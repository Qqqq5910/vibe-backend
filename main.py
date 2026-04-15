
import os
import uuid
import base64
import sqlite3
import json
import math
import httpx
import asyncio
import re
import time
import logging
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Optional, Dict, Deque, Tuple, List

import jwt
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from datetime import datetime

load_dotenv()

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("vibelocator-backend")

# -----------------------------
# Env config
# -----------------------------
APP_ENV = os.getenv("APP_ENV", "production")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")
DB_PATH = os.getenv("DB_PATH", "memories.db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(4 * 1024 * 1024)))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "300"))
BLIND_BOX_THRESHOLD = int(os.getenv("BLIND_BOX_THRESHOLD", "2"))
FREE_CAPSULE_LIMIT = int(os.getenv("FREE_CAPSULE_LIMIT", "30"))
FREE_AI_LIMIT = int(os.getenv("FREE_AI_LIMIT", "10"))
PRO_AI_LIMIT = int(os.getenv("PRO_AI_LIMIT", "100"))
UPLOAD_RATE_LIMIT_PER_MIN = int(os.getenv("UPLOAD_RATE_LIMIT_PER_MIN", "12"))
GENERAL_RATE_LIMIT_PER_MIN = int(os.getenv("GENERAL_RATE_LIMIT_PER_MIN", "120"))
UPLOADS_PERSISTENT_WARNING = os.getenv("UPLOADS_PERSISTENT_WARNING", "1") == "1"
ALLOW_INSECURE_UPGRADE = os.getenv("ALLOW_INSECURE_UPGRADE", "0") == "1"
PRO_PRODUCT_ID = os.getenv("PRO_PRODUCT_ID", "com.vibelocator.pro")
QWEN_MODEL_IMAGE = os.getenv("QWEN_MODEL_IMAGE", "qwen-vl-max")
QWEN_MODEL_TEXT = os.getenv("QWEN_MODEL_TEXT", "qwen-max")
AMAP_KEY = os.getenv("AMAP_KEY", "")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")

# Apple App Store Server API config
APPLE_ISSUER_ID = os.getenv("APPLE_ISSUER_ID", "")
APPLE_KEY_ID = os.getenv("APPLE_KEY_ID", "")
APPLE_BUNDLE_ID = os.getenv("APPLE_BUNDLE_ID", "")
APPLE_ENVIRONMENT = os.getenv("APPLE_ENVIRONMENT", "Auto")  # Auto | Production | Sandbox
APPLE_PRIVATE_KEY = os.getenv("APPLE_PRIVATE_KEY", "")
APPLE_PRIVATE_KEY_FILE = os.getenv("APPLE_PRIVATE_KEY_FILE", "")

# CORS config
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "*")
if ALLOWED_ORIGINS_RAW.strip() == "*":
    ALLOWED_ORIGINS = ["*"]
    ALLOW_CREDENTIALS = False
else:
    ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS_RAW.split(",") if o.strip()]
    ALLOW_CREDENTIALS = True

if not AMAP_KEY:
    logger.warning("AMAP_KEY is missing. Amap-related endpoints will fail.")
if not QWEN_API_KEY:
    logger.warning("QWEN_API_KEY is missing. AI parsing will fail.")
if UPLOADS_PERSISTENT_WARNING:
    logger.warning("Make sure %s and %s are on persistent storage in Zeabur.", DB_PATH, UPLOAD_DIR)

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="VibeLocator API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# -----------------------------
# Models
# -----------------------------
class UploadRequest(BaseModel):
    image_base64: Optional[str] = ""
    text_content: Optional[str] = ""
    source_url: Optional[str] = ""
    exact_name: Optional[str] = ""
    exact_location: Optional[str] = ""
    exact_district: Optional[str] = ""
    device_id: str = Field(min_length=8, max_length=128)
    lat: float
    lon: float


class NearbyRequest(BaseModel):
    lat: float
    lon: float
    device_id: str = Field(min_length=8, max_length=128)


class UpgradeRequest(BaseModel):
    device_id: str = Field(min_length=8, max_length=128)
    transaction_id: Optional[str] = None


# -----------------------------
# Lightweight rate limiter
# -----------------------------
class SlidingWindowRateLimiter:
    def __init__(self) -> None:
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        now = time.time()
        async with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] <= now - window_seconds:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True


rate_limiter = SlidingWindowRateLimiter()


async def enforce_rate_limit(request: Request, scope_key: str, limit: int) -> None:
    ip = request.client.host if request.client else "unknown"
    key = f"{scope_key}:{ip}"
    ok = await rate_limiter.check(key, limit)
    if not ok:
        raise HTTPException(status_code=429, detail="RATE_LIMITED")


# -----------------------------
# DB helpers
# -----------------------------
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=10000;")
        yield conn
        conn.commit()
    finally:
        conn.close()


# -----------------------------
# Startup init
# -----------------------------
def init_db() -> None:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_pool (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand TEXT NOT NULL,
                city TEXT NOT NULL,
                image_url TEXT,
                created_at TEXT NOT NULL,
                device_id TEXT DEFAULT 'anonymous',
                poi_lat REAL DEFAULT 0.0,
                poi_lon REAL DEFAULT 0.0
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                device_id TEXT PRIMARY KEY,
                is_pro INTEGER DEFAULT 0,
                ai_usage_month TEXT DEFAULT '',
                ai_usage_count INTEGER DEFAULT 0,
                last_verified_transaction_id TEXT DEFAULT ''
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_pool_device_id ON memory_pool(device_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_pool_brand ON memory_pool(brand)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_pool_created_at ON memory_pool(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_pool_device_created ON memory_pool(device_id, created_at DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_pool_device_brand ON memory_pool(device_id, brand)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_pool_brand_poi ON memory_pool(brand, poi_lat, poi_lon)")


@app.on_event("startup")
async def startup_event() -> None:
    init_db()
    logger.info("Server started. env=%s base_url=%s", APP_ENV, BASE_URL)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    response = None
    try:
        await enforce_rate_limit(request, "general", GENERAL_RATE_LIMIT_PER_MIN)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        logger.exception("Unhandled error | request_id=%s | path=%s", request_id, request.url.path)
        raise


# -----------------------------
# Geo helpers
# -----------------------------
pi = 3.1415926535897932384626
A = 6378245.0
EE = 0.00669342162296594323


def wgs84_to_gcj02(lng: float, lat: float) -> Tuple[float, float]:
    if not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271):
        return lng, lat

    def transform_lat(lng_: float, lat_: float) -> float:
        ret = -100.0 + 2.0 * lng_ + 3.0 * lat_ + 0.2 * lat_ * lat_ + 0.1 * lng_ * lat_ + 0.2 * math.sqrt(abs(lng_))
        ret += (20.0 * math.sin(6.0 * lng_ * pi) + 20.0 * math.sin(2.0 * lng_ * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat_ * pi) + 40.0 * math.sin(lat_ / 3.0 * pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat_ / 12.0 * pi) + 320 * math.sin(lat_ * pi / 30.0)) * 2.0 / 3.0
        return ret

    def transform_lng(lng_: float, lat_: float) -> float:
        ret = 300.0 + lng_ + 2.0 * lat_ + 0.1 * lng_ * lng_ + 0.1 * lng_ * lat_ + 0.1 * math.sqrt(abs(lng_))
        ret += (20.0 * math.sin(6.0 * lng_ * pi) + 20.0 * math.sin(2.0 * lng_ * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lng_ * pi) + 40.0 * math.sin(lng_ / 3.0 * pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lng_ / 12.0 * pi) + 300.0 * math.sin(lng_ / 30.0 * pi)) * 2.0 / 3.0
        return ret

    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((A * (1 - EE)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (A / sqrtmagic * math.cos(radlat) * pi)
    return lng + dlng, lat + dlat


def gcj02_to_wgs84(lng: float, lat: float) -> Tuple[float, float]:
    if not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271):
        return lng, lat
    m_lng, m_lat = wgs84_to_gcj02(lng, lat)
    return lng * 2 - m_lng, lat * 2 - m_lat


def calculate_haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> int:
    lon1, lat1, lon2, lat2 = map(math.radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a_ = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return int(2 * math.asin(math.sqrt(a_)) * 6371000)


def extract_core_brand(name: str) -> str:
    if not name:
        return ""
    return re.split(r"\(|（|·|-| ", name)[0].strip()


# -----------------------------
# Utility helpers
# -----------------------------
def validate_lat_lon(lat: float, lon: float) -> None:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="INVALID_COORDINATES")


def build_openai_client() -> AsyncOpenAI:
    if not QWEN_API_KEY:
        raise HTTPException(status_code=500, detail="QWEN_API_KEY_MISSING")
    return AsyncOpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def parse_location_string(location_str: str) -> Tuple[float, float]:
    try:
        gcj_lon, gcj_lat = map(float, location_str.split(","))
        return gcj02_to_wgs84(gcj_lon, gcj_lat)
    except Exception as e:
        raise HTTPException(status_code=400, detail="INVALID_LOCATION_STRING") from e


def image_url_to_local_path(image_url: str) -> Optional[str]:
    if not image_url:
        return None
    prefix = f"{BASE_URL}/uploads/"
    if image_url.startswith(prefix):
        filename = image_url[len(prefix):]
        return os.path.join(UPLOAD_DIR, filename)
    return None


def safe_base64url_json_decode(jws_token: str) -> dict:
    try:
        parts = jws_token.split(".")
        if len(parts) < 2:
            raise ValueError("Invalid JWS format")
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("utf-8"))
        return json.loads(decoded)
    except Exception as e:
        raise HTTPException(status_code=502, detail="APPLE_JWS_DECODE_FAILED") from e


def build_multimodal_hint(text_content: str, source_url: str) -> str:
    parts: List[str] = [
        "请提取这次分享里对应的商户或地点的核心品牌名与城市。",
        "优先识别实际店名、品牌名，不要输出“探店”“收藏”“推荐”“链接”等无关词。",
    ]
    if text_content:
        parts.append(f"补充文字：{text_content}")
    if source_url:
        parts.append(f"来源链接：{source_url}")
    return "\n".join(parts)


async def run_ai_extract(
    client: AsyncOpenAI,
    system_prompt: str,
    *,
    text_content: str,
    image_base64: str,
    source_url: str,
    image_only_fallback: bool = False,
) -> Tuple[str, str]:
    messages = [{"role": "system", "content": system_prompt}]

    if image_base64:
        if image_only_fallback:
            hint_text = "请只根据图片内容提取商户核心品牌名与城市。忽略任何模糊文案。"
        else:
            hint_text = build_multimodal_hint(text_content, source_url)

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": hint_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }
        )
        model = QWEN_MODEL_IMAGE
    else:
        plain_text = text_content or source_url
        messages.append({"role": "user", "content": f"请从这段文字中提取商户核心品牌名与城市：{plain_text}"})
        model = QWEN_MODEL_TEXT

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        timeout=25,
    )
    result_json = json.loads(response.choices[0].message.content)
    brand_name = (result_json.get("brand") or "").strip()
    city_hint = (result_json.get("city") or "").strip()
    return brand_name, city_hint


# -----------------------------
# App Store verification helpers
# -----------------------------
def load_apple_private_key() -> str:
    if APPLE_PRIVATE_KEY:
        return APPLE_PRIVATE_KEY.replace("\\n", "\n")
    if APPLE_PRIVATE_KEY_FILE:
        with open(APPLE_PRIVATE_KEY_FILE, "r", encoding="utf-8") as f:
            return f.read()
    raise HTTPException(status_code=500, detail="APPLE_PRIVATE_KEY_MISSING")


def build_apple_jwt() -> str:
    if not APPLE_ISSUER_ID or not APPLE_KEY_ID or not APPLE_BUNDLE_ID:
        raise HTTPException(status_code=500, detail="APPLE_SERVER_API_CONFIG_MISSING")
    private_key = load_apple_private_key()
    now = int(time.time())
    payload = {
        "iss": APPLE_ISSUER_ID,
        "iat": now,
        "exp": now + 1200,
        "aud": "appstoreconnect-v1",
        "bid": APPLE_BUNDLE_ID,
    }
    headers = {"alg": "ES256", "kid": APPLE_KEY_ID, "typ": "JWT"}
    return jwt.encode(payload, private_key, algorithm="ES256", headers=headers)


async def fetch_apple_transaction(transaction_id: str) -> dict:
    token = build_apple_jwt()
    environments: List[str]
    env_normalized = APPLE_ENVIRONMENT.strip().lower()
    if env_normalized == "production":
        environments = ["Production"]
    elif env_normalized == "sandbox":
        environments = ["Sandbox"]
    else:
        environments = ["Production", "Sandbox"]

    last_error = None
    for env in environments:
        base = (
            "https://api.storekit.itunes.apple.com"
            if env == "Production"
            else "https://api.storekit-sandbox.itunes.apple.com"
        )
        url = f"{base}/inApps/v1/transactions/{transaction_id}"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, headers={"Authorization": f"Bearer {token}"})
            if resp.status_code == 200:
                return {"environment": env, "data": resp.json()}
            last_error = f"{env}:{resp.status_code}:{resp.text[:200]}"
            logger.warning("Apple verify failed | env=%s | status=%s | body=%s", env, resp.status_code, resp.text[:300])
        except Exception as e:
            last_error = f"{env}:{repr(e)}"
            logger.exception("Apple verify exception | env=%s", env)

    raise HTTPException(status_code=502, detail=f"APPLE_VERIFY_FAILED:{last_error}")


async def verify_pro_purchase(transaction_id: str) -> dict:
    apple_response = await fetch_apple_transaction(transaction_id)
    data = apple_response["data"]
    signed_info = data.get("signedTransactionInfo")
    if not signed_info:
        raise HTTPException(status_code=502, detail="APPLE_SIGNED_TRANSACTION_MISSING")

    payload = safe_base64url_json_decode(signed_info)

    if payload.get("bundleId") != APPLE_BUNDLE_ID:
        raise HTTPException(status_code=403, detail="APPLE_BUNDLE_MISMATCH")
    if payload.get("productId") != PRO_PRODUCT_ID:
        raise HTTPException(status_code=403, detail="APPLE_PRODUCT_MISMATCH")
    if str(payload.get("transactionId")) != str(transaction_id):
        raise HTTPException(status_code=403, detail="APPLE_TRANSACTION_ID_MISMATCH")
    if payload.get("revocationDate"):
        raise HTTPException(status_code=403, detail="APPLE_TRANSACTION_REVOKED")

    return {
        "environment": apple_response["environment"],
        "transaction_payload": payload,
    }


# -----------------------------
# Quota logic
# -----------------------------
def get_or_create_user_profile(conn: sqlite3.Connection, device_id: str) -> sqlite3.Row:
    current_month = datetime.now().strftime("%Y-%m")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT device_id, is_pro, ai_usage_month, ai_usage_count, last_verified_transaction_id FROM user_profiles WHERE device_id = ?",
        (device_id,),
    )
    row = cursor.fetchone()
    if row is None:
        cursor.execute(
            "INSERT INTO user_profiles (device_id, is_pro, ai_usage_month, ai_usage_count, last_verified_transaction_id) VALUES (?, 0, ?, 0, '')",
            (device_id, current_month),
        )
        cursor.execute(
            "SELECT device_id, is_pro, ai_usage_month, ai_usage_count, last_verified_transaction_id FROM user_profiles WHERE device_id = ?",
            (device_id,),
        )
        row = cursor.fetchone()
    if row["ai_usage_month"] != current_month:
        cursor.execute(
            "UPDATE user_profiles SET ai_usage_month = ?, ai_usage_count = 0 WHERE device_id = ?",
            (current_month, device_id),
        )
        cursor.execute(
            "SELECT device_id, is_pro, ai_usage_month, ai_usage_count, last_verified_transaction_id FROM user_profiles WHERE device_id = ?",
            (device_id,),
        )
        row = cursor.fetchone()
    return row


def check_quota(device_id: str, is_ai_request: bool) -> Tuple[bool, str]:
    with get_db() as conn:
        profile = get_or_create_user_profile(conn, device_id)
        is_pro = bool(profile["is_pro"])
        ai_count = int(profile["ai_usage_count"])

        if not is_pro:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS cnt FROM memory_pool WHERE device_id = ?", (device_id,))
            count = int(cursor.fetchone()["cnt"])
            if count >= FREE_CAPSULE_LIMIT:
                return False, "CAPSULE_LIMIT_REACHED"

        if is_ai_request:
            quota = PRO_AI_LIMIT if is_pro else FREE_AI_LIMIT
            if ai_count >= quota:
                return False, "AI_LIMIT_REACHED"

        return True, ""


def increment_ai_usage(device_id: str) -> None:
    with get_db() as conn:
        get_or_create_user_profile(conn, device_id)
        conn.execute("UPDATE user_profiles SET ai_usage_count = ai_usage_count + 1 WHERE device_id = ?", (device_id,))


def mark_user_pro(device_id: str, transaction_id: str = "") -> None:
    current_month = datetime.now().strftime("%Y-%m")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO user_profiles (device_id, is_pro, ai_usage_month, ai_usage_count, last_verified_transaction_id)
            VALUES (?, 1, ?, 0, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                is_pro = 1,
                ai_usage_month = excluded.ai_usage_month,
                last_verified_transaction_id = excluded.last_verified_transaction_id
            """,
            (device_id, current_month, transaction_id or ""),
        )


# -----------------------------
# Routes
# -----------------------------
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat() + "Z",
        "db_path": DB_PATH,
        "upload_dir": UPLOAD_DIR,
    }


@app.get("/api/tips")
async def get_input_tips(keyword: str, lat: float, lon: float, request: Request):
    await enforce_rate_limit(request, "tips", 60)
    validate_lat_lon(lat, lon)
    keyword = (keyword or "").strip()
    if not keyword:
        return []
    if not AMAP_KEY:
        raise HTTPException(status_code=500, detail="AMAP_KEY_MISSING")

    gcj_lon, gcj_lat = wgs84_to_gcj02(lon, lat)
    url = "https://restapi.amap.com/v3/assistant/inputtips"
    params = {"key": AMAP_KEY, "keywords": keyword[:80], "location": f"{gcj_lon},{gcj_lat}", "datatype": "all"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = (await client.get(url, params=params)).json()
        if res.get("status") == "1":
            return [tip for tip in res.get("tips", []) if tip.get("location") and len(tip.get("location", "")) > 5]
        logger.warning("Amap inputtips failed | response=%s", res)
        return []
    except Exception:
        logger.exception("Amap inputtips exception")
        return []


@app.post("/api/upload")
async def upload_memory(request_data: UploadRequest, request: Request):
    await enforce_rate_limit(request, "upload", UPLOAD_RATE_LIMIT_PER_MIN)
    validate_lat_lon(request_data.lat, request_data.lon)

    device_id = request_data.device_id.strip()
    text_content = (request_data.text_content or "").strip()
    image_base64 = request_data.image_base64 or ""
    source_url = (request_data.source_url or "").strip()
    exact_name = (request_data.exact_name or "").strip()
    exact_location = (request_data.exact_location or "").strip()
    exact_district = (request_data.exact_district or "").strip()

    if text_content and len(text_content) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail="TEXT_TOO_LONG")
    if image_base64 and len(image_base64.encode("utf-8")) > int(MAX_IMAGE_BYTES * 1.45):
        raise HTTPException(status_code=413, detail="IMAGE_TOO_LARGE")

    logger.info(
        "upload payload summary | has_text=%s | text_len=%s | has_image=%s | has_source_url=%s | exact_name=%s",
        bool(text_content),
        len(text_content),
        bool(image_base64),
        bool(source_url),
        bool(exact_name),
    )

    is_ai_request = bool(image_base64) or bool(text_content) or bool(source_url)
    can_proceed, reason = check_quota(device_id, is_ai_request)
    if not can_proceed:
        raise HTTPException(status_code=403, detail=reason)

    brand_name = ""
    city_hint = ""
    poi_lat = 0.0
    poi_lon = 0.0
    image_url = ""

    if exact_name and exact_location:
        brand_name = exact_name
        city_hint = exact_district
        poi_lon, poi_lat = parse_location_string(exact_location)
    else:
        if not (text_content or image_base64 or source_url):
            raise HTTPException(status_code=400, detail="MUST_PROVIDE_TEXT_OR_IMAGE")

        client = build_openai_client()
        system_prompt = (
            "你是一个精准的位置实体提取引擎。"
            "请提取用户截图、分享文案或链接语境中的【核心品牌名】（brand）和【城市】（city）。"
            "品牌名必须尽量短，只保留真正的店名、品牌名，不要带菜系、分店、标点后缀。"
            "如果文字很模糊，但图片里有明确店名，请优先相信图片。"
            "如果是纯文本描述且未明确指明城市，city字段请留空。"
            "只返回 JSON，如 {\"brand\":\"xxx\",\"city\":\"xxx\"}。"
        )

        try:
            brand_name, city_hint = await run_ai_extract(
                client,
                system_prompt,
                text_content=text_content,
                image_base64=image_base64,
                source_url=source_url,
                image_only_fallback=False,
            )

            if not brand_name and image_base64 and text_content:
                logger.info("ai extract empty on mixed input, retrying image-only fallback")
                brand_name, city_hint = await run_ai_extract(
                    client,
                    system_prompt,
                    text_content="",
                    image_base64=image_base64,
                    source_url="",
                    image_only_fallback=True,
                )

            logger.info("ai extract result | brand=%s | city=%s", brand_name, city_hint)

            if not brand_name:
                raise HTTPException(status_code=422, detail="AI_EXTRACT_EMPTY")

            increment_ai_usage(device_id)
        except HTTPException:
            raise
        except Exception:
            logger.exception("AI parse failed")
            raise HTTPException(status_code=500, detail="AI_PARSE_FAILED")

        if not AMAP_KEY:
            raise HTTPException(status_code=500, detail="AMAP_KEY_MISSING")

        upload_gcj_lon, upload_gcj_lat = wgs84_to_gcj02(request_data.lon, request_data.lat)
        amap_url = "https://restapi.amap.com/v3/place/text"
        params = {
            "key": AMAP_KEY,
            "keywords": brand_name,
            "city": city_hint,
            "location": f"{upload_gcj_lon},{upload_gcj_lat}",
        }
        try:
            async with httpx.AsyncClient(timeout=10) as http_client:
                search_res = (await http_client.get(amap_url, params=params)).json()
            if search_res.get("status") == "1" and search_res.get("pois"):
                poi = search_res["pois"][0]
                loc_str = poi.get("location", "")
                if "," in loc_str:
                    gcj_p_lon, gcj_p_lat = map(float, loc_str.split(","))
                    poi_lon, poi_lat = gcj02_to_wgs84(gcj_p_lon, gcj_p_lat)
            else:
                logger.info("Amap place text no result | brand=%s | city=%s", brand_name, city_hint)
        except Exception:
            logger.exception("Amap place search failed")

        if image_base64:
            try:
                image_data = base64.b64decode(image_base64)
                if len(image_data) > MAX_IMAGE_BYTES:
                    raise HTTPException(status_code=413, detail="IMAGE_TOO_LARGE")
                filename = f"{uuid.uuid4().hex}.jpg"
                filepath = os.path.join(UPLOAD_DIR, filename)
                with open(filepath, "wb") as f:
                    f.write(image_data)
                image_url = f"{BASE_URL}/uploads/{filename}"
            except HTTPException:
                raise
            except Exception:
                logger.exception("Image save failed")
                image_url = ""

    clean_brand = extract_core_brand(brand_name) or brand_name
    if not clean_brand:
        raise HTTPException(status_code=422, detail="EMPTY_BRAND")

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO memory_pool (brand, city, image_url, created_at, device_id, poi_lat, poi_lon)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean_brand,
                city_hint,
                image_url,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                device_id,
                poi_lat,
                poi_lon,
            ),
        )

    dist_to_upload = 999999
    if poi_lat != 0.0 and poi_lon != 0.0:
        dist_to_upload = calculate_haversine_distance(request_data.lon, request_data.lat, poi_lon, poi_lat)

    return {
        "status": "success",
        "message": clean_brand,
        "poi_lat": poi_lat,
        "poi_lon": poi_lon,
        "is_immediate_nearby": dist_to_upload <= 500,
    }


@app.post("/api/nearby")
async def discover_nearby(request_data: NearbyRequest, request: Request):
    await enforce_rate_limit(request, "nearby", 30)
    validate_lat_lon(request_data.lat, request_data.lon)
    if request_data.lat == 0.0 and request_data.lon == 0.0:
        return []

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT brand FROM memory_pool WHERE device_id = ?", (request_data.device_id,))
        my_brands = [row["brand"] for row in cursor.fetchall()]
        cursor.execute(
            """
            SELECT brand, poi_lat, poi_lon, COUNT(DISTINCT device_id) AS wish_count
            FROM memory_pool
            WHERE device_id != ? AND poi_lat != 0.0 AND poi_lon != 0.0
            GROUP BY brand, poi_lat, poi_lon
            HAVING wish_count >= ?
            """,
            (request_data.device_id, BLIND_BOX_THRESHOLD),
        )
        public_boxes = cursor.fetchall()

    if not AMAP_KEY:
        raise HTTPException(status_code=500, detail="AMAP_KEY_MISSING")

    gcj_lon, gcj_lat = wgs84_to_gcj02(request_data.lon, request_data.lat)
    amap_url = "https://restapi.amap.com/v3/place/around"
    nearby_results = []

    async def fetch_brand_nearby(client: httpx.AsyncClient, brand_name: str):
        params = {
            "key": AMAP_KEY,
            "keywords": brand_name,
            "location": f"{gcj_lon},{gcj_lat}",
            "radius": "3000",
            "sortrule": "distance",
        }
        try:
            res = await client.get(amap_url, params=params)
            return brand_name, res.json()
        except Exception:
            logger.exception("Amap nearby fetch failed | brand=%s", brand_name)
            return brand_name, None

    brands_to_query = my_brands[:30]  # 防止单个用户收藏太多导致外部请求过多
    if brands_to_query:
        async with httpx.AsyncClient(timeout=10) as client:
            tasks = [fetch_brand_nearby(client, brand) for brand in brands_to_query]
            responses = await asyncio.gather(*tasks)
        for brand_name, search_res in responses:
            if search_res and search_res.get("status") == "1":
                for poi in search_res.get("pois", []):
                    dist = int(poi.get("distance", 9999))
                    if dist <= 3000:
                        loc_str = poi.get("location", "")
                        poi_lon, poi_lat = 0.0, 0.0
                        if "," in loc_str:
                            gcj_p_lon, gcj_p_lat = map(float, loc_str.split(","))
                            poi_lon, poi_lat = gcj02_to_wgs84(gcj_p_lon, gcj_p_lat)
                        nearby_results.append(
                            {
                                "brand": brand_name,
                                "name": poi.get("name", ""),
                                "address": poi.get("address", ""),
                                "distance": str(dist),
                                "lat": poi_lat,
                                "lon": poi_lon,
                                "is_public": False,
                                "wish_count": 1,
                            }
                        )

    for p in public_boxes:
        dist = calculate_haversine_distance(request_data.lon, request_data.lat, p["poi_lon"], p["poi_lat"])
        if dist <= 3000:
            nearby_results.append(
                {
                    "brand": p["brand"],
                    "name": f"神秘盲盒：{p['brand']}",
                    "address": "周边高人气打卡地",
                    "distance": str(dist),
                    "lat": p["poi_lat"],
                    "lon": p["poi_lon"],
                    "is_public": True,
                    "wish_count": p["wish_count"],
                }
            )

    unique_results = []
    seen = set()
    for item in sorted(nearby_results, key=lambda x: int(x["distance"])):
        identifier = f"{item['name']}_{item['lat']}_{item['lon']}"
        if identifier not in seen:
            seen.add(identifier)
            unique_results.append(item)
    return unique_results


@app.get("/api/memories")
def get_all_memories(device_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, brand, city, created_at, image_url FROM memory_pool WHERE device_id = ? ORDER BY created_at DESC",
            (device_id,),
        )
        rows = cursor.fetchall()
    return [
        {"id": r["id"], "brand": r["brand"], "city": r["city"], "created_at": r["created_at"], "image_url": r["image_url"]}
        for r in rows
    ]


@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: int, device_id: str):
    image_url = ""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image_url FROM memory_pool WHERE id = ? AND device_id = ?", (memory_id, device_id))
        row = cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="MEMORY_NOT_FOUND")
        image_url = row["image_url"] or ""
        cursor.execute("DELETE FROM memory_pool WHERE id = ? AND device_id = ?", (memory_id, device_id))

    local_path = image_url_to_local_path(image_url)
    if local_path and os.path.exists(local_path):
        try:
            os.remove(local_path)
        except Exception:
            logger.exception("Failed to delete image file | path=%s", local_path)

    return {"status": "success"}


@app.post("/api/upgrade")
async def upgrade_to_pro(request_data: UpgradeRequest, request: Request):
    await enforce_rate_limit(request, "upgrade", 20)
    device_id = request_data.device_id.strip()

    if request_data.transaction_id:
        verification = await verify_pro_purchase(str(request_data.transaction_id))
        mark_user_pro(device_id, str(request_data.transaction_id))
        return {
            "status": "success",
            "message": "Upgraded to Pro",
            "verification": {
                "environment": verification["environment"],
                "productId": verification["transaction_payload"].get("productId"),
                "transactionId": verification["transaction_payload"].get("transactionId"),
            },
        }

    if ALLOW_INSECURE_UPGRADE:
        logger.warning("INSECURE upgrade used for device_id=%s. Disable ALLOW_INSECURE_UPGRADE in production.", device_id)
        mark_user_pro(device_id)
        return {"status": "success", "message": "Upgraded to Pro (INSECURE MODE)"}

    raise HTTPException(status_code=400, detail="TRANSACTION_ID_REQUIRED")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
