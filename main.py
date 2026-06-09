
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
import hmac
import shutil
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Optional, Dict, Deque, Tuple, List
from urllib.parse import quote_plus

import jwt
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse
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
PERSISTENT_DATA_DIR = os.getenv("PERSISTENT_DATA_DIR", "/data")
DB_PATH = os.getenv("DB_PATH") or os.path.join(PERSISTENT_DATA_DIR, "memories.db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR") or os.path.join(PERSISTENT_DATA_DIR, "uploads")
LEGACY_DB_PATH = os.getenv("LEGACY_DB_PATH", "memories.db")
LEGACY_UPLOAD_DIR = os.getenv("LEGACY_UPLOAD_DIR", "uploads")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
ADMIN_COOKIE_NAME = "vibe_admin_token"
REQUIRE_PERSISTENT_STORAGE = os.getenv("REQUIRE_PERSISTENT_STORAGE", "0") == "1"
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(4 * 1024 * 1024)))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "300"))
BLIND_BOX_THRESHOLD = int(os.getenv("BLIND_BOX_THRESHOLD", "2"))
FREE_CAPSULE_LIMIT = int(os.getenv("FREE_CAPSULE_LIMIT", "30"))
FREE_AI_LIMIT = int(os.getenv("FREE_AI_LIMIT", "10"))
PRO_AI_LIMIT = int(os.getenv("PRO_AI_LIMIT", "100"))
UPLOAD_RATE_LIMIT_PER_MIN = int(os.getenv("UPLOAD_RATE_LIMIT_PER_MIN", "12"))
GENERAL_RATE_LIMIT_PER_MIN = int(os.getenv("GENERAL_RATE_LIMIT_PER_MIN", "120"))
TIPS_RATE_LIMIT_PER_MIN = int(os.getenv("TIPS_RATE_LIMIT_PER_MIN", "20"))
AMAP_TIPS_CACHE_TTL_SECONDS = int(os.getenv("AMAP_TIPS_CACHE_TTL_SECONDS", str(7 * 24 * 60 * 60)))
AMAP_PLACE_CACHE_TTL_SECONDS = int(os.getenv("AMAP_PLACE_CACHE_TTL_SECONDS", str(30 * 24 * 60 * 60)))
UPLOADS_PERSISTENT_WARNING = os.getenv("UPLOADS_PERSISTENT_WARNING", "1") == "1"
ALLOW_INSECURE_UPGRADE = os.getenv("ALLOW_INSECURE_UPGRADE", "0") == "1"
PRO_PRODUCT_ID = os.getenv("PRO_PRODUCT_ID", "com.vibelocator.pro")
QWEN_MODEL_IMAGE = os.getenv("QWEN_MODEL_IMAGE", "qwen-vl-max")
QWEN_MODEL_TEXT = os.getenv("QWEN_MODEL_TEXT", "qwen-max")
AMAP_KEY = os.getenv("AMAP_KEY", "")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
MIN_SUPPORTED_IOS_VERSION = os.getenv("MIN_SUPPORTED_IOS_VERSION", "1.3")
LATEST_IOS_VERSION = os.getenv("LATEST_IOS_VERSION", MIN_SUPPORTED_IOS_VERSION)
APP_STORE_URL = os.getenv("APP_STORE_URL") or "https://apps.apple.com/cn/app/id6770473628"
FORCE_UPDATE_MESSAGE = os.getenv("FORCE_UPDATE_MESSAGE", "这个版本已经停止使用，请更新到最新版后继续。")
FORCE_UPDATE_AFTER = os.getenv("FORCE_UPDATE_AFTER", "")
ENFORCE_WHEN_APP_STORE_VERSION_AVAILABLE = os.getenv("ENFORCE_WHEN_APP_STORE_VERSION_AVAILABLE", "1") != "0"
FORCE_LATEST_APP_STORE_VERSION = os.getenv("FORCE_LATEST_APP_STORE_VERSION", "1") != "0"

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

_storage_uses_persistent_dir = (
    os.path.abspath(DB_PATH).startswith(os.path.abspath(PERSISTENT_DATA_DIR) + os.sep)
    and os.path.abspath(UPLOAD_DIR).startswith(os.path.abspath(PERSISTENT_DATA_DIR) + os.sep)
)
if REQUIRE_PERSISTENT_STORAGE and not _storage_uses_persistent_dir:
    raise RuntimeError(
        "Persistent storage is required. Set DB_PATH=/data/memories.db and UPLOAD_DIR=/data/uploads."
    )

os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
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
    exact_address: Optional[str] = ""
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


class AdminLoginRequest(BaseModel):
    token: str = Field(min_length=16, max_length=256)


class AdminMigrateRequest(BaseModel):
    from_device_id: str = Field(min_length=8, max_length=128)
    to_device_id: str = Field(min_length=8, max_length=128)
    dry_run: bool = False


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
def _same_path(left: str, right: str) -> bool:
    return os.path.abspath(left) == os.path.abspath(right)


def sqlite_table_count(path: str, table_name: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    try:
        conn = sqlite3.connect(path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            if cursor.fetchone() is None:
                return 0
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return int(cursor.fetchone()[0])
        finally:
            conn.close()
    except Exception:
        logger.exception("Failed counting table %s in %s", table_name, path)
        return 0


def backup_sqlite_database(source_path: str, target_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    source = sqlite3.connect(source_path)
    target = sqlite3.connect(target_path)
    try:
        source.backup(target)
        target.commit()
    finally:
        target.close()
        source.close()


def copy_legacy_uploads_if_needed() -> None:
    if not LEGACY_UPLOAD_DIR or _same_path(LEGACY_UPLOAD_DIR, UPLOAD_DIR):
        return
    if not os.path.isdir(LEGACY_UPLOAD_DIR):
        return

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    copied = 0
    for name in os.listdir(LEGACY_UPLOAD_DIR):
        source = os.path.join(LEGACY_UPLOAD_DIR, name)
        target = os.path.join(UPLOAD_DIR, name)
        if not os.path.isfile(source) or os.path.exists(target):
            continue
        try:
            shutil.copy2(source, target)
            copied += 1
        except Exception:
            logger.exception("Failed copying legacy upload %s", source)
    if copied:
        logger.info("Copied %s legacy upload files from %s to %s", copied, LEGACY_UPLOAD_DIR, UPLOAD_DIR)


def migrate_legacy_storage_if_needed() -> None:
    if _same_path(LEGACY_DB_PATH, DB_PATH):
        copy_legacy_uploads_if_needed()
        return

    legacy_places = sqlite_table_count(LEGACY_DB_PATH, "memory_pool")
    target_places = sqlite_table_count(DB_PATH, "memory_pool")
    if legacy_places > 0 and target_places == 0:
        logger.warning(
            "Migrating legacy database from %s to %s | legacy_places=%s",
            LEGACY_DB_PATH,
            DB_PATH,
            legacy_places,
        )
        backup_sqlite_database(LEGACY_DB_PATH, DB_PATH)
    elif legacy_places > 0:
        logger.info(
            "Skipping legacy DB migration because target already has data | legacy=%s target=%s",
            legacy_places,
            target_places,
        )

    copy_legacy_uploads_if_needed()


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
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS amap_cache (
                cache_key TEXT PRIMARY KEY,
                response_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                ttl_seconds INTEGER NOT NULL
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_pool_device_id ON memory_pool(device_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_pool_brand ON memory_pool(brand)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_pool_created_at ON memory_pool(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_pool_device_created ON memory_pool(device_id, created_at DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_pool_device_brand ON memory_pool(device_id, brand)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_pool_brand_poi ON memory_pool(brand, poi_lat, poi_lon)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_amap_cache_created_at ON amap_cache(created_at)")

        cursor.execute("PRAGMA table_info(memory_pool)")
        existing_columns = {row["name"] for row in cursor.fetchall()}
        memory_columns = (
            ("amap_name", "amap_name TEXT DEFAULT ''"),
            ("amap_address", "amap_address TEXT DEFAULT ''"),
            ("amap_location", "amap_location TEXT DEFAULT ''"),
            ("amap_district", "amap_district TEXT DEFAULT ''"),
        )
        for column_name, column_sql in memory_columns:
            if column_name not in existing_columns:
                cursor.execute(f"ALTER TABLE memory_pool ADD COLUMN {column_sql}")


@app.on_event("startup")
async def startup_event() -> None:
    migrate_legacy_storage_if_needed()
    init_db()
    logger.info("Server started. env=%s base_url=%s db=%s uploads=%s", APP_ENV, BASE_URL, DB_PATH, UPLOAD_DIR)


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
    if not is_mainland_china_coordinate(lng, lat):
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
    if not is_mainland_china_coordinate(lng, lat):
        return lng, lat
    m_lng, m_lat = wgs84_to_gcj02(lng, lat)
    return lng * 2 - m_lng, lat * 2 - m_lat


def is_mainland_china_coordinate(lng: float, lat: float) -> bool:
    return 72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271


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
def normalize_amap_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "".join(normalize_amap_value(item) for item in value if item is not None).strip()
    return str(value).strip()


def rounded_coordinate_key(lon: float, lat: float, precision: int = 3) -> str:
    return f"{round(float(lon), precision):.{precision}f},{round(float(lat), precision):.{precision}f}"


def amap_cache_key(scope: str, params: dict) -> str:
    safe_params = {
        str(key): normalize_amap_value(value).lower()
        for key, value in params.items()
        if key != "key" and normalize_amap_value(value)
    }
    return f"{scope}:{json.dumps(safe_params, ensure_ascii=False, sort_keys=True)}"


def get_cached_amap_response(cache_key: str) -> Optional[dict]:
    now = time.time()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT response_json, created_at, ttl_seconds FROM amap_cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        if now - float(row["created_at"]) > int(row["ttl_seconds"]):
            cursor.execute("DELETE FROM amap_cache WHERE cache_key = ?", (cache_key,))
            return None
        try:
            return json.loads(row["response_json"])
        except Exception:
            cursor.execute("DELETE FROM amap_cache WHERE cache_key = ?", (cache_key,))
            return None


def store_amap_response(cache_key: str, response: dict, ttl_seconds: int) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO amap_cache (cache_key, response_json, created_at, ttl_seconds)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                response_json = excluded.response_json,
                created_at = excluded.created_at,
                ttl_seconds = excluded.ttl_seconds
            """,
            (cache_key, json.dumps(response, ensure_ascii=False), time.time(), ttl_seconds),
        )


async def fetch_amap_json(
    url: str,
    params: dict,
    cache_scope: str,
    cache_params: dict,
    ttl_seconds: int,
) -> dict:
    cache_key = amap_cache_key(cache_scope, cache_params)
    cached = get_cached_amap_response(cache_key)
    if cached is not None:
        return cached

    async with httpx.AsyncClient(timeout=10) as client:
        response = (await client.get(url, params=params)).json()

    if response.get("status") == "1":
        store_amap_response(cache_key, response, ttl_seconds)
    return response


def validate_lat_lon(lat: float, lon: float) -> None:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="INVALID_COORDINATES")


def require_admin(request: Request) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN_NOT_CONFIGURED")

    auth_header = request.headers.get("authorization", "")
    prefix = "Bearer "
    bearer_token = auth_header[len(prefix):].strip() if auth_header.startswith(prefix) else ""
    cookie_token = request.cookies.get(ADMIN_COOKIE_NAME, "").strip()
    token = bearer_token or cookie_token
    if not token or not hmac.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="ADMIN_FORBIDDEN")


def masked_device_id(device_id: str) -> str:
    value = (device_id or "").strip()
    if len(value) <= 12:
        return value[:2] + "***" if value else ""
    return f"{value[:8]}...{value[-4:]}"


def escape_sql_like(value: str) -> str:
    return (
        value
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def admin_place_payload(row: sqlite3.Row, include_device_id: bool = False) -> dict:
    name = row["amap_name"] or row["brand"] or ""
    address = row["amap_address"] or row["city"] or ""
    lat = float(row["poi_lat"] or 0.0)
    lon = float(row["poi_lon"] or 0.0)
    has_coordinates = lat != 0.0 and lon != 0.0
    map_url = ""
    if has_coordinates:
        map_url = f"https://maps.apple.com/?ll={lat},{lon}&q={quote_plus(name or address)}"

    payload = {
        "id": row["id"],
        "brand": row["brand"],
        "name": name,
        "city": row["city"],
        "district": row["amap_district"] or "",
        "address": address,
        "location": row["amap_location"] or "",
        "lat": lat,
        "lon": lon,
        "has_coordinates": has_coordinates,
        "created_at": row["created_at"],
        "image_url": row["image_url"] or "",
        "map_url": map_url,
    }
    if include_device_id:
        payload["device_id"] = row["device_id"]
    return payload


def normalized_admin_place_value(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", (value or "").strip().lower())


def admin_place_duplicate_key(row: sqlite3.Row) -> tuple:
    name_key = normalized_admin_place_value(row["amap_name"] or row["brand"] or "")
    address_key = normalized_admin_place_value(row["amap_address"] or "")
    lat = float(row["poi_lat"] or 0.0)
    lon = float(row["poi_lon"] or 0.0)
    if lat != 0.0 and lon != 0.0:
        return ("coord", name_key, round(lat, 5), round(lon, 5))
    return ("text", name_key, address_key)


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


CITY_KEYWORDS = [
    "上海", "北京", "广州", "深圳", "杭州", "成都", "重庆", "南京", "苏州", "武汉",
    "西安", "长沙", "青岛", "厦门", "天津", "宁波", "郑州", "合肥", "无锡", "佛山",
    "东莞", "昆明", "福州", "南昌", "沈阳", "大连", "哈尔滨", "长春", "济南",
]

DISTRICT_CITY_MAP = {
    "徐家汇": "上海",
    "陆家嘴": "上海",
    "静安寺": "上海",
    "南京东路": "上海",
    "南京西路": "上海",
    "淮海路": "上海",
    "新天地": "上海",
    "五角场": "上海",
    "中山公园": "上海",
    "前滩": "上海",
    "环球港": "上海",
    "百脑汇": "上海",
    "国金中心": "上海",
    "来福士": "上海",
    "合生汇": "上海",
    "K11": "上海",
    "iapm": "上海",
    "太古里": "成都",
    "春熙路": "成都",
    "天河城": "广州",
    "万象城": "深圳",
    "SKP": "北京",
}

LOCATION_SUFFIX_HINTS = [
    "百脑汇", "万象城", "大悦城", "太古里", "万达", "印象城", "龙湖天街", "环球港",
    "来福士", "合生汇", "国金中心", "徐家汇", "陆家嘴", "静安寺", "南京东路", "南京西路",
    "淮海路", "新天地", "五角场", "中山公园", "前滩", "K11", "SKP", "iapm",
]

GENERIC_BRAND_WORDS = {
    "狗咖", "猫咖", "咖啡", "咖啡馆", "餐厅", "酒吧", "生活记录", "citywalk", "增狗馆",
    "毛孩子", "店员", "环境", "朋友", "大家", "上海", "北京", "广州", "深圳", "杭州", "成都",
}


def normalize_social_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"https?://\S+", " ", text)
    cleaned = re.sub(r"www\.\S+", " ", cleaned)
    cleaned = re.sub(r"[#＃][^\s#＃]+", lambda m: f" {m.group(0)} ", cleaned)
    cleaned = re.sub(r"[\u2600-\u27BF\U0001F300-\U0001FAFF]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def infer_city_from_social_text(text: str) -> str:
    normalized = normalize_social_text(text)
    for city in CITY_KEYWORDS:
        if city in normalized:
            return city
    for keyword, city in DISTRICT_CITY_MAP.items():
        if keyword in normalized:
            return city
    return ""


def clean_brand_candidate(candidate: str) -> str:
    candidate = candidate.strip(" ，。！？!?,.·~～:：/\\|[]【】()（）")
    candidate = re.sub(r"^(打卡了?|去了|来到了|来到|安利了?|推荐了?|收藏了?|发现了?|种草了?|这家|一家|一个)+", "", candidate).strip()
    candidate = re.sub(r"^(没有异味的|干净的|超火的|宝藏的|可爱的|幸福的|附近的|新开的)+", "", candidate).strip()

    changed = True
    while changed:
        changed = False
        for token in LOCATION_SUFFIX_HINTS:
            if candidate.endswith(token) and len(candidate) > len(token) + 1:
                candidate = candidate[:-len(token)].strip()
                changed = True

    candidate = re.sub(r"(分店|门店|旗舰店|概念店|体验店|首店)$", "", candidate).strip()
    candidate = re.sub(r"(商圈)$", "", candidate).strip()
    candidate = re.sub(r"(狗咖|猫咖)$", "", candidate).strip()
    candidate = re.sub(r"\s+", " ", candidate).strip()

    if not candidate:
        return ""
    if candidate in GENERIC_BRAND_WORDS:
        return ""
    if len(candidate) < 2 or len(candidate) > 18:
        return ""
    return candidate


def extract_brand_from_social_text(text: str) -> str:
    normalized = normalize_social_text(text)
    if not normalized:
        return ""

    segments = [seg.strip() for seg in re.split(r"[\n。！？!?；;]", normalized) if seg.strip()]
    patterns = [
        r"([A-Za-z0-9\u4e00-\u9fa5· ]{2,24}?)\s*(?:[（(][^()（）]{1,10}[)）])?\s*店\b",
        r"([A-Za-z0-9\u4e00-\u9fa5· ]{2,24}?)\s*(?:馆|咖啡馆|咖啡|酒吧|居酒屋|面包店|甜品店)\b",
    ]

    candidates = []
    for seg in segments:
        for pattern in patterns:
            for match in re.finditer(pattern, seg):
                raw = match.group(1)
                cleaned = clean_brand_candidate(raw)
                if cleaned:
                    candidates.append(cleaned)

    if candidates:
        candidates.sort(key=lambda s: (len(s), s.count(" ")), reverse=True)
        return candidates[0]

    compact = normalized.replace(" ", "")
    fuzzy_patterns = [
        r"([A-Za-z0-9\u4e00-\u9fa5·]{2,20}?)(?:百脑汇|万象城|大悦城|太古里|徐家汇|陆家嘴|静安寺|南京东路|南京西路|淮海路|新天地|五角场|中山公园)?店",
        r"([A-Za-z0-9\u4e00-\u9fa5·]{2,16})(?:狗咖|猫咖)",
    ]
    for pattern in fuzzy_patterns:
        match = re.search(pattern, compact)
        if match:
            cleaned = clean_brand_candidate(match.group(1))
            if cleaned:
                return cleaned

    return ""


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
ADMIN_DASHBOARD_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>路过心动后台</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #152420;
      --muted: #718078;
      --line: rgba(21, 36, 32, 0.10);
      --surface: rgba(255, 255, 255, 0.78);
      --solid: #ffffff;
      --leaf: #2f7d5b;
      --lake: #227796;
      --warn: #b7791f;
      --bad: #b42318;
      --shadow: 0 18px 48px rgba(43, 75, 67, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "PingFang SC", "Helvetica Neue", Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 12%, rgba(47, 125, 91, 0.16), transparent 30%),
        radial-gradient(circle at 88% 6%, rgba(34, 119, 150, 0.13), transparent 28%),
        linear-gradient(180deg, #f7fbf7 0%, #eef7f4 48%, #f8faf8 100%);
    }
    main {
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 48px 0 64px;
    }
    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 24px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(30px, 5vw, 52px);
      letter-spacing: 0;
      line-height: 1.06;
    }
    .sub {
      margin: 0;
      color: var(--muted);
      font-size: 16px;
    }
    .panel {
      background: var(--surface);
      border: 1px solid rgba(255, 255, 255, 0.72);
      box-shadow: var(--shadow);
      border-radius: 24px;
      backdrop-filter: blur(18px);
    }
    .login {
      max-width: 520px;
      padding: 28px;
    }
    label {
      display: block;
      margin-bottom: 10px;
      font-weight: 700;
    }
    input {
      width: 100%;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.86);
      border-radius: 16px;
      padding: 14px 16px;
      font-size: 16px;
      outline: none;
    }
    input:focus {
      border-color: rgba(47, 125, 91, 0.52);
      box-shadow: 0 0 0 4px rgba(47, 125, 91, 0.12);
    }
    .row {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    button {
      appearance: none;
      border: 0;
      border-radius: 16px;
      background: linear-gradient(135deg, var(--leaf), var(--lake));
      color: white;
      font-weight: 800;
      font-size: 15px;
      padding: 12px 18px;
      cursor: pointer;
      box-shadow: 0 10px 22px rgba(34, 119, 150, 0.20);
    }
    button.secondary {
      background: rgba(255, 255, 255, 0.82);
      color: var(--ink);
      border: 1px solid var(--line);
      box-shadow: none;
    }
    button:disabled {
      opacity: 0.6;
      cursor: wait;
    }
    .status {
      min-height: 22px;
      margin-top: 14px;
      color: var(--muted);
    }
    .status.error { color: var(--bad); }
    .hidden { display: none !important; }
    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin: 22px 0;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }
    .card {
      padding: 20px;
      background: var(--solid);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 10px 28px rgba(43, 75, 67, 0.07);
    }
    .metric {
      font-size: 34px;
      line-height: 1;
      font-weight: 900;
      margin-bottom: 8px;
    }
    .label {
      color: var(--muted);
      font-size: 14px;
    }
    .storage {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      padding: 18px;
      margin-bottom: 14px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 12px;
      font-weight: 800;
      background: rgba(47, 125, 91, 0.12);
      color: var(--leaf);
    }
    .badge.warn {
      background: rgba(183, 121, 31, 0.14);
      color: var(--warn);
    }
    code {
      display: block;
      overflow-wrap: anywhere;
      color: var(--muted);
      margin-top: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      background: var(--solid);
      border-radius: 20px;
      box-shadow: 0 10px 28px rgba(43, 75, 67, 0.07);
    }
    th, td {
      padding: 14px 16px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-size: 13px;
      background: rgba(246, 250, 248, 0.92);
    }
    tr:last-child td { border-bottom: 0; }
    .device {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      overflow-wrap: anywhere;
      max-width: 360px;
    }
    button.small {
      border-radius: 12px;
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 800;
    }
    .places-row td {
      padding: 0;
      background: rgba(246, 250, 248, 0.72);
    }
    .places-panel {
      margin: 0;
      padding: 18px;
      border-top: 1px solid var(--line);
    }
    .places-note {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 12px;
    }
    .place-list {
      display: grid;
      gap: 10px;
    }
    .place-item {
      display: grid;
      grid-template-columns: minmax(180px, 1.1fr) minmax(220px, 1.6fr) auto;
      gap: 12px;
      align-items: start;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.86);
    }
    .place-name {
      font-weight: 900;
      margin-bottom: 6px;
    }
    .place-meta, .place-address {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }
    .place-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .place-actions a {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 7px 10px;
      color: var(--lake);
      background: rgba(34, 119, 150, 0.10);
      text-decoration: none;
      font-size: 13px;
      font-weight: 800;
    }
    .search-panel {
      padding: 18px;
      margin-bottom: 14px;
    }
    .search-form {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: end;
    }
    .migrate-form {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) minmax(180px, 1fr) auto;
      gap: 12px;
      align-items: end;
    }
    .search-results {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }
    .search-result {
      display: grid;
      grid-template-columns: minmax(180px, 0.9fr) minmax(220px, 1.3fr) minmax(180px, 0.9fr) auto;
      gap: 12px;
      align-items: start;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.86);
    }
    .search-device {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      overflow-wrap: anywhere;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .empty {
      padding: 28px;
      text-align: center;
      color: var(--muted);
      background: var(--solid);
      border-radius: 20px;
      border: 1px solid var(--line);
    }
    .switch {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
      user-select: none;
    }
    .switch input {
      width: 18px;
      height: 18px;
      padding: 0;
    }
    @media (max-width: 820px) {
      main { width: min(100vw - 20px, 1120px); padding-top: 28px; }
      header, .toolbar { align-items: flex-start; flex-direction: column; }
      .cards, .storage { grid-template-columns: 1fr 1fr; }
      .place-item { grid-template-columns: 1fr; }
      .search-form, .migrate-form, .search-result { grid-template-columns: 1fr; }
      .place-actions { justify-content: flex-start; }
      table { display: block; overflow-x: auto; }
    }
    @media (max-width: 560px) {
      .cards, .storage { grid-template-columns: 1fr; }
      .login { max-width: none; }
      th, td { padding: 12px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>路过心动后台</h1>
        <p class="sub">查看持久化状态、用户数量、地点数量和每个用户的保存情况。</p>
      </div>
      <div id="headerActions" class="row hidden">
        <button id="refreshButton" type="button">刷新数据</button>
        <button id="logoutButton" class="secondary" type="button">退出登录</button>
      </div>
    </header>

    <section id="loginPanel" class="panel login hidden">
      <form id="loginForm">
        <label for="tokenInput">管理员 Token</label>
        <input id="tokenInput" name="token" type="password" autocomplete="current-password" placeholder="输入 Zeabur 的 ADMIN_TOKEN" />
        <div class="row" style="margin-top: 16px;">
          <button id="loginButton" type="submit">进入后台</button>
        </div>
        <div id="loginStatus" class="status"></div>
      </form>
    </section>

    <section id="dashboard" class="hidden">
      <div class="toolbar">
        <div id="updatedAt" class="label"></div>
        <label class="switch">
          <input id="showIds" type="checkbox" />
          显示完整设备 ID
        </label>
      </div>

      <div class="cards">
        <div class="card"><div id="usersWithData" class="metric">-</div><div class="label">有地点的用户</div></div>
        <div class="card"><div id="usersInProfiles" class="metric">-</div><div class="label">注册过的设备档案</div></div>
        <div class="card"><div id="totalPlaces" class="metric">-</div><div class="label">总地点数</div></div>
        <div class="card"><div id="addressHealth" class="metric">-</div><div class="label">地址完整度</div></div>
      </div>

      <div class="panel storage">
        <div>
          <span id="persistentBadge" class="badge">检查中</span>
          <code id="dbPath"></code>
        </div>
        <div>
          <span id="mountBadge" class="badge">检查中</span>
          <code id="uploadDir"></code>
        </div>
      </div>

      <div class="panel search-panel">
        <form id="searchForm" class="search-form">
          <div>
            <label for="searchInput">搜索用户旧数据</label>
            <input id="searchInput" type="search" placeholder="输入店名、城市、地址或设备 ID 片段" />
          </div>
          <button id="searchButton" type="submit">搜索</button>
        </form>
        <div id="searchStatus" class="status"></div>
        <div id="searchResults" class="search-results"></div>
      </div>

      <div class="panel search-panel">
        <form id="migrateForm" class="migrate-form">
          <div>
            <label for="fromDeviceInput">旧设备 ID</label>
            <input id="fromDeviceInput" type="text" placeholder="要恢复的数据来自这个 ID" />
          </div>
          <div>
            <label for="toDeviceInput">新设备 ID</label>
            <input id="toDeviceInput" type="text" placeholder="复制到用户现在的 ID" />
          </div>
          <button id="migrateButton" type="submit">复制恢复</button>
        </form>
        <div id="migrateStatus" class="status">只复制地点到新 ID，不删除旧 ID 数据。</div>
      </div>

      <div id="emptyState" class="empty hidden">还没有用户地点数据。</div>
      <table id="usersTable" class="hidden">
        <thead>
          <tr>
            <th>用户设备</th>
            <th>地点数</th>
            <th>有坐标</th>
            <th>有地址</th>
            <th>首次保存</th>
            <th>最近保存</th>
            <th>存过地点</th>
          </tr>
        </thead>
        <tbody id="usersBody"></tbody>
      </table>
    </section>
  </main>

  <script>
    const loginPanel = document.getElementById("loginPanel");
    const dashboard = document.getElementById("dashboard");
    const headerActions = document.getElementById("headerActions");
    const loginForm = document.getElementById("loginForm");
    const loginButton = document.getElementById("loginButton");
    const loginStatus = document.getElementById("loginStatus");
    const refreshButton = document.getElementById("refreshButton");
    const logoutButton = document.getElementById("logoutButton");
    const showIds = document.getElementById("showIds");
    const usersBody = document.getElementById("usersBody");
    const usersTable = document.getElementById("usersTable");
    const emptyState = document.getElementById("emptyState");
    const searchForm = document.getElementById("searchForm");
    const searchInput = document.getElementById("searchInput");
    const searchButton = document.getElementById("searchButton");
    const searchStatus = document.getElementById("searchStatus");
    const searchResults = document.getElementById("searchResults");
    const migrateForm = document.getElementById("migrateForm");
    const fromDeviceInput = document.getElementById("fromDeviceInput");
    const toDeviceInput = document.getElementById("toDeviceInput");
    const migrateButton = document.getElementById("migrateButton");
    const migrateStatus = document.getElementById("migrateStatus");
    const placesCache = new Map();
    let latestSearchResults = [];

    function setLoginVisible(visible) {
      loginPanel.classList.toggle("hidden", !visible);
      dashboard.classList.toggle("hidden", visible);
      headerActions.classList.toggle("hidden", visible);
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function number(value) {
      return Number(value ?? 0).toLocaleString("zh-CN");
    }

    function percent(part, total) {
      if (!total) return "-";
      return `${Math.round((Number(part || 0) / Number(total)) * 100)}%`;
    }

    function maskDeviceId(value) {
      const text = String(value || "").trim();
      if (showIds.checked || text.length <= 12) return text;
      return `${text.slice(0, 8)}...${text.slice(-4)}`;
    }

    function renderPlaceList(places) {
      if (!Array.isArray(places) || places.length === 0) {
        return `<div class="empty">这个用户还没有地点记录。</div>`;
      }
      return `
        <div class="places-note">下面只展示用户存过的地点记录，不包含设备地址或 IP。</div>
        <div class="place-list">
          ${places.map(place => `
            <div class="place-item">
              <div>
                <div class="place-name">${escapeHtml(place.name || place.brand || "未命名地点")}</div>
                <div class="place-meta">${escapeHtml(place.created_at || "-")}</div>
              </div>
              <div>
                <div class="place-address">${escapeHtml(place.address || "地址未记录")}</div>
                <div class="place-meta">${escapeHtml(place.city || "")}${place.has_coordinates ? ` · ${escapeHtml(place.lat)}, ${escapeHtml(place.lon)}` : ""}</div>
              </div>
              <div class="place-actions">
                ${place.map_url ? `<a href="${escapeHtml(place.map_url)}" target="_blank" rel="noreferrer">地图</a>` : ""}
                ${place.image_url ? `<a href="${escapeHtml(place.image_url)}" target="_blank" rel="noreferrer">截图</a>` : ""}
              </div>
            </div>
          `).join("")}
        </div>
      `;
    }

    function renderSearchResults(results) {
      latestSearchResults = Array.isArray(results) ? results : [];
      if (latestSearchResults.length === 0) {
        searchResults.innerHTML = "";
        return;
      }

      searchResults.innerHTML = latestSearchResults.map(result => `
        <div class="search-result">
          <div>
            <div class="place-name">${escapeHtml(result.name || result.brand || "未命名地点")}</div>
            <div class="place-meta">${escapeHtml(result.created_at || "-")}</div>
          </div>
          <div>
            <div class="place-address">${escapeHtml(result.address || "地址未记录")}</div>
            <div class="place-meta">${escapeHtml(result.city || "")}${result.has_coordinates ? ` · ${escapeHtml(result.lat)}, ${escapeHtml(result.lon)}` : ""}</div>
          </div>
          <div>
            <div class="label">用户设备</div>
            <div class="search-device">${escapeHtml(maskDeviceId(result.device_id))}</div>
          </div>
          <div class="place-actions">
            <button class="secondary small fill-old-device-button" type="button" data-device-id="${escapeHtml(result.device_id)}">填旧 ID</button>
            ${result.map_url ? `<a href="${escapeHtml(result.map_url)}" target="_blank" rel="noreferrer">地图</a>` : ""}
            ${result.image_url ? `<a href="${escapeHtml(result.image_url)}" target="_blank" rel="noreferrer">截图</a>` : ""}
          </div>
        </div>
      `).join("");
    }

    function renderStats(data) {
      setLoginVisible(false);
      document.getElementById("usersWithData").textContent = number(data.users_with_data);
      document.getElementById("usersInProfiles").textContent = number(data.users_in_profiles);
      document.getElementById("totalPlaces").textContent = number(data.total_places);
      document.getElementById("addressHealth").textContent = percent(data.places_with_address, data.total_places);
      document.getElementById("updatedAt").textContent = `更新于 ${new Date().toLocaleString("zh-CN")}`;

      const persistentBadge = document.getElementById("persistentBadge");
      persistentBadge.textContent = data.is_likely_persistent ? "数据库已持久化" : "数据库未确认持久化";
      persistentBadge.classList.toggle("warn", !data.is_likely_persistent);

      const mountBadge = document.getElementById("mountBadge");
      mountBadge.textContent = data.persistent_data_dir_is_mount ? "/data 已挂载 Volume" : "/data 未确认挂载";
      mountBadge.classList.toggle("warn", !data.persistent_data_dir_is_mount);

      document.getElementById("dbPath").textContent = `DB: ${data.db_path || "-"}`;
      document.getElementById("uploadDir").textContent = `Uploads: ${data.upload_dir || "-"}`;

      const users = Array.isArray(data.users) ? data.users : [];
      emptyState.classList.toggle("hidden", users.length > 0);
      usersTable.classList.toggle("hidden", users.length === 0);
      usersBody.innerHTML = users.map((user, index) => `
        <tr>
          <td class="device">${escapeHtml(maskDeviceId(user.device_id))}</td>
          <td>${number(user.place_count)}</td>
          <td>${number(user.places_with_coordinates)}</td>
          <td>${number(user.places_with_address)}</td>
          <td>${escapeHtml(user.first_created_at || "-")}</td>
          <td>${escapeHtml(user.last_created_at || "-")}</td>
          <td><button class="secondary small places-button" type="button" data-device-id="${escapeHtml(user.device_id)}" data-target="places-${index}">展开</button></td>
        </tr>
        <tr id="places-${index}" class="places-row hidden">
          <td colspan="7">
            <div class="places-panel">
              <div class="places-note">正在读取地点...</div>
            </div>
          </td>
        </tr>
      `).join("");
    }

    async function loadStats() {
      refreshButton.disabled = true;
      const includeIds = "true";
      try {
        const response = await fetch(`/api/admin/stats?include_device_ids=${includeIds}&limit=1000`, {
          credentials: "same-origin"
        });
        if (response.status === 403 || response.status === 503) {
          setLoginVisible(true);
          loginStatus.textContent = response.status === 503 ? "后端还没有配置 ADMIN_TOKEN。" : "";
          loginStatus.classList.toggle("error", response.status === 503);
          return;
        }
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        renderStats(data);
      } catch (error) {
        setLoginVisible(true);
        loginStatus.textContent = `读取失败：${error.message}`;
        loginStatus.classList.add("error");
      } finally {
        refreshButton.disabled = false;
      }
    }

    async function toggleUserPlaces(button) {
      const deviceId = button.dataset.deviceId;
      const target = document.getElementById(button.dataset.target);
      if (!deviceId || !target) return;

      const isHidden = target.classList.contains("hidden");
      if (!isHidden) {
        target.classList.add("hidden");
        button.textContent = "展开";
        return;
      }

      target.classList.remove("hidden");
      button.textContent = "收起";

      const panel = target.querySelector(".places-panel");
      if (placesCache.has(deviceId)) {
        panel.innerHTML = renderPlaceList(placesCache.get(deviceId));
        return;
      }

      panel.innerHTML = `<div class="places-note">正在读取地点...</div>`;
      try {
        const response = await fetch(`/api/admin/users/${encodeURIComponent(deviceId)}/places?limit=500`, {
          credentials: "same-origin"
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        const places = Array.isArray(data.places) ? data.places : [];
        placesCache.set(deviceId, places);
        panel.innerHTML = renderPlaceList(places);
      } catch (error) {
        panel.innerHTML = `<div class="places-note">读取失败：${escapeHtml(error.message)}</div>`;
      }
    }

    async function searchPlaces(event) {
      event.preventDefault();
      const query = searchInput.value.trim();
      if (query.length < 2) {
        searchStatus.textContent = "至少输入 2 个字。";
        searchStatus.classList.add("error");
        renderSearchResults([]);
        return;
      }

      searchButton.disabled = true;
      searchStatus.textContent = "正在搜索...";
      searchStatus.classList.remove("error");
      renderSearchResults([]);

      try {
        const response = await fetch(`/api/admin/search?q=${encodeURIComponent(query)}&limit=100`, {
          credentials: "same-origin"
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        const results = Array.isArray(data.results) ? data.results : [];
        searchStatus.textContent = `找到 ${results.length} 条匹配记录`;
        renderSearchResults(results);
      } catch (error) {
        searchStatus.textContent = `搜索失败：${error.message}`;
        searchStatus.classList.add("error");
      } finally {
        searchButton.disabled = false;
      }
    }

    async function migrateUserData(event) {
      event.preventDefault();
      const fromDeviceId = fromDeviceInput.value.trim();
      const toDeviceId = toDeviceInput.value.trim();

      migrateStatus.classList.remove("error");
      if (fromDeviceId.length < 8 || toDeviceId.length < 8) {
        migrateStatus.textContent = "旧设备 ID 和新设备 ID 都至少需要 8 位。";
        migrateStatus.classList.add("error");
        return;
      }
      if (fromDeviceId === toDeviceId) {
        migrateStatus.textContent = "旧设备 ID 和新设备 ID 不能相同。";
        migrateStatus.classList.add("error");
        return;
      }

      const ok = window.confirm("确认把旧设备 ID 的地点复制到新设备 ID？旧数据不会删除。");
      if (!ok) return;

      migrateButton.disabled = true;
      migrateStatus.textContent = "正在复制恢复...";

      try {
        const response = await fetch("/api/admin/migrate", {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            from_device_id: fromDeviceId,
            to_device_id: toDeviceId,
            dry_run: false
          })
        });
        if (!response.ok) {
          const message = response.status === 404 ? "旧设备 ID 没有地点记录" : `HTTP ${response.status}`;
          throw new Error(message);
        }
        const data = await response.json();
        placesCache.delete(toDeviceId);
        migrateStatus.textContent = `恢复完成：复制 ${data.copied_count} 个地点，跳过 ${data.skipped_duplicate_count} 个重复。旧 ID 数据仍保留。`;
        await loadStats();
      } catch (error) {
        migrateStatus.textContent = `恢复失败：${error.message}`;
        migrateStatus.classList.add("error");
      } finally {
        migrateButton.disabled = false;
      }
    }

    loginForm.addEventListener("submit", async event => {
      event.preventDefault();
      loginStatus.textContent = "正在验证...";
      loginStatus.classList.remove("error");
      loginButton.disabled = true;
      try {
        const token = document.getElementById("tokenInput").value.trim();
        const response = await fetch("/api/admin/login", {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token })
        });
        if (!response.ok) {
          throw new Error(response.status === 403 ? "Token 不对" : `HTTP ${response.status}`);
        }
        document.getElementById("tokenInput").value = "";
        await loadStats();
      } catch (error) {
        loginStatus.textContent = `登录失败：${error.message}`;
        loginStatus.classList.add("error");
      } finally {
        loginButton.disabled = false;
      }
    });

    refreshButton.addEventListener("click", loadStats);
    showIds.addEventListener("change", () => {
      const rows = Array.from(usersBody.querySelectorAll("tr")).filter(row => !row.classList.contains("places-row"));
      rows.forEach(row => {
        const button = row.querySelector(".places-button");
        const cell = row.querySelector(".device");
        if (button && cell) cell.textContent = maskDeviceId(button.dataset.deviceId);
      });
      renderSearchResults(latestSearchResults);
    });
    searchForm.addEventListener("submit", searchPlaces);
    migrateForm.addEventListener("submit", migrateUserData);
    searchResults.addEventListener("click", event => {
      const button = event.target.closest(".fill-old-device-button");
      if (!button) return;
      fromDeviceInput.value = button.dataset.deviceId || "";
      migrateStatus.classList.remove("error");
      migrateStatus.textContent = "已填入旧设备 ID。请再填新设备 ID 后复制恢复。";
    });
    usersBody.addEventListener("click", event => {
      const button = event.target.closest(".places-button");
      if (button) toggleUserPlaces(button);
    });
    logoutButton.addEventListener("click", async () => {
      await fetch("/api/admin/logout", { method: "POST", credentials: "same-origin" });
      setLoginVisible(true);
    });

    loadStats();
  </script>
</body>
</html>
"""


@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard():
    return HTMLResponse(ADMIN_DASHBOARD_HTML)


@app.post("/api/admin/login")
def admin_login(payload: AdminLoginRequest, response: Response):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN_NOT_CONFIGURED")
    token = payload.token.strip()
    if not hmac.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="ADMIN_FORBIDDEN")
    response.set_cookie(
        key=ADMIN_COOKIE_NAME,
        value=token,
        max_age=30 * 24 * 60 * 60,
        httponly=True,
        secure=APP_ENV == "production",
        samesite="lax",
        path="/",
    )
    return {"ok": True}


@app.post("/api/admin/logout")
def admin_logout(response: Response):
    response.delete_cookie(key=ADMIN_COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/healthz")
def healthz():
    storage_uses_persistent_dir = (
        os.path.abspath(DB_PATH).startswith(os.path.abspath(PERSISTENT_DATA_DIR) + os.sep)
        and os.path.abspath(UPLOAD_DIR).startswith(os.path.abspath(PERSISTENT_DATA_DIR) + os.sep)
    )
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat() + "Z",
        "db_path": DB_PATH,
        "upload_dir": UPLOAD_DIR,
        "persistent_data_dir": PERSISTENT_DATA_DIR,
        "storage_uses_persistent_dir": storage_uses_persistent_dir,
        "persistent_data_dir_exists": os.path.isdir(PERSISTENT_DATA_DIR),
        "persistent_data_dir_is_mount": os.path.ismount(PERSISTENT_DATA_DIR),
    }


@app.get("/api/app-config")
def app_config(platform: str = "ios", bundle_id: str = "", version: str = ""):
    is_ios = not platform or platform.lower() == "ios"
    return {
        "min_supported_version": MIN_SUPPORTED_IOS_VERSION if is_ios else "",
        "latest_version": LATEST_IOS_VERSION if is_ios else "",
        "app_store_url": APP_STORE_URL,
        "enforce_after": FORCE_UPDATE_AFTER,
        "enforce_when_app_store_version_available": ENFORCE_WHEN_APP_STORE_VERSION_AVAILABLE,
        "force_latest_app_store_version": FORCE_LATEST_APP_STORE_VERSION,
        "message": FORCE_UPDATE_MESSAGE,
    }


@app.get("/api/admin/stats")
def admin_stats(request: Request, limit: int = 100, include_device_ids: bool = False):
    require_admin(request)
    limit = max(1, min(limit, 1000))

    with get_db() as conn:
        cursor = conn.cursor()
        users_with_data = cursor.execute(
            """
            SELECT COUNT(DISTINCT device_id)
            FROM memory_pool
            WHERE device_id IS NOT NULL AND TRIM(device_id) != ''
            """
        ).fetchone()[0]
        total_places = cursor.execute("SELECT COUNT(*) FROM memory_pool").fetchone()[0]
        users_in_profiles = cursor.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0]
        places_with_coordinates = cursor.execute(
            """
            SELECT COUNT(*)
            FROM memory_pool
            WHERE poi_lat != 0.0 AND poi_lon != 0.0
            """
        ).fetchone()[0]
        places_with_address = cursor.execute(
            """
            SELECT COUNT(*)
            FROM memory_pool
            WHERE TRIM(COALESCE(amap_address, '')) != ''
            """
        ).fetchone()[0]
        top_users = cursor.execute(
            """
            SELECT
                device_id,
                COUNT(*) AS place_count,
                MIN(created_at) AS first_created_at,
                MAX(created_at) AS last_created_at,
                SUM(CASE WHEN poi_lat != 0.0 AND poi_lon != 0.0 THEN 1 ELSE 0 END) AS places_with_coordinates,
                SUM(CASE WHEN TRIM(COALESCE(amap_address, '')) != '' THEN 1 ELSE 0 END) AS places_with_address
            FROM memory_pool
            WHERE device_id IS NOT NULL AND TRIM(device_id) != ''
            GROUP BY device_id
            ORDER BY place_count DESC, last_created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return {
        "db_path": DB_PATH,
        "upload_dir": UPLOAD_DIR,
        "persistent_data_dir": PERSISTENT_DATA_DIR,
        "is_likely_persistent": (
            os.path.abspath(DB_PATH).startswith(os.path.abspath(PERSISTENT_DATA_DIR) + os.sep)
            and os.path.abspath(UPLOAD_DIR).startswith(os.path.abspath(PERSISTENT_DATA_DIR) + os.sep)
        ),
        "persistent_data_dir_exists": os.path.isdir(PERSISTENT_DATA_DIR),
        "persistent_data_dir_is_mount": os.path.ismount(PERSISTENT_DATA_DIR),
        "users_with_data": users_with_data,
        "users_in_profiles": users_in_profiles,
        "total_places": total_places,
        "places_with_coordinates": places_with_coordinates,
        "places_with_address": places_with_address,
        "users": [
            {
                "device_id": row["device_id"] if include_device_ids else masked_device_id(row["device_id"]),
                "place_count": row["place_count"],
                "places_with_coordinates": row["places_with_coordinates"],
                "places_with_address": row["places_with_address"],
                "first_created_at": row["first_created_at"],
                "last_created_at": row["last_created_at"],
            }
            for row in top_users
        ],
    }


@app.get("/api/admin/search")
def admin_search(request: Request, q: str, limit: int = 100):
    require_admin(request)
    query = (q or "").strip()
    if len(query) < 2:
        raise HTTPException(status_code=400, detail="QUERY_TOO_SHORT")
    limit = max(1, min(limit, 300))
    pattern = f"%{escape_sql_like(query.lower())}%"

    with get_db() as conn:
        cursor = conn.cursor()
        rows = cursor.execute(
            """
            SELECT
                id,
                brand,
                city,
                created_at,
                image_url,
                device_id,
                poi_lat,
                poi_lon,
                amap_name,
                amap_address,
                amap_location,
                amap_district
            FROM memory_pool
            WHERE device_id IS NOT NULL
              AND TRIM(device_id) != ''
              AND (
                LOWER(COALESCE(device_id, '')) LIKE ? ESCAPE '\\'
                OR LOWER(COALESCE(brand, '')) LIKE ? ESCAPE '\\'
                OR LOWER(COALESCE(city, '')) LIKE ? ESCAPE '\\'
                OR LOWER(COALESCE(amap_name, '')) LIKE ? ESCAPE '\\'
                OR LOWER(COALESCE(amap_address, '')) LIKE ? ESCAPE '\\'
                OR LOWER(COALESCE(amap_location, '')) LIKE ? ESCAPE '\\'
                OR LOWER(COALESCE(amap_district, '')) LIKE ? ESCAPE '\\'
              )
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (pattern, pattern, pattern, pattern, pattern, pattern, pattern, limit),
        ).fetchall()

    results = [admin_place_payload(row, include_device_id=True) for row in rows]
    return {
        "query": query,
        "result_count": len(results),
        "matched_user_count": len({result["device_id"] for result in results}),
        "results": results,
    }


@app.post("/api/admin/migrate")
def admin_migrate_user_data(payload: AdminMigrateRequest, request: Request):
    require_admin(request)
    from_device_id = payload.from_device_id.strip()
    to_device_id = payload.to_device_id.strip()
    if from_device_id == to_device_id:
        raise HTTPException(status_code=400, detail="SAME_DEVICE_ID")

    with get_db() as conn:
        cursor = conn.cursor()
        source_rows = cursor.execute(
            """
            SELECT
                id,
                brand,
                city,
                created_at,
                image_url,
                device_id,
                poi_lat,
                poi_lon,
                amap_name,
                amap_address,
                amap_location,
                amap_district
            FROM memory_pool
            WHERE device_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (from_device_id,),
        ).fetchall()

        if not source_rows:
            raise HTTPException(status_code=404, detail="SOURCE_USER_HAS_NO_PLACES")

        target_rows = cursor.execute(
            """
            SELECT
                id,
                brand,
                city,
                created_at,
                image_url,
                device_id,
                poi_lat,
                poi_lon,
                amap_name,
                amap_address,
                amap_location,
                amap_district
            FROM memory_pool
            WHERE device_id = ?
            """,
            (to_device_id,),
        ).fetchall()

        existing_keys = {admin_place_duplicate_key(row) for row in target_rows}
        rows_to_copy = []
        skipped_duplicates = 0
        for row in source_rows:
            key = admin_place_duplicate_key(row)
            if key in existing_keys:
                skipped_duplicates += 1
                continue
            existing_keys.add(key)
            rows_to_copy.append(row)

        copied_ids = []
        if not payload.dry_run:
            get_or_create_user_profile(conn, to_device_id)
            for row in rows_to_copy:
                cursor.execute(
                    """
                    INSERT INTO memory_pool (
                        brand,
                        city,
                        image_url,
                        created_at,
                        device_id,
                        poi_lat,
                        poi_lon,
                        amap_name,
                        amap_address,
                        amap_location,
                        amap_district
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["brand"],
                        row["city"],
                        row["image_url"],
                        row["created_at"],
                        to_device_id,
                        row["poi_lat"],
                        row["poi_lon"],
                        row["amap_name"],
                        row["amap_address"],
                        row["amap_location"],
                        row["amap_district"],
                    ),
                )
                copied_ids.append(cursor.lastrowid)

    return {
        "status": "dry_run" if payload.dry_run else "success",
        "from_device_id": from_device_id,
        "to_device_id": to_device_id,
        "source_place_count": len(source_rows),
        "target_existing_place_count": len(target_rows),
        "skipped_duplicate_count": skipped_duplicates,
        "would_copy_count": len(rows_to_copy),
        "copied_count": 0 if payload.dry_run else len(copied_ids),
        "copied_ids": [] if payload.dry_run else copied_ids,
    }


@app.get("/api/admin/users/{device_id}/places")
def admin_user_places(device_id: str, request: Request, limit: int = 200):
    require_admin(request)
    device_id = device_id.strip()
    if not device_id:
        raise HTTPException(status_code=400, detail="DEVICE_ID_REQUIRED")
    limit = max(1, min(limit, 1000))

    with get_db() as conn:
        cursor = conn.cursor()
        rows = cursor.execute(
            """
            SELECT
                id,
                brand,
                city,
                created_at,
                image_url,
                poi_lat,
                poi_lon,
                amap_name,
                amap_address,
                amap_location,
                amap_district
            FROM memory_pool
            WHERE device_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (device_id, limit),
        ).fetchall()

    places = [admin_place_payload(row) for row in rows]

    return {
        "device_id": device_id,
        "place_count": len(places),
        "places": places,
    }


@app.get("/api/tips")
async def get_input_tips(keyword: str, lat: float, lon: float, request: Request):
    await enforce_rate_limit(request, "tips", TIPS_RATE_LIMIT_PER_MIN)
    validate_lat_lon(lat, lon)
    keyword = (keyword or "").strip()
    if not keyword:
        return []
    if not is_mainland_china_coordinate(lon, lat):
        logger.info("Skipping Amap inputtips for non-China coordinate | keyword=%s | lat=%s | lon=%s", keyword, lat, lon)
        return []
    if not AMAP_KEY:
        raise HTTPException(status_code=500, detail="AMAP_KEY_MISSING")

    gcj_lon, gcj_lat = wgs84_to_gcj02(lon, lat)
    url = "https://restapi.amap.com/v3/assistant/inputtips"
    params = {"key": AMAP_KEY, "keywords": keyword[:80], "location": f"{gcj_lon},{gcj_lat}", "datatype": "all"}
    cache_params = {
        "keywords": keyword[:80],
        "location": rounded_coordinate_key(gcj_lon, gcj_lat),
        "datatype": "all",
    }

    try:
        res = await fetch_amap_json(
            url,
            params,
            cache_scope="inputtips",
            cache_params=cache_params,
            ttl_seconds=AMAP_TIPS_CACHE_TTL_SECONDS,
        )
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
    exact_address = (request_data.exact_address or "").strip()

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
    amap_name = ""
    amap_address = ""
    amap_location = ""
    amap_district = ""

    if exact_name and exact_location:
        brand_name = exact_name
        city_hint = exact_district
        poi_lon, poi_lat = parse_location_string(exact_location)
        amap_name = exact_name
        amap_address = exact_address
        amap_location = exact_location
        amap_district = exact_district
    else:
        if not (text_content or image_base64 or source_url):
            raise HTTPException(status_code=400, detail="MUST_PROVIDE_INPUT")

        client = build_openai_client()
        system_prompt = (
            "你在解析小红书、点评、地图分享的探店内容。"
            "请提取最可能的【核心品牌名】brand 和【城市】city。"
            "品牌名必须尽量短，只保留真正的店名、品牌主名，不要带菜系、分店、商圈、标点后缀。"
            "如果是“品牌名 + 商场/地标 + 店”，只保留品牌名，例如“和它交个朋友 百脑汇店”应提取为 brand=和它交个朋友。"
            "如果图片和文字冲突，优先相信图片里的店名或商户卡片。"
            "如果是纯文本社媒文案，也要尽量还原最可能的店名，不要轻易返回空字符串。"
            "若文本里出现徐家汇、陆家嘴、静安寺、南京东路等地标，可将 city 推断为上海。"
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

            if not brand_name and text_content:
                social_text_prompt = (
                    "你正在解析小红书/点评口语化探店文案。"
                    "任务是尽量还原最可能的店名主名，而不是保守返回空。"
                    "如果出现“品牌名 + 商场/地标 + 店”，只提取品牌名。"
                    "例如“和它交个朋友 百脑汇店”应输出 "
                    "{\"brand\":\"和它交个朋友\",\"city\":\"上海\"}。"
                    "若看到 #上海、上海citywalk、徐家汇、百脑汇 等线索，可推断 city=上海。"
                    "只返回 JSON。"
                )
                logger.info("ai extract empty, retrying social-text prompt")
                brand_name, social_city_hint = await run_ai_extract(
                    client,
                    social_text_prompt,
                    text_content=text_content,
                    image_base64="",
                    source_url=source_url,
                    image_only_fallback=False,
                )
                city_hint = city_hint or social_city_hint

            if not brand_name and text_content:
                heuristic_brand = extract_brand_from_social_text(text_content)
                heuristic_city = infer_city_from_social_text(text_content)
                logger.info(
                    "social heuristic fallback | brand=%s | city=%s",
                    heuristic_brand,
                    heuristic_city,
                )
                brand_name = heuristic_brand or brand_name
                city_hint = city_hint or heuristic_city

            logger.info("ai extract result | brand=%s | city=%s", brand_name, city_hint)

            if not brand_name:
                raise HTTPException(status_code=422, detail="AI_EXTRACT_EMPTY")

            increment_ai_usage(device_id)
        except HTTPException:
            raise
        except Exception:
            logger.exception("AI parse failed")
            raise HTTPException(status_code=500, detail="AI_PARSE_FAILED")

        if is_mainland_china_coordinate(request_data.lon, request_data.lat):
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
            cache_params = {
                "keywords": brand_name,
                "city": city_hint,
                "location": rounded_coordinate_key(upload_gcj_lon, upload_gcj_lat),
            }
            try:
                search_res = await fetch_amap_json(
                    amap_url,
                    params,
                    cache_scope="place_text",
                    cache_params=cache_params,
                    ttl_seconds=AMAP_PLACE_CACHE_TTL_SECONDS,
                )
                if search_res.get("status") == "1" and search_res.get("pois"):
                    poi = search_res["pois"][0]
                    loc_str = normalize_amap_value(poi.get("location", ""))
                    amap_name = normalize_amap_value(poi.get("name")) or brand_name
                    amap_address = normalize_amap_value(poi.get("address"))
                    amap_location = loc_str
                    amap_district = (
                        normalize_amap_value(poi.get("adname"))
                        or normalize_amap_value(poi.get("district"))
                        or city_hint
                    )
                    city_hint = city_hint or amap_district
                    if "," in loc_str:
                        gcj_p_lon, gcj_p_lat = map(float, loc_str.split(","))
                        poi_lon, poi_lat = gcj02_to_wgs84(gcj_p_lon, gcj_p_lat)
                else:
                    logger.info("Amap place text no result | brand=%s | city=%s", brand_name, city_hint)
            except Exception:
                logger.exception("Amap place search failed")
        else:
            logger.info("Skipping Amap place search for non-China coordinate | brand=%s", brand_name)

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

    amap_name = amap_name or brand_name
    amap_district = amap_district or city_hint

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO memory_pool (
                brand, city, image_url, created_at, device_id, poi_lat, poi_lon,
                amap_name, amap_address, amap_location, amap_district
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean_brand,
                city_hint,
                image_url,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                device_id,
                poi_lat,
                poi_lon,
                amap_name,
                amap_address,
                amap_location,
                amap_district,
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
        "amap_name": amap_name,
        "amap_address": amap_address,
        "amap_location": amap_location,
        "amap_district": amap_district,
        "is_immediate_nearby": dist_to_upload <= 500,
    }


@app.post("/api/nearby")
async def discover_nearby(request_data: NearbyRequest, request: Request):
    await enforce_rate_limit(request, "nearby", 20)
    validate_lat_lon(request_data.lat, request_data.lon)
    if request_data.lat == 0.0 and request_data.lon == 0.0:
        return []

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT brand, city, poi_lat, poi_lon, amap_name, amap_address
            FROM memory_pool
            WHERE device_id = ? AND poi_lat != 0.0 AND poi_lon != 0.0
            ORDER BY created_at DESC
            """,
            (request_data.device_id,),
        )
        my_memories = cursor.fetchall()
        cursor.execute(
            """
            SELECT brand, poi_lat, poi_lon, amap_name, amap_address, COUNT(DISTINCT device_id) AS wish_count
            FROM memory_pool
            WHERE device_id != ? AND poi_lat != 0.0 AND poi_lon != 0.0
            GROUP BY brand, poi_lat, poi_lon, amap_name, amap_address
            HAVING wish_count >= ?
            """,
            (request_data.device_id, BLIND_BOX_THRESHOLD),
        )
        public_boxes = cursor.fetchall()

    nearby_results = []

    for memory in my_memories:
        dist = calculate_haversine_distance(request_data.lon, request_data.lat, memory["poi_lon"], memory["poi_lat"])
        if dist <= 3000:
            nearby_results.append(
                {
                    "brand": memory["brand"],
                    "name": memory["amap_name"] or memory["brand"],
                    "address": memory["amap_address"] or memory["city"],
                    "distance": str(dist),
                    "lat": memory["poi_lat"],
                    "lon": memory["poi_lon"],
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
                    "name": p["amap_name"] or f"神秘盲盒：{p['brand']}",
                    "address": p["amap_address"] or "周边高人气打卡地",
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
            """
            SELECT
                id, brand, city, created_at, image_url, poi_lat, poi_lon,
                amap_name, amap_address, amap_location, amap_district
            FROM memory_pool
            WHERE device_id = ?
            ORDER BY created_at DESC
            """,
            (device_id,),
        )
        rows = cursor.fetchall()
    return [
        {
            "id": r["id"],
            "brand": r["brand"],
            "city": r["city"],
            "created_at": r["created_at"],
            "image_url": r["image_url"],
            "poi_lat": r["poi_lat"],
            "poi_lon": r["poi_lon"],
            "amap_name": r["amap_name"] or r["brand"],
            "amap_address": r["amap_address"] or "",
            "amap_location": r["amap_location"] or "",
            "amap_district": r["amap_district"] or r["city"],
            "poi_name": r["amap_name"] or r["brand"],
            "poi_address": r["amap_address"] or "",
            "address": r["amap_address"] or "",
            "district": r["amap_district"] or r["city"],
            "location": r["amap_location"] or "",
        }
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
