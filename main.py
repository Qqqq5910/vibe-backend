import os
import uuid
import base64
import sqlite3
import json
import math
import httpx  # 🌟 核心替换：使用异步 HTTP 客户端代替 requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI  # 🌟 核心替换：使用异步 OpenAI 客户端
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

QWEN_API_KEY = "sk-caff47c35d20412c9561042bcbd14641"
AMAP_KEY = "2241cf1577bb0f2893404b727066270d"

def init_db():
    conn = sqlite3.connect("memories.db")
    cursor = conn.cursor()
    
    # 🌟 核心突破：开启 WAL 模式，允许多用户高并发读写，不再锁死数据库
    cursor.execute("PRAGMA journal_mode=WAL;")
    
    cursor.execute("""
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
    """)
    
    cursor.execute("PRAGMA table_info(memory_pool)")
    columns = [info[1] for info in cursor.fetchall()]
    if "poi_lat" not in columns:
        cursor.execute("ALTER TABLE memory_pool ADD COLUMN poi_lat REAL DEFAULT 0.0")
        cursor.execute("ALTER TABLE memory_pool ADD COLUMN poi_lon REAL DEFAULT 0.0")
        
    conn.commit()
    conn.close()

init_db()

class UploadRequest(BaseModel):
    image_base64: str
    device_id: str
    lat: float 
    lon: float

class NearbyRequest(BaseModel):
    lat: float
    lon: float
    device_id: str

pi = 3.1415926535897932384626
a = 6378245.0
ee = 0.00669342162296594323

def wgs84_to_gcj02(lng, lat):
    if not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271):
        return lng, lat
    def transform_lat(lng, lat):
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
        ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 * math.sin(2.0 * lng * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * pi) + 40.0 * math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 * math.sin(lat * pi / 30.0)) * 2.0 / 3.0
        return ret
    def transform_lng(lng, lat):
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
        ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 * math.sin(2.0 * lng * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lng * pi) + 40.0 * math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 * math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
        return ret
    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    return lng + dlng, lat + dlat

def gcj02_to_wgs84(lng, lat):
    if not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271):
        return lng, lat
    m_lng, m_lat = wgs84_to_gcj02(lng, lat)
    return lng * 2 - m_lng, lat * 2 - m_lat

def calculate_haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return int(c * 6371000)

@app.post("/api/upload")
async def upload_memory(request_data: UploadRequest, request: Request):
    # 🌟 核心并发突破：使用 AsyncOpenAI，释放主线程，不阻塞其他用户的雷达扫描
    client = AsyncOpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    system_prompt = """
    你是一个精准的商业实体提取引擎。提取截图中的【核心品牌名】（brand）和城市（city）。
    ⚠️ 核心规则：只提取最核心的招牌名字！绝对不要带上菜系、分店名或标点符号后缀。
    必须以 JSON 返回，示例：{"brand": "鸟喆", "city": "上海"}。
    """
    try:
        # await 将 I/O 等待时间交还给 FastAPI 调度器
        response = await client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "提取图片中的核心商户及城市。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request_data.image_base64}"}}
                ]}
            ],
            response_format={"type": "json_object"}
        )
        result_json = json.loads(response.choices[0].message.content)
        brand_name = result_json.get("brand")
        city_hint = result_json.get("city", "unknown")
    except Exception as e:
        raise HTTPException(status_code=500, detail="AI 解析失败")

    poi_lat, poi_lon = 0.0, 0.0
    upload_gcj_lon, upload_gcj_lat = wgs84_to_gcj02(request_data.lon, request_data.lat)
    
    amap_url = "https://restapi.amap.com/v3/place/text"
    params = {
        "key": AMAP_KEY, 
        "keywords": brand_name,
        "city": city_hint,
        "location": f"{upload_gcj_lon},{upload_gcj_lat}" 
    }
    
    try:
        # 🌟 核心并发突破：使用 httpx.AsyncClient 执行高德网络请求
        async with httpx.AsyncClient() as http_client:
            search_res = (await http_client.get(amap_url, params=params)).json()
            
        if search_res.get("status") == "1" and search_res.get("pois"):
            poi = search_res["pois"][0] 
            loc_str = poi.get("location", "")
            if "," in loc_str:
                gcj_p_lon, gcj_p_lat = map(float, loc_str.split(","))
                poi_lon, poi_lat = gcj02_to_wgs84(gcj_p_lon, gcj_p_lat)
    except Exception as e:
        print(f"⚠️ 寻找精准地理位置失败: {e}")

    try:
        image_data = base64.b64decode(request_data.image_base64)
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join("uploads", filename)
        with open(filepath, "wb") as f: f.write(image_data)
        base_url = str(request.base_url).rstrip('/')
        image_url = f"{base_url}/uploads/{filename}"
    except Exception as e: image_url = ""

    # 使用短连接快速写入，配合 WAL 模式极速释放
    conn = sqlite3.connect("memories.db", timeout=10)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO memory_pool (brand, city, image_url, created_at, device_id, poi_lat, poi_lon) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                   (brand_name, city_hint, image_url, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), request_data.device_id, poi_lat, poi_lon))
    conn.commit()
    conn.close()
    
    dist_to_upload = 9999
    if poi_lat != 0.0:
        dist_to_upload = calculate_haversine_distance(request_data.lon, request_data.lat, poi_lon, poi_lat)
    
    return {
        "status": "success", 
        "message": brand_name,
        "poi_lat": poi_lat, 
        "poi_lon": poi_lon,
        "is_immediate_nearby": (dist_to_upload <= 200) 
    }

@app.post("/api/nearby")
async def discover_nearby(request: NearbyRequest):
    if request.lat == 0.0 and request.lon == 0.0:
        return []

    # 直接查询该用户的所有胶囊
    conn = sqlite3.connect("memories.db", timeout=10)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT brand, city, poi_lat, poi_lon FROM memory_pool WHERE device_id = ?", (request.device_id,))
        saved_memories = cursor.fetchall()
    except sqlite3.OperationalError:
        saved_memories = []
    conn.close()

    if not saved_memories: 
        return []

    nearby_results = []
    
    # 🌟 完全依赖本地 CPU 微秒级运算，彻底摆脱多用户同时刷新高德造成的卡死或封 IP 风险
    for brand, city, poi_lat, poi_lon in saved_memories:
        if not poi_lat or poi_lat == 0.0:
            continue
            
        dist = calculate_haversine_distance(request.lon, request.lat, poi_lon, poi_lat)
        
        if dist <= 3000:
            nearby_results.append({
                "name": brand,
                "address": city,
                "distance": str(dist),
                "lat": poi_lat,
                "lon": poi_lon
            })
            
    nearby_results.sort(key=lambda x: int(x["distance"]))
    return nearby_results

@app.get("/api/memories")
def get_all_memories(device_id: str):
    conn = sqlite3.connect("memories.db", timeout=10)
    cursor = conn.cursor()
    cursor.execute("SELECT id, brand, city, created_at, image_url FROM memory_pool WHERE device_id = ? ORDER BY id DESC", (device_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "brand": r[1], "city": r[2], "created_at": r[3], "image_url": r[4]} for r in rows]

@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: int, device_id: str):
    conn = sqlite3.connect("memories.db", timeout=10)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM memory_pool WHERE id = ? AND device_id = ?", (memory_id, device_id))
    conn.commit()
    conn.close()
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
