import os
import uuid
import base64
import sqlite3
import json
import math
import httpx
import asyncio
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
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
    conn = sqlite3.connect("memories.db", timeout=10)
    cursor = conn.cursor()
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
            poi_lon REAL DEFAULT 0.0,
            is_public INTEGER DEFAULT 0
        )
    """)
    cursor.execute("PRAGMA table_info(memory_pool)")
    columns = [info[1] for info in cursor.fetchall()]
    if "poi_lat" not in columns:
        cursor.execute("ALTER TABLE memory_pool ADD COLUMN poi_lat REAL DEFAULT 0.0")
        cursor.execute("ALTER TABLE memory_pool ADD COLUMN poi_lon REAL DEFAULT 0.0")
    if "is_public" not in columns:
        cursor.execute("ALTER TABLE memory_pool ADD COLUMN is_public INTEGER DEFAULT 0")
    conn.commit()
    conn.close()

init_db()

class UploadRequest(BaseModel):
    image_base64: Optional[str] = ""
    text_content: Optional[str] = ""
    is_public: bool = False
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
    if not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271): return lng, lat
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
    if not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271): return lng, lat
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
    client = AsyncOpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # 🌟 AI 提示词重构：兼顾模糊文字理解
    system_prompt = """
    你是一个精准的位置实体提取引擎。请从用户的截图或模糊文本中提取【核心品牌名/地点名】（brand）和【城市】（city）。
    ⚠️ 规则：
    1. 如果是截图，提取招牌核心名字。
    2. 如果是用户输入的模糊描述（例如："五角场那家网红贝果"），请提取出最适合在地图上搜索的关键字（"网红贝果"），城市提取为"上海"（如果能推断出的话，否则填空）。
    必须以 JSON 返回，示例：{"brand": "鸟喆", "city": "上海"}。
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    if request_data.text_content:
        messages.append({"role": "user", "content": f"提取此文本中的地点：{request_data.text_content}"})
    elif request_data.image_base64:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": "提取图片中的核心商户及城市。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request_data.image_base64}"}}
        ]})
    else:
        raise HTTPException(status_code=400, detail="Must provide text or image")

    try:
        response = await client.chat.completions.create(
            model="qwen-vl-max" if request_data.image_base64 else "qwen-max",
            messages=messages,
            response_format={"type": "json_object"}
        )
        result_json = json.loads(response.choices[0].message.content)
        brand_name = result_json.get("brand")
        city_hint = result_json.get("city", "")
    except Exception as e:
        raise HTTPException(status_code=500, detail="AI 解析失败")

    poi_lat, poi_lon = 0.0, 0.0
    upload_gcj_lon, upload_gcj_lat = wgs84_to_gcj02(request_data.lon, request_data.lat)
    
    # 高德搜索
    amap_url = "https://restapi.amap.com/v3/place/text"
    params = {
        "key": AMAP_KEY, 
        "keywords": brand_name,
        "city": city_hint,
        "location": f"{upload_gcj_lon},{upload_gcj_lat}" 
    }
    
    try:
        async with httpx.AsyncClient() as http_client:
            search_res = (await http_client.get(amap_url, params=params)).json()
            
        if search_res.get("status") == "1" and search_res.get("pois"):
            poi = search_res["pois"][0] 
            loc_str = poi.get("location", "")
            brand_name = poi.get("name", brand_name) # 用高德的官方名字覆盖用户的模糊输入
            if "," in loc_str:
                gcj_p_lon, gcj_p_lat = map(float, loc_str.split(","))
                poi_lon, poi_lat = gcj02_to_wgs84(gcj_p_lon, gcj_p_lat)
    except Exception: pass

    # 保存图片（如果有）
    image_url = ""
    if request_data.image_base64:
        try:
            image_data = base64.b64decode(request_data.image_base64)
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join("uploads", filename)
            with open(filepath, "wb") as f: f.write(image_data)
            base_url = str(request.base_url).rstrip('/')
            image_url = f"{base_url}/uploads/{filename}"
        except Exception: pass

    is_pub = 1 if request_data.is_public else 0
    conn = sqlite3.connect("memories.db", timeout=10)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO memory_pool (brand, city, image_url, created_at, device_id, poi_lat, poi_lon, is_public) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (brand_name, city_hint, image_url, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), request_data.device_id, poi_lat, poi_lon, is_pub))
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
        "is_immediate_nearby": (dist_to_upload <= 500)
    }

@app.post("/api/nearby")
async def discover_nearby(request: NearbyRequest):
    if request.lat == 0.0 and request.lon == 0.0: return []

    conn = sqlite3.connect("memories.db", timeout=10)
    cursor = conn.cursor()
    
    # 🌟 取出自己的所有地点
    cursor.execute("SELECT DISTINCT brand FROM memory_pool WHERE device_id = ?", (request.device_id,))
    my_brands = [row[0] for row in cursor.fetchall()]
    
    # 🌟 取出别人公开的盲盒 (按坐标/品牌聚合，计算热度)
    cursor.execute("""
        SELECT brand, poi_lat, poi_lon, COUNT(*) as wish_count 
        FROM memory_pool 
        WHERE is_public = 1 AND device_id != ? AND poi_lat != 0.0
        GROUP BY brand, poi_lat, poi_lon
    """, (request.device_id,))
    public_boxes = cursor.fetchall()
    conn.close()

    gcj_lon, gcj_lat = wgs84_to_gcj02(request.lon, request.lat)
    amap_url = "https://restapi.amap.com/v3/place/around"
    nearby_results = []
    
    # 1. 动态查高德：找自己的店
    async def fetch_brand_nearby(client, brand_name):
        params = {"key": AMAP_KEY, "keywords": brand_name, "location": f"{gcj_lon},{gcj_lat}", "radius": "3000", "sortrule": "distance"}
        try:
            res = await client.get(amap_url, params=params)
            return brand_name, res.json()
        except Exception: return brand_name, None

    if my_brands:
        async with httpx.AsyncClient() as client:
            tasks = [fetch_brand_nearby(client, brand) for brand in my_brands]
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
                        nearby_results.append({
                            "name": poi.get("name", ""),
                            "address": poi.get("address", ""),
                            "distance": str(dist),
                            "lat": poi_lat, "lon": poi_lon,
                            "is_public": False, "wish_count": 1
                        })
                        
    # 2. 本地计算：找别人的盲盒（不用去高德查，因为存的时候坐标已经固定）
    for p_brand, p_lat, p_lon, w_count in public_boxes:
        dist = calculate_haversine_distance(request.lon, request.lat, p_lon, p_lat)
        if dist <= 3000:
            nearby_results.append({
                "name": f"神秘盲盒：{p_brand}",
                "address": "来自他人的公开胶囊",
                "distance": str(dist),
                "lat": p_lat, "lon": p_lon,
                "is_public": True, "wish_count": w_count
            })
            
    # 去重并排序
    unique_results = []
    seen = set()
    for item in sorted(nearby_results, key=lambda x: int(x["distance"])):
        # 相同名字和位置的算同一个
        identifier = f"{item['name']}_{item['lat']}_{item['lon']}"
        if identifier not in seen:
            seen.add(identifier)
            unique_results.append(item)
            
    return unique_results

@app.get("/api/memories")
def get_all_memories(device_id: str):
    conn = sqlite3.connect("memories.db", timeout=10)
    cursor = conn.cursor()
    cursor.execute("SELECT id, brand, city, created_at, image_url, is_public FROM memory_pool WHERE device_id = ? ORDER BY id DESC", (device_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "brand": r[1], "city": r[2], "created_at": r[3], "image_url": r[4], "is_public": bool(r[5])} for r in rows]

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
