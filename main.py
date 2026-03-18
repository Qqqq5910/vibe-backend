import os
import uuid
import base64
import sqlite3
import json
import math
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
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
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_pool (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT NOT NULL,
            city TEXT NOT NULL,
            image_url TEXT,
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute("PRAGMA table_info(memory_pool)")
    columns = [info[1] for info in cursor.fetchall()]
    if "device_id" not in columns:
        cursor.execute("ALTER TABLE memory_pool ADD COLUMN device_id TEXT DEFAULT 'anonymous'")
    conn.commit()
    conn.close()

init_db()

class UploadRequest(BaseModel):
    image_base64: str
    device_id: str

class NearbyRequest(BaseModel):
    lat: float
    lon: float
    device_id: str

pi = 3.1415926535897932384626
a = 6378245.0
ee = 0.00669342162296594323

# WGS84 转 GCJ02 (iOS 到 高德)
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

# 🌟 新增：GCJ02 转 WGS84 (高德 到 Apple MapKit)
def gcj02_to_wgs84(lng, lat):
    if not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271):
        return lng, lat
    m_lng, m_lat = wgs84_to_gcj02(lng, lat)
    # 利用偏移量进行近似逆推，对于 iOS 地图显示和 200米围栏精度已足够
    return lng * 2 - m_lng, lat * 2 - m_lat

@app.post("/api/upload")
async def upload_memory(request_data: UploadRequest, request: Request):
    client = OpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    system_prompt = """
    你是一个精准的商业实体提取引擎。提取截图中的【核心品牌名】（brand）和城市（city）。
    ⚠️ 核心规则：只提取最核心的招牌名字！绝对不要带上菜系、分店名或标点符号后缀。
    必须以 JSON 返回，示例：{"brand": "鸟喆", "city": "上海"}。
    """
    try:
        response = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": "提取图片中的核心商户及城市。"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request_data.image_base64}"}}]}
            ],
            response_format={"type": "json_object"}
        )
        result_json = json.loads(response.choices[0].message.content)
        brand_name = result_json.get("brand")
        city_hint = result_json.get("city", "unknown")
    except Exception as e:
        raise HTTPException(status_code=500, detail="AI 解析失败")

    try:
        image_data = base64.b64decode(request_data.image_base64)
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join("uploads", filename)
        with open(filepath, "wb") as f:
            f.write(image_data)
        base_url = str(request.base_url).rstrip('/')
        image_url = f"{base_url}/uploads/{filename}"
    except Exception as e:
        image_url = ""

    conn = sqlite3.connect("memories.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO memory_pool (brand, city, image_url, created_at, device_id) VALUES (?, ?, ?, ?, ?)", 
                   (brand_name, city_hint, image_url, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), request_data.device_id))
    conn.commit()
    conn.close()
    return {"status": "success", "message": f"成功保存：{brand_name}"}

@app.post("/api/nearby")
async def discover_nearby(request: NearbyRequest):
    if request.lat == 0.0 and request.lon == 0.0:
        return []

    gcj_lon, gcj_lat = wgs84_to_gcj02(request.lon, request.lat)
    
    conn = sqlite3.connect("memories.db")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT brand, city FROM memory_pool WHERE device_id = ?", (request.device_id,))
    saved_memories = cursor.fetchall()
    conn.close()

    if not saved_memories: 
        return []

    amap_url = "https://restapi.amap.com/v3/place/around"
    nearby_results = []
    
    for brand_name, city_hint in saved_memories:
        params = {
            "key": AMAP_KEY, 
            "keywords": brand_name,
            "location": f"{gcj_lon},{gcj_lat}",
            "radius": "3000",
            "sortrule": "distance"
        }
        try:
            res = requests.get(amap_url, params=params).json()
            if res.get("status") == "1":
                for poi in res.get("pois", []):
                    core_brand = brand_name.split('·')[0].split('(')[0].split('（')[0].strip().lower()
                    core_poi = poi.get('name','').split('(')[0].split('（')[0].strip().lower()
                    
                    if core_brand in core_poi or core_poi in core_brand:
                        raw_dist = poi.get("distance")
                        dist = int(raw_dist) if (isinstance(raw_dist, str) and raw_dist.isdigit()) else 9999
                        if dist <= 3000:
                            # 🌟 新增：解析高德坐标并转回 WGS84 给 iOS
                            loc_str = poi.get("location", "")
                            poi_lon, poi_lat = 0.0, 0.0
                            if "," in loc_str:
                                gcj_p_lon, gcj_p_lat = map(float, loc_str.split(","))
                                poi_lon, poi_lat = gcj02_to_wgs84(gcj_p_lon, gcj_p_lat)
                            
                            nearby_results.append({
                                "name": poi.get("name", ""),
                                "address": poi.get("address", ""),
                                "distance": str(dist),
                                "lat": poi_lat, # 传给前端用于打点和围栏
                                "lon": poi_lon
                            })
        except Exception: continue
    return nearby_results

@app.get("/api/memories")
def get_all_memories(device_id: str):
    conn = sqlite3.connect("memories.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, brand, city, created_at, image_url FROM memory_pool WHERE device_id = ? ORDER BY id DESC", (device_id,))
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "brand": r[1], "city": r[2], "created_at": r[3], "image_url": r[4]} for r in rows]

@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: int, device_id: str):
    conn = sqlite3.connect("memories.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM memory_pool WHERE id = ? AND device_id = ?", (memory_id, device_id))
    conn.commit()
    conn.close()
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
