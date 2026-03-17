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

# 开启跨域，确保 iOS 端访问顺畅
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载存储文件夹
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 🔑 你的 API Key (请确保这里是你真实的 Key)
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
    conn.commit()
    conn.close()

init_db()

class UploadRequest(BaseModel):
    image_base64: str

class NearbyRequest(BaseModel):
    lat: float
    lon: float

def calculate_haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return int(c * 6371000)

@app.post("/api/upload")
async def upload_memory(request_data: UploadRequest, request: Request):
    print(">>> 收到新截图，正在请求 AI 解析...")
    client = OpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    system_prompt = """
    你是一个精准的商业实体提取引擎。提取截图中的【核心品牌名】（brand）和城市（city）。
    ⚠️ 核心规则：只提取最核心的招牌名字！绝对不要带上菜系、分店名或标点符号后缀。
    例如：看到“鸟喆·烧鸟·炉端烧”，只提取“鸟喆”；看到“海底捞火锅(南京路店)”，只提取“海底捞”。
    必须以 JSON 返回，示例：{"brand": "鸟喆", "city": "上海"}。
    """
    try:
        response = client.chat.completions.create(
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

    # 🌟 核心升级：图片保存并动态适配公网 URL
    try:
        image_data = base64.b64decode(request_data.image_base64)
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join("uploads", filename)
        with open(filepath, "wb") as f:
            f.write(image_data)
        
        # 自动识别 vibe-backend.zeabur.app 域名
        base_url = str(request.base_url).rstrip('/')
        image_url = f"{base_url}/uploads/{filename}"
    except Exception as e:
        image_url = ""

    conn = sqlite3.connect("memories.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO memory_pool (brand, city, image_url, created_at) VALUES (?, ?, ?, ?)", 
                   (brand_name, city_hint, image_url, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    
    return {"status": "success", "message": f"成功保存：{brand_name}"}

@app.post("/api/nearby")
async def discover_nearby(request: NearbyRequest):
    conn = sqlite3.connect("memories.db")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT brand, city FROM memory_pool")
    saved_memories = cursor.fetchall()
    conn.close()

    if not saved_memories: return []

    amap_url = "https://restapi.amap.com/v3/place/text"
    nearby_results = []
    for brand_name, city_hint in saved_memories:
        params = {
            "key": AMAP_KEY, "keywords": brand_name,
            "city": city_hint if city_hint != "unknown" else "",
            "location": f"{request.lon},{request.lat}"
        }
        try:
            res = requests.get(amap_url, params=params).json()
            if res.get("status") == "1":
                for poi in res.get("pois", []):
                    # 模糊匹配逻辑：适配“大众点评名字”与“高德名字”不一致
                    core_brand = brand_name.split('·')[0].split('(')[0].split('（')[0].strip().lower()
                    core_poi = poi.get('name','').split('(')[0].split('（')[0].strip().lower()
                    
                    if core_brand in core_poi or core_poi in core_brand:
                        raw_dist = poi.get("distance")
                        dist = int(raw_dist) if (isinstance(raw_dist, str) and raw_dist.isdigit()) else 9999
                        if dist <= 3000:
                            nearby_results.append({
                                "name": poi.get("name", ""),
                                "address": poi.get("address", ""),
                                "distance": str(dist)
                            })
        except: continue
    return nearby_results

@app.get("/api/memories")
def get_all_memories():
    conn = sqlite3.connect("memories.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, brand, city, created_at, image_url FROM memory_pool ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "brand": r[1], "city": r[2], "created_at": r[3], "image_url": r[4]} for r in rows]

@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: int):
    conn = sqlite3.connect("memories.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM memory_pool WHERE id = ?", (memory_id,))
    conn.commit()
    conn.close()
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
