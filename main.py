from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import math
import sqlite3
from openai import OpenAI
import uvicorn
from datetime import datetime

app = FastAPI()

# ⚠️ 替换为你的真实 Key
QWEN_API_KEY = "sk-caff47c35d20412c9561042bcbd14641"
AMAP_KEY = "2241cf1577bb0f2893404b727066270d"

# ==========================================
# 数据库初始化 (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect("memories.db")
    cursor = conn.cursor()
    # 创建记忆表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_pool (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT NOT NULL,
            city TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 数据模型与辅助函数
# ==========================================
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

# ==========================================
# 接口 1：上传与解析 (入库)
# ==========================================
@app.post("/api/upload")
def upload_memory(request: UploadRequest):
    print(">>> 收到新截图，正在请求大模型解析...")
    client = OpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    system_prompt = """
    你是一个精准的商业实体提取引擎。提取截图中的品牌名（brand）和城市（city）。
    必须以 JSON 返回，示例：{"brand": "W Coffee", "city": "上海"}。若无城市线索填"unknown"。
    绝不要输出任何多余字符。
    """
    try:
        response = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "提取图片中的核心商户及城市。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request.image_base64}"}}
                ]}
            ],
            response_format={"type": "json_object"}
        )
        result_json = json.loads(response.choices[0].message.content)
        brand_name = result_json.get("brand")
        city_hint = result_json.get("city", "unknown")
    except Exception as e:
        raise HTTPException(status_code=500, detail="图片解析失败")

    if not brand_name:
        raise HTTPException(status_code=400, detail="未提取到有效商铺信息")

    # 存入数据库
    conn = sqlite3.connect("memories.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO memory_pool (brand, city, created_at) VALUES (?, ?, ?)", 
                   (brand_name, city_hint, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    
    print(f"✅ 已存入记忆库: [{brand_name}] - [{city_hint}]")
    return {"status": "success", "message": f"成功保存：{brand_name}"}

# ==========================================
# 接口 2：发现周边 (出库召回)
# ==========================================
@app.post("/api/nearby")
def discover_nearby(request: NearbyRequest):
    print(f">>> 开始扫描周边: 经度 {request.lon}, 纬度 {request.lat}")
    
    # 1. 从数据库捞出所有去重后的品牌
    conn = sqlite3.connect("memories.db")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT brand, city FROM memory_pool")
    saved_memories = cursor.fetchall()
    conn.close()

    if not saved_memories:
        return []

    amap_url = "https://restapi.amap.com/v3/place/text"
    nearby_results = []
    
    # 设定唤醒阈值：只展示距离你 3000 米以内的收藏
    DISTANCE_THRESHOLD = 3000 

    # 2. 批量请求高德验证位置
    for brand_name, city_hint in saved_memories:
        params = {
            "key": AMAP_KEY,
            "keywords": brand_name,
            "city": city_hint if city_hint != "unknown" else "",
            "citylimit": "true" if city_hint != "unknown" else "false",
            "location": f"{request.lon},{request.lat}",
            "extensions": "base"
        }
        
        try:
            amap_resp = requests.get(amap_url, params=params).json()
            if amap_resp.get("status") == "1":
                for poi in amap_resp.get("pois", []):
                    # 严格名称过滤
                    if brand_name.lower().replace(" ", "") in poi['name'].lower().replace(" ", ""):
                        raw_dist = poi.get("distance")
                        final_dist = 99999999 # 默认极大值
                        
                        if isinstance(raw_dist, str) and raw_dist.isdigit():
                            final_dist = int(raw_dist)
                        else:
                            poi_location = poi.get("location")
                            if poi_location and "," in poi_location:
                                poi_lon, poi_lat = poi_location.split(",")
                                final_dist = calculate_haversine_distance(request.lon, request.lat, poi_lon, poi_lat)

                        # 【核心过滤】只有在这个阈值内的店，才会被“唤醒”推给前端
                        if final_dist <= DISTANCE_THRESHOLD:
                            nearby_results.append({
                                "name": poi.get("name", ""),
                                "address": f"{poi.get('pname', '')}{poi.get('adname', '')}{poi.get('address', '')}",
                                "distance": str(final_dist)
                            })
        except Exception as e:
            print(f"高德 API 请求异常 ({brand_name}): {e}")
            continue

    print(f"✅ 扫描完毕，发现 {len(nearby_results)} 家在附近。")
    return nearby_results
# ==========================================
# 接口 3：查看所有胶囊 (查询库)
# ==========================================
@app.get("/api/memories")
def get_all_memories():
    conn = sqlite3.connect("memories.db")
    cursor = conn.cursor()
    # 按存入时间倒序拉取
    cursor.execute("SELECT id, brand, city, created_at FROM memory_pool ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    
    # 组装成 JSON 数组返回
    return [
        {"id": r[0], "brand": r[1], "city": r[2], "created_at": r[3]}
        for r in rows
    ]

# ==========================================
# 接口 4：删除指定胶囊 (出库/拔草)
# ==========================================
@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: int):
    conn = sqlite3.connect("memories.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM memory_pool WHERE id = ?", (memory_id,))
    conn.commit()
    conn.close()
    return {"status": "success", "message": "删除成功"}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)