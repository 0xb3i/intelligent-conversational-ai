"""评论数据加载工具：优先 Insforge，失败时回退本地 CSV。"""

import os
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
LOCAL_COMMENTS_CSV = ROOT / "RAG" / "data" / "processed" / "filtered_comments.csv"
INSFORGE_TIMEOUT = 8


def _normalize_comments_df(df: pd.DataFrame) -> pd.DataFrame:
    """统一评论表结构，保证以 _id 为索引并补齐前端依赖字段。"""
    if "_id" in df.columns:
        df["_id"] = df["_id"].astype(str)
        df.set_index("_id", inplace=True)
    else:
        df.index = df.index.map(str)

    # 本地 CSV 没有拆分后的 category1/2/3 与 star 字段，在线格式化会读取它们。
    if "star" not in df.columns and "score" in df.columns:
        df["star"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)

    if "images" not in df.columns:
        df["images"] = "[]"

    if "categories" in df.columns:
        parsed = (
            df["categories"]
            .fillna("")
            .astype(str)
            .str.strip("[]")
            .str.replace("'", "", regex=False)
            .str.split(r"\s*,\s*", regex=True)
        )
        for idx in range(3):
            col = f"category{idx + 1}"
            if col not in df.columns:
                df[col] = parsed.apply(
                    lambda items: items[idx] if isinstance(items, list) and len(items) > idx and items[idx] else None
                )

    for col in ["travel_type", "room_type", "fuzzy_room_type"]:
        if col not in df.columns:
            df[col] = ""

    return df


def _load_comments_from_local_csv() -> pd.DataFrame:
    """从本地离线 CSV 加载评论。"""
    if not LOCAL_COMMENTS_CSV.exists():
        raise FileNotFoundError(f"本地评论 CSV 不存在: {LOCAL_COMMENTS_CSV}")

    print(f"正在从本地 CSV 加载评论数据: {LOCAL_COMMENTS_CSV.name}")
    df = pd.read_csv(LOCAL_COMMENTS_CSV)
    df = _normalize_comments_df(df)
    print(f"✅ 已从本地 CSV 加载 {len(df)} 条评论数据")
    return df


def get_all_comments_from_insforge() -> pd.DataFrame:
    """从 Insforge 数据库获取所有评论数据。"""
    base_url = os.getenv("NEXT_PUBLIC_INSFORGE_BASE_URL")
    anon_key = os.getenv("NEXT_PUBLIC_INSFORGE_ANON_KEY")

    if not base_url or not anon_key:
        raise ValueError("缺少 Insforge 配置环境变量: NEXT_PUBLIC_INSFORGE_BASE_URL / NEXT_PUBLIC_INSFORGE_ANON_KEY")

    headers = {
        "apikey": anon_key,
        "Authorization": f"Bearer {anon_key}",
        "Content-Type": "application/json"
    }

    all_data = []
    batch_size = 1000
    offset = 0

    print("正在从 Insforge 数据库获取评论数据...")

    while True:
        url = f"{base_url}/api/database/records/comments?select=*"
        range_headers = {
            **headers,
            "Range-Unit": "items",
            "Range": f"{offset}-{offset + batch_size - 1}",
            "Prefer": "count=exact"
        }
        response = requests.get(url, headers=range_headers, timeout=INSFORGE_TIMEOUT)

        if response.status_code not in (200, 206):
            raise RuntimeError(f"Insforge API 调用失败: {response.status_code} {response.text}")

        data = response.json()
        if not data:
            break

        all_data.extend(data)
        print(f"  已获取 {len(all_data)} 条评论...")

        if len(data) < batch_size:
            break
        offset += batch_size

    df = pd.DataFrame(all_data)
    df = _normalize_comments_df(df)
    print(f"✅ 成功加载 {len(df)} 条评论数据")
    return df


def load_comments() -> pd.DataFrame:
    """优先使用 Insforge；缺失配置或请求失败时回退本地 CSV。"""
    try:
        return get_all_comments_from_insforge()
    except Exception as exc:
        print(f"Insforge 加载失败，回退本地 CSV: {exc}")
        return _load_comments_from_local_csv()
