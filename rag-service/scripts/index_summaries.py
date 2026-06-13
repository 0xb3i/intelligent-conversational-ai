"""
摘要 v2 入库脚本
==========================================

目的
----
把 `RAG/data/processed/category_summaries_v2.json` 写入两个新的 chroma collection，
供检索模块使用，**不动现有的 `summary_database` 集合**（保留以便 A/B 对比）。

产出 collection
----------------
1) summary_overview_v2  ：14 条，document = "类目：xxx\\n总览：...\\n关键词：..."
                          metadata = { category, comment_count, sampled_count, n_points }
   ── 用于「类目级粗筛」

2) summary_points_v2    ：≈ 84 条，document = "[类目][子主题][极性] xxx\\n关键词：..."
                          metadata = { category, subtopic, polarity, salience,
                                       keywords (用 \"|\" 拼成 str), comment_count, point_idx }
   ── 用于「要点级精排」（支持 polarity / category 过滤）

使用方法
--------
    cd /Users/shengonghui/Desktop/intelligent-conversational-ai
    no_proxy='*' NO_PROXY='*' .venv/bin/python rag-service/scripts/index_summaries.py

参数
----
    --input            v2 JSON 路径
    --chroma-path      chroma 持久化目录
    --reset            若 collection 已存在则删除重建（默认 True）
    --batch-size       embedding 批量大小，默认 32

注意
----
- chroma metadata 不支持 list，因此 keywords 用 "|" 拼接为 str；下游使用时再 split。
- 不会读取/修改任何现有 collection，仅新建两个 v2 集合。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 让脚本既能从 rag-service/ 也能从仓库根目录运行
SCRIPT_DIR = Path(__file__).resolve().parent
RAG_SERVICE_DIR = SCRIPT_DIR.parent
REPO_ROOT = RAG_SERVICE_DIR.parent
sys.path.insert(0, str(RAG_SERVICE_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(RAG_SERVICE_DIR / ".env")

import chromadb  # noqa: E402
from modules.clients import EmbeddingClient  # noqa: E402


# ------------------- 默认路径 -------------------
DEFAULT_INPUT = REPO_ROOT / "RAG" / "data" / "processed" / "category_summaries_v2.json"
DEFAULT_CHROMA = RAG_SERVICE_DIR / "data" / "chroma_db"

OVERVIEW_COLLECTION = "summary_overview_v2"
POINTS_COLLECTION = "summary_points_v2"


# ------------------- 工具函数 -------------------
def _build_overview_doc(cat: str, item: dict) -> str:
    """类目级 document：用于粗筛。把类目名/总览/关键词拼起来做 embedding。"""
    keywords = item.get("keywords") or []
    kw_str = "、".join(keywords)
    overview = (item.get("overview") or "").strip()
    return f"类目：{cat}\n总览：{overview}\n关键词：{kw_str}"


def _build_point_doc(cat: str, p: dict) -> str:
    """要点级 document：用于精排。把类目/子主题/极性/摘要/关键词拼起来做 embedding。"""
    polarity_zh = {"positive": "正面", "negative": "负面", "neutral": "中性"}.get(
        p.get("polarity", ""), p.get("polarity", "")
    )
    subtopic = (p.get("subtopic") or "").strip()
    summary = (p.get("summary") or "").strip()
    keywords = p.get("keywords") or []
    kw_str = "、".join(keywords)
    return (
        f"[{cat}][{subtopic}][{polarity_zh}]\n"
        f"{summary}\n"
        f"关键词：{kw_str}"
    )


def _embed_in_batches(
    client: EmbeddingClient, texts: list[str], batch_size: int
) -> list[list[float]]:
    """批量调用 embedding（DashScope text-embedding-v4 单次上限 10 条）。"""
    DASHSCOPE_BATCH_LIMIT = 10
    safe_batch = min(batch_size, DASHSCOPE_BATCH_LIMIT)
    all_vec: list[list[float]] = []
    for i in range(0, len(texts), safe_batch):
        batch = texts[i : i + safe_batch]
        all_vec.extend(client.embed_batch(batch))
    return all_vec


# ------------------- 主流程 -------------------
def main():
    parser = argparse.ArgumentParser(description="把 v2 摘要灌入 chroma 两个新集合")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--chroma-path", type=str, default=str(DEFAULT_CHROMA))
    parser.add_argument(
        "--reset",
        type=lambda x: str(x).lower() not in ("0", "false", "no"),
        default=True,
        help="若 collection 已存在则删除重建（默认 True）",
    )
    parser.add_argument("--batch-size", type=int, default=10,
                        help="embedding 单批大小，DashScope text-embedding-v4 上限 10")
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        print("[ERROR] 未读到合法的 DASHSCOPE_API_KEY，请检查 rag-service/.env")
        sys.exit(1)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] 输入文件不存在: {in_path}")
        sys.exit(1)

    print(f"[INFO] 读取摘要文件: {in_path}")
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or len(data) == 0:
        print("[ERROR] 输入 JSON 顶层必须是非空的 dict（类目名 -> 摘要对象）")
        sys.exit(1)

    print(f"[INFO] 类目数: {len(data)}")

    # ---- 准备 overview 与 points 两份待入库数据 ----
    ov_ids: list[str] = []
    ov_docs: list[str] = []
    ov_metas: list[dict] = []

    pt_ids: list[str] = []
    pt_docs: list[str] = []
    pt_metas: list[dict] = []

    for cat, item in data.items():
        # ---- overview 条目 ----
        ov_ids.append(f"ov::{cat}")
        ov_docs.append(_build_overview_doc(cat, item))
        ov_metas.append(
            {
                "category": cat,
                "comment_count": int(item.get("comment_count", 0)),
                "sampled_count": int(item.get("sampled_count", 0)),
                "n_points": len(item.get("points") or []),
                # 把全类关键词也存到 metadata，方便上游展示
                "keywords": "|".join(item.get("keywords") or []),
                "overview": (item.get("overview") or "").strip(),
            }
        )

        # ---- points 条目 ----
        for idx, p in enumerate(item.get("points") or []):
            polarity = p.get("polarity", "neutral")
            if polarity not in ("positive", "negative", "neutral"):
                polarity = "neutral"
            pt_ids.append(f"pt::{cat}::{idx}")
            pt_docs.append(_build_point_doc(cat, p))
            pt_metas.append(
                {
                    "category": cat,
                    "subtopic": (p.get("subtopic") or "").strip(),
                    "polarity": polarity,
                    "salience": float(p.get("salience", 0.0)),
                    "keywords": "|".join(p.get("keywords") or []),
                    "summary": (p.get("summary") or "").strip(),
                    "comment_count": int(item.get("comment_count", 0)),
                    "point_idx": idx,
                }
            )

    print(f"[INFO] overview 待写入: {len(ov_docs)} 条")
    print(f"[INFO] points   待写入: {len(pt_docs)} 条")

    # ---- 调 embedding ----
    emb_client = EmbeddingClient(api_key=api_key)

    print(f"[INFO] 计算 overview embedding ({len(ov_docs)} 条) …")
    t0 = time.time()
    ov_vecs = _embed_in_batches(emb_client, ov_docs, batch_size=args.batch_size)
    print(f"        done in {time.time() - t0:.1f}s")

    print(f"[INFO] 计算 points   embedding ({len(pt_docs)} 条) …")
    t0 = time.time()
    pt_vecs = _embed_in_batches(emb_client, pt_docs, batch_size=args.batch_size)
    print(f"        done in {time.time() - t0:.1f}s")

    # ---- 写 chroma ----
    chroma_path = Path(args.chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))

    existing = {c.name for c in client.list_collections()}
    for col_name in (OVERVIEW_COLLECTION, POINTS_COLLECTION):
        if col_name in existing:
            if args.reset:
                print(f"[INFO] 删除已存在 collection: {col_name}")
                client.delete_collection(col_name)
            else:
                print(f"[ERROR] collection 已存在: {col_name}（如需覆盖请传 --reset True）")
                sys.exit(1)

    ov_col = client.create_collection(
        name=OVERVIEW_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    pt_col = client.create_collection(
        name=POINTS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"[INFO] 写入 {OVERVIEW_COLLECTION} …")
    ov_col.add(ids=ov_ids, documents=ov_docs, metadatas=ov_metas, embeddings=ov_vecs)

    print(f"[INFO] 写入 {POINTS_COLLECTION} …")
    pt_col.add(ids=pt_ids, documents=pt_docs, metadatas=pt_metas, embeddings=pt_vecs)

    # ---- 校验 ----
    print()
    print("=" * 60)
    print("✅ 入库完成")
    print(f"   {OVERVIEW_COLLECTION}: {ov_col.count()} 条")
    print(f"   {POINTS_COLLECTION}:   {pt_col.count()} 条")
    print(f"   存储路径: {chroma_path}")
    print("=" * 60)
    print()
    print("现有 collection 列表：")
    for c in client.list_collections():
        print(f"   - {c.name}")


if __name__ == "__main__":
    main()
