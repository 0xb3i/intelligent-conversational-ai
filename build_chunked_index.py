"""
离线生成 chunked BM25 索引：
读取 RAG/data/processed/filtered_comments.csv，按 chunker 切分后构建
InvertedIndex，落盘到 rag-service/data/inverted_index_chunked.pkl，
同时落盘 chunk_meta.pkl 用于在线侧把 chunk_id 折叠回 comment_id 与
char_span。
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
# 直接把 modules 作为顶层 import 路径，避免触发 rag-service/modules/__init__.py
sys.path.insert(0, str(ROOT / "rag-service" / "modules"))

from index import InvertedIndex  # noqa: E402
from chunker import split_review  # noqa: E402

STOPWORDS = ROOT / "RAG" / "config" / "stopwords_chinese.txt"
COMMENTS_CSV = ROOT / "RAG" / "data" / "processed" / "filtered_comments.csv"
OUT_INDEX = ROOT / "rag-service" / "data" / "inverted_index_chunked.pkl"
OUT_META = ROOT / "rag-service" / "data" / "chunk_meta.pkl"

# BM25 v2：领域词典 + 同义词（同义词只影响 search，不影响 build 索引词项）
HOTEL_DICT = ROOT / "rag-service" / "data" / "dict" / "hotel_dict.txt"
SYNONYMS = ROOT / "rag-service" / "data" / "dict" / "synonyms.yaml"
BM25_K1 = 1.5
BM25_B = 0.75
KEEP_DIGITS = True


def main():
    df = pd.read_csv(COMMENTS_CSV)
    print(f"[load] {len(df)} comments from {COMMENTS_CSV.name}")

    chunks = []
    for _, row in df.iterrows():
        chunks.extend(split_review(row["comment"], comment_id=str(row["_id"])))

    print(f"[chunk] {len(chunks)} chunks "
          f"(avg {len(chunks)/len(df):.2f} chunks/comment)")

    docs = {c.chunk_id: c.text for c in chunks}
    meta = {
        c.chunk_id: {
            "comment_id": c.comment_id,
            "seq": c.seq,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "text": c.text,
        }
        for c in chunks
    }

    t0 = time.time()
    idx = InvertedIndex(
        k1=BM25_K1,
        b=BM25_B,
        stopwords_file=str(STOPWORDS),
        hotel_dict_file=str(HOTEL_DICT) if HOTEL_DICT.exists() else None,
        synonyms_file=str(SYNONYMS) if SYNONYMS.exists() else None,
        keep_digits=KEEP_DIGITS,
    )
    idx.build(docs)
    print(f"[index] built in {time.time()-t0:.2f}s; "
          f"{len(idx.index)} terms, {idx.num_docs} docs, "
          f"avg_doc_len={idx.avg_doc_length:.2f}")

    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    idx.save(str(OUT_INDEX))
    with open(OUT_META, "wb") as f:
        pickle.dump(meta, f)
    print(f"[save] index → {OUT_INDEX}")
    print(f"[save] meta  → {OUT_META}")


if __name__ == "__main__":
    main()
