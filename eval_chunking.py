"""
对比 BM25 检索：整条评论 baseline vs chunked。

为节省时间，直接评估"召回多样性 + chunk 命中率"等可离线计算的代理指标，
不调用 LLM。结果输出到 RAG/data/evaluation/chunking_compare.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAG_ROOT = ROOT / "rag-service" / "modules"
sys.path.insert(0, str(RAG_ROOT))

from index import InvertedIndex  # noqa: E402
from chunker import split_review  # noqa: E402

STOPWORDS = ROOT / "RAG" / "config" / "stopwords_chinese.txt"
COMMENTS_CSV = ROOT / "RAG" / "data" / "processed" / "filtered_comments.csv"
EVAL_SET = ROOT / "RAG" / "data" / "evaluation" / "eval_set.json"
OUT = ROOT / "RAG" / "data" / "evaluation" / "chunking_compare.json"

TOPK = 10


def build_baseline(df: pd.DataFrame) -> InvertedIndex:
    idx = InvertedIndex(stopwords_file=str(STOPWORDS))
    docs = {str(row["_id"]): str(row["comment"]) for _, row in df.iterrows()}
    idx.build(docs)
    return idx


def build_chunked(df: pd.DataFrame):
    chunks = []
    for _, row in df.iterrows():
        chunks.extend(split_review(row["comment"], comment_id=str(row["_id"])))
    idx = InvertedIndex(stopwords_file=str(STOPWORDS))
    docs = {c.chunk_id: c.text for c in chunks}
    idx.build(docs)
    chunk_meta = {c.chunk_id: c for c in chunks}
    return idx, chunk_meta


def chunked_to_comment(results, chunk_meta, topk):
    """把 chunk 级 Top-K 折叠到 comment 级 Top-K（保留每条评论 rrf 最高的 chunk）。"""
    seen = {}
    out = []
    for chunk_id, score in results:
        cid = chunk_meta[chunk_id].comment_id
        if cid in seen:
            continue
        seen[cid] = True
        out.append((cid, chunk_id, score))
        if len(out) >= topk:
            break
    return out


def main():
    df = pd.read_csv(COMMENTS_CSV)
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))

    print(f"[load] {len(df)} comments, {len(eval_set)} eval queries")

    t0 = time.time()
    base_idx = build_baseline(df)
    t_base = time.time() - t0
    print(f"[baseline] build {t_base:.1f}s, terms={len(base_idx.index)}, docs={base_idx.num_docs}")

    t0 = time.time()
    chunk_idx, chunk_meta = build_chunked(df)
    t_chunk = time.time() - t0
    print(f"[chunked]  build {t_chunk:.1f}s, terms={len(chunk_idx.index)}, docs={chunk_idx.num_docs}")

    avg_doc_len_base = base_idx.avg_doc_length
    avg_doc_len_chunk = chunk_idx.avg_doc_length

    per_query = []
    overlap_sum = 0
    only_base = 0
    only_chunk = 0
    chunk_avg_per_comment = []
    chunk_diversity = []

    for q in eval_set:
        question = q["question"]
        base_top = base_idx.search(question, topk=TOPK)
        base_ids = {doc_id for doc_id, _ in base_top}

        chunk_top_raw = chunk_idx.search(question, topk=TOPK * 5)
        chunk_top = chunked_to_comment(chunk_top_raw, chunk_meta, TOPK)
        chunk_ids = {cid for cid, _, _ in chunk_top}

        inter = base_ids & chunk_ids
        only_b = base_ids - chunk_ids
        only_c = chunk_ids - base_ids
        overlap_sum += len(inter)
        only_base += len(only_b)
        only_chunk += len(only_c)

        # chunk 命中多样性：每条命中评论平均被命中的 chunk 数
        comment_hits = {}
        for chunk_id, _ in chunk_top_raw[:TOPK * 3]:
            cid = chunk_meta[chunk_id].comment_id
            comment_hits[cid] = comment_hits.get(cid, 0) + 1
        if comment_hits:
            chunk_avg_per_comment.append(sum(comment_hits.values()) / len(comment_hits))
            chunk_diversity.append(len(comment_hits))

        per_query.append({
            "question_id": q["question_id"],
            "question": question,
            "baseline_top10": [c for c, _ in base_top],
            "chunked_top10": [
                {"comment_id": cid, "chunk_id": ck, "score": round(s, 3),
                 "snippet": chunk_meta[ck].text[:60]}
                for cid, ck, s in chunk_top
            ],
            "overlap@10": len(inter),
        })

    n = len(eval_set)
    summary = {
        "n_queries": n,
        "topk": TOPK,
        "baseline": {
            "build_time_s": round(t_base, 2),
            "num_docs": base_idx.num_docs,
            "num_terms": len(base_idx.index),
            "avg_doc_length": round(avg_doc_len_base, 2),
        },
        "chunked": {
            "build_time_s": round(t_chunk, 2),
            "num_docs": chunk_idx.num_docs,
            "num_terms": len(chunk_idx.index),
            "avg_doc_length": round(avg_doc_len_chunk, 2),
            "avg_chunks_per_comment": round(chunk_idx.num_docs / base_idx.num_docs, 2),
        },
        "compare": {
            "avg_overlap@10": round(overlap_sum / n, 2),
            "avg_only_baseline@10": round(only_base / n, 2),
            "avg_only_chunked@10": round(only_chunk / n, 2),
            "avg_chunks_hit_per_comment(top30 chunks)": round(
                sum(chunk_avg_per_comment) / len(chunk_avg_per_comment), 2
            ) if chunk_avg_per_comment else 0,
            "avg_distinct_comments(top30 chunks)": round(
                sum(chunk_diversity) / len(chunk_diversity), 2
            ) if chunk_diversity else 0,
        },
    }

    OUT.write_text(json.dumps({"summary": summary, "per_query": per_query},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
