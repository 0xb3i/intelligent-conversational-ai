"""BM25 倒排索引优化评测：baseline vs +dict vs +dict+expansion。

设计：
  evaluate 三套配置在同一份 chunked 数据上的代理指标
   1) baseline:   旧分词（无 hotel_dict / 无同义词 / 不保留数字）
   2) +dict     : 加载 hotel_dict.txt + keep_digits=True，无同义词扩展
   3) +dict+syn : 上面 + 同义词 OR 扩展（仅查询侧）

代理指标（无金标 relevant_id 时使用）：
  A. soft_recall@K：top-K 评论里包含 intent_direction 关联关键词的比例
                    关键词来自 RAG/config/categories.json 的 subcategories
                    + synonyms.yaml 中同组词
  B. unique_term_hit@K：query 词项在 top-K 命中至少 1 次的覆盖率
  C. distinct_comments@K：top-K * 5 chunk 折叠后的 distinct comment 数
  D. avg_search_ms / p95_search_ms：search 平均 / p95 延迟
  E. avg_query_tokens (after expansion)

输出：
  RAG/data/evaluation/bm25_compare.json + 控制台 markdown 表
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, quantiles

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # repo 根
sys.path.insert(0, str(ROOT / "rag-service" / "modules"))

from index import InvertedIndex  # noqa: E402
from chunker import split_review  # noqa: E402

# ── 配置路径 ────────────────────────────────────────────────
STOPWORDS = ROOT / "RAG" / "config" / "stopwords_chinese.txt"
HOTEL_DICT = ROOT / "rag-service" / "data" / "dict" / "hotel_dict.txt"
SYNONYMS = ROOT / "rag-service" / "data" / "dict" / "synonyms.yaml"
COMMENTS_CSV = ROOT / "RAG" / "data" / "processed" / "filtered_comments.csv"
EVAL_SET = ROOT / "RAG" / "data" / "evaluation" / "eval_set.json"
CATEGORIES = ROOT / "RAG" / "config" / "categories.json"
OUT = ROOT / "RAG" / "data" / "evaluation" / "bm25_compare.json"

TOPK = 10              # 评估命中粒度
TOPK_CHUNK_RAW = 50    # chunk 检索 raw 取数（再折叠到 comment）

# ── 关键词软金标：intent_direction → keyword set ─────────────
def _build_intent_keywords() -> dict[str, set[str]]:
    """从 categories.json 抽出 {leaf_name: set(subcategories+leaf_name)}，
    并合并 synonyms.yaml 中同组的扩展词。"""
    cats = json.loads(CATEGORIES.read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    for top in cats.get("categories", []):
        for leaf in top.get("subcategories", []):
            # leaf 形如 {name: '房间设施', subcategories: ['床', '卫浴', ...]}
            name = leaf.get("name")
            if not name:
                continue
            out[name] = {name}
            for kw in leaf.get("subcategories", []) or []:
                out[name].add(kw)

    # 用 synonyms 把每组中的 token 互相连通，扩大关键词覆盖
    try:
        import yaml  # type: ignore
        syn = yaml.safe_load(SYNONYMS.read_text(encoding="utf-8")) or {}
        groups = syn.get("groups", []) or []
    except Exception:
        groups = []

    # 对每个 intent，在每个组里"任意词命中" → 把整组都加进来
    for intent, kws in out.items():
        added: set[str] = set()
        for g in groups:
            if not isinstance(g, list):
                continue
            g_low = {str(w).lower() for w in g}
            if g_low & {k.lower() for k in kws}:
                added.update(g_low)
        kws.update(added)
    return out


# ── 三种 InvertedIndex 配置 ────────────────────────────────
def make_index(variant: str) -> InvertedIndex:
    if variant == "baseline":
        return InvertedIndex(
            stopwords_file=str(STOPWORDS),
            hotel_dict_file=None,
            synonyms_file=None,
            keep_digits=False,
        )
    if variant == "+dict":
        return InvertedIndex(
            stopwords_file=str(STOPWORDS),
            hotel_dict_file=str(HOTEL_DICT),
            synonyms_file=None,
            keep_digits=True,
        )
    if variant == "+dict+syn":
        return InvertedIndex(
            stopwords_file=str(STOPWORDS),
            hotel_dict_file=str(HOTEL_DICT),
            synonyms_file=str(SYNONYMS),
            keep_digits=True,
        )
    raise ValueError(variant)


# ── 评估单个变体 ───────────────────────────────────────────
def eval_variant(
    variant: str,
    docs: dict[str, str],
    chunk_meta: dict[str, dict],
    eval_queries: list[dict],
    intent_kw: dict[str, set[str]],
    use_expansion: bool,
) -> dict:
    print(f"\n========= variant: {variant} =========")
    t0 = time.time()
    idx = make_index(variant)
    idx.build(docs)
    build_time = time.time() - t0

    soft_hits: list[float] = []
    term_hits: list[float] = []
    distinct_cnts: list[int] = []
    search_ms: list[float] = []
    expand_lens: list[int] = []
    per_query_dump: list[dict] = []

    for q in eval_queries:
        question = q["question"]
        intent = (q.get("metadata") or {}).get("intent_direction")

        # 计时 search（含扩展）
        ts = time.time()
        raw = idx.search(question, topk=TOPK_CHUNK_RAW, use_query_expansion=use_expansion)
        search_ms.append((time.time() - ts) * 1000)

        # 折叠到 comment 级，取 top-K 评论
        seen: dict[str, str] = {}
        comment_top: list[tuple[str, str, float]] = []
        for chunk_id, score in raw:
            meta = chunk_meta.get(chunk_id)
            if not meta:
                continue
            cid = meta["comment_id"]
            if cid in seen:
                continue
            seen[cid] = chunk_id
            comment_top.append((cid, chunk_id, score))
            if len(comment_top) >= TOPK:
                break

        # 软金标命中：top-K 评论文本里含 intent 关键词的比例
        if intent and intent in intent_kw:
            kws = intent_kw[intent]
            hit = 0
            for cid, ck, _ in comment_top:
                txt = chunk_meta[ck]["text"]  # 用命中 chunk 的文本片段足够
                if any(kw in txt for kw in kws):
                    hit += 1
            soft_hits.append(hit / max(1, len(comment_top)))

        # query 词项命中率：query token 中有多少个出现在 top-K 任一 chunk 文本里
        q_tokens = idx.tokenize(question)
        if use_expansion:
            q_tokens_expanded = idx.expand_query_tokens(q_tokens)
        else:
            q_tokens_expanded = q_tokens
        expand_lens.append(len(q_tokens_expanded))

        if q_tokens:
            joined = " ".join(chunk_meta[ck]["text"].lower() for _, ck, _ in comment_top)
            hit_terms = sum(1 for t in q_tokens if t in joined)
            term_hits.append(hit_terms / len(q_tokens))

        distinct_cnts.append(len({cid for cid, _, _ in comment_top}))

        per_query_dump.append({
            "question_id": q.get("question_id"),
            "intent": intent,
            "tokens_orig": q_tokens,
            "tokens_expanded": q_tokens_expanded if use_expansion else None,
            "top10_comments": [cid for cid, _, _ in comment_top],
        })

    # 汇总
    def _safe_mean(arr):
        return round(mean(arr), 4) if arr else 0.0

    def _p95(arr):
        if len(arr) < 5:
            return round(max(arr) if arr else 0, 2)
        return round(quantiles(arr, n=20)[-1], 2)

    summary = {
        "variant": variant,
        "use_expansion": use_expansion,
        "build_time_s": round(build_time, 2),
        "num_terms": len(idx.index),
        "num_docs": idx.num_docs,
        "avg_doc_length": round(idx.avg_doc_length, 2),
        "soft_recall@10": _safe_mean(soft_hits),
        "term_hit@10": _safe_mean(term_hits),
        "distinct_comments@10": _safe_mean(distinct_cnts),
        "avg_search_ms": _safe_mean(search_ms),
        "p95_search_ms": _p95(search_ms),
        "avg_query_tokens": _safe_mean(expand_lens),
        "n_queries_with_intent": len(soft_hits),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return {"summary": summary, "per_query": per_query_dump}


def main():
    df = pd.read_csv(COMMENTS_CSV)
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    intent_kw = _build_intent_keywords()
    print(f"[load] {len(df)} comments, {len(eval_set)} eval queries, "
          f"{len(intent_kw)} intent groups")

    # chunk 一次，三套变体共用
    chunks = []
    for _, row in df.iterrows():
        chunks.extend(split_review(row["comment"], comment_id=str(row["_id"])))
    docs = {c.chunk_id: c.text for c in chunks}
    chunk_meta = {
        c.chunk_id: {
            "comment_id": c.comment_id,
            "text": c.text,
            "seq": c.seq,
        } for c in chunks
    }
    print(f"[chunk] {len(chunks)} chunks (avg {len(chunks)/len(df):.2f}/comment)")

    out: dict[str, dict] = {}
    out["baseline"] = eval_variant(
        "baseline", docs, chunk_meta, eval_set, intent_kw, use_expansion=False
    )
    out["+dict"] = eval_variant(
        "+dict", docs, chunk_meta, eval_set, intent_kw, use_expansion=False
    )
    out["+dict+syn"] = eval_variant(
        "+dict+syn", docs, chunk_meta, eval_set, intent_kw, use_expansion=True
    )

    # 控制台对比表
    print("\n=========================================")
    print("==           最终对比                  ==")
    print("=========================================")
    headers = [
        "variant", "soft_recall@10", "term_hit@10", "distinct@10",
        "avg_ms", "p95_ms", "avg_q_tok", "num_terms",
    ]
    rows = []
    for k in ["baseline", "+dict", "+dict+syn"]:
        s = out[k]["summary"]
        rows.append([
            k, s["soft_recall@10"], s["term_hit@10"], s["distinct_comments@10"],
            s["avg_search_ms"], s["p95_search_ms"], s["avg_query_tokens"], s["num_terms"],
        ])
    col_w = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in col_w]))
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))

    # 落盘
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] → {OUT}")


if __name__ == "__main__":
    main()
