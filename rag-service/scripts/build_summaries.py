"""
分类摘要生成（v2，结构化多要点版）
==========================================

目的
----
为 `filtered_comments.csv` 中的每个 category 标签生成更高质量、可被检索/生成
模块利用的「结构化摘要」，相对旧版 `category_summaries.json` 的区别：

旧版（v1）：每类 1 条 ~600 字综述，正负面 / 多子主题混在一起，难以被检索定位
新版（v2）：每类输出
    - overview        ：2-4 句客观总览（短、易读，参考 Exp1 风格）
    - points (N 条)   ：每条 = { subtopic, polarity, summary, keywords, salience }

并通过「分块 Map - 全局 Reduce」两阶段流水线，避免一次性把上千条评论塞给 LLM
导致超长上下文与质量不稳。

使用方法
--------
1) 确保已激活虚拟环境，并设置好环境变量 `DASHSCOPE_API_KEY`
   （也可写在 rag-service/.env 中，本脚本会自动加载）

    cd rag-service
    python scripts/build_summaries.py

2) 常用参数（任选）
    --input              输入 csv 路径
    --output             输出 v2 json 路径
    --output-flat        输出 v1 兼容版 json 路径（仅保留 overview）
    --model              LLM 模型名，默认 qwen-plus
    --workers            并发线程数，默认 6
    --max-comments       每个类目最多采样多少条进入 Map 阶段，默认 200
    --chunk-size         Map 阶段每块评论数，默认 30
    --max-points         每个类目最终输出的要点数上限，默认 6
    --only-category      仅处理某个类目，逗号分隔（用于试跑/调试）
    --dry-run            不调 LLM，只打印将要处理的类目和块统计
    --seed               采样随机种子，默认 42

输出示例（v2）
--------------
{
  "餐饮设施": {
    "category": "餐饮设施",
    "comment_count": 676,
    "sampled_count": 200,
    "keywords": ["早餐", "自助", "排队", "粤菜", "酒廊"],
    "overview": "餐饮整体口碑突出 …（2-4 句）",
    "points": [
      {
        "subtopic": "自助早餐",
        "polarity": "positive",
        "summary": "瀑布景观/旋转餐厅的自助早餐环境与品质受到广泛好评。",
        "keywords": ["早餐", "瀑布", "种类"],
        "salience": 0.85
      },
      ...
    ]
  },
  ...
}
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# 路径与默认配置
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "RAG" / "data" / "processed" / "filtered_comments.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "RAG" / "data" / "processed" / "category_summaries_v2.json"
DEFAULT_OUTPUT_FLAT = PROJECT_ROOT / "RAG" / "data" / "processed" / "category_summaries_v2_flat.json"

# 14 个允许的 category（与 Exp3 / Exp1 保持一致）
ALLOWED_CATEGORIES = [
    "房间设施", "公共设施", "餐饮设施",
    "前台服务", "客房服务", "退房/入住效率",
    "交通便利性", "周边配套", "景观/朝向",
    "性价比", "价格合理性",
    "整体满意度", "安静程度", "卫生状况",
]
ALLOWED_SET = set(ALLOWED_CATEGORIES)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
MAP_SYSTEM_PROMPT = """
你是一名酒店评论洞察分析师，擅长从大量真实评论中抽取「结构化要点」。
你会收到某个标签下的一批评论，请输出该批评论中**可被独立陈述**的要点列表。
要求：
- 每个要点是一个具体的子主题（如"自助早餐种类"、"卫生间布局"），不要写成长综述
- 每个要点必须标注极性 polarity ∈ {"positive","negative","neutral"}
- summary 必须忠实于评论，不得编造；语言简洁（1-2 句、不超过 60 字）
- keywords 给 2-5 个中文关键词
- mentions 估计该要点在本批评论中被提及的大致条数（整数）
- 同一子主题正负面要拆成两条（如有）
- 输出 JSON，且仅返回 JSON
""".strip()

MAP_USER_TEMPLATE = """
标签：{label}
本批评论数：{n_comments}
评论样本（每行 1 条，已编号）：
{comments_block}

请输出 JSON：
{{
  "points": [
    {{
      "subtopic": "...",
      "polarity": "positive | negative | neutral",
      "summary": "1-2 句客观陈述",
      "keywords": ["...", "..."],
      "mentions": 0
    }}
  ]
}}
""".strip()


REDUCE_SYSTEM_PROMPT = """
你是一名酒店评论洞察分析师。
你会收到某个标签下、由多批评论分别抽取出的「候选要点列表」（含重复、噪声）。
请将其去重、聚合、合并，输出该标签下的最终要点集合，并写一段精炼总览。
要求：
- points 数量不超过 {max_points} 条；按 salience（重要性 0-1）从高到低排序
- 同一 subtopic 下的多条要点要合并；正负面要分别保留为不同要点
- summary 客观、可读，1-2 句、不超过 80 字，不得编造
- salience 综合考虑 mentions 数量与该要点对此标签的代表性（0-1 浮点）
- overview：2-4 句精炼概述该标签下的整体情况，需同时反映优点与不足（如有）
- keywords：5-8 个能代表整个类目的高频中文关键词
- 仅返回 JSON
""".strip()

REDUCE_USER_TEMPLATE = """
标签：{label}
本类目评论总数：{total}
本次采样进入分析的评论数：{sampled}
候选要点列表（来自分块抽取）：
{candidates_block}

请输出 JSON：
{{
  "overview": "2-4 句的精炼总览",
  "keywords": ["...", "..."],
  "points": [
    {{
      "subtopic": "...",
      "polarity": "positive | negative | neutral",
      "summary": "1-2 句客观陈述",
      "keywords": ["...", "..."],
      "salience": 0.0
    }}
  ]
}}
""".strip()


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def parse_categories(raw: Any) -> list[str]:
    """容错解析 categories 列。"""
    if isinstance(raw, list):
        labels = raw
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            labels = parsed if isinstance(parsed, list) else [s]
        except Exception:
            labels = [s]
    else:
        return []
    out: list[str] = []
    for lb in labels:
        t = str(lb).strip()
        if t in ALLOWED_SET and t not in out:
            out.append(t)
    return out


def truncate(text: str, max_chars: int = 220) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def chunked(seq: list, size: int) -> list[list]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def stratified_sample(df_cat: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    简单分层采样：
    - 优先取 quality_score 高的评论（如果列存在）
    - 再按 publish_date 年度均匀采样，避免单一时段
    """
    if len(df_cat) <= n:
        return df_cat

    df = df_cat.copy()

    # 按年份分桶（解析失败的归到 "unknown"）
    if "publish_date" in df.columns:
        df["_year"] = pd.to_datetime(df["publish_date"], errors="coerce").dt.year
        df["_year"] = df["_year"].fillna(-1).astype(int)
    else:
        df["_year"] = -1

    # 每年份按 quality_score 降序排序（无该列则随机）
    if "quality_score" in df.columns:
        df = df.sort_values("quality_score", ascending=False, kind="stable")
    else:
        df = df.sample(frac=1.0, random_state=seed)

    years = sorted(df["_year"].unique())
    per_year = max(1, n // max(1, len(years)))
    pieces = []
    for y in years:
        sub = df[df["_year"] == y].head(per_year)
        pieces.append(sub)
    sampled = pd.concat(pieces)

    # 不足 n 时再补
    if len(sampled) < n:
        remain = df.drop(sampled.index, errors="ignore")
        sampled = pd.concat([sampled, remain.head(n - len(sampled))])

    sampled = sampled.head(n).drop(columns=["_year"], errors="ignore")
    return sampled.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# LLM 调用
# --------------------------------------------------------------------------- #
def make_client() -> OpenAI:
    load_dotenv(PROJECT_ROOT / "rag-service" / ".env")
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "环境变量 DASHSCOPE_API_KEY 未设置；请先在 rag-service/.env 或 shell 中配置"
        )
    return OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
    )


def call_llm_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.2,
    retries: int = 3,
) -> dict | None:
    """调 LLM 并强制 JSON 输出，失败重试。"""
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content.strip()
            return json.loads(content)
        except Exception as e:
            if i < retries - 1:
                time.sleep(1.5 * (i + 1))
                continue
            print(f"[LLM] 调用失败（已重试 {retries} 次）: {e}", file=sys.stderr)
            return None


# --------------------------------------------------------------------------- #
# Map / Reduce
# --------------------------------------------------------------------------- #
def map_chunk(
    client: OpenAI,
    model: str,
    label: str,
    chunk: list[str],
) -> list[dict]:
    """对单个 chunk 抽取候选要点。"""
    block_lines = [f"{i + 1}. {truncate(c, 220)}" for i, c in enumerate(chunk)]
    user_prompt = MAP_USER_TEMPLATE.format(
        label=label,
        n_comments=len(chunk),
        comments_block="\n".join(block_lines),
    )
    data = call_llm_json(client, model, MAP_SYSTEM_PROMPT, user_prompt)
    if not data:
        return []
    raw_points = data.get("points") or []
    cleaned: list[dict] = []
    for p in raw_points:
        if not isinstance(p, dict):
            continue
        polarity = str(p.get("polarity", "neutral")).strip().lower()
        if polarity not in {"positive", "negative", "neutral"}:
            polarity = "neutral"
        summary = str(p.get("summary", "")).strip()
        if not summary:
            continue
        cleaned.append(
            {
                "subtopic": str(p.get("subtopic", "")).strip()[:30] or "未分类",
                "polarity": polarity,
                "summary": summary[:200],
                "keywords": [str(k).strip() for k in (p.get("keywords") or []) if str(k).strip()][:5],
                "mentions": int(p.get("mentions") or 1),
            }
        )
    return cleaned


def reduce_points(
    client: OpenAI,
    model: str,
    label: str,
    total: int,
    sampled: int,
    candidates: list[dict],
    max_points: int,
) -> dict:
    """把所有 chunk 的候选要点合并成最终摘要。"""
    if not candidates:
        return {"overview": "暂无足够评论生成摘要。", "keywords": [], "points": []}

    cand_lines = []
    for i, p in enumerate(candidates, 1):
        cand_lines.append(
            f"{i}. [{p['polarity']}] {p['subtopic']}（提及≈{p['mentions']}）：{p['summary']}"
            f"  关键词：{', '.join(p.get('keywords', []))}"
        )
    user_prompt = REDUCE_USER_TEMPLATE.format(
        label=label,
        total=total,
        sampled=sampled,
        candidates_block="\n".join(cand_lines),
    )
    data = call_llm_json(
        client,
        model,
        REDUCE_SYSTEM_PROMPT.format(max_points=max_points),
        user_prompt,
        temperature=0.3,
    )
    if not data:
        return {"overview": "（生成失败）", "keywords": [], "points": []}

    # 清洗输出
    overview = str(data.get("overview", "")).strip()
    keywords = [str(k).strip() for k in (data.get("keywords") or []) if str(k).strip()][:8]
    raw_points = data.get("points") or []
    points: list[dict] = []
    for p in raw_points[:max_points]:
        if not isinstance(p, dict):
            continue
        polarity = str(p.get("polarity", "neutral")).strip().lower()
        if polarity not in {"positive", "negative", "neutral"}:
            polarity = "neutral"
        try:
            salience = float(p.get("salience", 0.5))
        except (TypeError, ValueError):
            salience = 0.5
        salience = max(0.0, min(1.0, salience))
        points.append(
            {
                "subtopic": str(p.get("subtopic", "")).strip()[:30] or "未分类",
                "polarity": polarity,
                "summary": str(p.get("summary", "")).strip()[:200],
                "keywords": [str(k).strip() for k in (p.get("keywords") or []) if str(k).strip()][:5],
                "salience": round(salience, 3),
            }
        )
    points.sort(key=lambda x: x["salience"], reverse=True)
    return {"overview": overview, "keywords": keywords, "points": points}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build_one_category(
    client: OpenAI,
    model: str,
    label: str,
    comments: list[str],
    *,
    max_comments: int,
    chunk_size: int,
    max_points: int,
    workers: int,
) -> dict:
    """对单个 category 跑完 Map + Reduce。"""
    total = len(comments)
    sampled_comments = comments[:max_comments]
    chunks = chunked(sampled_comments, chunk_size)

    # Map 阶段：并行
    candidates: list[dict] = []
    if chunks:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(map_chunk, client, model, label, ch) for ch in chunks]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"  Map [{label}]",
                leave=False,
            ):
                candidates.extend(fut.result())

    # Reduce 阶段
    reduced = reduce_points(
        client, model, label, total, len(sampled_comments), candidates, max_points
    )

    return {
        "category": label,
        "comment_count": total,
        "sampled_count": len(sampled_comments),
        "n_chunks": len(chunks),
        "n_candidates": len(candidates),
        "overview": reduced["overview"],
        "keywords": reduced["keywords"],
        "points": reduced["points"],
    }


def build_all(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    # 1) 读数据
    df = pd.read_csv(args.input)
    if "comment" not in df.columns or "categories" not in df.columns:
        raise RuntimeError(f"输入 CSV 缺少 comment / categories 列：{args.input}")
    df["_cats"] = df["categories"].apply(parse_categories)

    # 2) 按 category 收集评论（带 quality_score / publish_date 用于采样）
    keep_cols = ["comment", "_cats"]
    if "quality_score" in df.columns:
        keep_cols.append("quality_score")
    if "publish_date" in df.columns:
        keep_cols.append("publish_date")
    df = df[keep_cols].copy()

    only = (
        {x.strip() for x in args.only_category.split(",") if x.strip()}
        if args.only_category
        else None
    )

    # 每个 category 准备一份 DataFrame
    cat_to_df: dict[str, pd.DataFrame] = {}
    for label in ALLOWED_CATEGORIES:
        if only and label not in only:
            continue
        mask = df["_cats"].apply(lambda cs: label in cs)
        sub = df[mask].copy()
        if len(sub) == 0:
            continue
        cat_to_df[label] = sub

    # 3) 采样
    cat_to_comments: dict[str, list[str]] = {}
    for label, sub in cat_to_df.items():
        sampled = stratified_sample(sub, args.max_comments, seed=args.seed)
        comments = [
            str(c).strip()
            for c in sampled["comment"].tolist()
            if isinstance(c, str) and str(c).strip()
        ]
        cat_to_comments[label] = comments

    # 打印计划
    print("\n=== 摘要生成计划 ===")
    print(f"输入：{args.input}")
    print(f"模型：{args.model}   并发：{args.workers}")
    print(
        f"每类最多采样 {args.max_comments} 条，chunk_size={args.chunk_size}，"
        f"每类最多输出 {args.max_points} 个要点"
    )
    print(f"待处理类目数：{len(cat_to_comments)}")
    for label, cs in cat_to_comments.items():
        n_chunks = (len(cs) + args.chunk_size - 1) // args.chunk_size
        print(
            f"  - {label}: 全量 {len(cat_to_df[label])} 条 | "
            f"采样 {len(cs)} 条 | 分 {n_chunks} 块"
        )

    if args.dry_run:
        print("\n[dry-run] 不调 LLM，已退出。")
        return

    # 4) 逐类处理
    client = make_client()
    results: dict[str, dict] = {}
    for label, comments in tqdm(cat_to_comments.items(), desc="类目进度"):
        try:
            res = build_one_category(
                client,
                args.model,
                label,
                comments,
                max_comments=args.max_comments,
                chunk_size=args.chunk_size,
                max_points=args.max_points,
                workers=args.workers,
            )
            results[label] = res
        except Exception as e:
            print(f"[{label}] 处理失败: {e}", file=sys.stderr)
            results[label] = {
                "category": label,
                "comment_count": len(cat_to_df[label]),
                "sampled_count": len(comments),
                "overview": "（生成失败）",
                "keywords": [],
                "points": [],
                "error": str(e),
            }

    # 5) 保存
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] v2 摘要写入 -> {args.output}")

    # 6) 同时输出 v1 兼容版（仅 overview，便于和旧 category_summaries.json 对比）
    flat = []
    for label, item in results.items():
        flat.append(
            {
                "category": label,
                "summary": item.get("overview", ""),
                "keywords": item.get("keywords", []),
                "comment_count": item.get("comment_count", 0),
            }
        )
    with open(args.output_flat, "w", encoding="utf-8") as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)
    print(f"[OK] v1 兼容版摘要写入 -> {args.output_flat}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="基于 Map-Reduce 的分类摘要生成 (v2)")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--output-flat", type=Path, default=DEFAULT_OUTPUT_FLAT)
    p.add_argument("--model", type=str, default="qwen-plus")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-comments", type=int, default=200, help="每个类目最多采样多少条评论进入 Map 阶段")
    p.add_argument("--chunk-size", type=int, default=30, help="Map 阶段每块评论数")
    p.add_argument("--max-points", type=int, default=6, help="每个类目最终输出的要点数上限")
    p.add_argument("--only-category", type=str, default="", help="只处理某些类目，逗号分隔")
    p.add_argument("--dry-run", action="store_true", help="不调 LLM，只打印计划")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_all(args)


if __name__ == "__main__":
    main()
