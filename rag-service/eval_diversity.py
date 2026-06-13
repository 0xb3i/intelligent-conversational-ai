"""
多样性重排消融评估脚本

运行方式:
    cd rag-service
    /Users/xuyueming/miniforge3/bin/python eval_diversity.py

评估内容:
    - 7 种配置 × 前 30 题 = 210 次检索
    - 指标: 平均成对相似度、类别覆盖率、房型多样性、相关性保持率
"""

import json
import time
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path

API = "http://localhost:8000/api/v1/chat"
EVAL_SET = Path(__file__).parent.parent / "RAG" / "data" / "evaluation" / "eval_set.json"

# ── 评估配置 ─────────────────────────────────────────────────────────

CONFIGS = [
    # (name, diversity_method, param_key, param_val)
    ("baseline",      None,             None, None),
    ("mmr_lambda_0.7", "mmr",           "diversity_lambda", 0.7),
    ("mmr_lambda_0.5", "mmr",           "diversity_lambda", 0.5),
    ("mmr_lambda_0.3", "mmr",           "diversity_lambda", 0.3),
    ("dpp_theta_0.5",  "dpp",           "diversity_theta",  0.5),
    ("dpp_theta_1.0",  "dpp",           "diversity_theta",  1.0),
    ("dpp_theta_2.0",  "dpp",           "diversity_theta",  2.0),
]

N_QUESTIONS = 30  # 先用 30 题快速出结果，后续可扩展到 90


def call_api(query: str, diversity_method=None, **params) -> dict:
    """调用 RAG API（只检索，不生成）"""
    options = {"enable_generation": False}
    if diversity_method:
        options["diversity_method"] = diversity_method
        if "diversity_lambda" in params:
            options["diversity_lambda"] = params["diversity_lambda"]
        if "diversity_theta" in params:
            options["diversity_theta"] = params["diversity_theta"]

    body = json.dumps({"query": query, "options": options}).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err = json.loads(e.read())
        print(f"  !! API error: {err.get('detail', str(e))}")
        return {"error": str(err)}


def get_categories(comment: dict) -> set:
    """提取评论的类别标签"""
    cats = set()
    for key in ("category1", "category2", "category3"):
        v = comment.get(key)
        if v and isinstance(v, str):
            cats.add(v)
    return cats


def get_room_type(comment: dict) -> str:
    return comment.get("fuzzy_room_type", comment.get("room_type", ""))


# ── 指标计算 ─────────────────────────────────────────────────────────

def calc_diversity_metrics(comments: list, timing: dict) -> dict:
    """计算一组评论的多样性定量指标"""
    n = len(comments)
    if n < 2:
        return {"n_comments": n, "error": "too_few_comments"}

    # 1. 类别覆盖率: 去重类别数 / 总数
    all_cats = set()
    for c in comments:
        all_cats |= get_categories(c)
    # 14 个标准小类
    category_coverage = len(all_cats) / 14.0

    # 2. 房型多样性: 去重房型数 / n
    room_types = {get_room_type(c) for c in comments}
    room_diversity = len(room_types) / n

    # 3. 话题跨度: 涉及的一级类别数
    # 5 个大类: 设施, 服务, 位置, 价格, 体验
    primary_cats = set()
    for c in comments:
        for key in ("category1", "category2", "category3"):
            v = c.get(key)
            if not v or not isinstance(v, str):
                continue
            # 映射到 5 大类
            if v in ("房间设施", "公共设施", "餐饮设施"):
                primary_cats.add("设施")
            elif v in ("前台服务", "客房服务", "退房/入住效率"):
                primary_cats.add("服务")
            elif v in ("交通便利性", "周边配套", "景观/朝向"):
                primary_cats.add("位置")
            elif v in ("性价比", "价格合理性"):
                primary_cats.add("价格")
            elif v in ("整体满意度", "安静程度", "卫生状况"):
                primary_cats.add("体验")
    topic_breadth = len(primary_cats) / 5.0

    # 4. 相关性保持率: 多样性后的 avg(final_score) / baseline avg(final_score)
    avg_score = np.mean([c.get("relevance_score", 0) for c in comments])

    # 5. 多样性统计 — 优先从 ranking.diversity_stats 获取
    diversity_stats = timing.get("ranking", {}).get("diversity_stats")
    mean_pairwise_sim = None
    if diversity_stats and isinstance(diversity_stats, dict):
        mean_pairwise_sim = diversity_stats.get("mean_pairwise_cosine_sim")

    return {
        "n": n,
        "category_coverage": round(category_coverage, 4),
        "room_diversity": round(room_diversity, 4),
        "topic_breadth": round(topic_breadth, 4),
        "avg_relevance": round(avg_score, 4),
        "mean_pairwise_sim": round(mean_pairwise_sim, 4) if mean_pairwise_sim is not None else None,
    }


# ── 主流程 ───────────────────────────────────────────────────────────

def main():
    with open(EVAL_SET) as f:
        questions = json.load(f)[:N_QUESTIONS]

    print(f"{'='*70}")
    print(f"多样性消融评估: {N_QUESTIONS} 题 × {len(CONFIGS)} 配置 = {N_QUESTIONS * len(CONFIGS)} 次检索")
    print(f"{'='*70}\n")

    all_results = {}

    for cfg_name, method, param_key, param_val in CONFIGS:
        print(f"▶ 配置: {cfg_name} ", end="", flush=True)
        cfg_metrics = []
        errors = 0
        t0 = time.time()

        for i, q in enumerate(questions):
            query = q["question"]
            params = {}
            if param_key:
                params[param_key] = param_val

            result = call_api(query, diversity_method=method, **params)

            if "error" in result:
                errors += 1
                continue

            comments = result.get("references", {}).get("comments", [])
            timing = result.get("timing", {})

            if len(comments) >= 2:
                m = calc_diversity_metrics(comments, timing)
                m["question_id"] = q["question_id"]
                cfg_metrics.append(m)

            if (i + 1) % 10 == 0:
                print(".", end="", flush=True)

        elapsed = time.time() - t0
        print(f" {elapsed:.0f}s")
        print(f"  完成 {len(cfg_metrics)}/{N_QUESTIONS - errors} 题 "
              f"(错误 {errors}), 平均 {elapsed/max(len(cfg_metrics),1):.1f}s/题")

        all_results[cfg_name] = cfg_metrics

    # ── 汇总 ──────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("评估结果汇总")
    print(f"{'='*70}")

    baseline_metrics = None
    for cfg_name, metrics in all_results.items():
        if not metrics:
            continue

        avg_cat_cov = np.mean([m["category_coverage"] for m in metrics])
        avg_room_div = np.mean([m["room_diversity"] for m in metrics])
        avg_topic = np.mean([m["topic_breadth"] for m in metrics])
        avg_rel = np.mean([m["avg_relevance"] for m in metrics])

        # 平均成对相似度 (仅 diversity 配置有)
        sim_values = [m["mean_pairwise_sim"] for m in metrics
                      if m["mean_pairwise_sim"] is not None]
        avg_sim = np.mean(sim_values) if sim_values else None

        if cfg_name == "baseline":
            baseline_rel = avg_rel
            baseline_metrics = {
                "cat_cov": avg_cat_cov, "room_div": avg_room_div,
                "topic": avg_topic, "rel": avg_rel
            }

        # 相对提升
        rel_change = ""
        if baseline_metrics and cfg_name != "baseline":
            cat_delta = (avg_cat_cov - baseline_metrics["cat_cov"]) / baseline_metrics["cat_cov"] * 100
            room_delta = (avg_room_div - baseline_metrics["room_div"]) / baseline_metrics["room_div"] * 100
            topic_delta = (avg_topic - baseline_metrics["topic"]) / baseline_metrics["topic"] * 100
            rel_delta = (avg_rel - baseline_metrics["rel"]) / baseline_metrics["rel"] * 100
            rel_change = f"vs baseline: cat {cat_delta:+.1f}% room {room_delta:+.1f}% topic {topic_delta:+.1f}% rel {rel_delta:+.1f}%"

        sim_str = f"  avg_sim={avg_sim:.4f}" if avg_sim is not None else ""
        print(f"\n  {cfg_name}:")
        print(f"    cat_cov={avg_cat_cov:.4f}  room_div={avg_room_div:.4f}  "
              f"topic={avg_topic:.4f}  rel={avg_rel:.4f}{sim_str}")
        if rel_change:
            print(f"    {rel_change}")

    # ── 排名 ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("综合排名 (类别覆盖率 + 话题广度 + 房型多样性 - 相关性损失)")

    scores = {}
    for cfg_name, metrics in all_results.items():
        if not metrics:
            continue
        avg_cat = np.mean([m["category_coverage"] for m in metrics])
        avg_topic = np.mean([m["topic_breadth"] for m in metrics])
        avg_room = np.mean([m["room_diversity"] for m in metrics])
        avg_rel = np.mean([m["avg_relevance"] for m in metrics])
        # 综合: 多样性维度平均 - 相关性惩罚
        diversity_score = (avg_cat + avg_topic + avg_room) / 3
        scores[cfg_name] = {
            "diversity": round(diversity_score, 4),
            "relevance": round(avg_rel, 4),
            "composite": round(diversity_score * 0.6 + avg_rel * 0.4, 4),
        }

    ranked = sorted(scores.items(), key=lambda x: x[1]["composite"], reverse=True)
    for rank, (name, s) in enumerate(ranked, 1):
        print(f"  {rank}. {name:20s}  diversity={s['diversity']:.4f}  "
              f"relevance={s['relevance']:.4f}  composite={s['composite']:.4f}")

    print(f"\n{'='*70}")
    print("评估完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
