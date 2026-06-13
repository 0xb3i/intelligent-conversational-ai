"""
多样性消融评估 v2 — 带断点续传 + 中间结果留存

运行:
    python eval_diversity_v2.py          # 首次运行，或续传未完成的配置
    python eval_diversity_v2.py --clean  # 清除缓存重新跑

输出:
    eval_results/                          ← 结果目录
    ├── raw/         每条 API 响应原始 JSON
    ├── checkpoints/ 每个配置完成后存盘
    └── summary.json  最终汇总
"""

import json, time, os, sys, numpy as np
from pathlib import Path
from urllib import request, error

API = "http://localhost:8000/api/v1/chat"
EVAL_SET = Path(__file__).parent.parent / "RAG" / "data" / "evaluation" / "eval_set.json"
OUT_DIR = Path(__file__).parent / "eval_results"
RAW_DIR = OUT_DIR / "raw"
CKPT_DIR = OUT_DIR / "checkpoints"

CONFIGS = [
    ("baseline",      None, None, None),
    ("mmr_lambda_0.7", "mmr", "diversity_lambda", 0.7),
    ("mmr_lambda_0.5", "mmr", "diversity_lambda", 0.5),
    ("mmr_lambda_0.3", "mmr", "diversity_lambda", 0.3),
    ("dpp_theta_0.5",  "dpp", "diversity_theta",  0.5),
    ("dpp_theta_1.0",  "dpp", "diversity_theta",  1.0),
    ("dpp_theta_2.0",  "dpp", "diversity_theta",  2.0),
]
N_QUESTIONS = 30


def ensure_dirs():
    for d in (OUT_DIR, RAW_DIR, CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def call_api(query: str, diversity_method=None, **params) -> dict:
    options = {"enable_generation": False}
    if diversity_method:
        options["diversity_method"] = diversity_method
        for k, v in params.items():
            options[k] = v

    body = json.dumps({"query": query, "options": options}).encode()
    req = request.Request(API, data=body, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except error.HTTPError as e:
        return {"error": str(json.loads(e.read()).get("detail", str(e)))}


def load_raw(cfg_name: str, qid: int) -> dict | None:
    path = RAW_DIR / f"{cfg_name}_q{qid:03d}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_raw(cfg_name: str, qid: int, data: dict):
    (RAW_DIR / f"{cfg_name}_q{qid:03d}.json").write_text(
        json.dumps(data, ensure_ascii=False))


def save_checkpoint(cfg_name: str, metrics: list):
    (CKPT_DIR / f"{cfg_name}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))


def load_checkpoint(cfg_name: str) -> list | None:
    p = CKPT_DIR / f"{cfg_name}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ── 指标 ──────────────────────────────────────────────────────────────

def calc_metrics(comments: list, timing: dict) -> dict:
    n = len(comments)
    if n < 2:
        return {"n": n, "error": "too_few"}

    # 类别
    all_cats = set()
    for c in comments:
        for k in ("category1", "category2", "category3"):
            v = c.get(k)
            if v and isinstance(v, str):
                all_cats.add(v)
    category_coverage = len(all_cats) / 14.0

    # 房型
    rooms = {c.get("fuzzy_room_type", c.get("room_type", "")) for c in comments}
    room_diversity = len(rooms) / n

    # 一级大类
    primary = set()
    facility = {"房间设施", "公共设施", "餐饮设施"}
    service = {"前台服务", "客房服务", "退房/入住效率"}
    location = {"交通便利性", "周边配套", "景观/朝向"}
    price = {"性价比", "价格合理性"}
    experience = {"整体满意度", "安静程度", "卫生状况"}
    for c in comments:
        for k in ("category1", "category2", "category3"):
            v = c.get(k)
            if not v: continue
            if v in facility: primary.add("设施")
            elif v in service: primary.add("服务")
            elif v in location: primary.add("位置")
            elif v in price: primary.add("价格")
            elif v in experience: primary.add("体验")
    topic_breadth = len(primary) / 5.0

    # 相关性
    avg_rel = float(np.mean([c.get("relevance_score", 0) for c in comments]))

    # 多样性 stats（来自后端）
    ds = timing.get("ranking", {}).get("diversity_stats")
    avg_sim = None
    if isinstance(ds, dict) and "mean_pairwise_cosine_sim" in ds:
        avg_sim = round(ds["mean_pairwise_cosine_sim"], 4)

    return {
        "n": n, "category_coverage": round(category_coverage, 4),
        "room_diversity": round(room_diversity, 4),
        "topic_breadth": round(topic_breadth, 4),
        "avg_relevance": round(avg_rel, 4),
        "mean_pairwise_sim": avg_sim,
    }


# ── 主流程 ───────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    questions = json.loads(EVAL_SET.read_text())[:N_QUESTIONS]

    if "--clean" in sys.argv:
        import shutil
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        ensure_dirs()
        print("✓ 已清除旧缓存\n")

    print(f"{'='*65}")
    print(f"多样性消融评估 v2: {N_QUESTIONS} 题 × {len(CONFIGS)} 配置")
    print(f"结果目录: {OUT_DIR}")
    print(f"{'='*65}\n")

    for cfg_name, method, param_key, param_val in CONFIGS:
        # 检查 checkpoint
        existing = load_checkpoint(cfg_name)
        if existing and len(existing) >= N_QUESTIONS:
            print(f"⏭ {cfg_name}: 已完成 (从 checkpoint 加载)")
            continue

        print(f"▶ {cfg_name} ", end="", flush=True)
        completed = len(existing) if existing else 0
        cfg_metrics = existing if existing else []

        t0 = time.time()
        for i in range(completed, len(questions)):
            q = questions[i]

            # 尝试加载缓存
            result = load_raw(cfg_name, q["question_id"])
            if result is None:
                params = {}
                if param_key:
                    params[param_key] = param_val
                result = call_api(q["question"], diversity_method=method, **params)
                save_raw(cfg_name, q["question_id"], result)

            if "error" in result:
                cfg_metrics.append({"error": result["error"], "question_id": q["question_id"]})
                continue

            comments = result.get("references", {}).get("comments", [])
            timing = result.get("timing", {})
            m = calc_metrics(comments, timing)
            m["question_id"] = q["question_id"]
            cfg_metrics.append(m)

            if (i + 1) % 10 == 0:
                save_checkpoint(cfg_name, cfg_metrics)
                print(".", end="", flush=True)

        elapsed = time.time() - t0
        save_checkpoint(cfg_name, cfg_metrics)
        ok = sum(1 for m in cfg_metrics if "error" not in m)
        err = len(cfg_metrics) - ok
        print(f" {elapsed:.0f}s ({ok} ok, {err} err)")

    # ── 汇总 ─────────────────────────────────────────────────────────

    print(f"\n{'='*65}")
    print("评估结果汇总")
    print(f"{'='*65}")

    baseline_means = None
    config_means = {}

    for cfg_name, method, param_key, param_val in CONFIGS:
        metrics = load_checkpoint(cfg_name)
        if not metrics:
            continue
        metrics = [m for m in metrics if "error" not in m]
        if not metrics:
            continue

        avg = {
            "cat": round(np.mean([m["category_coverage"] for m in metrics]), 4),
            "room": round(np.mean([m["room_diversity"] for m in metrics]), 4),
            "topic": round(np.mean([m["topic_breadth"] for m in metrics]), 4),
            "rel": round(np.mean([m["avg_relevance"] for m in metrics]), 4),
        }
        sims = [m["mean_pairwise_sim"] for m in metrics if m["mean_pairwise_sim"] is not None]
        avg["sim"] = round(np.mean(sims), 4) if sims else None

        config_means[cfg_name] = avg

        if cfg_name == "baseline":
            baseline_means = avg

    # 打印每个配置
    for name in CONFIGS:
        _, name, _, _ = name
        m = config_means.get(name)
        if not m:
            continue
        sim_str = f" sim={m['sim']}" if m['sim'] else ""
        print(f"\n  {name}:")
        print(f"    cat_cov={m['cat']:.4f}  room_div={m['room']:.4f}  "
              f"topic={m['topic']:.4f}  rel={m['rel']:.4f}{sim_str}")

        if baseline_means and name != "baseline":
            b = baseline_means
            print(f"    vs baseline: "
                  f"cat {(m['cat']-b['cat'])/b['cat']*100:+.1f}%  "
                  f"room {(m['room']-b['room'])/b['room']*100:+.1f}%  "
                  f"topic {(m['topic']-b['topic'])/b['topic']*100:+.1f}%  "
                  f"rel {(m['rel']-b['rel'])/b['rel']*100:+.1f}%")

    # 排名
    print(f"\n{'='*65}")
    print("综合排名 (diversity×0.6 + relevance×0.4)")
    ranked = []
    for name, m in config_means.items():
        div = (m["cat"] + m["topic"] + m["room"]) / 3
        comp = div * 0.6 + m["rel"] * 0.4
        ranked.append((name, div, m["rel"], comp))
    ranked.sort(key=lambda x: x[3], reverse=True)
    for rank, (name, div, rel, comp) in enumerate(ranked, 1):
        print(f"  {rank}. {name:20s}  div={div:.4f}  rel={rel:.4f}  composite={comp:.4f}")

    # 保存汇总
    summary = {
        "n_questions": N_QUESTIONS,
        "configs": {name: m for name, m in config_means.items()},
        "ranking": [(name, round(comp, 4)) for name, _, _, comp in ranked],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n汇总已保存到: {OUT_DIR / 'summary.json'}")

    print(f"{'='*65}")
    print("评估完成")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
