"""
多样性消融评估 — 最终版（带可视化 + 正确的复合评分公式）

复合评分:
  diversity_quality = 1 − mean_pairwise_sim  (越高=越不相似=越多样)
  composite = relevance × 0.4 + diversity_quality × 0.6

输出:
  eval_results/summary.json         汇总数据
  eval_results/fig_composite.png    综合排名柱状图
  eval_results/fig_tradeoff.png     多样性-相关性权衡散点图
  eval_results/fig_sensitivity.png  λ/θ 敏感性曲线
"""

import json, time, os, sys
import numpy as np
from pathlib import Path
from urllib import request, error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})

API = "http://localhost:8000/api/v1/chat"
ROOT = Path(__file__).parent.parent
EVAL_SET = ROOT / "RAG/data/evaluation/eval_set.json"
OUT_DIR = Path(__file__).parent / "eval_results"
RAW_DIR = OUT_DIR / "raw"
CKPT_DIR = OUT_DIR / "checkpoints"

CONFIGS = [
    ("baseline"           , None  , None                , None),
    ("mmr_lambda_0.7"     , "mmr" , "diversity_lambda"  , 0.7),
    ("mmr_lambda_0.5"     , "mmr" , "diversity_lambda"  , 0.5),
    ("mmr_lambda_0.3"     , "mmr" , "diversity_lambda"  , 0.3),
    ("dpp_theta_0.5"      , "dpp" , "diversity_theta"   , 0.5),
    ("dpp_theta_1.0"      , "dpp" , "diversity_theta"   , 1.0),
    ("dpp_theta_2.0"      , "dpp" , "diversity_theta"   , 2.0),
]
N = 30


def ensure_dirs():
    for d in (OUT_DIR, RAW_DIR, CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)

def api(query, method=None, **p):
    opts = {"enable_generation": False}
    if method:
        opts["diversity_method"] = method
        opts.update(p)
    body = json.dumps({"query": query, "options": opts}).encode()
    req = request.Request(API, data=body, headers={"Content-Type":"application/json"})
    try:
        with request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except error.HTTPError as e:
        return {"error": str(e)}

def jsave(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False))

def jload(path):
    return json.loads(path.read_text()) if path.exists() else None

# ── 指标计算 ──────────────────────────────────────────────────────

FACILITY   = {"房间设施","公共设施","餐饮设施"}
SERVICE    = {"前台服务","客房服务","退房/入住效率"}
LOCATION   = {"交通便利性","周边配套","景观/朝向"}
PRICE      = {"性价比","价格合理性"}
EXPERIENCE = {"整体满意度","安静程度","卫生状况"}

def metrics(comments, timing):
    n = len(comments)
    if n < 2: return None
    all_cats = set()
    for c in comments:
        for k in ("category1","category2","category3"):
            v = c.get(k)
            if v and isinstance(v, str):
                all_cats.add(v)
    rooms = {c.get("fuzzy_room_type",c.get("room_type","")) for c in comments}

    primary = set()
    for c in comments:
        for k in ("category1","category2","category3"):
            v = c.get(k)
            if not v: continue
            if v in FACILITY:   primary.add("设施")
            elif v in SERVICE:  primary.add("服务")
            elif v in LOCATION: primary.add("位置")
            elif v in PRICE:    primary.add("价格")
            elif v in EXPERIENCE: primary.add("体验")

    rel = float(np.mean([c.get("relevance_score",0) for c in comments]))
    sim = timing.get("ranking",{}).get("mean_pairwise_sim")

    return dict(
        cat=len(all_cats)/14, room=len(rooms)/n,
        topic=len(primary)/5, rel=rel,
        sim=sim if sim is not None else None
    )

# ── 主流程 ────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    if "--clean" in sys.argv:
        import shutil; shutil.rmtree(OUT_DIR, ignore_errors=True); ensure_dirs()
        print("✓ 缓存已清除\n")

    questions = json.loads(EVAL_SET.read_text())[:N]
    print(f"多样性消融评估: {N} 题 × {len(CONFIGS)} 配置\n")

    for name, method, pk, pv in CONFIGS:
        existing = jload(CKPT_DIR / f"{name}.json")
        if existing and len(existing) >= N:
            print(f"⏭  {name}: 已缓存")
            continue

        print(f"▶  {name} ", end="", flush=True)
        t0 = time.time()
        results = existing if existing else []

        for i in range(len(results), N):
            q = questions[i]
            path = RAW_DIR / f"{name}_q{q['question_id']:03d}.json"
            result = jload(path)
            if result is None:
                result = api(q["question"], method, **{pk: pv} if pk else {})
                jsave(path, result)
            if result and "error" not in result:
                m = metrics(
                    result.get("references",{}).get("comments",[]),
                    result.get("timing",{})
                )
                if m: m["qid"] = q["question_id"]
                results.append(m or {"error":True,"qid":q["question_id"]})
            else:
                results.append({"error":True,"qid":q["question_id"]})
            if (i+1) % 10 == 0:
                jsave(CKPT_DIR / f"{name}.json", results)
                print(".", end="", flush=True)

        jsave(CKPT_DIR / f"{name}.json", results)
        ok = sum(1 for r in results if r and "error" not in r and "sim" in r)
        print(f" {time.time()-t0:.0f}s ({ok} ok)")

    # ── 汇总 ───────────────────────────────────────────────────

    vals = {}
    for name, *_ in CONFIGS:
        raw = jload(CKPT_DIR / f"{name}.json")
        if not raw: continue
        data = [d for d in raw if d and "error" not in d and d.get("sim") is not None]
        if not data: continue
        vals[name] = {k: round(float(np.mean([d[k] for d in data])), 4) for k in ("cat","room","topic","rel","sim")}

    b = vals.get("baseline")
    if not b:
        print("ERROR: no baseline")
        return

    # ── 排名 ───────────────────────────────────────────────────
    # composite = rel × 0.4 + (1-sim) × 0.6

    ranking = []
    for name, d in vals.items():
        dq = 1.0 - d["sim"]
        comp = d["rel"] * 0.4 + dq * 0.6
        ranking.append((name, d["rel"], dq, comp, d))

    ranking.sort(key=lambda x: x[3], reverse=True)

    print(f"\n{'='*65}")
    print("综合排名  composite = relevance×0.4 + diversity_quality×0.6")
    print("          diversity_quality = 1 − mean_pairwise_sim")
    print(f"{'='*65}")
    for rank, (name, rel, dq, comp, d) in enumerate(ranking, 1):
        flag = " ★" if rank == 1 else ""
        print(f"  {rank}. {name:20s}  rel={rel:.4f}  dq={dq:.4f}  "
              f"sim={d['sim']:.4f}  composite={comp:.4f}{flag}")

    # 保存
    summary = {
        "formula": "composite = relevance×0.4 + (1-mean_pairwise_sim)×0.6",
        "baseline": b,
        "configs": vals,
        "ranking": [(n, round(c,4)) for n,_,_,c,_ in ranking],
    }
    jsave(OUT_DIR / "summary.json", summary)
    print(f"\n✓ summary.json")

    # ── 可视化 ────────────────────────────────────────────────

    names   = [r[0] for r in ranking]
    rels    = [r[1] for r in ranking]
    div_qs  = [r[2] for r in ranking]
    comps   = [r[3] for r in ranking]
    sims    = [r[4]["sim"] for r in ranking]

    # ---- 图1: 综合排名 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if n == ranking[0][0] else
              "#e74c3c" if n == "baseline" else "#3498db" for n in names]
    bars = ax.bar(range(len(names)), comps, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, comps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Composite Score", fontweight="bold")
    ax.set_title("Diversity Re-ranking: Composite Score\n"
                 "(relevance×0.4 + diversity_quality×0.6)", fontweight="bold")
    ax.set_ylim(min(comps) * 0.92, max(comps) * 1.06)
    fig.savefig(OUT_DIR / "fig_composite.png")
    plt.close()
    print("✓ fig_composite.png")

    # ---- 图2: 权衡散点图 ----
    fig, ax = plt.subplots(figsize=(9, 7))
    for n, r, dq in zip(names, rels, div_qs):
        if n == "baseline":
            ax.scatter(r, dq, s=180, c="#e74c3c", marker="s",
                       edgecolors="black", linewidth=1.5, zorder=5)
        elif "mmr" in n:
            ax.scatter(r, dq, s=140, c="#3498db", marker="o",
                       edgecolors="white", zorder=4)
        else:
            ax.scatter(r, dq, s=140, c="#9b59b6", marker="^",
                       edgecolors="white", zorder=4)
        ax.annotate(n.replace("_lambda_"," λ=").replace("_theta_"," θ="),
                    (r, dq), textcoords="offset points", xytext=(5, 8),
                    ha="center", fontsize=8)
    ax.set_xlabel("Relevance Score →", fontweight="bold")
    ax.set_ylabel("Diversity Quality (1−sim) →", fontweight="bold")
    ax.set_title("Diversity–Relevance Trade-off\n(upper-right quadrant = strict improvement over baseline)",
                 fontweight="bold")
    ax.axhline(y=1-b.get("sim",0.5), color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=b["rel"], color="gray", linestyle=":", alpha=0.5)
    fig.savefig(OUT_DIR / "fig_tradeoff.png")
    plt.close()
    print("✓ fig_tradeoff.png")

    # ---- 图3: 敏感性 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    lambdas = [0.3, 0.5, 0.7]
    m_rel = [vals.get(f"mmr_lambda_{l}",{}).get("rel",np.nan) for l in lambdas]
    m_div = [1-vals.get(f"mmr_lambda_{l}",{}).get("sim",0) for l in lambdas]
    ax1.plot(lambdas, m_rel, "bo-", label="Relevance", linewidth=2, markersize=8)
    ax1.plot(lambdas, m_div, "go-", label="Diversity Quality", linewidth=2, markersize=8)
    ax1.axhline(y=b["rel"], color="red", linestyle="--", alpha=0.5)
    ax1.axhline(y=1-b.get("sim",0.5), color="orange", linestyle="--", alpha=0.5)
    ax1.set_xlabel("λ (higher → relevance-focused)")
    ax1.set_title("MMR λ Sensitivity")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    thetas = [0.5, 1.0, 2.0]
    d_rel = [vals.get(f"dpp_theta_{t}",{}).get("rel",np.nan) for t in thetas]
    d_div = [1-vals.get(f"dpp_theta_{t}",{}).get("sim",0) for t in thetas]
    ax2.plot(thetas, d_rel, "bo-", label="Relevance", linewidth=2, markersize=8)
    ax2.plot(thetas, d_div, "go-", label="Diversity Quality", linewidth=2, markersize=8)
    ax2.axhline(y=b["rel"], color="red", linestyle="--", alpha=0.5)
    ax2.axhline(y=1-b.get("sim",0.5), color="orange", linestyle="--", alpha=0.5)
    ax2.set_xlabel("θ (higher → quality-focused)")
    ax2.set_title("DPP θ Sensitivity")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.savefig(OUT_DIR / "fig_sensitivity.png")
    plt.close()
    print("✓ fig_sensitivity.png")

    print(f"\n全部结果: {OUT_DIR}/")


if __name__ == "__main__":
    main()
