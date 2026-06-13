"""DPP 修复验证 — 只跑 DPP 三组 + baseline 做参照"""
import json, time, urllib.request, urllib.error, numpy as np
from pathlib import Path

API = "http://localhost:8000/api/v1/chat"
questions = json.loads(Path(__file__).parent.parent.joinpath("RAG/data/evaluation/eval_set.json").read_text())[:30]

CONFIGS = [
    ("baseline",      None, None, None),
    ("dpp_theta_0.5", "dpp", "diversity_theta", 0.5),
    ("dpp_theta_1.0", "dpp", "diversity_theta", 1.0),
    ("dpp_theta_2.0", "dpp", "diversity_theta", 2.0),
]

def call(query, method=None, **p):
    opts = {"enable_generation": False}
    if method:
        opts["diversity_method"] = method
        for k,v in p.items(): opts[k] = v
    req = urllib.request.Request(API, data=json.dumps({"query": query, "options": opts}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

def cats(c): return {v for k in ("category1","category2","category3") if (v:=c.get(k)) and isinstance(v,str)}

all_r = {}
for name, method, pk, pv in CONFIGS:
    print(f"▶ {name} ", end="", flush=True)
    metrics = []
    for q in questions:
        r = call(q["question"], method, **{pk: pv} if pk else {})
        comments = r["references"]["comments"]
        cs = set(); [cs.update(cats(c)) for c in comments]
        rooms = {c.get("fuzzy_room_type", c.get("room_type","")) for c in comments}
        rel = np.mean([c.get("relevance_score",0) for c in comments])
        ds = r.get("timing",{}).get("ranking",{}).get("diversity_stats")
        sim = ds.get("mean_pairwise_cosine_sim") if isinstance(ds, dict) else None
        metrics.append({"cat": len(cs)/14, "room": len(rooms)/len(comments),
                        "rel": rel, "sim": sim})
    avg = lambda k: round(np.mean([m[k] for m in metrics]), 4)
    sims = [m["sim"] for m in metrics if m["sim"] is not None]
    print(f" cat={avg('cat'):.4f} room={avg('room'):.4f} rel={avg('rel'):.4f} sim={round(np.mean(sims),4) if sims else 'N/A'}")
    all_r[name] = metrics

if "baseline" in all_r:
    b = all_r["baseline"]
    for name in all_r:
        if name == "baseline": continue
        m = all_r[name]
        dc = (avg('cat') - avg_for('cat',b)) / max(avg_for('cat',b), 0.001) * 100 if False else ""
    # Quick compare
    b_avg = {k: np.mean([x[k] for x in b]) for k in ("cat","room","rel")}
    for name in all_r:
        if name == "baseline": continue
        m = all_r[name]
        m_avg = {k: np.mean([x[k] for x in m]) for k in ("cat","room","rel")}
        print(f"  {name}: cat {m_avg['cat']-b_avg['cat']:+.1%} room {m_avg['room']-b_avg['room']:+.1%} rel {m_avg['rel']-b_avg['rel']:+.1%}")

print("done")
