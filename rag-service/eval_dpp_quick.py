"""DPP 修复快速验证 — baseline + 3 组 DPP"""
import json, time, urllib.request, urllib.error, numpy as np
from pathlib import Path

API = "http://localhost:8000/api/v1/chat"
questions = json.loads(Path(__file__).parent.parent.joinpath("RAG/data/evaluation/eval_set.json").read_text())[:30]

CONFIGS = [
    ("baseline",      None),
    ("dpp_theta_0.5", ("dpp", {"diversity_theta": 0.5})),
    ("dpp_theta_1.0", ("dpp", {"diversity_theta": 1.0})),
    ("dpp_theta_2.0", ("dpp", {"diversity_theta": 2.0})),
]

def call(query, method_info):
    opts = {"enable_generation": False}
    if method_info:
        method, params = method_info
        opts["diversity_method"] = method
        opts.update(params)
    req = urllib.request.Request(API, data=json.dumps({"query": query, "options": opts}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

def extract(comments):
    cs = set()
    for c in comments:
        for k in ("category1","category2","category3"):
            v = c.get(k)
            if v and isinstance(v, str):
                cs.add(v)
    rooms = {c.get("fuzzy_room_type", c.get("room_type","")) for c in comments}
    rel = np.mean([c.get("relevance_score",0) for c in comments])
    return len(cs)/14, len(rooms)/len(comments), rel

results = {}
for name, method_info in CONFIGS:
    print(f"▶ {name} ", end="", flush=True)
    cats, rooms, rels, sims = [], [], [], []
    for i, q in enumerate(questions):
        r = call(q["question"], method_info)
        comments = r["references"]["comments"]
        ca, ro, re = extract(comments)
        cats.append(ca); rooms.append(ro); rels.append(re)
        ds = r.get("timing",{}).get("ranking",{}).get("diversity_stats")
        if isinstance(ds, dict) and ds.get("mean_pairwise_cosine_sim") is not None:
            sims.append(ds["mean_pairwise_cosine_sim"])
        if (i+1) % 10 == 0: print(".", end="", flush=True)
    results[name] = {"cat": np.mean(cats), "room": np.mean(rooms), "rel": np.mean(rels),
                      "sim": np.mean(sims) if sims else None}
    print(f" cat={results[name]['cat']:.4f} room={results[name]['room']:.4f} rel={results[name]['rel']:.4f}"
          f" sim={results[name]['sim']:.4f}" if sims else " sim=N/A")

b = results["baseline"]
print(f"\n{'='*60}")
print(f"vs baseline (cat={b['cat']:.4f} room={b['room']:.4f} rel={b['rel']:.4f}):")
for name in results:
    if name == "baseline": continue
    m = results[name]
    print(f"  {name}: cat {m['cat']-b['cat']:+.1%}  room {m['room']-b['room']:+.1%}  rel {m['rel']-b['rel']:+.1%}")
print("done")
