"""
对比 v0/v1 Prompt 的离线属性（不调用 LLM）：
- 长度
- 字符 token 估算
- 是否包含输出契约 / 自校验等结构标记
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "rag-service" / "modules"))

from prompts import build_prompt  # noqa: E402

EVAL_SET = ROOT / "RAG" / "data" / "evaluation" / "eval_set.json"
OUT = ROOT / "RAG" / "data" / "evaluation" / "prompt_compare.json"


def fake_context(question: str):
    """构造一份固定上下文，让 v0/v1 的差异只来自模板本身。"""
    rewritten = [
        {"query": question, "weight": 1.0},
        {"query": question + " 评价", "weight": 0.6},
    ]
    ranked = [
        {
            "comment_id": f"c{i}",
            "comment": "房间挺大的，干净；早餐种类不算多，但味道不错。前台办理稍慢。",
            "metadata": {
                "score": 4.5, "publish_date": "2024-08-12",
                "useful_count": 3, "review_count": 12,
                "room_type": "高级大床房", "fuzzy_room_type": "大床房",
            },
        }
        for i in range(1, 6)
    ]
    summaries = [
        {"summary": "总体评价正面，餐饮表现稳定。",
         "metadata": {"category": "整体满意度", "keywords": "干净 早餐 服务"}},
    ]
    return rewritten, ranked, summaries


def stats(prompt: str) -> dict:
    return {
        "chars": len(prompt),
        "lines": prompt.count("\n") + 1,
        "has_self_check": "自校验" in prompt,
        "has_output_contract": "输出契约" in prompt or "## 直接回答" in prompt,
        "has_chitchat_constraint": "禁词" in prompt or "不得提及" in prompt,
    }


def main():
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    today = datetime(2026, 6, 9)

    rows = []
    for q in eval_set[:20]:
        rewritten, ranked, summaries = fake_context(q["question"])
        p0 = build_prompt("v0", q["question"], rewritten, ranked, summaries,
                          need_retrieval=True, today=today)
        p1 = build_prompt("v1", q["question"], rewritten, ranked, summaries,
                          need_retrieval=True, today=today)
        rows.append({
            "question": q["question"],
            "v0": stats(p0),
            "v1": stats(p1),
        })

    # 闲聊分支
    chat_rows = []
    for q_text in ["你好啊", "今天天气怎么样？", "你是谁？", "再见"]:
        p0 = build_prompt("v0", q_text, need_retrieval=False, today=today)
        p1 = build_prompt("v1", q_text, need_retrieval=False, today=today)
        chat_rows.append({"q": q_text, "v0": stats(p0), "v1": stats(p1)})

    # 汇总
    def agg(rs, key):
        vals = [r[key]["chars"] for r in rs]
        return {
            "avg_chars": round(sum(vals) / len(vals), 1),
            "max_chars": max(vals),
            "min_chars": min(vals),
        }

    summary = {
        "retrieval": {"v0": agg(rows, "v0"), "v1": agg(rows, "v1")},
        "chitchat": {"v0": agg(chat_rows, "v0"), "v1": agg(chat_rows, "v1")},
        "v0_has_output_contract_rate": sum(r["v0"]["has_output_contract"] for r in rows) / len(rows),
        "v1_has_output_contract_rate": sum(r["v1"]["has_output_contract"] for r in rows) / len(rows),
        "v0_has_self_check_rate": sum(r["v0"]["has_self_check"] for r in rows) / len(rows),
        "v1_has_self_check_rate": sum(r["v1"]["has_self_check"] for r in rows) / len(rows),
    }

    OUT.write_text(json.dumps(
        {"summary": summary, "retrieval_samples": rows, "chitchat_samples": chat_rows},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # 展示一份示例 prompt
    sample_q = eval_set[0]["question"]
    rewritten, ranked, summaries = fake_context(sample_q)
    print("\n=== SAMPLE v1 PROMPT ===\n")
    print(build_prompt("v1", sample_q, rewritten, ranked, summaries,
                       need_retrieval=True, today=today))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
