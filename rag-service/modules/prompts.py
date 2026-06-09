"""
Prompt 模板集中管理。

- v0: 还原 generator.py 当前实现，用作基线。
- v1: 五段式结构化 + 输出契约 + 引用自校验 + 闲聊硬约束。

使用方式（drop-in 替换 ResponseGenerator._build_prompt）：

    from prompts import build_prompt, PROMPT_VERSION
    prompt = build_prompt(version="v1", user_query=..., ...)
"""

from __future__ import annotations

from datetime import datetime, date as _date
from typing import Optional


PROMPT_VERSION = "v1.0"

# 追问触发词：仅当 user_query 命中其中之一时，把上一轮历史注入 prompt
FOLLOWUP_TRIGGERS = (
    "那", "还有", "这个", "那个", "它", "他们", "她们",
    "是吗", "真的吗", "为什么", "怎么", "呢？", "呢?", "嗯",
)


def is_followup(user_query: str) -> bool:
    if not user_query:
        return False
    return any(t in user_query for t in FOLLOWUP_TRIGGERS)


def coarsen_date(publish_date: str, today: Optional[_date] = None) -> str:
    """把精确日期转为粗粒度时效标签，避免 LLM 直接抄日期。"""
    if not publish_date:
        return "时间未知"
    today = today or datetime.today().date()
    try:
        pd = datetime.strptime(str(publish_date)[:10], "%Y-%m-%d").date()
    except Exception:
        return "时间未知"
    days = (today - pd).days
    if days < 0:
        return "近期"
    if days <= 90:
        return "近期"
    if days <= 180:
        return "半年内"
    if days <= 365:
        return "一年内"
    return "一年以上"


# ── v0：基线（与 generator.py 现状对齐） ──────────────────────────────

def _v0_retrieval(user_query, rewritten_queries, ranked_comments, summaries,
                  today, history) -> str:
    history_context = ""
    if history and history.get("user") and history.get("assistant"):
        history_context = (
            f"\n【上一轮对话】\n用户：{history['user']}\n助手：{history['assistant']}\n"
        )
    if not today:
        today = datetime.today()
    date_str = f"{today.year}年{today.month}月{today.day}日"

    queries_context = ""
    if rewritten_queries:
        queries_context += "【问题解析】\n系统识别到用户可能关注以下方面：\n"
        queries_context += "\n".join(
            [f"- {q['query']}（意图权重为{q['weight']}）" for q in rewritten_queries]
        )
        queries_context += "\n注意：权重信息是用来帮助你区分意图主次的，**不得**向用户输出权重相关信息。"

    if ranked_comments:
        comments_context = "【相关用户评论】\n"
        for i, c in enumerate(ranked_comments, 1):
            comments_context += (
                f"\n【评论{i}】\n"
                f"评分: {c['metadata']['score']}（满分5分）\n"
                f"发布日期: {c['metadata']['publish_date']}\n"
                f"评论文本: {c['comment']}\n"
                f"点赞数: {c['metadata']['useful_count']}\n"
                f"评论数: {c['metadata']['review_count']}\n"
                f"房型: {c['metadata']['room_type']}\n"
            )
    else:
        comments_context = "【未检索到相关用户评论】\n"

    summaries_context = ""
    if summaries:
        summaries_context += "【相关评论摘要】\n"
        for s in summaries:
            summaries_context += (
                f"\n【{s['metadata']['category']}类别摘要】\n"
                f"关键词: {s['metadata']['keywords']}\n摘要: {s['summary']}\n"
            )
        summaries_context += (
            "\n注意：评论摘要是用来给到你更丰富的概览信息的，但用户只能看到【相关用户评论】"
            "的引用而看不到摘要的引用，因此在回复中你可以给出摘要中的模糊信息，但**不得过于精确"
            "因为用户无法溯源**，也**不得告诉用户你引用了摘要**，**更不得将其当作评论引用输出"
            '"评论x"**。若摘要中的信息与用户问题无关，直接忽略即可，**不需要**做出任何额外说明。\n'
        )

    return f"""
你是广州花园酒店的智能客服助手，需要基于用户评论为用户提供准确、高质量、有帮助、简洁的回答。

今天是：{date_str}
{history_context}
用户问题：{user_query}

{queries_context}

{comments_context}

{summaries_context}

【回答要求】
1. 综合以上评论信息，给出客观、全面的回答
2. 回答要有条理，突出重点
3. 如有正面和负面评价，都要提及，保持客观。注意给出的参考评论并不代表所有，切忌以偏概全给出"绝对化"的表述
4. 语气要专业、亲切
5. 回答长度适中，不要过于冗长
6. 不得大段或连续照抄用户评论，严禁全文都在引用用户评论却并没有思考提炼总结。相似内容能合并就合并，不要分开引用（合并后注意不得同时列出超过3条参考评论，使用"等"替代）
7. 一般来说越靠前的评论，其重要性越高，但你也可以自行判断自行选择
8. 不得在回复中罗列用户评论的具体日期，但当用户问题时效性敏感时，可以大致提一下参考评论的时间范围
9. 引用【相关用户评论】中某一条评论独特内容时，应使用引用标记 [[ref:N]]（N为评论序号）标注来源；针对参考评论总体或【xx类别摘要】进行归纳总结时**无需**标注。
10. 不得同时列出超过3条引用，即最多 [[ref:1,3,5]]。如需同时引用超过3条评论，则应只保留排名最靠前的2条并加"等"字。多条引用写在同一个标记内用逗号分隔。
11. 如果评论信息不足以回答问题，诚实说明
12. 所有的回复必须仅依赖检索到的用户评论及摘要，不得出现自作主张的幻觉回复
13. 使用Markdown格式输出，不得出现 "```markdown", "```" 标记

用户问题：{user_query}

请给出你的回答：
"""


def _v0_chitchat(user_query, history) -> str:
    history_context = ""
    if history and history.get("user") and history.get("assistant"):
        history_context = (
            f"\n【上一轮对话】\n用户：{history['user']}\n助手：{history['assistant']}\n"
        )
    return f"""
你是广州花园酒店的智能客服助手。
{history_context}
用户问题：{user_query}

请直接回答用户的问题。注意：
- 如果是问候或闲聊，友好回应
- 如果是通用问题，给出简洁准确的回答
- 如果用户的问题是对上一轮对话的追问，请结合上下文理解用户意图
- 语气要亲切专业
- 使用Markdown格式输出，不得出现 "```markdown", "```" 标记
"""


# ── v1：结构化 + 输出契约 + 自校验 ────────────────────────────────────

def _v1_retrieval(user_query, rewritten_queries, ranked_comments, summaries,
                  today, history) -> str:
    today = today or datetime.today()
    date_str = f"{today.year}年{today.month}月{today.day}日"

    # 历史按需注入
    history_block = ""
    if history and history.get("user") and history.get("assistant") and is_followup(user_query):
        history_block = f"上一轮对话｜用户：{history['user']}｜助手：{history['assistant'][:80]}"

    # 意图块（去掉权重括号说明，更紧凑）
    queries_block = "无"
    if rewritten_queries:
        queries_block = "；".join(
            [f"{q['query']}(w={q['weight']})" for q in rewritten_queries]
        )

    # 评论块：粗粒度日期 + 字段语义化 + 控长
    if ranked_comments:
        lines = []
        for i, c in enumerate(ranked_comments, 1):
            md = c.get("metadata", {})
            tag = coarsen_date(md.get("publish_date", ""), today.date()
                               if isinstance(today, datetime) else today)
            txt = (c.get("comment") or "").strip().replace("\n", " ")
            if len(txt) > 220:
                txt = txt[:220] + "…"
            primary = c.get("primary_chunk")
            head = (
                f"[评论{i}] 评分{md.get('score','?')}｜{tag}｜{md.get('room_type','-')}"
            )
            if primary and primary.get("text"):
                key_sent = primary["text"].strip().replace("\n", " ")
                if len(key_sent) > 80:
                    key_sent = key_sent[:80] + "…"
                # 关键句作为引用主体；评论全文仅当较长时才补充
                if len(txt) > 80:
                    lines.append(f"{head}\n  关键句：{key_sent}\n  全文：{txt}")
                else:
                    lines.append(f"{head}\n  关键句：{key_sent}")
            else:
                lines.append(f"{head}\n  内容：{txt}")
        comments_block = "\n".join(lines)
    else:
        comments_block = "（未检索到相关用户评论）"

    # 摘要块：把警示提到前面
    summaries_block = ""
    if summaries:
        summaries_block = (
            "【背景摘要：仅供你判断使用，**不得**作为引用源、不得在回答中提及"
            '"摘要"二字、**不得**用作 [[ref:..]] 的来源】\n'
        )
        for s in summaries:
            md = s.get("metadata", {})
            summaries_block += (
                f"- {md.get('category','?')}（关键词：{md.get('keywords','')}）：{s.get('summary','')}\n"
            )

    return f"""# 角色
你是广州花园酒店的智能客服助手。所有回答必须**仅基于**下面提供的住客评论与背景摘要。

# 上下文
- 今日：{date_str}
- 历史：{history_block or "（无相关上一轮）"}
- 用户问题：{user_query}
- 系统识别意图：{queries_block}

# 相关用户评论（编号 1..N，可被 [[ref:N]] 引用）
{comments_block}

{summaries_block}

# 任务
用自然中文回答用户问题，篇幅在 150–300 字之间，按问题复杂度自适应。结构如下，**禁止使用任何 ## 大标题**：

1. **开头一段**：1–2 句直接给出对用户问题的明确结论或概括（句末贴 [[ref:N]] 引用）。若评论不足以回答，第一句直说"目前评论中暂未涉及……"，无需 bullet。
2. **要点列表**：紧接着用 2–3 个 `- ` 起始的 Markdown 无序列表项展开关键评价；同主题合并、正负兼顾，每条末尾带一个 [[ref:..]] 标记。
3. **风险提示（仅在确有显著负面/限制项时输出）**：另起一段，以"需要留意的是"或"提醒一下"开头，单段 1 句话，句末附 [[ref:N]]。无显著负面则**整段省略**。

# 约束
C1 引用：仅在确凿对应时使用 [[ref:N]]。同一标记内逗号分隔，最多 3 条；超出时在 ]] **之外**加"等"字，例如 `[[ref:1,3]]等`。**严禁** `[[ref:1,3,等]]`、`[[ref:1]][[ref:2]]`。
C2 反幻觉：仅基于评论与摘要；不得调用 API、不得推测库存/活动/价格。
C3 客观：避免"全部 / 总是 / 绝对"；不以偏概全。
C4 表达：开头段不要写"## 直接回答"这种标题；正文不出现"评论"、"摘要"、"参考"等元词；不罗列日期；时效敏感时只用"近期/半年内/一年内"。
C5 输出：纯 Markdown，不要包 ```markdown / ``` 代码栅栏，不要任何 H1/H2/H3 标题。

# 自校验（内部完成，不要输出）
- [[ref:..]][[ref:..]] 紧邻 → 合并为 [[ref:N,M]]
- 单标记内 > 3 条 → 截到 2 条 + ]]外的"等"
- "等"字误写在 ]] 内 → 移到 ]] 之外
- 是否出现"##"标题或"摘要"二字 → 删除
- 引用编号是否真在文中相邻陈述里出现过？否则删掉

请直接输出最终回答。
"""


def _v1_chitchat(user_query, history) -> str:
    history_block = ""
    if history and history.get("user") and history.get("assistant") and is_followup(user_query):
        history_block = f"（上一轮｜{history['user']} → {history['assistant'][:60]}）"
    return f"""# 角色
你是广州花园酒店的智能客服助手。

# 任务
友好回应用户的问候 / 通用问询；≤ 80 字，单段。

# 严格约束
- 不得提及"评论 / 住客 / 评价"等字样
- 不得使用 [[ref:..]] 标记
- 不得编造房态、活动、价格、设施细节

# 当前用户输入
{user_query}
{history_block}

请输出回答。
"""


# ── 统一入口 ────────────────────────────────────────────────────────

def build_prompt(version: str, user_query: str, rewritten_queries=None,
                 ranked_comments=None, summaries=None, need_retrieval: bool = True,
                 today=None, history: Optional[dict] = None) -> str:
    if version == "v0":
        if not need_retrieval:
            return _v0_chitchat(user_query, history)
        return _v0_retrieval(user_query, rewritten_queries, ranked_comments,
                             summaries, today, history)
    if version == "v1":
        if not need_retrieval:
            return _v1_chitchat(user_query, history)
        return _v1_retrieval(user_query, rewritten_queries, ranked_comments,
                             summaries, today, history)
    raise ValueError(f"unknown prompt version: {version}")
