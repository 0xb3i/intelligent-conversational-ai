"""
多轮对话上下文管理：ConversationManager + QueryResolver + ContextCompressor

管理完整的多轮对话状态，包括：
- 轮次存储与滑动窗口
- 查询消解（指代/省略补全）
- 对话摘要压缩
- 分层上下文组装（用于 prompt 注入）
- Token 预算控制
- 序列化/反序列化（对接 sessionStorage）
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    """单轮对话记录"""
    turn_id: int
    user_query: str                          # 原始用户输入
    resolved_query: str                      # 消解后的完整 query
    assistant_response: str                  # 助手回复全文
    references: list = field(default_factory=list)   # 本轮的 reference comment IDs
    intent_expansion: Optional[dict] = None
    intent_detection: Optional[dict] = None
    timestamp: float = field(default_factory=time.time)


class ConversationManager:
    """
    多轮对话管理器

    ┌─────────────────────────────────────────┐
    │  turns: List[Turn]    完整轮次记录        │
    │  long_term_summary: str  旧轮次压缩摘要    │
    │  user_profile: dict   提取的用户偏好       │
    │                                          │
    │  + add_turn(q, a, refs)                  │
    │  + get_retrieval_context → 用于 query 补全 │
    │  + get_generation_context → 用于 prompt    │
    │  + maybe_summarize() → LLM 压缩旧轮次      │
    │  + to_dict() / from_dict() → 序列化       │
    └─────────────────────────────────────────┘
    """

    # ── 可配置参数 ────────────────────────────────────────────
    MAX_TURNS: int = 5               # 保留最近 N 轮完整内容
    MAX_SUMMARY_TURNS: int = 10      # 总共保留的轮次（超出部分入摘要）
    SUMMARY_TRIGGER_TURNS: int = 4   # 超过此轮数触发 LLM 摘要
    MAX_CONTEXT_TOKENS: int = 1200   # 注入 prompt 的历史 token 上限
    MAX_RECENT_RESPONSE_LEN: int = 200  # 最近轮次回答截断长度

    def __init__(self, session_id: str = ""):
        self.session_id = session_id
        self.turns: list[Turn] = []
        self.long_term_summary: str = ""       # 压缩后的旧轮次摘要
        self.user_preferences: dict = {}        # 提取的用户关注偏好
        self._turn_counter: int = 0
        self.created_at: float = time.time()
        self.last_active: float = time.time()

    # ── 轮次管理 ────────────────────────────────────────────

    def add_turn(
        self,
        user_query: str,
        resolved_query: str,
        assistant_response: str,
        references: list | None = None,
        intent_expansion: dict | None = None,
        intent_detection: dict | None = None,
    ) -> Turn:
        """记录新的一轮对话"""
        self._turn_counter += 1
        turn = Turn(
            turn_id=self._turn_counter,
            user_query=user_query,
            resolved_query=resolved_query,
            assistant_response=assistant_response,
            references=references or [],
            intent_expansion=intent_expansion,
            intent_detection=intent_detection,
            timestamp=time.time(),
        )
        self.turns.append(turn)
        self.last_active = time.time()

        # 滑动窗口：超过 MAX_SUMMARY_TURNS 的旧轮次移入摘要
        while len(self.turns) > self.MAX_SUMMARY_TURNS:
            self.turns.pop(0)

        return turn

    # ── 上下文构建 ───────────────────────────────────────────

    def get_retrieval_context(self) -> dict:
        """
        返回用于检索阶段的上下文信息。

        用于 query resolution：将对话历史中的关键实体和话题
        传递给 QueryResolver，帮助消解指代和省略。
        """
        if not self.turns:
            return {"has_history": False}

        # 最近 2 轮完整对话
        recent = []
        for t in self.turns[-2:]:
            recent.append({
                "user": t.user_query,
                "assistant": t.assistant_response[:150],  # 截断
                "entities": t.intent_detection or {},
            })

        # 提取最近的实体（房型偏好等）
        entities = {}
        for t in reversed(self.turns):
            if t.intent_detection and t.intent_detection.get("room_type"):
                entities["last_room_type"] = t.intent_detection["room_type"]
                break

        return {
            "has_history": True,
            "recent_turns": recent,
            "entity_memory": entities,
            "long_term_summary": self.long_term_summary,
        }

    def get_generation_context(self) -> str:
        """
        按 token 预算分层组装生成阶段的历史上下文文本。
        用于注入到 prompt 的「对话历史」区域。

        分层策略：
          Layer 1 — 最近 2 轮完整问答（追问时关键）
          Layer 2 — 更早轮次摘要（3-5 轮）
          Layer 3 — 长期摘要（5 轮以上）
        """
        if not self.turns:
            return ""

        budget = self.MAX_CONTEXT_TOKENS
        parts: list[str] = []

        # Layer 1: 最近 2 轮完整对话
        for t in self.turns[-2:]:
            answer_short = t.assistant_response[:self.MAX_RECENT_RESPONSE_LEN]
            if len(t.assistant_response) > self.MAX_RECENT_RESPONSE_LEN:
                answer_short += "…"
            block = f"用户：{t.user_query}\n助手：{answer_short}"
            cost = self._estimate_tokens(block)
            if budget - cost > 200:
                parts.append(block)
                budget -= cost

        # Layer 2: 中间轮次摘要（第 3 到第 MAX_TURNS 轮）
        middle_turns = self.turns[:-2]
        if middle_turns and budget > 100:
            mid_summary = self._build_middle_summary(middle_turns)
            cost = self._estimate_tokens(mid_summary)
            if budget - cost > 100:
                parts.append(mid_summary)
                budget -= cost

        # Layer 3: 长期摘要
        if self.long_term_summary and budget > 80:
            lt_block = f"【更早的对话摘要】{self.long_term_summary}"
            cost = self._estimate_tokens(lt_block)
            if budget - cost > 50:
                parts.append(lt_block)

        return "\n".join(parts) if parts else ""

    def is_followup(self, current_query: str) -> bool:
        """
        判断当前 query 是否为上一轮的追问。
        基于启发式规则（最终决策由服务端 QueryResolver 的 LLM 判断）。
        """
        if not self.turns:
            return False

        q = current_query.strip()

        # 规则 1: 极短 query（≤ 5 字）大概率是追问（除非是明显的完整问句）
        if len(q) <= 5:
            return True

        # 规则 2: 包含指代词 — 需要上下文
        if re.search(r'(那|这|它|他们|她们|刚才|上面|前面|之前)', q):
            return True

        # 规则 3: 短 query（≤ 10 字）且缺少完整问句结构 → 追问
        if len(q) <= 10 and not re.search(r'(什么|怎么|如何|哪里|多远|多少|能不能|有没有|可以|多少钱)', q):
            return True

        # 规则 4: 以追问/衔接词开头
        if re.match(r'^(还有|那么|所以|那|呢|吗|吧)', q):
            return True

        # 规则 5: 比较追问标记
        if re.search(r'(哪个更好|对比|比较|区别|差别)', q):
            return True

        # 规则 6: 省略主语的问句结构
        if re.match(r'^.{0,8}(呢|吗|吧|怎么样|好不好|行不行)', q):
            return True

        return False

    # ── 摘要压缩 ────────────────────────────────────────────

    def maybe_summarize(self) -> tuple[bool, str]:
        """
        检查是否需要触发摘要压缩。
        返回 (was_triggered, summary_text)。
        实际压缩逻辑由 rag_system 层调用 LLM 完成。
        """
        if len(self.turns) <= self.SUMMARY_TRIGGER_TURNS:
            return False, ""

        # 将最早的 2 轮移出并标记为待压缩
        oldest = self.turns[:2]
        return True, self._render_turns_for_summary(oldest)

    def apply_summary(self, summary_text: str):
        """应用 LLM 生成的摘要，清理旧轮次"""
        if summary_text:
            if self.long_term_summary:
                self.long_term_summary += " | " + summary_text
            else:
                self.long_term_summary = summary_text
        # 移除已被摘要的旧轮次
        if len(self.turns) > self.MAX_TURNS:
            self.turns = self.turns[-self.MAX_TURNS:]

    # ── 序列化 ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        """序列化为字典，用于前端 sessionStorage 存储"""
        return {
            "session_id": self.session_id,
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "user_query": t.user_query,
                    "resolved_query": t.resolved_query,
                    "assistant_response": t.assistant_response,
                    "references": t.references,
                    "timestamp": t.timestamp,
                }
                for t in self.turns
            ],
            "long_term_summary": self.long_term_summary,
            "turn_counter": self._turn_counter,
            "created_at": self.created_at,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationManager":
        """从字典恢复 ConversationManager 实例"""
        mgr = cls(session_id=data.get("session_id", ""))
        mgr._turn_counter = data.get("turn_counter", 0)
        mgr.created_at = data.get("created_at", time.time())
        mgr.last_active = data.get("last_active", time.time())
        mgr.long_term_summary = data.get("long_term_summary", "")
        for t_data in data.get("turns", []):
            turn = Turn(
                turn_id=t_data["turn_id"],
                user_query=t_data["user_query"],
                resolved_query=t_data.get("resolved_query", t_data["user_query"]),
                assistant_response=t_data["assistant_response"],
                references=t_data.get("references", []),
                timestamp=t_data.get("timestamp", time.time()),
            )
            mgr.turns.append(turn)
        return mgr

    def is_expired(self, ttl_seconds: int = 1800) -> bool:
        """检查会话是否过期（默认 30 分钟无活动）"""
        return (time.time() - self.last_active) > ttl_seconds

    # ── 辅助方法 ────────────────────────────────────────────

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略 token 估算：中文 1.5 chars/token，英文 1.3 chars/token"""
        if not text:
            return 0
        # 分别统计中英文字符
        chinese_chars = len(re.findall(r'[一-鿿]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.4)

    @staticmethod
    def _build_middle_summary(turns: list[Turn]) -> str:
        """为中间轮次构建简要摘要"""
        if not turns:
            return ""
        topics = [f'"{t.user_query}"' for t in turns]
        return f"此前用户还询问过：{'、'.join(topics)}。"

    @staticmethod
    def _render_turns_for_summary(turns: list[Turn]) -> str:
        """将多轮对话渲染为适合 LLM 摘要的文本"""
        lines = []
        for t in turns:
            answer_short = t.assistant_response[:150]
            lines.append(f"用户：{t.user_query}\n助手：{answer_short}")
        return "\n".join(lines)


class QueryResolver:
    """
    查询消解器：利用 LLM 将省略/指代 query 还原为完整 query

    示例：
      对话历史: [用户:"早餐怎么样？", 助手:"早餐很丰富..."]
      当前 query: "那午餐呢？"
      → resolved_query: "酒店的午餐怎么样？"
    """

    def __init__(self, llm_client):
        """
        参数:
            llm_client: LLMClient 实例（使用 qwen-flash 级别即可）
        """
        self.llm_client = llm_client

    def resolve(self, current_query: str, conversation_manager: ConversationManager) -> str:
        """
        消解当前 query 中的指代和省略。

        如果没有对话历史，直接返回原始 query。
        如果 query 已经是完整独立的，也返回原始 query。
        """
        if not conversation_manager.turns:
            return current_query

        # 快速判断：短 query 或包含指代才需要消解
        if not self._needs_resolution(current_query, conversation_manager):
            return current_query

        # 构建消解 prompt
        history_text = self._render_history(conversation_manager)

        prompt = f"""你是一个对话消解器。用户在和一个酒店客服助手多轮对话。
请将用户当前的输入还原为完整、独立的问题（消解指代和省略）。

【对话历史】
{history_text}

【当前用户输入】
{current_query}

【消解规则】
1. 如果用户输入中有"那"、"它"、"这个"、"刚才的"等指代词，替换为被指代的具体内容
2. 如果用户省略了主语（如直接说"早餐呢？"），从历史中补全缺失的上下文
3. 如果用户输入本身已经是完整的问题，直接返回原句
4. 消解后的问题应该像用户第一次问那样完整自然

只输出消解后的问题，不要有任何解释。
"""
        try:
            resolved = self.llm_client.generate(prompt, temperature=0.1)
            resolved = resolved.strip().strip('"').strip("'")
            if len(resolved) < 2:
                return current_query
            return resolved
        except Exception:
            return current_query

    @staticmethod
    def _needs_resolution(query: str, conv: ConversationManager) -> bool:
        """快速判断是否需要消解"""
        q = query.strip()
        # 长 query（> 20 字）通常已经完整
        if len(q) > 20:
            return False
        # 包含明显指代
        if re.search(r'(那|这|它|他们|她们|刚才|上面|前面|之前)', q):
            return True
        # 省略主语的短问句
        if re.match(r'^.{0,10}(呢|吗|吧|怎么样|如何)', q):
            return True
        # 短 query 且非首轮
        if len(q) <= 15 and conv.turns:
            return True
        return False

    @staticmethod
    def _render_history(conv: ConversationManager) -> str:
        """渲染最近几轮对话历史"""
        lines = []
        for t in conv.turns[-3:]:
            lines.append(f"用户：{t.user_query}")
            lines.append(f"助手：{t.assistant_response[:120]}")
        return "\n".join(lines)


class ContextCompressor:
    """
    对话上下文压缩器：调用 LLM 将多轮对话压缩为简短摘要
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def summarize_turns(self, turns_text: str, existing_summary: str = "") -> str:
        """将几轮对话压缩为 1-2 句摘要"""
        prompt = f"""请将以下酒店客服对话压缩为 1-2 句话的简短摘要。
保留关键信息（用户关心什么话题、得到什么结论），忽略细节。

{("【已有摘要】" + existing_summary) if existing_summary else ""}

【对话内容】
{turns_text}

只输出摘要句子，不要任何前缀或格式。"""
        try:
            summary = self.llm_client.generate(prompt, temperature=0.2)
            return summary.strip().strip('"').strip("'")
        except Exception:
            return ""
