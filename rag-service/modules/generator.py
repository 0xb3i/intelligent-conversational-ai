"""回复生成器：基于检索上下文生成最终回复"""

import re
import time
from datetime import datetime
from dashscope import Generation

from modules.prompts import build_prompt, PROMPT_VERSION


# 引用违规兜底：把 [[ref:1]][[ref:2]] 之类的连续引用合并为 [[ref:1,2]]
_REF_ADJ_RE = re.compile(r"\[\[ref:([0-9,\s]+)\]\]\s*\[\[ref:([0-9,\s]+)\]\]")
_REF_RE = re.compile(r"\[\[ref:([0-9,\s]+)\]\]")
# 模型有时会把"等"塞进 ]] 内部：[[ref:1,3,等]] → 提到 ]] 外面：[[ref:1,3]]等
_REF_ETC_INSIDE_RE = re.compile(r"\[\[ref:([0-9,\s]+?)[,，]?\s*等\s*\]\]")
# 自校验过程不应被输出，发现"自校验"开头的尾段就截断
_SELF_CHECK_TAIL_RE = re.compile(r"\n+#?\s*自校验[\s\S]*$")


def _merge_refs(match: re.Match) -> str:
    """合并连续两个 [[ref:..]] 标记，去重并保序，最多保留前 3 条。
    超出时输出 `[[ref:N,M]]等`（"等"在 ]] 之外），与前端 citation-parser 契约一致。"""
    parts = (match.group(1) + "," + match.group(2)).split(",")
    seen = []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.append(p)
    if len(seen) > 3:
        return f"[[ref:{','.join(seen[:2])}]]等"
    return f"[[ref:{','.join(seen)}]]"


def _truncate_refs(match: re.Match) -> str:
    """单个标记内 > 3 条引用时，截断为 `[[ref:前两条]]等`（"等"在 ]] 之外）。"""
    parts = [p.strip() for p in match.group(1).split(",") if p.strip()]
    if len(parts) > 3:
        return f"[[ref:{','.join(parts[:2])}]]等"
    return f"[[ref:{','.join(parts)}]]"


def sanitize_response(text: str) -> tuple[str, dict]:
    """对生成结果做轻量后处理 + 违规计数。"""
    stats = {"adjacent_refs_merged": 0, "over_length_refs_truncated": 0,
             "etc_inside_ref_fixed": 0,
             "self_check_tail_stripped": False}
    if not text:
        return text, stats

    # 0. 模型把"等"塞进 ]] 内部 → 提到 ]] 之外
    def _fix_etc_inside(m: re.Match) -> str:
        nums = m.group(1).strip().rstrip(",，")
        stats["etc_inside_ref_fixed"] += 1
        return f"[[ref:{nums}]]等"
    new_text = _REF_ETC_INSIDE_RE.sub(_fix_etc_inside, text)

    # 1. 合并连续 [[ref:..]][[ref:..]]
    merged_text, n = _REF_ADJ_RE.subn(_merge_refs, new_text)
    while n:
        stats["adjacent_refs_merged"] += n
        merged_text, n = _REF_ADJ_RE.subn(_merge_refs, merged_text)
    new_text = merged_text

    # 2. 单标记 > 3 条 → 截断
    def _count_and_truncate(m: re.Match) -> str:
        out = _truncate_refs(m)
        if out != m.group(0):
            stats["over_length_refs_truncated"] += 1
        return out

    new_text = _REF_RE.sub(_count_and_truncate, new_text)

    # 3. 自校验段误输出 → 截断
    if _SELF_CHECK_TAIL_RE.search(new_text):
        new_text = _SELF_CHECK_TAIL_RE.sub("", new_text).rstrip()
        stats["self_check_tail_stripped"] = True

    return new_text, stats


class ResponseGenerator:
    """回复生成器：基于检索上下文生成最终回复"""

    def __init__(self, api_key: str, model: str = "qwen-plus",
                 prompt_version: str = "v2"):
        self.api_key = api_key
        self.model = model
        self.prompt_version = prompt_version

    def _build_prompt(self, user_query: str, rewritten_queries=None,
                      ranked_comments=None, summaries=None,
                      need_retrieval: bool = True, today: datetime | None = None,
                      history: dict | None = None,
                      conversation_context: str = "") -> str:
        """构建生成 prompt（统一委托给 prompts.build_prompt）"""
        return build_prompt(
            version=self.prompt_version,
            user_query=user_query,
            rewritten_queries=rewritten_queries,
            ranked_comments=ranked_comments,
            summaries=summaries,
            need_retrieval=need_retrieval,
            today=today,
            history=history,
            conversation_context=conversation_context,
        )

    def _call_kwargs(self, prompt: str, temperature: float = 0.7) -> dict:
        """构建 Generation.call() 的通用参数"""
        return dict(
            api_key=self.api_key,
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            result_format="message",
            stream=True,
            incremental_output=True
        )

    def generate(self, user_query: str, rewritten_queries=None, ranked_comments=None,
                 summaries=None, need_retrieval: bool = True, print_response: bool = True,
                 today: datetime | None = None, history: dict | None = None,
                 conversation_context: str = "") -> tuple[str, float, float, float]:
        """
        生成回复（流式输出）

        返回:
            (response_text, ttft_model, subsequent_time, generation_time)
        """
        start_time = time.time()
        prompt = self._build_prompt(user_query, rewritten_queries, ranked_comments,
                                    summaries, need_retrieval, today, history,
                                    conversation_context)

        completion = Generation.call(**self._call_kwargs(prompt))

        response_content = ""
        ttft_model = 0
        subsequent_time = 0
        first_token_time = 0

        for chunk in completion:
            if chunk.status_code != 200:
                raise RuntimeError(f"回复生成失败: {chunk.message}")

            message = chunk.output.choices[0].message
            if message.content:
                if not ttft_model:
                    ttft_model = time.time() - start_time
                    first_token_time = time.time()
                if print_response:
                    print(message.content, end="", flush=True)
                response_content += message.content

        if print_response and response_content:
            print()

        if ttft_model:
            subsequent_time = time.time() - first_token_time

        # 引用违规兜底
        response_content, sanitize_stats = sanitize_response(response_content)
        if any(sanitize_stats.values()):
            print(f"[prompt={PROMPT_VERSION}] 引用违规兜底: {sanitize_stats}")

        generation_time = time.time() - start_time

        return response_content, ttft_model, subsequent_time, generation_time

    def generate_stream(self, user_query: str, rewritten_queries=None, ranked_comments=None,
                        summaries=None, need_retrieval: bool = True,
                        today: datetime | None = None, history: dict | None = None,
                        conversation_context: str = ""):
        """
        流式生成回复（yield 每个 chunk）

        Yields:
            str: 每个文本 chunk
        """
        prompt = self._build_prompt(user_query, rewritten_queries, ranked_comments,
                                    summaries, need_retrieval, today, history,
                                    conversation_context)

        completion = Generation.call(**self._call_kwargs(prompt))

        # 流式场景：本身在 chunk 间难以做"跨 chunk"的引用合并，
        # 因此采用"小缓冲 + 行级 flush"策略，避免把 [[ref:..]][[ref:..]]
        # 这种紧邻情况漏过。缓冲不超过 64 字符。
        buffer = ""
        BUFFER_FLUSH_LEN = 64

        for chunk in completion:
            if chunk.status_code != 200:
                raise RuntimeError(f"回复生成失败: {chunk.message}")

            message = chunk.output.choices[0].message
            if message.content:
                buffer += message.content
                # 遇到换行或缓冲过长就 flush（保留尾部一点防截断引用标记）
                while "\n" in buffer or len(buffer) >= BUFFER_FLUSH_LEN:
                    if "\n" in buffer:
                        head, buffer = buffer.split("\n", 1)
                        head += "\n"
                    else:
                        head, buffer = buffer[:BUFFER_FLUSH_LEN - 16], buffer[BUFFER_FLUSH_LEN - 16:]
                    cleaned, _ = sanitize_response(head)
                    yield cleaned

        if buffer:
            cleaned, _ = sanitize_response(buffer)
            yield cleaned
