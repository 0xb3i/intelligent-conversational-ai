"""FastAPI 入口 — 酒店评论 RAG 问答服务（支持多轮对话上下文管理）"""

import os
import sys
import json
import uuid
import asyncio
import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# 确保模块路径
sys.path.insert(0, str(Path(__file__).parent))

# 加载环境变量
load_dotenv()
load_dotenv(Path(__file__).parent.parent / ".env")


class NumpyEncoder(json.JSONEncoder):
    """处理 numpy 类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def json_dumps(obj) -> str:
    """带 numpy 支持的 JSON 序列化"""
    return json.dumps(obj, ensure_ascii=False, cls=NumpyEncoder)


app = FastAPI(title="Hotel Review RAG API", version="2.0.0")

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 全局单例 ────────────────────────────────────────────────────────

rag_system = None
query_resolver = None          # QueryResolver 实例（需 LLM client）
context_compressor = None      # ContextCompressor 实例
sessions: dict[str, "ConversationManager"] = {}  # session_id → 会话管理器
SESSION_TTL = 1800  # 30 分钟无活动过期


class ChatRequest(BaseModel):
    query: str
    options: dict = {}
    history: dict | None = None
    session_id: str | None = None  # 多轮对话会话 ID


# ── 启动 ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """启动时初始化 RAG 系统与对话管理组件"""
    global rag_system, query_resolver, context_compressor

    api_key = os.getenv("DASHSCOPE_API_KEY")
    intl_api_key = os.getenv("DASHSCOPE_INTL_API_KEY")
    dashvector_api_key = os.getenv("DASHVECTOR_API_KEY")
    dashvector_endpoint = os.getenv("DASHVECTOR_HOTEL_ENDPOINT")

    if not all([api_key, dashvector_api_key, dashvector_endpoint]):
        print("WARNING: 缺少 API Key 环境变量，RAG 系统未初始化")
        return

    try:
        from modules.rag_system import HotelReviewRAG
        from modules.clients import DASHSCOPE_INTL_API_BASE, LLMClient
        from modules.conversation import (
            ConversationManager, QueryResolver, ContextCompressor
        )
        data_dir = Path(__file__).parent / "data"

        # 国际模式
        if intl_api_key:
            import dashscope
            dashscope.base_http_api_url = DASHSCOPE_INTL_API_BASE

        mode = "新加坡" if intl_api_key else "北京"
        print(f"正在初始化 RAG 系统（{mode}）...")
        rag_system = HotelReviewRAG(
            api_key=api_key,
            dashvector_api_key=dashvector_api_key,
            dashvector_endpoint=dashvector_endpoint,
            data_dir=data_dir,
            intl_api_key=intl_api_key or None
        )
        print(f"RAG 系统初始化完成（{mode}）")

        # 初始化对话管理组件（复用 qwen-flash 做轻量查询消解和摘要）
        key = intl_api_key if intl_api_key else api_key
        resolver_llm = LLMClient(key, model="qwen-flash", json=False)
        query_resolver = QueryResolver(resolver_llm)
        context_compressor = ContextCompressor(resolver_llm)
        print("对话管理组件初始化完成")

    except Exception as e:
        print(f"系统初始化失败: {e}")


# ── 健康检查 ────────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "version": "2.0.0",
        "rag_ready": rag_system is not None,
        "conversation_ready": query_resolver is not None and context_compressor is not None,
        "active_sessions": len(sessions),
    }


# ── Session 管理 ────────────────────────────────────────────────────

def _get_or_create_session(session_id: str | None) -> "ConversationManager":
    """获取或创建会话管理器"""
    from modules.conversation import ConversationManager

    # 清理过期会话
    expired = [sid for sid, mgr in sessions.items() if mgr.is_expired(SESSION_TTL)]
    for sid in expired:
        del sessions[sid]

    if session_id and session_id in sessions:
        return sessions[session_id]

    # 创建新会话
    new_id = session_id or str(uuid.uuid4())
    conv = ConversationManager(session_id=new_id)
    sessions[new_id] = conv
    return conv


# ── RAG 问答 ────────────────────────────────────────────────────────

@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    """RAG 问答接口（支持多轮对话上下文管理）"""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")

    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG 系统未就绪")

    # ── 多轮对话管理 ──────────────────────────────────────────
    conv = _get_or_create_session(request.session_id)
    raw_query = request.query.strip()

    # 查询消解：将指代/省略还原为完整 query
    resolved_query = raw_query
    if query_resolver is not None and conv.turns:
        resolved_query = query_resolver.resolve(raw_query, conv)

    # 构建用于检索的上下文，用于 intent expansion
    retrieval_ctx = conv.get_retrieval_context() if conv.turns else None

    # 构建用于生成的多轮历史文本
    conv_context = conv.get_generation_context() if conv.turns else ""
    # ────────────────────────────────────────────────────────────

    enable_generation = request.options.get("enable_generation", True)
    query_options = {k: v for k, v in request.options.items() if k != "enable_generation"}
    query_options.setdefault("enable_hyde", False)

    # 非流式：只返回检索结果
    if not enable_generation:
        try:
            result = rag_system.query(
                resolved_query,
                enable_generation=False,
                print_response=False,
                history=request.history,
                conversation_context=conv_context,
                retrieval_context=retrieval_ctx,
                **query_options
            )

            comments = _format_comments(result['references']['comments'])
            summaries = [
                {"category": s['metadata'].get('category', ''), "content": s['summary']}
                for s in result['references']['summaries']
            ]

            # 记录对话轮次（非流式路径）
            ref_ids = [c.get("_id", "") for c in (comments or [])]
            conv.add_turn(
                user_query=raw_query,
                resolved_query=resolved_query,
                assistant_response="",
                references=ref_ids,
            )

            return JSONResponse(content=_sanitize({
                "references": {"comments": comments, "summaries": summaries},
                "timing": result['timing'],
                "session_id": conv.session_id,
                "resolved_query": resolved_query if resolved_query != raw_query else None,
            }))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 流式：SSE 响应
    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    query_text = resolved_query
    query_history = request.history
    query_conv = conv
    query_raw = raw_query
    query_resolver_ref = query_resolver
    query_compressor_ref = context_compressor

    def _run_query_stream():
        """在线程池中运行同步 RAG 流水线"""
        try:
            for event in rag_system.query_stream(
                query_text, history=query_history,
                conversation_context=conv_context,
                retrieval_context=retrieval_ctx,
                **query_options
            ):
                event_type = event.get("type")

                if event_type == "intent":
                    queue.put_nowait(f"data: {json_dumps(event)}\n\n")
                elif event_type == "references":
                    data = event["data"]
                    data["comments"] = _format_comments(data["comments"])
                    queue.put_nowait(f"data: {json_dumps(_sanitize(event))}\n\n")
                elif event_type == "chunk":
                    queue.put_nowait(f"data: {json_dumps(event)}\n\n")
                elif event_type == "done":
                    data = _sanitize(event)
                    data.setdefault("data", {})
                    data["data"]["session_id"] = query_conv.session_id
                    data["data"]["resolved_query"] = (
                        query_text if query_text != query_raw else None
                    )
                    queue.put_nowait(f"data: {json_dumps(data)}\n\n")

        except Exception as e:
            error_event = {"type": "error", "message": str(e)}
            queue.put_nowait(f"data: {json_dumps(error_event)}\n\n")
        finally:
            queue.put_nowait(_SENTINEL)

    async def generate_sse():
        """异步生成器：从 queue 取出 SSE 事件并 yield"""
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_query_stream)

        full_response = ""
        references_data = None

        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break

            # 收集回复内容用于对话记录
            line = item if isinstance(item, str) else str(item)
            if line.startswith("data: "):
                try:
                    evt = json.loads(line[6:])
                    if evt.get("type") == "chunk" and evt.get("content"):
                        full_response += evt["content"]
                    elif evt.get("type") == "references":
                        references_data = evt.get("data", {}).get("comments", [])
                except Exception:
                    pass

            yield item

        # 流结束后记录对话轮次
        _record_turn(query_raw, query_text, full_response, references_data,
                     query_conv, query_compressor_ref)

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


def _record_turn(raw_query, resolved_query, full_response, references_data,
                 conv, compressor):
    """记录一轮对话，必要时触发摘要压缩"""
    if not full_response:
        return

    ref_ids = [r.get("_id", "") for r in (references_data or [])]
    conv.add_turn(
        user_query=raw_query,
        resolved_query=resolved_query,
        assistant_response=full_response,
        references=ref_ids,
    )

    # 检查是否需要摘要压缩
    triggered, summary_text = conv.maybe_summarize()
    if triggered and compressor is not None:
        compressed = compressor.summarize_turns(
            summary_text, conv.long_term_summary
        )
        conv.apply_summary(compressed)


# ── Session 管理 API ────────────────────────────────────────────────

@app.post("/api/v1/session/reset")
async def reset_session(request: ChatRequest):
    """重置指定会话（开始全新对话）"""
    sid = request.session_id
    if sid and sid in sessions:
        del sessions[sid]
    return {"status": "ok", "message": "会话已重置"}


# ── 辅助函数 ────────────────────────────────────────────────────────

def _sanitize(obj):
    """递归将 numpy 类型转换为 Python 原生类型"""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _format_comments(raw_comments: list) -> list:
    """将 RAG 返回的评论格式转换为前端 Comment 类型"""
    formatted = []
    for i, c in enumerate(raw_comments):
        meta = c.get('metadata', {})
        comment_id = c.get('comment_id', '')

        full_data = {}
        if rag_system and comment_id:
            try:
                row = rag_system.df_comments.loc[comment_id]
                full_data = row.to_dict() if hasattr(row, 'to_dict') else {}
            except (KeyError, Exception):
                pass

        formatted.append(_sanitize({
            "_id": str(comment_id),
            "comment": str(c.get('comment', '')),
            "primary_chunk": c.get('primary_chunk'),
            "score": meta.get('score', 0),
            "star": full_data.get('star', int(meta.get('score', 0))),
            "useful_count": meta.get('useful_count', 0),
            "publish_date": str(meta.get('publish_date', '')),
            "room_type": str(meta.get('room_type', '')),
            "fuzzy_room_type": str(meta.get('fuzzy_room_type', '')),
            "travel_type": str(full_data.get('travel_type', '')),
            "review_count": meta.get('review_count', 0),
            "quality_score": meta.get('quality_score', 0),
            "category1": full_data.get('category1', None),
            "category2": full_data.get('category2', None),
            "category3": full_data.get('category3', None),
            "images": full_data.get('images', []),
            "relevance_score": c.get('final_score', c.get('rrf_score', 0)),
            "rank": c.get('final_rank', c.get('rrf_rank', i + 1))
        }))
    return formatted


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
