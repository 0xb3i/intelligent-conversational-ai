import os
import json
import time
import queue
import threading
from flask import Flask, request, Response, send_from_directory
from flask_cors import CORS

from pipeline import HotelReviewRAG

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

rag_instance = None


def get_rag():
    global rag_instance
    if rag_instance is None:
        rag_instance = HotelReviewRAG(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            dashvector_api_key=os.environ["DASHVECTOR_API_KEY"],
            dashvector_endpoint=os.environ["DASHVECTOR_HOTEL_ENDPOINT"],
            data_dir="data",
        )
    return rag_instance


def run_query_stream(query, enable_hyde, q):
    rag = get_rag()
    try:
        q.put({"stage": "query_process", "status": "running", "detail": "正在分析您的问题..."})

        t0 = time.time()
        query_info = rag._process_query(query, enable_hyde=enable_hyde)
        t_process = time.time() - t0

        q.put({
            "stage": "query_process", "status": "done", "duration": round(t_process, 2),
            "intent": query_info.intent,
            "sub_queries": query_info.sub_queries,
            "hyde": query_info.hyde_answer[:100] + "..." if query_info.hyde_answer else None,
        })

        q.put({"stage": "recall", "status": "running", "detail": "正在从多个渠道检索相关评论..."})

        t0 = time.time()
        all_recalls = []
        all_summaries = []
        for qi, sq in enumerate(query_info.sub_queries):
            sub_q = sq["query"]
            all_recalls.append(rag._recall_text(sub_q, qi, rag.topk_recall))
            all_recalls.append(rag._recall_vector(sub_q, qi, rag.topk_recall, query_info.intent))
            all_recalls.append(rag._recall_reverse(sub_q, qi, rag.topk_recall, query_info.intent))
            all_summaries.extend(rag._recall_summary(sub_q, qi))
        if enable_hyde and query_info.hyde_answer:
            all_recalls.append(rag._recall_hyde(query_info.hyde_answer, rag.topk_recall, query_info.intent))
        t_retrieve = time.time() - t0

        seen = set()
        unique_summaries = []
        for s in all_summaries:
            if s.category not in seen:
                seen.add(s.category)
                unique_summaries.append(s)

        q.put({
            "stage": "recall", "status": "done", "duration": round(t_retrieve, 2),
            "summary_categories": [s.category for s in unique_summaries],
        })

        q.put({"stage": "fusion", "status": "running", "detail": "正在融合多路检索结果..."})
        t0 = time.time()
        fused = rag._rrf_fuse(all_recalls)
        q.put({"stage": "fusion", "status": "done", "duration": round(time.time() - t0, 2), "fused_count": len(fused)})

        q.put({"stage": "rerank", "status": "running", "detail": "正在用 Rerank 模型精排..."})
        t0 = time.time()
        rerank_candidates = fused[: rag.topk_rerank * 2]
        reranked = rag._rerank(query, rerank_candidates, rag.topk_rerank)
        q.put({"stage": "rerank", "status": "done", "duration": round(time.time() - t0, 2), "reranked_count": len(reranked)})

        q.put({"stage": "sort", "status": "running", "detail": "正在综合排序..."})
        t0 = time.time()
        time_sensitive = query_info.intent.get("time_sensitivity") is not None
        ranked = rag._composite_score(reranked, time_sensitive=time_sensitive)
        top_comments = ranked[: rag.topk_final]
        q.put({"stage": "sort", "status": "done", "duration": round(time.time() - t0, 2)})

        q.put({"stage": "generate", "status": "running", "detail": "正在生成回答..."})
        t0 = time.time()
        answer = rag._generate_answer(query, top_comments, unique_summaries)
        q.put({"stage": "generate", "status": "done", "duration": round(time.time() - t0, 2)})

        comments_data = []
        for c in top_comments:
            comments_data.append({
                "rank": getattr(c, "_final_rank", 0),
                "score": round(getattr(c, "_final_score", 0), 4),
                "rerank_score": round(getattr(c, "_rerank_score", 0), 4),
                "doc_id": c.doc_id,
                "room_type": c.room_type,
                "score_val": c.score_val,
                "publish_date": c.publish_date,
                "quality_score": c.quality_score,
                "comment": c.comment[:300] + ("..." if len(c.comment) > 300 else ""),
                "routes": getattr(c, "_routes", []),
            })

        q.put({
            "stage": "result", "status": "done",
            "answer": answer,
            "summaries": [{"category": s.category, "summary": s.summary[:200] + "...", "comment_count": s.comment_count} for s in unique_summaries],
            "top_comments": comments_data,
        })

    except Exception as e:
        q.put({"stage": "error", "status": "error", "message": str(e)})
    finally:
        q.put(None)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/query", methods=["POST"])
def query_api():
    data = request.json
    user_query = data.get("query", "")
    enable_hyde = data.get("hyde", False)
    if not user_query.strip():
        return {"error": "查询不能为空"}, 400

    q = queue.Queue()
    t = threading.Thread(target=run_query_stream, args=(user_query, enable_hyde, q), daemon=True)
    t.start()

    def generate():
        while True:
            msg = q.get()
            if msg is None:
                break
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
