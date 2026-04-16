"""
酒店评论智能问答 RAG Pipeline
============================
完整流程：查询处理 → 多路召回 → RRF 融合 → Rerank 排序 → 综合排序 → LLM 生成

依赖安装：pip install dashscope dashvector chromadb jieba nltk pandas
"""

import os
import re
import json
import math
import time
import pickle
import nltk
import jieba
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from dashscope import TextEmbedding, TextReRank, Generation
import dashvector
from dashvector import Doc
import chromadb


try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class RecallResult:
    doc_id: str
    score: float
    source: str
    query_idx: int = 0
    comment: str = ""
    room_type: str = ""
    fuzzy_room_type: str = ""
    score_val: float = 0.0
    publish_date: str = ""
    quality_score: int = 0
    comment_len: int = 0
    useful_count: int = 0
    review_count: int = 0
    images: str = ""


@dataclass
class SummaryRecallResult:
    category: str
    keywords: str
    summary: str
    comment_count: int
    query_idx: int = 0


@dataclass
class QueryInfo:
    original: str = ""
    intent: dict = field(default_factory=dict)
    sub_queries: list = field(default_factory=list)
    hyde_answer: str = ""


@dataclass
class RAGResult:
    answer: str = ""
    query_info: QueryInfo = field(default_factory=QueryInfo)
    summaries: list = field(default_factory=list)
    top_comments: list = field(default_factory=list)
    latency: dict = field(default_factory=dict)


# ============================================================
# 基础组件
# ============================================================

class EmbeddingClient:
    def __init__(self, api_key: str, model: str = "text-embedding-v4", dimension: int = 1024):
        self.api_key = api_key
        self.model = model
        self.dimension = dimension

    def embed(self, text: str, text_type: str = "document", instruct: str = "") -> list[float]:
        return self.embed_batch([text], text_type=text_type, instruct=instruct)[0]

    def embed_batch(
        self,
        texts: list[str],
        text_type: str = "document",
        instruct: str = "",
    ) -> list[list[float]]:
        kwargs = dict(
            api_key=self.api_key,
            model=self.model,
            input=texts,
            dimension=self.dimension,
        )
        if text_type == "query":
            kwargs["text_type"] = "query"
            if instruct:
                kwargs["instruct"] = instruct

        response = TextEmbedding.call(**kwargs)
        if response.status_code == 200:
            return [item["embedding"] for item in response.output["embeddings"]]
        raise RuntimeError(f"Embedding 调用失败: {response.message}")


class InvertedIndex:
    def __init__(self, k1: float = 1.5, b: float = 0.75, stopwords_file: str = None):
        self.k1 = k1
        self.b = b
        self.index = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0.0
        self.num_docs = 0
        self.documents = {}

        self.stopwords = set()
        if stopwords_file and Path(stopwords_file).exists():
            with open(stopwords_file, encoding="utf-8") as f:
                self.stopwords.update(line.strip() for line in f if line.strip())
            try:
                self.stopwords.update(nltk.corpus.stopwords.words("english"))
            except Exception:
                pass
        jieba.initialize()

    def tokenize(self, text: str) -> list[str]:
        text = re.sub(r"\s+", "", text)
        tokens = jieba.lcut(text)
        pattern = re.compile(r"[^\u4e00-\u9fffa-zA-Z]")
        return [
            t.lower()
            for t in tokens
            if t.lower() not in self.stopwords and not pattern.search(t)
        ]

    def build(self, documents: dict[str, str]):
        self.documents = documents
        self.num_docs = len(documents)
        total_length = 0
        for doc_id, text in documents.items():
            tokens = self.tokenize(text)
            length = len(tokens)
            self.doc_lengths[doc_id] = length
            total_length += length
            for term, freq in Counter(tokens).items():
                self.index.setdefault(term, {})[doc_id] = freq
        self.avg_doc_length = total_length / self.num_docs if self.num_docs else 0

    def search(self, query: str, topk: int = 10) -> list[tuple[str, float]]:
        tokens = self.tokenize(query)
        if not tokens:
            return []
        idf = {}
        for t in tokens:
            if t in self.index:
                df = len(self.index[t])
                idf[t] = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
        scores: dict[str, float] = {}
        for t in tokens:
            if t not in self.index:
                continue
            for doc_id, tf in self.index[t].items():
                dl = self.doc_lengths[doc_id]
                norm = 1 - self.b + self.b * (dl / self.avg_doc_length)
                s = idf[t] * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
                scores[doc_id] = scores.get(doc_id, 0) + s
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "index": self.index,
                    "doc_lengths": self.doc_lengths,
                    "avg_doc_length": self.avg_doc_length,
                    "num_docs": self.num_docs,
                    "documents": self.documents,
                    "k1": self.k1,
                    "b": self.b,
                    "stopwords": self.stopwords,
                },
                f,
            )

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.index = data["index"]
        self.doc_lengths = data["doc_lengths"]
        self.avg_doc_length = data["avg_doc_length"]
        self.num_docs = data["num_docs"]
        self.documents = data["documents"]
        self.k1 = data["k1"]
        self.b = data["b"]
        self.stopwords = data.get("stopwords", set())


# ============================================================
# 主 Pipeline
# ============================================================

class HotelReviewRAG:

    INSTRUCT = "Given a hotel review query, retrieve relevant hotel reviews that answer the query."

    def __init__(
        self,
        api_key: str,
        dashvector_api_key: str,
        dashvector_endpoint: str,
        data_dir: str = "data",
        topk_recall: int = 150,
        topk_rerank: int = 40,
        topk_final: int = 10,
    ):
        self.api_key = api_key
        self.data_dir = Path(data_dir)
        self.topk_recall = topk_recall
        self.topk_rerank = topk_rerank
        self.topk_final = topk_final

        self.embedder = EmbeddingClient(api_key=api_key)

        self._init_vector_db(dashvector_api_key, dashvector_endpoint)
        self._init_inverted_index()
        self._load_metadata()

    # ----------------------------------------------------------
    # 初始化
    # ----------------------------------------------------------

    def _init_vector_db(self, dv_api_key: str, dv_endpoint: str):
        dv_client = dashvector.Client(api_key=dv_api_key, endpoint=dv_endpoint)
        self.comment_collection = dv_client.get("comment_database")
        self.query_collection = dv_client.get("reverse_query_database")
        if self.comment_collection is None:
            raise RuntimeError("DashVector comment_database 不存在，请先运行知识库构建")
        if self.query_collection is None:
            raise RuntimeError("DashVector reverse_query_database 不存在，请先运行知识库构建")

        chroma_path = str(self.data_dir / "chroma_db")
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.summary_collection = chroma_client.get_collection("summary_database")
        print("[初始化] 向量数据库连接成功")

    def _init_inverted_index(self):
        index_path = self.data_dir / "inverted_index.pkl"
        self.inverted_index = InvertedIndex()
        if index_path.exists():
            self.inverted_index.load(str(index_path))
        else:
            print("[初始化] 倒排索引文件不存在，将从评论数据构建")
            csv_path = self.data_dir / "filtered_comments.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, index_col=0)
                docs = {idx: row["comment"] for idx, row in df.iterrows()}
                self.inverted_index.build(docs)
                self.inverted_index.save(str(index_path))
            else:
                raise RuntimeError("缺少评论数据，无法构建倒排索引")

    def _load_metadata(self):
        csv_path = self.data_dir / "filtered_comments.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0)
            self.metadata = {}
            for idx, row in df.iterrows():
                self.metadata[str(idx)] = dict(
                    comment=row.get("comment", ""),
                    room_type=row.get("room_type", ""),
                    fuzzy_room_type=row.get("fuzzy_room_type", ""),
                    score=float(row.get("score", 0)),
                    publish_date=str(row.get("publish_date", "")),
                    quality_score=int(row.get("quality_score", 0)),
                    comment_len=int(row.get("comment_len", 0)),
                    useful_count=int(row.get("useful_count", 0)),
                    review_count=int(row.get("review_count", 0)),
                    images=str(row.get("images", "")),
                )
        else:
            self.metadata = {}

        summaries_path = self.data_dir / "category_summaries.json"
        if summaries_path.exists():
            with open(summaries_path, "r", encoding="utf-8") as f:
                self.summaries_data = json.load(f)
        else:
            self.summaries_data = []

        print(f"[初始化] 元数据加载完成: {len(self.metadata)} 条评论, {len(self.summaries_data)} 个摘要类别")

    # ----------------------------------------------------------
    # 查询处理
    # ----------------------------------------------------------

    def _detect_intent(self, query: str) -> dict:
        prompt = f"""请分析以下酒店评论查询的意图，提取关键信息，以 JSON 格式返回：
- room_type: 用户提到的具体房型（如"花园大床房"、"红棉大床套房"），未提及则为 null
- fuzzy_room_type: 用户提到的模糊房型分类（"大床房"/"双床房"/"套房"），未提及则为 null
- time_sensitivity: 用户是否关注时效性（近期评论更重要），是为 true，否为 null

查询：{query}

请仅返回 JSON，不要其他内容。"""

        resp = Generation.call(
            api_key=self.api_key,
            model="qwen-plus",
            prompt=prompt,
            result_format="message",
            max_tokens=256,
        )
        try:
            content = resp.output.choices[0].message.content
            content = re.sub(r"```json\s*|```\s*", "", content).strip()
            return json.loads(content)
        except Exception:
            return {"room_type": None, "fuzzy_room_type": None, "time_sensitivity": None}

    def _expand_query(self, query: str, intent: dict) -> list[dict]:
        prompt = f"""请将以下酒店评论查询拆解为 2-3 个更具体的子查询，每个子查询带一个权重（0-1，总和为1）。
以 JSON 数组格式返回：[{{"query": "子查询文本", "weight": 权重}}]

原始查询：{query}
意图信息：{json.dumps(intent, ensure_ascii=False)}

要求：
1. 子查询应覆盖原始查询的不同方面
2. 权重反映各子查询的重要性
3. 仅返回 JSON 数组，不要其他内容"""

        resp = Generation.call(
            api_key=self.api_key,
            model="qwen-plus",
            prompt=prompt,
            result_format="message",
            max_tokens=512,
        )
        try:
            content = resp.output.choices[0].message.content
            content = re.sub(r"```json\s*|```\s*", "", content).strip()
            return json.loads(content)
        except Exception:
            return [{"query": query, "weight": 1.0}]

    def _generate_hyde(self, query: str) -> str:
        prompt = f"""请根据以下问题，写一段详细的酒店评论风格的回答（假设你是一位住客，基于真实体验回答）。
问题：{query}

请直接写评论内容，不要加前缀说明。"""

        resp = Generation.call(
            api_key=self.api_key,
            model="qwen-plus",
            prompt=prompt,
            result_format="message",
            max_tokens=512,
        )
        try:
            return resp.output.choices[0].message.content.strip()
        except Exception:
            return ""

    def _process_query(self, query: str, enable_hyde: bool = False) -> QueryInfo:
        info = QueryInfo(original=query)

        t0 = time.time()
        intent = self._detect_intent(query)
        t_intent_detect = time.time() - t0

        t0 = time.time()
        sub_queries = self._expand_query(query, intent)
        t_expand = time.time() - t0

        hyde_answer = ""
        t_hyde = 0.0
        if enable_hyde:
            t0 = time.time()
            hyde_answer = self._generate_hyde(query)
            t_hyde = time.time() - t0

        info.intent = intent
        info.sub_queries = sub_queries
        info.hyde_answer = hyde_answer
        info._timings = {
            "intent_detect": t_intent_detect,
            "expand": t_expand,
            "hyde": t_hyde,
        }
        return info

    # ----------------------------------------------------------
    # 多路召回
    # ----------------------------------------------------------

    def _recall_text(self, sub_query: str, query_idx: int, topk: int) -> list[RecallResult]:
        results = self.inverted_index.search(sub_query, topk=topk)
        recall_list = []
        for doc_id, score in results:
            meta = self.metadata.get(str(doc_id), {})
            recall_list.append(
                RecallResult(
                    doc_id=str(doc_id),
                    score=score,
                    source="text",
                    query_idx=query_idx,
                    **meta,
                )
            )
        return recall_list

    def _recall_vector(
        self,
        sub_query: str,
        query_idx: int,
        topk: int,
        intent: dict,
    ) -> list[RecallResult]:
        q_emb = self.embedder.embed(
            sub_query, text_type="query", instruct=self.INSTRUCT
        )
        kwargs = dict(vector=q_emb, topk=topk)
        filter_parts = []
        if intent.get("fuzzy_room_type"):
            filter_parts.append(f"fuzzy_room_type = '{intent['fuzzy_room_type']}'")
        if intent.get("room_type"):
            filter_parts.append(f"room_type = '{intent['room_type']}'")
        if filter_parts:
            kwargs["filter"] = " and ".join(filter_parts)

        results = self.comment_collection.query(**kwargs)
        recall_list = []
        for doc in results:
            meta = self.metadata.get(doc.id, {})
            recall_list.append(
                RecallResult(
                    doc_id=doc.id,
                    score=doc.score,
                    source="vector",
                    query_idx=query_idx,
                    comment=doc.fields.get("comment", meta.get("comment", "")),
                    room_type=doc.fields.get("room_type", meta.get("room_type", "")),
                    fuzzy_room_type=doc.fields.get("fuzzy_room_type", meta.get("fuzzy_room_type", "")),
                    score_val=meta.get("score", 0),
                    publish_date=meta.get("publish_date", ""),
                    quality_score=meta.get("quality_score", 0),
                    comment_len=meta.get("comment_len", 0),
                    useful_count=meta.get("useful_count", 0),
                    review_count=meta.get("review_count", 0),
                )
            )
        return recall_list

    def _recall_reverse(
        self,
        sub_query: str,
        query_idx: int,
        topk: int,
        intent: dict,
    ) -> list[RecallResult]:
        q_emb = self.embedder.embed(
            sub_query, text_type="query", instruct=self.INSTRUCT
        )
        kwargs = dict(vector=q_emb, topk=topk)
        filter_parts = []
        if intent.get("fuzzy_room_type"):
            filter_parts.append(f"fuzzy_room_type = '{intent['fuzzy_room_type']}'")
        if intent.get("room_type"):
            filter_parts.append(f"room_type = '{intent['room_type']}'")
        if filter_parts:
            kwargs["filter"] = " and ".join(filter_parts)

        results = self.query_collection.query(**kwargs)
        recall_list = []
        for doc in results:
            comment_id = doc.fields.get("comment_id", "")
            meta = self.metadata.get(comment_id, {})
            recall_list.append(
                RecallResult(
                    doc_id=comment_id,
                    score=doc.score,
                    source="reverse",
                    query_idx=query_idx,
                    comment=doc.fields.get("comment", meta.get("comment", "")),
                    room_type=doc.fields.get("room_type", meta.get("room_type", "")),
                    fuzzy_room_type=doc.fields.get("fuzzy_room_type", meta.get("fuzzy_room_type", "")),
                    score_val=meta.get("score", 0),
                    publish_date=meta.get("publish_date", ""),
                    quality_score=meta.get("quality_score", 0),
                    comment_len=meta.get("comment_len", 0),
                    useful_count=meta.get("useful_count", 0),
                    review_count=meta.get("review_count", 0),
                )
            )
        return recall_list

    def _recall_hyde(
        self,
        hyde_answer: str,
        topk: int,
        intent: dict,
    ) -> list[RecallResult]:
        if not hyde_answer:
            return []
        q_emb = self.embedder.embed(hyde_answer, text_type="document")
        kwargs = dict(vector=q_emb, topk=topk)
        filter_parts = []
        if intent.get("fuzzy_room_type"):
            filter_parts.append(f"fuzzy_room_type = '{intent['fuzzy_room_type']}'")
        if intent.get("room_type"):
            filter_parts.append(f"room_type = '{intent['room_type']}'")
        if filter_parts:
            kwargs["filter"] = " and ".join(filter_parts)

        results = self.comment_collection.query(**kwargs)
        recall_list = []
        for doc in results:
            meta = self.metadata.get(doc.id, {})
            recall_list.append(
                RecallResult(
                    doc_id=doc.id,
                    score=doc.score,
                    source="hyde",
                    query_idx=0,
                    comment=doc.fields.get("comment", meta.get("comment", "")),
                    room_type=doc.fields.get("room_type", meta.get("room_type", "")),
                    fuzzy_room_type=doc.fields.get("fuzzy_room_type", meta.get("fuzzy_room_type", "")),
                    score_val=meta.get("score", 0),
                    publish_date=meta.get("publish_date", ""),
                    quality_score=meta.get("quality_score", 0),
                    comment_len=meta.get("comment_len", 0),
                    useful_count=meta.get("useful_count", 0),
                    review_count=meta.get("review_count", 0),
                )
            )
        return recall_list

    def _recall_summary(self, sub_query: str, query_idx: int) -> list[SummaryRecallResult]:
        q_emb = self.embedder.embed(
            sub_query, text_type="query", instruct=self.INSTRUCT
        )
        results = self.summary_collection.query(query_embeddings=[q_emb], n_results=3)
        recall_list = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            recall_list.append(
                SummaryRecallResult(
                    category=meta.get("category", ""),
                    keywords=meta.get("keywords", ""),
                    summary=results["documents"][0][i],
                    comment_count=meta.get("comment_count", 0),
                    query_idx=query_idx,
                )
            )
        return recall_list

    # ----------------------------------------------------------
    # RRF 融合
    # ----------------------------------------------------------

    def _rrf_fuse(self, recall_lists: list[list[RecallResult]], k: int = 60) -> list[RecallResult]:
        doc_scores: dict[str, float] = {}
        doc_info: dict[str, RecallResult] = {}
        doc_routes: dict[str, list[str]] = {}

        for recall_list in recall_lists:
            sorted_list = sorted(recall_list, key=lambda x: x.score, reverse=True)
            for rank, item in enumerate(sorted_list):
                if item.doc_id not in doc_scores:
                    doc_scores[item.doc_id] = 0.0
                    doc_info[item.doc_id] = item
                    doc_routes[item.doc_id] = []
                doc_scores[item.doc_id] += 1.0 / (k + rank + 1)
                route = f"{item.source}: 第{rank+1}名(Q{item.query_idx})"
                if route not in doc_routes[item.doc_id]:
                    doc_routes[item.doc_id].append(route)

        fused = []
        for doc_id, score in doc_scores.items():
            item = doc_info[doc_id]
            item.score = score
            item._routes = doc_routes.get(doc_id, [])
            fused.append(item)

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused

    # ----------------------------------------------------------
    # Rerank + 综合排序
    # ----------------------------------------------------------

    def _rerank(self, query: str, candidates: list[RecallResult], topk: int) -> list[RecallResult]:
        if not candidates:
            return []
        documents = [c.comment for c in candidates]
        response = TextReRank.call(
            api_key=self.api_key,
            model="qwen3-rerank",
            query=query,
            documents=documents,
            top_n=min(topk, len(documents)),
            return_documents=False,
        )
        reranked = []
        for item in response.output.results:
            c = candidates[item.index]
            c._rerank_score = item.relevance_score
            c._rerank_rank = len(reranked) + 1
            reranked.append(c)
        return reranked

    def _composite_score(self, candidates: list[RecallResult], time_sensitive: bool = False) -> list[RecallResult]:
        W_RELEVANCE = 0.40
        W_QUALITY = 0.25
        W_LENGTH = 0.05
        W_COMMENT = 0.05
        W_LIKES = 0.05
        W_TIMELINESS = 0.20

        if not candidates:
            return candidates

        max_len = max(c.comment_len for c in candidates) or 1
        max_reviews = max(c.review_count for c in candidates) or 1
        max_likes = max(c.useful_count for c in candidates) or 1

        ref_date = "2025-04-16"
        for c in candidates:
            relevance = getattr(c, "_rerank_score", 0)
            quality = c.quality_score / 10.0
            length = c.comment_len / max_len
            comment_n = c.review_count / max_reviews
            likes = c.useful_count / max_likes

            timeliness = 0.5
            if c.publish_date and time_sensitive:
                try:
                    from datetime import datetime
                    pub = datetime.strptime(c.publish_date, "%Y-%m-%d")
                    ref = datetime.strptime(ref_date, "%Y-%m-%d")
                    days_diff = (ref - pub).days
                    timeliness = max(0, 1 - days_diff / 730)
                except Exception:
                    timeliness = 0.5

            c._final_score = (
                W_RELEVANCE * relevance
                + W_QUALITY * quality
                + W_LENGTH * length
                + W_COMMENT * comment_n
                + W_LIKES * likes
                + W_TIMELINESS * timeliness
            )

        candidates.sort(key=lambda x: x._final_score, reverse=True)
        for i, c in enumerate(candidates):
            c._final_rank = i + 1
        return candidates

    # ----------------------------------------------------------
    # LLM 生成
    # ----------------------------------------------------------

    def _generate_answer(
        self,
        query: str,
        top_comments: list[RecallResult],
        summaries: list[SummaryRecallResult],
    ) -> str:
        context_parts = []
        if summaries:
            context_parts.append("=== 相关摘要 ===")
            for s in summaries:
                context_parts.append(f"【{s.category}】({s.comment_count}条评论)\n{s.summary}")
            context_parts.append("")

        if top_comments:
            context_parts.append("=== 相关评论 ===")
            for i, c in enumerate(top_comments, 1):
                context_parts.append(
                    f"评论{i}（房型: {c.room_type}，评分: {c.score_val}，"
                    f"发布: {c.publish_date}）:\n{c.comment}"
                )

        context = "\n\n".join(context_parts)

        prompt = f"""你是一位专业的酒店评论分析助手。请根据以下酒店评论信息，回答用户的问题。

要求：
1. 基于提供的评论信息回答，不要编造内容
2. 如果评论中有不同观点，请客观呈现各方意见
3. 引用具体评论时标注来源编号
4. 给出实用建议

用户问题：{query}

{context}

请用中文详细回答："""

        response = Generation.call(
            api_key=self.api_key,
            model="qwen-plus",
            prompt=prompt,
            result_format="message",
            max_tokens=2048,
        )
        try:
            return response.output.choices[0].message.content.strip()
        except Exception:
            return "抱歉，生成回答时出现错误。"

    # ----------------------------------------------------------
    # 主查询入口
    # ----------------------------------------------------------

    def query(self, user_query: str, enable_hyde: bool = False) -> RAGResult:
        result = RAGResult()
        latency = {}
        t_total_start = time.time()

        # 1. 查询处理
        t0 = time.time()
        query_info = self._process_query(user_query, enable_hyde=enable_hyde)
        t_process = time.time() - t0
        latency["query_process"] = t_process
        latency["intent_detect"] = query_info._timings.get("intent_detect", 0)
        latency["expand"] = query_info._timings.get("expand", 0)
        latency["hyde"] = query_info._timings.get("hyde", 0)
        result.query_info = query_info

        # 2. 多路召回
        t0 = time.time()
        all_recalls: list[list[RecallResult]] = []
        all_summaries: list[SummaryRecallResult] = []

        for qi, sq in enumerate(query_info.sub_queries):
            sub_q = sq["query"]
            weight = sq.get("weight", 1.0)

            t1 = time.time()
            text_results = self._recall_text(sub_q, qi, self.topk_recall)
            latency.setdefault("text_recall", 0)
            latency["text_recall"] += time.time() - t1

            t1 = time.time()
            vector_results = self._recall_vector(sub_q, qi, self.topk_recall, query_info.intent)
            latency.setdefault("vector_recall", 0)
            latency["vector_recall"] += time.time() - t1

            t1 = time.time()
            reverse_results = self._recall_reverse(sub_q, qi, self.topk_recall, query_info.intent)
            latency.setdefault("reverse_recall", 0)
            latency["reverse_recall"] += time.time() - t1

            t1 = time.time()
            summary_results = self._recall_summary(sub_q, qi)
            latency.setdefault("summary_recall", 0)
            latency["summary_recall"] += time.time() - t1

            all_recalls.extend([text_results, vector_results, reverse_results])
            all_summaries.extend(summary_results)

        if enable_hyde and query_info.hyde_answer:
            t1 = time.time()
            hyde_results = self._recall_hyde(query_info.hyde_answer, self.topk_recall, query_info.intent)
            latency["hyde_recall"] = time.time() - t1
            all_recalls.append(hyde_results)

        # 3. RRF 融合
        t1 = time.time()
        fused = self._rrf_fuse(all_recalls)
        latency["rrf_fusion"] = time.time() - t1
        t_retrieve = time.time() - t0
        latency["retrieve"] = t_retrieve

        # 去重摘要
        seen_categories = set()
        unique_summaries = []
        for s in all_summaries:
            if s.category not in seen_categories:
                seen_categories.add(s.category)
                unique_summaries.append(s)
        result.summaries = unique_summaries

        # 4. Rerank
        t0 = time.time()
        rerank_candidates = fused[: self.topk_rerank * 2]
        reranked = self._rerank(user_query, rerank_candidates, self.topk_rerank)
        t_rerank = time.time() - t0
        latency["rerank"] = t_rerank

        # 5. 综合排序
        t0 = time.time()
        time_sensitive = query_info.intent.get("time_sensitivity") is not None
        ranked = self._composite_score(reranked, time_sensitive=time_sensitive)
        latency["composite_score"] = time.time() - t0

        result.top_comments = ranked[: self.topk_final]

        # 6. LLM 生成
        t0 = time.time()
        answer = self._generate_answer(user_query, result.top_comments, result.summaries)
        t_generate = time.time() - t0
        latency["generate"] = t_generate
        result.answer = answer

        latency["total"] = time.time() - t_total_start
        result.latency = latency

        return result


# ============================================================
# 结果打印
# ============================================================

def print_rag_result(result: RAGResult):
    print(result.answer)

    lat = result.latency
    print(f"\n⏱️  延迟统计:")
    print(f"  • 查询处理: {lat.get('query_process', 0):.3f}s")
    print(f"    • 意图检测: {lat.get('intent_detect', 0):.3f}s")
    print(f"    • 意图扩展: {lat.get('expand', 0):.3f}s")
    if lat.get("hyde", 0) > 0:
        print(f"    • HyDE: {lat.get('hyde', 0):.3f}s")
    print(f"  • 混合检索: {lat.get('retrieve', 0):.3f}s")
    print(f"    • 文本召回: {lat.get('text_recall', 0):.3f}s")
    print(f"    • 向量召回: {lat.get('vector_recall', 0):.3f}s")
    print(f"    • 反向召回: {lat.get('reverse_recall', 0):.3f}s")
    if lat.get("hyde_recall", 0) > 0:
        print(f"    • HyDE召回: {lat.get('hyde_recall', 0):.3f}s")
    print(f"    • 摘要召回: {lat.get('summary_recall', 0):.3f}s")
    print(f"    • RRF融合: {lat.get('rrf_fusion', 0):.3f}s")
    print(f"  • 排序: {lat.get('rerank', 0) + lat.get('composite_score', 0):.3f}s"
          f"（Rerank {lat.get('rerank', 0):.3f}s + 排序 {lat.get('composite_score', 0):.3f}s）")
    print(f"  • 模型回复: {lat.get('generate', 0):.3f}s")
    print(f"  • 总延迟: {lat.get('total', 0):.3f}s")

    qi = result.query_info
    print(f"\n🔍 查询处理:")
    print(f"  • 意图检测: {qi.intent}")
    print(f"  • 意图扩展:")
    for sq in qi.sub_queries:
        print(f"      - {sq['query']} (weight={sq.get('weight', 1.0)})")

    if result.summaries:
        print(f"\n📚 召回摘要类别 ({len(result.summaries)}个):")
        for i, s in enumerate(result.summaries, 1):
            print(f"  [{i}] {s.category}（被 Query [{s.query_idx}] 召回）")
            print(f"      关键词: {s.keywords}")
            print(f"      评论数: {s.comment_count}")
            preview = s.summary[:100].replace("\n", " ")
            print(f"      摘要: {preview}...")

    if result.top_comments:
        print(f"\n🏆 Top {len(result.top_comments)} 评论:")
        for c in result.top_comments:
            rank = getattr(c, "_final_rank", "?")
            final = getattr(c, "_final_score", 0)
            rerank_s = getattr(c, "_rerank_score", 0)
            rerank_r = getattr(c, "_rerank_rank", "?")
            print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"  综合排名: # {rank} | 综合得分: {final:.4f}")
            print(f"  Rerank排名: # {rerank_r} | Rerank得分: {rerank_s:.4f}")
            print(f"  评论ID: {c.doc_id}")
            print(f"  房型: {c.room_type} | 评分: {c.score_val} | 质量: {c.quality_score}")
            print(f"  发布: {c.publish_date}")
            routes = getattr(c, "_routes", [])
            if routes:
                route_str = "\n    • ".join(routes)
                print(f"  召回路由:\n    • {route_str}")
            content_preview = c.comment[:200].replace("\n", " ")
            print(f"  内容: {content_preview}...")


# ============================================================
# 知识库构建
# ============================================================

def _get_or_create_collection(dv_client, name, dimension, metric, fields_schema):
    col = dv_client.get(name)
    if col is not None:
        stats = col.stats()
        if getattr(stats, "code", -1) == 0:
            doc_count = stats.output.partitions["default"].total_doc_count
            print(f"  {name} 已存在，含 {doc_count} 条数据（断点续传）")
            return col, doc_count
    try:
        dv_client.delete(name)
    except Exception:
        pass
    dv_client.create(name=name, dimension=dimension, metric=metric, fields_schema=fields_schema)
    col = dv_client.get(name)
    for _ in range(30):
        if col is not None:
            stats = col.stats()
            if getattr(stats, "code", -1) == 0:
                break
        time.sleep(1)
        col = dv_client.get(name)
    if col is None:
        raise RuntimeError(f"{name} 创建超时")
    print(f"  {name} 创建就绪")
    return col, 0


def _upsert_in_batches(col, df, text_col, id_prefix, fields_map, embedder, batch_size, skip_count=0):
    texts = df[text_col].tolist()
    total = len(texts)
    start = (skip_count // batch_size) * batch_size
    if start > 0:
        print(f"  跳过前 {start} 条（已插入），从第 {start + 1} 条继续")
    inserted = 0
    failed = 0
    for i in range(start, total, batch_size):
        batch = texts[i : i + batch_size]
        try:
            embs = embedder.embed_batch(batch)
        except Exception as e:
            print(f"  [WARN] Embedding 失败 batch {i}: {e}")
            time.sleep(2)
            continue
        docs = []
        for j, emb in enumerate(embs):
            idx = i + j
            row = df.iloc[idx]
            doc_id = f"{id_prefix}{row.name}" if id_prefix else str(row.name)
            fields = {k: str(row[v]) for k, v in fields_map.items()}
            docs.append(Doc(id=doc_id, vector=emb, fields=fields))
        resp = col.upsert(docs)
        if hasattr(resp, "code") and resp.code == 0:
            inserted += len(docs)
        else:
            failed += len(docs)
            print(f"  [WARN] upsert 失败 batch {i}: code={getattr(resp,'code','?')} msg={getattr(resp,'message','?')}")
        done = min(i + batch_size, total)
        if done % 100 == 0 or done == total:
            print(f"  进度: {done}/{total} (本轮成功 {inserted}, 失败 {failed})")
        time.sleep(0.5)
    return inserted, failed


def build_knowledge_base(
    api_key: str,
    dashvector_api_key: str,
    dashvector_endpoint: str,
    data_dir: str = "data",
):
    print("=" * 60)
    print("开始构建酒店评论知识库（支持断点续传）")
    print("=" * 60)

    data_path = Path(data_dir)
    embedder = EmbeddingClient(api_key=api_key)
    BATCH_SIZE = 10

    dv_client = dashvector.Client(api_key=dashvector_api_key, endpoint=dashvector_endpoint)

    chroma_path = str(data_path / "chroma_db")
    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    df_filtered = pd.read_csv(data_path / "filtered_comments.csv", index_col=0)
    df_queries = pd.read_csv(data_path / "reverse_queries.csv")
    with open(data_path / "category_summaries.json", "r", encoding="utf-8") as f:
        summaries = json.load(f)

    # --- 1. 评论向量库 ---
    print("\n[1/4] 构建评论向量库 (DashVector)...")
    comment_schema = {"comment": str, "room_type": str, "fuzzy_room_type": str}
    comment_col, skip = _get_or_create_collection(
        dv_client, "comment_database", 1024, "cosine", comment_schema
    )
    if skip < len(df_filtered):
        ins, fail = _upsert_in_batches(
            comment_col, df_filtered, "comment", "",
            {"comment": "comment", "room_type": "room_type", "fuzzy_room_type": "fuzzy_room_type"},
            embedder, BATCH_SIZE, skip,
        )
        print(f"  评论数据库构建完成, 本轮新增 {ins} 条" + (f", 失败 {fail} 条" if fail else ""))
    else:
        print(f"  评论数据库已完整，跳过")

    # --- 2. 反向 Query 向量库 ---
    print("\n[2/4] 构建反向 Query 向量库 (DashVector)...")
    query_schema = {"query": str, "comment_id": str, "comment": str, "room_type": str, "fuzzy_room_type": str}
    query_col, skip = _get_or_create_collection(
        dv_client, "reverse_query_database", 1024, "cosine", query_schema
    )
    if skip < len(df_queries):
        ins, fail = _upsert_in_batches(
            query_col, df_queries, "query", "query_",
            {"query": "query", "comment_id": "comment_id", "comment": "comment",
             "room_type": "room_type", "fuzzy_room_type": "fuzzy_room_type"},
            embedder, BATCH_SIZE, skip,
        )
        print(f"  反向 Query 数据库构建完成, 本轮新增 {ins} 条" + (f", 失败 {fail} 条" if fail else ""))
    else:
        print(f"  反向 Query 数据库已完整，跳过")

    # --- 3. 摘要向量库 ---
    print("\n[3/4] 构建摘要向量库 (ChromaDB)...")
    try:
        existing = chroma_client.get_collection("summary_database")
        existing_count = existing.count()
        if existing_count >= len(summaries):
            print(f"  摘要数据库已完整（{existing_count} 条），跳过")
        else:
            print(f"  摘要数据库不完整（{existing_count}/{len(summaries)}），重建")
            chroma_client.delete_collection("summary_database")
            raise Exception("rebuild")
    except Exception:
        summary_col = chroma_client.create_collection(
            name="summary_database", metadata={"hnsw:space": "cosine"}
        )
        keywords_list = [s["keywords"] for s in summaries]
        for i in range(0, len(keywords_list), BATCH_SIZE):
            batch = keywords_list[i : i + BATCH_SIZE]
            embs = embedder.embed_batch(batch)
            num = len(embs)
            summary_col.add(
                ids=[f"summary_{j}" for j in range(i, i + num)],
                embeddings=embs,
                documents=[s["summary"] for s in summaries[i : i + num]],
                metadatas=[
                    {"category": s["category"], "keywords": s["keywords"], "comment_count": s["comment_count"]}
                    for s in summaries[i : i + num]
                ],
            )
        print(f"  摘要数据库构建完成, 共 {len(summaries)} 条")

    # --- 4. 倒排索引 ---
    print("\n[4/4] 构建倒排索引 (BM25)...")
    inv = InvertedIndex(k1=1.5, b=0.75, stopwords_file=str(data_path / "stopwords_chinese.txt"))
    documents = {str(idx): row["comment"] for idx, row in df_filtered.iterrows()}
    inv.build(documents)
    inv.save(str(data_path / "inverted_index.pkl"))

    print("\n" + "=" * 60)
    print("知识库构建完成！")
    print("=" * 60)


# ============================================================
# 主入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="酒店评论智能问答 RAG Pipeline")
    parser.add_argument("--build", action="store_true", help="构建知识库（首次使用需执行）")
    parser.add_argument("--query", type=str, help="查询问题")
    parser.add_argument("--hyde", action="store_true", help="启用 HyDE 召回")
    parser.add_argument("--interactive", action="store_true", help="交互式查询模式")
    parser.add_argument("--data-dir", type=str, default="data", help="数据目录")
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    dv_api_key = os.environ.get("DASHVECTOR_API_KEY")
    dv_endpoint = os.environ.get("DASHVECTOR_HOTEL_ENDPOINT")

    if not api_key:
        raise EnvironmentError("缺少环境变量 DASHSCOPE_API_KEY")
    if not args.build and (not dv_api_key or not dv_endpoint):
        raise EnvironmentError("缺少环境变量 DASHVECTOR_API_KEY 或 DASHVECTOR_HOTEL_ENDPOINT")

    if args.build:
        if not dv_api_key or not dv_endpoint:
            raise EnvironmentError("构建知识库需要 DASHVECTOR_API_KEY 和 DASHVECTOR_HOTEL_ENDPOINT")
        build_knowledge_base(api_key, dv_api_key, dv_endpoint, args.data_dir)
        return

    print("初始化 RAG 系统...")
    rag = HotelReviewRAG(
        api_key=api_key,
        dashvector_api_key=dv_api_key,
        dashvector_endpoint=dv_endpoint,
        data_dir=args.data_dir,
    )
    print("初始化完成！\n")

    if args.query:
        result = rag.query(args.query, enable_hyde=args.hyde)
        print_rag_result(result)
    elif args.interactive:
        print("进入交互模式（输入 q 退出）")
        while True:
            try:
                user_input = input("\n🧑 请输入问题: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in ("q", "quit", "exit"):
                break
            if not user_input:
                continue
            result = rag.query(user_input, enable_hyde=args.hyde)
            print_rag_result(result)
        print("再见！")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
