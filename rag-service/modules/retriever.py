"""混合检索器：多路召回 + RRF 融合"""

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


class HybridRetriever:
    """混合检索器：多路召回 + RRF 融合"""

    def __init__(self, inverted_index, comments_collection, reverse_queries_collection,
                 summaries_collection, embedding_client, df_comments, hyde_generator,
                 chunked_index=None, chunk_meta=None):
        """
        参数:
            chunked_index: 可选，chunk 级 BM25 倒排索引（InvertedIndex）。提供后
                BM25 路改为在 chunk 上检索，再按 comment_id 折叠回评论级别。
            chunk_meta: 可选，{chunk_id: {comment_id, seq, char_start, char_end, text}}
                用于把命中 chunk 折叠回 comment 并透出 primary_chunk 的字符跨度。
        """
        self.inverted_index = inverted_index
        self.comments_collection = comments_collection
        self.reverse_queries_collection = reverse_queries_collection
        self.summaries_collection = summaries_collection
        self.embedding_client = embedding_client
        self.hyde_generator = hyde_generator
        self.df_comments = df_comments
        self.chunked_index = chunked_index
        self.chunk_meta = chunk_meta or {}
        # comment_id → 该评论的 chunk_id 列表（按 seq 升序）
        self._comment_to_chunks: dict[str, list[str]] = {}
        if self.chunk_meta:
            tmp: dict[str, list[tuple[int, str]]] = {}
            for cid_, m in self.chunk_meta.items():
                tmp.setdefault(m['comment_id'], []).append((m['seq'], cid_))
            for k, v in tmp.items():
                v.sort()
                self._comment_to_chunks[k] = [cid for _, cid in v]

    def retrieve(self, rewritten_queries, room_type=None, fuzzy_room_type=None, topk=150,
                 final_topk=100, enable_bm25=True, enable_vector=True,
                 enable_reverse=True, enable_hyde=True, enable_summary=True,
                 dashvector_filter=None, bm25_filter_keywords=None):
        """
        混合检索

        参数:
            rewritten_queries: 改写后的 Query 列表及权重
            room_type: 精确房型约束
            fuzzy_room_type: 模糊房型约束
            topk: 每次召回的评论数量
            final_topk: 最终返回的评论数量
            dashvector_filter: DashVector 兼容过滤表达式（优化项 10）
            bm25_filter_keywords: BM25 关键词约束列表（优化项 10）
        """
        timing = {}
        retrieve_start_time = time.time()

        queries = [item['query'] for item in rewritten_queries]
        weights = [item['weight'] for item in rewritten_queries]
        embedding_time = 0
        if sum([enable_vector, enable_reverse, enable_summary]):
            embedding_start_time = time.time()
            query_embeddings = self.embedding_client.embed_batch(queries)
            embedding_time = time.time() - embedding_start_time

        # 构建过滤条件：优先使用增强约束的 dashvector_filter，回退到旧方式
        room_filter = dashvector_filter
        if not room_filter:
            if room_type:
                room_filter = f"room_type = '{room_type}'"
            elif fuzzy_room_type:
                room_filter = f"fuzzy_room_type = '{fuzzy_room_type}'"

        # BM25 关键词约束
        bm25_kws = bm25_filter_keywords or []

        enabled_routes = sum([enable_bm25, enable_vector, enable_reverse, enable_hyde, enable_summary])
        if enabled_routes == 0:
            raise ValueError("至少需要启用一路召回")

        with ThreadPoolExecutor(max_workers=enabled_routes) as executor:
            futures = {}
            if enable_bm25:
                futures[executor.submit(self._route_bm25, queries, topk, bm25_kws)] = 'bm25'
            if enable_vector:
                futures[executor.submit(self._route_vector, query_embeddings, topk, room_filter)] = 'vector'
            if enable_reverse:
                futures[executor.submit(self._route_reverse, query_embeddings, topk, room_filter)] = 'reverse'
            if enable_hyde:
                futures[executor.submit(self._route_hyde, queries, topk, room_filter)] = 'hyde'
            if enable_summary:
                futures[executor.submit(self._route_summary, query_embeddings)] = 'summary'

            comment_results = []
            summary_results = []
            route_results = {}
            hyde_results = {}

            for future in as_completed(futures):
                route_name = futures[future]

                if route_name == 'summary':
                    results, route_timing = future.result()
                    timing[route_name] = route_timing + embedding_time
                    summary_results = results

                elif route_name == 'hyde':
                    results, route_timing, hyde_generated = future.result()
                    timing[route_name] = route_timing
                    route_results[route_name] = results
                    comment_results.extend(results)
                    hyde_results = hyde_generated

                else:
                    results, route_timing = future.result()
                    timing[route_name] = route_timing if route_name == 'bm25' else route_timing + embedding_time
                    route_results[route_name] = results
                    comment_results.extend(results)

        # 设置未启用通路的默认延迟
        if not enable_bm25:
            timing['bm25'] = 0
        if not enable_vector:
            timing['vector'] = 0
        if not enable_reverse:
            timing['reverse'] = 0
        if not enable_hyde:
            timing['hyde'] = {'total': 0, 'generation': 0, 'retrieval': 0}
        if not enable_summary:
            timing['summary'] = 0

        # RRF 融合
        rrf_start_time = time.time()
        rrf_scores = self._rrf_fusion(comment_results, weights, k=60)

        rrf_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        rrf_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(rrf_sorted, 1)}

        # 构建检索结果
        final_comment_results = []
        for doc_id, rrf_score in rrf_sorted[:final_topk]:
            comment_row = self.df_comments.loc[doc_id]

            route_ranks = {}
            primary_chunk = None
            best_chunk_rank = None
            for route_name, results in route_results.items():
                for d_id, r_name, rank, metadata in results:
                    if d_id == doc_id:
                        if route_name not in route_ranks:
                            route_ranks[route_name] = []
                        route_ranks[route_name].append({
                            'rank': rank,
                            'metadata': metadata
                        })
                        # BM25 chunked 路会带 primary_chunk；取 rank 最靠前的
                        if metadata.get('primary_chunk') and (
                            best_chunk_rank is None or rank < best_chunk_rank
                        ):
                            primary_chunk = metadata['primary_chunk']
                            best_chunk_rank = rank

            comment_text = comment_row['comment']
            # 兜底：未命中 chunk 时，把首个 chunk 当代表（如果 chunk_meta 可用）
            if primary_chunk is None and self._comment_to_chunks.get(doc_id):
                first_id = self._comment_to_chunks[doc_id][0]
                first_chunk = self.chunk_meta[first_id]
                primary_chunk = {
                    'chunk_id': first_id,
                    'text': first_chunk['text'],
                    'char_start': first_chunk['char_start'],
                    'char_end': first_chunk['char_end'],
                    'seq': first_chunk['seq'],
                }

            final_comment_results.append({
                'comment_id': doc_id,
                'comment': comment_text,
                'primary_chunk': primary_chunk,
                'rrf_score': rrf_score,
                'rrf_rank': rrf_ranks[doc_id],
                'route_ranks': route_ranks,
                'metadata': {
                    'score': comment_row['score'],
                    'publish_date': comment_row['publish_date'],
                    'quality_score': comment_row['quality_score'],
                    'review_count': comment_row['review_count'],
                    'useful_count': comment_row['useful_count'],
                    'room_type': comment_row['room_type'],
                    'fuzzy_room_type': comment_row['fuzzy_room_type']
                }
            })

        timing['rrf_fusion'] = time.time() - rrf_start_time
        timing_info = {
            'routes': timing,
            'total': time.time() - retrieve_start_time
        }

        return final_comment_results, summary_results, timing_info, hyde_results

    # ── BM25 路 ──────────────────────────────────────────────

    def _route_bm25(self, queries, topk, bm25_kws=None):
        """第一路：BM25 文本召回。chunked_index 存在时走 chunk 级检索 + 折叠。"""
        start = time.time()
        results = []
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = [
                executor.submit(self._single_bm25_query, query_idx, query, topk, bm25_kws)
                for query_idx, query in enumerate(queries)
            ]
            for future in as_completed(futures):
                results.extend(future.result())
        return results, time.time() - start

    def _single_bm25_query(self, query_idx, query, topk, bm25_kws=None):
        """单次 BM25 查询，支持关键词约束增强"""
        # 将 BM25 过滤关键词追加到 query 中增强检索精度
        enhanced_query = query
        if bm25_kws:
            enhanced_query = query + " " + " ".join(bm25_kws)

        # 优先使用 chunked 索引
        if self.chunked_index is not None and self.chunk_meta:
            raw = self.chunked_index.search(enhanced_query, topk=topk * 5)
            seen = {}
            collapsed = []
            for chunk_id, score in raw:
                meta = self.chunk_meta.get(chunk_id)
                if not meta:
                    continue
                cid = meta["comment_id"]
                if cid in seen:
                    continue
                seen[cid] = True
                collapsed.append((cid, score, chunk_id, meta))
                if len(collapsed) >= topk:
                    break
            return [
                (cid, 'bm25', rank, {
                    'query_idx': query_idx,
                    'primary_chunk': {
                        'chunk_id': chunk_id,
                        'text': meta['text'],
                        'char_start': meta['char_start'],
                        'char_end': meta['char_end'],
                        'seq': meta['seq'],
                    },
                })
                for rank, (cid, _score, chunk_id, meta) in enumerate(collapsed, 1)
            ]

        # 回退：旧的整条评论 BM25
        bm25_results = self.inverted_index.search(enhanced_query, topk=topk)
        return [(doc_id, 'bm25', rank, {'query_idx': query_idx})
                for rank, (doc_id, score) in enumerate(bm25_results, 1)]

    # ── 向量路 ────────────────────────────────────────────────

    def _route_vector(self, query_embeddings, topk, room_filter):
        """第二路：基础向量召回"""
        start = time.time()
        results = []
        with ThreadPoolExecutor(max_workers=len(query_embeddings)) as executor:
            futures = [
                executor.submit(self._single_vector_query, query_idx, emb, room_filter, topk)
                for query_idx, emb in enumerate(query_embeddings)
            ]
            for future in as_completed(futures):
                results.extend(future.result())
        return results, time.time() - start

    def _single_vector_query(self, query_idx, embedding, room_filter, topk):
        response = self.comments_collection.query(vector=embedding, topk=topk, filter=room_filter)
        docs = response.output if response.output else []
        return [(doc.id, 'vector', rank, {'query_idx': query_idx})
                for rank, doc in enumerate(docs, 1)]

    # ── 反向 Query 路 ────────────────────────────────────────

    def _route_reverse(self, query_embeddings, topk, room_filter):
        """第三路：反向 Query 召回"""
        start = time.time()
        results = []
        with ThreadPoolExecutor(max_workers=len(query_embeddings)) as executor:
            futures = [
                executor.submit(self._single_reverse_query, query_idx, emb, room_filter, topk)
                for query_idx, emb in enumerate(query_embeddings)
            ]
            for future in as_completed(futures):
                results.extend(future.result())
        return results, time.time() - start

    def _single_reverse_query(self, query_idx, embedding, room_filter, topk):
        response = self.reverse_queries_collection.query(vector=embedding, topk=topk, filter=room_filter)
        docs = response.output if response.output else []
        return [(doc.fields.get('comment_id'), 'reverse', rank, {'query_idx': query_idx})
                for rank, doc in enumerate(docs, 1)]

    # ── HyDE 路 ──────────────────────────────────────────────

    def _route_hyde(self, queries, topk, room_filter):
        """第四路：HyDE 增强召回"""
        route_start = time.time()
        results = []
        hyde_generated = {}
        generation_time = []
        retrieval_time = []

        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = [
                executor.submit(self._single_hyde_pipeline, query_idx, query, topk, room_filter)
                for query_idx, query in enumerate(queries)
            ]
            for future in as_completed(futures):
                query_results, gen_time, ret_time, query_idx, hyde_responses = future.result()
                results.extend(query_results)
                generation_time.append(gen_time)
                retrieval_time.append(ret_time)
                hyde_generated[query_idx] = hyde_responses

        timing = {
            'total': time.time() - route_start,
            'generation': max(generation_time),
            'retrieval': max(retrieval_time)
        }
        return results, timing, hyde_generated

    def _single_hyde_pipeline(self, query_idx, query, topk, room_filter):
        """单个 Query 的 HyDE 生成与召回"""
        gen_start = time.time()
        hyde_responses = self.hyde_generator.generate(query)
        generation_time = time.time() - gen_start

        ret_start = time.time()
        hyde_embeddings = self.embedding_client.embed_batch(hyde_responses)

        raw_results = []
        with ThreadPoolExecutor(max_workers=len(hyde_embeddings)) as executor:
            futures = [
                executor.submit(self._single_hyde_query, query_idx, hyde_idx, emb, room_filter, topk)
                for hyde_idx, emb in enumerate(hyde_embeddings)
            ]
            for future in as_completed(futures):
                raw_results.extend(future.result())

        # 去重
        best_candidates = {}
        for item in raw_results:
            doc_id, route_name, rank, metadata = item
            if doc_id not in best_candidates or rank < best_candidates[doc_id][0]:
                best_candidates[doc_id] = (rank, item)

        results = [item for rank, item in best_candidates.values()]
        retrieval_time = time.time() - ret_start

        return results, generation_time, retrieval_time, query_idx, hyde_responses

    def _single_hyde_query(self, query_idx, hyde_idx, embedding, room_filter, topk):
        response = self.comments_collection.query(vector=embedding, topk=topk, filter=room_filter)
        docs = response.output if response.output else []
        return [(doc.id, 'hyde', rank, {'query_idx': query_idx, 'hyde_idx': hyde_idx})
                for rank, doc in enumerate(docs, 1)]

    # ── 摘要路 ────────────────────────────────────────────────

    def _route_summary(self, query_embeddings):
        """第五路：类别摘要召回"""
        start = time.time()

        summary_results = self.summaries_collection.query(
            query_embeddings=query_embeddings, n_results=1
        )

        category_map = {}
        for query_idx, (category_ids, documents, metadatas) in enumerate(zip(
            summary_results['ids'],
            summary_results['documents'],
            summary_results['metadatas']
        )):
            if category_ids:
                category_id = category_ids[0]
                if category_id not in category_map:
                    category_map[category_id] = {
                        'summary': documents[0],
                        'metadata': metadatas[0] if metadatas else {},
                        'retrieved_by_queries': []
                    }
                category_map[category_id]['retrieved_by_queries'].append(query_idx)

        summaries = list(category_map.values())
        return summaries, time.time() - start

    # ── RRF 融合 ─────────────────────────────────────────────

    def _rrf_fusion(self, all_results, weights, k=60):
        """RRF 融合"""
        rrf_scores = defaultdict(float)
        for doc_id, route_name, rank, metadata in all_results:
            rrf_scores[doc_id] += (1 / (k + rank)) * weights[metadata['query_idx']]
        return dict(rrf_scores)
