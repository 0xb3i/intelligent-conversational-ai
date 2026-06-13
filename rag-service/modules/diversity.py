"""
多样性重排模块：MMR / DPP / Fast-DPP

在 MultiFactorRanker 完成多因子打分后，从候选池（retrieval_topk 条）
中选出最终展示的 ranking_topk 条评论，兼顾相关性与多样性。

算法一览：
- MMR  (Maximal Marginal Relevance): λ·rel - (1-λ)·max_sim
- DPP  (Determinantal Point Process): 贪心核矩阵行列式最大化
- Fast-DPP: k-means 聚类 + 簇内预筛选 + DPP 精排（O(N²·K) → O(N·c + M²·K)）
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans


def _safe_cosine_sim(embeddings: NDArray) -> NDArray:
    """计算 N×N 余弦相似度矩阵，处理零向量边界情况"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = embeddings / norms
    sim = normalized @ normalized.T
    np.fill_diagonal(sim, 0.0)
    return np.clip(sim, -1.0, 1.0)


class DiversityReranker:
    """多样性重排器：从候选池中选出兼顾相关性与多样性的子集"""

    def __init__(self):
        pass

    # ── MMR ────────────────────────────────────────────────────────

    def mmr_select(
        self,
        final_scores: NDArray,
        sim_matrix: NDArray,
        topk: int,
        lambda_param: float = 0.5,
    ) -> list[int]:
        """
        Maximal Marginal Relevance 贪心选择

        MMR(D_i) = λ·score_i - (1-λ)·max sim(D_i, D_j∈S)

        参数:
            final_scores: shape (N,), 多因子综合得分（0-1 归一化后）
            sim_matrix: shape (N, N), 候选评论间的余弦相似度
            topk: 需要选出的评论数量
            lambda_param: 相关性权重，λ 越大越偏相关性，越小越偏多样性
        返回:
            selected_indices: 选中的评论在原始候选池中的索引列表
        """
        n = len(final_scores)
        if topk >= n:
            return list(range(n))

        scores = final_scores.copy().astype(np.float64)
        selected: list[int] = []
        remaining = set(range(n))

        # 第一轮选分数最高的
        first = int(np.argmax(scores))
        selected.append(first)
        remaining.remove(first)

        # 贪心迭代
        for _ in range(topk - 1):
            best_idx = -1
            best_val = -1e12
            remaining_list = list(remaining)

            # MMR 得分 = λ·score_i - (1-λ)·max_{j∈selected} sim(i, j)
            max_sim_to_selected = np.max(
                sim_matrix[remaining_list][:, selected], axis=1
            )
            mmr_scores = (
                lambda_param * scores[remaining_list]
                - (1.0 - lambda_param) * max_sim_to_selected
            )

            best_local = int(np.argmax(mmr_scores))
            best_idx = remaining_list[best_local]

            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected

    # ── DPP (贪心 MAP 推断) ─────────────────────────────────────────

    def dpp_select(
        self,
        final_scores: NDArray,
        sim_matrix: NDArray,
        topk: int,
        theta: float = 1.0,
    ) -> list[int]:
        """
        Determinantal Point Process — 正确贪心 MAP 推断

        核矩阵 L_{ij} = q_i · S_{ij} · q_j
        贪心增益：gain(i) = L[i,i] − L[i,S] · L[S,S]⁻¹ · L[S,i]
        第二项是「候选 i 已被已选集 S 解释掉的方差」，越大越冗余。

        参数:
            final_scores: shape (N,), 多因子综合得分
            sim_matrix: shape (N, N), 余弦相似度
            topk: 目标数量
            theta: 质量放大指数。θ > 1 拉大高分差 → 偏相关性；
                   θ < 1 压缩分数 → 更多样性
        返回:
            selected_indices
        """
        n = len(final_scores)
        if topk >= n:
            return list(range(n))

        # 质量向量：min-max 归一化 → [0.1, 1.0] → 幂函数放大
        s = final_scores.astype(np.float64)
        s_min, s_max = s.min(), s.max()
        if s_max - s_min > 1e-8:
            quality = (s - s_min) / (s_max - s_min)
            quality = quality * 0.9 + 0.1
        else:
            quality = np.full(n, 0.5, dtype=np.float64)
        quality = quality ** max(theta, 0.1)

        # 相似度映射到 [0, 1]
        S = (sim_matrix + 1.0) / 2.0

        # DPP 核矩阵 L_{ij} = q_i · S_{ij} · q_j
        L = quality[:, np.newaxis] * S * quality[np.newaxis, :]
        # 小正则化保证正定
        L += np.eye(n) * 1e-8

        selected: list[int] = []
        remaining = set(range(n))

        for _ in range(topk):
            if not selected:
                # 第一轮选分数最高的（用原始 score）
                remaining_list = list(remaining)
                best_idx = remaining_list[int(np.argmax(final_scores[remaining_list]))]
            else:
                # 计算 L[S,S] 的逆（|S| ≤ topk ≤ 10，代价很小）
                s_arr = np.array(selected, dtype=int)
                L_SS = L[s_arr][:, s_arr]
                try:
                    L_SS_inv = np.linalg.inv(L_SS)
                except np.linalg.LinAlgError:
                    L_SS_inv = np.linalg.pinv(L_SS)

                best_gain = -np.inf
                best_idx = -1

                for i in remaining:
                    L_iS = L[i, s_arr]  # shape (|S|,)
                    # Schur complement: gain = L_ii - L_iS · L_SS⁻¹ · L_Si
                    conditional_variance = L_iS @ L_SS_inv @ L_iS
                    gain = L[i, i] - conditional_variance
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i

            if best_idx == -1:
                remaining_list = list(remaining)
                best_idx = remaining_list[0]

            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected

    # ── Fast-DPP ───────────────────────────────────────────────────

    def fast_dpp_select(
        self,
        final_scores: NDArray,
        embeddings: NDArray,
        topk: int,
        theta: float = 1.0,
        n_clusters: int | None = None,
        candidates_per_cluster: int = 3,
        random_state: int = 42,
    ) -> list[int]:
        """
        Fast-DPP: k-means 聚类预筛选 + DPP 精排

        1. 用 k-means（k = topk * 2）将候选按 embedding 聚类
        2. 每个簇内保留 final_score 最高的 candidates_per_cluster 条
        3. 在缩减后的候选集上运行标准 DPP

        复杂度从 O(N²·K) 降到 O(N·c + M²·K)，M ≤ n_clusters × candidates_per_cluster

        参数:
            final_scores: shape (N,)
            embeddings: shape (N, D)
            topk: 最终选出数量
            theta: DPP 多样性强度
            n_clusters: 聚类数，默认 topk * 2
            candidates_per_cluster: 每个簇保留的候选数
        返回:
            selected_indices: 相对于原始候选池的索引列表
        """
        n = len(final_scores)
        if topk >= n:
            return list(range(n))

        if n_clusters is None:
            n_clusters = min(topk * 2, n)

        # 确保每个簇至少有 1 条
        n_clusters = min(n_clusters, n)

        # Step 1: k-means 聚类
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(embeddings)

        # Step 2: 簇内 top-k 预筛选
        pruned_indices: list[int] = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_scores = final_scores[cluster_indices]
            top_local = min(candidates_per_cluster, len(cluster_indices))
            best_local = cluster_indices[
                np.argpartition(cluster_scores, -top_local)[-top_local:]
            ]
            pruned_indices.extend(best_local.tolist())

        pruned_indices = sorted(set(pruned_indices))

        if len(pruned_indices) <= topk:
            # 直接按分数排序返回
            order = np.argsort(final_scores[pruned_indices])[::-1]
            return [pruned_indices[i] for i in order[:topk]]

        # Step 3: 在缩减集上运行 DPP 精排
        pruned_embeddings = embeddings[pruned_indices]
        pruned_scores = final_scores[pruned_indices]
        pruned_sim = _safe_cosine_sim(pruned_embeddings)

        local_selected = self.dpp_select(
            pruned_scores, pruned_sim, topk, theta
        )

        return [pruned_indices[i] for i in local_selected]

    # ── 统一入口 ───────────────────────────────────────────────────

    def rerank(
        self,
        candidates: list[dict],
        final_scores: NDArray,
        embeddings: NDArray | None,
        topk: int,
        method: str = "mmr",
        lambda_param: float = 0.5,
        theta: float = 1.0,
    ) -> tuple[list[dict], dict]:
        """
        多样性重排统一入口

        参数:
            candidates: 候选评论列表（含 comment 文本和元数据）
            final_scores: shape (N,), 多因子综合得分
            embeddings: shape (N, D), 可选，候选评论的 embedding 向量。
                        若为 None 则用随机 embedding 占位（仅用于退化测试）。
            topk: 最终选出数量
            method: "mmr" | "dpp" | "fast_dpp"
            lambda_param: MMR 的 λ 参数
            theta: DPP 的 θ 参数

        返回:
            (selected_candidates, diversity_stats)
        """
        n = len(candidates)
        if topk >= n:
            return candidates, {"method": method, "effective": False,
                                "reason": "topk >= candidate_count"}

        if embeddings is None:
            # 退化：无 embedding 时回退到贪心 top-k
            order = np.argsort(final_scores)[::-1]
            return [candidates[i] for i in order[:topk]], {
                "method": method, "effective": False,
                "reason": "no_embeddings"
            }

        sim_matrix = _safe_cosine_sim(embeddings.astype(np.float64))

        if method == "mmr":
            indices = self.mmr_select(final_scores, sim_matrix, topk, lambda_param)
        elif method == "dpp":
            indices = self.dpp_select(final_scores, sim_matrix, topk, theta)
        elif method == "fast_dpp":
            indices = self.fast_dpp_select(final_scores, embeddings.astype(np.float64),
                                           topk, theta)
        else:
            raise ValueError(f"不支持的多样性重排方法: {method}")

        selected = [candidates[i] for i in indices]

        # 多样性统计
        if len(indices) >= 2:
            selected_sim = sim_matrix[indices][:, indices]
            upper_tri = np.triu(selected_sim, k=1)
            n_pairs = len(indices) * (len(indices) - 1) / 2
            mean_pairwise_sim = float(upper_tri.sum() / n_pairs) if n_pairs > 0 else 0.0
        else:
            mean_pairwise_sim = 0.0

        stats = {
            "method": method,
            "effective": True,
            "params": {"lambda": lambda_param, "theta": theta},
            "input_count": n,
            "output_count": len(indices),
            "mean_pairwise_cosine_sim": mean_pairwise_sim,
        }

        return selected, stats
