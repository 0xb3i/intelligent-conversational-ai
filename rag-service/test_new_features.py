"""
新功能测试脚本：多样性重排 + 多轮对话上下文管理

运行方式:
    cd rag-service
    python test_new_features.py
"""

import numpy as np

from modules.diversity import DiversityReranker, _safe_cosine_sim
from modules.conversation import ConversationManager, QueryResolver


# ── 测试 1: DiversityReranker ────────────────────────────────────────

def test_mmr_select():
    """MMR 贪心选择：验证多样性提升"""
    # 使用低维 embedding + 明显聚类构造：
    # 簇A: idx 0,1 高度相似 (cos_sim ≈ 0.99)
    # 簇B: idx 2,3 高度相似 (cos_sim ≈ 0.99)
    # 独立: idx 4 与所有其他都低相似
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],           # 0: 簇A
        [0.99, 0.02, 0.0, 0.0],         # 1: 簇A (cos_sim ≈ 0.9998)
        [0.0, 1.0, 0.0, 0.0],           # 2: 簇B
        [0.02, 0.99, 0.0, 0.0],         # 3: 簇B (cos_sim ≈ 0.9998)
        [0.0, 0.0, 1.0, 0.0],           # 4: 独立
    ], dtype=np.float64)

    # 分数：贪心 top-3 = [0, 1, 2]，其中 0 和 1 高度冗余
    scores = np.array([0.9, 0.85, 0.82, 0.78, 0.75], dtype=np.float64)
    sim_matrix = _safe_cosine_sim(embeddings)
    reranker = DiversityReranker()

    greedy = list(np.argsort(scores)[::-1][:3])
    greedy_sim = (sim_matrix[greedy][:, greedy].sum() - 3) / 6
    print(f"  Baseline (greedy) indices={greedy}, 平均成对相似度: {greedy_sim:.4f}")

    # MMR lambda=0.5: 应选出更分散的集合 [0, 2, 4] 而非 [0, 1, 2]
    mmr_indices = reranker.mmr_select(scores, sim_matrix, topk=3, lambda_param=0.5)
    mmr_sim = (sim_matrix[mmr_indices][:, mmr_indices].sum() - 3) / 6
    print(f"  MMR (λ=0.5) indices={mmr_indices}, 平均成对相似度: {mmr_sim:.4f}")

    assert mmr_sim < greedy_sim, f"MMR 应降低平均相似度，但 {mmr_sim:.4f} >= {greedy_sim:.4f}"
    # 验证 MMR 确实跨簇选择（含 idx 4）
    assert 4 in mmr_indices, f"MMR 应包含独立评论 idx 4，实际 {mmr_indices}"
    print("  ✓ MMR 多样性提升验证通过")

    # MMR lambda=1.0: 退化为贪心
    mmr_greedy = reranker.mmr_select(scores, sim_matrix, topk=3, lambda_param=1.0)
    assert mmr_greedy == greedy, f"λ=1.0 时 MMR 应退化为贪心，但得到 {mmr_greedy}"
    print("  ✓ MMR λ=1.0 退化验证通过")


def test_dpp_select():
    """DPP 贪心 MAP：验证质量-多样性平衡"""
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.99, 0.02, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.02, 0.99, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float64)

    scores = np.array([0.95, 0.85, 0.80, 0.75, 0.70], dtype=np.float64)
    sim_matrix = _safe_cosine_sim(embeddings)
    reranker = DiversityReranker()

    greedy = list(np.argsort(scores)[::-1][:3])
    dpp_indices = reranker.dpp_select(scores, sim_matrix, topk=3, theta=1.0)
    assert len(dpp_indices) == 3
    dpp_sim = (sim_matrix[dpp_indices][:, dpp_indices].sum() - 3) / 6
    greedy_sim = (sim_matrix[greedy][:, greedy].sum() - 3) / 6
    print(f"  DPP (θ=1.0) indices={dpp_indices}, 平均成对相似度: {dpp_sim:.4f} vs 贪心: {greedy_sim:.4f}")
    assert 4 in dpp_indices, f"DPP 应包含独立评论 idx 4，实际 {dpp_indices}"
    print("  ✓ DPP 选择验证通过")


def test_fast_dpp():
    """Fast-DPP：k-means 预选 + DPP 精排"""
    np.random.seed(456)
    embeddings = np.random.randn(100, 128).astype(np.float64)
    scores = np.random.rand(100).astype(np.float64)

    reranker = DiversityReranker()
    indices = reranker.fast_dpp_select(scores, embeddings, topk=10, theta=1.0)
    assert len(indices) == 10, f"Fast-DPP 应选 10 条，实际 {len(indices)}"
    assert len(set(indices)) == 10, "应有不重复结果"
    print(f"  Fast-DPP 选出 {len(indices)} 条不重复结果")
    print("  ✓ Fast-DPP 基本验证通过")


def test_rerank_entry():
    """rerank() 统一入口测试"""
    np.random.seed(789)
    n = 20
    embeddings = np.random.randn(n, 128).astype(np.float64)
    scores = np.random.rand(n).astype(np.float64)
    candidates = [
        {"comment_id": f"id_{i}", "comment": f"评论 {i}", "metadata": {"score": 4.0}}
        for i in range(n)
    ]

    reranker = DiversityReranker()

    for method in ["mmr", "dpp", "fast_dpp"]:
        selected, stats = reranker.rerank(candidates, scores, embeddings, topk=5, method=method)
        assert len(selected) == 5
        print(f"  {method}: 平均成对相似度 {stats['mean_pairwise_cosine_sim']:.4f}")
    print("  ✓ rerank() 三种模式验证通过")

    selected, stats = reranker.rerank(candidates, scores, embeddings, topk=20, method="mmr")
    assert stats["effective"] is False
    print("  ✓ topk >= candidate_count 退化验证通过")


# ── 测试 2: ConversationManager ──────────────────────────────────────

def test_conversation_manager_basic():
    """ConversationManager 基础功能"""
    conv = ConversationManager(session_id="test-1")

    conv.add_turn("早餐怎么样？", "早餐怎么样？", "早餐很丰富，有中式西式可选。", ["id1", "id2"])
    assert len(conv.turns) == 1
    print("  ✓ add_turn 基本功能验证通过")

    conv.add_turn("那午餐呢？", "酒店午餐怎么样？", "午餐在二楼中餐厅，有粤菜套餐。", ["id3"])
    assert len(conv.turns) == 2
    assert conv.turns[1].resolved_query == "酒店午餐怎么样？"

    ctx = conv.get_retrieval_context()
    assert ctx["has_history"] is True
    assert len(ctx["recent_turns"]) == 2

    gen_ctx = conv.get_generation_context()
    assert "早餐" in gen_ctx
    assert "午餐" in gen_ctx

    d = conv.to_dict()
    conv2 = ConversationManager.from_dict(d)
    assert len(conv2.turns) == 2
    assert conv2.turns[0].user_query == "早餐怎么样？"
    print("  ✓ to_dict / from_dict 序列化验证通过")


def test_conversation_manager_followup_detection():
    """追问检测"""
    conv = ConversationManager()
    conv.add_turn("早餐怎么样？", "早餐怎么样？", "早餐很丰富。", [])

    assert conv.is_followup("那午餐呢？") is True
    assert conv.is_followup("这个好吗？") is True
    assert conv.is_followup("早餐和午餐比哪个更好？") is True
    assert conv.is_followup("广州塔离酒店多远？") is False
    print("  ✓ is_followup 追问检测验证通过")


def test_conversation_manager_summary():
    """摘要触发"""
    conv = ConversationManager()
    for i in range(5):
        conv.add_turn(f"问题{i}？", f"问题{i}？", f"回答{i}", [])

    triggered, text = conv.maybe_summarize()
    assert triggered
    assert len(text) > 0
    conv.apply_summary("用户询问了多个方面的问题。")
    assert len(conv.long_term_summary) > 0
    print("  ✓ 摘要触发与压缩验证通过")


def test_conversation_expiry():
    """会话过期"""
    conv = ConversationManager()
    assert conv.is_expired(ttl_seconds=1800) is False
    conv.last_active = 0
    assert conv.is_expired(ttl_seconds=1800) is True
    print("  ✓ 会话过期检测验证通过")


# ── 运行 ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Part A: 多样性重排测试")
    print("=" * 60)
    test_mmr_select()
    test_dpp_select()
    test_fast_dpp()
    test_rerank_entry()
    print()

    print("=" * 60)
    print("Part B: 多轮对话上下文管理测试")
    print("=" * 60)
    test_conversation_manager_basic()
    test_conversation_manager_followup_detection()
    test_conversation_manager_summary()
    test_conversation_expiry()
    print()

    print("=" * 60)
    print("全部测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
