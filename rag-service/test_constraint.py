"""约束检测模块单元测试

用法：
    cd rag-service
    python -m pytest test_constraint.py -v

或直接运行：
    python test_constraint.py
"""

import json
import sys
from pathlib import Path

# 确保 rag-service 在 path 中
sys.path.insert(0, str(Path(__file__).parent))

# 直接导入 intent 模块，避免触发 __init__.py 的链式导入
import importlib.util
_intent_spec = importlib.util.spec_from_file_location(
    "intent", str(Path(__file__).parent / "modules" / "intent.py")
)
_intent = importlib.util.module_from_spec(_intent_spec)
_intent_spec.loader.exec_module(_intent)

ConstraintDetector = _intent.ConstraintDetector
TRAVEL_TYPES = _intent.TRAVEL_TYPES
CATEGORIES = _intent.CATEGORIES

# 模拟 LLMClient（用于离线测试）
class MockLLMClient:
    """模拟 LLM 客户端，返回预定义的 JSON 响应"""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = None

    def generate(self, prompt, temperature=0.1):
        self.call_count += 1
        self.last_prompt = prompt
        # 根据 prompt 中的 query 返回对应响应
        for query_key, response in self.responses.items():
            if query_key in prompt:
                return response
        # 默认返回空约束
        return json.dumps({
            "room_type": None,
            "fuzzy_room_type": None,
            "time_sensitivity": None,
            "score_range": None,
            "travel_type": None,
            "categories": None,
            "date_range": None,
            "dashvector_filter": None,
            "bm25_filter_keywords": []
        }, ensure_ascii=False)


EXACT_ROOM_TYPES = ["花园大床房", "花园双床房", "红棉大床套房"]
FUZZY_ROOM_TYPES = ["大床房", "双床房", "套房"]


def test_default_result():
    """测试默认空约束"""
    detector = ConstraintDetector(MockLLMClient(), EXACT_ROOM_TYPES, FUZZY_ROOM_TYPES)
    result = detector._default_result()
    assert result["room_type"] is None
    assert result["fuzzy_room_type"] is None
    assert result["time_sensitivity"] is None
    assert result["score_range"] is None
    assert result["travel_type"] is None
    assert result["categories"] is None
    assert result["date_range"] is None
    assert result["dashvector_filter"] is None
    assert result["bm25_filter_keywords"] == []


def test_validate_room_type():
    """测试房型校验"""
    detector = ConstraintDetector(MockLLMClient(), EXACT_ROOM_TYPES, FUZZY_ROOM_TYPES)

    # 有效精确房型
    data = {"room_type": "花园大床房", "fuzzy_room_type": None, "time_sensitivity": None,
            "score_range": None, "travel_type": None, "categories": None,
            "date_range": None, "dashvector_filter": None, "bm25_filter_keywords": []}
    result = detector._validate_and_clean(data)
    assert result["room_type"] == "花园大床房"

    # 无效房型 → None
    data["room_type"] = "不存在的房型"
    result = detector._validate_and_clean(data)
    assert result["room_type"] is None

    # 有效模糊房型
    data["room_type"] = None
    data["fuzzy_room_type"] = "大床房"
    result = detector._validate_and_clean(data)
    assert result["fuzzy_room_type"] == "大床房"


def test_validate_score_range():
    """测试评分约束校验"""
    detector = ConstraintDetector(MockLLMClient(), EXACT_ROOM_TYPES, FUZZY_ROOM_TYPES)

    # 有效评分范围
    data = {"room_type": None, "fuzzy_room_type": None, "time_sensitivity": None,
            "score_range": {"min": 4, "max": 5}, "travel_type": None,
            "categories": None, "date_range": None,
            "dashvector_filter": None, "bm25_filter_keywords": []}
    result = detector._validate_and_clean(data)
    assert result["score_range"] == {"min": 4, "max": 5}

    # 超出范围 → 裁剪
    data["score_range"] = {"min": 0, "max": 10}
    result = detector._validate_and_clean(data)
    assert result["score_range"] == {"min": 1, "max": 5}

    # 无效格式 → None
    data["score_range"] = "invalid"
    result = detector._validate_and_clean(data)
    assert result["score_range"] is None


def test_validate_travel_type():
    """测试出行类型校验"""
    detector = ConstraintDetector(MockLLMClient(), EXACT_ROOM_TYPES, FUZZY_ROOM_TYPES)

    data = {"room_type": None, "fuzzy_room_type": None, "time_sensitivity": None,
            "score_range": None, "travel_type": "商务出差", "categories": None,
            "date_range": None, "dashvector_filter": None, "bm25_filter_keywords": []}
    result = detector._validate_and_clean(data)
    assert result["travel_type"] == "商务出差"

    data["travel_type"] = "无效类型"
    result = detector._validate_and_clean(data)
    assert result["travel_type"] is None


def test_validate_categories():
    """测试话题类别校验"""
    detector = ConstraintDetector(MockLLMClient(), EXACT_ROOM_TYPES, FUZZY_ROOM_TYPES)

    data = {"room_type": None, "fuzzy_room_type": None, "time_sensitivity": None,
            "score_range": None, "travel_type": None,
            "categories": ["餐饮设施", "前台服务", "无效类别"],
            "date_range": None, "dashvector_filter": None, "bm25_filter_keywords": []}
    result = detector._validate_and_clean(data)
    assert result["categories"] == ["餐饮设施", "前台服务"]  # 无效类别被过滤

    data["categories"] = ["全部无效"]
    result = detector._validate_and_clean(data)
    assert result["categories"] is None


def test_validate_time_sensitivity():
    """测试时效性校验"""
    detector = ConstraintDetector(MockLLMClient(), EXACT_ROOM_TYPES, FUZZY_ROOM_TYPES)

    for valid_ts in ["clear", "implied", None]:
        data = {"room_type": None, "fuzzy_room_type": None,
                "time_sensitivity": valid_ts, "score_range": None,
                "travel_type": None, "categories": None,
                "date_range": None, "dashvector_filter": None, "bm25_filter_keywords": []}
        result = detector._validate_and_clean(data)
        assert result["time_sensitivity"] == valid_ts

    data["time_sensitivity"] = "invalid"
    result = detector._validate_and_clean(data)
    assert result["time_sensitivity"] is None


def test_build_dashvector_filter():
    """测试 DashVector 过滤表达式构建"""
    detector = ConstraintDetector(MockLLMClient(), EXACT_ROOM_TYPES, FUZZY_ROOM_TYPES)

    # 仅房型
    constraints = {"room_type": "花园大床房", "fuzzy_room_type": None,
                   "travel_type": None, "score_range": None}
    assert detector.build_dashvector_filter(constraints) == "room_type = '花园大床房'"

    # 房型 + 评分
    constraints = {"room_type": "花园大床房", "fuzzy_room_type": None,
                   "travel_type": None, "score_range": {"min": 4, "max": 5}}
    result = detector.build_dashvector_filter(constraints)
    assert "room_type = '花园大床房'" in result
    assert "score >= 4" in result
    assert "score <= 5" in result

    # 出行类型 + 评分
    constraints = {"room_type": None, "fuzzy_room_type": None,
                   "travel_type": "商务出差", "score_range": {"min": 4, "max": 5}}
    result = detector.build_dashvector_filter(constraints)
    assert "travel_type = '商务出差'" in result
    assert "score >= 4" in result

    # 空约束
    constraints = {"room_type": None, "fuzzy_room_type": None,
                   "travel_type": None, "score_range": None}
    assert detector.build_dashvector_filter(constraints) is None


def test_mock_detect():
    """测试模拟 LLM 的完整检测流程"""
    mock_responses = {
        "商务出差且评分高的住客对早餐评价如何": json.dumps({
            "room_type": None,
            "fuzzy_room_type": None,
            "time_sensitivity": None,
            "score_range": {"min": 4, "max": 5},
            "travel_type": "商务出差",
            "categories": ["餐饮设施"],
            "date_range": None,
            "dashvector_filter": "travel_type = '商务出差' and score >= 4",
            "bm25_filter_keywords": ["商务", "出差", "早餐", "餐饮", "好评"]
        }, ensure_ascii=False),
        "最近花园大床房的评价怎么样": json.dumps({
            "room_type": "花园大床房",
            "fuzzy_room_type": None,
            "time_sensitivity": "clear",
            "score_range": None,
            "travel_type": None,
            "categories": None,
            "date_range": None,
            "dashvector_filter": "room_type = '花园大床房'",
            "bm25_filter_keywords": ["花园大床房", "最近", "评价"]
        }, ensure_ascii=False),
    }

    client = MockLLMClient(responses=mock_responses)
    detector = ConstraintDetector(client, EXACT_ROOM_TYPES, FUZZY_ROOM_TYPES)

    # 测试复合约束
    result = detector.detect("商务出差且评分高的住客对早餐评价如何")
    assert result["travel_type"] == "商务出差"
    assert result["score_range"] == {"min": 4, "max": 5}
    assert result["categories"] == ["餐饮设施"]
    assert "travel_type" in result["dashvector_filter"]
    assert len(result["bm25_filter_keywords"]) > 0

    # 测试房型 + 时效性
    result = detector.detect("最近花园大床房的评价怎么样")
    assert result["room_type"] == "花园大床房"
    assert result["time_sensitivity"] == "clear"


def test_bm25_keywords_limit():
    """测试 BM25 关键词数量限制"""
    detector = ConstraintDetector(MockLLMClient(), EXACT_ROOM_TYPES, FUZZY_ROOM_TYPES)

    data = {"room_type": None, "fuzzy_room_type": None, "time_sensitivity": None,
            "score_range": None, "travel_type": None, "categories": None,
            "date_range": None, "dashvector_filter": None,
            "bm25_filter_keywords": ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10"]}
    result = detector._validate_and_clean(data)
    assert len(result["bm25_filter_keywords"]) <= 8


if __name__ == "__main__":
    # 运行所有测试
    tests = [
        ("test_default_result", test_default_result),
        ("test_validate_room_type", test_validate_room_type),
        ("test_validate_score_range", test_validate_score_range),
        ("test_validate_travel_type", test_validate_travel_type),
        ("test_validate_categories", test_validate_categories),
        ("test_validate_time_sensitivity", test_validate_time_sensitivity),
        ("test_build_dashvector_filter", test_build_dashvector_filter),
        ("test_mock_detect", test_mock_detect),
        ("test_bm25_keywords_limit", test_bm25_keywords_limit),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"结果: {passed} passed, {failed} failed, {len(tests)} total")
