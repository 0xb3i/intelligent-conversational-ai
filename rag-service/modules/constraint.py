"""约束检测模块：从用户 Query 中提取多维过滤条件，生成结构化过滤语句"""

import json
import time
from typing import Optional


# 出行类型列表
TRAVEL_TYPES = ["商务出差", "家庭亲子", "情侣出游", "朋友出游", "独自旅行"]

# 14 个话题类别
CATEGORIES = [
    "房间设施", "公共设施", "餐饮设施",
    "前台服务", "客房服务", "退房/入住效率",
    "交通便利性", "周边配套", "景观/朝向",
    "性价比", "价格合理性",
    "整体满意度", "安静程度", "卫生状况"
]


class ConstraintDetector:
    """约束检测器：从用户 Query 中提取多维过滤条件

    扩展 IntentDetector，新增：
    - score_range: 评分约束
    - travel_type: 出行类型
    - categories: 话题类别
    - date_range: 时间范围
    - dashvector_filter: DashVector 兼容过滤表达式
    - bm25_filter_keywords: BM25 关键词约束
    """

    def __init__(self, llm_client, exact_room_types: list, fuzzy_room_types: list):
        self.llm_client = llm_client
        self.exact_room_types = exact_room_types
        self.fuzzy_room_types = fuzzy_room_types

    def detect(self, query: str) -> dict:
        """
        检测用户查询中的多维约束

        返回:
            {
                "room_type": str | None,
                "fuzzy_room_type": str | None,
                "time_sensitivity": "clear" | "implied" | None,
                "score_range": {"min": int, "max": int} | None,
                "travel_type": str | None,
                "categories": [str] | None,
                "date_range": {"start": str, "end": str} | None,
                "dashvector_filter": str | None,
                "bm25_filter_keywords": [str]
            }
        """
        prompt = f"""
你是一个酒店智能客服助手，需要从用户查询中提取检索过滤条件。

【任务】从用户查询中提取以下信息：

1. 房型约束：用户是否提到特定房型
2. 时效性需求：用户是否关注最新信息
3. 评分约束：用户是否暗示了评分偏好
   - "好评"/"推荐"/"满意"/"不错" → 4-5分
   - "差评"/"不好"/"糟糕"/"失望" → 1-2分
   - "中评"/"一般" → 3分
   - 未提及 → None
4. 出行类型：用户是否指定了出行目的
   - 可选值：{json.dumps(TRAVEL_TYPES, ensure_ascii=False)}
   - 未提及 → None
5. 话题类别：用户关注的具体方面
   - 可选值：{json.dumps(CATEGORIES, ensure_ascii=False)}
   - 未提及 → None
6. 时间范围：用户是否指定了时间范围
   - 如"去年"/"2024年" → 对应日期范围
   - 未提及 → None

【精确房型列表】
{json.dumps(self.exact_room_types, ensure_ascii=False)}

【模糊房型列表】
{json.dumps(self.fuzzy_room_types, ensure_ascii=False)}

【房型检测规则】
- 优先检测精确房型，如检测到则填入 room_type，若模棱两可或只能检测到模糊房型则视为未检测到，填入 None
- 如未检测到精确房型，尝试检测模糊房型，如检测到则填入 fuzzy_room_type，若模棱两可则视为未检测到，填入 None

【时效性判断标准】
- clear: 用户明确提到"最近"、"今年"、"最新"、"现在"等词汇
- implied: 用户隐含关注当前现状，但未明确表达
- None: 用户未表现出时效性关注

【DashVector 过滤表达式生成规则】
- 根据检测到的约束，生成 DashVector 兼容的过滤表达式
- 支持的字段：room_type, fuzzy_room_type, score, travel_type
- 格式示例：
  - 仅房型: "room_type = '花园大床房'"
  - 房型+评分: "room_type = '花园大床房' and score >= 4"
  - 出行类型+评分: "travel_type = '商务出差' and score >= 4"
  - 模糊房型: "fuzzy_room_type = '大床房'"
  - 无约束: null
- 注意：字符串值用单引号包裹，and 连接多个条件

【BM25 过滤关键词生成规则】
- 从用户查询和检测到的约束中，提取 3-8 个最能代表检索意图的关键词
- 关键词应覆盖：话题类别相关词、出行类型相关词、评分倾向相关词
- 例如查询"商务出差且评分高的住客对早餐评价如何？" → ["商务", "出差", "早餐", "餐饮", "好评", "推荐"]

【用户查询】
{query}

【输出格式】
严格以 JSON 格式输出：
{{
    "room_type": "花园大床房" 或 null,
    "fuzzy_room_type": "大床房" 或 null,
    "time_sensitivity": "clear" 或 "implied" 或 null,
    "score_range": {{"min": 4, "max": 5}} 或 null,
    "travel_type": "商务出差" 或 null,
    "categories": ["餐饮设施", "前台服务"] 或 null,
    "date_range": {{"start": "2024-01-01", "end": "2024-12-31"}} 或 null,
    "dashvector_filter": "room_type = '花园大床房' and score >= 4" 或 null,
    "bm25_filter_keywords": ["早餐", "餐饮", "商务"]
}}
"""

        for i in range(2):
            try:
                response = self.llm_client.generate(prompt, temperature=0.1)
                response = response.replace("```json", "").replace("```", "").strip()
                data = json.loads(response)

                # 校验并清理结果
                result = self._validate_and_clean(data)
                return result
            except Exception as e:
                print(f"约束检测第 {i+1} 次尝试失败: {e}")
                if i < 1:
                    time.sleep(0.1)
                    continue

        print("约束检测失败，已返回默认值")
        return self._default_result()

    def _validate_and_clean(self, data: dict) -> dict:
        """校验并清理 LLM 返回的约束数据"""
        result = {}

        # 房型约束
        result["room_type"] = (
            data.get("room_type")
            if data.get("room_type") in self.exact_room_types
            else None
        )
        result["fuzzy_room_type"] = (
            data.get("fuzzy_room_type")
            if data.get("fuzzy_room_type") in self.fuzzy_room_types
            else None
        )

        # 时效性
        ts = data.get("time_sensitivity")
        result["time_sensitivity"] = ts if ts in ("clear", "implied") else None

        # 评分约束
        sr = data.get("score_range")
        if isinstance(sr, dict) and "min" in sr and "max" in sr:
            result["score_range"] = {
                "min": max(1, min(5, int(sr["min"]))),
                "max": max(1, min(5, int(sr["max"]))),
            }
        else:
            result["score_range"] = None

        # 出行类型
        tt = data.get("travel_type")
        result["travel_type"] = tt if tt in TRAVEL_TYPES else None

        # 话题类别
        cats = data.get("categories")
        if isinstance(cats, list):
            result["categories"] = [c for c in cats if c in CATEGORIES] or None
        else:
            result["categories"] = None

        # 时间范围
        dr = data.get("date_range")
        if isinstance(dr, dict) and "start" in dr and "end" in dr:
            result["date_range"] = {"start": str(dr["start"]), "end": str(dr["end"])}
        else:
            result["date_range"] = None

        # DashVector 过滤表达式
        df_filter = data.get("dashvector_filter")
        result["dashvector_filter"] = df_filter if isinstance(df_filter, str) and df_filter.strip() else None

        # BM25 过滤关键词
        bm25_kws = data.get("bm25_filter_keywords")
        if isinstance(bm25_kws, list) and len(bm25_kws) > 0:
            result["bm25_filter_keywords"] = [str(kw) for kw in bm25_kws[:8]]
        else:
            result["bm25_filter_keywords"] = []

        return result

    def _default_result(self) -> dict:
        """返回默认的空约束"""
        return {
            "room_type": None,
            "fuzzy_room_type": None,
            "time_sensitivity": None,
            "score_range": None,
            "travel_type": None,
            "categories": None,
            "date_range": None,
            "dashvector_filter": None,
            "bm25_filter_keywords": [],
        }

    def build_dashvector_filter(self, constraints: dict) -> Optional[str]:
        """根据约束字典构建 DashVector 过滤表达式（程序化构建，不依赖 LLM 输出）

        作为 LLM 生成 dashvector_filter 的兜底方案。
        """
        parts = []

        if constraints.get("room_type"):
            parts.append(f"room_type = '{constraints['room_type']}'")
        elif constraints.get("fuzzy_room_type"):
            parts.append(f"fuzzy_room_type = '{constraints['fuzzy_room_type']}'")

        if constraints.get("travel_type"):
            parts.append(f"travel_type = '{constraints['travel_type']}'")

        sr = constraints.get("score_range")
        if sr:
            parts.append(f"score >= {sr['min']}")
            parts.append(f"score <= {sr['max']}")

        return " and ".join(parts) if parts else None
