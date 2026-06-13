"""意图处理模块：识别、检测、扩展、HyDE 生成、约束检测"""

import json
import time
from typing import Optional
from dashscope import Generation

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


class IntentRecognizer:
    """意图识别器：判断问题是否需要检索知识库"""

    def __init__(self, api_key: str, model: str = "qwen-flash"):
        self.api_key = api_key
        self.model = model

    def recognize(self, query: str, **kwargs) -> str:
        """识别用户意图，返回 True 表示需要检索"""
        system_prompt = """你是广州花园酒店的意图分类器。根据用户的问题，判断是否需要检索酒店评论知识库。

分类规则：
- RETRIEVAL：问题涉及酒店的设施、服务、房间、位置、餐饮、价格、体验等具体信息，需要检索评论才能回答
- DIRECT：问候、闲聊、常识性问题等，不涉及该酒店的具体信息，可以直接回答

只回复 RETRIEVAL 或 DIRECT，不要输出任何其他内容。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        response = Generation.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            result_format="message"
        )

        if response.status_code == 200:
            intent = response.output.choices[0].message.content.strip()
            return intent == "RETRIEVAL"
        else:
            raise RuntimeError(f"意图识别失败: {response.message}")


class IntentDetector:
    """意图检测器：提取房型约束与时效性需求"""

    def __init__(self, llm_client, exact_room_types: list, fuzzy_room_types: list):
        self.llm_client = llm_client
        self.exact_room_types = exact_room_types
        self.fuzzy_room_types = fuzzy_room_types

    def detect(self, query: str) -> dict:
        """
        检测用户意图

        返回:
            {
                "room_type": "花园大床房" | ... | None,
                "fuzzy_room_type": "大床房" | ... | None,
                "time_sensitivity": "clear" | "implied" | None
            }
        """
        prompt = f"""
你是一个酒店智能客服助手，需要分析用户查询并提取关键信息。

【任务】
从用户查询中提取以下信息：
1. 房型约束：用户是否提到特定房型
2. 时效性需求：用户是否关注最新信息

【精确房型列表】
{json.dumps(self.exact_room_types, ensure_ascii=False)}

【模糊房型列表】
{json.dumps(self.fuzzy_room_types, ensure_ascii=False)}

【房型检测规则】
- 优先检测精确房型，如检测到则填入 room_type，若模棱两可或只能检测到模糊房型则视为未检测到，填入 None。填入的内容只能是【精确房型列表】中的房型名称或 None
- 如未检测到精确房型，尝试检测模糊房型，如检测到则填入 fuzzy_room_type，若模棱两可则视为未检测到，填入 None。填入的内容只能是【模糊房型列表】中的房型名称或 None
- 如都未检测到，两者均为 None

【时效性判断标准】
- clear: 用户明确提到"最近"、"今年"、"最新"、"现在"等词汇
- implied: 用户隐含关注当前现状，但未明确表达，表现弱时效性
- None: 用户未表现出时效性关注

【用户查询】
{query}

【输出格式】
严格以 JSON 格式输出：
{{
    "room_type": "花园大床房" 或 None,
    "fuzzy_room_type": "大床房" 或 None,
    "time_sensitivity": "clear" 或 "implied" 或 None
}}
"""

        for i in range(2):
            try:
                response = self.llm_client.generate(prompt, temperature=0.1)
                response = response.replace('```json', '').replace('```', '').strip()
                data = json.loads(response)
                if data['room_type'] and data['room_type'] not in self.exact_room_types:
                    data['room_type'] = None
                if data['fuzzy_room_type'] and data['fuzzy_room_type'] not in self.fuzzy_room_types:
                    data['fuzzy_room_type'] = None
                if data['time_sensitivity'] and data['time_sensitivity'] not in ['clear', 'implied']:
                    data['time_sensitivity'] = None
                return data
            except Exception as e:
                print(f"意图检测第 {i+1} 次尝试失败: {e}")
                if i < 1:
                    time.sleep(0.1)
                    continue

        print("意图检测失败，已返回全 None 字典")
        return {
            "room_type": None,
            "fuzzy_room_type": None,
            "time_sensitivity": None
        }


class IntentExpander:
    """意图扩展器：改写 Query 并计算权重"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def expand(self, query: str) -> dict:
        """
        扩展用户意图

        返回:
            [
                {"query": "改写的查询1", "weight": 0.6},
                {"query": "改写的查询2", "weight": 0.2},
                {"query": "改写的查询3", "weight": 0.2}
            ]
        """
        prompt = f"""
你是一个酒店智能客服助手，需要深度理解用户查询意图。

【任务】
1. 分析用户查询，检测用户的核心关注点
2. 生成1-3个改写后的查询，每个查询更清晰、更具体地表达一个关注点
3. 为每个改写查询分配权重，表示该关注点的重要性（权重之和为1，且只允许使用0.2的倍数，即0.2,0.4,0.6,0.8,1.0）

【用户查询】
{query}

【要求】
- 改写的查询应该比原查询更具体、更明确
- 每个改写查询应该聚焦一个具体方面
- 权重应该反映该方面在原查询中的重要性
- 对于模糊的查询，使用尽可能多的改写来覆盖更大范围的意图；对于明确的查询，不要对其过度展开

【输出格式】
严格以 JSON 格式输出：
{{
    "rewritten_queries": [
        {{"query": "酒店交通是否便利？", "weight": 0.6}},
        {{"query": "酒店周边有哪些配套设施？", "weight": 0.2}},
        {{"query": "酒店的服务效率如何？", "weight": 0.2}}
    ]
}}

【注意】
- rewritten_queries 数组长度为1-3
- 所有 weight 之和必须等于1，且只允许使用0.2的倍数
"""

        for i in range(2):
            try:
                response = self.llm_client.generate(prompt, temperature=0.3)
                response = response.replace('```json', '').replace('```', '').strip()
                data = json.loads(response)
                queries = data['rewritten_queries']
                if isinstance(queries, list):
                    for item in queries:
                        item['query'] = item['query']
                        item['weight'] = float(item['weight'])
                    return queries
                else:
                    raise TypeError(f"queries 数据类型错误: 期望 list, 实际为 {type(queries).__name__}")
            except Exception as e:
                print(f"意图扩展第 {i+1} 次尝试失败: {e}")
                if i < 1:
                    time.sleep(0.1)
                    continue

        print("意图扩展失败，已返回 None")
        return None


class HyDEGenerator:
    """假设性回复生成器：为单个 Query 生成假设回复用于增强检索"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate(self, query: str) -> list[str]:
        """
        为单个查询生成假设性回复

        策略：生成2条正面回复 + 1条负面回复
        """
        prompt = f"""
你是一个酒店评论撰写者，需要为以下查询生成假设性的评论回复。

【查询】
{query}

【任务】
针对上述查询，生成3条假设性的酒店评论：
- 2条正面评论：积极评价酒店相关方面
- 1条负面评论：指出可能存在的不足

【要求】
- 每条评论50-100字
- 评论要具体、真实，包含细节
- 评论风格要像真实用户写的
- 尽量增大3条评论之间的差异性

【输出格式】
严格以 JSON 格式输出：
{{
    "hypothetical_responses": [
        "正面评论1",
        "正面评论2",
        "负面评论"
    ]
}}
"""

        for i in range(2):
            try:
                response = self.llm_client.generate(prompt, temperature=0.7)
                response = response.replace('```json', '').replace('```', '').strip()
                data = json.loads(response)
                responses = data['hypothetical_responses']
                if isinstance(responses, list):
                    return responses
                else:
                    raise TypeError(f"responses 数据类型错误: 期望 list, 实际为 {type(responses).__name__}")
            except Exception as e:
                print(f"假设性回复生成第 {i+1} 次尝试失败: {e}")
                if i < 1:
                    time.sleep(0.1)
                    continue

        print("假设性回复生成失败，已返回原查询")
        return [query]


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
        """检测用户查询中的多维约束"""
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
- 优先检测精确房型，如检测到则填入 room_type
- 如未检测到精确房型，尝试检测模糊房型
- 如都未检测到，两者均为 None

【时效性判断标准】
- clear: 用户明确提到"最近"、"今年"、"最新"、"现在"等词汇
- implied: 用户隐含关注当前现状
- None: 用户未表现出时效性关注

【DashVector 过滤表达式生成规则】
- 根据检测到的约束，生成 DashVector 兼容的过滤表达式
- 支持的字段：room_type, fuzzy_room_type, score, travel_type
- 格式示例：
  - 仅房型: "room_type = '花园大床房'"
  - 房型+评分: "room_type = '花园大床房' and score >= 4"
  - 出行类型+评分: "travel_type = '商务出差' and score >= 4"
  - 无约束: null

【BM25 过滤关键词生成规则】
- 从用户查询和检测到的约束中，提取 3-8 个最能代表检索意图的关键词
- 关键词应覆盖：话题类别相关词、出行类型相关词、评分倾向相关词

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

        ts = data.get("time_sensitivity")
        result["time_sensitivity"] = ts if ts in ("clear", "implied") else None

        sr = data.get("score_range")
        if isinstance(sr, dict) and "min" in sr and "max" in sr:
            result["score_range"] = {
                "min": max(1, min(5, int(sr["min"]))),
                "max": max(1, min(5, int(sr["max"]))),
            }
        else:
            result["score_range"] = None

        tt = data.get("travel_type")
        result["travel_type"] = tt if tt in TRAVEL_TYPES else None

        cats = data.get("categories")
        if isinstance(cats, list):
            result["categories"] = [c for c in cats if c in CATEGORIES] or None
        else:
            result["categories"] = None

        dr = data.get("date_range")
        if isinstance(dr, dict) and "start" in dr and "end" in dr:
            result["date_range"] = {"start": str(dr["start"]), "end": str(dr["end"])}
        else:
            result["date_range"] = None

        df_filter = data.get("dashvector_filter")
        result["dashvector_filter"] = (
            df_filter if isinstance(df_filter, str) and df_filter.strip() else None
        )

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
        """根据约束字典构建 DashVector 过滤表达式（程序化兜底）"""
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
