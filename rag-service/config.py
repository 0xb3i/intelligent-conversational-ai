"""常量配置"""

from datetime import datetime
from pathlib import Path

# 时间基准常量
TODAY = datetime(2025, 4, 17)

# ── 路径 ────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
DICT_DIR = DATA_DIR / "dict"

# ── BM25 倒排索引相关 ───────────────────────────────────────
# 以下参数同时影响离线 build（build_chunked_index.py）与在线 search。
# 修改 BM25_K1 / BM25_B / 词典 / KEEP_DIGITS 后，必须重建 inverted_index_chunked.pkl，
# 否则索引词项与 query 分词不一致会导致召回崩塌。
# 仅 SYNONYMS / USE_QUERY_EXPANSION 是纯查询侧扩展，可热改不需重建。
BM25_K1 = 1.5                    # 词频饱和度，常见 1.2~2.0
BM25_B = 0.75                    # 长度归一化强度，常见 0.5~0.8
BM25_KEEP_DIGITS = True          # tokenize 是否保留数字（"500元"/"5G"/"2人间"）
BM25_USE_QUERY_EXPANSION = True  # search 时是否做同义词 OR 扩展
BM25_HOTEL_DICT_PATH = str(DICT_DIR / "hotel_dict.txt")
BM25_SYNONYMS_PATH = str(DICT_DIR / "synonyms.yaml")

# 精确房型列表
EXACT_ROOM_TYPES = [
    '花园大床房', '花园双床房', '红棉大床套房', '红棉双床套房',
    '城央绿意大床房', '城央绿意双床房', '粤韵大床套房', '粤韵双床套房',
    '花园行政大床套房', '花园行政双床套房', '羊羊得意主题大床房',
    '羊羊得意主题大床套房', '大嘴猴亲子主题大床房',
    '盼酷小黄鸭亲子主题大床房', '盼酷小黄鸭亲子主题套房'
]

# 模糊房型列表
FUZZY_ROOM_TYPES = ['大床房', '双床房', '套房', '主题房']
