"""基于 BM25 的倒排索引

支持的扩展（v2）：
- 自定义词典：jieba.load_userdict(hotel_dict_path) 优先识别房型/设施等复合词
- 数字保留：KEEP_DIGITS=True 时不再过滤纯数字与数字+单位 token
- 查询时同义词扩展：search 时把 query_tokens 经 synonym map 做 OR 扩展，
  扩展只影响"参与 BM25 计分的词项"，不改 BM25 公式本身。
  扩展后的同义词 token 不带新权重；它们各自走原 IDF/TF。
"""

import re
import math
import nltk
import jieba
import pickle
from pathlib import Path
from collections import Counter

# 仅在 IPython/Jupyter 环境用 notebook 版，否则用普通 tqdm
def _select_tqdm():
    try:
        from IPython import get_ipython  # type: ignore
        if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
            from tqdm.notebook import tqdm as _t
            return _t
    except Exception:
        pass
    from tqdm import tqdm as _t
    return _t

tqdm = _select_tqdm()


# ── 工具：加载同义词组 ──────────────────────────────────────
def _load_synonyms(path: str | None) -> dict[str, set[str]]:
    """读取 synonyms.yaml，返回 {token: {同组的所有 token}}（含自身）。
    解析失败、文件不存在、PyYAML 缺失时返回空 dict（静默降级）。"""
    if not path or not Path(path).exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        print("[index] PyYAML 未安装，跳过同义词扩展（pip install pyyaml）")
        return {}
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception as e:
        print(f"[index] synonyms 解析失败，跳过：{e}")
        return {}

    syn_map: dict[str, set[str]] = {}
    for group in data.get("groups", []) or []:
        if not isinstance(group, list) or len(group) < 2:
            continue
        # 对组内每个词项，统一成 lower-case 与全角转半角后的小写形式
        members = {str(w).strip().lower() for w in group if str(w).strip()}
        if len(members) < 2:
            continue
        for w in members:
            syn_map.setdefault(w, set()).update(members)
    return syn_map


class InvertedIndex:
    """基于 BM25 的倒排索引"""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords_file: str | None = None,
        hotel_dict_file: str | None = None,
        synonyms_file: str | None = None,
        keep_digits: bool = True,
    ):
        """
        参数:
            k1: BM25 参数，控制词频饱和度
            b: BM25 参数，控制文档长度归一化程度
            stopwords_file: 中文停用词表，附加 nltk 英文停用词
            hotel_dict_file: 领域自定义词典（jieba.load_userdict 格式）
            synonyms_file: 同义词组 yaml；查询侧 OR 扩展用，不影响索引构建
            keep_digits: 是否保留数字 token（如 "500" / "2人间" 中的 "2"）
        """
        self.k1 = k1
        self.b = b
        self.keep_digits = keep_digits
        self.index: dict[str, dict[str, int]] = {}    # {term: {doc_id: tf}}
        self.doc_lengths: dict[str, int] = {}         # {doc_id: doc_length}
        self.avg_doc_length = 0.0
        self.num_docs = 0
        self.documents: dict[str, str] = {}           # {doc_id: text}
        # 配置文件路径在保存索引时一并落盘，加载后可被覆盖（便于热改同义词）
        self.hotel_dict_file = hotel_dict_file
        self.synonyms_file = synonyms_file

        # 加载停用词
        self.stopwords: set[str] = set()
        if stopwords_file and Path(stopwords_file).exists():
            with open(stopwords_file, encoding='utf-8') as f:
                self.stopwords.update([line.strip() for line in f])
            try:
                self.stopwords.update(nltk.corpus.stopwords.words('english'))
            except Exception:
                print("警告: 未能加载 NLTK 英文停用词")

        # 加载领域词典（影响 jieba 全局；不重复加载同一文件）
        if hotel_dict_file and Path(hotel_dict_file).exists():
            jieba.load_userdict(hotel_dict_file)
            print(f"[index] 已加载领域词典: {hotel_dict_file}")

        # 加载同义词
        self._synonyms = _load_synonyms(synonyms_file)
        if self._synonyms:
            print(f"[index] 已加载同义词组: {len(self._synonyms)} token "
                  f"覆盖")

        # 字典预加载
        jieba.initialize()

        # token 过滤正则
        # - 保留中英文
        # - keep_digits=True 时也保留数字
        if self.keep_digits:
            self._allowed_re = re.compile(r'^[\u4e00-\u9fffa-zA-Z0-9]+$')
        else:
            self._allowed_re = re.compile(r'^[\u4e00-\u9fffa-zA-Z]+$')

    def tokenize(self, text: str) -> list[str]:
        """分词与过滤。"""
        text = re.sub(r'\s+', '', str(text))
        tokens_raw = jieba.lcut(text)
        out: list[str] = []
        for tok in tokens_raw:
            t = tok.lower()
            if not t or t in self.stopwords:
                continue
            if not self._allowed_re.match(t):
                continue
            # 单字符纯数字也保留（"5G"/"3楼"切出的"5""3"），但单字符纯英文不丢
            out.append(t)
        return out

    # ── query 侧同义词扩展 ────────────────────────────────────
    def expand_query_tokens(self, tokens: list[str]) -> list[str]:
        """OR 扩展：每个 token 若命中同义词组，就把整组都加进来；否则保留原样。
        - 只加进 self.index 中存在的词（避免引入永远命中不到的噪声）
        - 扩展后做去重，但保留原 query 词序在前
        """
        if not self._synonyms:
            return tokens
        seen: set[str] = set()
        expanded: list[str] = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                expanded.append(t)
            for syn in self._synonyms.get(t, ()):  # type: ignore[arg-type]
                if syn in seen:
                    continue
                if syn not in self.index:
                    continue  # 不在索引词项中，扩展也无意义
                seen.add(syn)
                expanded.append(syn)
        return expanded

    def build(self, documents: dict[str, str]):
        """构建倒排索引。
        参数:
            documents: {doc_id: document_text}
        """
        self.documents = documents
        self.num_docs = len(documents)

        total_length = 0
        for doc_id, text in tqdm(documents.items(), desc="分词与统计"):
            tokens = self.tokenize(text)
            doc_length = len(tokens)
            self.doc_lengths[doc_id] = doc_length
            total_length += doc_length

            term_freq = Counter(tokens)
            for term, freq in term_freq.items():
                if term not in self.index:
                    self.index[term] = {}
                self.index[term][doc_id] = freq

        self.avg_doc_length = total_length / self.num_docs if self.num_docs > 0 else 0
        print(f"倒排索引构建完成: {len(self.index)} 个词项, {self.num_docs} 篇文档")
        print(f"平均文档长度: {self.avg_doc_length:.2f} 个词")

    def search(
        self,
        query: str,
        topk: int = 10,
        use_query_expansion: bool = True,
    ) -> list[tuple[str, float]]:
        """
        BM25 检索

        参数:
            query: 查询文本
            topk: 返回 Top-K 结果
            use_query_expansion: 是否对 query tokens 做同义词扩展（None=按构造默认）
        返回:
            [(doc_id, bm25_score), ...]
        """
        query_tokens = self.tokenize(query)
        if use_query_expansion:
            query_tokens = self.expand_query_tokens(query_tokens)

        if not query_tokens:
            return []

        # 计算 IDF
        idf: dict[str, float] = {}
        for term in query_tokens:
            if term in self.index:
                df = len(self.index[term])
                idf[term] = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

        # 计算 BM25 分数
        scores: dict[str, float] = {}
        for term in query_tokens:
            if term not in self.index:
                continue
            for doc_id, tf in self.index[term].items():
                doc_length = self.doc_lengths[doc_id]
                norm_factor = 1 - self.b + self.b * (doc_length / self.avg_doc_length)
                term_score = idf[term] * (tf * (self.k1 + 1)) / (tf + self.k1 * norm_factor)
                scores[doc_id] = scores.get(doc_id, 0.0) + term_score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        return sorted_docs

    def save(self, filepath: str):
        """保存索引到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'doc_lengths': self.doc_lengths,
                'avg_doc_length': self.avg_doc_length,
                'num_docs': self.num_docs,
                'documents': self.documents,
                'k1': self.k1,
                'b': self.b,
                'stopwords': self.stopwords,
                'keep_digits': self.keep_digits,
                'hotel_dict_file': self.hotel_dict_file,
                'synonyms_file': self.synonyms_file,
            }, f)
        print(f"倒排索引已保存: {filepath}")

    def load(self, filepath: str):
        """从文件加载索引。
        加载完成后会按需加载 hotel_dict + synonyms（供查询使用）。"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.index = data['index']
            self.doc_lengths = data['doc_lengths']
            self.avg_doc_length = data['avg_doc_length']
            self.num_docs = data['num_docs']
            self.documents = data['documents']
            self.k1 = data['k1']
            self.b = data['b']
            self.stopwords = data.get('stopwords', set())
            self.keep_digits = data.get('keep_digits', False)
            self.hotel_dict_file = data.get('hotel_dict_file', None)
            self.synonyms_file = data.get('synonyms_file', None)

        # 与构造一致地激活领域词典 + 同义词
        if self.hotel_dict_file and Path(self.hotel_dict_file).exists():
            jieba.load_userdict(self.hotel_dict_file)
        self._synonyms = _load_synonyms(self.synonyms_file)
        if self.keep_digits:
            self._allowed_re = re.compile(r'^[\u4e00-\u9fffa-zA-Z0-9]+$')
        else:
            self._allowed_re = re.compile(r'^[\u4e00-\u9fffa-zA-Z]+$')
        print(
            f"倒排索引已加载: {len(self.index)} 词项, {self.num_docs} 文档"
            f"; keep_digits={self.keep_digits}"
            f"; synonyms={len(self._synonyms)}"
        )
