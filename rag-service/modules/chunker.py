"""
评论文本分块（Chunking）模块。

策略：句切（中文标点为主） + 长 chunk 滑窗合并 + 短 chunk 与相邻合并。
仅处理纯文本，不引入额外依赖。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# 中文 + 英文 句末标点 + 换行作为切分点
_SENT_SPLIT_RE = re.compile(r"([。！？；…!?\.\n]+)")

# chunk 控制参数（可在外部覆盖）
DEFAULT_MIN_LEN = 8       # 短句下限：低于此长度合并到相邻
DEFAULT_MAX_LEN = 120     # 长 chunk 上限：超过此长度做滑窗
DEFAULT_OVERLAP = 30      # 滑窗重叠


@dataclass
class Chunk:
    chunk_id: str
    comment_id: str
    seq: int
    text: str
    char_start: int
    char_end: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "comment_id": self.comment_id,
            "seq": self.seq,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


def _raw_sentences(comment: str) -> list[tuple[str, int, int]]:
    """按标点切句，返回 (句子, start, end)；保留标点附在前一句末尾。"""
    if not comment:
        return []
    parts = _SENT_SPLIT_RE.split(comment)
    sents: list[tuple[str, int, int]] = []
    cursor = 0
    buf = ""
    buf_start = 0
    for piece in parts:
        if not piece:
            continue
        if buf == "":
            buf_start = cursor
        buf += piece
        cursor += len(piece)
        # 标点段（_SENT_SPLIT_RE 捕获组）作为切分触发
        if _SENT_SPLIT_RE.fullmatch(piece):
            text = buf.strip()
            if text:
                sents.append((text, buf_start, buf_start + len(buf)))
            buf = ""
    if buf.strip():
        sents.append((buf.strip(), buf_start, buf_start + len(buf)))
    return sents


def _merge_short(sents: list[tuple[str, int, int]], min_len: int) -> list[tuple[str, int, int]]:
    """把过短的句子合并到相邻句子。"""
    if not sents:
        return []
    merged: list[tuple[str, int, int]] = []
    for s, st, ed in sents:
        if merged and (len(s) < min_len or len(merged[-1][0]) < min_len):
            ps, pst, ped = merged[-1]
            merged[-1] = (ps + s, pst, ed)
        else:
            merged.append((s, st, ed))
    return merged


def _slide_long(sent: tuple[str, int, int], max_len: int, overlap: int) -> list[tuple[str, int, int]]:
    """把过长的句子按滑窗切。"""
    text, start, end = sent
    if len(text) <= max_len:
        return [sent]
    out: list[tuple[str, int, int]] = []
    step = max(1, max_len - overlap)
    i = 0
    while i < len(text):
        seg = text[i: i + max_len]
        out.append((seg, start + i, start + i + len(seg)))
        if i + max_len >= len(text):
            break
        i += step
    return out


def split_review(
    comment: str,
    comment_id: str,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """把单条评论切成 chunk 列表。"""
    if not comment or not str(comment).strip():
        return []
    sents = _raw_sentences(str(comment))
    sents = _merge_short(sents, min_len=min_len)

    final: list[tuple[str, int, int]] = []
    for s in sents:
        final.extend(_slide_long(s, max_len=max_len, overlap=overlap))

    chunks: list[Chunk] = []
    for seq, (text, st, ed) in enumerate(final):
        chunks.append(
            Chunk(
                chunk_id=f"{comment_id}::{seq}",
                comment_id=str(comment_id),
                seq=seq,
                text=text,
                char_start=st,
                char_end=ed,
            )
        )
    return chunks


def chunk_dataframe(df, id_col: str = "_id", text_col: str = "comment") -> list[Chunk]:
    """对 DataFrame 整体切分。"""
    chunks: list[Chunk] = []
    for _, row in df.iterrows():
        chunks.extend(
            split_review(comment=row[text_col], comment_id=str(row[id_col]))
        )
    return chunks


if __name__ == "__main__":
    demo = (
        "房间非常好 装修很厚重奢华。一开始看评论 看酒店自己po的照片 感觉跟快捷酒店一样！"
        "退房时候liren还帮忙直接登记了退房 不用排队了。\n"
        "另外吃的真的很好，晚上去楼顶旋转自助吃 螃蟹腿鲍鱼进口生蚝 吃撑了 还有最爱吃的金蒜牛肉。"
    )
    for c in split_review(demo, comment_id="demo001"):
        print(c)
