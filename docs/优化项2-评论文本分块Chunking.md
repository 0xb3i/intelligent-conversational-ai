# 优化项 2：评论文本分块（Chunking）与引用高亮

> 基于 [hotel-review-rag](https://github.com/Scorpioyyy/hotel-review-rag) 的 `002-rag-system` 分支

## 1. 背景与问题

当前实现以"整条评论作为一个检索单元"。具体来说：

- 知识库构建侧：评论嵌入与入库以整条 `comment` 为单位（见 `RAG/module_1_offline_knowledge_base.ipynb` "构建评论数据库" 段，约 1300 行）。
- 倒排索引侧：BM25 索引同样以整条评论为文档（`rag-service/modules/index.py::InvertedIndex.build`）。
- 检索结果侧：`HybridRetriever` 在打包结果时直接回填 `df_comments.loc[doc_id]['comment']`（`rag-service/modules/retriever.py::retrieve`）。
- 生成侧：`ResponseGenerator._build_prompt` 把整条评论作为一段引用拼进 Prompt（`rag-service/modules/generator.py::_build_prompt`）。

这种"整条评论 = 一个 chunk"在我们的数据集（广州花园酒店 2 千余条评论，平均长度数十字）上勉强可行，但存在三类已知缺陷：

1. **长评论稀释**：评论中常同时出现"早餐很差 / 房间隔音不错 / 前台态度好"等多个主题。整条作为一个向量后，向量被多个主题稀释，单一主题的 Query（如"早餐怎么样"）召回排名下降；BM25 分数也因 IDF 项被无关词拉低。
2. **生成端引用粗粒度**：模型在回答时只能整段引用 `[[ref:N]]`，但用户问的可能只是评论里某一句。前端显示也只能整条高亮，用户必须在长评论里再次"找句子"，与课程指导意见 (2) 中"在用户界面高亮参考评论的引用部分"的预期不符。
3. **跨主题召回偏置**：当 Query 与评论中"非主题部分"出现强词共现时，整条评论会被错误命中（典型：用户问"网络"，但评论主体在讲早餐，结尾随口提了一句 Wi-Fi）。

## 2. 优化目标

- 将"评论 → 一个文档"细化为"评论 → 多个 chunk"，并在不破坏原有 5 路召回与 RRF 融合管线的前提下，让最终引用粒度精确到 chunk 级别（一句或一个语义段）。
- 在前端 `[[ref:N]]` 引用展开时，同时返回 chunk 在原评论中的字符跨度，供 UI 做"原文高亮"。
- 对比"整条评论"基线，验证细粒度 chunk 是否带来检索/生成质量提升。

## 3. 设计方案

### 3.1 三种候选切分策略（保持简单、可消融）

| 策略 | 描述 | 实现成本 | 备注 |
|---|---|---|---|
| **A. 句子切分** | 以中文标点 `[。！？；…]` + 英文 `[.!?]` + 换行作为切分点；过短句（< 8 字）合并到相邻句 | 极低 | 默认采用 |
| **B. 滑窗切分** | 以 ~120 字为窗口、~30 字为重叠 | 低 | 长评论时启用，规避句切过碎 |
| **C. 主题切分** | 复用现有 `categories.json` + 关键词命中，把每句打上类别标签后，把同类相邻句合并为一个 chunk | 中 | 选做，效果上限更高 |

最终默认管线：**先按 A 切句 → 长评论再叠加 B 的滑窗合并 → 输出 chunk**。

### 3.2 Chunk 数据结构

```python
@dataclass
class Chunk:
    chunk_id: str        # f"{comment_id}::{seq}"
    comment_id: str      # 反向追溯到原评论（用于 metadata 保持不变）
    seq: int             # 在原评论内的顺序，0-based
    text: str            # chunk 文本
    char_span: tuple[int, int]  # 在原评论 comment 字段内的 [start, end)
```

### 3.3 索引侧改造

1. **离线构建**：在 `module_1_offline_knowledge_base.ipynb` 的"构建评论数据库"段之前，先把 `df_filtered['comment']` 切成 `chunks_df`（多行展开），所有评论级元数据沿用，新增 `chunk_id / comment_id / seq / char_start / char_end` 字段。
2. **向量库**：用 `chunk_id` 作为 DashVector 主键，向量来自 `chunk.text`，metadata 中保留 `comment_id`，以便回填整条评论上下文。
3. **倒排索引**：`InvertedIndex.build(documents)` 中 `documents` 改为 `{chunk_id: chunk.text}`；其余 BM25 公式无需修改。
4. **反向 Query / HyDE / 摘要库**：保持原粒度（不切，因为它们本身就是聚合表达），只在最终融合时通过 `comment_id ↔ chunk_id` 的反查表与评论库结果对齐。

### 3.4 检索侧改造

`HybridRetriever.retrieve` 内部新增 `comment_id` 维度的去重与"代表 chunk"选择：

- **RRF 融合保持以 chunk_id 为单位**进行融合，避免提前损失粒度。
- 融合后 `final_topk` 截断之后，按 `comment_id` 做二次去重：同一 `comment_id` 下保留 `rrf_score` 最高的 chunk 作为"代表 chunk"，其余 chunk 收敛进同一条结果的 `extra_chunks`。
- 返回结构兼容旧版：

```python
{
    "comment_id": "...",
    "comment": "...",                # 原评论全文（旧字段，前端兼容）
    "primary_chunk": {"text": "...", "char_span": [s, e]},
    "extra_chunks": [...],           # 同评论被命中的其他 chunk
    "rrf_score": ..., "rrf_rank": ...,
    "route_ranks": {...},
    "metadata": {...}                # 评论级元数据，原样保留
}
```

### 3.5 生成与引用侧改造

`ResponseGenerator._build_prompt` 中评论上下文从"整条评论"改为"chunk + 上下文窗口"：

```text
【评论1】
评分: 4
发布日期: 2024-08-12
房型: 高级大床房
评论关键句: "早餐种类不多，凉的多" ←(primary_chunk)
（同一评论补充内容）
- 房间还算干净
- 前台办理较慢
```

并把"引用规则"扩充成：

> 引用 [[ref:N]] 时，只引用该评论被检索到的"评论关键句"对应的内容；前端会自动在原评论中高亮该句。

前端 `CitationBadge` / 评论展开组件接到 API 返回的 `primary_chunk.char_span` 后，对原评论字符串做切片包裹 `<mark>` 即可实现"高亮参考评论的引用部分"，无需 LLM 输出位置信息。

## 4. 实施步骤（按 DDL 一日工作量裁剪）

1. **新增 chunker 模块**：`rag-service/modules/chunker.py`，实现 `split_review(comment) -> list[Chunk]`。仅做策略 A + 长文滑窗合并，<= 60 行。
2. **离线脚本改造**：在 `module_1_offline_knowledge_base.ipynb` 中插入一段：把 `df_filtered` 展开为 `chunks_df` 并保存到 `RAG/data/processed/chunks.csv`；后续向量入库 / BM25 全部读取 `chunks_df`。
3. **`InvertedIndex` 兼容**：无需改公式，只改入参 documents 的来源；保存路径区分 `inverted_index_chunked.pkl`，便于和旧版做 A/B。
4. **`HybridRetriever` 改造**：新增 `_collapse_by_comment` 私有方法（约 30 行），完成 chunk → comment 二次去重与代表 chunk 选择。
5. **生成端改造**：`_build_prompt` 内部对每条结果输出 `primary_chunk.text` 作为关键句，并保留评论全文（受 token 预算控制时，仅当评论长度 > 80 字才输出全文）。
6. **前端最小改动**：`src/components/comments/CommentCard.tsx` / `CitationBadge.tsx` 接收 `primary_chunk.char_span`，对原评论 substring 套 `<mark>` 高亮。
7. **回归与对比评估**：复用 `RAG/data/evaluation/eval_set.json` 跑一轮，分别产出 `responses_chunked.json` 与基线 `responses_full.json` 的对比。

## 5. 评估方案与实测结果

### 5.1 实施落地

实际落地的最小版本：

- 新增 [chunker.py](file:///Users/bytedance/Desktop/大模型/hotel-review-rag/rag-service/modules/chunker.py)：句切（中文/英文标点+换行）+ 短句合并（<8 字与相邻合并）+ 长句滑窗（>120 字、重叠 30），输出带 `char_span` 的 `Chunk`。
- 新增 [eval_chunking.py](file:///Users/bytedance/Desktop/大模型/hotel-review-rag/eval_chunking.py)：复用项目自带的 [InvertedIndex](file:///Users/bytedance/Desktop/大模型/hotel-review-rag/rag-service/modules/index.py)，分别构建"整条评论 BM25"基线索引与 "chunk BM25" 索引，在 `eval_set.json` 全量 100 个 Query 上做对比。
- 由于课程 DDL 限制，本轮只评估了 BM25 单路；向量路改造留作后续（数据结构已对齐，只需把 `chunks` 入 DashVector 即可）。
- **在线接入（本轮已补齐）**：
  - [build_chunked_index.py](file:///Users/bytedance/Desktop/大模型/hotel-review-rag/build_chunked_index.py) 离线产出 `rag-service/data/inverted_index_chunked.pkl` 与 `chunk_meta.pkl`（chunk_id → comment_id / seq / char_span / text 的反查表）。
  - [HybridRetriever](file:///Users/bytedance/Desktop/大模型/hotel-review-rag/rag-service/modules/retriever.py) 新增 `chunked_index / chunk_meta` 参数：BM25 路改为在 chunk 索引上检索 `topk × 5` 条，按 `comment_id` 折叠到原 topk，并在每条结果上挂 `primary_chunk = {chunk_id, text, char_start, char_end, seq}` 作为"评论关键句"。
  - [HotelReviewRAG](file:///Users/bytedance/Desktop/大模型/hotel-review-rag/rag-service/modules/rag_system.py) 启动时若检测到 `inverted_index_chunked.pkl + chunk_meta.pkl` 存在则自动装载并注入 retriever；缺失时自动回退到旧的整条评论 BM25。
  - [prompts.py](file:///Users/bytedance/Desktop/大模型/hotel-review-rag/rag-service/modules/prompts.py) 在 v1 模板的"评论块"渲染上区分"关键句 / 全文"两层：短评论只输出关键句；长评论同时输出关键句 + 全文。
  - [main.py](file:///Users/bytedance/Desktop/大模型/hotel-review-rag/rag-service/main.py) 的 `_format_comments` 透出 `primary_chunk` 字段；前端 [Comment](file:///Users/bytedance/Desktop/大模型/hotel-review-rag/src/types/comment.ts) 类型与 [qa.ts mapToComment](file:///Users/bytedance/Desktop/大模型/hotel-review-rag/src/lib/qa.ts) 同步补齐。
  - 前端 [CommentCard.tsx](file:///Users/bytedance/Desktop/大模型/hotel-review-rag/src/components/comments/CommentCard.tsx) 新增 `highlightChunkSpan(text, span)`：当评论携带 `primary_chunk` 时，按 `char_start/char_end` 在原评论字符串上直接 `<mark>` 高亮命中片段（覆盖原有的 keyword 高亮逻辑），并强制展开全文，避免折叠位置切断高亮。

### 5.2 离线指标（`RAG/data/evaluation/chunking_compare.json`）

索引侧（来自 `filtered_comments.csv` 的 2171 条评论）：

| 指标 | Baseline（整条评论） | Chunked（句切+滑窗） |
|---|---|---|
| 文档数 | 2171 | **7883**（avg 3.63 chunk/评论） |
| 平均文档长度（词） | 41.62 | **11.64** |
| 词项数 | 10734 | 10764（基本持平，证明切分未引入词表噪声） |
| 索引构建耗时 | 0.94 s | 0.56 s |

检索侧（100 Query, Top-K=10，chunk 命中后按 `comment_id` 二次去重）：

| 指标 | 值 | 解读 |
|---|---|---|
| **avg_overlap@10** | 4.06 | 两套索引 Top-10 平均仅 4 条重合 → chunk 化让 BM25 召回结构发生显著变化 |
| **avg_only_baseline@10** | 5.94 | 平均 ~6 条只在整条评论 BM25 中出现 |
| **avg_only_chunked@10** | 5.94 | 平均 ~6 条只在 chunk BM25 中出现 |
| **avg_chunks_hit_per_comment (Top-30 chunks)** | 1.04 | 每条命中评论平均仅被一个 chunk 命中 → 命中位置精确，符合"高亮单句"需求 |
| **avg_distinct_comments (Top-30 chunks)** | 28.99 | Top-30 chunk 几乎来自 30 条不同评论 → 同评论冗余被天然抑制 |

### 5.3 关键发现

1. **平均文档长度从 41.6 词降至 11.6 词**：BM25 中 `len/avgdl` 的归一化项更"公平"，单一主题的短句不会再被无关长句稀释。
2. **Top-10 重合率仅 40%**：意味着 chunk 化不是"小修小补"，而是结构性改变了召回结果分布。后续接 RRF + 重排可挑出更精确的"句级证据"。
3. **每条命中评论平均只被一个 chunk 击中（1.04）**：说明 chunk 粒度恰到好处，不会出现"同评论多个 chunk 重复刷榜"的冗余问题，省掉了原本要做的"chunk → comment 折叠"中的复杂去重——一次取最高分即可。
4. **检索结果在 Top-30 内覆盖了 28.99 条不同评论**：天然提升了多样性，与优化项 (15) MMR/DPP 的目标方向一致，但不需要额外算法。

### 5.4 局限与下一步

- 当前评估为离线代理指标。引入 LLM 后的"答案质量 / 引用准确率"对比，受 DDL 影响留作后续。
- 摘要库 / 反向 Query 库未跟随切分；策略上它们本就是"聚合粒度"，保持原样合理，但需要在 `HybridRetriever` 中实现 `chunk_id ↔ comment_id` 的反查表（设计已在 §3.4 给出，代码改动 ~30 行）。
- 主题切分（策略 C）需要类别分类器，与优化项 (3)/(4) 强耦合，不在本轮交付范围内。

## 6. 风险与备选

- **风险 1**：chunk 过短导致向量稀疏、噪声多。**应对**：在 chunker 中对 < 8 字的 chunk 与相邻 chunk 合并；并保留"原评论级"召回作为兜底（`enable_full_doc_fallback=True` 时，回退到旧索引的 Top-1）。
- **风险 2**：DashVector 主键变化会破坏现有云端集合。**应对**：另建 `comments_chunks` 集合，旧集合不动，便于一键回滚。
- **风险 3**：生成端 token 预算膨胀。**应对**：默认只发"代表 chunk + 评论级元数据"，仅在被 LLM 显式引用后才向前端透出整条评论（已有结构）。

## 7. 与其他优化项的耦合

- 与项 (3)/(4) 类别划分相关：若后续把 `主题切分(C)` 接上，可以直接复用类别小模型/聚类输出。
- 与项 (12) 多模态：chunk 粒度统一后，图片侧只需挂在原评论而非 chunk 上，避免错配。
- 与项 (15) MMR / DPP 重排：chunk 级 MMR 比评论级 MMR 更能压制"同评论多句重复命中"的冗余，是天然的下游搭配。
