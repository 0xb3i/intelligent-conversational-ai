# scripts —— 离线脚本

## `build_summaries.py` —— 分类摘要生成（v2 结构化）

参考 Exp1 的 LLM 调用范式，对 `RAG/data/processed/filtered_comments.csv`
中每个 `category` 标签生成更高质量的「结构化摘要」。

### 优化点（相对旧版 `category_summaries.json`）

| 维度 | 旧 v1 | 新 v2 |
|------|-------|-------|
| 摘要粒度 | 每类 1 段 ~600 字 | 每类 1 段 overview + N 个要点 (subtopic / polarity / 关键词 / salience) |
| 极性 | 正负面混在一段 | 拆成 positive / negative / neutral 独立要点 |
| 输入策略 | 单次塞入全部评论 → token 易超限 | 分块抽取 (Map) → 全局聚合 (Reduce) |
| 评论选择 | 顺序拼接 | 按 quality_score + 年份分层采样 |
| 容错 | 单次调用失败整类丢失 | 每个 chunk 独立重试，单点失败不影响整体 |
| 检索友好度 | 长段文本难匹配 query | 要点级别可被独立检索/打分 |

### 前置条件

1. 已激活项目虚拟环境：
   ```bash
   cd <project_root>
   source .venv/bin/activate
   ```

2. 已安装额外依赖（首次运行时）：
   ```bash
   pip install openai
   ```

3. 已设置 DashScope API Key（任选其一）：
   - 写在 `rag-service/.env`：
     ```
     DASHSCOPE_API_KEY=sk-xxxx
     ```
   - 或临时导出：
     ```bash
     export DASHSCOPE_API_KEY=sk-xxxx
     ```

### 运行

```bash
# 1) 先 dry-run 看看处理计划，不消耗 API
python rag-service/scripts/build_summaries.py --dry-run

# 2) 完整运行（默认 qwen-plus / 14 个类目 / 每类最多 200 条评论 / 每类最多 6 个要点）
python rag-service/scripts/build_summaries.py

# 3) 调试单个类目
python rag-service/scripts/build_summaries.py --only-category 餐饮设施

# 4) 调整采样规模与并发
python rag-service/scripts/build_summaries.py \
    --max-comments 150 --chunk-size 25 --workers 8 --max-points 5
```

### 输出

- `RAG/data/processed/category_summaries_v2.json` —— v2 结构化摘要（推荐使用）
- `RAG/data/processed/category_summaries_v2_flat.json` —— 兼容旧 schema 的扁平版
  （仅保留 `overview` 作为 summary，方便和旧 `category_summaries.json` 做 A/B 对比）

v2 数据结构示例：

```json
{
  "餐饮设施": {
    "category": "餐饮设施",
    "comment_count": 676,
    "sampled_count": 200,
    "n_chunks": 7,
    "n_candidates": 38,
    "overview": "餐饮整体口碑突出 …（2-4 句）",
    "keywords": ["早餐", "粤菜", "酒廊", "排队", "瀑布"],
    "points": [
      {
        "subtopic": "自助早餐",
        "polarity": "positive",
        "summary": "瀑布景观/旋转餐厅的自助早餐环境受到广泛好评。",
        "keywords": ["早餐", "瀑布", "种类"],
        "salience": 0.85
      },
      {
        "subtopic": "自助早餐",
        "polarity": "negative",
        "summary": "高峰期排队明显，部分客人反映等位 15-20 分钟。",
        "keywords": ["早餐", "排队", "高峰"],
        "salience": 0.42
      }
    ]
  }
}
```

### 后续衔接（不在本脚本范围内）

- 把 v2 摘要灌入 ChromaDB（每个 point 作为一条独立记录，索引在 `summary + keywords` 上）
- 在 `retriever._route_summary` 中改为返回相关 points 而非整段文本
- 在 `generator` 的 prompt 中以"要点列表"形式注入摘要，提升 LLM 利用率
- 通过 `eval_prompt.py` 风格的消融评估对比 v1 vs v2 提升幅度
