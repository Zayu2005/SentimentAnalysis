# SentimentAnalysis

中文社交媒体舆情分析系统，覆盖从数据采集到知识图谱构建的全流水线。

## 系统架构

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                          SentimentAnalysis 舆情分析系统                              │
├───────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌──────────────┐ ┌───────────────┐ ┌──────────────┐ ┌─────────────┐             │
│  │SentimentSpider│ │SentimentProces│ │SentimentModel│ │ TopicCluster│             │
│  │  数据采集     │▶│  数据预处理    │▶│  情感分析    │▶│  话题聚类   │             │
│  └──────────────┘ └───────────────┘ └──────────────┘ └──────┬──────┘             │
│                                                              │                    │
│                                                              ▼                    │
│                                            ┌────────────────────────────┐         │
│                                            │      KnowledgeGraph       │         │
│                                            │  实体关系抽取 + 知识图谱    │         │
│                                            │    (OneKE + Neo4j)         │         │
│                                            └────────────────────────────┘         │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                              存储层                                          │  │
│  │   MySQL (原始/统一/预处理/情感/话题/抽取)    Neo4j (实体-关系图谱)            │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## 数据处理流程

```
社交媒体平台 (小红书/抖音/微博/B站/知乎/快手/贴吧)
    │
    ▼
[SentimentSpider] 数据采集 ─── Playwright 自动化 + 平台 API 逆向
    │
    ▼
[SentimentProcessor] 数据预处理 ─── 正则清洗 → OpenCC 繁简转换 → jieba 分词 → TF-IDF/TextRank 提关键词
    │
    ▼
[SentimentModel] 情感分析 ─── BERT (RoBERTa-wwm-ext) 三分类 / Qwen2.5 LoRA 微调 (18 情绪)
    │
    ▼
[TopicCluster] 话题聚类 ─── BERT [CLS] 嵌入 → Faiss Single-Pass 聚类 → 话题合并/演化
    │
    ▼
[KnowledgeGraph] 知识图谱 ─── OneKE 三 Agent 抽取 → 实体聚合去重 → Neo4j MERGE 写入
```

---

## 各模块算法详解

### 1. SentimentProcessor — 数据预处理

#### 1.1 文本清洗 (TextCleaner)

多层正则表达式流水线，按序执行：

| 步骤 | 算法 | 说明 |
|------|------|------|
| HTML 清除 | `re.sub(r'<[^>]+>', '')` | 去除残留标签 |
| URL 移除 | 正则匹配 `https?://` 和 `www.` 链接 | 消除噪声 |
| @提及移除 | `@[\w\u4e00-\u9fff]+` | 去除用户提及 |
| 话题标签 | `#([^#\s]+)#?` → 保留话题文字，移除 `#` 和 `[话题]` 后缀 | 保留语义 |
| Emoji 移除 | Unicode 范围 `U+1F600-U+1FAFF` + 平台表情 `[笑哭R]` | 去除表情 |
| 繁简转换 | **OpenCC** `t2s` 模式 | 繁体→简体 |
| 网络用语 | 自定义映射表 `SlangNormalizer` | 如 "yyds"→"永远的神" |
| 空白规范 | `\s+` → 单空格 | 统一格式 |

#### 1.2 中文分词 (Segmenter)

基于 **jieba** 分词引擎：

- **默认精确模式**: `jieba.cut(text)` — 基于前缀词典 + 动态规划 (DAG) 求解最大概率路径，对未登录词使用 HMM (隐马尔可夫模型) 进行识别
- **搜索引擎模式**: `jieba.cut_for_search(text)` — 在精确模式基础上对长词再次切分，提高召回率
- **停用词过滤**: 加载停用词表，过滤高频无意义词

> jieba 分词算法参考: 基于前缀词典的有向无环图 (DAG) + Viterbi 算法 (HMM)

#### 1.3 关键词提取 (KeywordExtractor)

支持两种算法：

| 算法 | 原理 | 适用场景 |
|------|------|---------|
| **TF-IDF** | 词频 (TF) × 逆文档频率 (IDF)，使用 jieba 内置 IDF 语料库 | 通用关键词提取 |
| **TextRank** | 基于 PageRank 的无监督排序算法，构建词共现图计算节点重要度 | 强调上下文语义 |

> TextRank 算法来源: Mihalcea R, Tarau P. **TextRank: Bringing Order into Texts**. EMNLP 2004.

---

### 2. SentimentModel — 情感分析

#### 2.1 BERT 情感分类器

**模型架构**: `chinese-roberta-wwm-ext` + Dropout + Linear 分类头

```
Input → BERT Encoder → [CLS] Token (768-dim) → Dropout(0.1) → Linear(768→3) → Softmax → 三分类
```

| 组件 | 说明 |
|------|------|
| 预训练模型 | **RoBERTa-wwm-ext** (哈工大讯飞联合实验室)，全词遮蔽 (Whole Word Masking) 预训练 |
| 分类头初始化 | Xavier Uniform 初始化权重，零初始化偏置 |
| 损失函数 | CrossEntropyLoss |
| 情感分数 | `score = P(positive) - P(negative)`，范围 [-1, 1] |
| 训练优化器 | AdamW + Linear Warmup 学习率调度 |
| 混合精度 | PyTorch AMP `autocast` + `GradScaler` |
| 评估指标 | Accuracy, Macro-F1, Weighted-F1, Per-class P/R/F1, Confusion Matrix |

> 预训练模型来源: Cui Y, Che W, Liu T, et al. **Pre-Training with Whole Word Masking for Chinese BERT**. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2021.

#### 2.2 Qwen2.5 情感分析器

**模型架构**: `Qwen2.5-1.5B-Instruct` + LoRA 微调

| 组件 | 说明 |
|------|------|
| 基座模型 | **Qwen2.5-1.5B-Instruct** (通义千问，阿里云) |
| 微调方法 | **LoRA** (Low-Rank Adaptation)，通过 `peft` 库加载适配器 |
| 量化支持 | 可选 4-bit NF4 量化 (`BitsAndBytesConfig`) |
| 输入格式 | Chat Template: System Prompt + User Prompt → JSON 输出 |
| 输出 | `sentiment` (三分类) + `sentiment_score` (-1~1) + `emotion_tags` (18 种情绪) |
| 解码策略 | `temperature=0.1`, `max_new_tokens=256` |
| 后处理 | 正则提取 JSON + 字段验证 + 标签过滤 |

**18 种情绪标签** (参考 GoEmotions 针对中文场景优化):

| 类别 | 情绪 |
|------|------|
| 正面 | 喜悦、兴奋、满足、感激、爱 |
| 负面 | 愤怒、厌恶、悲伤、恐惧、失望 |
| 复杂 | 惊讶、困惑、好奇、期待、焦虑 |
| 中性 | 平静、无聊、冷漠 |

> LoRA 来源: Hu E J, Shen Y, Wallis P, et al. **LoRA: Low-Rank Adaptation of Large Language Models**. ICLR 2022.
>
> GoEmotions 来源: Demszky D, Movshovitz-Attias D, Ko J, et al. **GoEmotions: A Dataset of Fine-Grained Emotions**. ACL 2020.

---

### 3. TopicCluster — 话题聚类

#### 3.1 文本嵌入 (BertEmbedder)

使用 `chinese-roberta-wwm-ext` 提取 768 维 [CLS] 向量：

```
Input Text → BERT Tokenizer → BERT Encoder → last_hidden_state[:, 0, :] → L2 Normalize → 768-dim Vector
```

- **L2 归一化**: `v = v / ||v||₂`，使得内积 (Inner Product) 等价于余弦相似度
- **批处理**: 按 batch_size 分批，`padding=True, truncation=True, max_length=128`

#### 3.2 向量索引 (FaissIndex)

基于 **Faiss** (`IndexFlatIP`) 的精确内积检索：

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| `search(query, k)` | O(n) | 暴力扫描所有质心，返回 Top-k |
| `add_topic(id, centroid)` | O(n) | 添加并重建索引 |
| `get_all_pairs_similarity()` | O(n²) | 矩阵乘法计算所有话题对相似度 |

> Faiss 来源: Johnson J, Douze M, Jégou H. **Billion-scale similarity search with GPUs**. IEEE Transactions on Big Data, 2021. (Meta AI Research)

#### 3.3 Single-Pass 增量聚类 (ClusterEngine)

核心算法流程：

```
对于每条新内容:
  1. 计算 BERT 嵌入向量 e
  2. 在 Faiss 索引中搜索最近的话题质心: (topic_id, similarity) = search(e, k=1)
  3. 如果 similarity ≥ threshold (默认 0.84):
       → 归入现有话题
       → 增量更新质心 (Running Average):
         c_new = (c_old × n + e) / (n + 1)
         c_new = c_new / ||c_new||₂    (L2 归一化)
  4. 否则:
       → 创建新话题，以当前嵌入作为初始质心
```

**特点**:
- 在线增量处理，无需预设簇数 k
- 质心滑动平均更新，避免全量重算
- 阈值动态控制聚类粒度

> Single-Pass 聚类思路参考:
> - Cao Y, Ngo C, Zhang J, et al. **HISEvent: A Large-Scale High Inter-Intra Similarity Benchmark for Social Media Event Detection**. AAAI 2024.
> - 本系统在 HISEvent 的嵌入+检索框架基础上简化为 Single-Pass 模式，以适应流式增量场景。

#### 3.4 话题合并 (TopicMaintainer.merge_topics)

全局相似度扫描合并：

```
1. 加载所有活跃话题质心到 Faiss
2. 计算所有话题对的余弦相似度矩阵: S = V × Vᵀ
3. 按相似度降序遍历话题对:
   如果 similarity ≥ merge_threshold (默认 0.92):
     → 小话题合并到大话题 (content_count 多者为主)
     → 重新分配内容归属
     → 源话题标记为 merged 状态
```

#### 3.5 话题生命周期管理

状态机模型：

```
emerging ──[content_count ≥ 10]──▶ active
   │                                  │
   ├──[inactive_days ≥ 3]──▶ declining ──[inactive_days ≥ 7]──▶ ended
   │
   └──[合并]──▶ merged
```

#### 3.6 热度评估

```
热度分数: hot_score = content_delta × 10 + interaction × 0.01
热度等级:
  score ≥ 500 → critical
  score ≥ 100 → high
  score ≥ 20  → medium
  score < 20  → low
```

#### 3.7 LLM 话题命名 (TopicNamer)

使用 `Qwen2.5-1.5B-Instruct` 为话题生成结构化描述：

- **输入**: 话题下 Top-10 代表性内容 (含平台、情感标签、标题、正文截取)
- **输出**: JSON `{event_name, event_description, keywords[{word, weight}]}`
- **Prompt 策略**: 要求"主体+事件+影响"结构命名，描述需涵盖事件背景、涉及主体、公众态度、潜在风险

---

### 4. KnowledgeGraph — 知识图谱构建

#### 4.1 实体关系 Schema

预定义 7 类实体和 8 类关系，针对中文社交媒体舆情场景：

| 实体类型 | 说明 | 示例 |
|----------|------|------|
| 人物 | 自然人 | 雷军、马斯克 |
| 组织机构 | 企业、政府机构 | 小米集团、消防局 |
| 品牌 | 产品品牌 | 小米、特斯拉 |
| 产品 | 具体产品 | SU7、iPhone 16 |
| 地点 | 地理位置 | 营口、北京 |
| 事件 | 具体事件 | 起火事故、发布会 |
| 平台 | 社交媒体平台 | 微博、小红书 |

| 关系类型 | 说明 |
|----------|------|
| 涉及 | 人物/组织参与事件 |
| 生产 | 品牌生产产品 |
| 位于 | 事件/组织关联地点 |
| 共同提及 | 同一上下文共现 |
| 导致 | 事件间因果关系 |
| 回应 | 对事件做出回应 |
| 属于 | 从属关系 |
| 竞争 | 品牌/产品间竞争 |

#### 4.2 OneKE 三 Agent 抽取流水线

使用 **OneKE** 框架 (WWW 2025) + DeepSeek API 进行实体关系抽取：

```
Input Text
    │
    ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Schema Agent │ ──▶ │ Extraction Agent  │ ──▶ │ Reflection Agent │
│  模式推演    │     │   信息抽取         │     │   质量反思        │
└─────────────┘     └──────────────────┘     └──────────────────┘
```

**Schema Agent (模式推演)**:
- 根据任务类型 (NER/RE/EE/Triple) 和约束条件选择合适的 Schema
- Triple 任务使用 `get_retrieved_schema` 从 Schema Repository 检索匹配模式
- 输出: 实体类型列表 + 关系类型列表

**Extraction Agent (信息抽取)**:
- 将输入文本按 NLTK 句子分割为 chunks
- 对每个 chunk 构建 Constraint Prompt，包含 Schema 定义 + 抽取规则
- 调用 LLM 抽取三元组: `(head, head_type, relation, relation_type, tail, tail_type)`
- `summarize_answer`: 聚合多 chunk 结果并去重

**Reflection Agent (质量反思)** (standard 模式):
- **Self-Consistency (自一致性检验)**: 以 3 种温度 (T=0.2, 0.5, 1.0) 重复抽取，投票选出一致性最高的结果
- **Case-Based Reflection (案例反思)**: 从 Case Repository 检索相似历史案例，利用 bad_case 进行错误纠正
- **Case Repository**: 混合相似度检索 (50% Embedding 余弦相似度 + 50% RapidFuzz 字符串匹配)，返回 Top-2 案例

> OneKE 来源: **OneKE: A Dockerized Schema-Guided LLM Agent-based Knowledge Extraction System**. The Web Conference (WWW) 2025.

#### 4.3 实体聚合去重

从 MySQL `kg_extraction` 表读取话题下所有抽取结果，进行跨文档聚合：

```
实体去重:
  key = (name.strip().lower(), entity_type)
  相同 key 的实体合并，mention_count 累加，properties 合并

关系聚合:
  key = (head.lower(), tail.lower(), relation)
  相同 key 的关系合并:
    confidence = avg(所有来源的 confidence)
    source_count = 累加计数
```

#### 4.4 Neo4j 图模型与写入

```
(:TopicEvent {topic_id, name, description})
(:Entity {name, entity_type, mention_count, properties})

(:Entity)-[:BELONGS_TO_TOPIC]->(:TopicEvent)
(:Entity)-[:RELATES_TO {relation_type, confidence, source_count, topic_id}]->(:Entity)
```

写入使用 **MERGE 语义** (幂等)，支持重复执行不产生重复数据。

---

### 5. SentimentSpider — 数据采集

#### 5.1 MediaCrawler 多平台爬虫

| 平台 | 支持内容 | 支持评论 | 技术方案 |
|------|---------|---------|---------|
| 小红书 | ✅ | ✅ | Playwright + 签名算法 |
| 抖音 | ✅ | ✅ | Playwright + API |
| 快手 | ✅ | ✅ | GraphQL API |
| 微博 | ✅ | ✅ | Cookie 登录 + API |
| B站 | ✅ | ✅ | HTTP API |
| 贴吧 | ✅ | ✅ | HTTP API |
| 知乎 | ✅ | ✅ | HTTP API |

#### 5.2 热点新闻模块

- 聚合百度、微博、知乎、B站、抖音等 10+ 平台热榜
- LLM 领域匹配: 判断热点是否属于监测领域
- 自动提取搜索关键词，触发定向爬取

---

## 项目结构

```
SentimentAnalysis/
├── SentimentSpider/              # 数据采集模块
│   ├── MediaCrawler/             # 多平台社交媒体爬虫
│   │   ├── media_platform/       # 各平台爬虫实现 (xhs/douyin/weibo/bilibili/kuaishou/tieba/zhihu)
│   │   ├── store/                # 数据存储适配器
│   │   ├── proxy/                # 代理 IP 池管理
│   │   ├── cache/                # Redis/本地缓存
│   │   └── api/                  # FastAPI 管理接口
│   └── hot_news/                 # 热点新闻采集
│       ├── analyzer/             # LLM 领域匹配 + 关键词提取
│       ├── fetcher/              # 热榜抓取客户端
│       ├── sync/                 # 统一数据同步
│       └── database/migrations/  # SQL 迁移脚本 (001-005)
│
├── SentimentProcessor/           # 数据预处理模块
│   ├── processor/
│   │   ├── cleaner.py            # 文本清洗 (正则 + OpenCC)
│   │   ├── segmenter.py          # jieba 中文分词
│   │   └── extractor.py          # TF-IDF / TextRank 关键词提取
│   └── utils/
│       ├── stopwords.py          # 停用词管理
│       └── slang.py              # 网络用语规范化
│
├── SentimentModel/               # 情感分析模块
│   ├── models/
│   │   └── bert_classifier.py    # BERT (RoBERTa-wwm-ext) 三分类器
│   ├── qwen/
│   │   ├── predictor.py          # Qwen2.5 LoRA 推理器
│   │   ├── finetune.py           # LoRA 微调脚本
│   │   ├── trainer.py            # 训练循环
│   │   └── data_prepare.py       # 指令微调数据准备
│   ├── training/
│   │   ├── trainer.py            # BERT 训练器 (AdamW + AMP)
│   │   └── metrics.py            # Accuracy/F1/Confusion Matrix
│   └── inference/
│       └── predictor.py          # BERT 推理器
│
├── TopicCluster/                 # 话题聚类模块
│   ├── cluster/
│   │   ├── embedder.py           # BERT [CLS] 嵌入提取
│   │   ├── index.py              # Faiss IndexFlatIP 向量索引
│   │   ├── engine.py             # Single-Pass 增量聚类引擎
│   │   └── maintainer.py         # 话题合并 / 生命周期 / 演化快照
│   └── llm/
│       └── namer.py              # Qwen2.5 话题命名/描述生成
│
├── KnowledgeGraph/               # 知识图谱模块
│   ├── extraction/
│   │   ├── schema.py             # 7 实体类型 + 8 关系类型定义
│   │   └── extractor.py          # OneKE Pipeline 实体关系抽取
│   ├── graph/
│   │   └── builder.py            # 实体聚合去重 + Neo4j MERGE 写入
│   └── database/
│       ├── mysql_connection.py   # MySQL 操作
│       ├── neo4j_connection.py   # Neo4j 驱动 + session 管理
│       └── repository.py         # kg_extraction / kg_build_log 仓库
│
├── OneKE/                        # OneKE 信息抽取框架 (WWW 2025)
│   └── src/
│       ├── pipeline.py           # Schema → Extraction → Reflection 三 Agent 流水线
│       ├── modules/
│       │   ├── schema_agent.py   # Schema 推演 Agent
│       │   ├── extraction_agent.py # 信息抽取 Agent
│       │   └── reflection_agent.py # 自一致性反思 Agent
│       └── utils/
│           └── process.py        # JSON 解析与后处理
│
├── .env.example                  # 环境变量模板
├── requirements.txt              # 统一依赖
└── README.md
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Zayu2005/SentimentAnalysis.git
cd SentimentAnalysis
```

### 2. 创建虚拟环境

```bash
# Conda (推荐)
conda create -n sentiment python=3.10
conda activate sentiment

# 或 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. 安装依赖

```bash
# 安装所有依赖 (CPU)
pip install -r requirements.txt

# 安装 Playwright 浏览器 (爬虫需要)
playwright install chromium
```

**GPU 环境 (可选)**:
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，设置必要配置
```

必须配置的项：
```env
# MySQL
MYSQL_DB_PWD=your_password_here

# Neo4j (KnowledgeGraph 模块)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here

# DeepSeek API (KnowledgeGraph 模块)
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### 5. 初始化数据库

```sql
CREATE DATABASE sentiment DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

```bash
# 依次执行迁移脚本
mysql -u root -p sentiment < SentimentSpider/hot_news/database/migrations/001_initial.sql
mysql -u root -p sentiment < SentimentSpider/hot_news/database/migrations/002_unified_tables.sql
mysql -u root -p sentiment < SentimentSpider/hot_news/database/migrations/003_topic_cluster.sql
mysql -u root -p sentiment < SentimentSpider/hot_news/database/migrations/004_wordcloud.sql
mysql -u root -p sentiment < SentimentSpider/hot_news/database/migrations/005_knowledge_graph.sql
```

### 6. 验证安装

```bash
python -m SentimentProcessor stats
python test_qwen_base.py
```

## 使用示例

### 完整舆情分析工作流

```bash
# Step 1: 采集数据
cd SentimentSpider/MediaCrawler
python main.py --platform xhs --keywords "新能源汽车" --type search
cd ../..

# Step 2: 数据预处理
python -m SentimentProcessor all

# Step 3: 情感分析
python run_qwen_analyze.py

# Step 4: 话题聚类
python -m TopicCluster cluster --batch-size 64
python -m TopicCluster describe --all --include-ended
python -m TopicCluster merge
python -m TopicCluster evolve

# Step 5: 知识图谱构建
python -m KnowledgeGraph extract --topic-id 1 --limit 50
python -m KnowledgeGraph build --topic-id 1 --clear

# Step 6: 查看结果
python -m TopicCluster stats
python -m KnowledgeGraph query --topic-id 1
python -m KnowledgeGraph stats
```

## CLI 命令汇总

### SentimentSpider (数据采集与同步)

| 命令 | 说明 |
|------|------|
| `cd SentimentSpider && python -m hot_news.cli.main fetch` | 获取热点新闻 |
| `cd SentimentSpider && python -m hot_news.cli.main fetch xhs wb bili -l 50` | 指定平台获取热点 |
| `cd SentimentSpider && python -m hot_news.cli.main sync` | 增量同步各平台数据到统一表 |
| `cd SentimentSpider && python -m hot_news.cli.main sync xhs wb` | 指定平台同步 |
| `cd SentimentSpider && python -m hot_news.cli.main sync --full` | 全量同步（清空后重建） |
| `cd SentimentSpider && python -m hot_news.cli.main sync --no-comments` | 只同步内容，不同步评论 |
| `cd SentimentSpider && python -m hot_news.cli.main sync-stats` | 查看同步统计 |
| `cd SentimentSpider && python -m hot_news.cli.main show content` | 查看内容数据 |
| `cd SentimentSpider && python -m hot_news.cli.main show comment` | 查看评论数据 |
| `cd SentimentSpider && python -m hot_news.cli.main show hot` | 查看热点数据 |
| `cd SentimentSpider && python -m hot_news.cli.main analyze` | 分析热点领域匹配 |
| `cd SentimentSpider && python -m hot_news.cli.main run` | 运行完整流水线 |
| `cd SentimentSpider/MediaCrawler && python main.py --platform xhs --keywords "关键词" --type search` | 爬取指定平台内容 |

**完整数据采集到分析工作流**:

```bash
# Step 1: 采集热点
cd SentimentSpider
python -m hot_news.cli.main fetch

# Step 2: 领域匹配分析
python -m hot_news.cli.main analyze

# Step 3: 触发爬取（根据热点关键词）
cd MediaCrawler
python main.py --platform xhs --keywords "新能源汽车" --type search

# Step 4: 同步数据到统一表
cd ..
python -m hot_news.cli.main sync

# Step 5: 查看同步结果
python -m hot_news.cli.main sync-stats
```

### SentimentProcessor

| 命令 | 说明 |
|------|------|
| `python -m SentimentProcessor content` | 处理内容 |
| `python -m SentimentProcessor comments` | 处理评论 |
| `python -m SentimentProcessor all` | 处理全部 |
| `python -m SentimentProcessor stats` | 查看统计 |

### TopicCluster

| 命令 | 说明 |
|------|------|
| `python -m TopicCluster cluster` | 增量聚类 |
| `python -m TopicCluster describe [--all]` | LLM 话题命名 |
| `python -m TopicCluster merge [--dry-run]` | 合并相似话题 |
| `python -m TopicCluster evolve` | 更新生命周期/统计/演化快照 |
| `python -m TopicCluster wordcloud [--all]` | 生成词云数据 |
| `python -m TopicCluster recluster` | 全量重聚类 |
| `python -m TopicCluster stats` | 查看统计 |

### KnowledgeGraph

| 命令 | 说明 |
|------|------|
| `python -m KnowledgeGraph extract -t <ID>` | 抽取实体关系 |
| `python -m KnowledgeGraph build -t <ID>` | 构建 Neo4j 图 |
| `python -m KnowledgeGraph pipeline -t <ID>` | 完整流水线 (抽取+构建) |
| `python -m KnowledgeGraph query -t <ID>` | 查询图谱信息 |
| `python -m KnowledgeGraph stats` | 全局统计 |

## 配置说明

| 变量 | 必须 | 默认值 | 说明 |
|------|------|--------|------|
| `MYSQL_DB_HOST` | 否 | localhost | MySQL 主机 |
| `MYSQL_DB_PORT` | 否 | 3306 | MySQL 端口 |
| `MYSQL_DB_USER` | 否 | root | MySQL 用户 |
| `MYSQL_DB_PWD` | **是** | - | MySQL 密码 |
| `MYSQL_DB_NAME` | 否 | sentiment | 数据库名 |
| `HF_ENDPOINT` | 否 | https://hf-mirror.com | HuggingFace 镜像 |
| `QWEN_MODEL_NAME` | 否 | Qwen/Qwen2.5-1.5B-Instruct | Qwen 模型 |
| `BERT_MODEL_NAME` | 否 | hfl/chinese-roberta-wwm-ext | BERT 模型 |
| `NEO4J_URI` | 否 | bolt://localhost:7687 | Neo4j 连接地址 |
| `NEO4J_USER` | 否 | neo4j | Neo4j 用户 |
| `NEO4J_PASSWORD` | KG模块需要 | - | Neo4j 密码 |
| `DEEPSEEK_API_KEY` | KG模块需要 | - | DeepSeek API Key |
| `DEEPSEEK_API_BASE` | 否 | https://api.deepseek.com | DeepSeek API 地址 |
| `DEEPSEEK_MODEL` | 否 | deepseek-chat | DeepSeek 模型 |

## 数据迁移

### 从本地数据库迁移到远程

#### 方式1: 使用 mysqldump (推荐)

```bash
# 导出单个表
mysqldump -u root -p1234 sentiment 表名 --single-transaction | mysql -h 远程IP -u 用户名 -p密码 目标数据库

# 批量迁移示例
for table in xhs_note weibo_note bilibili_video kuaishou_video tieba_note zhihu_content; do
  mysqldump -u root -p1234 sentiment $table | mysql -h 202.200.205.108 -u sentiment -pM22hbxnfhDyRCwaT sentiment
done
```

#### 方式2: SQL导入 (统一表)

```sql
-- 迁移内容到统一表 (以微博为例)
INSERT INTO unified_content (
    platform, content_id, content_type, user_id, nickname, title, content, content_url, media_type, 
    liked_count, comment_count, share_count, source_keyword, add_ts
)
SELECT 
    'wb', CAST(note_id AS CHAR), 'note', user_id, nickname, '', content, note_url, 'text',
    COALESCE(CAST(liked_count AS UNSIGNED), 0),
    COALESCE(CAST(comments_count AS UNSIGNED), 0),
    COALESCE(CAST(shared_count AS UNSIGNED), 0),
    source_keyword, add_ts
FROM weibo_note;

-- 迁移评论到统一表
INSERT INTO unified_comment (
    platform, comment_id, content_id, user_id, nickname, content, add_ts
)
SELECT 
    'wb', CAST(comment_id AS CHAR), CAST(note_id AS CHAR), user_id, nickname, content, add_ts
FROM weibo_note_comment;
```

### 常用数据库查询

```sql
-- 查看各平台数据分布
SELECT platform, COUNT(*) as cnt FROM unified_content GROUP BY platform;

-- 查看已聚类内容
SELECT platform, COUNT(*) as cnt FROM unified_content WHERE topic_id IS NOT NULL GROUP BY platform;

-- 话题统计
SELECT id, event_name, content_count, platform_distribution FROM topic_event;

-- 话题互动数据
SELECT te.id, te.content_count, 
       SUM(uc.liked_count) as total_likes,
       SUM(uc.comment_count) as total_comments
FROM topic_event te
JOIN unified_content uc ON uc.topic_id = te.id
GROUP BY te.id;
```

## 数据库表结构

### 核心表

| 表名 | 说明 |
|------|------|
| `unified_content` | 统一内容表 (含 topic_id, sentiment, keywords 等) |
| `unified_comment` | 统一评论表 |
| `processed_content` | 预处理结果 (清洗/分词/关键词) |
| `topic_event` | 话题事件 (质心嵌入、状态、情感/热度统计) |
| `topic_evolution` | 话题演化快照 (每日热度、情感、平台分布) |
| `topic_merge_log` | 话题合并日志 |
| `kg_extraction` | 实体关系抽取结果 (entities JSON, relations JSON) |
| `kg_build_log` | Neo4j 构建日志 |

### 原始数据表

| 表名 | 说明 |
|------|------|
| `xhs_note` / `xhs_note_comment` | 小红书 |
| `weibo_note` / `weibo_note_comment` | 微博 |
| `bilibili_video` / `bilibili_video_comment` | B站 |
| `kuaishou_video` / `kuaishou_video_comment` | 快手 |
| `tieba_note` / `tieba_comment` | 贴吧 |
| `zhihu_content` / `zhihu_comment` | 知乎 |

## Troubleshooting

### 问题1: 模块找不到
```bash
# 确保在正确的目录
cd E:/Code/Project/SentimentAnalysis

# 确保 conda 环境已激活
conda activate sentiment
```

### 问题2: 数据库连接失败
```bash
# 检查 .env 配置
cat .env

# 测试连接
mysql -h 202.200.205.108 -u sentiment -pM22hbxnfhDyRCwaT sentiment -e "SELECT 1"
```

### 问题3: 依赖缺失
```bash
# 安装依赖
pip install -r requirements.txt

# 或使用 conda
conda install pytorch transformers jieba faiss-cpu
```

### 问题4: 话题聚类全部归为一个话题
- 可能是相似度阈值设置过低 (默认 0.75)
- 可以调整 `TopicCluster/config/settings.py` 中的 `similarity_threshold`
- 或使用 `--dry-run` 参数预览聚类效果

## 参考文献

> 以下为本项目各模块所引用的核心算法与模型的学术出处。

### 情感分析

- **[1]** Cui Y, Che W, Liu T, Qin B, Wang S, Hu G. **Revisiting Pre-Trained Models for Chinese Natural Language Processing**. *Findings of the Association for Computational Linguistics: EMNLP 2020*, pp. 657–668, 2020.
  - 用途: `chinese-roberta-wwm-ext` 预训练模型，用于情感分类基座和文本嵌入提取
  - 模块: `SentimentModel/models/bert_classifier.py`, `TopicCluster/cluster/embedder.py`

- **[2]** Cui Y, Che W, Liu T, Qin B, Yang Z. **Pre-Training with Whole Word Masking for Chinese BERT**. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 29, pp. 3504–3514, 2021.
  - 用途: 全词遮蔽 (Whole Word Masking) 预训练策略，提升中文语义理解
  - 模块: `SentimentModel`, `TopicCluster`

- **[3]** Hu E J, Shen Y, Wallis P, Allen-Zhu Z, Li Y, Wang S, Wang L, Chen W. **LoRA: Low-Rank Adaptation of Large Language Models**. *International Conference on Learning Representations (ICLR)*, 2022.
  - 用途: Qwen2.5-1.5B-Instruct 的参数高效微调 (rank=8, alpha=16)
  - 模块: `SentimentModel/qwen/trainer.py`

- **[4]** Dettmers T, Pagnoni A, Holtzman A, Zettlemoyer L. **QLoRA: Efficient Finetuning of Quantized Large Language Models**. *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.
  - 用途: 4-bit NF4 量化，降低 Qwen2.5 微调显存需求
  - 模块: `SentimentModel/qwen/trainer.py` (BitsAndBytesConfig)

- **[5]** Demszky D, Movshovitz-Attias D, Ko J, Cowen A, Nemade G, Ravi S. **GoEmotions: A Dataset of Fine-Grained Emotions**. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, pp. 4040–4054, 2020.
  - 用途: 18 类情绪标签体系设计参考，适配中文社交媒体场景
  - 模块: `SentimentModel/qwen/data_prepare.py`

### 文本预处理

- **[6]** Mihalcea R, Tarau P. **TextRank: Bringing Order into Texts**. *Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 404–411, 2004.
  - 用途: 基于 PageRank 的无监督关键词提取
  - 模块: `SentimentProcessor/processor/extractor.py`

### 话题聚类

- **[7]** Johnson J, Douze M, Jégou H. **Billion-scale Similarity Search with GPUs**. *IEEE Transactions on Big Data*, vol. 7, no. 3, pp. 535–547, 2021.
  - 用途: Faiss IndexFlatIP 内积向量检索，用于话题质心匹配
  - 模块: `TopicCluster/cluster/index.py`

- **[8]** Cao Y, Ngo C, Zhang J, Chua T. **HISEvent: A Large-Scale High Inter-Intra Similarity and Hard Cases Benchmark for Social Media Event Detection**. *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, 2024.
  - 用途: Single-Pass 增量聚类框架参考，本项目在其嵌入+检索框架上简化适配流式场景
  - 模块: `TopicCluster/cluster/engine.py`

### 知识图谱

- **[9]** Xiao N, Hu Z, Zheng J, Liu J, Cochez M, Chen J, Deng S, Ye H, Zhang N, Chen H, et al. **OneKE: A Dockerized Schema-Guided LLM Agent-based Knowledge Extraction System**. *Proceedings of the ACM Web Conference (WWW)*, 2025.
  - 用途: Schema Agent → Extraction Agent → Reflection Agent 三阶段实体关系抽取流水线
  - 模块: `OneKE/src/pipeline.py`, `KnowledgeGraph/extraction/extractor.py`

### 训练优化

- **[10]** Loshchilov I, Hutter F. **Decoupled Weight Decay Regularization**. *International Conference on Learning Representations (ICLR)*, 2019.
  - 用途: AdamW 优化器，解耦权重衰减与梯度更新
  - 模块: `SentimentModel/training/trainer.py`

## 项目规划

- [x] 数据采集模块 (SentimentSpider)
- [x] 数据预处理模块 (SentimentProcessor)
- [x] 情感分析模型 (SentimentModel)
- [x] 话题聚类与事件监测 (TopicCluster)
- [x] 知识图谱构建 (KnowledgeGraph + OneKE)
- [ ] API 服务 (Spring Boot + Neo4j/MySQL)
- [ ] 可视化仪表板 (前端知识图谱展示)
- [ ] 舆情预测与应对方案 (见微知著)

## 作者

**Zayu2005** - [GitHub](https://github.com/Zayu2005)

## 许可证

MIT License
