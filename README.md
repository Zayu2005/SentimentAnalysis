# SentimentAnalysis

中文社交媒体舆情分析系统，支持多平台数据采集、文本预处理、情感分析和话题聚类。

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          SentimentAnalysis 舆情分析系统                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌───────────────┐  ┌────────────────┐  ┌───────────────┐  ┌──────────────┐    │
│  │ SentimentSpider│  │SentimentProcessor│ │ SentimentModel│  │ TopicCluster │    │
│  │   数据采集     │─▶│   数据预处理    │─▶│   情感分析    │─▶│  话题聚类    │    │
│  └───────────────┘  └────────────────┘  └───────────────┘  └──────────────┘    │
│         │                  │                    │                  │             │
│         ▼                  ▼                    ▼                  ▼             │
│  ┌──────────────────────────────────────────────────────────────────────┐       │
│  │                           MySQL 数据库                                │       │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐            │       │
│  │  │原始数据 │ │统一数据 │ │预处理  │ │情感结果 │ │话题事件 │            │       │
│  │  │xhs_note│ │unified_│ │cleaned │ │sentiment│ │topic_  │            │       │
│  │  │douyin_ │ │content │ │segment │ │score   │ │event   │            │       │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘            │       │
│  └──────────────────────────────────────────────────────────────────────┘       │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## 数据处理流程

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ 社交媒体  │   │  数据采集 │   │ 数据清洗  │   │ 情感分析  │   │ 话题聚类  │   │  结果存储 │
│  平台    │──▶│  Spider  │──▶│Processor │──▶│  Model   │──▶│  Cluster │──▶│ Database │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
     │              │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼              ▼
   小红书        爬取内容       文本清洗      Qwen2.5/BERT   BERT嵌入       话题事件
   抖音          爬取评论       中文分词      情感分类       Single-Pass    演化快照
   微博          数据同步       关键词提取    情绪识别       Faiss检索      合并日志
   B站           统一格式       停用词过滤    18种情绪       Qwen命名       统计分析
   知乎            ...            ...          ...           话题合并         ...
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Zayu2005/SentimentAnalysis.git
cd SentimentAnalysis
```

### 2. 创建虚拟环境

**方式一：使用 Conda (推荐)**
```bash
conda create -n sentiment python=3.10
conda activate sentiment
```

**方式二：使用 venv**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖

```bash
# 安装所有依赖 (CPU 版本)
pip install -r requirements.txt

# 安装 Playwright 浏览器 (爬虫需要)
playwright install chromium
```

**GPU 环境配置 (可选，用于加速模型推理)**

```bash
# 先安装 CUDA 版本的 PyTorch (根据你的 CUDA 版本选择)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 然后安装其他依赖
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，设置数据库密码
# Windows 用户可以用记事本打开
notepad .env
```

必须配置的项：
```env
MYSQL_DB_PWD=your_password_here
```

### 5. 初始化数据库

```sql
-- 创建数据库
CREATE DATABASE sentiment DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

```bash
# 执行话题聚类表迁移
mysql -u root -p sentiment < SentimentSpider/hot_news/database/migrations/003_topic_cluster.sql
```

### 6. 验证安装

```bash
# 测试数据库连接
python -m SentimentProcessor stats

# 测试模型加载 (首次运行会下载模型)
python test_qwen_base.py
```

## 项目结构

```
SentimentAnalysis/
├── SentimentSpider/          # 数据采集模块
│   ├── MediaCrawler/         # 多平台社交媒体爬虫
│   │   ├── media_platform/   # 各平台爬虫实现
│   │   ├── store/            # 数据存储
│   │   └── ...
│   └── hot_news/             # 热点新闻采集
│
├── SentimentProcessor/       # 数据预处理模块
│   ├── config/               # 配置管理
│   ├── database/             # 数据库操作
│   ├── processor/            # 预处理核心
│   │   ├── cleaner.py        # 文本清洗
│   │   ├── segmenter.py      # 中文分词
│   │   └── extractor.py      # 关键词提取
│   ├── utils/                # 工具类
│   │   ├── stopwords.py      # 停用词管理
│   │   └── slang.py          # 网络用语规范化
│   └── cli/                  # 命令行工具
│
├── SentimentModel/           # 情感分析模型
│   ├── config/               # 配置管理
│   ├── models/               # BERT 模型
│   ├── training/             # 模型训练
│   ├── inference/            # 模型推理
│   ├── qwen/                 # Qwen2.5 大模型
│   ├── database/             # 数据库操作
│   └── cli/                  # 命令行工具
│
├── TopicCluster/             # 话题聚类模块
│   ├── config/               # 配置管理
│   ├── database/             # 数据库操作
│   ├── cluster/              # 聚类核心
│   │   ├── embedder.py       # BERT 嵌入提取
│   │   ├── index.py          # Faiss 向量索引
│   │   ├── engine.py         # 增量聚类引擎
│   │   └── maintainer.py     # 话题合并/生命周期/统计
│   ├── llm/                  # LLM 话题命名
│   │   └── namer.py          # Qwen 话题描述生成
│   └── cli/                  # 命令行工具
│
├── .env.example              # 环境变量模板
├── requirements.txt          # 统一依赖文件
├── run_qwen_analyze.py       # Qwen 分析脚本
├── test_qwen_base.py         # 模型测试脚本
└── README.md
```

## 功能特性

### 数据采集 (SentimentSpider)

支持多个主流社交媒体平台的数据采集：

| 平台 | 支持内容 | 支持评论 |
|------|---------|---------|
| 小红书 | ✅ | ✅ |
| 抖音 | ✅ | ✅ |
| 快手 | ✅ | ✅ |
| 微博 | ✅ | ✅ |
| B站 | ✅ | ✅ |
| 贴吧 | ✅ | ✅ |
| 知乎 | ✅ | ✅ |

### 数据预处理 (SentimentProcessor)

- **文本清洗**: 移除 URL、邮箱、HTML、@提及、表情符号
- **文本规范化**: 繁简转换、网络用语规范化、重复字符压缩
- **中文分词**: jieba 分词、自定义词典、停用词过滤
- **关键词提取**: TF-IDF、TextRank

### 情感分析 (SentimentModel)

| 模型 | 说明 | 输出 |
|------|------|------|
| BERT | chinese-roberta-wwm-ext | 三分类 (正面/中性/负面) |
| Qwen2.5 | 1.5B 参数大模型 | 情感分数 + 18种情绪标签 |

**支持的 18 种情绪标签：**
- 正面：喜悦、兴奋、满足、感激、爱
- 负面：愤怒、厌恶、悲伤、恐惧、失望
- 中性：惊讶、困惑、好奇、期待、焦虑、平静、无聊、冷漠

### 话题聚类 (TopicCluster)

基于 HISEvent + RagSEDE 思路的增量话题聚类系统：

- **BERT 嵌入**: 使用 `chinese-roberta-wwm-ext` 提取 768 维 [CLS] 向量，L2 归一化
- **Single-Pass 聚类**: Faiss IndexFlatIP 检索最近话题质心，阈值匹配归入或新建话题
- **话题合并**: 全局话题对相似度检查，自动合并高相似话题
- **生命周期管理**: emerging → active → declining → ended 状态自动流转
- **LLM 命名**: Qwen2.5 自动生成话题名称、舆情描述和加权关键词
- **演化追踪**: 每日快照记录话题热度、情感趋势、平台分布变化

| 命令 | 功能 |
|------|------|
| `python -m TopicCluster cluster` | 增量聚类 |
| `python -m TopicCluster describe` | LLM 话题命名 |
| `python -m TopicCluster merge` | 合并相似话题 |
| `python -m TopicCluster evolve` | 更新生命周期/统计/演化快照 |
| `python -m TopicCluster stats` | 查看统计信息 |
| `python -m TopicCluster recluster` | 全量重聚类 |

## 使用示例

### 示例 1：完整的舆情分析工作流

```bash
# Step 1: 采集小红书数据
cd SentimentSpider/MediaCrawler
python main.py --platform xhs --keywords "新能源汽车" --type search

# Step 2: 回到根目录，进行数据预处理
cd ../..
python -m SentimentProcessor all

# Step 3: 情感分析
python run_qwen_analyze.py

# Step 4: 话题聚类
python -m TopicCluster cluster --batch-size 64

# Step 5: LLM 话题命名
python -m TopicCluster describe --all --include-ended

# Step 6: 话题合并 + 统计更新
python -m TopicCluster merge
python -m TopicCluster evolve

# Step 7: 查看统计结果
python -m SentimentProcessor stats
python -m TopicCluster stats
```

### 示例 2：只分析内容（不分析评论）

```bash
# 预处理内容
python -m SentimentProcessor content

# 情感分析内容
python run_qwen_analyze.py --type content
```

### 示例 3：试运行模式（不写入数据库）

```bash
# 预处理试运行
python -m SentimentProcessor all --dry-run

# 情感分析试运行
python run_qwen_analyze.py --dry-run
```

### 示例 4：批量处理大数据

```bash
# 使用较小的批次大小，避免内存溢出
python -m SentimentProcessor all -b 50
python run_qwen_analyze.py --batch-size 20
```

### 示例 5：Python API 使用

```python
# 文本预处理
from SentimentProcessor import TextCleaner, Segmenter, KeywordExtractor

cleaner = TextCleaner()
segmenter = Segmenter()
extractor = KeywordExtractor()

text = "这个产品真的太好用了！强烈推荐给大家 #好物分享#"

# 清洗文本
cleaned = cleaner.clean(text)
print(f"清洗后: {cleaned}")

# 分词
words = segmenter.segment(cleaned)
print(f"分词: {words}")

# 提取关键词
keywords = extractor.extract_tfidf(cleaned, topk=5)
print(f"关键词: {keywords}")
```

```python
# 情感分析
from SentimentModel import SentimentPredictor

predictor = SentimentPredictor(model_path="models/best_model")
result = predictor.predict("这个产品真的太好用了！")

print(f"情感: {result.label}")
print(f"置信度: {result.confidence:.2%}")
```

## 配置说明

### 环境变量

| 变量名 | 必须 | 默认值 | 说明 |
|--------|------|--------|------|
| `MYSQL_DB_HOST` | 否 | localhost | 数据库主机 |
| `MYSQL_DB_PORT` | 否 | 3306 | 数据库端口 |
| `MYSQL_DB_USER` | 否 | root | 数据库用户 |
| `MYSQL_DB_PWD` | **是** | - | 数据库密码 |
| `MYSQL_DB_NAME` | 否 | sentiment | 数据库名 |
| `HF_ENDPOINT` | 否 | https://hf-mirror.com | HuggingFace 镜像 |

### 模型下载

首次运行会自动下载模型，国内用户建议配置 HuggingFace 镜像：

```env
HF_ENDPOINT=https://hf-mirror.com
```

模型大小参考：
- `Qwen2.5-1.5B-Instruct`: ~3GB
- `chinese-roberta-wwm-ext`: ~400MB

## 数据库表结构

### 原始数据表

| 表名 | 说明 |
|------|------|
| `unified_content` | 统一内容表 (各平台帖子/笔记) |
| `unified_comment` | 统一评论表 (各平台评论) |
| `xhs_note` | 小红书笔记 |
| `xhs_note_comment` | 小红书评论 |
| `douyin_aweme` | 抖音视频 |
| `douyin_aweme_comment` | 抖音评论 |

### 预处理字段

| 字段 | 说明 |
|------|------|
| `title_cleaned` / `content_cleaned` | 清洗后的文本 |
| `title_segmented` / `content_segmented` | 分词结果 (JSON) |
| `keywords` | 关键词 (JSON) |
| `sentiment` | 情感标签 (positive/neutral/negative) |
| `sentiment_score` | 情感分数 (-1.0 ~ 1.0) |
| `emotion_tags` | 情绪标签 (JSON) |

### 话题聚类表

| 表名 | 说明 |
|------|------|
| `topic_event` | 话题事件 (质心嵌入、名称、状态、情感/热度统计) |
| `topic_evolution` | 话题演化快照 (每日热度、情感、平台分布) |
| `topic_merge_log` | 话题合并日志 (合并记录、相似度、关键词) |

## 常见问题

### Q: 模型下载太慢？

配置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
或在 `.env` 文件中设置。

### Q: GPU 内存不足？

1. 使用更小的批次大小：`--batch-size 10`
2. 使用 CPU 推理：在 `.env` 中设置 `DEVICE=cpu`
3. 使用更小的模型：`Qwen2.5-0.5B-Instruct`

### Q: 数据库连接失败？

1. 检查 MySQL 服务是否启动
2. 检查 `.env` 中的密码是否正确
3. 检查数据库是否已创建

### Q: 爬虫无法运行？

1. 确保已安装 Playwright：`playwright install chromium`
2. 某些平台需要登录 Cookie，请查看 MediaCrawler 文档

## 项目规划

- [x] 数据采集模块 (SentimentSpider)
- [x] 数据预处理模块 (SentimentProcessor)
- [x] 情感分析模型 (SentimentModel)
- [x] 话题聚类与事件监测 (TopicCluster)
- [ ] API 服务 (SentimentAPI)
- [ ] 可视化仪表板 (SentimentDashboard)

## 作者

**Zayu2005** - [GitHub](https://github.com/Zayu2005)

## 许可证

MIT License
