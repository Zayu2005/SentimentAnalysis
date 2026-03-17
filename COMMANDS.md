# SentimentAnalysis 完整操作指南

## 一、数据库配置

### 配置文件位置
- 根目录 `.env` 文件

### 环境变量
```env
# MySQL 数据库
MYSQL_DB_HOST=202.200.205.108
MYSQL_DB_PORT=3306
MYSQL_DB_USER=sentiment
MYSQL_DB_PWD=M22hbxnfhDyRCwaT
MYSQL_DB_NAME=sentiment

# HuggingFace
HF_ENDPOINT=https://hf-mirror.com

# 模型配置
QWEN_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
BERT_MODEL_NAME=hfl/chinese-roberta-wwm-ext

# Neo4j (可选)
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=

# DeepSeek API (可选)
DEEPSEEK_API_KEY=
DEEPSEEK_API_BASE=https://api.deepseek.com
```

---

## 二、数据迁移 (从 localhost 到远程)

### 方式1: 使用 mysqldump (推荐)

```bash
# 导出单个表
mysqldump -u root -p1234 sentiment 表名 --single-transaction | mysql -h 远程IP -u 用户名 -p密码 目标数据库

# 批量导出所有表
for table in xhs_note weibo_note bilibili_video kuaishou_video tieba_note zhihu_content; do
  mysqldump -u root -p1234 sentiment $table | mysql -h 202.200.205.108 -u sentiment -pM22hbxnfhDyRCwaT sentiment
done
```

### 方式2: SQL导入 (统一表)

```sql
-- 迁移内容到统一表
INSERT INTO unified_content (platform, content_id, content_type, user_id, nickname, title, content, ...)
SELECT platform, ... FROM 原始表;

-- 迁移评论到统一表
INSERT INTO unified_comment (platform, comment_id, content_id, ...)
SELECT platform, ... FROM 原始评论表;
```

---

## 三、完整数据处理流程

### Step 1: 数据预处理

```bash
# 使用 conda 环境
conda activate sentiment

# 处理内容 + 评论
python -m SentimentProcessor all

# 或分别处理
python -m SentimentProcessor content    # 只处理内容
python -m SentimentProcessor comments   # 只处理评论
python -m SentimentProcessor stats      # 查看统计
```

### Step 2: 情感分析 (可选)

```bash
# 使用 Qwen 进行情感分析
python run_qwen_analyze.py

# 或使用 BERT 模型
python -m SentimentModel predict
```

### Step 3: 话题聚类

```bash
# 增量聚类 (不删除已有话题)
python -m TopicCluster cluster

# 全量重聚类 (删除所有话题)
python -m TopicCluster recluster

# LLM 自动命名话题
python -m TopicCluster describe --all

# 合并相似话题
python -m TopicCluster merge --dry-run  # 预览
python -m TopicCluster merge             # 执行合并

# 更新话题统计和生命周期
python -m TopicCluster evolve

# 生成词云
python -m TopicCluster wordcloud --all

# 查看统计
python -m TopicCluster stats
```

### Step 4: 知识图谱 (可选)

```bash
# 实体关系抽取
python -m KnowledgeGraph extract --topic-id 1 --limit 50

# 构建 Neo4j 图谱
python -m KnowledgeGraph build --topic-id 1 --clear

# 完整流水线
python -m KnowledgeGraph pipeline --topic-id 1

# 查询图谱
python -m KnowledgeGraph query --topic-id 1

# 查看统计
python -m KnowledgeGraph stats
```

---

## 四、数据采集 (SentimentSpider)

```bash
cd SentimentSpider

# 获取热点新闻
python -m hot_news.cli.main fetch

# 指定平台获取热点
python -m hot_news.cli.main fetch xhs wb bili -l 50

# 同步数据到统一表
python -m hot_news.cli.main sync

# 指定平台同步
python -m hot_news.cli.main sync xhs wb

# 全量同步 (清空重建)
python -m hot_news.cli.main sync --full

# 查看同步统计
python -m hot_news.cli.main sync-stats

# 触发爬虫
cd MediaCrawler
python main.py --platform xhs --keywords "新能源汽车" --type search
```

---

## 五、常用数据库查询

### 查看各平台数据分布

```sql
-- 原始表
SELECT platform, COUNT(*) as cnt FROM unified_content GROUP BY platform;

-- 已聚类内容
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

---

## 六、一键执行完整流程

```bash
#!/bin/bash
# 完整流水线脚本

echo "=== Step 1: 数据预处理 ==="
python -m SentimentProcessor all

echo "=== Step 2: 情感分析 ==="
python run_qwen_analyze.py

echo "=== Step 3: 话题聚类 ==="
python -m TopicCluster cluster
python -m TopicCluster describe --all
python -m TopicCluster evolve

echo "=== Step 4: 查看结果 ==="
python -m TopicCluster stats
```

---

## 七、Troubleshooting

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

---

## 八、表结构说明

### 核心业务表

| 表名 | 说明 |
|------|------|
| xhs_note / weibo_note / bilibili_video / ... | 各平台原始数据 |
| unified_content | 统一内容表 |
| unified_comment | 统一评论表 |
| processed_content | 预处理结果 (清洗/分词/关键词) |
| processed_comment | 评论预处理结果 |
| topic_event | 话题事件 |
| topic_evolution | 话题演化快照 |
| kg_extraction | 实体关系抽取结果 |
