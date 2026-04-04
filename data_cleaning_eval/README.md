# 数据清洗质量评估模块

对应论文章节: **5.4.1.4 数据清洗流水线评估**

## 功能概述

本模块用于对中文舆情文本数据清洗流水线的各环节进行定量评估，输出结构化的评估报告。

### 评估环节

| 环节 | 评估内容 | 评估指标 |
|------|---------|---------|
| HTML清洗 | 标签/URL/邮箱/Emoji等过滤 | Accuracy |
| 分词质量 | jieba 默认模式 vs +自定义词典 | P / R / F1 |
| 停用词过滤 | 过滤精度与情感词误删率 | Accuracy / FP Rate / 情感词损失率 |
| 近似去重 | SimHash vs TF-IDF余弦相似度 | P / R / F1 / PR曲线 |

## 项目结构

```
data_cleaning_eval/
├── main.py                    # 主入口
├── config.py                  # 路径配置、阈值参数
├── requirements.txt           # 依赖
├── evaluators/
│   ├── __init__.py
│   ├── html_cleaner_eval.py   # HTML/特殊符号清洗评估
│   ├── segmentation_eval.py   # 分词质量评估（F1）
│   ├── stopword_eval.py       # 停用词过滤评估
│   └── dedup_eval.py          # 近似去重评估
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         # 数据集加载工具
│   ├── metrics.py             # 通用指标计算
│   └── report_generator.py    # 报告生成（CSV+JSON+控制台）
└── output/                     # 输出目录
    ├── eval_report.csv
    ├── eval_report.json
    ├── pr_curve_SimHash.png
    └── pr_curve_Cosine.png
```

## 快速开始

### 1. 安装依赖

```bash
cd data_cleaning_eval
pip install -r requirements.txt
```

### 2. 准备数据集

#### SIGHAN 2005 PKU 分词测试集

下载地址: http://sighan.cs.uchicago.edu/bakeoff2005/

文件名: `pku_test_gold.utf8`

放置路径: `data/sighan2005/pku_test_gold.utf8`

格式示例:
```
迈向 充满 希望 的 新 世纪
新华社 上海 二月 十日 电
```

#### NLPCC 2016 微博分词测试集

联系 NLPCC 官网获取评测数据

放置路径: `data/nlpcc2016/weibo_test_gold.txt`

#### LCQMC 相似句对数据集

下载地址: https://huggingface.co/datasets/shibing624/lcqmc

格式: `句子1\t句子2\t标签` (1=语义相似/0=不相似)

放置路径: `data/lcqmc/test.txt`

#### 停用词表

可使用哈工大停用词表或从项目 SentimentProcessor 中提取

放置路径: `dict/stopwords.txt`

#### (可选) 自定义词典

每行一个词，格式: `词语 词频 词性`

放置路径: `dict/custom_dict.txt`

#### (可选) 情感词典

用于检测停用词过滤中的情感词误删情况

放置路径: `dict/sentiment_words.txt`

### 3. 运行评估

```bash
# 运行完整评估 (推荐)
python -m data_cleaning_eval

# 仅检查数据文件
python -m data_cleaning_eval --check-only

# 仅运行单个模块
python -m data_cleaning_eval --module html      # HTML清洗
python -m data_cleaning_eval --module seg       # 分词评估
python -m data_cleaning_eval --module stopword   # 停用词过滤
python -m data_cleaning_eval --module dedup      # 去重评估

# 自定义阈值
python -m data_cleaning_eval --simhash-threshold 5 --cosine-threshold 0.80
```

## 输出说明

### 控制台报告 (Rich 表格)

```
┌─────────────────┬──────────────┬──────────────┬──────────────────┐
│ 清洗环节        │ 评估数据集    │ 指标          │ 结果             │
├─────────────────┼──────────────┼──────────────┼──────────────────┤
│ HTML清洗        │ 自建500条    │ 准确率        │ 98.50%           │
│ 分词(SIGHAN PKU)│ 默认模式     │ F1           │ 92.30%           │
│ 分词(NLPCC微博) │ +自定义词典  │ F1           │ 85.60%           │
│ 停用词过滤      │ 自建500条    │ 准确率/误删率  │ 95.20% / 3.10%   │
│ 近似去重(SimHash)│ LCQMC 1000对│ P/R/F1       │ 85.3/78.2/81.5   │
│ 近似去重(Cosine) │ LCQMC 1000对│ P/R/F1       │ 88.7/82.1/85.2   │
└─────────────────┴──────────────┴──────────────┴──────────────────┘
```

### CSV 报告 (`output/eval_report.csv`)

每行一个评估结果，列为: module / dataset / metric / value

### JSON 报告 (`output/eval_report.json`)

完整嵌套 JSON，包含所有中间数据和错误案例

### PR 曲线图

- `output/pr_curve_SimHash.png`
- `output/pr_curve_Cosine.png`

## 核心算法说明

### 分词 F1 计算

采用基于**词边界**的标准方法:

```
P = 正确预测词边界数 / 预测总词边界数
R = 正确预测词边界数 / 标准词边界数
F1 = 2 * P * R / (P + R)
```

每个词用 `(起始位置, 结束位置)` 表示其边界，通过集合交集计算匹配数。

### SimHash 去重

1. 对中文文本进行 jieba 分词
2. 计算每个分词结果的 SimHash 指纹
3. 计算两个指纹的汉明距离
4. 距离 <= 阈值 → 判定为重复

### TF-IDF 余弦去重

1. 使用 jieba 分词作为 tokenizer 构建 TF-IDF 向量
2. 计算两个句子的余弦相似度
3. 相似度 >= 阈值 → 判定为重复

## 配置参数

在 `config.py` 中修改以下参数:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `simhash_threshold` | 3 | SimHash 汉明距离阈值 |
| `cosine_threshold` | 0.75 | 余弦相似度阈值 |
| `lcqmc_positive_samples` | 500 | LCQMC 正样本数量 |
| `lcqmc_negative_samples` | 500 | LCQMC 负样本数量 |

## 注意事项

1. **SIGHAN 数据集分隔符**: 兼容全角空格(`\u3000`)和半角空格
2. **jieba 自定义词典**: 每行 `词语 词频 词性`，后两项可省略
3. **SimHash 必须先分词**: 不能直接对原始字符串计算哈希
4. **LCQMC 正负样本均衡**: 各取 500 对避免类别不平衡
5. **自动生成测试数据**: HTML 和停用词评估支持自动生成模拟测试数据
