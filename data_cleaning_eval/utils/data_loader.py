# -*- coding: utf-8 -*-
"""
数据集加载工具

对应论文章节: 5.4.1.4 数据清洗流水线评估
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from ..config import config


def load_sighan_pku(path: Optional[str] = None) -> List[Tuple[str, List[str]]]:
    """
    加载 SIGHAN 2005 PKU 分词数据集

    格式: 每行为已分词文本，词语间以空格分隔
          兼容全角空格(\\u3000)和半角空格

    Args:
        path: 数据文件路径 (默认使用配置)

    Returns:
        [(原始句子, [词1, 词2, ...]), ...]
    """
    file_path = Path(path) if path else config.resolve(config.sighan_pku_test_path)
    if not file_path.exists():
        raise FileNotFoundError(f"SIGHAN PKU 数据文件不存在: {file_path}")

    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = _split_sighan_line(line)
            original = "".join(tokens)
            if tokens and original.strip():
                results.append((original, tokens))

    return results


def load_nlpcc_weibo(path: Optional[str] = None) -> List[Tuple[str, List[str]]]:
    """
    加载 NLPCC 2016 微博分词数据集

    Args:
        path: 数据文件路径

    Returns:
        [(原始句子, [词1, 词2, ...]), ...]
    """
    file_path = Path(path) if path else config.resolve(config.nlpcc_test_path)
    if not file_path.exists():
        raise FileNotFoundError(f"NLPCC 微博数据文件不存在: {file_path}")

    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = _split_sighan_line(line)
            original = "".join(tokens)
            if tokens and original.strip():
                results.append((original, tokens))

    return results


def load_lcqmc(
    positive_n: int = 500,
    negative_n: int = 500,
    path: Optional[str] = None,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """
    加载 LCQMC 相似句对数据集

    格式: 句子1\\t句子2\\t标签 (1=相似/0=不相似)

    Args:
        positive_n: 正样本数量
        negative_n: 负样本数量
        path: 数据文件路径

    Returns:
        (正样本列表, 负样本列表), 每个元素为 (句子1, 句子2, 标签)
    """
    file_path = Path(path) if path else config.resolve(config.lcqmc_test_path)
    if not file_path.exists():
        raise FileNotFoundError(f"LCQMC 数据文件不存在: {file_path}")

    positives = []
    negatives = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            s1, s2, label = parts[0], parts[1], int(parts[2])

            if label == 1 and len(positives) < positive_n:
                positives.append((s1, s2, label))
            elif label == 0 and len(negatives) < negative_n:
                negatives.append((s1, s2, label))

            if len(positives) >= positive_n and len(negatives) >= negative_n:
                break

    return positives, negatives


def load_stopwords(path: Optional[str] = None) -> set:
    """加载停用词表"""
    file_path = Path(path) if path else config.resolve(config.stopword_path)
    if not file_path.exists():
        raise FileNotFoundError(f"停用词文件不存在: {file_path}")

    stopwords = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.add(word)
    return stopwords


def load_sentiment_words(path: Optional[str] = None) -> set:
    """加载情感词典"""
    file_path = Path(path) if path else config.resolve(config.sentiment_dict_path)
    if not file_path.exists():
        print(f"[警告] 情感词典不存在: {file_path}, 将使用内置情感词")
        return _get_builtin_sentiment_words()

    words = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word and not word.startswith("#"):
                words.add(word.split("\t")[0])
    return words


def load_custom_dict(path: Optional[str] = None) -> List[str]:
    """加载自定义分词词典"""
    file_path = Path(path) if path else config.resolve(config.custom_dict_path)
    if not file_path.exists():
        return []

    words = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                word = line.split()[0]
                if word:
                    words.append(word)
    return words


def load_html_test_data(path: Optional[str] = None) -> Dict:
    """加载 HTML 清洗测试数据 (JSON格式)"""
    file_path = Path(path) if path else config.resolve(config.html_test_data_path)
    if not file_path.exists():
        raise FileNotFoundError(f"HTML测试数据文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_stopword_test_data(path: Optional[str] = None) -> Dict:
    """加载停用词过滤测试数据 (JSON格式)"""
    file_path = Path(path) if path else config.resolve(config.stopword_test_data_path)
    if not file_path.exists():
        raise FileNotFoundError(f"停用词测试数据文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_html_test_data(output_path: Optional[str] = None, n: int = 500) -> Dict:
    """
    生成 HTML 清洗测试数据 (自建模拟数据)

    用于没有真实标注数据时的评估

    Returns:
        {
            "raw_texts": [...],
            "cleaned_texts": [...],
            "gold_texts": [...]
        }
    """
    import re

    html_patterns = [
        ("<p>{text}</p>", "{text}"),
        ("<div class='content'>{text}</div>", "{text}"),
        ("<a href='http://example.com'>{text}</a>", "{text}"),
        ("<strong>{text}</strong><br/>", "{text}"),
        ("{text}<img src='x.jpg'/>", "{text}"),
        ("{text}&nbsp;&nbsp;{more}", "{text} {more}"),
        ("{text}@用户名 评论", "{text} 评论"),
        ("{text} #话题标签# 更多内容", "{text} 话题标签 更多内容"),
        ("{text} https://t.cn/abc123 结束", "{text} 结束"),
        ("{text} 😂😊👍 表情", "{text} 表情"),
        ("<script>alert('xss')</script>{text}", "{text}"),
        ("&lt;{text}&gt;", "{text}"),
        ("{text}&amp;更多", "{text}更多"),
    ]

    sample_texts = [
        "这款产品的售后服务真的太差了，客服态度恶劣",
        "今天天气不错，出门散步心情很好",
        "新发布的手机配置很强，但价格有点贵",
        "这家餐厅的菜品味道一般，环境还可以",
        "最近看的一部电影非常精彩，推荐大家去看",
        "公司宣布了新的福利政策，员工都很开心",
        "交通拥堵问题越来越严重，需要改善",
        "这个品牌的护肤品效果不错，值得购买",
        "学校食堂的饭菜质量有所提升",
        "周末去公园玩，人很多但是风景很美",
    ]

    import random
    random.seed(42)

    raw_texts = []
    cleaned_texts = []
    gold_texts = []

    for i in range(n):
        base_text = random.choice(sample_texts)
        pattern = random.choice(html_patterns)

        more = random.choice(sample_texts)[:20] if "{more}" in pattern[0] else ""

        raw = pattern[0].format(text=base_text, more=more)
        gold = pattern[1].format(text=base_text, more=more)

        raw_texts.append(raw)
        gold_texts.append(gold)

        cleaned_texts.append(gold)

    data = {
        "raw_texts": raw_texts,
        "cleaned_texts": cleaned_texts,
        "gold_texts": gold_texts,
    }

    out_path = Path(output_path) if output_path else config.resolve(config.html_test_data_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[DataLoader] 已生成 HTML 测试数据: {out_path} ({n} 条)")
    return data


def generate_stopword_test_data(output_path: Optional[str] = None, n: int = 500) -> Dict:
    """
    生成停用词过滤测试数据 (自建模拟数据)

    Returns:
        {
            "tokenized_texts": [[...], ...],
            "gold_filtered": [[...], ...]
        }
    """
    import random
    random.seed(42)

    sentences = [
        ["我", "觉得", "这个", "产品", "的", "质量", "非常", "好"],
        ["今天", "的", "天气", "真", "是", "不错", "啊"],
        ["这", "家", "店", "的", "服务", "态度", "很", "差"],
        ["虽然", "价格", "有点", "贵", "但", "是", "值", "得"],
        ["我们", "都", "认为", "这个", "品牌", "很", "不错"],
        ["在", "这个", "平台", "上", "买", "到", "了", "好", "东西"],
        ["因为", "它", "的", "设计", "很", "好看", "所以", "买", "了"],
        ["如果", "不", "考虑", "价格", "的话", "还", "可以"],
        ["对于", "学生", "来说", "可能", "有", "点", "贵"],
        ["从", "整体", "来看", "这", "个", "产品", "还是", "OK", "的"],
    ]

    default_stopwords = {
        "的", "了", "是", "在", "我", "有", "和", "就", "不", "人",
        "都", "一", "上", "也", "很", "到", "说", "要", "去", "你",
        "会", "着", "没", "看", "好", "自", "己", "这", "那", "他",
        "们", "什", "么", "怎", "为", "把", "被", "让", "给", "从",
        "而", "但", "是", "可", "以", "能", "会", "对", "与", "或",
    }

    tokenized_texts = []
    gold_filtered = []

    for i in range(n):
        tokens = random.choice(sentences).copy()
        random.shuffle(tokens)
        tokenized_texts.append(tokens[:])

        filtered = [t for t in tokens if t not in default_stopwords]
        gold_filtered.append(filtered)

    data = {
        "tokenized_texts": tokenized_texts,
        "gold_filtered": gold_filtered,
    }

    out_path = Path(output_path) if output_path else config.resolve(config.stopword_test_data_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[DataLoader] 已生成停用词测试数据: {out_path} ({n} 条)")
    return data


def _split_sighan_line(line: str) -> List[str]:
    """
    分割 SIGHAN/NLPCC 格式的分词行

    兼容全角空格(\\u3000)和半角空格
    """
    line = line.replace("\u3000", " ")
    tokens = [t for t in line.split(" ") if t.strip()]
    return tokens


def _get_builtin_sentiment_words() -> set:
    """获取内置情感词集合 (BosonNLP 精简版)"""
    return {
        "喜欢", "爱", "开心", "快乐", "高兴", "幸福", "满意", "美好", "精彩",
        "优秀", "棒", "赞", "好", "不错", "可以", "支持", "推荐", "期待", "希望",
        "信任", "感谢", "感激", "感动", "温暖", "舒适", "方便", "实用", "划算",
        "便宜", "实惠", "超值", "物美价廉", "性价比高", "值得", "必须", "应该",
        "讨厌", "厌恶", "反感", "愤怒", "生气", "失望", "难过", "伤心", "痛苦",
        "糟糕", "差劲", "垃圾", "烂", "坑", "骗", "假", "劣质", "低劣", "难用",
        "不便", "麻烦", "复杂", "混乱", "错误", "失败", "崩溃", "卡顿", "延迟",
        "昂贵", "贵", "不值", "浪费", "后悔", "遗憾", "担心", "焦虑", "害怕",
        "恐惧", "紧张", "压力", "疲惫", "累", "烦", "无聊", "寂寞", "孤独",
        "惊讶", "震惊", "意外", "惊喜", "兴奋", "激动", "热情", "冷漠", "冷淡",
        "鄙视", "轻视", "嘲笑", "讽刺", "挖苦", "指责", "批评", "抱怨", "投诉",
        "维权", "退款", "赔偿", "道歉", "解释", "澄清", "否认", "承认", "隐瞒",
    }
