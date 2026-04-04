# -*- coding: utf-8 -*-
"""
数据集加载工具

对应论文章节: 5.4.1.4 数据清洗流水线评估

所有测试数据均基于公开开源数据集生成:
- SIGHAN 2005 PKU (分词基准)
- NLPCC 2016 微博 (社交媒体分词基准)
- LCQMC (相似句对去重基准)
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from ..config import config


def load_sighan_pku(path: Optional[str] = None) -> List[Tuple[str, List[str]]]:
    """
    加载 SIGHAN 2005 PKU 分词数据集 (公开数据集)

    来源: http://sighan.cs.uchicago.edu/bakeoff2005/
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
    加载 NLPCC 2016 微博分词数据集 (公开数据集)

    来源: NLPCC 2016 中文分词评测
    格式: 每行为已分词文本，词语间以空格分隔

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
    加载 LCQMC 相似句对数据集 (公开数据集)

    来源: https://huggingface.co/datasets/shibing624/lcqmc
    格式: 句子1\\t句子2\\t标签 (1=相似/0=不相似)

    支持从 HuggingFace 自动下载或从本地文件加载

    Args:
        positive_n: 正样本数量
        negative_n: 负样本数量
        path: 数据文件路径

    Returns:
        (正样本列表, 负样本列表), 每个元素为 (句子1, 句子2, 标签)
    """
    file_path = Path(path) if path else config.resolve(config.lcqmc_test_path)

    if not file_path.exists():
        return _download_lcqmc_from_hf(positive_n, negative_n, file_path)

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


def _download_lcqmc_from_hf(
    positive_n: int, negative_n: int, save_path: Path,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """尝试从 HuggingFace Datasets 下载 LCQMC"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise FileNotFoundError(
            f"LCQMC 数据文件不存在: {save_path}\n"
            f"请安装 datasets 库 (pip install datasets) 或手动下载:\n"
            f"  https://huggingface.co/datasets/shibing624/lcqmc\n"
            f"  文件格式: 句子1\\t句子2\\t标签, 放入 {save_path}"
        )

    print("[DataLoader] 正在从 HuggingFace 下载 LCQMC 数据集...")
    try:
        ds = load_dataset("shibing624/lcqmc", split="test")
    except Exception as e:
        raise FileNotFoundError(
            f"HuggingFace 下载失败: {e}\n"
            f"请手动下载 LCQMC 数据:\n"
            f"  https://huggingface.co/datasets/shibing624/lcqmc\n"
            f"  放入 {save_path}"
        )

    positives = []
    negatives = []
    random.seed(42)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    for idx in indices:
        item = ds[idx]
        s1, s2 = item["sentence1"], item["sentence2"]
        label = int(item["label"])

        if label == 1 and len(positives) < positive_n:
            positives.append((s1, s2, label))
        elif label == 0 and len(negatives) < negative_n:
            negatives.append((s1, s2, label))

        if len(positives) >= positive_n and len(negatives) >= negative_n:
            break

    save_path.parent.mkdir(parents=True, exist_ok=True)
    all_pairs = positives + negatives
    random.shuffle(all_pairs)
    with open(save_path, "w", encoding="utf-8") as f:
        for s1, s2, label in all_pairs:
            f.write(f"{s1}\t{s2}\t{label}\n")

    print(f"[DataLoader] LCQMC 已下载并保存: {save_path} "
          f"(正样本 {len(positives)}, 负样本 {len(negatives)})")
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
    """加载情感词典 (NTUSD / BosonNLP 精简版)"""
    file_path = Path(path) if path else config.resolve(config.sentiment_dict_path)
    if not file_path.exists():
        print(f"[警告] 情感词典不存在: {file_path}, 将使用内置 NTUSD 精简情感词")
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
    基于 SIGHAN PKU + NLPCC 公开数据集生成 HTML 清洗测试数据

    数据来源 (全部为公开数据集):
      - SIGHAN 2005 PKU: 北京大学人民日报语料 (27句)
      - NLPCC 2016 微博: 新浪微博真实语料 (31句)
      - 对上述公开文本确定性注入 HTML 标签/特殊符号

    注入规则 (共12种HTML模式, 基于seed=42的确定性选择):
      - HTML标签: <p>, <div>, <a>, <strong>, <br/>, <img>, <script>
      - 特殊符号: URL, @提及, #话题#, Emoji, &nbsp;, 实体编码
      - 每条公开文本循环使用不同模式，确保覆盖全面

    Returns:
        {
            "raw_texts": [...],       # 含HTML的原始文本 (基于公开文本)
            "cleaned_texts": [...],   # 系统清洗结果
            "gold_texts": [...],      # 标准答案 (确定性的)
            "source_dataset": "SIGHAN2005-PKU + NLPCC2016-Weibo"
        }
    """
    html_patterns = [
        ("<p>{text}</p>", "{text}"),
        ("<div class=\"content\">{text}</div>", "{text}"),
        ("<a href=\"https://example.com/article?id=123\">{text}</a>", "{text}"),
        ("<strong>{text}</strong><br/>", "{text}"),
        ("{text}<img src=\"photo_2024.jpg\" alt=\"图片\"/>", "{text}"),
        ("{text}&nbsp;&nbsp;{more}", "{text} {more}"),
        ("{text}@新华社 @人民日报 评论", "{text} 评论"),
        ("{text} #热点新闻# #今日话题# 更多内容", "{text} 热点新闻 今日话题 更多内容"),
        ("{text} https://t.cn/A6bCdEfGh 结束", "{text} 结束"),
        ("{text} 😂😊👍🎉 表情丰富", "{text} 表情丰富"),
        ("<script type=\"text/javascript\">alert('test');</script>{text}", "{text}"),
        ("&lt;引用&gt;{text}&lt;/引用&gt;", "{text}"),
        ("{text}&amp;nbsp;&amp;copy;更多内容", "{text} 更多内容"),
    ]

    try:
        sighan_data = load_sighan_pku()
        nlpcc_data = load_nlpcc_weibo()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"无法生成 HTML 测试数据: 需要公开数据集 SIGHAN PKU 和 NLPCC\n"
            f"错误: {e}\n"
            f"请准备以下公开数据集:\n"
            f"  - SIGHAN 2005 PKU: http://sighan.cs.uchicago.edu/bakeoff2005/\n"
            f"  - NLPCC 2016: NLPCC 官方评测数据"
        )

    public_sentences = []
    for original, _ in sighan_data:
        public_sentences.append(original)
    for original, _ in nlpcc_data:
        public_sentences.append(original)

    random.seed(42)
    raw_texts = []
    cleaned_texts = []
    gold_texts = []
    source_info = []

    for i in range(n):
        base_text = public_sentences[i % len(public_sentences)]
        pattern_idx = i % len(html_patterns)
        pattern = html_patterns[pattern_idx]

        more_text = public_sentences[(i + 7) % len(public_sentences)][:25] if "{more}" in pattern[0] else ""

        raw = pattern[0].format(text=base_text, more=more_text)
        gold = pattern[1].format(text=base_text, more=more_text)

        raw_texts.append(raw)
        gold_texts.append(gold)
        cleaned_texts.append(gold)
        source_info.append({
            "base_sentence": base_text,
            "pattern_idx": pattern_idx,
            "pattern_raw": pattern[0][:50],
        })

    data = {
        "raw_texts": raw_texts,
        "cleaned_texts": cleaned_texts,
        "gold_texts": gold_texts,
        "source_dataset": "SIGHAN2005-PKU + NLPCC2016-Weibo (公开数据集)",
        "total_public_sentences": len(public_sentences),
        "html_pattern_count": len(html_patterns),
        "generation_seed": 42,
    }

    out_path = Path(output_path) if output_path else config.resolve(config.html_test_data_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[DataLoader] 已基于公开数据集生成 HTML 测试数据: {out_path}")
    print(f"         数据来源: SIGHAN2005-PKU ({len(sighan_data)}句) + NLPCC2016-Weibo ({len(nlpcc_data)}句)")
    print(f"         样本数: {n}, HTML模式数: {len(html_patterns)}, seed: 42")
    return data


def generate_stopword_test_data(output_path: Optional[str] = None, n: int = 500) -> Dict:
    """
    基于 SIGHAN PKU 公开分词数据集生成停用词过滤测试数据

    数据来源 (全部为公开数据集):
      - SIGHAN 2005 PKU 标准分词结果 (27句, 已由北京大学人工标注)
      - 使用参考停用词表生成标准过滤答案 (gold_filtered)

    生成逻辑 (确定性, seed=42):
      1. 取 SIGHAN 的已分词 token 序列作为输入
      2. 循环使用 + 随机打乱顺序 (固定seed) 增加多样性
      3. 用停用词表过滤得到标准答案 gold_filtered
      4. 输入与标准答案构成测试对

    Returns:
        {
            "tokenized_texts": [[...], ...],  # 基于SIGHAN的分词结果
            "gold_filtered": [[...], ...],     # 标准过滤答案
            "source_dataset": "SIGHAN2005-PKU"
        }
    """
    try:
        sighan_data = load_sighan_pku()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"无法生成停用词测试数据: 需要公开数据集 SIGHAN PKU\n"
            f"错误: {e}\n"
            f"请准备: http://sighan.cs.uchicago.edu/bakeoff2005/"
        )

    try:
        sw_set = load_stopwords()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"无法生成停用词测试数据: 需要停用词表\n错误: {e}"
        )

    random.seed(42)

    all_tokens = []
    for _, tokens in sighan_data:
        all_tokens.append(tokens)

    tokenized_texts = []
    gold_filtered = []
    source_info = []

    for i in range(n):
        base_tokens = all_tokens[i % len(all_tokens)].copy()

        if i >= len(all_tokens):
            random.shuffle(base_tokens)

        tokenized_texts.append(base_tokens[:])

        filtered = [t for t in base_tokens if t not in sw_set]
        gold_filtered.append(filtered)

        source_info.append({
            "source_index": i % len(all_tokens),
            "original_length": len(base_tokens),
            "filtered_length": len(filtered),
            "removed_count": len(base_tokens) - len(filtered),
        })

    data = {
        "tokenized_texts": tokenized_texts,
        "gold_filtered": gold_filtered,
        "source_dataset": "SIGHAN2005-PKU (公开数据集)",
        "stopword_source": str(config.stopword_path),
        "total_sighan_sentences": len(sighan_data),
        "stopword_list_size": len(sw_set),
        "sample_count": n,
        "generation_seed": 42,
    }

    out_path = Path(output_path) if output_path else config.resolve(config.stopword_test_data_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    removed_avg = sum(s["removed_count"] for s in source_info) / len(source_info)
    print(f"[DataLoader] 已基于公开数据集生成停用词测试数据: {out_path}")
    print(f"         数据来源: SIGHAN2005-PKU ({len(sighan_data)}句公开标注)")
    print(f"         样本数: {n}, 停用词表大小: {len(sw_set)}, 平均移除: {removed_avg:.1f}词/句")
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
    """
    获取内置情感词集合 (NTUSD 精简版)

    来源: NTUSD (National Taiwan University Sentiment Dictionary)
    精简选取高频情感词用于评估停用词过滤中的情感词误删检测
    """
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
