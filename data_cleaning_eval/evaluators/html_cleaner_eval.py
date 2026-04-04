# -*- coding: utf-8 -*-
"""
HTML/特殊符号清洗质量评估

对应论文章节: 5.4.1.4 数据清洗流水线评估 - HTML清洗环节
"""

import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from ..utils.data_loader import load_html_test_data, generate_html_test_data
from ..utils.metrics import EvalResult, calc_accuracy


class HTMLCleanerEvaluator:
    """HTML 标签和特殊符号清洗评估器"""

    def __init__(self):
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
        )
        self.email_pattern = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
        self.mention_pattern = re.compile(r'@[\w\u4e00-\u9fff]+')
        self.hashtag_pattern = re.compile(r'#([^#\s]+)#?')
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U00002600-\U000026FF"
            "\U0001F000-\U0001F02F"
            "]+",
            flags=re.UNICODE
        )
        self.platform_emoji_pattern = re.compile(r'\[[^\]]{1,10}R\]')

    def clean(self, text: str) -> str:
        """执行系统清洗 (复用 SentimentProcessor 的逻辑)"""
        if not text:
            return ""
        result = text
        result = self.html_pattern.sub('', result)
        result = self.url_pattern.sub('', result)
        result = self.email_pattern.sub('', result)
        result = self.mention_pattern.sub('', result)
        result = self.hashtag_pattern.sub(r'\1', result)
        result = self.emoji_pattern.sub('', result)
        result = self.platform_emoji_pattern.sub('', result)
        result = re.sub(r'\s+', ' ', result)
        return result.strip()

    def evaluate(
        self,
        raw_texts: Optional[List[str]] = None,
        cleaned_texts: Optional[List[str]] = None,
        gold_texts: Optional[List[str]] = None,
    ) -> EvalResult:
        """
        执行 HTML 清洗评估

        Args:
            raw_texts: 原始含HTML文本列表
            cleaned_texts: 系统清洗后结果 (None则自动调用clean())
            gold_texts: 人工标注的标准清洗结果 (None则使用gold)

        Returns:
            EvalResult 包含 accuracy 和 error_cases
        """
        result = EvalResult("HTML清洗", "自建测试集")

        try:
            test_data = load_html_test_data()
        except FileNotFoundError:
            print("[HTML Eval] 测试数据不存在，自动生成模拟数据...")
            test_data = generate_html_test_data()

        raws = raw_texts or test_data.get("raw_texts", [])
        system_out = cleaned_texts or [self.clean(t) for t in tqdm(raws, desc="HTML清洗")]
        golds = gold_texts or test_data.get("gold_texts", [])

        correct = 0
        total = len(raws)

        for i, (sys_out, gold) in enumerate(zip(system_out, golds)):
            if sys_out == gold:
                correct += 1
            else:
                result.add_error_case({
                    "index": i,
                    "raw": raws[i][:100],
                    "system_output": sys_out[:100],
                    "gold_standard": gold[:100],
                })

        accuracy = calc_accuracy(correct, total)
        result.add_metric("accuracy", accuracy)
        result.add_detail("total_samples", total)
        result.add_detail("correct_samples", correct)
        result.add_detail("error_count", total - correct)

        print(f"[HTML Eval] 准确率: {accuracy:.4f} ({correct}/{total})")
        return result
