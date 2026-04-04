# -*- coding: utf-8 -*-
"""
停用词过滤质量评估

对应论文章节: 5.4.1.4 数据清洗流水线评估 - 停用词过滤环节
"""

from typing import List, Dict, Set, Optional
from tqdm import tqdm

from ..utils.data_loader import (
    load_stopwords,
    load_sentiment_words,
    load_stopword_test_data,
    generate_stopword_test_data,
)
from ..utils.metrics import EvalResult, calc_accuracy


class StopwordEvaluator:
    """停用词过滤评估器"""

    def __init__(self):
        self._stopwords: Optional[Set[str]] = None
        self._sentiment_words: Optional[Set[str]] = None

    @property
    def stopwords(self) -> Set[str]:
        if self._stopwords is None:
            self._stopwords = load_stopwords()
        return self._stopwords

    @property
    def sentiment_words(self) -> Set[str]:
        if self._sentiment_words is None:
            self._sentiment_words = load_sentiment_words()
        return self._sentiment_words

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """使用停用词表过滤"""
        return [t for t in tokens if t not in self.stopwords]

    def evaluate(
        self,
        tokenized_texts: Optional[List[List[str]]] = None,
        gold_filtered: Optional[List[List[str]]] = None,
    ) -> EvalResult:
        """
        执行停用词过滤评估

        Args:
            tokenized_texts: 已分词文本列表
            gold_filtered: 人工标注的标准过滤结果

        Returns:
            EvalResult 包含 accuracy, false_positive_rate, sentiment_word_loss_rate
        """
        result = EvalResult("停用词过滤", "自建测试集")

        try:
            test_data = load_stopword_test_data()
        except FileNotFoundError:
            print("[Stopword Eval] 测试数据不存在，自动生成模拟数据...")
            test_data = generate_stopword_test_data()

        inputs = tokenized_texts or test_data.get("tokenized_texts", [])
        golds = gold_filtered or test_data.get("gold_filtered", [])

        total_correct_filter = 0
        total_should_filter = 0
        total_actual_filter = 0
        total_false_positive = 0
        total_sentiment_in_gold = 0
        total_sentiment_lost = 0

        for i, (tokens, gold) in enumerate(tqdm(
            list(zip(inputs, golds)), desc="停用词过滤评估"
        )):
            pred = self.filter_tokens(tokens)

            gold_set = set(gold)
            pred_set = set(pred)
            token_set = set(tokens)

            should_remove = token_set - gold_set
            actually_removed = token_set - pred_set

            correct_removals = should_remove & actually_removed
            false_positives = actually_removed - should_remove

            total_correct_filter += len(correct_removals)
            total_should_filter += len(should_remove)
            total_actual_filter += len(actually_removed)
            total_false_positive += len(false_positives)

            sentiment_in_gold = gold_set & self.sentiment_words
            sentiment_lost = sentiment_in_gold - pred_set
            total_sentiment_in_gold += len(sentiment_in_gold)
            total_sentiment_lost += len(sentiment_lost)

            if pred != gold:
                lost_important = sentiment_lost
                result.add_error_case({
                    "index": i,
                    "input": tokens,
                    "predicted": pred,
                    "gold": gold,
                    "false_positives": list(false_positives)[:5],
                    "sentiment_lost": list(lost_important)[:5],
                })

        n = len(inputs)

        filter_precision = total_correct_filter / total_actual_filter if total_actual_filter > 0 else 0.0
        filter_recall = total_correct_filter / total_should_filter if total_should_filter > 0 else 0.0
        accuracy = calc_accuracy(sum(1 for p, g in zip(
            [self.filter_tokens(t) for t in inputs], golds
        ) if p == g), n)

        false_positive_rate = total_false_positive / total_actual_filter if total_actual_filter > 0 else 0.0
        sentiment_loss_rate = total_sentiment_lost / total_sentiment_in_gold if total_sentiment_in_gold > 0 else 0.0

        result.add_metric("accuracy", accuracy)
        result.add_metric("filter_precision", filter_precision)
        result.add_metric("filter_recall", filter_recall)
        result.add_metric("false_positive_rate", false_positive_rate)
        result.add_metric("sentiment_word_loss_rate", sentiment_loss_rate)

        result.add_detail("total_samples", n)
        result.add_detail("correct_samples", int(accuracy * n))
        result.add_detail("error_count", n - int(accuracy * n))
        result.add_detail("stopword_list_size", len(self.stopwords))
        result.add_detail("sentiment_word_total", total_sentiment_in_gold)
        result.add_detail("sentiment_word_lost", total_sentiment_lost)

        print(f"[Stopword Eval] 准确率: {accuracy:.4f}")
        print(f"[Stopword Eval] 过滤精度(P): {filter_precision:.4f}, 召回(R): {filter_recall:.4f}")
        print(f"[Stopword Eval] 误删率: {false_positive_rate:.4f}, 情感词误删率: {sentiment_loss_rate:.4f}")

        return result
