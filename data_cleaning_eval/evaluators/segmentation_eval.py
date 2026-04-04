# -*- coding: utf-8 -*-
"""
分词质量评估 - 基于 SIGHAN PKU 和 NLPCC 微博数据集

对应论文章节: 5.4.1.4 数据清洗流水线评估 - 分词环节
"""

from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

import jieba

from ..utils.data_loader import (
    load_sighan_pku,
    load_nlpcc_weibo,
    load_custom_dict,
)
from ..utils.metrics import (
    EvalResult,
    calc_segmentation_f1_batch,
)


class SegmentationEvaluator:
    """分词质量评估器"""

    def __init__(self):
        self._custom_dict_loaded = False

    def _load_custom_dict(self):
        """加载自定义词典"""
        if self._custom_dict_loaded:
            return
        custom_words = load_custom_dict()
        if custom_words:
            for word in custom_words:
                jieba.add_word(word)
            print(f"[Seg Eval] 已加载自定义词典: {len(custom_words)} 个词")
        self._custom_dict_loaded = True

    def _reset_jieba(self):
        """重置 jieba 到默认状态"""
        import importlib
        import jieba as jieba_module
        jieba_module.initialize()
        self._custom_dict_loaded = False

    def segment_default(self, text: str) -> List[str]:
        """jieba 默认模式分词"""
        return list(jieba.cut(text))

    def segment_with_dict(self, text: str) -> List[str]:
        """jieba + 自定义词典模式分词"""
        self._load_custom_dict()
        return list(jieba.cut(text))

    def _evaluate_dataset(
        self,
        dataset_name: str,
        sentences: List[Tuple[str, List[str]]],
        mode_name: str,
        seg_fn,
    ) -> Dict[str, float]:
        """
        对单个数据集 + 模式组合进行评估

        Args:
            dataset_name: 数据集名称
            sentences: [(原始句子, gold_tokens), ...]
            mode_name: 模式名称
            seg_fn: 分词函数

        Returns:
            {"precision": ..., "recall": ..., "f1": ...}
        """
        gold_sentences = [tokens for _, tokens in sentences]

        pred_sentences = []
        for original, _ in tqdm(sentences, desc=f"分词-{dataset_name}-{mode_name}"):
            pred_tokens = seg_fn(original)
            pred_sentences.append(pred_tokens)

        return calc_segmentation_f1_batch(gold_sentences, pred_sentences)

    def evaluate(self) -> Dict[str, EvalResult]:
        """
        执行完整分词评估

        运行四种配置:
        1. SIGHAN PKU + 默认模式
        2. SIGHAN PKU + 自定义词典
        3. NLPCC微博 + 默认模式
        4. NLPCC微博 + 自定义词典

        Returns:
            {配置名: EvalResult}
        """
        results = {}

        datasets = {
            "SIGHAN PKU": lambda: load_sighan_pku(),
            "NLPCC微博": lambda: load_nlpcc_weibo(),
        }

        modes = [
            ("默认模式", self.segment_default),
            ("+自定义词典", self.segment_with_dict),
        ]

        for ds_name, loader in datasets.items():
            try:
                sentences = loader()
            except FileNotFoundError as e:
                print(f"[Seg Eval] 跳过 {ds_name}: {e}")
                continue

            for mode_name, seg_fn in modes:
                key = f"分词({ds_name}){mode_name}"

                if mode_name == "+自定义词典":
                    self._reset_jieba()

                metrics = self._evaluate_dataset(ds_name, sentences, mode_name, seg_fn)

                result = EvalResult(f"分词({ds_name})", mode_name)
                result.add_metric("precision", metrics["precision"])
                result.add_metric("recall", metrics["recall"])
                result.add_metric("f1", metrics["f1"])
                result.add_detail("sample_count", len(sentences))

                results[key] = result
                print(
                    f"[Seg Eval] {ds_name} / {mode_name}: "
                    f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
                )

        return results
