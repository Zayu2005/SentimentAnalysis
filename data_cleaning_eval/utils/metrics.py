# -*- coding: utf-8 -*-
"""
通用指标计算工具

对应论文章节: 5.4.1.4 数据清洗流水线评估
"""

from typing import Tuple, List, Dict, Any


def calc_precision_recall_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> Tuple[float, float, float]:
    """
    计算 Precision / Recall / F1

    Args:
        true_positives: TP
        false_positives: FP
        false_negatives: FN

    Returns:
        (precision, recall, f1)
    """
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def calc_accuracy(correct: int, total: int) -> float:
    """计算准确率"""
    return correct / total if total > 0 else 0.0


def calc_segmentation_f1(gold_tokens: List[str], pred_tokens: List[str]) -> Tuple[float, float, float]:
    """
    计算分词 F1 (基于词边界的标准方法)

    使用位置对齐方式：将 gold 和 pred 的词边界转换为位置标记，
    然后计算匹配度。

    Args:
        gold_tokens: 标准分词结果
        pred_tokens: 预测分词结果

    Returns:
        (precision, recall, f1)
    """
    gold_boundaries = _tokens_to_boundaries(gold_tokens)
    pred_boundaries = _tokens_to_boundaries(pred_tokens)

    tp = len(gold_boundaries & pred_boundaries)
    fp = len(pred_boundaries - gold_boundaries)
    fn = len(gold_boundaries - pred_boundaries)

    return calc_precision_recall_f1(tp, fp, fn)


def calc_segmentation_f1_batch(
    gold_sentences: List[List[str]],
    pred_sentences: List[List[str]],
) -> Dict[str, float]:
    """
    批量计算分词 F1，聚合所有句子的结果

    Args:
        gold_sentences: 标准分词结果列表
        pred_sentences: 预测分词结果列表

    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for gold_tokens, pred_tokens in zip(gold_sentences, pred_sentences):
        gold_bounds = _tokens_to_boundaries(gold_tokens)
        pred_bounds = _tokens_to_boundaries(pred_tokens)

        total_tp += len(gold_bounds & pred_bounds)
        total_fp += len(pred_bounds - gold_bounds)
        total_fn += len(gold_bounds - pred_bounds)

    p, r, f1 = calc_precision_recall_f1(total_tp, total_fp, total_fn)
    return {"precision": p, "recall": r, "f1": f1}


def _tokens_to_boundaries(tokens: List[str]) -> set:
    """
    将词序列转换为词边界位置集合

    每个词用 (起始位置, 结束位置) 表示其边界

    示例:
        tokens = ["迈向", "充满", "希望"]
        boundaries = {(0, 2), (2, 4), (4, 6)}
    """
    boundaries = set()
    pos = 0
    for token in tokens:
        end = pos + len(token)
        boundaries.add((pos, end))
        pos = end
    return boundaries


def format_result(value: float, decimals: int = 4) -> str:
    """格式化数值为百分比或小数"""
    if value <= 1.0:
        return f"{value:.{decimals}f}"
    return f"{value:.{decimals}%}"


class EvalResult:
    """评估结果容器"""

    def __init__(self, module_name: str, dataset_name: str):
        self.module_name = module_name
        self.dataset_name = dataset_name
        self.metrics: Dict[str, float] = {}
        self.details: Dict[str, Any] = {}
        self.error_cases: List[Dict] = []

    def add_metric(self, name: str, value: float):
        self.metrics[name] = value

    def add_detail(self, key: str, value: Any):
        self.details[key] = value

    def add_error_case(self, case: Dict):
        self.error_cases.append(case)

    def to_dict(self) -> Dict:
        return {
            "module": self.module_name,
            "dataset": self.dataset_name,
            "metrics": self.metrics,
            "details": self.details,
            "error_cases_count": len(self.error_cases),
        }
