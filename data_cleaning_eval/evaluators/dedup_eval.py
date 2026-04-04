# -*- coding: utf-8 -*-
"""
近似去重质量评估 - SimHash vs TF-IDF 余弦相似度

对应论文章节: 5.4.1.4 数据清洗流水线评估 - 去重环节
"""

from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

try:
    from simhash import Simhash
    _SIMHASH_AVAILABLE = True
except ImportError:
    _SIMHASH_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..config import config
from ..utils.data_loader import load_lcqmc
from ..utils.metrics import (
    EvalResult,
    calc_precision_recall_f1,
    calc_accuracy,
)


def _get_features(text: str) -> List[str]:
    """SimHash 分词特征提取"""
    import jieba
    return list(jieba.cut(text))


class DedupEvaluator:
    """近似去重评估器"""

    def __init__(
        self,
        simhash_threshold: int = None,
        cosine_threshold: float = None,
    ):
        self.simhash_threshold = simhash_threshold or config.simhash_threshold
        self.cosine_threshold = cosine_threshold or config.cosine_threshold
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None

    def _compute_simhash_distance(self, s1: str, s2: str) -> int:
        """计算两个文本的 SimHash 汉明距离"""
        if not _SIMHASH_AVAILABLE:
            raise ImportError("请安装 simhash 库: pip install simhash")
        h1 = Simhash(_get_features(s1))
        h2 = Simhash(_get_features(s2))
        return h1.distance(h2)

    def _predict_simhash(self, s1: str, s2: str) -> int:
        """SimHash 方式判断是否重复"""
        dist = self._compute_simhash_distance(s1, s2)
        return 1 if dist <= self.simhash_threshold else 0

    def _predict_cosine(self, s1: str, s2: str) -> int:
        """TF-IDF 余弦相似度方式判断是否重复"""
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(tokenizer=_get_features, lowercase=False)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([s1, s2])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return 1 if sim >= self.cosine_threshold else 0

    def _evaluate_method(
        self,
        pairs: List[Tuple[str, str, int]],
        method_name: str,
        predict_fn,
    ) -> Dict[str, float]:
        """
        评估单个去重方法

        Args:
            pairs: [(句子1, 句子2, gold_label), ...]
            method_name: 方法名称
            predict_fn: 预测函数 (s1, s2) -> 0/1

        Returns:
            {"precision": ..., "recall": ..., "f1": ..., "accuracy": ...}
        """
        tp = fp = fn = tn = 0

        preds = []
        labels = []
        scores = []

        for s1, s2, gold_label in tqdm(pairs, desc=f"去重评估-{method_name}"):
            pred = predict_fn(s1, s2)
            preds.append(pred)
            labels.append(gold_label)

            if method_name == "simhash":
                dist = self._compute_simhash_distance(s1, s2)
                score = max(0, 1 - dist / 64.0)
            else:
                if self.tfidf_vectorizer is None:
                    self.tfidf_vectorizer = TfidfVectorizer(
                        tokenizer=_get_features, lowercase=False
                    )
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([s1, s2])
                score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
            scores.append(score)

            if pred == 1 and gold_label == 1:
                tp += 1
            elif pred == 1 and gold_label == 0:
                fp += 1
            elif pred == 0 and gold_label == 1:
                fn += 1
            else:
                tn += 1

        precision, recall, f1 = calc_precision_recall_f1(tp, fp, fn)
        accuracy = calc_accuracy(tp + tn, tp + fp + fn + tn)

        self._plot_pr_curve(labels, scores, method_name)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    def _plot_pr_curve(self, labels: List[int], scores: List[float], method_name: str):
        """绘制 PR 曲线"""
        from sklearn.metrics import precision_recall_curve, average_precision_score

        output_dir = config.output_path
        output_dir.mkdir(parents=True, exist_ok=True)

        precision_vals, recall_vals, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)

        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, 'b-', linewidth=2,
                 label=f'{method_name} (AP={ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve - {method_name}')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plot_path = output_dir / f"pr_curve_{method_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Dedup Eval] PR曲线已保存: {plot_path}")

    def evaluate(self) -> Dict[str, EvalResult]:
        """
        执行完整去重评估

        对比 SimHash 和 TF-IDF余弦 两种方式

        Returns:
            {"simhash": EvalResult, "cosine": EvalResult}
        """
        results = {}

        try:
            positives, negatives = load_lcqmc(
                positive_n=config.lcqmc_positive_samples,
                negative_n=config.lcqmc_negative_samples,
            )
        except FileNotFoundError as e:
            print(f"[Dedup Eval] LCQMC 数据文件不存在: {e}")
            print("[Dedup Eval] 请从 https://huggingface.co/datasets/shibing624/lcqmc 下载")
            return results

        pairs = positives + negatives
        print(f"[Dedup Eval] 加载测试对: 正样本 {len(positives)}, 负样本 {len(negatives)}")

        # SimHash 评估
        if _SIMHASH_AVAILABLE:
            self.tfidf_vectorizer = None
            simhash_metrics = self._evaluate_method(
                pairs, "SimHash", self._predict_simhash
            )

            sim_result = EvalResult("近似去重(SimHash)", f"LCQMC {len(pairs)}对")
            for k, v in simhash_metrics.items():
                sim_result.add_metric(k, v)
            sim_result.add_detail("threshold", self.simhash_threshold)
            sim_result.add_detail("positive_samples", len(positives))
            sim_result.add_detail("negative_samples", len(negatives))

            results["simhash"] = sim_result
            print(
                f"[Dedup Eval] SimHash: P={simhash_metrics['precision']:.4f} "
                f"R={simhash_metrics['recall']:.4f} F1={simhash_metrics['f1']:.4f}"
            )
        else:
            print("[Dedup Eval] simhash 库未安装，跳过 SimHash 评估")

        # Cosine 评估
        self.tfidf_vectorizer = None
        cosine_metrics = self._evaluate_method(
            pairs, "Cosine", self._predict_cosine
        )

        cos_result = EvalResult("近似去重(Cosine)", f"LCQMC {len(pairs)}对")
        for k, v in cosine_metrics.items():
            cos_result.add_metric(k, v)
        cos_result.add_detail("threshold", self.cosine_threshold)
        cos_result.add_detail("positive_samples", len(positives))
        cos_result.add_detail("negative_samples", len(negatives))

        results["cosine"] = cos_result
        print(
            f"[Dedup Eval] Cosine: P={cosine_metrics['precision']:.4f} "
            f"R={cosine_metrics['recall']:.4f} F1={cosine_metrics['f1']:.4f}"
        )

        # 对比差异
        if "simhash" in results:
            diff_f1 = abs(simhash_metrics["f1"] - cosine_metrics["f1"])
            print(f"[Dedup Eval] 两种方法 F1 差异: {diff_f1:.4f}")

        return results
