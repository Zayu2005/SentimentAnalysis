# -*- coding: utf-8 -*-
"""
SentimentModel - 中文情感分析模块

基于预训练的中文 BERT 模型进行情感分析

使用方法:
    # 命令行
    python -m SentimentModel stats
    python -m SentimentModel train --dataset weibo_senti_100k
    python -m SentimentModel analyze --model models/best_model.pt

    # Python API
    from SentimentModel import SentimentPredictor

    predictor = SentimentPredictor(model_path="models/best_model.pt")
    result = predictor.predict("这个产品真的太好用了！")
    print(result.label, result.score)
"""

__version__ = "0.1.0"
__author__ = "Zayy2005x"

# 导出主要类和函数
from .config import (
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    Settings,
    get_settings
)

from .models import BertSentimentClassifier

from .inference import SentimentPredictor, SentimentResult

from .training import SentimentTrainer, compute_metrics, MetricResult

from .database import (
    SentimentContentRepo,
    SentimentCommentRepo
)

from .data import (
    SentimentDataset,
    create_dataloader,
    load_public_dataset
)

# Qwen 模块 (可选，需要额外依赖)
try:
    from .qwen import (
        QwenSentimentTrainer,
        QwenSentimentPredictor,
        prepare_sentiment_data,
        EMOTION_TAGS
    )
    _QWEN_AVAILABLE = True
except ImportError:
    _QWEN_AVAILABLE = False


__all__ = [
    # 版本信息
    "__version__",
    "__author__",

    # 配置
    "ModelConfig",
    "TrainingConfig",
    "InferenceConfig",
    "Settings",
    "get_settings",

    # 模型
    "BertSentimentClassifier",

    # 推理
    "SentimentPredictor",
    "SentimentResult",

    # 训练
    "SentimentTrainer",
    "compute_metrics",
    "MetricResult",

    # 数据库
    "SentimentContentRepo",
    "SentimentCommentRepo",

    # 数据
    "SentimentDataset",
    "create_dataloader",
    "load_public_dataset",

    # Qwen (可选)
    "QwenSentimentTrainer",
    "QwenSentimentPredictor",
    "prepare_sentiment_data",
    "EMOTION_TAGS",
    "_QWEN_AVAILABLE",
]
