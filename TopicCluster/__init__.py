# -*- coding: utf-8 -*-
"""
TopicCluster - 话题聚类与事件检测模块

基于 HISEvent (AAAI 2024) + RagSEDE (2026) 思路:
BERT 嵌入 + Single-Pass 阈值匹配 + Qwen 生成话题描述

使用方法:
    # 命令行
    python -m TopicCluster cluster --batch-size 64
    python -m TopicCluster stats
    python -m TopicCluster merge
    python -m TopicCluster describe
    python -m TopicCluster evolve

    # Python API
    from TopicCluster import ClusterEngine
    engine = ClusterEngine()
    result = engine.run()
"""

__version__ = "0.1.0"
__author__ = "Zayy2005x"

# 配置
from .config import (
    ClusteringConfig,
    LLMConfig,
    DatabaseConfig,
    Settings,
    get_settings,
)

# 数据库
from .database import (
    TopicEventRepo,
    TopicEvolutionRepo,
    TopicMergeRepo,
    TopicContentRepo,
)

# 聚类 (延迟导入重量级依赖)
try:
    from .cluster import (
        BertEmbedder,
        FaissIndex,
        ClusterEngine,
        ClusterResult,
        TopicMaintainer,
        MergeResult,
    )
    _CLUSTER_AVAILABLE = True
except ImportError:
    _CLUSTER_AVAILABLE = False

# LLM (可选)
try:
    from .llm import TopicNamer
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False


__all__ = [
    # 版本
    "__version__",
    "__author__",

    # 配置
    "ClusteringConfig",
    "LLMConfig",
    "DatabaseConfig",
    "Settings",
    "get_settings",

    # 数据库
    "TopicEventRepo",
    "TopicEvolutionRepo",
    "TopicMergeRepo",
    "TopicContentRepo",

    # 聚类
    "BertEmbedder",
    "FaissIndex",
    "ClusterEngine",
    "ClusterResult",
    "TopicMaintainer",
    "MergeResult",

    # LLM
    "TopicNamer",

    # 可用性标志
    "_CLUSTER_AVAILABLE",
    "_LLM_AVAILABLE",
]
