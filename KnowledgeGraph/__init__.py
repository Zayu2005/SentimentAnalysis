# -*- coding: utf-8 -*-
"""
KnowledgeGraph - 知识图谱构建模块

基于 OneKE 思路: DeepSeek API 实体关系抽取 + Neo4j 图存储

使用方法:
    # 命令行
    python -m KnowledgeGraph extract --topic-id 42
    python -m KnowledgeGraph build --topic-id 42
    python -m KnowledgeGraph pipeline --topic-id 42
    python -m KnowledgeGraph query --topic-id 42
    python -m KnowledgeGraph stats

    # Python API
    from KnowledgeGraph import EntityRelationExtractor, GraphBuilder
    extractor = EntityRelationExtractor()
    result = extractor.extract_for_topic(topic_id=42)
"""

__version__ = "0.1.0"
__author__ = "Zayy2005x"

# 配置
from .config import (
    Neo4jConfig,
    DeepSeekConfig,
    ExtractionConfig,
    KGSettings,
    get_kg_settings,
)

# 数据库
from .database import (
    KGExtractionRepo,
    KGBuildLogRepo,
    KGContentRepo,
)

# 抽取 (延迟导入)
try:
    from .extraction import (
        EntityRelationExtractor,
        KG_ENTITY_TYPES,
        KG_RELATION_TYPES,
        EXTRACTION_SCHEMA,
    )
    _EXTRACTION_AVAILABLE = True
except ImportError:
    _EXTRACTION_AVAILABLE = False

# 图构建 (延迟导入)
try:
    from .graph import GraphBuilder
    _GRAPH_AVAILABLE = True
except ImportError:
    _GRAPH_AVAILABLE = False


__all__ = [
    # 版本
    "__version__",
    "__author__",

    # 配置
    "Neo4jConfig",
    "DeepSeekConfig",
    "ExtractionConfig",
    "KGSettings",
    "get_kg_settings",

    # 数据库
    "KGExtractionRepo",
    "KGBuildLogRepo",
    "KGContentRepo",

    # 抽取
    "EntityRelationExtractor",
    "KG_ENTITY_TYPES",
    "KG_RELATION_TYPES",
    "EXTRACTION_SCHEMA",

    # 图构建
    "GraphBuilder",

    # 可用性标志
    "_EXTRACTION_AVAILABLE",
    "_GRAPH_AVAILABLE",
]
