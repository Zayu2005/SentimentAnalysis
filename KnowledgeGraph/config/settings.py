# -*- coding: utf-8 -*-
"""
知识图谱配置管理

使用 Pydantic 进行配置验证和管理
"""

import os
from pydantic import BaseModel, Field
from functools import lru_cache

from config.settings import (
    get_mysql_config,
    get_neo4j_config,
    get_deepseek_config,
    Neo4jConfig,
    DeepSeekConfig,
)


class ExtractionConfig(BaseModel):
    """抽取配置"""
    batch_size: int = Field(
        default=5, ge=1, description="每批发送API的内容条数"
    )
    max_content_length: int = Field(
        default=500, description="截取的最大内容长度"
    )
    rate_limit_delay: float = Field(
        default=1.0, ge=0.0, description="API调用间隔(秒)"
    )
    max_retries: int = Field(
        default=3, ge=1, description="API失败重试次数"
    )


class KGSettings(BaseModel):
    """KnowledgeGraph 全局设置"""
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)
    database: object = Field(default=None)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)

    base_dir: str = Field(default="", description="项目根目录")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.base_dir:
            self.base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )

    @classmethod
    def from_env(cls) -> "KGSettings":
        """从环境变量加载设置"""
        return cls(
            neo4j=get_neo4j_config(),
            deepseek=get_deepseek_config(),
            database=get_mysql_config(),
            extraction=ExtractionConfig(),
        )


@lru_cache()
def get_kg_settings() -> KGSettings:
    """获取全局设置 (单例模式)"""
    return KGSettings.from_env()
