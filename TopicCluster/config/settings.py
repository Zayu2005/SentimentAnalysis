# -*- coding: utf-8 -*-
"""
话题聚类配置管理

使用 Pydantic 进行配置验证和管理
"""

import os
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field
from functools import lru_cache

# 加载 .env 文件 (优先从项目根目录加载)
try:
    from dotenv import load_dotenv

    def _find_env_file() -> Path:
        """查找 .env 文件，优先级: 根目录 > MediaCrawler"""
        root_dir = Path(__file__).parent.parent.parent
        root_env = root_dir / ".env"
        if root_env.exists():
            return root_env
        legacy_env = root_dir / "SentimentSpider" / "MediaCrawler" / ".env"
        if legacy_env.exists():
            return legacy_env
        return root_env

    env_file = _find_env_file()
    load_dotenv(str(env_file))
except ImportError:
    pass


class ClusteringConfig(BaseModel):
    """聚类配置"""
    similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="话题归属相似度阈值"
    )
    merge_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="话题合并相似度阈值"
    )
    min_topic_size: int = Field(
        default=2,
        ge=1,
        description="话题最小内容数"
    )
    embedding_model: str = Field(
        default="hfl/chinese-roberta-wwm-ext",
        description="嵌入模型名称"
    )
    embedding_dim: int = Field(
        default=768,
        description="嵌入向量维度"
    )
    max_length: int = Field(
        default=128,
        description="输入序列最大长度"
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        description="批处理大小"
    )
    device: str = Field(
        default="auto",
        description="计算设备 (auto/cpu/cuda/cuda:0)"
    )
    inactive_days: int = Field(
        default=7,
        ge=1,
        description="话题不活跃天数阈值"
    )
    ended_days: int = Field(
        default=30,
        ge=1,
        description="话题结束天数阈值"
    )


class LLMConfig(BaseModel):
    """LLM 话题命名配置"""
    model_config = {"protected_namespaces": ()}

    model_name: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="LLM 模型名称"
    )
    generate_names: bool = Field(
        default=True,
        description="是否启用 LLM 话题命名"
    )
    max_new_tokens: int = Field(
        default=200,
        ge=1,
        description="最大生成 token 数"
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="生成温度"
    )


class DatabaseConfig(BaseModel):
    """数据库配置"""
    host: str = Field(default="localhost", description="数据库主机")
    port: int = Field(default=3306, description="数据库端口")
    user: str = Field(default="root", description="数据库用户名")
    password: str = Field(default="", description="数据库密码")
    database: str = Field(default="sentiment", description="数据库名称")
    charset: str = Field(default="utf8mb4", description="字符集")

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """从环境变量加载配置"""
        return cls(
            host=os.getenv("MYSQL_DB_HOST", "localhost"),
            port=int(os.getenv("MYSQL_DB_PORT", "3306")),
            user=os.getenv("MYSQL_DB_USER", "root"),
            password=os.getenv("MYSQL_DB_PWD", ""),
            database=os.getenv("MYSQL_DB_NAME", "sentiment")
        )


class Settings(BaseModel):
    """全局设置"""
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    base_dir: str = Field(default="", description="项目根目录")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.base_dir:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量加载设置"""
        return cls(
            clustering=ClusteringConfig(
                embedding_model=os.getenv("BERT_MODEL_NAME", "hfl/chinese-roberta-wwm-ext"),
            ),
            llm=LLMConfig(
                model_name=os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
            ),
            database=DatabaseConfig.from_env()
        )


@lru_cache()
def get_settings() -> Settings:
    """获取全局设置 (单例模式)"""
    return Settings.from_env()
