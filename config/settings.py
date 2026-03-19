# -*- coding: utf-8 -*-
"""
全局共享配置

统一管理 MySQL、Neo4j、DeepSeek、模型名称等配置，
各业务模块从此处引用，避免重复定义。
"""

import os
from pathlib import Path
from functools import lru_cache

from pydantic import BaseModel, Field

# ─── .env 加载 (整个项目只执行一次) ────────────────────────
try:
    from dotenv import load_dotenv

    def _find_env_file() -> Path:
        """查找 .env 文件，优先级: 项目根目录 > MediaCrawler 目录"""
        root_dir = Path(__file__).parent.parent
        root_env = root_dir / ".env"
        if root_env.exists():
            return root_env
        legacy_env = root_dir / "SentimentSpider" / "MediaCrawler" / ".env"
        if legacy_env.exists():
            return legacy_env
        return root_env

    _env_file = _find_env_file()
    load_dotenv(str(_env_file))
except ImportError:
    pass


# ─── MySQL 配置 ──────────────────────────────────────────
class MySQLConfig(BaseModel):
    """MySQL 数据库配置"""
    host: str = Field(default="localhost", description="数据库主机")
    port: int = Field(default=3306, description="数据库端口")
    user: str = Field(default="root", description="数据库用户名")
    password: str = Field(default="", description="数据库密码")
    database: str = Field(default="sentiment", description="数据库名称")
    charset: str = Field(default="utf8mb4", description="字符集")

    @property
    def connection_kwargs(self) -> dict:
        """返回 pymysql 连接参数"""
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "charset": self.charset,
        }

    @classmethod
    def from_env(cls) -> "MySQLConfig":
        """从环境变量加载配置"""
        return cls(
            host=os.getenv("MYSQL_DB_HOST", "localhost"),
            port=int(os.getenv("MYSQL_DB_PORT", "3306")),
            user=os.getenv("MYSQL_DB_USER", "root"),
            password=os.getenv("MYSQL_DB_PWD", ""),
            database=os.getenv("MYSQL_DB_NAME", "sentiment"),
        )


# ─── Neo4j 配置 ──────────────────────────────────────────
class Neo4jConfig(BaseModel):
    """Neo4j 图数据库配置"""
    uri: str = Field(default="bolt://localhost:7687", description="Neo4j 连接 URI")
    user: str = Field(default="neo4j", description="Neo4j 用户名")
    password: str = Field(default="", description="Neo4j 密码")
    database: str = Field(default="neo4j", description="Neo4j 数据库名")

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """从环境变量加载配置"""
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )


# ─── DeepSeek API 配置 ───────────────────────────────────
class DeepSeekConfig(BaseModel):
    """DeepSeek API 配置"""
    model_config = {"protected_namespaces": ()}

    api_key: str = Field(default="", description="DeepSeek API Key")
    api_base: str = Field(
        default="https://api.deepseek.com", description="DeepSeek API 基础 URL"
    )
    model_name: str = Field(default="deepseek-chat", description="模型名称")
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="生成温度"
    )
    max_tokens: int = Field(default=4096, ge=1, description="最大生成 token 数")
    timeout: float = Field(default=60.0, description="API 超时时间(秒)")

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        """从环境变量加载配置"""
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        )


# ─── 共享模型名称 ────────────────────────────────────────
class ModelNameConfig(BaseModel):
    """共享预训练模型名称"""
    model_config = {"protected_namespaces": ()}

    bert_model: str = Field(
        default="hfl/chinese-roberta-wwm-ext", description="BERT 模型名称"
    )
    qwen_model: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct", description="Qwen 模型名称"
    )

    @classmethod
    def from_env(cls) -> "ModelNameConfig":
        """从环境变量加载配置"""
        return cls(
            bert_model=os.getenv("BERT_MODEL_NAME", "hfl/chinese-roberta-wwm-ext"),
            qwen_model=os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
        )


# ─── 工厂函数 (单例) ─────────────────────────────────────

@lru_cache()
def get_mysql_config() -> MySQLConfig:
    """获取 MySQL 配置 (单例)"""
    return MySQLConfig.from_env()


@lru_cache()
def get_neo4j_config() -> Neo4jConfig:
    """获取 Neo4j 配置 (单例)"""
    return Neo4jConfig.from_env()


@lru_cache()
def get_deepseek_config() -> DeepSeekConfig:
    """获取 DeepSeek 配置 (单例)"""
    return DeepSeekConfig.from_env()


@lru_cache()
def get_model_names() -> ModelNameConfig:
    """获取共享模型名称 (单例)"""
    return ModelNameConfig.from_env()
