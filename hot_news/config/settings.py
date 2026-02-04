# =====================================================
# Hot News Module - Configuration Loader
# =====================================================

import os
from typing import Optional
from pathlib import Path

import pymysql
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


# 尝试从多个路径加载 .env 文件
env_paths = [
    "../MediaCrawler/.env",  # 从 SentimentSpider 根目录
    "../../MediaCrawler/.env",  # 从 hot_news 目录
    "./.env",  # 当前目录
    "../.env",  # 上级目录
]

loaded = False
for env_path in env_paths:
    if load_dotenv(env_path):
        print(f"[Config] 加载 .env: {env_path}")
        loaded = True
        break

if not loaded:
    load_dotenv()  # 使用默认路径


class DBConfig(BaseModel):
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = "1234"
    database: str = "sentiment"

    @property
    def connection_kwargs(self):
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "charset": "utf8mb4",
        }


class HotPlatformConfig(BaseModel):
    id: int
    platform_code: str
    platform_name: str
    is_enabled: int = 1
    priority: int = 0


class DomainConfig(BaseModel):
    id: int
    domain_name: str
    domain_keywords: str = ""
    is_enabled: int = 1
    description: str = ""


class LLMConfig(BaseModel):
    id: int
    provider: str
    api_base: str = ""
    api_key: str = ""
    model_name: str = ""
    temperature: float = 0.7
    max_tokens: int = 2000
    is_enabled: int = 1
    is_default: int = 0

    model_config = {"protected_namespaces": ()}


class CrawlerPlatformConfig(BaseModel):
    id: int
    platform_code: str
    platform_name: str
    is_enabled: int = 1
    max_notes_count: int = 50
    max_comments_count: int = 20
    priority: int = 0


class TaskScheduleConfig(BaseModel):
    id: int
    task_name: str
    cron_expression: str = ""
    is_enabled: int = 1
    max_hot_count: int = 100
    need_llm_check: int = 1
    need_keyword_extract: int = 1
    execute_immediately: int = 0
    last_execute_time: Optional[datetime] = None


class HotNewsSettings:
    """热点新闻模块配置加载器"""

    _instance = None
    _db_config: Optional[DBConfig] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._db_config is None:
            self._load_db_config()

    def _load_db_config(self):
        """从环境变量加载数据库配置"""
        self._db_config = DBConfig(
            host=os.getenv("MYSQL_DB_HOST", "localhost"),
            port=int(os.getenv("MYSQL_DB_PORT", 3306)),
            user=os.getenv("MYSQL_DB_USER", "root"),
            password=os.getenv("MYSQL_DB_PWD", ""),
            database=os.getenv("MYSQL_DB_NAME", "sentiment"),
        )

    def _get_connection(self):
        """获取数据库连接"""
        return pymysql.connect(**self._db_config.connection_kwargs)

    def get_hot_platforms(self, enabled_only: bool = True) -> List[HotPlatformConfig]:
        """获取热点平台配置"""
        sql = "SELECT * FROM hot_platform_config"
        if enabled_only:
            sql += " WHERE is_enabled = 1"
        sql += " ORDER BY priority DESC"

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                return [
                    HotPlatformConfig(
                        id=row[0],
                        platform_code=row[1],
                        platform_name=row[2],
                        is_enabled=row[3],
                        priority=row[4],
                    )
                    for row in rows
                ]

    def get_domains(self, enabled_only: bool = True) -> List[DomainConfig]:
        """获取领域配置"""
        sql = "SELECT * FROM domain_config"
        if enabled_only:
            sql += " WHERE is_enabled = 1"

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                return [
                    DomainConfig(
                        id=row[0],
                        domain_name=row[1],
                        domain_keywords=row[2],
                        is_enabled=row[3],
                        description=row[4],
                    )
                    for row in rows
                ]

    def get_llm_config(self, provider: Optional[str] = None) -> List[LLMConfig]:
        """获取LLM配置"""
        sql = "SELECT * FROM llm_config WHERE is_enabled = 1"
        if provider:
            sql += f" AND provider = '{provider}'"

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                return [
                    LLMConfig(
                        id=row[0],
                        provider=row[1],
                        api_base=row[2],
                        api_key=row[3],
                        model_name=row[4],
                        temperature=row[5],
                        max_tokens=row[6],
                        is_enabled=row[7],
                        is_default=row[8],
                    )
                    for row in rows
                ]

    def get_default_llm(self) -> Optional[LLMConfig]:
        """获取默认LLM配置"""
        sql = "SELECT * FROM llm_config WHERE is_enabled = 1 AND is_default = 1 LIMIT 1"

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                row = cursor.fetchone()
                if row:
                    return LLMConfig(
                        id=row[0],
                        provider=row[1],
                        api_base=row[2],
                        api_key=row[3],
                        model_name=row[4],
                        temperature=row[5],
                        max_tokens=row[6],
                        is_enabled=row[7],
                        is_default=row[8],
                    )
                return None

    def get_crawler_platforms(
        self, enabled_only: bool = True
    ) -> List[CrawlerPlatformConfig]:
        """获取爬虫平台配置"""
        sql = "SELECT * FROM crawler_platform_config"
        if enabled_only:
            sql += " WHERE is_enabled = 1"
        sql += " ORDER BY priority DESC"

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                return [
                    CrawlerPlatformConfig(
                        id=row[0],
                        platform_code=row[1],
                        platform_name=row[2],
                        is_enabled=row[3],
                        max_notes_count=row[4],
                        max_comments_count=row[5],
                        priority=row[6],
                    )
                    for row in rows
                ]

    def get_task_schedule(
        self, task_name: Optional[str] = None
    ) -> List[TaskScheduleConfig]:
        """获取任务调度配置"""
        sql = "SELECT * FROM task_schedule_config WHERE is_enabled = 1"
        if task_name:
            sql += f" AND task_name = '{task_name}'"

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                return [
                    TaskScheduleConfig(
                        id=row[0],
                        task_name=row[1],
                        cron_expression=row[2],
                        is_enabled=row[3],
                        max_hot_count=row[4],
                        need_llm_check=row[5],
                        need_keyword_extract=row[6],
                        execute_immediately=row[7],
                        last_execute_time=row[8],
                    )
                    for row in rows
                ]


def get_settings() -> HotNewsSettings:
    """获取配置实例"""
    return HotNewsSettings()
