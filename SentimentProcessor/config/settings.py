# =====================================================
# SentimentProcessor - Settings
# 配置加载器
# =====================================================

from typing import Optional, List

from pydantic import BaseModel

from config.settings import get_mysql_config


class ProcessorConfig(BaseModel):
    """预处理器配置"""
    # 批处理大小
    batch_size: int = 100

    # 分词配置
    use_paddle: bool = False  # 是否使用paddle模式(更准确但更慢)
    cut_all: bool = False  # 是否全模式分词

    # 停用词配置
    use_stopwords: bool = True
    custom_stopwords: List[str] = []

    # 清洗配置
    remove_urls: bool = True
    remove_emails: bool = True
    remove_mentions: bool = True  # @用户
    remove_hashtags: bool = False  # #话题# (保留话题内容)
    remove_emojis: bool = True
    remove_html: bool = True
    normalize_whitespace: bool = True
    to_simplified: bool = True  # 繁体转简体

    # 网络用语规范化
    normalize_slang: bool = True


class Settings:
    """全局配置"""

    _instance: Optional["Settings"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_config()
        return cls._instance

    def _init_config(self):
        """初始化配置"""
        self.db = get_mysql_config()
        self.processor = ProcessorConfig()

    def reload(self):
        """重新加载配置"""
        self._init_config()


def get_settings() -> Settings:
    """获取配置实例"""
    return Settings()
