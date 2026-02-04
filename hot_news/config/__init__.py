# =====================================================
# Hot News Module - Config Package
# =====================================================

from .settings import get_settings, HotNewsSettings
from .settings import (
    DBConfig,
    HotPlatformConfig,
    DomainConfig,
    LLMConfig,
    CrawlerPlatformConfig,
    TaskScheduleConfig,
)

__all__ = [
    "get_settings",
    "HotNewsSettings",
    "DBConfig",
    "HotPlatformConfig",
    "DomainConfig",
    "LLMConfig",
    "CrawlerPlatformConfig",
    "TaskScheduleConfig",
]
