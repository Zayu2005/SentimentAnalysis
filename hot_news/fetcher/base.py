# =====================================================
# Hot News Module - Base Fetcher
# =====================================================

from abc import ABC, abstractmethod
from typing import List
from ..models.entities import HotNewsItem


class BaseFetcher(ABC):
    """热点获取器基类"""

    def __init__(self, platform_code: str):
        self.platform_code = platform_code

    @abstractmethod
    async def fetch(self, limit: int = 100) -> List[HotNewsItem]:
        """获取热点新闻"""
        pass
