# =====================================================
# Hot News Module - Orz.ai API Client
# =====================================================

import httpx
import json
from typing import List, Optional
from datetime import datetime
import hashlib

from .base import BaseFetcher
from ..models.entities import HotNewsItem


class OrzAiClient:
    """orz.ai API 客户端"""

    BASE_URL = "https://orz.ai/api/v1"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()

    async def get_hot_news(self, platform: str, limit: int = 100) -> List[dict]:
        """
        获取热点新闻

        Args:
            platform: 平台代码 (baidu/weibo/zhihu/bilibili/douyin等)
            limit: 返回数量限制

        Returns:
            热点新闻列表
        """
        url = f"{self.BASE_URL}/dailynews/"
        params = {"platform": platform}

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        try:
            response = await self.client.get(
                url, params=params, headers=headers, follow_redirects=True
            )
            if response.status_code >= 400:
                print(f"[OrzAiClient] HTTP错误: 状态码 {response.status_code}")
                return []

            data = response.json()

            if data.get("status") == "200":
                news_list = data.get("data", [])[:limit]
                return news_list
            else:
                print(f"[OrzAiClient] API返回错误: {data.get('msg')}")
                return []
        except httpx.HTTPError as e:
            print(f"[OrzAiClient] HTTP错误: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"[OrzAiClient] JSON解析错误: {e}")
            return []

    async def get_website_meta(self, url: str) -> Optional[dict]:
        """获取网站元信息"""
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/tools/website-meta/", params={"url": url}
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "200":
                return data.get("data", {})
            return None
        except httpx.HTTPError as e:
            print(f"[OrzAiClient] 获取元信息错误: {e}")
            return None


class OrzAiFetcher(BaseFetcher):
    """orz.ai 热点获取器"""

    SUPPORTED_PLATFORMS = [
        "baidu",
        "weibo",
        "zhihu",
        "bilibili",
        "douyin",
        "juejin",
        "github",
        "hackernews",
        "sina_finance",
        "xueqiu",
        "douban",
        "hupu",
        "tieba",
        "cls",
        "tenxunwang",
        "v2ex",
        "jinritoutiao",
        "stackoverflow",
        "tskr",
        "ftpojie",
        "sspai",
        "eastmoney",
    ]

    def __init__(self, platform_code: str):
        super().__init__(platform_code)
        self.client = OrzAiClient()

    async def fetch(self, limit: int = 100) -> List[HotNewsItem]:
        """获取热点新闻"""
        if self.platform_code not in self.SUPPORTED_PLATFORMS:
            print(f"[OrzAiFetcher] 不支持的平台: {self.platform_code}")
            return []

        news_list = await self.client.get_hot_news(self.platform_code, limit)

        items = []
        for news in news_list:
            news_id = self._generate_news_id(self.platform_code, news)
            item = HotNewsItem(
                news_id=news_id,
                platform_code=self.platform_code,
                title=news.get("title", ""),
                url=news.get("url", ""),
                score=news.get("hot", ""),
                description=news.get("content", "") or news.get("description", ""),
                created_at=datetime.now(),
            )
            items.append(item)

        return items

    def _generate_news_id(self, platform: str, news: dict) -> str:
        """生成唯一的新闻ID"""
        title = news.get("title", "")
        url = news.get("url", "")
        unique_str = f"{platform}:{title}:{url}"
        return hashlib.md5(unique_str.encode()).hexdigest()

    async def close(self):
        """关闭客户端"""
        await self.client.close()


class HotNewsFactory:
    """热点获取器工厂"""

    _fetchers = {}

    @classmethod
    def get_fetcher(cls, platform_code: str) -> OrzAiFetcher:
        """获取指定平台的获取器"""
        if platform_code not in cls._fetchers:
            cls._fetchers[platform_code] = OrzAiFetcher(platform_code)
        return cls._fetchers[platform_code]

    @classmethod
    async def close_all(cls):
        """关闭所有获取器"""
        for fetcher in cls._fetchers.values():
            await fetcher.close()
        cls._fetchers.clear()
