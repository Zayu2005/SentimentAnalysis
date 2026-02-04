# =====================================================
# Hot News Module - Fetch Command
# =====================================================

import typer
from typing import List, Optional
import asyncio
from hot_news.fetcher import HotNewsFactory
from hot_news.config import get_settings
from hot_news.database import HotNewsRepository
from hot_news.models.entities import HotNewsItem

cmd_fetch = typer.Typer(help="获取热点新闻")


@cmd_fetch.command("fetch")
def fetch(
    platforms: Optional[List[str]] = typer.Argument(
        None, help="平台列表，如 weibo zhihu bilibili"
    ),
    limit: int = typer.Option(50, "--limit", "-l", help="每个平台获取的数量"),
):
    """
    从 orz.ai 获取热点新闻

    示例:
        hot-news fetch weibo zhihu bilibili
        hot-news fetch --limit 100
    """
    settings = get_settings()

    if platforms:
        enabled_platforms = platforms
    else:
        platform_configs = settings.get_hot_platforms()
        enabled_platforms = [p.platform_code for p in platform_configs]

    async def _fetch():
        repo = HotNewsRepository()
        total = 0

        for platform in enabled_platforms:
            print(f"[Fetch] 获取 {platform} 热点...")
            try:
                fetcher = HotNewsFactory.get_fetcher(platform)
                news_list = await fetcher.fetch(limit)

                if news_list:
                    repo.bulk_save(news_list)
                    total += len(news_list)
                    print(f"  -> 获取 {len(news_list)} 条热点")
                else:
                    print(f"  -> 无数据")
            except Exception as e:
                print(f"  -> 错误: {e}")

        await HotNewsFactory.close_all()
        print(f"\n[Summary] 共获取 {total} 条热点")

    asyncio.run(_fetch())
