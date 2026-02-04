# =====================================================
# Hot News Module - Crawl Command
# =====================================================

import typer
from typing import List, Optional
from hot_news.config import get_settings
from hot_news.database import KeywordRepository
from hot_news.crawler import CrawlTrigger

cmd_crawl = typer.Typer(help="触发爬虫")


@cmd_crawl.command()
def trigger_crawl(
    platforms: List[str] = typer.Argument(..., help="爬虫平台列表，如 xhs dy bili"),
    keywords: Optional[List[str]] = typer.Option(
        None, "--keyword", "-k", help="指定关键词"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="触发爬取的数量"),
):
    """
    使用关键词触发爬虫爬取数据

    示例:
        hot-news crawl xhs dy --keyword "AI 编程"
        hot-news crawl bili --limit 5
    """
    settings = get_settings()
    trigger = CrawlTrigger()

    if keywords:
        success_count = trigger.trigger_batch(keywords, platforms)
    else:
        success_count = trigger.trigger_from_keywords(
            keyword_ids=None, domain_id=None, platforms=platforms, limit=limit
        )

    print(f"[Summary] 成功触发 {success_count} 次爬取")
