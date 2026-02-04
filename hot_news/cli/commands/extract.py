# =====================================================
# Hot News Module - Extract Command
# =====================================================

import typer
from typing import List, Optional
import asyncio
from hot_news.config import get_settings
from hot_news.database import HotNewsRepository, AnalysisRepository, KeywordRepository
from hot_news.analyzer import KeywordExtractor
from hot_news.models.entities import HotNewsItem

cmd_extract = typer.Typer(help="提取关键词")


@cmd_extract.command()
def extract_keywords(
    domains: Optional[List[str]] = typer.Argument(None, help="领域列表，如 科技 金融"),
    limit: int = typer.Option(50, "--limit", "-l", help="提取的热点数量"),
):
    """
    使用LLM从匹配的热点中提取关键词

    示例:
        hot-news extract 科技 金融
        hot-news extract --limit 100
    """
    settings = get_settings()
    domain_configs = settings.get_domains()

    if domains:
        selected_domains = [d for d in domain_configs if d.domain_name in domains]
    else:
        selected_domains = domain_configs

    if not selected_domains:
        print("[Error] 未找到领域配置")
        raise typer.Exit(1)

    repo = HotNewsRepository()
    analysis_repo = AnalysisRepository()
    keyword_repo = KeywordRepository()
    extractor = KeywordExtractor()

    async def _extract():
        total_keywords = 0

        for domain in selected_domains:
            print(f"[Extract] 领域: {domain.domain_name}")

            matched_news = analysis_repo.get_matched_news(domain.id, limit)
            print(f"  待提取热点: {len(matched_news)} 条")

            for idx, news_row in enumerate(matched_news):
                news = HotNewsItem(
                    news_id=news_row["news_id"],
                    platform_code=news_row["platform_code"],
                    title=news_row["title"],
                    url=news_row.get("url", ""),
                    description=news_row.get("description", ""),
                )

                try:
                    print(
                        f"  [{idx + 1}/{len(matched_news)}] 提取: {news.title[:30]}..."
                    )

                    keywords = await extractor.extract_keywords(news, domain)

                    if keywords:
                        keyword_repo.bulk_save(keywords)
                        total_keywords += len(keywords)
                        kw_list = ", ".join([k.keyword for k in keywords[:5]])
                        if len(keywords) > 5:
                            kw_list += f"..."
                        print(f"    -> 提取 {len(keywords)} 个关键词: {kw_list}")

                except Exception as e:
                    print(f"    -> 错误: {e}")

        print(f"\n[Summary] 共提取 {total_keywords} 个关键词")

    asyncio.run(_extract())
