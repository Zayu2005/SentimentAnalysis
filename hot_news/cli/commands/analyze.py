# =====================================================
# Hot News Module - Analyze Command
# =====================================================

import typer
from typing import List, Optional
import asyncio
from hot_news.config import get_settings
from hot_news.database import HotNewsRepository, AnalysisRepository
from hot_news.analyzer import DomainChecker
from hot_news.models.entities import HotNewsItem

cmd_analyze = typer.Typer(help="分析热点领域匹配")


@cmd_analyze.command()
def analyze_domain_match(
    domains: Optional[List[str]] = typer.Argument(None, help="领域列表，如 科技 金融"),
    limit: int = typer.Option(100, "--limit", "-l", help="分析的热点数量"),
    min_confidence: float = typer.Option(
        0.5, "--min-confidence", "-c", help="最小置信度"
    ),
):
    """
    使用LLM分析热点是否符合指定领域

    示例:
        hot-news analyze 科技 金融
        hot-news analyze --limit 50 --min-confidence 0.7
    """
    settings = get_settings()
    domain_configs = settings.get_domains()

    if domains:
        selected_domains = [d for d in domain_configs if d.domain_name in domains]
        if not selected_domains:
            print("[Error] 未找到匹配的领域配置")
            raise typer.Exit(1)
    else:
        if not domain_configs:
            print("[Warn] 未配置领域，将跳过分析")
            return
        selected_domains = domain_configs

    repo = HotNewsRepository()
    analysis_repo = AnalysisRepository()
    checker = DomainChecker()

    async def _analyze():
        matched_count = 0

        news_list = repo.get_recent(limit)
        print(f"[Analyze] 待分析热点: {len(news_list)} 条")
        print(f"[Analyze] 领域: {[d.domain_name for d in selected_domains]}")

        for idx, news_row in enumerate(news_list):
            news = HotNewsItem(
                news_id=news_row["news_id"],
                platform_code=news_row["platform_code"],
                title=news_row["title"],
                url=news_row.get("url", ""),
                description=news_row.get("description", ""),
                score=news_row.get("score", ""),
            )

            for domain in selected_domains:
                try:
                    print(
                        f"  [{idx + 1}/{len(news_list)}] 分析: {news.title[:30]}... -> {domain.domain_name}"
                    )

                    result = await checker.check_domain(news, domain)

                    analysis_repo.save_analysis(
                        news_id=result.news_id,
                        domain_id=result.domain_id,
                        is_match=result.is_match,
                        llm_provider="",
                        analysis_content=result.reason,
                        confidence=result.confidence,
                    )

                    if result.is_match and result.confidence >= min_confidence:
                        matched_count += 1
                        print(f"    -> 匹配! (置信度: {result.confidence:.2f})")

                except Exception as e:
                    print(f"    -> 错误: {e}")

        print(f"\n[Summary] 分析完成，匹配 {matched_count} 条")

    asyncio.run(_analyze())
