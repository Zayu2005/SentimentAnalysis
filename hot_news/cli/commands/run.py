# =====================================================
# Hot News Module - Run Command (一键执行)
# =====================================================

import typer
from typing import List, Optional
import asyncio
from datetime import datetime
import time
from hot_news.config import get_settings
from hot_news.fetcher import HotNewsFactory
from hot_news.analyzer import DomainChecker, KeywordExtractor
from hot_news.crawler import CrawlTrigger
from hot_news.database import (
    HotNewsRepository,
    AnalysisRepository,
    KeywordRepository,
    TaskLogRepository,
)
from hot_news.models.entities import HotNewsItem

cmd_run = typer.Typer(help="一键执行完整流程")


async def _analyze_domain(news_list, selected_domains):
    """异步分析领域匹配"""
    checker = DomainChecker()
    matched_count = 0

    for idx, news_row in enumerate(news_list):
        news = HotNewsItem(
            news_id=news_row["news_id"],
            platform_code=news_row["platform_code"],
            title=news_row["title"],
            url=news_row.get("url", ""),
            description=news_row.get("description", ""),
        )

        for domain in selected_domains:
            try:
                result = await checker.check_domain(news, domain)
                AnalysisRepository().save_analysis(
                    news_id=result.news_id,
                    domain_id=result.domain_id,
                    is_match=result.is_match,
                    llm_provider="",
                    analysis_content=result.reason,
                    confidence=result.confidence,
                )
                if result.is_match:
                    matched_count += 1
            except Exception:
                pass

    return matched_count


async def _extract_keywords(selected_domains, keyword_limit, all_news=None, run_batch_id: int = None):
    """异步提取关键词

    Args:
        selected_domains: 选定的领域列表，如果为空则使用all_news
        keyword_limit: 每个领域最多提取的关键词数
        all_news: 当selected_domains为空时，从这个新闻列表中提取关键词
        run_batch_id: 运行批次ID，用于追踪该批次提取的关键词
    """
    extractor = KeywordExtractor()
    keyword_repo = KeywordRepository()
    total_keywords = 0

    # 如果有domain配置，从匹配的热点中提取
    if selected_domains:
        for domain in selected_domains:
            matched_news = AnalysisRepository().get_matched_news(domain.id, keyword_limit)
            for news_row in matched_news:
                news = HotNewsItem(
                    news_id=news_row["news_id"],
                    platform_code=news_row["platform_code"],
                    title=news_row["title"],
                    url=news_row.get("url", ""),
                    description=news_row.get("description", ""),
                )

                try:
                    keywords = await extractor.extract_keywords(news, domain)
                    if keywords:
                        keyword_repo.bulk_save(keywords, run_batch_id=run_batch_id)
                        total_keywords += len(keywords)
                except Exception:
                    pass

    # 如果没有domain配置，从所有热点中提取（不筛选）
    elif all_news:
        # 创建一个虚拟的domain对象用于关键词提取
        from hot_news.models.entities import DomainConfig
        virtual_domain = DomainConfig(
            id=0,
            domain_name="通用",
            domain_keywords="热点,新闻",
            is_enabled=1,
        )

        for news_row in all_news[:keyword_limit]:
            news = HotNewsItem(
                news_id=news_row["news_id"],
                platform_code=news_row["platform_code"],
                title=news_row["title"],
                url=news_row.get("url", ""),
                description=news_row.get("description", ""),
            )

            try:
                keywords = await extractor.extract_keywords(news, virtual_domain)
                if keywords:
                    keyword_repo.bulk_save(keywords, run_batch_id=run_batch_id)
                    total_keywords += len(keywords)
            except Exception:
                pass

    return total_keywords


async def _run_pipeline_inner(
    platforms, domains, crawl_platforms, hot_limit, keyword_limit, no_llm, no_crawl
):
    """内部异步执行函数"""
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    settings = get_settings()
    log_repo = TaskLogRepository()

    log_id = log_repo.start_task("hot_pipeline")

    hot_count = 0
    matched_count = 0
    keyword_count = 0
    crawl_count = 0

    try:
        # =============== 显示执行配置 ===============
        print("\n" + "=" * 70)
        print("  🚀 热点新闻获取与分析流程")
        print("=" * 70)
        print(f"⏰ 开始时间: {start_datetime}")
        print(f"📋 批次ID: {log_id}")
        print()

        # 显示配置信息
        print("📝 执行配置:")
        print(f"  • 热点限制: {hot_limit} 条/平台")
        print(f"  • 关键词限制: {keyword_limit} 个/领域")
        print(f"  • LLM分析: {'✓ 启用' if not no_llm else '✗ 禁用'}")
        print(f"  • 爬虫触发: {'✓ 启用' if not no_crawl else '✗ 禁用'}")

        if platforms:
            print(f"  • 热点平台: {', '.join(platforms)}")

        if domains:
            print(f"  • 分析领域: {', '.join(domains)}")

        if crawl_platforms:
            print(f"  • 爬虫平台: {', '.join(crawl_platforms)}")
        print()

        # =============== Step 1: 获取热点 ===============
        print("[Step 1/4] 🔍 获取热点新闻")
        print("-" * 70)

        if platforms:
            selected_platforms = platforms
        else:
            selected_platforms = [p.platform_code for p in settings.get_hot_platforms()]

        repo = HotNewsRepository()

        print(f"从 {len(selected_platforms)} 个平台获取热点...")

        total = 0
        platform_results = []
        for platform in selected_platforms:
            try:
                fetcher = HotNewsFactory.get_fetcher(platform)
                news_list = await fetcher.fetch(hot_limit)
                if news_list:
                    repo.bulk_save(news_list)
                    total += len(news_list)
                    platform_results.append((platform, len(news_list), "✓"))
                    print(f"  ✓ {platform:15} {len(news_list):3} 条")
                else:
                    platform_results.append((platform, 0, "✗"))
                    print(f"  - {platform:15}  无数据")
            except Exception as e:
                platform_results.append((platform, 0, "✗"))
                print(f"  ✗ {platform:15} 错误: {str(e)[:50]}")

        await HotNewsFactory.close_all()
        hot_count = total
        print(f"\n✅ 步骤完成: 共获取 {hot_count} 条热点")

        # =============== Step 2: 领域分析 ===============
        print("\n[Step 2/4] 🎯 分析领域匹配")
        print("-" * 70)

        domain_configs = settings.get_domains()
        if domains:
            selected_domains = [d for d in domain_configs if d.domain_name in domains]
        else:
            selected_domains = domain_configs

        if selected_domains and not no_llm:
            print(f"使用 {len(selected_domains)} 个领域进行分析...")
            for domain in selected_domains:
                print(f"  • {domain.domain_name} (ID:{domain.id})")

            news_list = repo.get_recent(hot_limit)
            print(f"\n从 {len(news_list)} 条热点中进行分析...")
            matched_count = await _analyze_domain(news_list, selected_domains)
            print(f"\n✅ 步骤完成: 匹配 {matched_count} 条热点")
        elif not selected_domains and not no_llm:
            print("⏭️  跳过分析 (未配置领域)")
            print("✅ 步骤完成: 跳过")
        else:
            print("⏭️  跳过分析 (--no-llm 标志)")
            print("✅ 步骤完成: 跳过")

        # =============== Step 3: 提取关键词 ===============
        print("\n[Step 3/4] 🔑 提取关键词")
        print("-" * 70)

        if not no_llm:
            if selected_domains:
                print(f"从 {len(selected_domains)} 个领域的匹配热点中提取关键词...")
                keyword_count = await _extract_keywords(selected_domains, keyword_limit, run_batch_id=log_id)
                print(f"\n✅ 步骤完成: 提取 {keyword_count} 个关键词")
            else:
                print("未配置领域，从所有热点中提取关键词...")
                all_news_list = repo.get_recent(hot_limit)
                keyword_count = await _extract_keywords([], keyword_limit, all_news=all_news_list, run_batch_id=log_id)
                print(f"\n✅ 步骤完成: 提取 {keyword_count} 个关键词")
        else:
            print("⏭️  跳过提取 (--no-llm 标志)")
            print("✅ 步骤完成: 跳过")

        # =============== Step 4: 触发爬虫 ===============
        print("\n[Step 4/4] 🕷️  触发爬虫")
        print("-" * 70)

        if not no_crawl:
            trigger = CrawlTrigger()
            keyword_repo = KeywordRepository()

            if crawl_platforms:
                crawl_platform_list = crawl_platforms
            else:
                crawl_platform_list = [
                    p.platform_code for p in settings.get_crawler_platforms()
                ]

            print(f"目标爬虫平台: {', '.join(crawl_platform_list)}")

            # 使用run_batch_id过滤，只爬取当前运行批次的关键词
            keywords = keyword_repo.get_by_batch_never_crawled(run_batch_id=log_id, limit=20)

            if keywords:
                print(f"准备爬取 {len(keywords)} 个关键词...")
                success_count = 0
                fail_count = 0

                for kw in keywords:
                    for platform in crawl_platform_list:
                        try:
                            if trigger.trigger_crawl(kw["keyword"], platform, 30, 10):
                                keyword_repo.increment_search_count(kw["id"])
                                crawl_count += 1
                                success_count += 1
                                print(f"  ✓ {kw['keyword']:20} @ {platform}")
                            else:
                                fail_count += 1
                                print(f"  ✗ {kw['keyword']:20} @ {platform} (失败)")
                        except Exception as e:
                            fail_count += 1
                            print(f"  ✗ {kw['keyword']:20} @ {platform} (错误: {str(e)[:30]})")

                print(f"\n✅ 步骤完成: 成功触发 {success_count} 次, 失败 {fail_count} 次")
            else:
                print("⏭️  无待爬取关键词 (当前批次无提取的关键词)")
                print("✅ 步骤完成: 跳过")
        else:
            print("⏭️  跳过爬虫 (--no-crawl 标志)")
            print("✅ 步骤完成: 跳过")

        # =============== 执行总结 ===============
        end_time = time.time()
        elapsed_time = end_time - start_time
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("\n" + "=" * 70)
        print("  ✅ 执行完成!")
        print("=" * 70)
        print(f"📊 执行结果:")
        print(f"  • 获取热点: {hot_count:4} 条")
        print(f"  • 匹配热点: {matched_count:4} 条")
        print(f"  • 提取关键词: {keyword_count:3} 个")
        print(f"  • 触发爬虫: {crawl_count:3} 次")
        print()
        print(f"⏱️  执行耗时: {elapsed_time:.2f} 秒")
        print(f"🔚 结束时间: {end_datetime}")
        print("=" * 70)

        log_repo.complete_task(
            log_id, "success", hot_count, matched_count, keyword_count, crawl_count
        )

    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time

        print("\n" + "=" * 70)
        print("  ❌ 执行失败!")
        print("=" * 70)
        print(f"❌ 错误信息: {str(e)}")
        print(f"⏱️  执行耗时: {elapsed_time:.2f} 秒")
        print("=" * 70)

        log_repo.complete_task(
            log_id,
            "failed",
            hot_count,
            matched_count,
            keyword_count,
            crawl_count,
            str(e),
        )
        raise


@cmd_run.command()
def run_pipeline(
    platforms: Optional[List[str]] = typer.Argument(None, help="热点平台列表"),
    domains: Optional[List[str]] = typer.Option(
        None, "--domains", "-d", help="领域列表"
    ),
    crawl_platforms: Optional[List[str]] = typer.Option(
        None, "--crawl-platforms", "-cp", help="爬虫平台"
    ),
    hot_limit: int = typer.Option(50, "--hot-limit", "-hl", help="获取热点数量"),
    keyword_limit: int = typer.Option(
        20, "--keyword-limit", "-kl", help="提取关键词数量"
    ),
    no_llm: bool = typer.Option(False, "--no-llm", help="跳过LLM分析，直接提取关键词"),
    no_crawl: bool = typer.Option(False, "--no-crawl", help="跳过爬虫触发"),
):
    """
    一键执行完整流程：获取热点 -> 分析匹配 -> 提取关键词 -> 触发爬虫

    示例:
        hot-news run weibo zhihu 科技 金融
        hot-news run --hot-limit 100 --crawl-platforms xhs dy
    """
    asyncio.run(
        _run_pipeline_inner(
            platforms,
            domains,
            crawl_platforms,
            hot_limit,
            keyword_limit,
            no_llm,
            no_crawl,
        )
    )
