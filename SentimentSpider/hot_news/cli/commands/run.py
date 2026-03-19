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
from hot_news.logger import logger, info, debug, warning, error
from hot_news.sync import UnifiedDataSync

# 并发控制：最多同时执行的LLM调用数
MAX_CONCURRENT_LLM_CALLS = 5

cmd_run = typer.Typer(help="一键执行完整流程")


async def _analyze_domain(news_list, selected_domains):
    """异步分析领域匹配 - 使用并发加速"""
    checker = DomainChecker()
    analysis_repo = AnalysisRepository()
    matched_count = 0

    # 创建信号量限制并发数
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

    async def check_and_save(news_row, domain):
        """检查单个新闻-领域对，并保存结果"""
        async with semaphore:  # 限制并发数
            news = HotNewsItem(
                news_id=news_row["news_id"],
                platform_code=news_row["platform_code"],
                title=news_row["title"],
                url=news_row.get("url", ""),
                description=news_row.get("description", ""),
            )

            try:
                result = await checker.check_domain(news, domain)
                analysis_repo.save_analysis(
                    news_id=result.news_id,
                    domain_id=result.domain_id,
                    is_match=result.is_match,
                    llm_provider="",
                    analysis_content=result.reason,
                    confidence=result.confidence,
                )
                return 1 if result.is_match else 0
            except Exception as e:
                error(
                    f"  [错误] {news_row['title'][:20]}... @ {domain.domain_name}: {str(e)[:40]}"
                )
                return 0

    # 创建所有的检查任务
    tasks = []
    for news_row in news_list:
        for domain in selected_domains:
            task = check_and_save(news_row, domain)
            tasks.append(task)

    # 并发执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 统计匹配数
    for result in results:
        if isinstance(result, int):
            matched_count += result

    return matched_count


async def _extract_keywords(
    selected_domains, keyword_limit, all_news=None, run_batch_id: int = None
):
    """异步提取关键词 - 使用并发加速

    Args:
        selected_domains: 选定的领域列表，如果为空则使用all_news
        keyword_limit: 每个领域最多提取的关键词数
        all_news: 当selected_domains为空时，从这个新闻列表中提取关键词
        run_batch_id: 运行批次ID，用于追踪该批次提取的关键词
    """
    extractor = KeywordExtractor()
    keyword_repo = KeywordRepository()
    total_keywords = 0

    # 创建信号量限制并发数
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)

    async def extract_and_save(news_row, domain):
        """提取关键词并保存"""
        async with semaphore:
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
                    return len(keywords)
            except Exception as e:
                error(
                    f"  [错误] 提取关键词失败 {news_row['title'][:20]}...: {str(e)[:40]}"
                )
            return 0

    tasks = []

    # 如果有domain配置，从匹配的热点中提取
    if selected_domains:
        for domain in selected_domains:
            matched_news = AnalysisRepository().get_matched_news(
                domain.id, keyword_limit
            )
            for news_row in matched_news:
                task = extract_and_save(news_row, domain)
                tasks.append(task)

    # 如果没有domain配置，从所有热点中提取（不筛选）
    elif all_news:
        from hot_news.config.settings import DomainConfig

        virtual_domain = DomainConfig(
            id=0,
            domain_name="通用",
            domain_keywords="热点,新闻",
            is_enabled=1,
        )

        for news_row in all_news[:keyword_limit]:
            task = extract_and_save(news_row, virtual_domain)
            tasks.append(task)

    # 并发执行所有任务
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, int):
                total_keywords += result

    return total_keywords


async def _run_pipeline_inner(
    platforms,
    domains,
    crawl_platforms,
    hot_limit,
    keyword_limit,
    no_llm,
    no_crawl,
    no_sync=False,
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
    sync_count = 0

    try:
        # =============== 显示执行配置 ===============
        info("\n" + "=" * 70)
        info("  🚀 热点新闻获取与分析流程")
        info("=" * 70)
        info(f"⏰ 开始时间: {start_datetime}")
        info(f"📋 批次ID: {log_id}")
        info("")

        # 显示配置信息
        info("📝 执行配置:")
        info(f"  • 热点限制: {hot_limit} 条/平台")
        info(f"  • 关键词限制: {keyword_limit} 个/领域")
        info(f"  • LLM分析: {'✓ 启用' if not no_llm else '✗ 禁用'}")
        info(f"  • 爬虫触发: {'✓ 启用' if not no_crawl else '✗ 禁用'}")
        info(f"  • 数据同步: {'✓ 启用' if not no_sync else '✗ 禁用'}")

        if platforms:
            info(f"  • 热点平台: {', '.join(platforms)}")

        if domains:
            info(f"  • 分析领域: {', '.join(domains)}")

        if crawl_platforms:
            info(f"  • 爬虫平台: {', '.join(crawl_platforms)}")
        info("")

        # =============== Step 1: 获取热点 ===============
        info("[Step 1/5] 🔍 获取热点新闻")
        info("-" * 70)

        if platforms:
            selected_platforms = platforms
        else:
            selected_platforms = [p.platform_code for p in settings.get_hot_platforms()]

        repo = HotNewsRepository()

        info(f"从 {len(selected_platforms)} 个平台获取热点...")

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
                    info(f"  ✓ {platform:15} {len(news_list):3} 条")
                else:
                    platform_results.append((platform, 0, "✗"))
                    info(f"  - {platform:15}  无数据")
            except Exception as e:
                platform_results.append((platform, 0, "✗"))
                error(f"  ✗ {platform:15} 错误: {str(e)[:50]}")

        await HotNewsFactory.close_all()
        hot_count = total
        info(f"\n✅ 步骤完成: 共获取 {hot_count} 条热点")

        # =============== Step 2: 领域分析 ===============
        info("\n[Step 2/5] 🎯 分析领域匹配")
        info("-" * 70)

        domain_configs = settings.get_domains()
        if domains:
            selected_domains = [d for d in domain_configs if d.domain_name in domains]
        else:
            selected_domains = domain_configs

        if selected_domains and not no_llm:
            info(f"使用 {len(selected_domains)} 个领域进行分析...")
            for domain in selected_domains:
                info(f"  • {domain.domain_name} (ID:{domain.id})")

            news_list = repo.get_recent(hot_limit)
            info(f"\n从 {len(news_list)} 条热点中进行分析...")
            info(f"⚡ 并发执行 (最多 {MAX_CONCURRENT_LLM_CALLS} 并发)")
            matched_count = await _analyze_domain(news_list, selected_domains)
            info(f"\n✅ 步骤完成: 匹配 {matched_count} 条热点")
        elif not selected_domains and not no_llm:
            info("⏭️  跳过分析 (未配置领域)")
            info("✅ 步骤完成: 跳过")
        else:
            info("⏭️  跳过分析 (--no-llm 标志)")
            info("✅ 步骤完成: 跳过")

        # =============== Step 3: 提取关键词 ===============
        info("\n[Step 3/5] 🔑 提取关键词")
        info("-" * 70)

        if not no_llm:
            if selected_domains:
                info(f"从 {len(selected_domains)} 个领域的匹配热点中提取关键词...")
                info(f"⚡ 并发执行 (最多 {MAX_CONCURRENT_LLM_CALLS} 并发)")
                keyword_count = await _extract_keywords(
                    selected_domains, keyword_limit, run_batch_id=log_id
                )
                info(f"\n✅ 步骤完成: 提取 {keyword_count} 个关键词")
            else:
                info("未配置领域，从所有热点中提取关键词...")
                info(f"⚡ 并发执行 (最多 {MAX_CONCURRENT_LLM_CALLS} 并发)")
                all_news_list = repo.get_recent(hot_limit)
                keyword_count = await _extract_keywords(
                    [], keyword_limit, all_news=all_news_list, run_batch_id=log_id
                )
                info(f"\n✅ 步骤完成: 提取 {keyword_count} 个关键词")
        else:
            info("⏭️  跳过提取 (--no-llm 标志)")
            info("✅ 步骤完成: 跳过")

        # =============== Step 4: 触发爬虫 ===============
        info("\n[Step 4/5] 🕷️  触发爬虫")
        info("-" * 70)

        # 记录爬虫开始前的时间戳（用于后续同步）
        crawl_start_ts = int(time.time() * 1000)

        if not no_crawl:
            trigger = CrawlTrigger()
            keyword_repo = KeywordRepository()

            if crawl_platforms:
                crawl_platform_list = crawl_platforms
            else:
                crawl_platform_list = [
                    p.platform_code for p in settings.get_crawler_platforms()
                ]

            info(f"目标爬虫平台: {', '.join(crawl_platform_list)}")

            # 使用run_batch_id过滤，只爬取当前运行批次的关键词
            keywords = keyword_repo.get_by_batch_never_crawled(
                run_batch_id=log_id, limit=20
            )

            if keywords:
                info(f"准备爬取 {len(keywords)} 个关键词...")

                # 提取关键词列表
                keyword_list = [kw["keyword"] for kw in keywords]
                keyword_ids = {kw["keyword"]: kw["id"] for kw in keywords}

                # 使用批量爬取方法，一次性传递所有关键词
                success_count = trigger.trigger_batch(
                    keywords=keyword_list,
                    platforms=crawl_platform_list,
                    max_comments=10,
                )

                # 更新关键词的搜索次数
                for kw in keywords:
                    keyword_repo.increment_search_count(kw["id"])
                    crawl_count += 1

                info(f"\n✅ 步骤完成: 成功触发 {success_count} 个关键词")
            else:
                info("⏭️  无待爬取关键词 (当前批次无提取的关键词)")
                info("✅ 步骤完成: 跳过")
        else:
            info("⏭️  跳过爬虫 (--no-crawl 标志)")
            info("✅ 步骤完成: 跳过")

        # =============== Step 5: 数据同步 ===============
        info("\n[Step 5/5] 🔄 同步到统一表")
        info("-" * 70)

        if not no_sync:
            syncer = UnifiedDataSync()

            if not no_crawl and crawl_count > 0:
                # 有爬虫执行：同步本次爬取的数据（基于爬虫开始时间）
                info("同步本次爬取的数据到统一表...")
                crawl_platform_list_for_sync = crawl_platforms or [
                    p.platform_code for p in settings.get_crawler_platforms()
                ]
                try:
                    sync_stats = await syncer.sync_all_platforms(
                        platforms=crawl_platform_list_for_sync,
                        sync_comments=True,
                        batch_size=500,
                        incremental=True,
                        since_ts=crawl_start_ts,
                    )

                    for key, count in sync_stats.items():
                        if count > 0:
                            info(f"  ✓ {key}: {count} 条")
                        sync_count += count

                    info(f"\n✅ 步骤完成: 同步 {sync_count} 条数据")
                except Exception as e:
                    error(f"  ✗ 同步失败: {str(e)[:50]}")
                    info("✅ 步骤完成: 同步出错")
            else:
                # 无爬虫执行：增量同步所有平台的新数据
                info("增量同步各平台数据...")
                try:
                    sync_stats = await syncer.sync_all_platforms(
                        platforms=None,
                        sync_comments=True,
                        batch_size=500,
                        incremental=True,
                    )

                    for key, count in sync_stats.items():
                        if count > 0:
                            info(f"  ✓ {key}: {count} 条")
                        sync_count += count

                    info(f"\n✅ 步骤完成: 同步 {sync_count} 条数据")
                except Exception as e:
                    error(f"  ✗ 同步失败: {str(e)[:50]}")
                    info("✅ 步骤完成: 同步出错")
        else:
            info("⏭️  跳过同步 (--no-sync 标志)")
            info("✅ 步骤完成: 跳过")

        # =============== 执行总结 ===============
        end_time = time.time()
        elapsed_time = end_time - start_time
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        info("\n" + "=" * 70)
        info("  ✅ 执行完成!")
        info("=" * 70)
        info(f"📊 执行结果:")
        info(f"  • 获取热点: {hot_count:4} 条")
        info(f"  • 匹配热点: {matched_count:4} 条")
        info(f"  • 提取关键词: {keyword_count:3} 个")
        info(f"  • 触发爬虫: {crawl_count:3} 次")
        info(f"  • 数据同步: {sync_count:4} 条")
        info("")
        info(f"⏱️  执行耗时: {elapsed_time:.2f} 秒")
        info(f"🔚 结束时间: {end_datetime}")
        info("=" * 70)

        log_repo.complete_task(
            log_id, "success", hot_count, matched_count, keyword_count, crawl_count
        )

    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time

        info("\n" + "=" * 70)
        info("  ❌ 执行失败!")
        info("=" * 70)
        error(f"❌ 错误信息: {str(e)}")
        info(f"⏱️  执行耗时: {elapsed_time:.2f} 秒")
        info("=" * 70)

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
    no_sync: bool = typer.Option(False, "--no-sync", help="跳过数据同步到统一表"),
):
    """
    一键执行完整流程：获取热点 -> 分析匹配 -> 提取关键词 -> 触发爬虫 -> 数据同步

    示例:
        hot-news run weibo zhihu 科技 金融
        hot-news run --hot-limit 100 --crawl-platforms xhs dy
        hot-news run --no-sync  # 跳过数据同步
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
            no_sync,
        )
    )
