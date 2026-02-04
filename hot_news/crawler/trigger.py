# =====================================================
# Hot News Module - Crawler Trigger
# =====================================================

import subprocess
import sys
import os
from typing import List, Optional
from ..config.settings import get_settings
from ..database import KeywordRepository, CrawlLogRepository
from ..models.entities import CrawlTask


class CrawlTrigger:
    """爬虫触发器"""

    def __init__(self):
        self.settings = get_settings()
        self.keyword_repo = KeywordRepository()
        self.crawl_log_repo = CrawlLogRepository()

    def trigger_crawl(
        self, keyword: str, platform: str, max_comments: int = 10
    ) -> bool:
        """
        触发爬虫

        Args:
            keyword: 搜索关键词
            platform: 平台代码
            max_comments: 最大评论数

        Returns:
            是否成功触发
        """
        process = None
        try:
            # 获取 MediaCrawler 目录的绝对路径
            mediacrawler_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "..", "MediaCrawler"
            ))

            # 优先使用 conda sentiment 环境
            conda_python = r"E:\develop\anaconda3\envs\sentiment\python.exe"
            if os.path.exists(conda_python):
                python_exe = conda_python
            else:
                python_exe = sys.executable

            cmd = [
                python_exe,
                "main.py",
                "--platform",
                platform,
                "--lt",
                "cookie",
                "--type",
                "search",
                "--keywords",
                keyword,
                "--max_comments_count_singlenotes",
                str(max_comments),
                "--save_data_option",
                "db",
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = mediacrawler_dir
            env.pop("PYTHONHOME", None)

            # 关键：设置工作目录为 MediaCrawler 目录
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=mediacrawler_dir  # 设置工作目录
            )

            stdout, stderr = process.communicate(timeout=600)

            if process.returncode == 0:
                print(f"[CrawlTrigger] 成功爬取: {keyword} @ {platform}")
                return True
            else:
                stdout_text = stdout.decode("utf-8", errors="ignore")
                stderr_text = stderr.decode("utf-8", errors="ignore")
                print(f"[CrawlTrigger] 爬取失败: {keyword} @ {platform}")
                print(f"  返回码: {process.returncode}")
                print(f"  命令: {' '.join(cmd)}")
                print(f"  工作目录: {mediacrawler_dir}")
                if stdout_text.strip():
                    print(f"  标准输出:\n{stdout_text[-2000:]}")
                if stderr_text.strip():
                    print(f"  标准错误:\n{stderr_text[-2000:]}")
                return False

        except subprocess.TimeoutExpired:
            if process:
                process.kill()
            print(f"[CrawlTrigger] 爬取超时: {keyword} @ {platform}")
            return False
        except Exception as e:
            print(f"[CrawlTrigger] 触发失败: {e}")
            return False

    async def trigger_crawl_async(
        self,
        keyword: str,
        platform: str,
        keyword_id: int = 0,
        max_comments: int = 10,
    ) -> bool:
        """
        异步触发爬虫（实际上还是同步执行）
        """
        log_id = None
        if keyword_id:
            log_id = self.crawl_log_repo.create_log(keyword_id, platform, "processing")

        success = self.trigger_crawl(keyword, platform, max_comments)

        if log_id:
            if success:
                self.crawl_log_repo.update_log(log_id, "completed", 0, 0)
            else:
                self.crawl_log_repo.update_log(log_id, "failed", 0, 0, "爬取失败")

        if keyword_id and success:
            self.keyword_repo.increment_search_count(keyword_id)

        return success

    def trigger_batch(
        self,
        keywords: List[str],
        platforms: List[str],
        max_comments: int = 10,
    ) -> int:
        """
        批量触发爬虫

        Returns:
            成功触发的数量
        """
        success_count = 0

        for keyword in keywords:
            for platform in platforms:
                if self.trigger_crawl(keyword, platform, max_comments):
                    success_count += 1

        return success_count

    def trigger_from_keywords(
        self,
        keyword_ids: Optional[List[int]] = None,
        domain_id: Optional[int] = None,
        platforms: Optional[List[str]] = None,
        limit: int = 10,
    ) -> int:
        """
        从数据库关键词触发爬虫

        Args:
            keyword_ids: 关键词ID列表
            domain_id: 领域ID
            platforms: 平台列表
            limit: 数量限制

        Returns:
            成功触发的数量
        """
        if keyword_ids:
            keywords = []
            for kid in keyword_ids:
                kw = self.keyword_repo.get_by_id(kid)
                if kw:
                    keywords.append(kw)
        elif domain_id:
            keywords = self.keyword_repo.get_by_domain(domain_id, limit)
        else:
            keywords = self.keyword_repo.get_never_crawled(limit=limit)

        if not platforms:
            crawler_platforms = self.settings.get_crawler_platforms(enabled_only=True)
            platforms = [p.platform_code for p in crawler_platforms]

        success_count = 0
        for kw in keywords:
            for platform in platforms:
                if self.trigger_crawl(kw["keyword"], platform, 10):
                    success_count += 1
                    self.keyword_repo.increment_search_count(kw["id"])

        return success_count
