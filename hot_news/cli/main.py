# =====================================================
# Hot News Module - CLI Main Entry
# =====================================================

import typer
from typing import Optional
from pathlib import Path

from .commands.config import cmd_config
from .commands.show import cmd_show
from .commands.fetch import fetch
from .commands.analyze import analyze_domain_match
from .commands.extract import extract_keywords
from .commands.crawl import trigger_crawl
from .commands.run import run_pipeline
import typer
from typing import Optional, List
import asyncio
from pathlib import Path

app = typer.Typer(name="hot-news", add_completion=False, help="热点新闻获取与分析模块")

app.add_typer(cmd_config, name="config", help="配置管理")
app.add_typer(cmd_show, name="show", help="查看数据")


@app.command("fetch")
def fetch_cmd(
    platforms: Optional[List[str]] = typer.Argument(
        None, help="平台列表，如 weibo zhihu bilibili"
    ),
    limit: int = typer.Option(50, "--limit", "-l", help="每个平台获取的数量"),
):
    fetch(platforms, limit)


@app.command("analyze")
def analyze_cmd(
    domains: Optional[List[str]] = typer.Argument(None, help="领域列表，如 科技 金融"),
    limit: int = typer.Option(100, "--limit", "-l", help="分析的热点数量"),
    min_confidence: float = typer.Option(
        0.5, "--min-confidence", "-c", help="最小置信度"
    ),
):
    analyze_domain_match(domains, limit, min_confidence)


@app.command("extract")
def extract_cmd(
    domains: Optional[List[str]] = typer.Argument(None, help="领域列表，如 科技 金融"),
    limit: int = typer.Option(50, "--limit", "-l", help="提取的热点数量"),
):
    extract_keywords(domains, limit)


@app.command("crawl")
def crawl_cmd(
    platforms: List[str] = typer.Argument(..., help="爬虫平台列表，如 xhs dy bili"),
    keywords: Optional[List[str]] = typer.Option(
        None, "--keyword", "-k", help="指定关键词"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="触发爬取的数量"),
):
    trigger_crawl(platforms, keywords, limit)


@app.command("run")
def run_cmd(
    platforms: Optional[List[str]] = typer.Argument(None, help="热点平台列表"),
    domains: Optional[List[str]] = typer.Option(
        None, "-d", "--domains", help="领域列表"
    ),
    crawl_platforms: Optional[List[str]] = typer.Option(
        None, "--crawl-platforms", "-cp", help="爬虫平台"
    ),
    hot_limit: int = typer.Option(50, "--hot-limit", "-hl", help="获取热点数量"),
    keyword_limit: int = typer.Option(
        20, "--keyword-limit", "-kl", help="提取关键词数量"
    ),
    no_llm: bool = typer.Option(False, "--no-llm", help="跳过LLM分析"),
    no_crawl: bool = typer.Option(False, "--no-crawl", help="跳过爬虫触发"),
):
    # ✅ 直接调用run_pipeline，不需要asyncio.run()
    # 因为run_pipeline函数内部已经处理了asyncio
    run_pipeline(
        platforms,
        domains,
        crawl_platforms,
        hot_limit,
        keyword_limit,
        no_llm,
        no_crawl,
    )


@app.command()
def init_db():
    """初始化数据库表"""
    import sys
    from pathlib import Path

    sql_file = (
        Path(__file__).parent.parent / "database" / "migrations" / "001_initial.sql"
    )

    if not sql_file.exists():
        print(f"[Error] SQL文件不存在: {sql_file}")
        raise typer.Exit(1)

    from ..config.settings import get_settings
    import pymysql

    settings = get_settings()

    print(f"[Init] 连接数据库: {settings._db_config.host}:{settings._db_config.port}")

    try:
        conn = pymysql.connect(**settings._db_config.connection_kwargs)
        with conn.cursor() as cursor:
            with open(sql_file, "r", encoding="utf-8") as f:
                sql = f.read()
                for statement in sql.split(";"):
                    statement = statement.strip()
                    if statement:
                        cursor.execute(statement)
            conn.commit()
        conn.close()

        print("[Success] 数据库初始化完成！")

    except Exception as e:
        print(f"[Error] 初始化失败: {e}")
        raise typer.Exit(1)


def cli():
    """CLI入口点"""
    app()


if __name__ == "__main__":
    cli()
