# =====================================================
# Hot News Module - Models/Entities
# =====================================================

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class HotNewsItem:
    """热点新闻项"""

    news_id: str
    platform_code: str
    title: str
    url: str = ""
    score: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DomainInfo:
    """领域信息"""

    id: int
    name: str
    keywords: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class DomainMatchResult:
    """领域匹配结果"""

    news_id: str
    domain_id: int
    domain_name: str
    is_match: bool
    confidence: float = 0.0
    match_level: str = ""
    reason: str = ""


@dataclass
class KeywordResult:
    """关键词提取结果"""

    keyword: str
    source_news_id: str
    domain_id: int
    llm_provider: str = ""
    confidence: float = 0.0
    primary_keyword: str = ""


@dataclass
class LLMResponse:
    """LLM响应"""

    content: str
    provider: str
    model_name: str
    tokens_used: int = 0


@dataclass
class TaskResult:
    """任务执行结果"""

    task_name: str
    status: str  # running/success/failed
    hot_count: int = 0
    matched_count: int = 0
    keyword_count: int = 0
    crawl_triggered: int = 0
    error_message: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class CrawlTask:
    """爬虫任务"""

    keyword: str
    platform: str
    keyword_id: int = 0
    max_notes: int = 30
    max_comments: int = 10
