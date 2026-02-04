# =====================================================
# Hot News Module - Keyword Repository
# =====================================================

from typing import List, Optional, Dict
from datetime import datetime
from ..connection import get_db
from hot_news.models.entities import KeywordResult


class KeywordRepository:
    """关键词数据访问层"""

    def __init__(self):
        self.db = get_db()

    def save_keyword(self, kw: KeywordResult, run_batch_id: int = None) -> int:
        """保存关键词，返回ID"""
        sql = """
            INSERT INTO extracted_keywords
            (keyword, source_news_id, domain_id, llm_provider, confidence, run_batch_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE
                source_news_id=VALUES(source_news_id),
                domain_id=VALUES(domain_id),
                llm_provider=VALUES(llm_provider),
                confidence=VALUES(confidence),
                run_batch_id=VALUES(run_batch_id)
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(
                sql,
                (
                    kw.keyword,
                    kw.source_news_id,
                    kw.domain_id,
                    kw.llm_provider,
                    kw.confidence,
                    run_batch_id,
                ),
            )
            cursor.execute("SELECT LAST_INSERT_ID()")
            return cursor.fetchone()[0]

    def bulk_save(self, keywords: List[KeywordResult], run_batch_id: int = None) -> int:
        """批量保存关键词"""
        if not keywords:
            return 0

        sql = """
            INSERT INTO extracted_keywords
            (keyword, source_news_id, domain_id, llm_provider, confidence, run_batch_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE
                source_news_id=VALUES(source_news_id),
                domain_id=VALUES(domain_id),
                llm_provider=VALUES(llm_provider),
                confidence=VALUES(confidence),
                run_batch_id=VALUES(run_batch_id)
        """
        data = [
            (k.keyword, k.source_news_id, k.domain_id, k.llm_provider, k.confidence, run_batch_id)
            for k in keywords
        ]

        with self.db.get_cursor() as cursor:
            cursor.executemany(sql, data)
            return len(keywords)

    def get_by_id(self, keyword_id: int) -> Optional[dict]:
        """根据ID获取关键词"""
        sql = "SELECT * FROM extracted_keywords WHERE id = %s"
        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, (keyword_id,))
            row = cursor.fetchone()
            return dict(zip([c[0] for c in cursor.description], row)) if row else None

    def get_all(self, limit: int = 100, domain_id: int = None) -> List[dict]:
        """获取所有关键词"""
        sql = "SELECT * FROM extracted_keywords"
        params = []
        if domain_id:
            sql += " WHERE domain_id = %s"
            params.append(domain_id)
        sql += " ORDER BY confidence DESC, created_at DESC LIMIT %s"
        params.append(limit)

        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            return [dict(zip([c[0] for c in cursor.description], row)) for row in rows]

    def get_by_keyword(self, keyword: str) -> Optional[dict]:
        """根据关键词获取"""
        sql = "SELECT * FROM extracted_keywords WHERE keyword = %s"
        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, (keyword,))
            row = cursor.fetchone()
            return dict(zip([c[0] for c in cursor.description], row)) if row else None

    def get_by_domain(self, domain_id: int, limit: int = 100) -> List[dict]:
        """根据领域获取关键词"""
        sql = """
            SELECT * FROM extracted_keywords 
            WHERE domain_id = %s 
            ORDER BY confidence DESC, created_at DESC 
            LIMIT %s
        """
        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, (domain_id, limit))
            rows = cursor.fetchall()
            return [dict(zip([c[0] for c in cursor.description], row)) for row in rows]

    def increment_search_count(self, keyword_id: int) -> bool:
        """增加搜索次数"""
        sql = """
            UPDATE extracted_keywords 
            SET search_count = search_count + 1, last_used = NOW()
            WHERE id = %s
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(sql, (keyword_id,))
            return cursor.rowcount > 0

    def get_never_crawled(self, domain_id: int = None, limit: int = 50, run_batch_id: int = None) -> List[dict]:
        """获取从未爬取过的关键词

        Args:
            domain_id: 领域ID，可选
            limit: 限制数量
            run_batch_id: 运行批次ID，如果指定则只返回该批次的关键词
        """
        sql = """
            SELECT * FROM extracted_keywords
            WHERE search_count = 0
        """
        params = []
        if domain_id:
            sql += " AND domain_id = %s"
            params.append(domain_id)
        if run_batch_id:
            sql += " AND run_batch_id = %s"
            params.append(run_batch_id)
        sql += " ORDER BY confidence DESC LIMIT %s"
        params.append(limit)

        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            return [dict(zip([c[0] for c in cursor.description], row)) for row in rows]

    def get_by_batch(self, run_batch_id: int, limit: int = 100) -> List[dict]:
        """获取指定运行批次中的所有关键词

        Args:
            run_batch_id: 运行批次ID
            limit: 限制数量

        Returns:
            关键词列表
        """
        sql = """
            SELECT * FROM extracted_keywords
            WHERE run_batch_id = %s
            ORDER BY confidence DESC, created_at DESC
            LIMIT %s
        """
        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, (run_batch_id, limit))
            rows = cursor.fetchall()
            return [dict(zip([c[0] for c in cursor.description], row)) for row in rows]

    def get_by_batch_never_crawled(self, run_batch_id: int, domain_id: int = None, limit: int = 50) -> List[dict]:
        """获取指定运行批次中未爬取过的关键词

        这是推荐用的方法，用于爬虫触发，确保只爬取当前运行提取的关键词

        Args:
            run_batch_id: 运行批次ID
            domain_id: 领域ID，可选
            limit: 限制数量

        Returns:
            未爬取的关键词列表
        """
        sql = """
            SELECT * FROM extracted_keywords
            WHERE run_batch_id = %s AND search_count = 0
        """
        params = [run_batch_id]
        if domain_id:
            sql += " AND domain_id = %s"
            params.append(domain_id)
        sql += " ORDER BY confidence DESC LIMIT %s"
        params.append(limit)

        with self.db.get_cursor(commit=False) as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            return [dict(zip([c[0] for c in cursor.description], row)) for row in rows]

    def count(self, domain_id: int = None) -> int:
        """统计数量"""
        sql = "SELECT COUNT(*) FROM extracted_keywords"
        params = []
        if domain_id:
            sql += " WHERE domain_id = %s"
            params.append(domain_id)

        with self.db.get_cursor(commit=False) as cursor:
            if params:
                cursor.execute(sql, tuple(params))
            else:
                cursor.execute(sql)
            return cursor.fetchone()[0]

    def delete(self, keyword_id: int) -> bool:
        """删除关键词"""
        sql = "DELETE FROM extracted_keywords WHERE id = %s"
        with self.db.get_cursor() as cursor:
            cursor.execute(sql, (keyword_id,))
            return cursor.rowcount > 0
