# -*- coding: utf-8 -*-
"""
话题聚类数据仓库

包含 TopicEventRepo, TopicEvolutionRepo, TopicMergeRepo, TopicContentRepo
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, date
import json
import numpy as np

from .connection import (
    get_connection,
    execute_query,
    execute_many,
    execute_update,
    execute_insert,
)


class TopicEventRepo:
    """话题事件仓库 - 操作 topic_event 表"""

    @staticmethod
    def insert(
        event_name: str,
        centroid_embedding: np.ndarray,
        keywords: Optional[List[Dict]] = None,
        similarity_threshold: float = 0.75,
        embedding_model: str = "hfl/chinese-roberta-wwm-ext",
    ) -> int:
        """
        插入新话题事件

        Args:
            event_name: 话题名称
            centroid_embedding: 质心嵌入向量
            keywords: 关键词列表
            similarity_threshold: 相似度阈值
            embedding_model: 嵌入模型名称

        Returns:
            新插入的话题ID
        """
        sql = """
            INSERT INTO topic_event
                (event_name, centroid_embedding, keywords, similarity_threshold,
                 embedding_model, content_count, first_content_at, last_content_at)
            VALUES (%s, %s, %s, %s, %s, 1, NOW(), NOW())
        """
        centroid_blob = centroid_embedding.astype(np.float32).tobytes()
        keywords_json = json.dumps(keywords, ensure_ascii=False) if keywords else None
        return execute_insert(
            sql, (event_name, centroid_blob, keywords_json, similarity_threshold, embedding_model)
        )

    @staticmethod
    def update_centroid(topic_id: int, centroid: np.ndarray, content_count: int) -> int:
        """
        更新话题质心和内容计数

        Args:
            topic_id: 话题ID
            centroid: 新的质心向量
            content_count: 新的内容计数

        Returns:
            影响的行数
        """
        sql = """
            UPDATE topic_event
            SET centroid_embedding = %s,
                content_count = %s,
                last_content_at = NOW()
            WHERE id = %s
        """
        centroid_blob = centroid.astype(np.float32).tobytes()
        return execute_update(sql, (centroid_blob, content_count, topic_id))

    @staticmethod
    def get_active_topics() -> List[Dict[str, Any]]:
        """
        获取所有活跃话题 (非 merged/ended)

        Returns:
            活跃话题列表，包含 id, event_name, centroid_embedding, content_count
        """
        sql = """
            SELECT id, event_name, centroid_embedding, content_count,
                   keywords, status
            FROM topic_event
            WHERE status NOT IN ('merged', 'ended')
            ORDER BY id
        """
        return execute_query(sql)

    @staticmethod
    def get_non_merged_topics() -> List[Dict[str, Any]]:
        """
        获取所有未合并话题 (包含 ended)

        Returns:
            话题列表，包含 id, event_name, centroid_embedding, content_count
        """
        sql = """
            SELECT id, event_name, centroid_embedding, content_count,
                   keywords, status
            FROM topic_event
            WHERE status != 'merged'
            ORDER BY id
        """
        return execute_query(sql)

    @staticmethod
    def update_status(topic_id: int, status: str, merged_into_id: Optional[int] = None) -> int:
        """
        更新话题状态

        Args:
            topic_id: 话题ID
            status: 新状态 (emerging/active/declining/ended/merged)
            merged_into_id: 合并目标话题ID

        Returns:
            影响的行数
        """
        if merged_into_id is not None:
            sql = "UPDATE topic_event SET status = %s, merged_into_id = %s WHERE id = %s"
            return execute_update(sql, (status, merged_into_id, topic_id))
        sql = "UPDATE topic_event SET status = %s WHERE id = %s"
        return execute_update(sql, (status, topic_id))

    @staticmethod
    def update_description(
        topic_id: int,
        event_name: str,
        event_description: Optional[str] = None,
        keywords: Optional[List[Dict]] = None,
    ) -> int:
        """
        更新话题描述信息 (LLM 生成)

        Args:
            topic_id: 话题ID
            event_name: 话题名称
            event_description: 话题描述
            keywords: 关键词列表

        Returns:
            影响的行数
        """
        sql = """
            UPDATE topic_event
            SET event_name = %s, event_description = %s, keywords = %s
            WHERE id = %s
        """
        keywords_json = json.dumps(keywords, ensure_ascii=False) if keywords else None
        return execute_update(sql, (event_name, event_description, keywords_json, topic_id))

    @staticmethod
    def update_sentiment_stats(
        topic_id: int,
        avg_score: float,
        sentiment_dist: Dict[str, int],
        dominant_sentiment: str,
        dominant_emotions: Optional[str] = None,
    ) -> int:
        """更新话题情感聚合统计"""
        sql = """
            UPDATE topic_event
            SET avg_sentiment_score = %s,
                sentiment_distribution = %s,
                dominant_sentiment = %s,
                dominant_emotions = %s
            WHERE id = %s
        """
        dist_json = json.dumps(sentiment_dist, ensure_ascii=False)
        return execute_update(
            sql, (avg_score, dist_json, dominant_sentiment, dominant_emotions, topic_id)
        )

    @staticmethod
    def update_heat_and_platform(
        topic_id: int,
        heat_level: str,
        platform_distribution: Dict[str, int],
        comment_count: int = 0,
    ) -> int:
        """更新话题热度和平台分布"""
        sql = """
            UPDATE topic_event
            SET heat_level = %s,
                platform_distribution = %s,
                comment_count = %s
            WHERE id = %s
        """
        platform_json = json.dumps(platform_distribution, ensure_ascii=False)
        return execute_update(sql, (heat_level, platform_json, comment_count, topic_id))

    @staticmethod
    def count_by_status() -> List[Dict[str, Any]]:
        """按状态统计话题数量"""
        sql = """
            SELECT status, COUNT(*) as cnt
            FROM topic_event
            GROUP BY status
            ORDER BY FIELD(status, 'emerging', 'active', 'declining', 'ended', 'merged')
        """
        return execute_query(sql)

    @staticmethod
    def get_topic_stats() -> Dict[str, Any]:
        """获取话题总体统计"""
        sql = """
            SELECT
                COUNT(*) as total_topics,
                SUM(CASE WHEN status NOT IN ('merged', 'ended') THEN 1 ELSE 0 END) as active_topics,
                SUM(content_count) as total_content,
                AVG(content_count) as avg_content_per_topic,
                MAX(content_count) as max_content_per_topic
            FROM topic_event
            WHERE status != 'merged'
        """
        return execute_query(sql, fetch_one=True)

    @staticmethod
    def reset_all_topics() -> int:
        """重置所有话题 (全量重聚类前调用)"""
        sql = "DELETE FROM topic_event"
        return execute_update(sql)


class TopicEvolutionRepo:
    """话题演化快照仓库 - 操作 topic_evolution 表"""

    @staticmethod
    def insert(
        event_id: int,
        snapshot_date: date,
        content_count_delta: int = 0,
        comment_count_delta: int = 0,
        content_count_total: int = 0,
        avg_sentiment_score: Optional[float] = None,
        sentiment_distribution: Optional[Dict] = None,
        hot_score: float = 0.0,
        interaction_count: int = 0,
        keywords: Optional[List[Dict]] = None,
        platform_distribution: Optional[Dict] = None,
    ) -> int:
        """
        插入或更新演化快照 (UPSERT)

        Returns:
            影响的行数
        """
        sql = """
            INSERT INTO topic_evolution
                (event_id, snapshot_date, content_count_delta, comment_count_delta,
                 content_count_total, avg_sentiment_score, sentiment_distribution,
                 hot_score, interaction_count, keywords, platform_distribution)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                content_count_delta = VALUES(content_count_delta),
                comment_count_delta = VALUES(comment_count_delta),
                content_count_total = VALUES(content_count_total),
                avg_sentiment_score = VALUES(avg_sentiment_score),
                sentiment_distribution = VALUES(sentiment_distribution),
                hot_score = VALUES(hot_score),
                interaction_count = VALUES(interaction_count),
                keywords = VALUES(keywords),
                platform_distribution = VALUES(platform_distribution)
        """
        sentiment_json = (
            json.dumps(sentiment_distribution, ensure_ascii=False) if sentiment_distribution else None
        )
        keywords_json = json.dumps(keywords, ensure_ascii=False) if keywords else None
        platform_json = (
            json.dumps(platform_distribution, ensure_ascii=False) if platform_distribution else None
        )
        return execute_update(
            sql,
            (
                event_id, snapshot_date, content_count_delta, comment_count_delta,
                content_count_total, avg_sentiment_score, sentiment_json,
                hot_score, interaction_count, keywords_json, platform_json,
            ),
        )

    @staticmethod
    def get_by_event(event_id: int, limit: int = 30) -> List[Dict[str, Any]]:
        """获取话题的演化快照"""
        sql = """
            SELECT * FROM topic_evolution
            WHERE event_id = %s
            ORDER BY snapshot_date DESC
            LIMIT %s
        """
        return execute_query(sql, (event_id, limit))

    @staticmethod
    def get_latest_snapshot_date() -> Optional[date]:
        """获取最新快照日期"""
        sql = "SELECT MAX(snapshot_date) as latest FROM topic_evolution"
        result = execute_query(sql, fetch_one=True)
        return result["latest"] if result else None


class TopicMergeRepo:
    """话题合并记录仓库 - 操作 topic_merge_log 表"""

    @staticmethod
    def insert(
        source_event_id: int,
        target_event_id: int,
        similarity_score: float,
        merge_reason: str,
        source_content_count: int = 0,
        source_keywords: Optional[List[Dict]] = None,
    ) -> int:
        """记录话题合并"""
        sql = """
            INSERT INTO topic_merge_log
                (source_event_id, target_event_id, similarity_score,
                 merge_reason, source_content_count, source_keywords)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        keywords_json = json.dumps(source_keywords, ensure_ascii=False) if source_keywords else None
        return execute_insert(
            sql,
            (source_event_id, target_event_id, similarity_score,
             merge_reason, source_content_count, keywords_json),
        )

    @staticmethod
    def get_by_target(target_event_id: int) -> List[Dict[str, Any]]:
        """获取合并到指定话题的合并记录"""
        sql = """
            SELECT * FROM topic_merge_log
            WHERE target_event_id = %s
            ORDER BY merged_at DESC
        """
        return execute_query(sql, (target_event_id,))


class TopicContentRepo:
    """话题内容关联仓库 - 操作 unified_content + processed_content"""

    @staticmethod
    def get_unclustered(limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取未聚类的内容

        JOIN processed_content 获取清洗后文本和关键词

        Returns:
            未聚类内容列表
        """
        sql = """
            SELECT
                uc.id AS unified_id,
                uc.platform,
                uc.title,
                uc.content_id,
                pc.content_cleaned,
                pc.title_cleaned,
                pc.keywords,
                uc.liked_count,
                uc.comment_count,
                uc.share_count,
                uc.original_created_at
            FROM unified_content uc
            JOIN processed_content pc ON pc.unified_id = uc.id
            WHERE uc.topic_id IS NULL
              AND pc.process_status = 'completed'
              AND pc.content_cleaned IS NOT NULL
              AND pc.content_cleaned != ''
            ORDER BY uc.id
            LIMIT %s
        """
        return execute_query(sql, (limit,))

    @staticmethod
    def assign_topic(
        unified_id: int,
        topic_id: int,
        similarity: float,
    ) -> int:
        """为单条内容分配话题"""
        sql = """
            UPDATE unified_content
            SET topic_id = %s,
                topic_similarity = %s,
                topic_assigned_at = NOW()
            WHERE id = %s
        """
        return execute_update(sql, (topic_id, similarity, unified_id))

    @staticmethod
    def batch_assign_topic(assignments: List[Dict[str, Any]]) -> int:
        """
        批量分配话题

        Args:
            assignments: 列表，每个元素包含 unified_id, topic_id, similarity
        """
        if not assignments:
            return 0

        sql = """
            UPDATE unified_content
            SET topic_id = %s,
                topic_similarity = %s,
                topic_assigned_at = NOW()
            WHERE id = %s
        """
        count = 0
        with get_connection() as conn:
            with conn.cursor() as cursor:
                for a in assignments:
                    cursor.execute(sql, (a["topic_id"], a["similarity"], a["unified_id"]))
                    count += cursor.rowcount
        return count

    @staticmethod
    def reassign_merged_topic(source_topic_id: int, target_topic_id: int) -> int:
        """将已合并话题的内容重新分配到目标话题"""
        sql = """
            UPDATE unified_content
            SET topic_id = %s
            WHERE topic_id = %s
        """
        return execute_update(sql, (target_topic_id, source_topic_id))

    @staticmethod
    def clear_all_topic_assignments() -> int:
        """清除所有话题分配 (全量重聚类前调用)"""
        sql = """
            UPDATE unified_content
            SET topic_id = NULL,
                topic_similarity = NULL,
                topic_assigned_at = NULL
            WHERE topic_id IS NOT NULL
        """
        return execute_update(sql)

    @staticmethod
    def count_unclustered() -> int:
        """统计未聚类的内容数量"""
        sql = """
            SELECT COUNT(*) as cnt
            FROM unified_content uc
            JOIN processed_content pc ON pc.unified_id = uc.id
            WHERE uc.topic_id IS NULL
              AND pc.process_status = 'completed'
              AND pc.content_cleaned IS NOT NULL
              AND pc.content_cleaned != ''
        """
        result = execute_query(sql, fetch_one=True)
        return result["cnt"] if result else 0

    @staticmethod
    def count_clustered() -> int:
        """统计已聚类的内容数量"""
        sql = "SELECT COUNT(*) as cnt FROM unified_content WHERE topic_id IS NOT NULL"
        result = execute_query(sql, fetch_one=True)
        return result["cnt"] if result else 0

    @staticmethod
    def get_topic_content_stats(topic_id: int) -> Dict[str, Any]:
        """
        获取话题内容聚合统计

        返回: content_count, avg_sentiment, platform分布, interaction总量
        """
        sql = """
            SELECT
                COUNT(*) as content_count,
                AVG(uc.sentiment_score) as avg_sentiment,
                SUM(CASE WHEN uc.sentiment = 'positive' THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN uc.sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
                SUM(CASE WHEN uc.sentiment = 'negative' THEN 1 ELSE 0 END) as negative_count,
                SUM(COALESCE(uc.liked_count, 0)) as total_likes,
                SUM(COALESCE(uc.comment_count, 0)) as total_comments,
                SUM(COALESCE(uc.share_count, 0)) as total_shares,
                MIN(uc.original_created_at) as first_content_at,
                MAX(uc.original_created_at) as last_content_at
            FROM unified_content uc
            WHERE uc.topic_id = %s
        """
        result = execute_query(sql, (topic_id,), fetch_one=True)

        # 平台分布
        platform_sql = """
            SELECT platform, COUNT(*) as cnt
            FROM unified_content
            WHERE topic_id = %s
            GROUP BY platform
        """
        platform_rows = execute_query(platform_sql, (topic_id,))
        platform_dist = {row["platform"]: row["cnt"] for row in platform_rows} if platform_rows else {}

        # 主要情绪标签
        emotion_sql = """
            SELECT emotion_tags
            FROM unified_content
            WHERE topic_id = %s AND emotion_tags IS NOT NULL
        """
        emotion_rows = execute_query(emotion_sql, (topic_id,))
        emotion_counts: Dict[str, int] = {}
        for row in emotion_rows or []:
            try:
                tags = json.loads(row["emotion_tags"]) if isinstance(row["emotion_tags"], str) else row["emotion_tags"]
                if isinstance(tags, list):
                    for tag in tags:
                        emotion_counts[tag] = emotion_counts.get(tag, 0) + 1
            except (json.JSONDecodeError, TypeError):
                pass
        top_emotions = sorted(emotion_counts.items(), key=lambda x: -x[1])[:3]
        dominant_emotions = ",".join(e[0] for e in top_emotions) if top_emotions else None

        if result:
            result["platform_distribution"] = platform_dist
            result["dominant_emotions"] = dominant_emotions
            result["total_interaction"] = (
                (result.get("total_likes") or 0)
                + (result.get("total_comments") or 0)
                + (result.get("total_shares") or 0)
            )
        return result or {}

    @staticmethod
    def get_content_for_topic(topic_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """获取话题下的内容 (用于 LLM 命名)"""
        sql = """
            SELECT
                uc.id AS unified_id,
                uc.title,
                uc.platform,
                pc.content_cleaned,
                pc.title_cleaned,
                pc.keywords,
                uc.sentiment,
                uc.sentiment_score
            FROM unified_content uc
            JOIN processed_content pc ON pc.unified_id = uc.id
            WHERE uc.topic_id = %s
            ORDER BY uc.topic_similarity DESC
            LIMIT %s
        """
        return execute_query(sql, (topic_id, limit))

    @staticmethod
    def get_daily_stats(topic_id: int, target_date: date) -> Dict[str, Any]:
        """获取话题某日的增量统计"""
        sql = """
            SELECT
                COUNT(*) as content_count_delta,
                AVG(uc.sentiment_score) as avg_sentiment,
                SUM(COALESCE(uc.liked_count, 0) + COALESCE(uc.comment_count, 0)
                    + COALESCE(uc.share_count, 0)) as interaction_count
            FROM unified_content uc
            WHERE uc.topic_id = %s
              AND DATE(uc.topic_assigned_at) = %s
        """
        result = execute_query(sql, (topic_id, target_date), fetch_one=True)
        return result or {"content_count_delta": 0, "avg_sentiment": None, "interaction_count": 0}
