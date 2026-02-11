# -*- coding: utf-8 -*-
"""
话题维护器

处理话题合并、生命周期管理、统计更新、演化快照
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Dict, Any, List
import json

from ..config import get_settings
from ..database import TopicEventRepo, TopicEvolutionRepo, TopicMergeRepo, TopicContentRepo
from ..utils import get_logger
from .index import FaissIndex

logger = get_logger("TopicCluster.maintainer")


@dataclass
class MergeResult:
    """合并结果统计"""
    pairs_checked: int = 0
    merged_count: int = 0
    content_reassigned: int = 0


class TopicMaintainer:
    """话题维护器"""

    def __init__(self, merge_threshold: Optional[float] = None):
        settings = get_settings()
        self.merge_threshold = merge_threshold or settings.clustering.merge_threshold
        self.inactive_days = settings.clustering.inactive_days
        self.ended_days = settings.clustering.ended_days

    def merge_topics(self, dry_run: bool = False) -> MergeResult:
        """
        合并相似话题

        遍历所有话题对，相似度超过 merge_threshold 时合并 (小话题 → 大话题)

        Args:
            dry_run: 试运行

        Returns:
            合并结果统计
        """
        settings = get_settings()
        result = MergeResult()

        # 加载所有活跃话题到索引
        index = FaissIndex(dim=settings.clustering.embedding_dim)
        topics = TopicEventRepo.get_active_topics()
        index.load_from_topics(topics)

        # 建立 topic_id -> topic 的映射
        topic_map = {t["id"]: t for t in topics}

        # 计算所有话题对相似度
        pairs = index.get_all_pairs_similarity()
        result.pairs_checked = len(pairs)

        # 已合并的话题集合
        merged_ids = set()

        for topic_a, topic_b, similarity in pairs:
            if similarity < self.merge_threshold:
                break  # 已按降序排列

            if topic_a in merged_ids or topic_b in merged_ids:
                continue

            # 小话题合并到大话题
            count_a = topic_map.get(topic_a, {}).get("content_count", 0)
            count_b = topic_map.get(topic_b, {}).get("content_count", 0)

            if count_a >= count_b:
                target_id, source_id = topic_a, topic_b
                source_count = count_b
            else:
                target_id, source_id = topic_b, topic_a
                source_count = count_a

            source_topic = topic_map.get(source_id, {})
            source_keywords = source_topic.get("keywords")
            if isinstance(source_keywords, str):
                try:
                    source_keywords = json.loads(source_keywords)
                except (json.JSONDecodeError, TypeError):
                    source_keywords = None

            if dry_run:
                source_name = source_topic.get("event_name", "?")
                target_name = topic_map.get(target_id, {}).get("event_name", "?")
                logger.info(
                    f"[试运行] 合并: {source_name}({source_id}) -> "
                    f"{target_name}({target_id}), 相似度={similarity:.4f}"
                )
            else:
                # 记录合并日志
                TopicMergeRepo.insert(
                    source_event_id=source_id,
                    target_event_id=target_id,
                    similarity_score=similarity,
                    merge_reason=f"质心相似度={similarity:.4f}",
                    source_content_count=source_count,
                    source_keywords=source_keywords,
                )

                # 内容重新分配
                reassigned = TopicContentRepo.reassign_merged_topic(source_id, target_id)
                result.content_reassigned += reassigned

                # 更新源话题状态
                TopicEventRepo.update_status(source_id, "merged", merged_into_id=target_id)

                logger.info(
                    f"合并完成: {source_id} -> {target_id}, "
                    f"相似度={similarity:.4f}, 重新分配 {reassigned} 条内容"
                )

            merged_ids.add(source_id)
            result.merged_count += 1

        logger.info(
            f"合并检查完成: 检查 {result.pairs_checked} 对, "
            f"合并 {result.merged_count} 个话题"
        )

        return result

    def update_lifecycle(self):
        """
        更新话题生命周期状态

        - emerging → active: content_count >= 10
        - active/emerging → declining: 超过 inactive_days 无新内容
        - declining → ended: 超过 ended_days 无新内容
        """
        topics = TopicEventRepo.get_active_topics()
        now = datetime.now()
        updated = 0

        for topic in topics:
            status = topic.get("status")
            content_count = topic.get("content_count", 0)
            topic_id = topic["id"]

            # 计算不活跃天数
            stats = TopicContentRepo.get_topic_content_stats(topic_id)
            last_content_at = stats.get("last_content_at")

            if last_content_at is None:
                continue

            if isinstance(last_content_at, str):
                last_content_at = datetime.fromisoformat(last_content_at)

            days_inactive = (now - last_content_at).days

            new_status = None

            if status == "emerging" and content_count >= 10:
                new_status = "active"
            elif status in ("emerging", "active") and days_inactive >= self.inactive_days:
                new_status = "declining"
            elif status == "declining" and days_inactive >= self.ended_days:
                new_status = "ended"

            if new_status:
                TopicEventRepo.update_status(topic_id, new_status)
                updated += 1
                logger.info(f"话题 {topic_id} 状态: {status} -> {new_status}")

        logger.info(f"生命周期更新: {updated} 个话题状态变更")

    def update_all_stats(self):
        """更新所有非合并话题的聚合统计"""
        topics = TopicEventRepo.get_non_merged_topics()
        updated = 0

        for topic in topics:
            topic_id = topic["id"]
            stats = TopicContentRepo.get_topic_content_stats(topic_id)

            if not stats or stats.get("content_count", 0) == 0:
                continue

            # 情感统计
            avg_sentiment = stats.get("avg_sentiment")
            sentiment_dist = {
                "positive": int(stats.get("positive_count", 0) or 0),
                "neutral": int(stats.get("neutral_count", 0) or 0),
                "negative": int(stats.get("negative_count", 0) or 0),
            }

            # 主导情感
            dominant = max(sentiment_dist, key=sentiment_dist.get)
            total = sum(sentiment_dist.values())
            if total > 0 and sentiment_dist[dominant] / total < 0.4:
                dominant = "mixed"

            TopicEventRepo.update_sentiment_stats(
                topic_id=topic_id,
                avg_score=float(avg_sentiment) if avg_sentiment else 0.0,
                sentiment_dist=sentiment_dist,
                dominant_sentiment=dominant,
                dominant_emotions=stats.get("dominant_emotions"),
            )

            # 热度和平台
            interaction = int(stats.get("total_interaction", 0) or 0)
            content_count = int(stats.get("content_count", 0) or 0)

            heat_level = self._calculate_heat_level(content_count, interaction)

            TopicEventRepo.update_heat_and_platform(
                topic_id=topic_id,
                heat_level=heat_level,
                platform_distribution=stats.get("platform_distribution", {}),
                comment_count=stats.get("total_comments", 0),
            )

            updated += 1

        logger.info(f"统计更新: {updated} 个话题")

    def generate_evolution(self, snapshot_date: Optional[date] = None):
        """
        生成话题演化快照

        Args:
            snapshot_date: 快照日期 (默认今天)
        """
        snapshot_date = snapshot_date or date.today()
        topics = TopicEventRepo.get_non_merged_topics()
        generated = 0

        for topic in topics:
            topic_id = topic["id"]

            # 当日增量
            daily = TopicContentRepo.get_daily_stats(topic_id, snapshot_date)
            delta = int(daily.get("content_count_delta", 0) or 0)
            interaction = int(daily.get("interaction_count", 0) or 0)

            # 总量
            stats = TopicContentRepo.get_topic_content_stats(topic_id)
            total_count = int(stats.get("content_count", 0) or 0)

            # 热度分数
            hot_score = self._calculate_hot_score(delta, interaction)

            # 情感
            avg_sentiment = daily.get("avg_sentiment")
            sentiment_dist = {
                "positive": int(stats.get("positive_count", 0) or 0),
                "neutral": int(stats.get("neutral_count", 0) or 0),
                "negative": int(stats.get("negative_count", 0) or 0),
            }

            TopicEvolutionRepo.insert(
                event_id=topic_id,
                snapshot_date=snapshot_date,
                content_count_delta=delta,
                content_count_total=total_count,
                avg_sentiment_score=float(avg_sentiment) if avg_sentiment else None,
                sentiment_distribution=sentiment_dist,
                hot_score=hot_score,
                interaction_count=interaction,
                platform_distribution=stats.get("platform_distribution"),
            )
            generated += 1

        logger.info(f"演化快照: {snapshot_date}, {generated} 个话题")

    @staticmethod
    def _calculate_hot_score(delta: int, interaction: int) -> float:
        """
        计算热度分数

        hot_score = delta * 10 + interaction * 0.01
        """
        return delta * 10.0 + interaction * 0.01

    @staticmethod
    def _calculate_heat_level(content_count: int, interaction: int) -> str:
        """
        计算热度等级

        基于内容数量和互动量综合判断
        """
        score = content_count * 5 + interaction * 0.01

        if score >= 500:
            return "critical"
        elif score >= 100:
            return "high"
        elif score >= 20:
            return "medium"
        return "low"
