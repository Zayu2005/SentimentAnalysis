# -*- coding: utf-8 -*-
"""
增量聚类引擎

Single-Pass 增量聚类: BERT 嵌入 → Faiss 搜索 → 阈值匹配/新建话题
"""

import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
from tqdm import tqdm

from ..config import get_settings
from ..database import TopicEventRepo, TopicContentRepo
from ..utils import get_logger
from .embedder import BertEmbedder
from .index import FaissIndex

logger = get_logger("TopicCluster.engine")


@dataclass
class ClusterResult:
    """聚类结果统计"""
    total_processed: int = 0
    assigned_to_existing: int = 0
    new_topics_created: int = 0
    errors: int = 0


class ClusterEngine:
    """增量聚类主引擎"""

    def __init__(
        self,
        embedder: Optional[BertEmbedder] = None,
        similarity_threshold: Optional[float] = None,
    ):
        settings = get_settings()
        self.embedder = embedder or BertEmbedder()
        self.similarity_threshold = similarity_threshold or settings.clustering.similarity_threshold
        self.embedding_model = settings.clustering.embedding_model

        self.index: Optional[FaissIndex] = None
        # 缓存: topic_id -> content_count (用于 running avg)
        self._topic_counts: Dict[int, int] = {}
        # 缓存: topic_id -> centroid (用于 running avg)
        self._topic_centroids: Dict[int, np.ndarray] = {}

    def _load_index(self):
        """从数据库加载活跃话题质心到 Faiss 索引"""
        settings = get_settings()
        self.index = FaissIndex(dim=settings.clustering.embedding_dim)

        topics = TopicEventRepo.get_active_topics()
        self.index.load_from_topics(topics)

        # 缓存计数和质心
        self._topic_counts.clear()
        self._topic_centroids.clear()
        for topic in topics:
            blob = topic.get("centroid_embedding")
            if blob is not None:
                centroid = np.frombuffer(blob, dtype=np.float32).copy()
                if centroid.shape[0] == settings.clustering.embedding_dim:
                    self._topic_counts[topic["id"]] = topic.get("content_count", 1)
                    self._topic_centroids[topic["id"]] = centroid

        logger.info(f"加载 {self.index.size} 个活跃话题到索引")

    def _update_centroid_running_avg(self, topic_id: int, new_embedding: np.ndarray):
        """
        增量更新话题质心 (running average)

        new_centroid = (old_centroid * n + new_embedding) / (n + 1)
        然后 L2 归一化
        """
        old_centroid = self._topic_centroids[topic_id]
        n = self._topic_counts[topic_id]

        new_centroid = (old_centroid * n + new_embedding) / (n + 1)

        # L2 归一化
        norm = np.linalg.norm(new_centroid)
        if norm > 1e-12:
            new_centroid = new_centroid / norm

        self._topic_centroids[topic_id] = new_centroid
        self._topic_counts[topic_id] = n + 1

        # 更新 Faiss 索引
        self.index.update_centroid(topic_id, new_centroid)

    def _create_new_topic(self, embedding: np.ndarray, content: Dict[str, Any]) -> int:
        """
        创建新话题

        使用关键词拼接为占位名，后续由 LLM 生成正式名

        Returns:
            新话题ID
        """
        # 提取关键词作为占位名
        keywords_raw = content.get("keywords")
        keyword_list = []
        if keywords_raw:
            try:
                if isinstance(keywords_raw, str):
                    keyword_list = json.loads(keywords_raw)
                elif isinstance(keywords_raw, list):
                    keyword_list = keywords_raw
            except (json.JSONDecodeError, TypeError):
                pass

        if keyword_list:
            # 取前5个关键词拼接
            kw_names = []
            for kw in keyword_list[:5]:
                if isinstance(kw, dict):
                    kw_names.append(kw.get("word", str(kw)))
                else:
                    kw_names.append(str(kw))
            placeholder_name = "/".join(kw_names)
        else:
            # 使用标题截取
            title = content.get("title_cleaned") or content.get("title") or ""
            placeholder_name = title[:50] if title else f"话题-未命名"

        # 构建关键词 JSON
        keywords_json = None
        if keyword_list:
            structured = []
            for kw in keyword_list[:10]:
                if isinstance(kw, dict):
                    structured.append(kw)
                else:
                    structured.append({"word": str(kw), "weight": 1.0})
            keywords_json = structured

        topic_id = TopicEventRepo.insert(
            event_name=placeholder_name,
            centroid_embedding=embedding,
            keywords=keywords_json,
            similarity_threshold=self.similarity_threshold,
            embedding_model=self.embedding_model,
        )

        # 更新缓存
        self._topic_counts[topic_id] = 1
        self._topic_centroids[topic_id] = embedding.copy()
        self.index.add_topic(topic_id, embedding)

        return topic_id

    def run(
        self,
        batch_size: Optional[int] = None,
        max_items: Optional[int] = None,
        dry_run: bool = False,
    ) -> ClusterResult:
        """
        运行增量聚类

        Args:
            batch_size: 每次获取的批次大小
            max_items: 最大处理数量
            dry_run: 试运行 (不写入数据库)

        Returns:
            聚类结果统计
        """
        settings = get_settings()
        batch_size = batch_size or settings.clustering.batch_size
        result = ClusterResult()

        # 1. 加载索引
        self._load_index()

        # 统计待聚类
        unclustered_count = TopicContentRepo.count_unclustered()
        logger.info(f"待聚类内容: {unclustered_count} 条")

        if unclustered_count == 0:
            logger.info("没有待聚类的内容")
            return result

        if max_items:
            total_to_process = min(unclustered_count, max_items)
        else:
            total_to_process = unclustered_count

        processed = 0
        pbar = tqdm(total=total_to_process, desc="聚类中")

        while True:
            if max_items and processed >= max_items:
                break

            # 2. 获取一批未聚类内容
            current_batch = min(batch_size, total_to_process - processed) if max_items else batch_size
            contents = TopicContentRepo.get_unclustered(limit=current_batch)
            if not contents:
                break

            # 3. 提取文本并计算 BERT 嵌入
            texts = []
            for c in contents:
                text = c.get("content_cleaned") or c.get("title_cleaned") or ""
                texts.append(text[:500])

            try:
                embeddings = self.embedder.embed_texts(texts, batch_size=batch_size)
            except Exception as e:
                logger.error(f"嵌入计算失败: {e}")
                result.errors += len(contents)
                break

            # 4. Single-Pass 聚类
            assignments = []  # 待批量写入的分配

            for i, (content, embedding) in enumerate(zip(contents, embeddings)):
                try:
                    # 搜索最近话题
                    matches = self.index.search(embedding, k=1)

                    if matches and matches[0][1] >= self.similarity_threshold:
                        # 归入现有话题
                        topic_id, similarity = matches[0]

                        if not dry_run:
                            self._update_centroid_running_avg(topic_id, embedding)

                        assignments.append({
                            "unified_id": content["unified_id"],
                            "topic_id": topic_id,
                            "similarity": float(similarity),
                        })
                        result.assigned_to_existing += 1
                    else:
                        # 创建新话题
                        if not dry_run:
                            topic_id = self._create_new_topic(embedding, content)
                            assignments.append({
                                "unified_id": content["unified_id"],
                                "topic_id": topic_id,
                                "similarity": 1.0,
                            })
                        result.new_topics_created += 1

                    result.total_processed += 1

                except Exception as e:
                    logger.error(f"处理内容 {content.get('unified_id')} 失败: {e}")
                    result.errors += 1

            # 5. 批量写入 DB
            if not dry_run and assignments:
                TopicContentRepo.batch_assign_topic(assignments)

                # 批量更新话题质心到 DB
                for topic_id, centroid in self._topic_centroids.items():
                    if topic_id in self._topic_counts:
                        TopicEventRepo.update_centroid(
                            topic_id, centroid, self._topic_counts[topic_id]
                        )

            processed += len(contents)
            pbar.update(len(contents))

            if dry_run and processed >= min(batch_size, total_to_process):
                # 试运行只处理一个批次
                break

        pbar.close()

        logger.info(
            f"聚类完成: 处理 {result.total_processed}, "
            f"归入现有 {result.assigned_to_existing}, "
            f"新建话题 {result.new_topics_created}, "
            f"错误 {result.errors}"
        )

        return result
