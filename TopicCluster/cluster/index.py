# -*- coding: utf-8 -*-
"""
Faiss 向量索引

管理话题质心的向量检索，使用 IndexFlatIP (内积 = L2归一化后的余弦相似度)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from ..utils import get_logger

logger = get_logger("TopicCluster.index")


class FaissIndex:
    """Faiss 向量索引 - 管理话题质心"""

    def __init__(self, dim: int = 768):
        import faiss

        self.dim = dim
        self._centroids: Dict[int, np.ndarray] = {}  # topic_id -> centroid (ground truth)
        self._id_list: List[int] = []  # Faiss 索引顺序对应的 topic_id
        self._index = faiss.IndexFlatIP(dim)  # 内积索引

    def add_topic(self, topic_id: int, centroid: np.ndarray):
        """
        添加话题质心

        Args:
            topic_id: 话题ID
            centroid: L2 归一化的质心向量
        """
        self._centroids[topic_id] = centroid.astype(np.float32)
        self._rebuild()

    def search(self, query: np.ndarray, k: int = 1) -> List[Tuple[int, float]]:
        """
        搜索最近的话题质心

        Args:
            query: L2 归一化的查询向量
            k: 返回的最近邻数

        Returns:
            [(topic_id, similarity_score), ...]
        """
        if self._index.ntotal == 0:
            return []

        query = query.astype(np.float32).reshape(1, -1)
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._id_list):
                results.append((self._id_list[idx], float(score)))
        return results

    def update_centroid(self, topic_id: int, new_centroid: np.ndarray):
        """
        更新话题质心

        Args:
            topic_id: 话题ID
            new_centroid: 新的 L2 归一化质心向量
        """
        if topic_id not in self._centroids:
            self.add_topic(topic_id, new_centroid)
            return
        self._centroids[topic_id] = new_centroid.astype(np.float32)
        self._rebuild()

    def remove_topic(self, topic_id: int):
        """移除话题"""
        if topic_id in self._centroids:
            del self._centroids[topic_id]
            self._rebuild()

    def load_from_topics(self, topics: List[Dict]):
        """
        从数据库记录加载话题质心

        Args:
            topics: 话题列表，每个包含 id, centroid_embedding (BLOB)
        """
        self._centroids.clear()
        loaded = 0
        for topic in topics:
            blob = topic.get("centroid_embedding")
            if blob is None:
                continue
            centroid = np.frombuffer(blob, dtype=np.float32).copy()
            if centroid.shape[0] == self.dim:
                self._centroids[topic["id"]] = centroid
                loaded += 1
            else:
                logger.warning(f"话题 {topic['id']} 质心维度不匹配: {centroid.shape[0]} != {self.dim}")
        self._rebuild()
        logger.info(f"加载了 {loaded} 个话题质心到索引")

    def get_all_pairs_similarity(self) -> List[Tuple[int, int, float]]:
        """
        计算所有话题对的相似度 (用于合并检查)

        Returns:
            [(topic_id_a, topic_id_b, similarity), ...] 按相似度降序
        """
        if len(self._centroids) < 2:
            return []

        ids = list(self._centroids.keys())
        vectors = np.array([self._centroids[tid] for tid in ids], dtype=np.float32)

        # 内积矩阵 = 余弦相似度矩阵 (已 L2 归一化)
        sim_matrix = vectors @ vectors.T

        pairs = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ids[i], ids[j], float(sim_matrix[i, j])))

        pairs.sort(key=lambda x: -x[2])
        return pairs

    @property
    def size(self) -> int:
        """当前索引中的话题数"""
        return len(self._centroids)

    def _rebuild(self):
        """从 _centroids 重建 Faiss 索引"""
        import faiss

        self._index = faiss.IndexFlatIP(self.dim)
        self._id_list = list(self._centroids.keys())

        if self._id_list:
            vectors = np.array(
                [self._centroids[tid] for tid in self._id_list],
                dtype=np.float32,
            )
            self._index.add(vectors)
