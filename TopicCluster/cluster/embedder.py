# -*- coding: utf-8 -*-
"""
BERT 嵌入提取器

使用预训练 BERT 模型提取 [CLS] 向量作为文本嵌入
"""

import os
import numpy as np
from typing import List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from ..config import get_settings
from ..utils import get_logger

logger = get_logger("TopicCluster.embedder")


class BertEmbedder:
    """BERT [CLS] 嵌入提取器"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: Optional[int] = None,
        device: Optional[str] = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.clustering.embedding_model
        self.max_length = max_length or settings.clustering.max_length
        self.dim = settings.clustering.embedding_dim

        # 设备选择
        device_cfg = device or settings.clustering.device
        if device_cfg == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_cfg)

        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """延迟加载模型"""
        if self.model is not None:
            return

        # 设置 HuggingFace 镜像
        if not os.getenv("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        logger.info(f"加载嵌入模型: {self.model_name} -> {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info("嵌入模型加载完成")

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        批量提取 [CLS] 嵌入

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            normalize: 是否 L2 归一化

        Returns:
            嵌入矩阵 (N, dim)
        """
        self._load_model()

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # 取 [CLS] token 的输出
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

        embeddings = np.vstack(all_embeddings).astype(np.float32)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms

        return embeddings

    def embed_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        单条文本嵌入提取

        Args:
            text: 文本
            normalize: 是否 L2 归一化

        Returns:
            嵌入向量 (dim,)
        """
        return self.embed_texts([text], normalize=normalize)[0]
