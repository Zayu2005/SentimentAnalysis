# -*- coding: utf-8 -*-
"""
话题分类模块

提供基于 LLM 的多维话题分类服务
支持本地模型和远程 API (DeepSeek/OpenAI)
"""

from .llm_classifier import LLMTopicClassifier, ClassifierFactory

__all__ = ["LLMTopicClassifier", "ClassifierFactory"]
