# -*- coding: utf-8 -*-
"""
话题命名器

使用 Qwen LLM 为话题生成描述性名称和摘要
"""

import os
import re
import json
from typing import Optional, Dict, Any, List

from ..config import get_settings
from ..database import TopicEventRepo, TopicContentRepo
from ..utils import get_logger

logger = get_logger("TopicCluster.namer")

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的舆情监测分析师。请根据用户提供的多篇社交媒体内容，识别并总结其核心舆情话题。

请只输出 JSON 格式结果，不要有其他内容。JSON 格式如下:
{
    "event_name": "舆情话题名称(10-20字，如'小米SU7起火事故引发安全争议')",
    "event_description": "舆情描述(80-150字，包含：事件概述、涉及主体、公众关注焦点、舆论情感倾向)",
    "keywords": [{"word": "关键词", "weight": 0.9}]
}

要求:
- event_name 采用"主体+事件+影响"结构，简洁概括舆情焦点
- event_description 需涵盖：①事件背景 ②涉及品牌/人物/机构 ③公众态度与情绪 ④潜在风险或影响
- keywords 提取3-5个舆情核心关键词，weight反映其在舆情中的重要程度(0-1)
- 如果内容涉及负面舆情，需在描述中明确指出风险点"""


class TopicNamer:
    """话题命名器 - 使用 Qwen 生成话题描述"""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.llm.model_name
        self.max_new_tokens = settings.llm.max_new_tokens
        self.temperature = settings.llm.temperature

        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """延迟加载 Qwen 模型"""
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not os.getenv("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        logger.info(f"加载 LLM 模型: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        logger.info("LLM 模型加载完成")

    def generate_name(self, topic_id: int) -> Dict[str, Any]:
        """
        为话题生成名称和描述

        Args:
            topic_id: 话题ID

        Returns:
            {event_name, event_description, keywords}
        """
        import torch

        self._load_model()

        # 获取话题代表性内容
        contents = TopicContentRepo.get_content_for_topic(topic_id, limit=10)
        if not contents:
            logger.warning(f"话题 {topic_id} 没有关联内容")
            return {}

        # 构建内容摘要
        content_texts = []
        for i, c in enumerate(contents[:10], 1):
            title = c.get("title_cleaned") or c.get("title") or ""
            body = c.get("content_cleaned") or ""
            sentiment = c.get("sentiment") or "未知"
            platform = c.get("platform", "?")
            text = f"{i}. [{platform}] [情感:{sentiment}] {title}"
            if body:
                text += f"\n   {body[:200]}"
            content_texts.append(text)

        user_content = (
            f"以下是舆情监测系统聚类到同一话题的 {len(content_texts)} 篇社交媒体内容，"
            f"请从舆情分析角度总结该话题:\n\n" + "\n\n".join(content_texts)
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return self._parse_response(response)

    def describe_topics(
        self,
        topic_ids: Optional[List[int]] = None,
        only_unnamed: bool = True,
        include_ended: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """
        批量为话题生成/更新描述

        Args:
            topic_ids: 指定话题ID列表 (None=全部)
            only_unnamed: 仅处理未命名话题
            include_ended: 是否包含已结束话题
            dry_run: 试运行

        Returns:
            {total, success, error}
        """
        stats = {"total": 0, "success": 0, "error": 0}

        # 获取需要命名的话题
        if include_ended:
            topics = TopicEventRepo.get_non_merged_topics()
        else:
            topics = TopicEventRepo.get_active_topics()

        if topic_ids:
            topics = [t for t in topics if t["id"] in topic_ids]

        if only_unnamed:
            topics = [
                t for t in topics
                if t.get("event_name", "").startswith("话题-") or "/" in t.get("event_name", "")
            ]

        stats["total"] = len(topics)
        logger.info(f"待命名话题: {len(topics)} 个")

        for topic in topics:
            topic_id = topic["id"]
            try:
                result = self.generate_name(topic_id)

                if not result or not result.get("event_name"):
                    logger.warning(f"话题 {topic_id} 命名生成为空")
                    stats["error"] += 1
                    continue

                if dry_run:
                    logger.info(
                        f"[试运行] 话题 {topic_id}: "
                        f"{topic.get('event_name')} -> {result['event_name']}"
                    )
                    if result.get("event_description"):
                        logger.info(f"  描述: {result['event_description'][:80]}...")
                else:
                    TopicEventRepo.update_description(
                        topic_id=topic_id,
                        event_name=result["event_name"],
                        event_description=result.get("event_description"),
                        keywords=result.get("keywords"),
                    )
                    logger.info(f"话题 {topic_id} 命名: {result['event_name']}")

                stats["success"] += 1

            except Exception as e:
                logger.error(f"话题 {topic_id} 命名失败: {e}")
                stats["error"] += 1

        logger.info(
            f"命名完成: 总计 {stats['total']}, "
            f"成功 {stats['success']}, 失败 {stats['error']}"
        )
        return stats

    @staticmethod
    def _parse_response(response: str) -> Dict[str, Any]:
        """
        解析 LLM 响应中的 JSON

        Args:
            response: LLM 原始输出

        Returns:
            解析后的字典
        """
        # 尝试提取 JSON
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)

        if json_match:
            try:
                result = json.loads(json_match.group())

                # 验证必填字段
                event_name = result.get("event_name", "").strip()
                if not event_name:
                    return {}

                # 规范化关键词
                keywords = result.get("keywords", [])
                if isinstance(keywords, list):
                    normalized = []
                    for kw in keywords:
                        if isinstance(kw, dict) and "word" in kw:
                            weight = float(kw.get("weight", 1.0))
                            weight = max(0.0, min(1.0, weight))
                            normalized.append({"word": kw["word"], "weight": weight})
                        elif isinstance(kw, str):
                            normalized.append({"word": kw, "weight": 1.0})
                    keywords = normalized

                return {
                    "event_name": event_name[:200],
                    "event_description": result.get("event_description", ""),
                    "keywords": keywords,
                }
            except (json.JSONDecodeError, ValueError):
                pass

        logger.warning(f"无法解析 LLM 响应: {response[:200]}")
        return {}
