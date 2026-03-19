# -*- coding: utf-8 -*-
"""
LLM 话题分类器

支持两种模式:
1. 本地模型: 使用 HuggingFace Transformers 加载 Qwen 等模型
2. 远程 API: 支持 DeepSeek / OpenAI / Claude 等 API
"""

import os
import re
import json
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

from ..config import get_settings
from ..database import TopicEventRepo, TopicContentRepo, TopicClassificationRepo
from ..utils import get_logger

logger = get_logger("TopicCluster.classifier")

SYSTEM_PROMPT = """你是一个话题分类专家。请分析以下舆情话题，为其分类。

## 分类维度

### 1. 一级分类 (必选)
从以下类别中选择最合适的一个:
- 社会事件: 突发事件、民生热点、公共安全、司法案件
- 娱乐明星: 影视综艺、明星绯闻、演唱会、作品发布
- 科技产品: 手机数码、软件更新、AI技术、发布会
- 游戏电竞: 游戏发布、赛事动态、直播热点
- 汽车: 新车发布、电动车、事故维权
- 美妆服饰: 护肤彩妆、时装周、产品测评
- 食品饮料: 食品安全、品牌动态、餐饮探店
- 金融财经: 股市动态、基金理财、经济政策
- 教育培训: 考试动态、教育政策、留学资讯
- 医疗健康: 疫情动态、医疗技术、药品上市
- 旅游出行: 景点推荐、出行安全、交通出行
- 房产家居: 房价动态、楼市政策、装修案例
- 体育运动: 赛事结果、运动员动态
- 法律舆情: 法律法规、案件判决、维权事件
- 国际关系: 外交动态、国际会议、贸易关系

### 2. 二级分类 (必选)
根据一级分类选择更细分的领域，如:
- 社会事件下: 突发事件/民生热点/公共安全/司法案件/企业舆情
- 科技产品下: 手机数码/软件更新/AI技术/智能汽车/消费电子
- 娱乐明星下: 影视综艺/明星绯闻/演唱会/作品发布/选秀偶像

### 3. 事件类型 (必选)
- product_issue: 产品问题(质量缺陷、召回、安全隐患)
- service_dispute: 服务纠纷(售后推诿、客服态度)
- price_dispute: 价格争议(涨价、降价、价格欺诈)
- safety_incident: 安全事件(数据泄露、信息安全)
- personnel_change: 人事变动(高管离职、裁员)
- policy_release: 政策发布(法规出台、监管加强)
- marketing_blunder: 营销翻车(代言争议、广告抄袭)
- celebrity_gossip: 明星八卦(绯闻、婚恋、言行)
- sports_event: 体育赛事(比赛结果、运动员表现)
- tech_breakthrough: 技术突破(新品发布、技术创新)
- public_opinion: 舆论事件(热搜话题、网络争议)
- other: 其他

### 4. 受众范围 (必选)
- local: 本地(城市级)
- regional: 区域(省份/地区级)
- national: 全国(全国范围)
- global: 全球(国际范围)

### 5. 时间敏感度 (必选)
- breaking: 突发(24小时内爆发的重大事件)
- trending: 热点(正在发酵的热门话题)
- normal: 普通(一般性话题)
- evergreen: 长尾(持续关注的长效话题)

### 6. 情感强度 (必选)
- mild: 轻度(平静讨论)
- moderate: 中度(有一定情绪但可控)
- intense: 激烈(情绪激动、争议较大)
- extreme: 极端(极度愤怒、恐慌或失控)

### 7. 争议程度 (必选)
- low: 低(共识性强)
- medium: 中(存在不同观点)
- high: 高(对立观点明显)
- extreme: 极端(严重对立、水火不容)

### 8. 风险等级 (必选)
- safe: 安全(普通话题，无风险)
- attention: 需关注(潜在风险，需监控)
- warning: 警告(明显风险，需处理)
- dangerous: 危险(高危风险，需立即处理)

## 输出要求
请以 JSON 格式输出完整的分类结果，不要有其他内容:
{
    "primary_category": "一级分类",
    "secondary_category": "二级分类",
    "event_type": "事件类型",
    "audience_scope": "受众范围",
    "time_sensitivity": "时间敏感度",
    "sentiment_intensity": "情感强度",
    "controversy_level": "争议程度",
    "risk_level": "风险等级",
    "industry_tags": ["行业标签1", "行业标签2"],
    "risk_keywords": ["风险关键词1", "风险关键词2"],
    "classification_confidence": 0.85,
    "classification_reason": "分类理由简述(30字内)"
}"""


class BaseClassifier(ABC):
    """分类器基类"""

    @abstractmethod
    def classify(self, topic_info: str) -> Dict[str, Any]:
        """执行分类"""
        pass

    @abstractmethod
    def name(self) -> str:
        """分类器名称"""
        pass


class LocalClassifier(BaseClassifier):
    """本地模型分类器 (HuggingFace Transformers)"""

    def __init__(self, model_name: str, max_new_tokens: int = 200, temperature: float = 0.3):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None

    def name(self) -> str:
        return f"local:{self.model_name}"

    def _load_model(self):
        """延迟加载模型"""
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not os.getenv("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        logger.info(f"加载本地模型: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        logger.info("本地模型加载完成")

    def classify(self, topic_info: str) -> Dict[str, Any]:
        """使用本地模型分类"""
        import torch

        self._load_model()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": topic_info},
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

    @staticmethod
    def _parse_response(response: str) -> Dict[str, Any]:
        """解析 LLM 响应中的 JSON"""
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)

        if json_match:
            try:
                result = json.loads(json_match.group())

                required_fields = ["primary_category", "event_type", "audience_scope",
                                   "time_sensitivity", "sentiment_intensity",
                                   "controversy_level", "risk_level"]
                for field in required_fields:
                    if field not in result:
                        logger.warning(f"缺少必填字段: {field}")
                        return {}

                return {
                    "primary_category": result.get("primary_category", ""),
                    "secondary_category": result.get("secondary_category", ""),
                    "tertiary_category": result.get("tertiary_category", ""),
                    "event_type": result.get("event_type", ""),
                    "audience_scope": result.get("audience_scope", "national"),
                    "time_sensitivity": result.get("time_sensitivity", "normal"),
                    "sentiment_intensity": result.get("sentiment_intensity", "moderate"),
                    "controversy_level": result.get("controversy_level", "low"),
                    "risk_level": result.get("risk_level", "safe"),
                    "industry_tags": result.get("industry_tags", []),
                    "risk_keywords": result.get("risk_keywords", []),
                    "classification_confidence": float(result.get("classification_confidence", 0.5)),
                    "classification_reason": result.get("classification_reason", ""),
                }
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"JSON 解析失败: {e}")

        logger.warning(f"无法解析 LLM 响应: {response[:200]}")
        return {}


class DeepSeekClassifier(BaseClassifier):
    """DeepSeek API 分类器"""

    def __init__(self, api_key: str, api_base: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", max_tokens: int = 500, temperature: float = 0.3):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def name(self) -> str:
        return f"deepseek:{self.model}"

    def classify(self, topic_info: str) -> Dict[str, Any]:
        """使用 DeepSeek API 分类"""
        import requests

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": topic_info},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"]
            return self._parse_response(content)
        except Exception as e:
            logger.error(f"DeepSeek API 调用失败: {e}")
            return {}

    @staticmethod
    def _parse_response(response: str) -> Dict[str, Any]:
        """解析 LLM 响应中的 JSON"""
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)

        if json_match:
            try:
                result = json.loads(json_match.group())

                required_fields = ["primary_category", "event_type", "audience_scope",
                                   "time_sensitivity", "sentiment_intensity",
                                   "controversy_level", "risk_level"]
                for field in required_fields:
                    if field not in result:
                        logger.warning(f"缺少必填字段: {field}")
                        return {}

                return {
                    "primary_category": result.get("primary_category", ""),
                    "secondary_category": result.get("secondary_category", ""),
                    "tertiary_category": result.get("tertiary_category", ""),
                    "event_type": result.get("event_type", ""),
                    "audience_scope": result.get("audience_scope", "national"),
                    "time_sensitivity": result.get("time_sensitivity", "normal"),
                    "sentiment_intensity": result.get("sentiment_intensity", "moderate"),
                    "controversy_level": result.get("controversy_level", "low"),
                    "risk_level": result.get("risk_level", "safe"),
                    "industry_tags": result.get("industry_tags", []),
                    "risk_keywords": result.get("risk_keywords", []),
                    "classification_confidence": float(result.get("classification_confidence", 0.5)),
                    "classification_reason": result.get("classification_reason", ""),
                }
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"JSON 解析失败: {e}")

        logger.warning(f"无法解析 LLM 响应: {response[:200]}")
        return {}


class OpenAIClassifier(BaseClassifier):
    """OpenAI API 分类器"""

    def __init__(self, api_key: str, api_base: str = "https://api.openai.com/v1",
                 model: str = "gpt-4o-mini", max_tokens: int = 500, temperature: float = 0.3):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def name(self) -> str:
        return f"openai:{self.model}"

    def classify(self, topic_info: str) -> Dict[str, Any]:
        """使用 OpenAI API 分类"""
        import requests

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": topic_info},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"]
            return self._parse_response(content)
        except Exception as e:
            logger.error(f"OpenAI API 调用失败: {e}")
            return {}

    @staticmethod
    def _parse_response(response: str) -> Dict[str, Any]:
        """解析 LLM 响应中的 JSON"""
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)

        if json_match:
            try:
                result = json.loads(json_match.group())

                required_fields = ["primary_category", "event_type", "audience_scope",
                                   "time_sensitivity", "sentiment_intensity",
                                   "controversy_level", "risk_level"]
                for field in required_fields:
                    if field not in result:
                        logger.warning(f"缺少必填字段: {field}")
                        return {}

                return {
                    "primary_category": result.get("primary_category", ""),
                    "secondary_category": result.get("secondary_category", ""),
                    "tertiary_category": result.get("tertiary_category", ""),
                    "event_type": result.get("event_type", ""),
                    "audience_scope": result.get("audience_scope", "national"),
                    "time_sensitivity": result.get("time_sensitivity", "normal"),
                    "sentiment_intensity": result.get("sentiment_intensity", "moderate"),
                    "controversy_level": result.get("controversy_level", "low"),
                    "risk_level": result.get("risk_level", "safe"),
                    "industry_tags": result.get("industry_tags", []),
                    "risk_keywords": result.get("risk_keywords", []),
                    "classification_confidence": float(result.get("classification_confidence", 0.5)),
                    "classification_reason": result.get("classification_reason", ""),
                }
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"JSON 解析失败: {e}")

        logger.warning(f"无法解析 LLM 响应: {response[:200]}")
        return {}


class ClassifierFactory:
    """分类器工厂"""

    @staticmethod
    def create(provider: str = "local", **kwargs) -> BaseClassifier:
        """
        创建分类器

        Args:
            provider: 提供者类型
                - "local": 本地模型 (需要本地模型路径)
                - "deepseek": DeepSeek API
                - "openai": OpenAI API
            **kwargs: 其他参数

        Returns:
            分类器实例
        """
        provider = provider.lower()

        if provider == "local":
            model_name = kwargs.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
            max_tokens = kwargs.get("max_new_tokens", 200)
            temperature = kwargs.get("temperature", 0.3)
            logger.info(f"创建本地分类器: {model_name}")
            return LocalClassifier(model_name, max_tokens, temperature)

        elif provider == "deepseek":
            api_key = kwargs.get("api_key") or os.getenv("DEEPSEEK_API_KEY", "")
            api_base = kwargs.get("api_base") or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
            model = kwargs.get("model") or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            max_tokens = kwargs.get("max_tokens", 500)
            temperature = kwargs.get("temperature", 0.3)
            if not api_key:
                raise ValueError("DeepSeek API key is required")
            logger.info(f"创建 DeepSeek 分类器: {model}")
            return DeepSeekClassifier(api_key, api_base, model, max_tokens, temperature)

        elif provider == "openai":
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY", "")
            api_base = kwargs.get("api_base") or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            model = kwargs.get("model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            max_tokens = kwargs.get("max_tokens", 500)
            temperature = kwargs.get("temperature", 0.3)
            if not api_key:
                raise ValueError("OpenAI API key is required")
            logger.info(f"创建 OpenAI 分类器: {model}")
            return OpenAIClassifier(api_key, api_base, model, max_tokens, temperature)

        else:
            raise ValueError(f"Unknown provider: {provider}")


class LLMTopicClassifier:
    """话题分类器 - 统一接口"""

    def __init__(self, provider: str = "local", **kwargs):
        """
        初始化分类器

        Args:
            provider: 分类器类型 ("local", "deepseek", "openai")
            **kwargs: 传递给具体分类器的参数
        """
        self.classifier = ClassifierFactory.create(provider, **kwargs)
        self.provider = provider

    def classify_topic(self, topic_id: int) -> Dict[str, Any]:
        """
        为单个话题生成分类

        Args:
            topic_id: 话题ID

        Returns:
            分类结果字典
        """
        topic = TopicEventRepo.get_non_merged_topics()
        topic = [t for t in topic if t["id"] == topic_id]
        if not topic:
            logger.warning(f"话题 {topic_id} 不存在")
            return {}
        topic = topic[0]

        keywords = topic.get("keywords", [])
        if isinstance(keywords, str):
            try:
                keywords = json.loads(keywords)
            except json.JSONDecodeError:
                keywords = []

        keywords_str = ", ".join([k.get("word", "") if isinstance(k, dict) else str(k) for k in keywords[:10]])

        topic_info = f"""## 话题信息
- 话题名称: {topic.get("event_name", "未知")}
- 话题描述: {topic.get("event_description", "无描述")}
- 关键词: {keywords_str or "无关键词"}
- 主导情感: {topic.get("dominant_sentiment", "未知")}
- 主要情绪: {topic.get("dominant_emotions", "无")}
- 内容数量: {topic.get("content_count", 0)}
- 话题状态: {topic.get("status", "未知")}"""

        return self.classifier.classify(topic_info)

    def classify_and_save(self, topic_id: int, dry_run: bool = False) -> Dict[str, Any]:
        """
        分类话题并保存到数据库

        Args:
            topic_id: 话题ID
            dry_run: 试运行

        Returns:
            分类结果
        """
        result = self.classify_topic(topic_id)

        if not result or not result.get("primary_category"):
            logger.warning(f"话题 {topic_id} 分类结果为空")
            return {}

        if dry_run:
            logger.info(f"[试运行] 话题 {topic_id} 分类: {json.dumps(result, ensure_ascii=False)}")
            return result

        TopicClassificationRepo.upsert(
            topic_id=topic_id,
            primary_category=result.get("primary_category"),
            secondary_category=result.get("secondary_category"),
            tertiary_category=result.get("tertiary_category"),
            industry_tags=result.get("industry_tags"),
            event_type=result.get("event_type"),
            audience_scope=result.get("audience_scope", "national"),
            time_sensitivity=result.get("time_sensitivity", "normal"),
            sentiment_intensity=result.get("sentiment_intensity", "moderate"),
            controversy_level=result.get("controversy_level", "low"),
            risk_level=result.get("risk_level", "safe"),
            risk_keywords=result.get("risk_keywords"),
            classification_confidence=result.get("classification_confidence"),
            classified_by=self.provider,
            classification_reason=result.get("classification_reason"),
        )

        sql = """
            UPDATE topic_event
            SET primary_category = %s, event_type = %s, risk_level = %s,
                classification_version = classification_version + 1
            WHERE id = %s
        """
        from ..database.connection import execute_update
        execute_update(sql, (
            result.get("primary_category"),
            result.get("event_type"),
            result.get("risk_level"),
            topic_id
        ))

        logger.info(f"话题 {topic_id} 分类完成: {result.get('primary_category')} / {result.get('event_type')}")
        return result

    def batch_classify(
        self,
        topic_ids: Optional[List[int]] = None,
        only_unclassified: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """
        批量分类话题

        Args:
            topic_ids: 指定话题ID列表 (None=全部)
            only_unclassified: 仅处理未分类话题
            dry_run: 试运行

        Returns:
            {total, success, error}
        """
        stats = {"total": 0, "success": 0, "error": 0}

        if topic_ids:
            topics = TopicEventRepo.get_non_merged_topics()
            topics = [t for t in topics if t["id"] in topic_ids]
        elif only_unclassified:
            topics = TopicClassificationRepo.get_unclassified_topics(limit=1000)
        else:
            topics = TopicEventRepo.get_non_merged_topics()

        stats["total"] = len(topics)
        logger.info(f"待分类话题: {len(topics)} 个, 使用分类器: {self.classifier.name()}")

        for topic in topics:
            topic_id = topic["id"]
            try:
                self.classify_and_save(topic_id, dry_run=dry_run)
                stats["success"] += 1
            except Exception as e:
                logger.error(f"话题 {topic_id} 分类失败: {e}")
                stats["error"] += 1

        logger.info(
            f"分类完成: 总计 {stats['total']}, "
            f"成功 {stats['success']}, 失败 {stats['error']}"
        )
        return stats

    def sync_to_topic_event(self) -> int:
        """
        同步分类信息到 topic_event 表

        Returns:
            更新的行数
        """
        sql = """
            UPDATE topic_event te
            JOIN topic_classification tc ON te.id = tc.topic_id
            SET te.primary_category = tc.primary_category,
                te.event_type = tc.event_type,
                te.risk_level = tc.risk_level,
                te.classification_version = te.classification_version + 1
            WHERE tc.classified_by = %s
        """
        from ..database.connection import execute_update
        count = execute_update(sql, (self.provider,))
        logger.info(f"同步分类信息到 topic_event: {count} 条")
        return count
