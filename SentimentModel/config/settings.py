# -*- coding: utf-8 -*-
"""
配置管理模块

使用 Pydantic 进行配置验证和管理
"""

import os
from typing import Optional, Literal
from pydantic import BaseModel, Field
from functools import lru_cache

from config.settings import get_mysql_config, get_model_names


class ModelConfig(BaseModel):
    """模型配置"""
    name: str = Field(
        default="hfl/chinese-roberta-wwm-ext",
        description="预训练模型名称"
    )
    num_labels: int = Field(
        default=3,
        description="分类标签数量 (0=负面, 1=中性, 2=正面)"
    )
    max_length: int = Field(
        default=128,
        description="输入序列最大长度"
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout 概率"
    )
    device: str = Field(
        default="auto",
        description="计算设备 (auto/cpu/cuda/cuda:0)"
    )

    # 标签映射
    id2label: dict = Field(
        default={0: "negative", 1: "neutral", 2: "positive"},
        description="ID 到标签的映射"
    )
    label2id: dict = Field(
        default={"negative": 0, "neutral": 1, "positive": 2},
        description="标签到 ID 的映射"
    )


class TrainingConfig(BaseModel):
    """训练配置"""
    batch_size: int = Field(
        default=32,
        ge=1,
        description="批次大小"
    )
    learning_rate: float = Field(
        default=2e-5,
        gt=0,
        description="学习率"
    )
    epochs: int = Field(
        default=3,
        ge=1,
        description="训练轮数"
    )
    warmup_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="预热步数比例"
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="权重衰减"
    )
    fp16: bool = Field(
        default=True,
        description="是否使用混合精度训练"
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        description="梯度累积步数"
    )
    max_grad_norm: float = Field(
        default=1.0,
        gt=0,
        description="梯度裁剪阈值"
    )
    output_dir: str = Field(
        default="models",
        description="模型输出目录"
    )
    logging_steps: int = Field(
        default=100,
        ge=1,
        description="日志记录间隔"
    )
    save_steps: int = Field(
        default=500,
        ge=1,
        description="模型保存间隔"
    )
    eval_steps: int = Field(
        default=500,
        ge=1,
        description="评估间隔"
    )
    seed: int = Field(
        default=42,
        description="随机种子"
    )


class InferenceConfig(BaseModel):
    """推理配置"""
    model_config = {"protected_namespaces": ()}  # 允许 model_ 前缀字段

    batch_size: int = Field(
        default=64,
        ge=1,
        description="推理批次大小"
    )
    model_path: str = Field(
        default="models/best_model",
        description="模型路径"
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="置信度阈值"
    )
    device: str = Field(
        default="auto",
        description="计算设备"
    )


class Settings(BaseModel):
    """全局设置"""
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    database: object = Field(default=None)

    # 路径配置
    base_dir: str = Field(
        default="",
        description="项目根目录"
    )
    data_dir: str = Field(
        default="data",
        description="数据目录"
    )
    log_dir: str = Field(
        default="logs",
        description="日志目录"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if not self.base_dir:
            # 设置为 SentimentModel 模块所在目录
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量加载设置"""
        model_names = get_model_names()
        return cls(
            model=ModelConfig(
                name=model_names.bert_model,
            ),
            database=get_mysql_config(),
        )


@lru_cache()
def get_settings() -> Settings:
    """获取全局设置 (单例模式)"""
    return Settings.from_env()


# 标签常量
LABEL_NEGATIVE = "negative"
LABEL_NEUTRAL = "neutral"
LABEL_POSITIVE = "positive"

LABEL_NAMES = [LABEL_NEGATIVE, LABEL_NEUTRAL, LABEL_POSITIVE]
LABEL_NAMES_CN = ["负面", "中性", "正面"]

# 情感分数范围
SENTIMENT_SCORE_MIN = -1.0
SENTIMENT_SCORE_MAX = 1.0
