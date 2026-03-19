# -*- coding: utf-8 -*-
"""配置模块"""

from .settings import (
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    Settings,
    get_settings,
    LABEL_NEGATIVE,
    LABEL_NEUTRAL,
    LABEL_POSITIVE,
    LABEL_NAMES,
    LABEL_NAMES_CN,
    SENTIMENT_SCORE_MIN,
    SENTIMENT_SCORE_MAX
)

__all__ = [
    'ModelConfig',
    'TrainingConfig',
    'InferenceConfig',
    'Settings',
    'get_settings',
    'LABEL_NEGATIVE',
    'LABEL_NEUTRAL',
    'LABEL_POSITIVE',
    'LABEL_NAMES',
    'LABEL_NAMES_CN',
    'SENTIMENT_SCORE_MIN',
    'SENTIMENT_SCORE_MAX'
]
