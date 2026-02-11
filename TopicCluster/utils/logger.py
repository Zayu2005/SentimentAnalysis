# -*- coding: utf-8 -*-
"""
日志工具

提供统一的日志配置
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "TopicCluster",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志名称
        level: 日志级别
        log_file: 日志文件路径 (可选)

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "TopicCluster") -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 日志名称

    Returns:
        Logger 实例
    """
    return logging.getLogger(name)
