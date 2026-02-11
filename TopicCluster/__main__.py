# -*- coding: utf-8 -*-
"""
TopicCluster 命令行入口

使用方法:
    python -m TopicCluster cluster
    python -m TopicCluster stats
    python -m TopicCluster merge
    python -m TopicCluster describe
    python -m TopicCluster evolve
    python -m TopicCluster recluster --confirm
"""

from .cli import main

if __name__ == "__main__":
    main()
