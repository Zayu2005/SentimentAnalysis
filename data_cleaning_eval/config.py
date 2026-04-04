# -*- coding: utf-8 -*-
"""
数据清洗质量评估模块 - 配置文件

对应论文章节: 5.4.1.4 数据清洗流水线评估
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvalConfig:
    """评估配置"""

    # ==================== 数据路径 ====================
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    # SIGHAN 2005 PKU 分词测试集 (词语间以空格分隔)
    sighan_pku_test_path: str = "data/sighan2005/pku_test_gold.utf8"

    # NLPCC 2016 微博分词测试集
    nlpcc_test_path: str = "data/nlpcc2016/weibo_test_gold.txt"

    # LCQMC 相似句对数据集 (格式: 句子1\t句子2\t标签)
    lcqmc_test_path: str = "data/lcqmc/test.txt"

    # 自定义词典 (每行: 词语 词频 词性)
    custom_dict_path: str = "dict/custom_dict.txt"

    # 停用词表 (每行一个词)
    stopword_path: str = "dict/stopwords.txt"

    # 情感词典 (用于检测情感词误删)
    sentiment_dict_path: str = "dict/sentiment_words.txt"

    # HTML清洗测试数据 (JSON格式)
    html_test_data_path: str = "data/html_test_data.json"

    # 停用词过滤测试数据 (JSON格式)
    stopword_test_data_path: str = "data/stopword_test_data.json"

    # ==================== 阈值参数 ====================
    simhash_threshold: int = 3
    cosine_threshold: float = 0.75

    # LCQMC 正负样本数量 (保持均衡)
    lcqmc_positive_samples: int = 500
    lcqmc_negative_samples: int = 500

    # HTML测试样本数
    html_test_samples: int = 500

    # 停用词测试样本数
    stopword_test_samples: int = 500

    # ==================== 输出目录 ====================
    output_dir: str = "output/"

    @property
    def output_path(self) -> Path:
        return self.base_dir / self.output_dir

    def resolve(self, path: str) -> Path:
        """解析相对路径为绝对路径"""
        return self.base_dir / path

    def check_required_files(self) -> List[str]:
        """检查必需文件是否存在，返回缺失列表"""
        missing = []
        required_paths = [
            self.sighan_pku_test_path,
            self.nlpcc_test_path,
            self.lcqmc_test_path,
            self.stopword_path,
        ]
        for p in required_paths:
            if not self.resolve(p).exists():
                missing.append(p)

        optional_paths = [
            self.custom_dict_path,
            self.sentiment_dict_path,
            self.html_test_data_path,
            self.stopword_test_data_path,
        ]
        return missing


# 全局配置实例
config = EvalConfig()
