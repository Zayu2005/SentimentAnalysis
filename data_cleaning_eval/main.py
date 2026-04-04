# -*- coding: utf-8 -*-
"""
数据清洗质量评估 - 主入口

对应论文章节: 5.4.1.4 数据清洗流水线评估

执行流程:
    1. 检查数据文件
    2. HTML 清洗评估
    3. 分词质量评估 (SIGHAN PKU + NLPCC 微博)
    4. 停用词过滤评估
    5. 近似去重评估 (SimHash vs Cosine)
    6. 生成报告 (CSV + JSON + 控制台)
"""

import sys
from datetime import datetime

from .config import config, EvalConfig
from .evaluators.html_cleaner_eval import HTMLCleanerEvaluator
from .evaluators.segmentation_eval import SegmentationEvaluator
from .evaluators.stopword_eval import StopwordEvaluator
from .evaluators.dedup_eval import DedupEvaluator
from .utils.report_generator import ReportGenerator


def check_data_files() -> bool:
    """
    检查必需的数据文件是否存在

    Returns:
        True: 所有必要文件存在; False: 缺少文件
    """
    missing = config.check_required_files()

    if missing:
        print("\n" + "=" * 60)
        print("  [错误] 缺少以下必需的数据文件:")
        print("=" * 60)
        for f in missing:
            resolved = config.resolve(f)
            print(f"    - {f}")
            print(f"      路径: {resolved}")
        print()
        print("=" * 60)
        print("  数据集下载说明:")
        print("-" * 60)
        print("""
  1. SIGHAN 2005 PKU 分词测试集:
     下载地址: http://sighan.cs.uchicago.edu/bakeoff2005/
     文件名: pku_test_gold.utf8
     放置路径: data/sighan2005/pku_test_gold.utf8

  2. NLPCC 2016 微博分词测试集:
     联系 NLPCC 官网获取评测数据
     文件名: weibo_test_gold.txt
     放置路径: data/nlpcc2016/weibo_test_gold.txt

  3. LCQMC 相似句对数据集:
     下载地址: https://huggingface.co/datasets/shibing624/lcqmc
     格式: 句子1\\t句子2\\t标签 (1=相似/0=不相似)
     放置路径: data/lcqmc/test.txt

  4. 停用词表:
     可从项目 SentimentProcessor/utils/stopwords.py 中提取
     或使用哈工大停用词表
     放置路径: dict/stopwords.txt
""")
        print("=" * 60)
        return False

    return True


def run_evaluation(custom_config: Optional[EvalConfig] = None) -> Dict:
    """
    运行完整的评估流程

    Args:
        custom_config: 自定义配置 (可选)

    Returns:
        完整报告字典
    """
    global config
    if custom_config:
        config = custom_config

    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"  数据清洗质量评估")
    print(f"  论文章节: 5.4.1.4 数据清洗流水线评估")
    print(f"  开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # 1. 检查数据文件
    if not check_data_files():
        sys.exit(1)

    report = ReportGenerator()
    report.set_time(start_time, None)

    # 2. HTML 清洗评估
    print("\n[1/4] HTML/特殊符号清洗评估")
    print("-" * 40)
    html_eval = HTMLCleanerEvaluator()
    html_result = html_eval.evaluate()
    report.add_result(html_result.to_dict())

    # 3. 分词质量评估
    print("\n[2/4] 分词质量评估 (SIGHAN PKU / NLPCC微博)")
    print("-" * 40)
    seg_eval = SegmentationEvaluator()
    seg_results = seg_eval.evaluate()
    for key, result in seg_results.items():
        report.add_result(result.to_dict())

    # 4. 停用词过滤评估
    print("\n[3/4] 停用词过滤评估")
    print("-" * 40)
    sw_eval = StopwordEvaluator()
    sw_result = sw_eval.evaluate()
    report.add_result(sw_result.to_dict())

    # 5. 近似去重评估
    print("\n[4/4] 近似去重评估 (SimHash vs Cosine)")
    print("-" * 40)
    dedup_eval = DedupEvaluator()
    dedup_results = dedup_eval.evaluate()
    for key, result in dedup_results.items():
        report.add_result(result.to_dict())

    # 6. 生成报告
    end_time = datetime.now()
    report.set_time(start_time, end_time)

    print("\n" + "=" * 60)
    print("  生成评估报告...")
    print("=" * 60)

    final_report = report.generate_all()

    duration = (end_time - start_time).total_seconds()
    print(f"\n  总运行时间: {duration:.2f} 秒")
    print(f"{'='*60}\n")

    return final_report


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="数据清洗质量评估工具 - 论文 5.4.1.4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    python -m data_cleaning_eval                    # 运行完整评估
    python -m data_cleaning_eval --check-only       # 仅检查数据文件
    python -m data_cleaning_eval --module html      # 仅运行HTML评估
    python -m data_cleaning_eval --module seg       # 仅运行分词评估
    python -m data_cleaning_eval --module stopword   # 仅运行停用词评估
    python -m data_cleaning_eval --module dedup      # 仅运行去重评估
        """,
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="仅检查数据文件是否完整"
    )
    parser.add_argument(
        "--module", "-m", choices=["html", "seg", "stopword", "dedup"],
        help="仅运行指定模块的评估"
    )
    parser.add_argument(
        "--simhash-threshold", type=int, default=None,
        help="SimHash 汉明距离阈值 (默认: 3)"
    )
    parser.add_argument(
        "--cosine-threshold", type=float, default=None,
        help="余弦相似度阈值 (默认: 0.75)"
    )

    args = parser.parse_args()

    if args.check_only:
        ok = check_data_files()
        sys.exit(0 if ok else 1)

    if args.module:
        from .config import config as cfg
        if args.simhash_threshold is not None:
            cfg.simhash_threshold = args.simhash_threshold
        if args.cosine_threshold is not None:
            cfg.cosine_threshold = args.cosine_threshold

        if args.module == "html":
            eval_obj = HTMLCleanerEvaluator()
            result = eval_obj.evaluate()
        elif args.module == "seg":
            eval_obj = SegmentationEvaluator()
            results = eval_obj.evaluate()
            result = list(results.values())[0] if results else None
        elif args.module == "stopword":
            eval_obj = StopwordEvaluator()
            result = eval_obj.evaluate()
        elif args.module == "dedup":
            eval_obj = DedupEvaluator(
                simhash_threshold=args.simhash_threshold,
                cosine_threshold=args.cosine_threshold,
            )
            results = eval_obj.evaluate()
            result = list(results.values())[0] if results else None

        if result:
            rg = ReportGenerator()
            rg.add_result(result.to_dict())
            rg.generate_all()
        sys.exit(0)

    run_evaluation()


if __name__ == "__main__":
    main()
