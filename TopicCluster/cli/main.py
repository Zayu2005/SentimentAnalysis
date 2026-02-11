# -*- coding: utf-8 -*-
"""
命令行工具

提供话题聚类的各种操作命令
"""

import argparse
import logging
import sys

from ..config import get_settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def cmd_cluster(args):
    """增量聚类命令"""
    from ..cluster import ClusterEngine

    settings = get_settings()
    threshold = args.threshold or settings.clustering.similarity_threshold

    print(f"\n开始增量聚类...")
    print(f"相似度阈值: {threshold}")
    print(f"批次大小: {args.batch_size}")
    if args.limit:
        print(f"最大处理数: {args.limit}")
    if args.dry_run:
        print("[试运行模式]")

    engine = ClusterEngine(similarity_threshold=threshold)
    result = engine.run(
        batch_size=args.batch_size,
        max_items=args.limit,
        dry_run=args.dry_run,
    )

    print(f"\n聚类结果:")
    print(f"  处理总数:   {result.total_processed}")
    print(f"  归入现有:   {result.assigned_to_existing}")
    print(f"  新建话题:   {result.new_topics_created}")
    print(f"  错误:       {result.errors}")


def cmd_stats(args):
    """统计信息命令"""
    from ..database import TopicEventRepo, TopicContentRepo

    print("\n" + "=" * 60)
    print("话题聚类统计信息")
    print("=" * 60)

    # 内容统计
    unclustered = TopicContentRepo.count_unclustered()
    clustered = TopicContentRepo.count_clustered()

    print(f"\n内容统计:")
    print(f"  已聚类:     {clustered}")
    print(f"  待聚类:     {unclustered}")
    print(f"  总计:       {clustered + unclustered}")

    # 话题统计
    topic_stats = TopicEventRepo.get_topic_stats()
    if topic_stats:
        print(f"\n话题统计:")
        print(f"  总话题数:       {topic_stats.get('total_topics', 0)}")
        print(f"  活跃话题:       {topic_stats.get('active_topics', 0)}")
        print(f"  关联内容总数:   {topic_stats.get('total_content', 0)}")
        avg = topic_stats.get('avg_content_per_topic')
        if avg is not None:
            print(f"  平均内容/话题:  {float(avg):.1f}")
        print(f"  最大内容/话题:  {topic_stats.get('max_content_per_topic', 0)}")

    # 状态分布
    status_counts = TopicEventRepo.count_by_status()
    if status_counts:
        print(f"\n状态分布:")
        for row in status_counts:
            print(f"  {row['status']:12s} {row['cnt']}")

    print("\n" + "=" * 60)


def cmd_merge(args):
    """话题合并命令"""
    from ..cluster import TopicMaintainer

    settings = get_settings()
    threshold = args.threshold or settings.clustering.merge_threshold

    print(f"\n开始话题合并检查...")
    print(f"合并阈值: {threshold}")
    if args.dry_run:
        print("[试运行模式]")

    maintainer = TopicMaintainer(merge_threshold=threshold)
    result = maintainer.merge_topics(dry_run=args.dry_run)

    print(f"\n合并结果:")
    print(f"  检查话题对:   {result.pairs_checked}")
    print(f"  合并数:       {result.merged_count}")
    if not args.dry_run:
        print(f"  重分配内容:   {result.content_reassigned}")


def cmd_evolve(args):
    """演化快照 + 生命周期更新 + 统计更新"""
    from ..cluster import TopicMaintainer

    maintainer = TopicMaintainer()

    print(f"\n更新话题生命周期...")
    maintainer.update_lifecycle()

    print(f"\n更新话题统计...")
    maintainer.update_all_stats()

    print(f"\n生成演化快照...")
    maintainer.generate_evolution()

    print(f"\n演化更新完成!")


def cmd_describe(args):
    """LLM 话题描述命令"""
    from ..llm import TopicNamer

    print(f"\n开始话题命名...")
    print(f"模型: {args.model or '默认'}")
    if args.dry_run:
        print("[试运行模式]")

    namer = TopicNamer(model_name=args.model)
    stats = namer.describe_topics(
        only_unnamed=not args.all,
        include_ended=args.include_ended,
        dry_run=args.dry_run,
    )

    print(f"\n命名结果:")
    print(f"  总计:   {stats['total']}")
    print(f"  成功:   {stats['success']}")
    print(f"  失败:   {stats['error']}")


def cmd_recluster(args):
    """全量重聚类命令"""
    from ..database import TopicEventRepo, TopicContentRepo
    from ..cluster import ClusterEngine

    if not args.confirm:
        print("\n全量重聚类将清除所有现有话题和分配!")
        print("请使用 --confirm 参数确认操作")
        sys.exit(1)

    settings = get_settings()
    threshold = args.threshold or settings.clustering.similarity_threshold

    print(f"\n开始全量重聚类...")
    print(f"相似度阈值: {threshold}")

    # 清除现有数据
    print("清除话题分配...")
    cleared = TopicContentRepo.clear_all_topic_assignments()
    print(f"已清除 {cleared} 条内容分配")

    print("清除话题...")
    deleted = TopicEventRepo.reset_all_topics()
    print(f"已删除 {deleted} 个话题")

    # 重新聚类
    engine = ClusterEngine(similarity_threshold=threshold)
    result = engine.run(batch_size=args.batch_size)

    print(f"\n重聚类结果:")
    print(f"  处理总数:   {result.total_processed}")
    print(f"  归入现有:   {result.assigned_to_existing}")
    print(f"  新建话题:   {result.new_topics_created}")
    print(f"  错误:       {result.errors}")


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog="TopicCluster",
        description="话题聚类与事件检测"
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # cluster 命令
    cluster_parser = subparsers.add_parser("cluster", help="增量聚类")
    cluster_parser.add_argument(
        "--threshold", "-t", type=float, default=None,
        help="相似度阈值 (默认: 0.75)"
    )
    cluster_parser.add_argument(
        "--batch-size", "-b", type=int, default=64,
        help="批次大小 (默认: 64)"
    )
    cluster_parser.add_argument(
        "--limit", "-l", type=int, default=None,
        help="最大处理数量"
    )
    cluster_parser.add_argument(
        "--dry-run", action="store_true",
        help="试运行 (不保存结果)"
    )

    # stats 命令
    subparsers.add_parser("stats", help="查看统计信息")

    # merge 命令
    merge_parser = subparsers.add_parser("merge", help="合并相似话题")
    merge_parser.add_argument(
        "--threshold", "-t", type=float, default=None,
        help="合并阈值 (默认: 0.90)"
    )
    merge_parser.add_argument(
        "--dry-run", action="store_true",
        help="试运行 (不执行合并)"
    )

    # evolve 命令
    subparsers.add_parser("evolve", help="更新演化快照和生命周期")

    # describe 命令
    describe_parser = subparsers.add_parser("describe", help="LLM 话题命名")
    describe_parser.add_argument(
        "--model", "-m", default=None,
        help="LLM 模型名称"
    )
    describe_parser.add_argument(
        "--all", action="store_true",
        help="重新命名所有话题 (默认只命名未命名的)"
    )
    describe_parser.add_argument(
        "--include-ended", action="store_true",
        help="包含已结束话题 (默认仅处理活跃话题)"
    )
    describe_parser.add_argument(
        "--dry-run", action="store_true",
        help="试运行"
    )

    # recluster 命令
    recluster_parser = subparsers.add_parser("recluster", help="全量重聚类")
    recluster_parser.add_argument(
        "--confirm", action="store_true",
        help="确认执行 (必须指定)"
    )
    recluster_parser.add_argument(
        "--threshold", "-t", type=float, default=None,
        help="相似度阈值"
    )
    recluster_parser.add_argument(
        "--batch-size", "-b", type=int, default=64,
        help="批次大小"
    )

    return parser


# 命令映射
COMMANDS = {
    "cluster": cmd_cluster,
    "stats": cmd_stats,
    "merge": cmd_merge,
    "evolve": cmd_evolve,
    "describe": cmd_describe,
    "recluster": cmd_recluster,
}


def main():
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command in COMMANDS:
        try:
            COMMANDS[args.command](args)
        except KeyboardInterrupt:
            print("\n操作已取消")
            sys.exit(1)
        except Exception as e:
            logger.error(f"执行失败: {e}")
            raise
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
