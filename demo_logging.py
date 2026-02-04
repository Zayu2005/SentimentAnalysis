#!/usr/bin/env python3
"""
Demo script to show the new logging system

Run: python3 demo_logging.py
Output will show to console AND be saved to hot_news/logs/hot_news_*.log
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from hot_news.logger import info, debug, warning, error, critical, logger

def main():
    print("\n" + "="*70)
    print("🎯 Hot News Pipeline - Logging System Demo")
    print("="*70 + "\n")

    # Show log file location
    log_dir = Path(__file__).parent / "hot_news" / "logs"
    print(f"📝 日志文件位置: {log_dir.resolve()}\n")

    # Demonstrate different log levels
    info("ℹ️  This is an INFO message - appears in console and file")
    debug("🐛 This is a DEBUG message - only in file")
    warning("⚠️  This is a WARNING message")
    error("❌ This is an ERROR message")
    critical("🚨 This is a CRITICAL message")

    # Example from actual pipeline
    print("\n" + "-"*70)
    print("📋 Example: Pipeline Execution Logs\n")

    info("\n" + "="*70)
    info("  🚀 热点新闻获取与分析流程")
    info("="*70)
    info("⏰ 开始时间: 2024-12-04 15:30:45")
    info("📋 批次ID: 12345")
    info("")
    info("📝 执行配置:")
    info("  • 热点限制: 50 条/平台")
    info("  • 关键词限制: 20 个/领域")
    info("  • LLM分析: ✓ 启用")
    info("  • 爬虫触发: ✓ 启用")
    info("")

    info("[Step 1/4] 🔍 获取热点新闻")
    info("-"*70)
    info("从 5 个平台获取热点...")
    info("  ✓ weibo             45 条")
    info("  ✓ zhihu             38 条")
    info("  ✓ bilibili          42 条")

    info("")
    info("[Step 2/4] 🎯 分析领域匹配")
    info("-"*70)
    info("使用 3 个领域进行分析...")
    info("  • 科技 (ID:1)")
    info("  • 金融 (ID:2)")
    info("  • 医疗 (ID:3)")
    info("")
    info("从 50 条热点中进行分析...")
    info("⚡ 并发执行 (最多 5 并发)")
    info("✅ 步骤完成: 匹配 15 条热点")

    error("  [错误] Some error occurred during processing")

    print("\n" + "="*70)
    print("✅ 日志系统演示完成")
    print("="*70)
    print(f"\n📂 查看日志文件: {log_dir.resolve()}")
    print("📄 今日日志: hot_news_*.log\n")

if __name__ == "__main__":
    main()
