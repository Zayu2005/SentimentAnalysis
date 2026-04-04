# -*- coding: utf-8 -*-
"""
报告生成器 - 输出 CSV + JSON + 控制台表格

对应论文章节: 5.4.1.4 数据清洗流水线评估
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..config import config


class ReportGenerator:
    """评估报告生成器"""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def add_result(self, result: Dict[str, Any]):
        """添加单个模块的评估结果"""
        self.results.append(result)

    def set_time(self, start: datetime, end: datetime):
        """设置运行时间"""
        self.start_time = start
        self.end_time = end

    def generate_all(self) -> Dict[str, Any]:
        """生成所有格式的报告"""
        output_dir = config.output_path
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "eval_report.csv"
        json_path = output_dir / "eval_report.json"

        report_data = self._build_report()

        self._save_csv(csv_path, report_data)
        self._save_json(json_path, report_data)
        self._print_console(report_data)

        return report_data

    def _build_report(self) -> Dict[str, Any]:
        """构建完整报告数据"""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        return {
            "title": "数据清洗质量评估报告",
            "section": "5.4.1.4 数据清洗流水线评估",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(duration, 2) if duration else None,
            "modules": self.results,
            "summary": self._build_summary(),
        }

    def _build_summary(self) -> List[Dict[str, str]]:
        """构建汇总表格行"""
        rows = []
        for r in self.results:
            module = r.get("module", "未知")
            dataset = r.get("dataset", "-")
            metrics = r.get("metrics", {})

            for metric_name, value in metrics.items():
                value_str = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                rows.append({
                    "清洗环节": module,
                    "评估数据集": dataset,
                    "指标": metric_name,
                    "结果": value_str,
                })
        return rows

    def _save_csv(self, path: Path, data: Dict):
        """保存 CSV 报告"""
        rows = data.get("summary", [])
        if not rows:
            return

        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[Report] CSV 报告已保存: {path}")

    def _save_json(self, path: Path, data: Dict):
        """保存 JSON 报告"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"[Report] JSON 报告已保存: {path}")

    def _print_console(self, data: Dict):
        """打印控制台报告 (使用 rich 库美化)"""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel

            console = Console()
        except ImportError:
            self._print_plain(data)
            return

        console.print()
        console.print(Panel(
            f"[bold cyan]{data['title']}[/bold cyan]\n"
            f"论文章节: {data.get('section', '-')}\n"
            f"生成时间: {data['generated_at']}\n"
            f"运行耗时: {data.get('duration_seconds', '-')} 秒",
            title="[bold]SentimentAnalysis 数据清洗评估[/bold]",
            border_style="cyan",
        ))

        summary = data.get("summary", [])
        if not summary:
            console.print("[yellow]暂无评估结果[/yellow]")
            return

        table = Table(
            title="评估结果汇总",
            show_header=True,
            header_style="bold magenta",
            title_style="bold",
        )
        table.add_column("清洗环节", style="cyan", min_width=18)
        table.add_column("评估数据集", style="green", min_width=14)
        table.add_column("指标", style="yellow", min_width=12)
        table.add_column("结果", style="white bold", justify="right")

        for row in summary:
            val = row["结果"]
            if "/" in val:
                style = "white"
            elif float(val) >= 0.9:
                style = "green"
            elif float(val) >= 0.7:
                style = "yellow"
            else:
                style = "red"
            table.add_row(row["清洗环节"], row["评估数据集"], row["指标"], f"[{style}]{val}[/{style}]")

        console.print(table)
        console.print()

    @staticmethod
    def _print_plain(data: Dict):
        """无 rich 库时的纯文本输出"""
        print()
        print("=" * 72)
        print(f"  {data['title']}")
        print(f"  论文章节: {data.get('section', '-')}")
        print(f"  生成时间: {data['generated_at']}")
        print("=" * 72)

        summary = data.get("summary", [])
        if not summary:
            print("  暂无评估结果")
            return

        header = f"  {'清洗环节':<18} {'评估数据集':<14} {'指标':<12} {'结果':>10}"
        print(header)
        print("  " + "-" * 60)

        for row in summary:
            print(f"  {row['清洗环节']:<18} {row['评估数据集']:<14} {row['指标']:<12} {row['结果']:>10}")

        print("=" * 72)
        print()
