"""Run the full local benchmark suite and write a Markdown report."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class SuiteCommand:
    name: str
    command: list[str]
    output_file: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Fusion's full benchmark suite.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark-results")
    parser.add_argument("--quick", action="store_true", help="Use smaller local sizes for a fast comparison run.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = _commands(quick=args.quick)
    sections = [_run(item, args.output_dir) for item in commands]
    report_path = args.output_dir / "full-benchmark-report.md"
    report_path.write_text(_markdown_report(sections, quick=args.quick), encoding="utf-8")
    print(report_path)
    return int(any(section["returncode"] != 0 for section in sections))


def _commands(*, quick: bool) -> list[SuiteCommand]:
    py = sys.executable
    if quick:
        event = ["--bars", "512", "--symbols", "8", "--repeat", "3", "--warmup", "1"]
        factor = ["--dates", "250", "--symbols", "500", "--repeat", "3", "--warmup", "1"]
        rebalance = ["--dates", "100", "--symbols", "40", "--top", "8", "--repeat", "3", "--warmup", "1"]
        sweep = ["--dates", "1200", "--symbols", "120", "--cases", "12", "--workers", "4", "--repeat", "1", "--warmup", "0"]
    else:
        event = ["--bars", "5000", "--symbols", "20", "--repeat", "3", "--warmup", "1"]
        factor = ["--dates", "500", "--symbols", "1000", "--repeat", "3", "--warmup", "1"]
        rebalance = ["--dates", "500", "--symbols", "200", "--top", "20", "--repeat", "3", "--warmup", "1"]
        sweep = ["--dates", "1500", "--symbols", "160", "--cases", "16", "--workers", "4", "--repeat", "1", "--warmup", "0"]
    return [
        SuiteCommand(
            "event_engines",
            [py, "benchmarks/event_engine_compare.py", *event, "--scenario", "all"],
            "event_engines.csv",
        ),
        SuiteCommand(
            "event_modes_buy_once",
            [py, "benchmarks/event_engine_modes_compare.py", "--scenario", "buy_once", *event],
            "event_modes_buy_once.csv",
        ),
        SuiteCommand(
            "event_modes_multi_order",
            [py, "benchmarks/event_engine_modes_compare.py", "--scenario", "multi_order", *event],
            "event_modes_multi_order.csv",
        ),
        SuiteCommand(
            "record_export",
            [py, "benchmarks/event_record_export_compare.py", *event],
            "record_export.csv",
        ),
        SuiteCommand(
            "rebalance_execution",
            [py, "benchmarks/rebalance_execution_compare.py", *rebalance],
            "rebalance_execution.csv",
        ),
        SuiteCommand(
            "factor_research",
            [py, "benchmarks/factor_research_compare.py", *factor],
            "factor_research.csv",
        ),
        SuiteCommand(
            "execution_sweep_parallel",
            [py, "benchmarks/execution_sweep_parallel_compare.py", *sweep],
            "execution_sweep_parallel.csv",
        ),
    ]


def _run(item: SuiteCommand, output_dir: Path) -> dict:
    completed = subprocess.run(
        item.command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output_path = output_dir / item.output_file
    output_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path = output_dir / f"{item.name}.stderr.txt"
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return {
        "name": item.name,
        "command": " ".join(item.command),
        "returncode": completed.returncode,
        "output_path": output_path,
        "stderr_path": stderr_path,
        "rows": _csv_rows(completed.stdout),
    }


def _csv_rows(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line and "," in line]
    if not lines:
        return []
    header_index = 0
    for index, line in enumerate(lines):
        first_column = line.split(",", 1)[0]
        if first_column in {"scenario", "mode", "engine"}:
            header_index = index
            break
    return list(csv.DictReader(lines[header_index:]))


def _markdown_report(sections: list[dict], *, quick: bool) -> str:
    lines = [
        "# Full Benchmark Suite",
        "",
        f"Date: {datetime.now(UTC).isoformat()}",
        f"Mode: {'quick' if quick else 'full'}",
        "",
    ]
    for section in sections:
        lines.extend(
            [
                f"## {section['name']}",
                "",
                f"Command: `{section['command']}`",
                f"Return code: `{section['returncode']}`",
                f"CSV: `{section['output_path']}`",
                f"stderr: `{section['stderr_path']}`",
                "",
            ]
        )
        rows = section["rows"]
        if rows:
            columns = ["" if column is None else str(column) for column in rows[0]]
            lines.append("| " + " | ".join(columns) + " |")
            lines.append("| " + " | ".join("---" for _ in columns) + " |")
            for row in rows[:40]:
                lines.append("| " + " | ".join(str(row.get(column or None, "")) for column in columns) + " |")
            if len(rows) > 40:
                lines.append("")
                lines.append(f"Showing 40 of {len(rows)} rows.")
        else:
            lines.append("No CSV rows captured.")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
