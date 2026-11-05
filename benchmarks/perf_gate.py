"""Run the CI performance-regression gate's fixed measurement set.

Five dimensions, one committed size each, chosen to run in well under a
minute total so this can sit in CI without becoming the thing people wait on:

- **open-order scan throughput** (`event_engine_compare.py`) — the scenario
  that stresses the open-order index rather than fill/mark bookkeeping.
- **large-scale throughput & peak memory** (`extreme_scaling_compare.py`) —
  100k and 1M bars, `fusion` only (no reference clones, so this runs on a
  plain `pip install -e .[dev]`).
- **serial vs. process execution-sweep throughput**
  (`execution_sweep_parallel_compare.py`) — both scheduling modes, compared
  against their own history, not against each other (process has fixed
  per-task overhead that serial does not, so a slower `process` number here is
  not by itself a regression signal at this case count).
- **Polars feed construction** (`feed_construction_compare.py`) — `from_polars`
  and `from_bars`, the two ways a caller turns data into a `ColumnarBarFeed`.
- **multi-symbol stress throughput** (`large_scale_stress_compare.py`) —
  OCO brackets, trailing stops, long-lived resting limits, and partial fills
  together, at 300 symbols x 400 sparse sessions (~4,100 open orders by the
  end). Added after profiling found `cancel_oco_siblings` scanning every open
  order in the book, any symbol, once per fill — a real ~13x regression this
  set would have caught, and none of the other four dimensions would have
  (they don't use OCO at a scale where the bug bites).

Each existing script keeps its own methodology (warmup, best-of-N, subprocess
memory isolation where it already does that); this just runs them at fixed
sizes and normalizes their stdout into one flat metrics dict, written as JSON.

    python benchmarks/perf_gate.py --output benchmark-results/perf-gate/latest.json

Compare the result against a committed baseline with `compare_perf_gate.py`.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# name -> (higher_is_better, unit)
METRIC_KINDS: dict[str, tuple[bool, str]] = {
    "open_order_scan.bars_per_second": (True, "bars/s"),
    "extreme_scaling.fusion_100000.bars_per_second": (True, "bars/s"),
    "extreme_scaling.fusion_100000.peak_rss_mb": (False, "MB"),
    "extreme_scaling.fusion_1000000.bars_per_second": (True, "bars/s"),
    "extreme_scaling.fusion_1000000.peak_rss_mb": (False, "MB"),
    "stress.bar_observations_per_second": (True, "obs/s"),
    "execution_sweep.serial.cases_per_second": (True, "cases/s"),
    "execution_sweep.process.cases_per_second": (True, "cases/s"),
    "feed_construction.from_polars.rows_per_second": (True, "rows/s"),
    "feed_construction.from_bars.rows_per_second": (True, "rows/s"),
}


def _run(args: list[str]) -> str:
    completed = subprocess.run(
        [PYTHON, str(ROOT / args[0]), *args[1:]],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{args[0]} failed (exit {completed.returncode}):\n{completed.stderr}")
    return completed.stdout


def _csv_rows(stdout: str) -> list[dict[str, str]]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    header_index = next(i for i, line in enumerate(lines) if "," in line and not line[0].isdigit())
    return list(csv.DictReader(io.StringIO("\n".join(lines[header_index:]))))


def _measure_open_order_scan() -> dict[str, float]:
    stdout = _run([
        "event_engine_compare.py",
        "--scenario", "open_order_scan",
        "--bars", "20000",
        "--symbols", "5",
        "--repeat", "3",
        "--warmup", "1",
    ])
    row = _csv_rows(stdout)[0]
    return {"open_order_scan.bars_per_second": float(row["bars_per_second"])}


def _measure_extreme_scaling() -> dict[str, float]:
    # This script only writes structured CSV to --output; its stdout is a
    # human-readable progress table, not data.
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "extreme_scaling.csv"
        _run([
            "extreme_scaling_compare.py",
            "--engines", "fusion",
            "--bars", "100000,1000000",
            "--repeats", "2",
            "--output", str(csv_path),
        ])
        with csv_path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    metrics: dict[str, float] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("engine") != "fusion":
            continue
        bars = row["bars"]
        metrics[f"extreme_scaling.fusion_{bars}.bars_per_second"] = float(row["bars_per_second"])
        metrics[f"extreme_scaling.fusion_{bars}.peak_rss_mb"] = float(row["peak_rss_mb"])
    return metrics


def _measure_execution_sweep() -> dict[str, float]:
    stdout = _run([
        "execution_sweep_parallel_compare.py",
        "--dates", "250",
        "--symbols", "50",
        "--cases", "8",
        "--workers", "2",
        "--repeat", "3",
        "--warmup", "1",
    ])
    metrics: dict[str, float] = {}
    for row in _csv_rows(stdout):
        metrics[f"execution_sweep.{row['mode']}.cases_per_second"] = float(row["cases_per_second"])
    return metrics


def _measure_feed_construction() -> dict[str, float]:
    stdout = _run([
        "feed_construction_compare.py",
        "--dates", "500",
        "--symbols", "200",
        "--repeat", "3",
    ])
    metrics: dict[str, float] = {}
    for row in _csv_rows(stdout):
        metrics[f"feed_construction.{row['source']}.rows_per_second"] = float(row["rows_per_second"])
    return metrics


def _measure_stress() -> dict[str, float]:
    # --subprocess: perf_gate's own _run() already isolates this in a
    # subprocess: the script's own re-exec-for-RSS-isolation wrapper would
    # otherwise nest a second one for no reason.
    stdout = _run([
        "large_scale_stress_compare.py",
        "--symbols", "300",
        "--bars", "400",
        "--subprocess",
    ])
    row = json.loads(stdout)
    return {"stress.bar_observations_per_second": float(row["bar_observations_per_second"])}


def run_all() -> dict[str, float]:
    metrics: dict[str, float] = {}
    for fn in (
        _measure_open_order_scan,
        _measure_extreme_scaling,
        _measure_execution_sweep,
        _measure_feed_construction,
        _measure_stress,
    ):
        metrics.update(fn())
    missing = set(METRIC_KINDS) - set(metrics)
    if missing:
        raise RuntimeError(f"perf gate did not produce expected metrics: {sorted(missing)}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = run_all()
    payload = {
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    width = max(len(name) for name in metrics)
    for name, value in sorted(metrics.items()):
        _, unit = METRIC_KINDS[name]
        print(f"{name:<{width}}  {value:>14,.1f} {unit}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
