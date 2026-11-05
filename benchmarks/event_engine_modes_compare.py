"""Compare Fusion event-engine execution modes."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.event_engine_compare import BuyFirstSlice, MultiOrderStrategy, make_fusion_bars  # noqa: E402
from cortexbacktest import BacktestEngine, BacktestResult, ColumnarBarFeed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Fusion event-engine modes.")
    parser.add_argument("--bars", type=int, default=5_000, help="Bars per symbol.")
    parser.add_argument("--symbols", type=int, default=10, help="Symbol count.")
    parser.add_argument("--scenario", choices=("buy_once", "multi_order"), default="buy_once")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Optional CSV destination for reproducible benchmark artifacts.")
    args = parser.parse_args()

    bars = make_fusion_bars(args.bars, args.symbols)
    feed = ColumnarBarFeed.from_bars(bars)
    strategy_factory = BuyFirstSlice if args.scenario == "buy_once" else MultiOrderStrategy
    total_bars = args.bars * args.symbols

    modes: list[tuple[str, Callable[[], BacktestResult]]] = [
        ("list_bar_snapshots", lambda: BacktestEngine(snapshot_frequency="bar").run(bars, strategy_factory())),
        ("feed_bar_snapshots", lambda: BacktestEngine(snapshot_frequency="bar").run(feed, strategy_factory())),
        ("feed_session_snapshots", lambda: BacktestEngine(snapshot_frequency="session").run(feed, strategy_factory())),
        ("feed_final_snapshot", lambda: BacktestEngine(snapshot_frequency="final").run(feed, strategy_factory())),
        (
            "feed_final_summary_retention",
            lambda: BacktestEngine(snapshot_frequency="final", result_retention="summary").run(feed, strategy_factory()),
        ),
        ("feed_no_snapshots", lambda: BacktestEngine(snapshot_frequency="none").run(feed, strategy_factory())),
    ]

    rows = []
    for mode, fn in modes:
        for _ in range(args.warmup):
            fn()
        seconds, equity_points = _time_best(fn, repeat=args.repeat)
        bars_per_second = total_bars / seconds if seconds > 0 else float("inf")
        rows.append(
            {
                "scenario": args.scenario,
                "mode": mode,
                "seconds": f"{seconds:.6f}",
                "bars_per_second": f"{bars_per_second:.2f}",
                "equity_points": equity_points,
            }
        )
    fieldnames = ["scenario", "mode", "seconds", "bars_per_second", "equity_points"]
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(",".join(fieldnames))
    for row in rows:
        print(",".join(str(row[field]) for field in fieldnames))
    return 0


def _time_best(fn: Callable[[], BacktestResult], *, repeat: int) -> tuple[float, int]:
    best = float("inf")
    equity_points = 0
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        if elapsed < best:
            best, equity_points = elapsed, result.performance.equity_curve().height
    return best, equity_points


if __name__ == "__main__":
    raise SystemExit(main())
