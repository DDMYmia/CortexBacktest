"""Compare combined input/snapshot modes for target-weight rebalance.

List input uses per-bar snapshots; feed input uses one final snapshot. Timing
differences do not isolate either feed materialization or snapshot overhead.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean

import polars as pl

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cortexbacktest import Bar  # noqa: E402
from cortexbacktest.event import (  # noqa: E402
    BacktestEngine,
    BrokerSimulator,
    ColumnarBarFeed,
    RebalanceConfig,
    TargetWeightStrategy,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare target-weight rebalance execution modes.")
    parser.add_argument("--dates", type=int, default=500)
    parser.add_argument("--symbols", type=int, default=100)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    bars = make_bars(args.dates, args.symbols)
    feed = ColumnarBarFeed.from_bars(bars, assume_sorted=True)
    weights = make_target_weights(args.dates, args.symbols, args.top)
    modes = [
        ("list_bar_snapshots", lambda: run_once(bars, weights)),
        ("feed_final_snapshot", lambda: run_once(feed, weights, snapshot_frequency="final")),
    ]

    print("mode,seconds,bars_per_second,fills,orders,final_equity")
    total_bars = args.dates * args.symbols
    for name, fn in modes:
        for _ in range(args.warmup):
            fn()
        timings = []
        last = None
        for _ in range(args.repeat):
            start = time.perf_counter()
            last = fn()
            timings.append(time.perf_counter() - start)
        seconds = mean(timings)
        assert last is not None
        print(
            f"{name},{seconds:.6f},{total_bars / seconds:.2f},"
            f"{len(last.fills)},{len(last.orders)},{last.summary()['final_equity']:.6f}"
        )
    return 0


def run_once(bars, weights, *, snapshot_frequency: str = "bar"):
    broker = BrokerSimulator(initial_cash=1_000_000.0)
    strategy = TargetWeightStrategy(
        weights,
        broker=broker,
        config=RebalanceConfig(allow_fractional=False, min_trade_value=100.0),
    )
    return BacktestEngine(broker=broker, snapshot_frequency=snapshot_frequency).run(
        bars,
        strategy,
        assume_sorted=True,
    )


def make_bars(dates: int, symbols: int) -> list[Bar]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = []
    for day in range(dates):
        timestamp = start + timedelta(days=day)
        for symbol_index in range(symbols):
            price = 100.0 + symbol_index * 0.1 + day * 0.01
            bars.append(
                Bar(
                    f"S{symbol_index:04d}",
                    timestamp,
                    price,
                    price * 1.01,
                    price * 0.99,
                    price,
                    volume=1_000_000,
                )
            )
    return bars


def make_target_weights(dates: int, symbols: int, top: int) -> pl.DataFrame:
    top = max(1, min(top, symbols))
    start = datetime(2024, 1, 1)
    rows = []
    for day in range(dates):
        active_start = day % symbols
        active = {(active_start + offset) % symbols for offset in range(top)}
        row = {"date": start + timedelta(days=day)}
        for symbol_index in range(symbols):
            row[f"S{symbol_index:04d}"] = 1.0 / top if symbol_index in active else 0.0
        rows.append(row)
    return pl.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
