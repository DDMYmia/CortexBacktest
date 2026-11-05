"""Where the time goes on real minute bars: storage, feed, engine loop, strategy.

Reads research-layer monthly partitions (timestamp, open, high, low, close,
volume, ticker) through PartitionedBarFeed, the reader production runs use,
and times each layer on its own so the engine loop's share can be read off
directly. The question it answers: would a native (Rust) event loop or
matching core pay for itself, or is the cost elsewhere?

    python benchmarks/minute_partition_profile.py --partition part-2024-01.parquet \
        --symbols 500 --output benchmark-results/minute-profile.json --profile
"""
from __future__ import annotations

import argparse
import cProfile
from collections import defaultdict, deque
import json
from pathlib import Path
import pstats
import resource
import sys
from time import perf_counter

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cortexbacktest.contracts import FillPrice, Order, Side  # noqa: E402
from cortexbacktest.data.bar_stream import PartitionedBarFeed  # noqa: E402
from cortexbacktest.engine import BacktestEngine  # noqa: E402
from cortexbacktest.event.broker import BrokerSimulator  # noqa: E402
from cortexbacktest.strategy import Strategy  # noqa: E402


class Idle(Strategy):
    """Receives every bar and never trades: the engine's own per-bar cost."""

    def on_bar(self, bar):
        return []


class RollingSma(Strategy):
    """A per-bar Python strategy that trades: the cost a real user pays on top."""

    def __init__(self, fast=20, slow=50):
        self.fast, self.slow = fast, slow
        self.windows = defaultdict(deque)
        self.sums = defaultdict(float)
        self.long = defaultdict(bool)

    def on_bar(self, bar):
        values = self.windows[bar.symbol]
        values.append(bar.close)
        self.sums[bar.symbol] += bar.close
        if len(values) > self.slow:
            self.sums[bar.symbol] -= values.popleft()
        if len(values) < self.slow:
            return []
        fast = sum(values[i] for i in range(self.slow - self.fast, self.slow)) / self.fast
        target = fast > self.sums[bar.symbol] / self.slow
        if target == self.long[bar.symbol]:
            return []
        self.long[bar.symbol] = target
        return [Order(bar.symbol, bar.timestamp, Side.BUY if target else Side.SELL, 1)]


def rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--partition", type=Path, action="append", required=True)
    parser.add_argument("--symbols", type=int, default=500, help="most-traded N tickers; 0 for all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true", help="also cProfile the idle engine run")
    args = parser.parse_args()

    scans = [pl.scan_parquet(path).rename({"ticker": "symbol"}) for path in args.partition]
    universe = None
    if args.symbols:
        universe = (pl.concat(scans).group_by("symbol").agg(pl.col("volume").sum())
                    .sort("volume", descending=True).head(args.symbols).collect()["symbol"])

    def partitions():
        for scan in scans:
            yield scan.filter(pl.col("symbol").is_in(universe.to_list())) if universe is not None else scan

    stages: dict[str, dict] = {}

    def timed(name, fn, **extra):
        start = perf_counter()
        value = fn()
        stages[name] = {"seconds": round(perf_counter() - start, 3), "peak_rss_mb": round(rss_mb(), 1), **extra}
        return value

    rows = timed("read_sort", lambda: sum(
        part.sort(["timestamp", "symbol"]).collect(engine="streaming").height for part in partitions()))
    stages["read_sort"]["rows"] = rows
    slices = timed("feed_iterate", lambda: sum(1 for _ in PartitionedBarFeed(partitions)))
    stages["feed_iterate"]["slices"] = slices

    def engine_run(strategy):
        return BacktestEngine(broker=BrokerSimulator(initial_cash=1e9, fill_price=FillPrice.OPEN),
                              trade_on_close=False, result_retention="summary",
                              snapshot_frequency="session").run(PartitionedBarFeed(partitions), strategy)

    timed("engine_idle", lambda: engine_run(Idle()))
    result = timed("engine_sma", lambda: engine_run(RollingSma()))
    stages["engine_sma"]["fills"] = result.fill_count

    feed_s = stages["feed_iterate"]["seconds"]
    idle_s = stages["engine_idle"]["seconds"]
    sma_s = stages["engine_sma"]["seconds"]
    per_bar = {name: round(seconds / rows * 1e6, 3) for name, seconds in (
        ("feed_us", feed_s), ("engine_loop_us", idle_s - feed_s), ("strategy_us", sma_s - idle_s))}
    payload = {"partitions": [str(p) for p in args.partition], "symbols": args.symbols, "rows": rows,
               "stages": stages, "per_bar": per_bar}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(engine_run, Idle())
        text_path = args.output.with_suffix(".profile.txt")
        with text_path.open("w") as handle:
            stats = pstats.Stats(profiler, stream=handle).strip_dirs()
            stats.sort_stats("tottime").print_stats(30)
            stats.sort_stats("cumulative").print_stats(30)
        payload["profile"] = str(text_path)

    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
