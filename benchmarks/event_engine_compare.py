"""Time CortexBacktest's event engine on synthetic order-flow scenarios.

Other engines are compared on real data in `benchmarks/real_data_comparison/`.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cortexbacktest import BacktestEngine, Bar, Order, OrderType, Side, Strategy  # noqa: E402
from cortexbacktest.event import BrokerSimulator  # noqa: E402

SCENARIOS = ("buy_once", "multi_asset", "multi_order", "order_types", "partial_fill", "open_order_scan")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    scenario: str
    name: str
    available: bool
    seconds: float | None = None
    bars_per_second: float | None = None
    reason: str | None = None


class BuyFirstSlice(Strategy):
    def __init__(self) -> None:
        self.ordered = False

    def on_bars(self, bars: list[Bar]) -> list[Order]:
        if self.ordered:
            return []
        self.ordered = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 1) for bar in bars]


class MultiOrderStrategy(Strategy):
    def on_bars(self, bars: list[Bar]) -> list[Order]:
        orders = []
        for bar in bars:
            if bar.timestamp.minute % 5 == 0:
                orders.append(Order(bar.symbol, bar.timestamp, Side.BUY, 1))
            elif bar.timestamp.minute % 5 == 3:
                position = self.position(bar.symbol)
                if position is not None and position.quantity > 0:
                    orders.append(Order(bar.symbol, bar.timestamp, Side.SELL, min(position.quantity, 1)))
        return orders


class OrderTypesStrategy(Strategy):
    def __init__(self) -> None:
        self.ordered = False

    def on_bars(self, bars: list[Bar]) -> list[Order]:
        if self.ordered:
            return []
        self.ordered = True
        orders = []
        for index, bar in enumerate(bars):
            if index % 3 == 0:
                orders.append(
                    Order(
                        bar.symbol,
                        bar.timestamp,
                        Side.BUY,
                        1,
                        order_type=OrderType.LIMIT,
                        limit_price=bar.close + 0.01,
                    )
                )
            elif index % 3 == 1:
                orders.append(
                    Order(
                        bar.symbol,
                        bar.timestamp,
                        Side.BUY,
                        1,
                        order_type=OrderType.STOP,
                        stop_price=bar.close - 0.01,
                    )
                )
            else:
                orders.append(
                    Order(
                        bar.symbol,
                        bar.timestamp,
                        Side.BUY,
                        1,
                        order_type=OrderType.STOP_LIMIT,
                        stop_price=bar.close - 0.01,
                        limit_price=bar.close + 0.01,
                    )
                )
        return orders


class PartialFillStrategy(Strategy):
    def __init__(self) -> None:
        self.ordered = False

    def on_bars(self, bars: list[Bar]) -> list[Order]:
        if self.ordered:
            return []
        self.ordered = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 80) for bar in bars]


class OpenOrderScanStrategy(Strategy):
    def __init__(self) -> None:
        self.ordered = False

    def on_bars(self, bars: list[Bar]) -> list[Order]:
        if self.ordered:
            return []
        self.ordered = True
        return [
            Order(
                bar.symbol,
                bar.timestamp,
                Side.BUY,
                1,
                order_type=OrderType.LIMIT,
                limit_price=bar.low * 0.01 - index,
            )
            for bar in bars
            for index in range(10)
        ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=10_000, help="Bars per symbol.")
    parser.add_argument("--symbols", type=int, default=10, help="Fusion symbols.")
    parser.add_argument("--repeat", type=int, default=3, help="Runs per engine.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs before timing.")
    parser.add_argument(
        "--scenario",
        choices=("all", *SCENARIOS),
        default="buy_once",
        help="Benchmark scenario to run.",
    )
    args = parser.parse_args()

    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    results = []
    for scenario in scenarios:
        results.append(
            run_benchmark(
                scenario,
                "fusion_event",
                lambda scenario=scenario: run_fusion(args.bars, args.symbols, scenario),
                total_bars=args.bars * args.symbols,
                repeat=args.repeat,
                warmup=args.warmup,
            )
        )

    print("scenario,engine,available,seconds,bars_per_second,reason")
    for result in results:
        seconds = "" if result.seconds is None else f"{result.seconds:.6f}"
        bps = "" if result.bars_per_second is None else f"{result.bars_per_second:.2f}"
        reason = "" if result.reason is None else result.reason.replace(",", ";")
        print(f"{result.scenario},{result.name},{result.available},{seconds},{bps},{reason}")
    return 0


def run_benchmark(
    scenario: str,
    name: str,
    fn: Callable[[], None],
    *,
    total_bars: int,
    repeat: int,
    warmup: int,
) -> BenchmarkResult:
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    seconds = mean(durations)
    return BenchmarkResult(
        scenario=scenario,
        name=name,
        available=True,
        seconds=seconds,
        bars_per_second=total_bars / seconds if seconds else None,
    )


def run_fusion(bar_count: int, symbol_count: int, scenario: str) -> None:
    bars = make_fusion_bars(bar_count, symbol_count, volume=100 if scenario == "partial_fill" else 1_000_000)
    broker = BrokerSimulator(max_volume_share=0.25 if scenario == "partial_fill" else 1.0)
    strategy = {
        "buy_once": BuyFirstSlice,
        "multi_asset": BuyFirstSlice,
        "multi_order": MultiOrderStrategy,
        "order_types": OrderTypesStrategy,
        "partial_fill": PartialFillStrategy,
        "open_order_scan": OpenOrderScanStrategy,
    }[scenario]()
    BacktestEngine(broker=broker).run(bars, strategy)


def make_fusion_bars(bar_count: int, symbol_count: int, *, volume: float = 1_000_000) -> list[Bar]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = []
    for i in range(bar_count):
        timestamp = start + timedelta(minutes=i)
        for symbol_index in range(symbol_count):
            price = 100.0 + i * 0.01 + symbol_index
            bars.append(
                Bar(
                    symbol=f"S{symbol_index:04d}",
                    timestamp=timestamp,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=volume,
                )
            )
    return bars


if __name__ == "__main__":
    raise SystemExit(main())
