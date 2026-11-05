"""A synthetic stress benchmark combining what a single feature test never does.

Every other benchmark in this directory isolates one thing: `extreme_scaling_
compare.py` is throughput and memory at scale with the simplest possible
strategy; `execution_sweep_parallel_compare.py` is serial vs. process
scheduling; `event_engine_compare.py`'s `open_order_scan` scenario is the
open-order index specifically. None of them run sparse multi-symbol data,
long-lived resting limit orders, OCO brackets, trailing stops, and partial
fills *together*, at a symbol count and bar count large enough that a
regression in any one of them would actually show up.

This does. Each run generates:

- `--symbols` symbols, each trading on only `--trade-fraction` of `--bars`
  sessions (sparse, not every symbol every day — real illiquid names look
  like this, and a feed/broker path that assumes dense data will not notice
  a bug here on dense synthetic data).
- A fixed fraction of sessions per symbol open a bracket (market entry +
  OCO limit take-profit / stop loss), attempt a trailing-stop exit, or place
  a limit order far enough off market that it never fills and sits in the
  open-order book for the rest of the run — the accumulate-long-lived-orders
  case `open_order_scan` (a single symbol, single order shape) does not
  cover at multi-symbol scale.
- `max_volume_share` well under 1.0, so most fills are partial across
  several bars rather than immediate and whole.
- One corporate action (a cash dividend) partway through, on the most-observed symbol; actual holdings at that point are not asserted.

Peak RSS is measured in a subprocess (like extreme_scaling_compare.py):
running this in-process would report whichever configuration used the most
memory, attributed to both.

    python benchmarks/large_scale_stress_compare.py
    python benchmarks/large_scale_stress_compare.py --symbols 1000 --bars 10000

The second line is the full scale named in the 2026-09-04 optimization plan
(1,000 symbols, order counts into the hundreds of thousands from long-lived
limits plus brackets, bars in the low millions of bar-observations) — the
default is smaller so this remains something to run routinely rather than
only on request; scale it up deliberately, not as the checked-in default.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cortexbacktest import (  # noqa: E402
    Bar,
    BacktestEngine,
    BrokerSimulator,
    CorporateAction,
    CorporateActionType,
    Order,
    OrderType,
    Side,
    Strategy,
)


class StressStrategy(Strategy):
    """Brackets, trailing stops, and long-lived off-market limits, at random."""

    def __init__(self, *, seed: int, long_lived_limit_fraction: float, bracket_fraction: float, trailing_fraction: float) -> None:
        super().__init__()
        self.rng = random.Random(seed)
        self.long_lived_limit_fraction = long_lived_limit_fraction
        self.bracket_fraction = bracket_fraction
        self.trailing_fraction = trailing_fraction
        self._bracket_serial = 0

    def on_bars(self, bars: list[Bar]) -> list[Order]:
        orders: list[Order] = []
        for bar in bars:
            draw = self.rng.random()
            if draw < self.bracket_fraction:
                self._bracket_serial += 1
                oco = f"{bar.symbol}-{self._bracket_serial}"
                orders.append(Order(bar.symbol, bar.timestamp, Side.BUY, 20))
                orders.append(
                    Order(
                        bar.symbol, bar.timestamp, Side.SELL, 20,
                        order_type=OrderType.LIMIT, limit_price=bar.close * 1.05, oco_group=oco,
                    )
                )
                orders.append(
                    Order(
                        bar.symbol, bar.timestamp, Side.SELL, 20,
                        order_type=OrderType.STOP, stop_price=bar.close * 0.95, oco_group=oco,
                    )
                )
            elif draw < self.bracket_fraction + self.trailing_fraction:
                orders.append(
                    Order(
                        bar.symbol, bar.timestamp, Side.SELL, 10,
                        order_type=OrderType.TRAILING_STOP, trail_percent=0.03,
                    )
                )
            elif draw < self.bracket_fraction + self.trailing_fraction + self.long_lived_limit_fraction:
                # Half market, well clear of any plausible fill: accumulates
                # in the open-order book for the rest of the run.
                orders.append(
                    Order(bar.symbol, bar.timestamp, Side.BUY, 1, order_type=OrderType.LIMIT, limit_price=bar.close * 0.5)
                )
        return orders


def _generate_bars(symbols: int, bars: int, trade_fraction: float, seed: int) -> list[Bar]:
    rng = random.Random(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    generated: list[Bar] = []
    for symbol_index in range(symbols):
        symbol = f"S{symbol_index:05d}"
        price = 50.0 + rng.random() * 150.0
        for day in range(bars):
            if rng.random() > trade_fraction:
                continue  # sparse: this symbol did not trade this session
            drift = (rng.random() - 0.48) * 0.03
            price = max(0.5, price * (1 + drift))
            spread = price * 0.01
            generated.append(
                Bar(
                    symbol=symbol,
                    timestamp=start + timedelta(days=day),
                    open=price,
                    high=price + spread,
                    low=max(0.01, price - spread),
                    close=price,
                    volume=1_000 + rng.random() * 9_000,
                )
            )
    return generated


def _dividend_action(bars: list[Bar]) -> CorporateAction:
    # Choose the most-observed symbol; this does not prove an open position.
    counts = Counter(bar.symbol for bar in bars)
    symbol = max(counts, key=lambda name: counts[name])
    midpoint = sorted({bar.timestamp for bar in bars if bar.symbol == symbol})[counts[symbol] // 2]
    return CorporateAction(symbol, midpoint.date(), CorporateActionType.CASH_DIVIDEND, amount=0.25)


def _run(
    symbols: int,
    bars: int,
    trade_fraction: float,
    long_lived_limit_fraction: float,
    bracket_fraction: float,
    trailing_fraction: float,
    max_volume_share: float,
    seed: int,
) -> dict:
    generated = _generate_bars(symbols, bars, trade_fraction, seed)
    dividend = _dividend_action(generated)
    broker = BrokerSimulator(
        initial_cash=10_000_000.0,
        max_volume_share=max_volume_share,
        corporate_actions=[dividend],
    )
    strategy = StressStrategy(
        seed=seed,
        long_lived_limit_fraction=long_lived_limit_fraction,
        bracket_fraction=bracket_fraction,
        trailing_fraction=trailing_fraction,
    )
    engine = BacktestEngine(broker=broker, snapshot_frequency="final", result_retention="summary", build_manifest=False)

    started = time.perf_counter()
    result = engine.run(generated, strategy)
    elapsed = time.perf_counter() - started

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    summary = result.summary()
    return {
        "symbols": symbols,
        "bars_requested": bars,
        "bar_observations": len(generated),
        "seconds": elapsed,
        "bar_observations_per_second": len(generated) / elapsed if elapsed > 0 else float("inf"),
        "peak_rss_mb": peak_mb,
        "open_orders": result.open_order_count,
        "orders": summary["orders"],
        "fills": summary["fills"],
        "final_equity": summary["final_equity"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", type=int, default=200)
    parser.add_argument("--bars", type=int, default=1_000, help="Sessions per symbol (before sparsity).")
    parser.add_argument("--trade-fraction", type=float, default=0.6, help="Fraction of sessions a symbol actually trades.")
    parser.add_argument("--bracket-fraction", type=float, default=0.03, help="Per-bar chance of opening an OCO bracket.")
    parser.add_argument("--trailing-fraction", type=float, default=0.02, help="Per-bar chance of a trailing-stop order.")
    parser.add_argument("--long-lived-limit-fraction", type=float, default=0.05, help="Per-bar chance of a resting off-market limit.")
    parser.add_argument("--max-volume-share", type=float, default=0.1, help="Caps fill size, forcing most fills to be partial.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--subprocess", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.subprocess:
        # Re-invoke isolated so peak RSS is this run's alone.
        command = [sys.executable, __file__, "--subprocess", *sys.argv[1:]]
        completed = subprocess.run(command, capture_output=True, text=True)
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        return completed.returncode

    row = _run(
        args.symbols,
        args.bars,
        args.trade_fraction,
        args.long_lived_limit_fraction,
        args.bracket_fraction,
        args.trailing_fraction,
        args.max_volume_share,
        args.seed,
    )
    print(json.dumps(row, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(row, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
