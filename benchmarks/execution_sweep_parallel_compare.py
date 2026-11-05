from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, time as datetime_time, timedelta
from datetime import UTC
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cortexbacktest import Bar, FactorSweepCase, RebalanceConfig, momentum, run_execution_sweep, run_factor_sweep  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare serial and parallel execution-sweep throughput.")
    parser.add_argument("--dates", type=int, default=250)
    parser.add_argument("--symbols", type=int, default=50)
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    prices = _prices(args.dates, args.symbols)
    bars = _bars(prices)
    factor_sweep = run_factor_sweep(
        prices,
        [
            FactorSweepCase(
                f"mom_{window}",
                lambda frame, window=window: momentum(frame, window=window),
                top_quantile=0.2,
                bottom_quantile=None,
            )
            for window in range(1, args.cases + 1)
        ],
        horizons=(1,),
        quantiles=5,
    )

    print("mode,workers,cases,seconds,cases_per_second,final_equity_sum")
    for mode, workers in (("serial", 1), ("process", args.workers)):
        for _ in range(args.warmup):
            _run(factor_sweep, bars, workers=workers)
        seconds, final_equity = _time_repeated(
            lambda: _run(factor_sweep, bars, workers=workers),
            repeat=args.repeat,
        )
        cases_per_second = args.cases / seconds if seconds > 0 else float("inf")
        print(f"{mode},{workers},{args.cases},{seconds:.6f},{cases_per_second:.2f},{final_equity:.6f}")
    return 0


def _run(factor_sweep, bars, *, workers: int):
    result = run_execution_sweep(
        factor_sweep,
        bars,
        initial_cash=100_000,
        rebalance_config=RebalanceConfig(cash_buffer=0.01, min_trade_value=500.0, allow_fractional=False),
        max_workers=workers,
        parallel_backend="process",
    )
    return result.summary().select(pl.col("event_final_equity").sum()).item()


def _time_repeated(fn, *, repeat: int) -> tuple[float, float]:
    best = float("inf")
    final_equity = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        final_equity = float(fn())
        best = min(best, time.perf_counter() - start)
    return best, final_equity


def _prices(dates: int, symbols: int) -> pl.DataFrame:
    date_values = pl.date_range(
        start=date(2024, 1, 2),
        end=date(2024, 1, 2) + timedelta(days=dates - 1),
        interval="1d",
        eager=True,
    )
    columns = {"date": date_values}
    for idx in range(symbols):
        base = 100.0 + idx
        columns[f"S{idx:04d}"] = [base + day * (0.01 + (idx % 7) * 0.001) + ((day + idx) % 5) * 0.1 for day in range(dates)]
    return pl.DataFrame(columns)


def _bars(prices: pl.DataFrame) -> list[Bar]:
    rows = prices.iter_rows(named=True)
    bars = []
    symbols = [column for column in prices.columns if column != "date"]
    for row in rows:
        timestamp = datetime.combine(row["date"], datetime_time(), tzinfo=UTC)
        for symbol in symbols:
            price = float(row[symbol])
            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=100_000,
                )
            )
    return bars


if __name__ == "__main__":
    raise SystemExit(main())
