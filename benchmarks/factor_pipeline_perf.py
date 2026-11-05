"""Time the cross-sectional factor pipeline on a panel of realistic size.

The panel is synthetic but shaped like a real equity universe: a few thousand
symbols over about ten years of business days, with symbols that list late
(leading nulls), delist early (trailing nulls) and a heavy-tailed return
distribution. Factors are built in Polars from the prices (momentum, short
reversal, low volatility, and a noisy latent-quality signal), so the timing
covers factor construction as well as `run_factor_research`.

    python benchmarks/factor_pipeline_perf.py --symbols 1500 --dates 2520 \
        --output benchmark-results/factor-pipeline/<label>.json

Results are wall-clock seconds for the median-of-repeats, plus the process peak
RSS. Run each Polars version in its own interpreter; `ru_maxrss` is a
process-lifetime high-water mark.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cortexbacktest import run_factor_research  # noqa: E402

FACTOR_NAMES = ("momentum_252", "reversal_5", "low_vol_60", "latent_quality")


def business_dates(start: date, count: int) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def build_panel(symbols: int, dates: int, seed: int = 20260101) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """Return (prices, factors) as wide frames: a `date` column plus one column per symbol."""
    rng = np.random.default_rng(seed)
    days = business_dates(date(2016, 1, 4), dates)
    names = [f"S{i:05d}" for i in range(symbols)]

    quality = rng.normal(0.0, 1.0, symbols)
    drift = 0.00015 + 0.0004 * quality
    vol = rng.uniform(0.008, 0.025, symbols)
    # Student-t innovations: fat tails without the cost of a regime model.
    shocks = rng.standard_t(df=4, size=(dates, symbols)) / np.sqrt(2.0)
    returns = drift + vol * shocks
    prices = 50.0 * np.exp(np.cumsum(returns, axis=0))

    # Realistic listing history: ~10% list late, ~5% delist early.
    listed_late = rng.uniform(size=symbols) < 0.10
    list_day = np.where(listed_late, rng.integers(1, dates // 2, symbols), 0)
    delisted = rng.uniform(size=symbols) < 0.05
    delist_day = np.where(delisted, rng.integers(dates // 2, dates, symbols), dates)
    day_index = np.arange(dates)[:, None]
    alive = (day_index >= list_day[None, :]) & (day_index < delist_day[None, :])
    prices = np.where(alive, prices, np.nan)

    price_frame = pl.DataFrame({"date": days, **{n: prices[:, i] for i, n in enumerate(names)}})
    latent = quality[None, :] + rng.normal(0.0, 0.8, size=(dates, symbols))
    latent_frame = pl.DataFrame({"date": days, **{n: latent[:, i] for i, n in enumerate(names)}})

    # Factor construction in Polars, same shape as the inputs.
    long = price_frame.unpivot(index="date", variable_name="symbol", value_name="close").sort("symbol", "date")
    long = long.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0).alias("ret"),
    )
    factored = long.with_columns(
        (pl.col("close") / pl.col("close").shift(252).over("symbol") - 1.0).alias("momentum_252"),
        (-(pl.col("close") / pl.col("close").shift(5).over("symbol") - 1.0)).alias("reversal_5"),
        (-pl.col("ret").rolling_std(60).over("symbol")).alias("low_vol_60"),
    )
    factors: dict[str, pl.DataFrame] = {}
    for name in ("momentum_252", "reversal_5", "low_vol_60"):
        factors[name] = factored.pivot(on="symbol", index="date", values=name).sort("date")
    factors["latent_quality"] = latent_frame
    return price_frame, factors


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def run_once(prices: pl.DataFrame, factors: dict[str, pl.DataFrame]) -> dict[str, float]:
    per_factor: dict[str, float] = {}
    summary: dict[str, float] = {}
    for name in FACTOR_NAMES:
        start = time.perf_counter()
        report = run_factor_research(
            prices,
            factors[name],
            horizons=(1, 5, 20),
            quantiles=5,
            top_quantile=0.2,
            bottom_quantile=0.2,
            transaction_cost_bps=5.0,
        )
        per_factor[name] = time.perf_counter() - start
        summary[f"{name}.mean_rank_ic_5d"] = float(report.horizon_results[5].mean_rank_ic)
    return {"total_seconds": sum(per_factor.values()), **{f"seconds.{k}": v for k, v in per_factor.items()}, **summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=int, default=1500)
    parser.add_argument("--dates", type=int, default=2520)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import polars

    build_start = time.perf_counter()
    prices, factors = build_panel(args.symbols, args.dates)
    build_seconds = time.perf_counter() - build_start

    for _ in range(args.warmup):
        run_once(prices, factors)

    runs = [run_once(prices, factors) for _ in range(args.repeat)]
    totals = [run["total_seconds"] for run in runs]
    cells = args.symbols * args.dates * len(FACTOR_NAMES)
    result = {
        "polars": polars.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "symbols": args.symbols,
        "dates": args.dates,
        "factors": list(FACTOR_NAMES),
        "panel_build_seconds": build_seconds,
        "repeat": args.repeat,
        "total_seconds_median": statistics.median(totals),
        "total_seconds_all": totals,
        "cells_per_second_median": cells / statistics.median(totals),
        "peak_rss_mb": _peak_rss_mb(),
        "last_run_metrics": runs[-1],
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
