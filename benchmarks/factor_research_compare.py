"""Compare CortexBacktest rank IC with Alphalens Reloaded (`pip install alphalens-reloaded`)."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cortexbacktest.research import factor_information_coefficient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", type=int, default=500)
    parser.add_argument("--symbols", type=int, default=500)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    fusion_data, alphalens_data = make_factor_data(args.dates, args.symbols, args.quantiles)
    rows = [
        run_case(
            "fusion_polars",
            lambda: run_fusion(fusion_data),
            observations=fusion_data.height,
            repeat=args.repeat,
            warmup=args.warmup,
        ),
        run_case(
            "alphalens_reloaded",
            lambda: run_alphalens(alphalens_data),
            observations=fusion_data.height,
            repeat=args.repeat,
            warmup=args.warmup,
        ),
    ]
    print("engine,available,seconds,observations_per_second,mean_rank_ic,reason")
    for row in rows:
        seconds = "" if row["seconds"] is None else f"{row['seconds']:.6f}"
        throughput = "" if row["throughput"] is None else f"{row['throughput']:.2f}"
        rank_ic = "" if row["mean_rank_ic"] is None else f"{row['mean_rank_ic']:.12f}"
        reason = row["reason"].replace(",", ";") if row["reason"] else ""
        print(f"{row['engine']},{row['available']},{seconds},{throughput},{rank_ic},{reason}")
    return 0


def make_factor_data(dates: int, symbols: int, quantiles: int) -> tuple[pl.DataFrame, pd.DataFrame]:
    if dates < 1 or symbols < quantiles:
        raise ValueError("dates must be positive and symbols must be >= quantiles")
    rng = np.random.default_rng(7)
    date_index = pd.bdate_range("2020-01-01", periods=dates)
    assets = [f"S{index:05d}" for index in range(symbols)]
    factor = rng.normal(size=(dates, symbols))
    forward = factor * 0.02 + rng.normal(scale=0.01, size=(dates, symbols))
    long = pl.DataFrame(
        {
            "date": np.repeat(date_index.to_numpy(), symbols),
            "symbol": np.tile(assets, dates),
            "factor": factor.ravel(),
            "forward_return": forward.ravel(),
        }
    )
    index = pd.MultiIndex.from_product([date_index, assets], names=["date", "asset"])
    reference = pd.DataFrame(
        {
            "factor": factor.ravel(),
            "1D": forward.ravel(),
            "factor_quantile": np.tile(np.arange(symbols) * quantiles // symbols + 1, dates),
        },
        index=index,
    )
    return long, reference


def run_fusion(data: pl.DataFrame) -> float:
    result = factor_information_coefficient(data, method="spearman")
    value = result["rank_ic"].mean()
    return float(value) if value is not None else float("nan")


def run_alphalens(data: pd.DataFrame) -> float:
    performance = importlib.import_module("alphalens.performance")
    result = performance.factor_information_coefficient(data)
    return float(result["1D"].mean())


def run_case(engine: str, fn, *, observations: int, repeat: int, warmup: int) -> dict:
    try:
        for _ in range(warmup):
            fn()
        durations = []
        values = []
        for _ in range(repeat):
            start = time.perf_counter()
            values.append(fn())
            durations.append(time.perf_counter() - start)
        seconds = mean(durations)
        return {
            "engine": engine,
            "available": True,
            "seconds": seconds,
            "throughput": observations / seconds if seconds else None,
            "mean_rank_ic": mean(values),
            "reason": None,
        }
    except Exception as exc:
        return {
            "engine": engine,
            "available": False,
            "seconds": None,
            "throughput": None,
            "mean_rank_ic": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }


if __name__ == "__main__":
    raise SystemExit(main())
