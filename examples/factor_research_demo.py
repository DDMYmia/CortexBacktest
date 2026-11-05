"""Run a complete cross-sectional factor research demo."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from cortexbacktest import run_factor_research
if __package__:
    from ._demo_data import business_dates as _business_days
else:
    from _demo_data import business_dates as _business_days


def build_demo_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(7)
    dates = _business_days(date(2024, 1, 2), 180)
    symbols = [f"STK{i:02d}" for i in range(30)]

    latent_quality = rng.normal(0.0, 1.0, len(symbols))
    noise = rng.normal(0.0, 0.012, (len(dates), len(symbols)))
    daily_returns = 0.0002 + 0.0015 * latent_quality + noise
    prices = 100.0 * np.cumprod(1.0 + daily_returns, axis=0)

    factor_noise = rng.normal(0.0, 0.8, (len(dates), len(symbols)))
    factors = latent_quality + factor_noise

    price_frame = pl.DataFrame({"date": dates, **{symbol: prices[:, i] for i, symbol in enumerate(symbols)}})
    factor_frame = pl.DataFrame({"date": dates, **{symbol: factors[:, i] for i, symbol in enumerate(symbols)}})
    return price_frame, factor_frame


def main() -> None:
    prices, factors = build_demo_data()
    report = run_factor_research(
        prices,
        factors,
        horizons=(1, 5, 20),
        quantiles=5,
        top_quantile=0.2,
        bottom_quantile=0.2,
        transaction_cost_bps=5.0,
    )
    result = report.horizon_results[5]

    print("CortexBacktest - factor research demo")
    print("multi_horizon_summary:")
    print(report.summary())
    print()
    print(f"mean_coverage: {report.coverage_mean:.4f}")
    print(f"mean_turnover: {report.turnover_mean:.4f}")
    print(f"mean_ic: {result.mean_ic:.4f}")
    print(f"mean_rank_ic: {result.mean_rank_ic:.4f}")
    print(f"mean_top_bottom_spread_5d: {result.mean_spread:.4f}")
    print()
    print("portfolio:")
    print(f"final_equity: {report.final_equity:.4f}")


if __name__ == "__main__":
    main()
