"""Run factor research with winsorization, group neutralization, and z-scoring."""

from __future__ import annotations

import polars as pl

from cortexbacktest import FactorPreprocessConfig, momentum, run_factor_research
if __package__:
    from ._demo_data import parsed_business_dates as _business_days
else:
    from _demo_data import parsed_business_dates as _business_days



def main() -> None:
    dates = _business_days("2024-01-01", 60)
    prices = pl.DataFrame(
        {
            "date": dates,
            "AAPL": [100 + i * 0.80 for i in range(len(dates))],
            "MSFT": [100 + i * 0.45 for i in range(len(dates))],
            "NVDA": [100 + i * 1.10 for i in range(len(dates))],
            "XOM": [100 - i * 0.20 for i in range(len(dates))],
            "CVX": [100 - i * 0.10 for i in range(len(dates))],
            "JPM": [100 + (-1) ** i * 0.3 + i * 0.1 for i in range(len(dates))],
        }
    )
    raw_factor = momentum(prices, window=10)
    sectors = {
        "AAPL": "technology",
        "MSFT": "technology",
        "NVDA": "technology",
        "XOM": "energy",
        "CVX": "energy",
        "JPM": "financials",
    }

    preprocessing = FactorPreprocessConfig(
        winsorize_quantiles=(0.05, 0.95),
        neutralize_groups=sectors,
        zscore=True,
    )
    report = run_factor_research(
        prices,
        raw_factor,
        preprocessing=preprocessing,
        horizons=(1, 5),
        quantiles=3,
        top_quantile=1 / 3,
        bottom_quantile=1 / 3,
        transaction_cost_bps=2.0,
    )

    print("CortexBacktest - factor preprocessing demo")
    print(report.summary())
    print(f"coverage: {report.coverage_mean:.4f}")
    print(f"turnover: {report.turnover_mean:.4f}")
    print(f"final_equity: {report.final_equity:.4f}")


if __name__ == "__main__":
    main()
