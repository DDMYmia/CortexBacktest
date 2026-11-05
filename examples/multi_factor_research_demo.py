"""Build a composite factor and run cross-sectional research."""

from __future__ import annotations

import polars as pl
if __package__:
    from ._demo_data import parsed_business_dates as _business_days
else:
    from _demo_data import parsed_business_dates as _business_days

from cortexbacktest import (
    FactorPreprocessConfig,
    combine_factor_scores,
    factor_decay_linear,
    momentum,
    rolling_volatility,
    run_factor_research,
    sma_ratio,
)



def main() -> None:
    dates = _business_days("2024-01-01", 80)
    prices = pl.DataFrame(
        {
            "date": dates,
            "AAPL": [100 + i * 0.80 for i in range(len(dates))],
            "MSFT": [100 + i * 0.50 for i in range(len(dates))],
            "NVDA": [100 + i * 1.10 for i in range(len(dates))],
            "XOM": [100 - i * 0.15 for i in range(len(dates))],
            "CVX": [100 - i * 0.05 for i in range(len(dates))],
            "JPM": [100 + (-1) ** i * 0.4 + i * 0.12 for i in range(len(dates))],
        }
    )

    raw_momentum = momentum(prices, window=10)
    trend = sma_ratio(prices, window=20)
    volatility = rolling_volatility(prices, window=10, annualization=252**0.5)
    # Lower volatility is a better signal, so flip the sign of the value columns.
    low_volatility = volatility.with_columns(pl.exclude("date") * -1)

    preprocessing = FactorPreprocessConfig(
        winsorize_quantiles=(0.05, 0.95),
        zscore=True,
    )
    composite = combine_factor_scores(
        {
            "momentum": factor_decay_linear(raw_momentum, window=3, min_periods=1),
            "trend": trend,
            "low_volatility": low_volatility,
        },
        weights={"momentum": 0.5, "trend": 0.3, "low_volatility": 0.2},
        preprocessing=preprocessing,
        min_count=2,
    )
    report = run_factor_research(
        prices,
        composite,
        horizons=(1, 5, 10),
        quantiles=3,
        top_quantile=1 / 3,
        bottom_quantile=1 / 3,
        transaction_cost_bps=2.0,
    )

    print("CortexBacktest - multi-factor research demo")
    print(report.summary())
    print(f"coverage: {report.coverage_mean:.4f}")
    print(f"turnover: {report.turnover_mean:.4f}")
    print(f"final_equity: {report.final_equity:.4f}")


if __name__ == "__main__":
    main()
