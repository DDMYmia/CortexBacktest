"""Compare factor parameter configurations with a research sweep."""

from __future__ import annotations

import polars as pl
if __package__:
    from ._demo_data import parsed_business_dates as _business_days
else:
    from _demo_data import parsed_business_dates as _business_days

from cortexbacktest import (
    FactorPreprocessConfig,
    FactorSweepCase,
    factor_decay_linear,
    momentum,
    run_factor_sweep,
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

    preprocessing = FactorPreprocessConfig(
        winsorize_quantiles=(0.05, 0.95),
        zscore=True,
    )
    cases = [
        FactorSweepCase(
            "mom_5",
            lambda frame: momentum(frame, window=5),
            params={"window": 5, "decay": "none"},
            preprocessing=preprocessing,
            transaction_cost_bps=2.0,
        ),
        FactorSweepCase(
            "mom_10",
            lambda frame: momentum(frame, window=10),
            params={"window": 10, "decay": "none"},
            preprocessing=preprocessing,
            transaction_cost_bps=2.0,
        ),
        FactorSweepCase(
            "mom_10_decay_3",
            lambda frame: factor_decay_linear(momentum(frame, window=10), window=3, min_periods=1),
            params={"window": 10, "decay": "linear_3"},
            preprocessing=preprocessing,
            transaction_cost_bps=2.0,
        ),
    ]

    result = run_factor_sweep(prices, cases, horizons=(1, 5, 10), quantiles=3)

    print("CortexBacktest - factor sweep demo")
    print(result.summary())
    print()
    print(result.stability())
    print(f"best_case: {result.best_case()}")


if __name__ == "__main__":
    main()
