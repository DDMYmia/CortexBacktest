"""Run a factor sweep and execute each case through the event engine."""

from __future__ import annotations

import polars as pl

if __package__:
    from ._demo_data import business_date_strings as _business_days, flat_bars as build_bars
else:
    from _demo_data import business_date_strings as _business_days, flat_bars as build_bars



from cortexbacktest import (
    FactorSweepCase,
    RebalanceConfig,
    factor_decay_linear,
    momentum,
    run_execution_sweep,
    run_factor_sweep,
)

def build_prices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": _business_days("2024-01-02", 8),
            "AAPL": [100.0, 102.0, 103.0, 104.0, 106.0, 108.0, 109.0, 111.0],
            "MSFT": [100.0, 99.0, 98.0, 97.0, 98.0, 99.0, 98.0, 97.0],
            "NVDA": [100.0, 101.0, 104.0, 108.0, 111.0, 115.0, 117.0, 120.0],
            "XOM": [100.0, 100.0, 99.0, 99.0, 98.0, 98.0, 97.0, 97.0],
        }
    )


def main() -> None:
    prices = build_prices()
    factor_sweep = run_factor_sweep(
        prices,
        [
            FactorSweepCase(
                "mom_1",
                lambda frame: momentum(frame, window=1),
                params={"window": 1, "decay": "none"},
                top_quantile=0.25,
                bottom_quantile=None,
            ),
            FactorSweepCase(
                "mom_3_decay",
                lambda frame: factor_decay_linear(momentum(frame, window=3), window=2, min_periods=1),
                params={"window": 3, "decay": "linear_2"},
                top_quantile=0.25,
                bottom_quantile=None,
            ),
        ],
        horizons=(1,),
        quantiles=4,
    )
    execution = run_execution_sweep(
        factor_sweep,
        build_bars(prices),
        initial_cash=100_000,
        commission_bps=1.0,
        slippage_bps=1.0,
        rebalance_config=RebalanceConfig(cash_buffer=0.01, min_trade_value=500.0, allow_fractional=False),
    )

    print("CortexBacktest - research to execution sweep demo")
    print(execution.summary())
    print(f"best_case: {execution.best_case()}")


if __name__ == "__main__":
    main()
