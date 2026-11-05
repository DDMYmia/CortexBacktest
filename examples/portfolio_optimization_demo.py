"""Build lightweight optimized weights and simulate a portfolio."""

from __future__ import annotations

import polars as pl

from cortexbacktest import (
    apply_volatility_target,
    inverse_volatility_weights,
    returns_matrix,
    simulate_target_weights,
)


def main() -> None:
    dates = [f"2024-01-{day:02d}" for day in (2, 3, 4, 5, 8, 9, 10, 11)]
    prices = pl.DataFrame(
        {
            "date": dates,
            "LOW_VOL": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "HIGH_VOL": [100.0, 110.0, 95.0, 115.0, 90.0, 120.0, 85.0, 125.0],
        }
    )
    returns = returns_matrix(prices)
    weights = inverse_volatility_weights(returns, window=3, min_periods=2)
    base_equity = simulate_target_weights(returns, weights)

    portfolio_returns = base_equity.select(
        pl.col("date"), pl.col("equity").pct_change().alias("return")
    )
    scaled_weights = apply_volatility_target(
        weights,
        portfolio_returns,
        target_volatility=0.10,
        window=3,
        max_leverage=1.0,
    )
    scaled_equity = simulate_target_weights(returns, scaled_weights)

    print("CortexBacktest - portfolio optimization demo")
    print("latest weights")
    print(weights.tail(1))
    print(f"base_final_equity: {base_equity['equity'][-1]:.4f}")
    print(f"scaled_final_equity: {scaled_equity['equity'][-1]:.4f}")


if __name__ == "__main__":
    main()
