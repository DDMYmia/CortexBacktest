"""Normalize yfinance-style data and run a dataset workflow."""

from __future__ import annotations

import pandas as pd
import polars as pl

from cortexbacktest import load_backtest_dataset, normalize_yfinance_history


def main() -> None:
    history = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [102, 103, 105],
            "Low": [99, 100, 101],
            "Close": [101, 102, 104],
            "Adj Close": [100.5, 101.5, 103.5],
            "Volume": [1_000_000, 1_100_000, 1_200_000],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    bars = normalize_yfinance_history(history, symbols="AAPL")
    factors = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "symbol": ["AAPL", "AAPL"],
            "factor": [1.0, 1.1],
        }
    )
    dataset = load_backtest_dataset(bars=bars, factors=factors)

    print("CortexBacktest - yfinance source demo")
    print(f"bars: {len(bars)}")
    print(f"symbols: {[c for c in dataset.prices.columns if c != 'date']}")
    print(f"latest_price: {dataset.prices.drop('date').row(-1)[0]:.2f}")


if __name__ == "__main__":
    main()
