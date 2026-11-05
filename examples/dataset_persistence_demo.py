"""Persist and reload a standard backtest dataset directory."""

from __future__ import annotations

import pandas as pd

from cortexbacktest import load_backtest_dataset, load_backtest_dataset_dir


def main() -> None:
    bars = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "adj_close": [100.0, 200.0, 101.0, 198.0],
        }
    )
    factors = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "symbol": ["AAPL", "MSFT"],
            "factor": [1.0, -1.0],
        }
    )
    output_dir = "examples/output/sample_dataset"
    dataset = load_backtest_dataset(bars=bars, factors=factors)
    dataset.save(output_dir, format="csv")
    loaded = load_backtest_dataset_dir(output_dir)

    print("CortexBacktest - dataset persistence demo")
    print(f"bars: {len(loaded.bars)}")
    print(f"factors: {len(loaded.factors)}")
    print(f"prices_shape: {loaded.prices.shape}")


if __name__ == "__main__":
    main()
