"""Load a standard BacktestDataset and run factor research."""

from __future__ import annotations

import polars as pl

from cortexbacktest import load_backtest_dataset


def main() -> None:
    bars = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
            "adj_close": [100, 100, 103, 98, 106, 96],
        }
    )
    factors = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "factor": [1.0, -1.0, 1.0, -1.0],
        }
    )
    membership = pl.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "start": ["2024-01-01", "2024-01-01"],
            "end": [None, None],
        }
    )

    dataset = load_backtest_dataset(bars=bars, factors=factors, membership=membership)
    report = dataset.run_factor_research(horizons=(1,), quantiles=2)

    print("CortexBacktest - dataset loader demo")
    print(f"prices_shape: {dataset.prices.shape}")
    print(f"factor_shape: {dataset.factor_values.shape}")
    print(f"calendar_sessions: {len(dataset.calendar.sessions)}")
    symbols = [c for c in dataset.universe.mask.columns if c != "date"] if dataset.universe else []
    print(f"universe_symbols: {symbols}")
    print(report.summary())


if __name__ == "__main__":
    main()
