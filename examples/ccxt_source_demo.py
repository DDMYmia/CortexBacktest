"""Normalize ccxt-style OHLCV rows and run a dataset workflow."""

from __future__ import annotations

import pandas as pd

from cortexbacktest import load_backtest_dataset, normalize_ccxt_ohlcv


def main() -> None:
    rows = [
        [1_704_067_200_000, 42_000, 43_000, 41_000, 42_500, 123.4],
        [1_704_153_600_000, 42_500, 44_000, 42_000, 43_500, 234.5],
        [1_704_240_000_000, 43_500, 45_000, 43_000, 44_500, 345.6],
    ]
    bars = normalize_ccxt_ohlcv(rows, symbol="BTC/USDT")
    factors = pd.DataFrame(
        {
            "date": bars["date"].iloc[:2],
            "symbol": ["BTC/USDT", "BTC/USDT"],
            "factor": [1.0, 1.2],
        }
    )
    dataset = load_backtest_dataset(bars=bars, factors=factors)

    print("CortexBacktest - ccxt source demo")
    print(f"bars: {len(bars)}")
    print(f"symbols: {[c for c in dataset.prices.columns if c != 'date']}")
    print(f"latest_price: {dataset.prices.drop('date').row(-1)[0]:.2f}")


if __name__ == "__main__":
    main()
