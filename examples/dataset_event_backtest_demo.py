"""Run factor research and event backtesting from one standard dataset."""

from __future__ import annotations

import polars as pl

from cortexbacktest import Bar, Order, Side, Strategy, load_backtest_dataset


class BuyStrongestFirstDay(Strategy):
    def __init__(self, first_day_scores: dict) -> None:
        self.symbol = max(first_day_scores, key=first_day_scores.get)
        self.bought = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if self.bought or bar.symbol != self.symbol:
            return []
        self.bought = True
        return [Order(symbol=bar.symbol, timestamp=bar.timestamp, side=Side.BUY, quantity=50)]


def main() -> None:
    bars = pl.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
            ],
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
            "open": [100, 200, 104, 198, 108, 196],
            "high": [102, 202, 106, 200, 110, 198],
            "low": [99, 198, 103, 195, 107, 194],
            "close": [101, 199, 105, 196, 109, 195],
            "volume": [1_000_000, 900_000, 1_100_000, 950_000, 1_200_000, 975_000],
            "adj_close": [101, 199, 105, 196, 109, 195],
        }
    )
    factors = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "factor": [1.2, -0.4, 1.0, -0.2],
        }
    )

    dataset = load_backtest_dataset(bars=bars, factors=factors)
    research = dataset.run_factor_research(horizons=(1,), quantiles=2)
    scores = {s: v for s, v in dataset.factor_values.row(0, named=True).items() if s != "date" and v is not None}
    event_result = dataset.run_event_backtest(BuyStrongestFirstDay(scores))

    print("CortexBacktest - dataset event demo")
    print(f"quality_ok: {dataset.quality.ok}")
    print("factor_summary:")
    print(research.summary())
    print(f"fills: {len(event_result.fills)}")
    print(f"final_equity: {event_result.summary()['final_equity']:.2f}")


if __name__ == "__main__":
    main()
