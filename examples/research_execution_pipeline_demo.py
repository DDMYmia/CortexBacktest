"""Run the standard dataset -> research -> event-execution pipeline."""

from __future__ import annotations

import pandas as pd

from cortexbacktest import (
    BrokerExecutionConfig,
    RebalanceConfig,
    ResearchExecutionPipelineConfig,
    load_backtest_dataset,
)


def build_bars() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=8)
    prices = {
        "AAPL": [100, 102, 103, 105, 108, 109, 111, 114],
        "MSFT": [100, 99, 98, 98, 97, 98, 99, 100],
        "NVDA": [100, 101, 104, 108, 111, 116, 120, 125],
        "XOM": [100, 100, 99, 99, 98, 98, 97, 97],
    }
    rows = []
    for offset, date in enumerate(dates):
        for symbol, series in prices.items():
            price = float(series[offset])
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "adj_close": price,
                    "volume": 25_000,
                }
            )
    return pd.DataFrame(rows)


def build_factors() -> pd.DataFrame:
    bars = build_bars()
    prices = bars.pivot(index="date", columns="symbol", values="adj_close")
    momentum = prices.pct_change(fill_method=None)
    factors = momentum.stack(future_stack=True).rename("factor").reset_index()
    return factors.dropna()


def main() -> None:
    dataset = load_backtest_dataset(bars=build_bars(), factors=build_factors())
    result = dataset.run_research_execution_pipeline(
        config=ResearchExecutionPipelineConfig(
            horizons=(1,),
            quantiles=4,
            top_quantile=0.25,
            bottom_quantile=None,
            rebalance=RebalanceConfig(cash_buffer=0.01, min_trade_value=500.0),
            broker=BrokerExecutionConfig(
                initial_cash=100_000,
                commission_bps=1.0,
                slippage_bps=1.0,
            ),
        )
    )

    print("CortexBacktest - research execution pipeline demo")
    print(pd.Series(result.summary()).to_string())


if __name__ == "__main__":
    main()
