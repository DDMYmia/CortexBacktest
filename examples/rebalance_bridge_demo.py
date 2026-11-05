"""Run target weights through the event-driven execution engine."""

from __future__ import annotations

import polars as pl

from cortexbacktest import Bar
from cortexbacktest.event import BacktestEngine, BrokerSimulator, RebalanceConfig, TargetWeightStrategy
from cortexbacktest.research import top_bottom_weights
if __package__:
    from ._demo_data import business_date_strings as _business_days
else:
    from _demo_data import business_date_strings as _business_days


from datetime import UTC, datetime


def build_prices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": _business_days("2024-01-02", 6),
            "AAPL": [100.0, 102.0, 103.0, 104.0, 106.0, 108.0],
            "MSFT": [100.0, 99.0, 98.0, 97.0, 98.0, 99.0],
            "NVDA": [100.0, 101.0, 104.0, 108.0, 111.0, 115.0],
        }
    )


def build_bars(prices: pl.DataFrame) -> list[Bar]:
    symbols = [column for column in prices.columns if column != "date"]
    prices = prices.with_columns(pl.col("date").str.to_datetime())
    bars = []
    for row in prices.iter_rows(named=True):
        day = row["date"]
        timestamp = datetime(day.year, day.month, day.day, tzinfo=UTC)
        for symbol in symbols:
            price = float(row[symbol])
            bars.append(Bar(symbol=symbol, timestamp=timestamp, open=price, high=price, low=price, close=price))
    return bars


def main() -> None:
    prices = build_prices()
    factors = prices.select(
        pl.col("date"),
        pl.lit(1.0).alias("AAPL"),
        pl.lit(-1.0).alias("MSFT"),
        pl.lit(2.0).alias("NVDA"),
    )
    target_weights = top_bottom_weights(
        factors,
        top_quantile=1 / 3,
        bottom_quantile=1 / 3,
        long_weight=1.0,
        short_weight=0.0,
    )

    broker = BrokerSimulator(initial_cash=100_000, commission_bps=1.0, slippage_bps=1.0)
    strategy = TargetWeightStrategy(
        target_weights,
        broker=broker,
        config=RebalanceConfig(cash_buffer=0.01, min_trade_value=500.0, allow_fractional=False),
    )
    result = BacktestEngine(broker=broker).run(build_bars(prices), strategy)

    print("CortexBacktest - rebalance bridge demo")
    print(f"orders: {result.summary()['orders']}")
    print(f"fills: {result.summary()['fills']}")
    print(f"final_equity: {result.summary()['final_equity']:.2f}")
    print("positions:")
    for symbol, position in sorted(broker.state.positions.items()):
        print(f"  {symbol}: qty={position.quantity:.0f}, last={position.last_price:.2f}")


if __name__ == "__main__":
    main()
