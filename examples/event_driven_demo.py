"""Run a minimal event-driven backtest demo."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from cortexbacktest import BacktestEngine, Bar, Order, Side, Strategy
from cortexbacktest.broker import BrokerSimulator


class BuyAndHold(Strategy):
    def __init__(self, quantity: float) -> None:
        self.quantity = quantity
        self.has_bought = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if self.has_bought:
            return []
        self.has_bought = True
        return [Order(symbol=bar.symbol, timestamp=bar.timestamp, side=Side.BUY, quantity=self.quantity)]


def build_bars() -> list[Bar]:
    start = datetime(2024, 1, 2, tzinfo=UTC)
    prices = [100, 102, 101, 105, 109, 108, 112]
    return [
        Bar("AAPL", start + timedelta(days=i), price, price, price, price, 1_000_000)
        for i, price in enumerate(prices)
    ]


def main() -> None:
    broker = BrokerSimulator(commission_bps=1.0, slippage_bps=2.0)
    result = BacktestEngine(broker=broker).run(
        build_bars(),
        BuyAndHold(quantity=10),
    )
    summary = result.summary()

    print("CortexBacktest - event-driven demo")
    print(f"orders: {summary['orders']}")
    print(f"fills: {summary['fills']}")
    print(f"first_fill_price: {result.fills[0].price:.2f}")
    print(f"trade_summary: {result.trades.fill_summary()}")
    print(f"round_trip_summary: {result.trades.summary()}")
    print(f"performance_summary: {result.performance.report().summary}")
    print(f"final_equity: {summary['final_equity']:.2f}")
    print(f"total_return: {summary['total_return']:.4f}")
    print(f"sharpe: {summary['sharpe']:.4f}")
    print(f"max_drawdown: {summary['max_drawdown']:.4f}")


if __name__ == "__main__":
    main()
