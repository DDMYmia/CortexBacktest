"""Default fill timing: a decision on bar T fills at the open of bar T+1."""

import random
from datetime import UTC, datetime, timedelta

import pytest

from cortexbacktest import (
    BacktestEngine, Bar, BrokerSimulator, FillPrice, Order, OrderType, Side, Strategy, TimeInForce,
)


class BuyOnSecondBar(Strategy):
    def on_bar(self, bar):
        if bar.timestamp.day == 2:
            return [Order(bar.symbol, bar.timestamp, Side.BUY, 100)]
        return []


def _bars():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    # open, close per day: (10, 11), (20, 21), (30, 31), (40, 41)
    return [Bar("X", start + timedelta(days=i), o, o + 1, o - 1, o + 1, 1e6) for i, o in enumerate((10, 20, 30, 40))]


def _fill_price(engine):
    return [fill.price for fill in engine.run(_bars(), BuyOnSecondBar()).fills]


def test_default_engine_fills_at_next_bar_open():
    assert _fill_price(BacktestEngine()) == [30]


def test_default_broker_fills_at_next_bar_open():
    assert _fill_price(BacktestEngine(broker=BrokerSimulator())) == [30]


def test_engine_built_broker_follows_same_bar_close_timing():
    assert _fill_price(BacktestEngine(trade_on_close=True)) == [21]


def test_next_bar_close_is_explicit():
    assert _fill_price(BacktestEngine(broker=BrokerSimulator(fill_price=FillPrice.CLOSE))) == [31]


class _FullPassBroker(BrokerSimulator):
    """Overriding process_bar makes the engine run the opening fill pass on every bar."""

    def process_bar(self, bar, *, expire_orders=True, mark=True):
        return super().process_bar(bar, expire_orders=expire_orders, mark=mark)


class _SparseFlow(Strategy):
    """Mostly an empty book: occasional market orders and short-lived limits."""

    def __init__(self, seed):
        super().__init__()
        self.rng = random.Random(seed)
        self.notified = []

    def notify_order(self, record):
        self.notified.append((record.order.id, record.status))

    def on_bars(self, bars):
        orders = []
        for bar in bars:
            roll = self.rng.random()
            if roll < 0.08:
                orders.append(Order(bar.symbol, bar.timestamp, self.rng.choice([Side.BUY, Side.SELL]), 10))
            elif roll < 0.12:
                orders.append(Order(bar.symbol, bar.timestamp, Side.BUY, 5, order_type=OrderType.LIMIT,
                                    limit_price=bar.close * 0.99, time_in_force=TimeInForce.DAY))
        return orders


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_idle_opening_pass_skip_matches_the_full_pass(seed):
    rng = random.Random(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = []
    for day in range(60):
        for symbol in ("A", "B"):
            o = rng.uniform(95, 105)
            c = o * rng.uniform(0.98, 1.02)
            bars.append(Bar(symbol, start + timedelta(days=day), o, max(o, c) + 1, min(o, c) - 1, c, 1e6))

    def run(broker):
        strategy = _SparseFlow(seed)
        result = BacktestEngine(broker=broker).run(bars, strategy)
        return result.performance.equity_curve()["equity"].to_list(), result.fills, strategy.notified

    assert run(BrokerSimulator(allow_short=False)) == run(_FullPassBroker(allow_short=False))


def test_same_bar_open_fill_is_rejected_as_look_ahead():
    with pytest.raises(ValueError, match="trade_on_close=True needs a non-open fill_price"):
        BacktestEngine(broker=BrokerSimulator(fill_price=FillPrice.OPEN), trade_on_close=True)
