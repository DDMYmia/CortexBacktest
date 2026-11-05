"""Differential property test: the linear and indexed matching paths must be
behaviorally identical over the same order flow.

The broker picks a path from the open-order count (a performance heuristic), so
any semantic gap between the paths silently changes backtest results when a
strategy crosses the threshold. This test forces both paths over identical
randomized order flow (market/limit/stop orders, OCO groups, bracket children,
volume caps, partial fills) and asserts field-identical fills, order records,
and equity curves.

Resting-order boundaries are checked independently below: both paths share a
prefilter, so path equivalence alone cannot establish its correctness.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pytest

from cortexbacktest import BacktestEngine, Bar, Order, Side, Strategy
from cortexbacktest.contracts import OrderStatus, OrderType
from cortexbacktest.event.broker import BrokerSimulator
from cortexbacktest.event.matching import MatchingModel
from cortexbacktest import FillPrice

SYMBOLS = ["AAA", "BBB", "CCC"]


class ForcedPathBroker(BrokerSimulator):
    def __init__(self, path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._path = path

    def process_bar(self, bar, *, expire_orders=True, mark=True):
        if mark:
            self.mark(bar)
        if expire_orders:
            self.expire_orders(bar.timestamp)
        if self.order_manager.open_order_count == 0:
            return []
        if self._path == "linear":
            return self._process_bar_linear(bar, self.order_manager.open_orders)
        return self._process_bar_indexed(bar)


class RandomFlowStrategy(Strategy):
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.counter = iter(range(100_000))

    def on_bars(self, bars):
        rng = self.rng
        timestamp = bars[0].timestamp
        orders = []
        oco_group = f"G{rng.randint(0, 3)}" if rng.random() < 0.3 else None
        for _ in range(rng.randint(0, 4)):
            symbol = rng.choice(SYMBOLS)
            side = rng.choice([Side.BUY, Side.SELL])
            quantity = rng.randint(1, 20)
            order_id = f"O{next(self.counter):05d}"
            kind = rng.random()
            if kind < 0.4:
                orders.append(Order(symbol, timestamp, side, quantity, id=order_id, oco_group=oco_group))
            elif kind < 0.7:
                orders.append(Order(
                    symbol, timestamp, side, quantity, order_type=OrderType.LIMIT,
                    limit_price=round(rng.uniform(80, 120), 2), id=order_id, oco_group=oco_group,
                ))
            else:
                orders.append(Order(
                    symbol, timestamp, side, quantity, order_type=OrderType.STOP,
                    stop_price=round(rng.uniform(80, 120), 2), id=order_id, oco_group=oco_group,
                ))
            if rng.random() < 0.2:
                orders.append(Order(
                    symbol, timestamp, Side.SELL if side == Side.BUY else Side.BUY, quantity,
                    order_type=OrderType.LIMIT, limit_price=round(rng.uniform(80, 120), 2),
                    id=f"O{next(self.counter):05d}", parent_id=order_id,
                ))
        return orders


def _bars(rng: random.Random, days: int = 15) -> list[Bar]:
    start = datetime(2024, 1, 1)
    bars = []
    for day in range(days):
        timestamp = start + timedelta(days=day)
        for symbol in SYMBOLS:
            open_ = rng.uniform(90, 110)
            high = open_ * rng.uniform(1.0, 1.15)
            low = open_ * rng.uniform(0.85, 1.0)
            close = rng.uniform(low, high)
            volume = rng.choice([None, rng.uniform(5, 500)])
            bars.append(Bar(symbol, timestamp, open_, high, low, close, volume=volume))
    return bars


def _run(path: str, seed: int):
    bars = _bars(random.Random(seed))
    broker = ForcedPathBroker(
        path, fill_price=FillPrice.CLOSE, initial_cash=1_000_000, commission_bps=5, slippage_bps=3, max_volume_share=0.5,
    )
    engine = BacktestEngine(broker=broker, trade_on_close=(seed % 2 == 0))
    result = engine.run(bars, RandomFlowStrategy(seed))
    fills = [
        (fill.timestamp, fill.order_id, fill.symbol, fill.side, fill.quantity, fill.price)
        for fill in result.fills
    ]
    records = [(record.order.id, record.status) for record in result.orders]
    equity = result.performance.equity_curve()["equity"].to_list()
    return fills, records, equity


@pytest.mark.parametrize("seed", [17, 21, 34, 0, 1, 2, 3, 4])
def test_linear_and_indexed_paths_are_identical(seed):
    linear = _run("linear", seed)
    indexed = _run("indexed", seed)
    assert linear[0] == indexed[0], "fill sequences differ"
    assert linear[1] == indexed[1], "order record sequences differ"
    assert linear[2] == pytest.approx(indexed[2]), "equity curves differ"


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 17, 34])
def test_automatic_small_book_symbol_selection_matches_full_linear_reference(seed):
    bars = _bars(random.Random(seed))
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=1_000_000, commission_bps=5,
                             slippage_bps=3, max_volume_share=.5)
    result = BacktestEngine(broker=broker, trade_on_close=(seed % 2 == 0)).run(bars, RandomFlowStrategy(seed))
    reference = _run('linear', seed)
    assert [(f.timestamp, f.order_id, f.symbol, f.side, f.quantity, f.price) for f in result.fills] == reference[0]
    assert [(r.order.id, r.status) for r in result.orders] == reference[1]
    assert result.performance.equity_curve()['equity'].to_list() == pytest.approx(reference[2])


@pytest.mark.parametrize('path,open_during_rejection', [('linear', True), ('indexed', False)])
def test_matching_failure_preserves_record_sink_ordering(path, open_during_rejection, monkeypatch):
    broker = ForcedPathBroker(path)
    at = datetime(2024, 1, 1)
    broker.submit_order(Order('AAA', at, Side.BUY, 1, id='bad'))
    observed = []

    def record_sink(record):
        if record.status is OrderStatus.REJECTED:
            observed.append(broker.order_manager.is_open('bad'))

    def fail_matching(*args, **kwargs):
        raise ValueError('fixture rejection')

    broker.order_manager.record_sink = record_sink
    monkeypatch.setattr(broker.execution_handler, 'try_fill', fail_matching)
    assert broker.process_bar(Bar('AAA', at, 100, 101, 99, 100)) == []
    assert observed == [open_during_rejection]
    assert broker.open_orders == []


SYMBOL = "AAA"
START = datetime(2024, 1, 1)
# Every bar is identical, so a resting order either fills on the first bar it is
# live for or never fills at all.
LOW, HIGH, OPEN, CLOSE = 90.0, 110.0, 100.0, 100.0


class SubmitOnce(Strategy):
    def __init__(self, orders: list[Order]) -> None:
        self._orders = orders
        self._sent = False

    def on_bars(self, bars):
        if self._sent:
            return []
        self._sent = True
        return [
            Order(
                SYMBOL, bars[0].timestamp, order.side, order.quantity,
                order_type=order.order_type, limit_price=order.limit_price,
                stop_price=order.stop_price, id=order.id,
            )
            for order in self._orders
        ]


def _resting_bars(count: int = 3) -> list[Bar]:
    return [
        Bar(SYMBOL, START + timedelta(days=day), OPEN, HIGH, LOW, CLOSE, volume=1_000_000.0)
        for day in range(count)
    ]


def _fill_count(path: str, order: Order, *, matching_model: MatchingModel | None = None) -> int:
    broker = ForcedPathBroker(path, fill_price=FillPrice.CLOSE, initial_cash=1_000_000, max_volume_share=1.0,
                              matching_model=matching_model)
    engine = BacktestEngine(broker=broker, trade_on_close=True)
    result = engine.run(_resting_bars(), SubmitOnce([order]))
    return len(result.fills)


def _order(side: Side, order_type: OrderType, *, limit=None, stop=None) -> Order:
    return Order(SYMBOL, START, side, 1, order_type=order_type,
                 limit_price=limit, stop_price=stop, id="O1")


@pytest.mark.parametrize("path", ["linear", "indexed"])
@pytest.mark.parametrize(
    "order, should_fill, description",
    [
        # Limit buy fills when the bar trades at or below the limit. The band
        # check is `bar.low > limit_price`, so limit == low must still fill.
        (_order(Side.BUY, OrderType.LIMIT, limit=LOW), True, "buy limit exactly at the low"),
        (_order(Side.BUY, OrderType.LIMIT, limit=LOW - 0.01), False, "buy limit just under the low"),
        (_order(Side.BUY, OrderType.LIMIT, limit=HIGH), True, "buy limit above the whole bar"),
        # Limit sell mirrors it against the high.
        (_order(Side.SELL, OrderType.LIMIT, limit=HIGH), True, "sell limit exactly at the high"),
        (_order(Side.SELL, OrderType.LIMIT, limit=HIGH + 0.01), False, "sell limit just over the high"),
        (_order(Side.SELL, OrderType.LIMIT, limit=LOW), True, "sell limit below the whole bar"),
        # Stop buy triggers on `bar.high >= stop`, so stop == high must fill.
        (_order(Side.BUY, OrderType.STOP, stop=HIGH), True, "buy stop exactly at the high"),
        (_order(Side.BUY, OrderType.STOP, stop=HIGH + 0.01), False, "buy stop just over the high"),
        # Stop sell triggers on `bar.low <= stop`.
        (_order(Side.SELL, OrderType.STOP, stop=LOW), True, "sell stop exactly at the low"),
        (_order(Side.SELL, OrderType.STOP, stop=LOW - 0.01), False, "sell stop just under the low"),
        # Stop-limit is prefiltered on its stop only; the limit leg still applies.
        (_order(Side.BUY, OrderType.STOP_LIMIT, stop=HIGH, limit=HIGH), True,
         "buy stop-limit triggering exactly at the high"),
        (_order(Side.BUY, OrderType.STOP_LIMIT, stop=HIGH + 0.01, limit=HIGH), False,
         "buy stop-limit whose stop is out of band"),
    ],
)
def test_prefilter_matches_the_fill_rules_at_the_band_boundary(path, order, should_fill, description):
    assert _fill_count(path, order) == (1 if should_fill else 0), description


class _FillEverything(MatchingModel):
    """A model that fills any order regardless of price — the prefilter must not pre-empt it."""

    def match(self, order, bar, *, default_price, max_quantity=None):
        return default_price, max_quantity


@pytest.mark.parametrize("path", ["linear", "indexed"])
def test_custom_matching_model_still_sees_out_of_band_orders(path):
    # A custom model may fill on conditions the default band knows nothing
    # about, so the prefilter has to switch itself off when one is installed.
    far_below = _order(Side.BUY, OrderType.LIMIT, limit=LOW - 50.0)
    assert _fill_count(path, far_below) == 0, "sanity: unreachable without a custom model"
    assert _fill_count(path, far_below, matching_model=_FillEverything()) == 1
