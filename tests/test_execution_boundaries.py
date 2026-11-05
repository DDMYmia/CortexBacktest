from datetime import UTC, datetime, timedelta

import pytest

from cortexbacktest.broker import BrokerSimulator
from cortexbacktest.contracts import Bar, FillPrice, Order, OrderType, Side

TS = datetime(2024, 1, 1, tzinfo=UTC)


def bar(day=0, *, open=100, high=110, low=90, close=100, volume=100):
    return Bar("X", TS + timedelta(days=day), open, high, low, close, volume)


def pad(broker, count):
    for i in range(count):
        broker.submit_order(Order("Y", TS, Side.BUY, 1, order_type=OrderType.LIMIT, limit_price=1, id=f"pad{i}"))


@pytest.mark.parametrize("padding", [0, 40])
def test_triggered_stop_limit_persists_across_bars(padding):
    broker = BrokerSimulator(fill_price=FillPrice.OPEN)
    pad(broker, padding)
    broker.submit_order(Order("X", TS, Side.BUY, 1, order_type=OrderType.STOP_LIMIT,
                              stop_price=105, limit_price=103, id="entry"))
    assert broker.process_bar(bar(open=110, high=112, low=108, close=110)) == []
    fills = broker.process_bar(bar(1, open=102, high=104, low=101, close=103))
    assert [(fill.order_id, fill.price) for fill in fills] == [("entry", 102)]


@pytest.mark.parametrize("padding", [0, 40])
def test_oco_partial_fill_cancels_other_leg(padding):
    broker = BrokerSimulator()
    pad(broker, padding)
    broker.submit_order(Order("X", TS, Side.SELL, 10, order_type=OrderType.LIMIT, limit_price=105, id="tp", oco_group="exit"))
    broker.submit_order(Order("X", TS, Side.SELL, 10, order_type=OrderType.STOP, stop_price=95, id="sl", oco_group="exit"))
    fills = broker.process_bar(bar(volume=5))
    assert [(fill.order_id, fill.quantity) for fill in fills] == [("tp", 5)]
    assert not broker.order_manager.is_open("sl")
    assert broker.order_manager.get_order("tp").quantity == 5


@pytest.mark.parametrize("side,stop,open,high,low,expected", [
    (Side.BUY, 105, 112, 115, 110, 112),
    (Side.SELL, 95, 88, 90, 85, 88),
])
def test_stop_gap_fills_at_open_beyond_stop(side, stop, open, high, low, expected):
    broker = BrokerSimulator(fill_price=FillPrice.OPEN)
    broker.submit_order(Order("X", TS, side, 1, order_type=OrderType.STOP, stop_price=stop))
    assert broker.process_bar(bar(open=open, high=high, low=low, close=open))[0].price == expected


def test_cancel_before_matching_prevents_fill_and_after_fill_cannot_reverse_it():
    broker = BrokerSimulator()
    broker.submit_order(Order("X", TS, Side.BUY, 1, id="cancelled"))
    broker.cancel_order("cancelled")
    assert broker.process_bar(bar()) == []
    broker.submit_order(Order("X", TS, Side.BUY, 1, id="filled"))
    assert len(broker.process_bar(bar(1))) == 1
    assert broker.cancel_order("filled") is None
    assert broker.state.positions["X"].quantity == 1
