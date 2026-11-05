"""Trigger and quantity matrix for `BarLimitMatching`.

`BarLimitMatching` replaces the default matching semantics wholesale, so a
defect here produces wrong backtest results rather than an error. It was the
least-covered correctness-critical module in the engine (60%, with
`_trigger_price` almost entirely untested), and the resting-order prefilter
raised the stakes: the prefilter disables itself whenever a custom
`MatchingModel` is installed, so this is now the path that does *not* get the
fast route and has to be right on its own.

The cases below walk every order type and side against prices that sit exactly
on, just inside, and just outside the bar's high/low band, plus the
`price_improvement` and `participation` options.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from cortexbacktest import Bar, Order, Side
from cortexbacktest.contracts import FillPrice, OrderType
from cortexbacktest.event.matching import BarLimitMatching

TS = datetime(2024, 1, 1)
LOW, HIGH, OPEN, CLOSE = 90.0, 110.0, 95.0, 105.0
DEFAULT = CLOSE  # what the caller would have filled at without a trigger


def _bar(volume: float | None = 1000.0) -> Bar:
    return Bar("AAA", TS, OPEN, HIGH, LOW, CLOSE, volume=volume)


def _order(side: Side, order_type: OrderType, *, limit=None, stop=None, quantity=10.0) -> Order:
    return Order("AAA", TS, side, quantity, order_type=order_type,
                 limit_price=limit, stop_price=stop)


def _price(order: Order, *, model: BarLimitMatching | None = None, bar: Bar | None = None):
    matched = (model or BarLimitMatching()).match(
        order, bar or _bar(), default_price=DEFAULT, max_quantity=None
    )
    return None if matched is None else matched[0]


# --- market orders ---------------------------------------------------------

def test_market_order_fills_at_the_default_price():
    assert _price(_order(Side.BUY, OrderType.MARKET)) == DEFAULT
    assert _price(_order(Side.SELL, OrderType.MARKET)) == DEFAULT


# --- limit orders ----------------------------------------------------------

@pytest.mark.parametrize(
    "side, limit, expected, description",
    [
        # A buy limit needs the bar to trade at or below it. Price improvement
        # then fills at the better of the default and the limit.
        (Side.BUY, LOW, LOW, "buy limit exactly at the low fills at the limit"),
        (Side.BUY, LOW - 0.01, None, "buy limit just under the low does not trigger"),
        (Side.BUY, CLOSE, CLOSE, "buy limit at the default price fills there"),
        (Side.BUY, HIGH, DEFAULT, "buy limit above the default improves to the default"),
        # A sell limit mirrors it against the high.
        (Side.SELL, HIGH, HIGH, "sell limit exactly at the high fills at the limit"),
        (Side.SELL, HIGH + 0.01, None, "sell limit just over the high does not trigger"),
        (Side.SELL, LOW, DEFAULT, "sell limit below the default improves to the default"),
    ],
)
def test_limit_trigger_and_price_improvement(side, limit, expected, description):
    assert _price(_order(side, OrderType.LIMIT, limit=limit)) == expected, description


@pytest.mark.parametrize("side, limit", [(Side.BUY, HIGH), (Side.SELL, LOW)])
def test_price_improvement_disabled_fills_at_the_limit(side, limit):
    # Without improvement the fill is the limit price itself, even when the
    # default price is better for the order's side.
    model = BarLimitMatching(price_improvement=False)
    assert _price(_order(side, OrderType.LIMIT, limit=limit), model=model) == limit


def test_limit_order_without_a_limit_price_does_not_trigger():
    assert _price(_order(Side.BUY, OrderType.LIMIT)) is None


# --- stop and trailing-stop orders ----------------------------------------

@pytest.mark.parametrize("order_type", [OrderType.STOP, OrderType.TRAILING_STOP])
@pytest.mark.parametrize(
    "side, stop, expected, description",
    [
        # A buy stop triggers once the bar reaches it, and never fills better
        # than the stop.
        (Side.BUY, HIGH, HIGH, "buy stop exactly at the high fills at the stop"),
        (Side.BUY, HIGH + 0.01, None, "buy stop just over the high does not trigger"),
        (Side.BUY, LOW, DEFAULT, "buy stop below the default fills at the default"),
        # A sell stop triggers on the way down.
        (Side.SELL, LOW, LOW, "sell stop exactly at the low fills at the stop"),
        (Side.SELL, LOW - 0.01, None, "sell stop just under the low does not trigger"),
        (Side.SELL, HIGH, DEFAULT, "sell stop above the default fills at the default"),
    ],
)
def test_stop_trigger_matrix(order_type, side, stop, expected, description):
    assert _price(_order(side, order_type, stop=stop)) == expected, description


def test_stop_order_without_a_stop_price_does_not_trigger():
    assert _price(_order(Side.BUY, OrderType.STOP)) is None


# --- stop-limit orders -----------------------------------------------------

@pytest.mark.parametrize(
    "side, stop, limit, expected, description",
    [
        # Both legs must pass: the stop arms the order, the limit caps the fill.
        (Side.BUY, LOW, HIGH, DEFAULT, "buy stop-limit with both legs satisfied"),
        (Side.BUY, HIGH + 0.01, HIGH, None, "buy stop-limit whose stop never triggers"),
        (Side.BUY, LOW, LOW - 0.01, None, "buy stop-limit whose limit is unreachable"),
        (Side.SELL, HIGH, LOW, DEFAULT, "sell stop-limit with both legs satisfied"),
        (Side.SELL, LOW - 0.01, LOW, None, "sell stop-limit whose stop never triggers"),
        (Side.SELL, HIGH, HIGH + 0.01, None, "sell stop-limit whose limit is unreachable"),
    ],
)
def test_stop_limit_requires_both_legs(side, stop, limit, expected, description):
    assert _price(_order(side, OrderType.STOP_LIMIT, stop=stop, limit=limit)) == expected, description


@pytest.mark.parametrize("stop, limit", [(None, HIGH), (LOW, None)])
def test_stop_limit_missing_either_price_does_not_trigger(stop, limit):
    assert _price(_order(Side.BUY, OrderType.STOP_LIMIT, stop=stop, limit=limit)) is None


# --- quantity and participation -------------------------------------------

def test_participation_caps_quantity_against_bar_volume():
    model = BarLimitMatching(participation=0.1)
    matched = model.match(_order(Side.BUY, OrderType.MARKET), _bar(volume=500.0),
                          default_price=DEFAULT, max_quantity=None)
    assert matched is not None
    assert matched[1] == pytest.approx(50.0), "10% of a 500-share bar"


def test_participation_and_max_quantity_take_the_tighter_bound():
    model = BarLimitMatching(participation=0.5)
    matched = model.match(_order(Side.BUY, OrderType.MARKET), _bar(volume=500.0),
                          default_price=DEFAULT, max_quantity=10.0)
    assert matched is not None
    assert matched[1] == pytest.approx(10.0), "max_quantity is tighter than 250"


def test_volumeless_bar_leaves_quantity_to_the_caller():
    matched = BarLimitMatching().match(_order(Side.BUY, OrderType.MARKET), _bar(volume=None),
                                       default_price=DEFAULT, max_quantity=None)
    assert matched is not None
    assert matched[1] is None


@pytest.mark.parametrize("participation", [0.0, -0.1, 1.5])
def test_participation_outside_the_unit_interval_is_rejected(participation):
    with pytest.raises(ValueError, match="participation"):
        BarLimitMatching(participation=participation)


# --- fill price selection --------------------------------------------------

@pytest.mark.parametrize(
    "fill_price, expected",
    [
        (FillPrice.OPEN, OPEN),
        (FillPrice.CLOSE, CLOSE),
        (FillPrice.MID, (HIGH + LOW) / 2.0),
    ],
)
def test_fill_price_setting_is_carried_by_the_engine_not_the_model(fill_price, expected):
    # The model receives `default_price` already resolved by the execution
    # handler, so its own `fill_price` does not re-derive it. Pinning this so a
    # future change to that division of labour is a deliberate one.
    model = BarLimitMatching(fill_price=fill_price)
    assert _price(_order(Side.BUY, OrderType.MARKET), model=model) == DEFAULT
    assert model.fill_price is fill_price
