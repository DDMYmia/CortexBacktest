"""Futures are settled by variation margin, not by exchanging principal.

Treating a futures contract like a share has one visible symptom and one
invisible one. Visibly, the account is asked for the full notional to open a
position — roughly $250,000 for one ES contract against the ~$13,000 of margin a
real account posts — so leveraged strategies simply never fill. Invisibly, the
notional sits in equity as though the account owned it, and the day's profit is
never converted to cash, so nothing is available to meet a margin call.

The invariant that holds both models together: **equity is identical either
way**. A futures position's P&L does not depend on how it is booked. Only the
cash placement and the buying-power test change. Several tests below assert that
equivalence directly, because it is what proves this is an accounting change
rather than a change in results.
"""

from cortexbacktest import FillPrice
from datetime import datetime

import pytest

from cortexbacktest import (
    BacktestEngine,
    Bar,
    BrokerSimulator,
    Order,
    Side,
    Strategy,
)

SYMBOL = "ES"
MULTIPLIER = 50.0


class BuyOnce(Strategy):
    def __init__(self, quantity: float = 1.0, side: Side = Side.BUY) -> None:
        self.done = False
        self.quantity = quantity
        self.side = side

    def on_bars(self, bars):
        if self.done:
            return []
        self.done = True
        return [Order(bars[0].symbol, bars[0].timestamp, self.side, self.quantity)]


class ScriptedOrders(Strategy):
    """Place a signed quantity on the bar at each given index."""

    def __init__(self, schedule: dict[int, float]) -> None:
        self.schedule = schedule
        self.index = -1

    def on_bars(self, bars):
        self.index += 1
        signed = self.schedule.get(self.index)
        if not signed:
            return []
        side = Side.BUY if signed > 0 else Side.SELL
        return [Order(bars[0].symbol, bars[0].timestamp, side, abs(signed))]


def _bars(prices: list[float], symbol: str = SYMBOL) -> list[Bar]:
    return [
        Bar(symbol, datetime(2024, 1, day + 1), price, price, price, price)
        for day, price in enumerate(prices)
    ]


def _broker(cash: float, *, futures: bool = True, **kwargs) -> BrokerSimulator:
    kwargs.setdefault("fill_price", FillPrice.CLOSE)
    return BrokerSimulator(
        initial_cash=cash,
        contract_multipliers={SYMBOL: MULTIPLIER},
        futures_symbols=[SYMBOL] if futures else None,
        **kwargs,
    )


def _curve(result, column: str = "equity") -> list[float]:
    return result.performance.equity_curve()[column].to_list()


# --- buying power ----------------------------------------------------------

def test_opening_a_contract_does_not_require_its_notional():
    # One ES at 5000 x 50 is $250,000 of notional. A $50,000 account trades it
    # every day; under principal accounting it could not open at all.
    broker = _broker(50_000.0)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 5000.0]), BuyOnce())

    assert len(result.fills) == 1
    assert broker.state.positions[SYMBOL].quantity == pytest.approx(1.0)


def test_the_same_account_cannot_open_it_under_principal_accounting():
    # The contrast that makes the previous test meaningful: identical inputs,
    # futures accounting off.
    broker = _broker(50_000.0, futures=False)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 5000.0]), BuyOnce())

    assert result.fills == []


def test_opening_costs_no_cash_beyond_commission():
    broker = _broker(50_000.0, commission_bps=1.0)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 5000.0]), BuyOnce())

    commission = 250_000.0 * 0.0001
    assert _curve(result, "cash")[0] == pytest.approx(50_000.0 - commission)


# --- daily settlement ------------------------------------------------------

def test_the_daily_move_is_settled_into_cash():
    broker = _broker(50_000.0)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 5010.0, 4990.0, 5020.0]), BuyOnce())

    # Settlement happens at the start of the next session, on the prior close:
    # +10 then -20 index points, at $50 a point.
    assert _curve(result, "cash") == pytest.approx([50_000.0, 50_000.0, 50_500.0, 49_500.0])
    assert broker.variation_margin_charges[0].amount == pytest.approx(500.0)
    assert broker.variation_margin_charges[1].amount == pytest.approx(-1_000.0)


def test_equity_tracks_the_position_between_settlements():
    broker = _broker(50_000.0)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 5010.0, 4990.0, 5020.0]), BuyOnce())

    # Equity follows the mark continuously even though cash only moves daily.
    assert _curve(result) == pytest.approx([50_000.0, 50_500.0, 49_500.0, 51_000.0])


@pytest.mark.parametrize("prices", [
    [5000.0, 5010.0, 4990.0, 5020.0],
    [5000.0, 4900.0, 4800.0, 5300.0],
    [5000.0, 5000.0, 5000.0, 5000.0],
])
def test_equity_is_identical_with_and_without_futures_accounting(prices):
    # The whole point: this is a change in where cash sits and what buying power
    # means, not a change in what the strategy earned. Funded well enough that
    # principal accounting can also open the position.
    futures = BacktestEngine(trade_on_close=True, broker=_broker(400_000.0)).run(_bars(prices), BuyOnce())
    principal = BacktestEngine(trade_on_close=True, broker=_broker(400_000.0, futures=False)).run(_bars(prices), BuyOnce())

    assert _curve(futures) == pytest.approx(_curve(principal))


def test_a_short_position_settles_with_the_opposite_sign():
    broker = _broker(50_000.0)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(
        _bars([5000.0, 5010.0, 4990.0]), BuyOnce(side=Side.SELL)
    )

    # Price up is a loss for the short, price down a gain.
    assert _curve(result, "cash") == pytest.approx([50_000.0, 50_000.0, 49_500.0])
    assert _curve(result) == pytest.approx([50_000.0, 49_500.0, 50_500.0])


# --- realising P&L on the way out -----------------------------------------

def test_closing_realises_the_unsettled_move_immediately():
    # Bought at 5000, closed at 5010 within the same session: the $500 is
    # realised on the fill, not left for the next settlement.
    broker = _broker(50_000.0)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(
        _bars([5000.0, 5010.0, 5010.0]), ScriptedOrders({0: 1.0, 1: -1.0})
    )

    assert SYMBOL not in broker.state.positions
    assert _curve(result, "cash")[-1] == pytest.approx(50_500.0)
    assert _curve(result)[-1] == pytest.approx(50_500.0)


def test_a_partial_close_realises_only_the_closed_quantity():
    broker = _broker(200_000.0)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(
        _bars([5000.0, 5010.0, 5010.0]), ScriptedOrders({0: 2.0, 1: -1.0})
    )

    # One contract realised (+$500); the other still carries its move.
    assert broker.state.positions[SYMBOL].quantity == pytest.approx(1.0)
    assert _curve(result, "cash")[1] == pytest.approx(200_500.0)
    assert _curve(result)[-1] == pytest.approx(201_000.0)


def test_adding_to_a_position_blends_the_settlement_basis():
    broker = _broker(200_000.0)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(
        _bars([5000.0, 5010.0, 5020.0, 5020.0]), ScriptedOrders({0: 1.0, 1: 1.0})
    )

    # One lot from 5000 and one from 5010 held to 5020: 20 + 10 = 30 points.
    assert broker.state.positions[SYMBOL].quantity == pytest.approx(2.0)
    assert _curve(result)[-1] == pytest.approx(200_000.0 + 30.0 * MULTIPLIER)


def test_flipping_through_zero_starts_a_fresh_basis():
    broker = _broker(200_000.0)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(
        _bars([5000.0, 5010.0, 5000.0, 5000.0]), ScriptedOrders({0: 1.0, 1: -2.0})
    )

    # Long one from 5000 closed at 5010 (+$500), then short one from 5010 held
    # to 5000 (+$500 more).
    assert broker.state.positions[SYMBOL].quantity == pytest.approx(-1.0)
    assert _curve(result)[-1] == pytest.approx(201_000.0)


# --- interaction with the rest of the account -----------------------------

def test_a_cash_instrument_in_the_same_account_still_pays_principal():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE,
        initial_cash=100_000.0,
        contract_multipliers={SYMBOL: MULTIPLIER},
        futures_symbols=[SYMBOL],
    )

    feed = sorted(
        _bars([5000.0, 5000.0]) + _bars([100.0, 100.0], symbol="AAA"),
        key=lambda bar: (bar.timestamp, bar.symbol),
    )

    class BuyBoth(Strategy):
        def __init__(self) -> None:
            self.done = False

        def on_bars(self, bars):
            if self.done:
                return []
            self.done = True
            return [Order(bar.symbol, bar.timestamp, Side.BUY, 1) for bar in bars]

    result = BacktestEngine(trade_on_close=True, broker=broker).run(feed, BuyBoth())

    # AAA cost its full price; ES cost nothing. Futures accounting must not
    # leak into the equity instrument sharing the account.
    assert _curve(result, "cash")[0] == pytest.approx(99_900.0)
    assert _curve(result)[0] == pytest.approx(100_000.0)


def test_the_final_session_settles_when_the_run_ends():
    # Settlement runs at session boundaries, and the last bar has none after it,
    # so without an end-of-run settle the account finishes short by the closing
    # day's P&L — equity right, cash stale.
    broker = _broker(50_000.0)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 5100.0]), BuyOnce())

    assert broker.state.cash == pytest.approx(55_000.0)
    assert _curve(result)[-1] == pytest.approx(55_000.0)


def test_end_of_run_settlement_does_not_move_equity():
    # It relocates value from unsettled variation margin into cash; if equity
    # changed, the P&L would have been counted twice.
    broker = _broker(50_000.0)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars([5000.0, 5100.0, 4950.0]), BuyOnce())

    assert _curve(result)[-1] == pytest.approx(47_500.0)
    assert broker.state.cash == pytest.approx(47_500.0)


def test_settlement_is_a_no_op_for_an_account_holding_no_futures():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars([100.0, 110.0], symbol="AAA"), BuyOnce(quantity=10.0))

    assert broker.variation_margin_charges == []
    assert _curve(result)[-1] == pytest.approx(100_100.0)
