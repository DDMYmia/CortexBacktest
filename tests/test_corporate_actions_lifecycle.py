"""Delisting, stock dividends, symbol changes, and mergers.

These four change the economics of a held position, so the broker has to apply
them; the alternative is silent and expensive. A delisted holding left open is
marked at its last trade for the rest of the run, so a bankrupt position keeps
contributing its pre-bankruptcy value to terminal equity — the most common way a
long backtest overstates its return. A renamed holding stops receiving bars and
freezes at its last pre-rename price for the same reason.

The invariant each test pins is economic, not structural: what happened to
equity, cash, and the share count, on the ex-date and after.
"""

from cortexbacktest import FillPrice
from datetime import date, datetime

import pytest

from cortexbacktest import (
    BacktestEngine,
    Bar,
    BrokerSimulator,
    CorporateAction,
    CorporateActionType,
    Order,
    OrderType,
    Side,
    Strategy,
)


class BuyFirstBar(Strategy):
    def __init__(self, quantity: float = 100.0, side: Side = Side.BUY, symbol: str | None = None) -> None:
        self.done = False
        self.quantity = quantity
        self.side = side
        self.symbol = symbol

    def on_bars(self, bars: list[Bar]) -> list[Order]:
        if self.done:
            return []
        self.done = True
        return [
            Order(bar.symbol, bar.timestamp, self.side, self.quantity)
            for bar in bars
            if self.symbol is None or bar.symbol == self.symbol
        ]


def _bars(prices: list[float], *, symbol: str = "AAA") -> list[Bar]:
    return [
        Bar(symbol, datetime(2024, 1, day + 1), price, price, price, price)
        for day, price in enumerate(prices)
    ]


def _equity(result) -> list[float]:
    return result.performance.equity_curve()["equity"].to_list()


def _cash(result) -> list[float]:
    return result.performance.equity_curve()["cash"].to_list()


# --- delisting -------------------------------------------------------------

def test_worthless_delisting_removes_the_position_and_the_equity_it_carried():
    # 100 shares at 100 = 10,000 of the 100,000 account. A delisting at zero
    # must cost exactly that, and must not keep marking the shares afterwards.
    actions = [CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.DELISTING, amount=0.0)]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars([100.0, 100.0, 100.0, 100.0]), BuyFirstBar())

    assert "AAA" not in broker.state.positions
    assert _equity(result) == pytest.approx([100_000.0, 100_000.0, 90_000.0, 90_000.0])
    assert _cash(result) == pytest.approx([90_000.0] * 4)


def test_delisting_with_a_cash_out_value_pays_it_and_leaves_equity_unchanged():
    actions = [CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.DELISTING, amount=100.0)]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars([100.0, 100.0, 100.0, 100.0]), BuyFirstBar())

    assert "AAA" not in broker.state.positions
    # Cashed out at the mark, so nothing is gained or lost.
    assert _equity(result) == pytest.approx([100_000.0] * 4)
    assert _cash(result) == pytest.approx([90_000.0, 90_000.0, 100_000.0, 100_000.0])


def test_delisting_a_short_position_buys_it_back_at_the_cash_out_value():
    actions = [CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.DELISTING, amount=100.0)]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions, allow_short=True)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(
        _bars([100.0, 100.0, 100.0, 100.0]), BuyFirstBar(side=Side.SELL)
    )

    assert "AAA" not in broker.state.positions
    assert _equity(result) == pytest.approx([100_000.0] * 4)


def test_a_worthless_delisting_is_a_gain_for_a_short():
    # The borrow is closed for nothing, so the short keeps the sale proceeds.
    actions = [CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.DELISTING, amount=0.0)]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions, allow_short=True)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(
        _bars([100.0, 100.0, 100.0, 100.0]), BuyFirstBar(side=Side.SELL)
    )

    assert _equity(result) == pytest.approx([100_000.0, 100_000.0, 110_000.0, 110_000.0])


def test_delisting_cancels_orders_resting_on_the_dead_symbol():
    class RestingLimit(Strategy):
        def __init__(self) -> None:
            self.done = False

        def on_bars(self, bars):
            if self.done:
                return []
            self.done = True
            bar = bars[0]
            return [
                Order(bar.symbol, bar.timestamp, Side.BUY, 100),
                # Parked far below the market so it never fills on its own.
                Order(bar.symbol, bar.timestamp, Side.BUY, 10,
                      order_type=OrderType.LIMIT, limit_price=1.0),
            ]

    actions = [CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.DELISTING, amount=0.0)]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([100.0, 100.0, 100.0, 100.0]), RestingLimit())

    # A symbol that will never trade again cannot leave a live order behind.
    assert broker.open_orders == []


# --- stock dividend --------------------------------------------------------

def test_stock_dividend_issues_shares_and_leaves_equity_unchanged():
    # A 10% stock dividend: 100 shares become 110, and the quote drops by the
    # same factor, so the holding is worth what it was.
    actions = [CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.STOCK_DIVIDEND, ratio=0.1)]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(
        _bars([110.0, 110.0, 100.0, 100.0]), BuyFirstBar()
    )

    position = broker.state.positions["AAA"]
    assert position.quantity == pytest.approx(110.0)
    assert _equity(result) == pytest.approx([100_000.0] * 4)


def test_stock_dividend_is_reported_as_its_own_action_type():
    actions = [CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.STOCK_DIVIDEND, ratio=0.5)]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([150.0, 150.0, 100.0, 100.0]), BuyFirstBar())

    (event,) = broker.corporate_action_events
    # Implemented via the split path, but it must not be reported as a split.
    assert event.action_type is CorporateActionType.STOCK_DIVIDEND
    assert event.quantity_before == pytest.approx(100.0)
    assert event.quantity_after == pytest.approx(150.0)


# --- symbol change ---------------------------------------------------------

def test_symbol_change_moves_the_position_and_keeps_marking_it():
    # The holding must follow the rename: bars arrive under BBB from the
    # ex-date on, and the position has to be there to receive them.
    actions = [
        CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.SYMBOL_CHANGE, new_symbol="BBB")
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    feed = [
        Bar("AAA", datetime(2024, 1, 1), 100.0, 100.0, 100.0, 100.0),
        Bar("AAA", datetime(2024, 1, 2), 100.0, 100.0, 100.0, 100.0),
        Bar("BBB", datetime(2024, 1, 3), 120.0, 120.0, 120.0, 120.0),
        Bar("BBB", datetime(2024, 1, 4), 120.0, 120.0, 120.0, 120.0),
    ]
    result = BacktestEngine(trade_on_close=True, broker=broker).run(feed, BuyFirstBar(symbol="AAA"))

    assert "AAA" not in broker.state.positions
    assert broker.state.positions["BBB"].quantity == pytest.approx(100.0)
    # The rename itself is economically neutral; the gain is the new price.
    assert _equity(result)[-1] == pytest.approx(102_000.0)


def test_symbol_change_reports_the_destination():
    actions = [
        CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.SYMBOL_CHANGE, new_symbol="bbb")
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([100.0, 100.0, 100.0, 100.0]), BuyFirstBar())

    (event,) = broker.corporate_action_events
    assert event.new_symbol == "BBB", "destination symbols are normalised like every other symbol"
    assert event.quantity_before == event.quantity_after == pytest.approx(100.0)


def test_symbol_change_merges_into_an_existing_holding_of_the_destination():
    from cortexbacktest.event.ledger import Ledger

    ledger = Ledger(initial_cash=100_000.0)
    ledger.apply_fill(_fill("AAA", 100.0, 10.0))
    ledger.apply_fill(_fill("BBB", 50.0, 20.0))

    carried = ledger.apply_symbol_change("AAA", "BBB")

    assert carried == pytest.approx(100.0)
    assert "AAA" not in ledger.state.positions
    # Two tickers becoming one must not lose either side's shares.
    assert ledger.state.positions["BBB"].quantity == pytest.approx(150.0)


# --- merger ----------------------------------------------------------------

def test_all_share_merger_exchanges_the_position_at_the_ratio():
    # 0.5 shares of BBB per AAA share: 100 shares become 50, and the mark
    # doubles so the holding's value is continuous across the exchange.
    actions = [
        CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.MERGER,
                        new_symbol="BBB", ratio=0.5)
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    feed = [
        Bar("AAA", datetime(2024, 1, 1), 100.0, 100.0, 100.0, 100.0),
        Bar("AAA", datetime(2024, 1, 2), 100.0, 100.0, 100.0, 100.0),
        Bar("BBB", datetime(2024, 1, 3), 200.0, 200.0, 200.0, 200.0),
        Bar("BBB", datetime(2024, 1, 4), 200.0, 200.0, 200.0, 200.0),
    ]
    result = BacktestEngine(trade_on_close=True, broker=broker).run(feed, BuyFirstBar(symbol="AAA"))

    assert "AAA" not in broker.state.positions
    assert broker.state.positions["BBB"].quantity == pytest.approx(50.0)
    assert _equity(result) == pytest.approx([100_000.0] * 4)


def test_cash_and_share_merger_pays_the_cash_leg_too():
    actions = [
        CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.MERGER,
                        new_symbol="BBB", ratio=0.5, amount=10.0)
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    feed = [
        Bar("AAA", datetime(2024, 1, 1), 100.0, 100.0, 100.0, 100.0),
        Bar("AAA", datetime(2024, 1, 2), 100.0, 100.0, 100.0, 100.0),
        Bar("BBB", datetime(2024, 1, 3), 200.0, 200.0, 200.0, 200.0),
        Bar("BBB", datetime(2024, 1, 4), 200.0, 200.0, 200.0, 200.0),
    ]
    result = BacktestEngine(trade_on_close=True, broker=broker).run(feed, BuyFirstBar(symbol="AAA"))

    # 100 shares x 10 cash consideration on top of the share leg.
    assert _cash(result)[-1] == pytest.approx(91_000.0)
    assert _equity(result)[-1] == pytest.approx(101_000.0)


def test_all_cash_acquisition_closes_the_position():
    # ratio=0 with cash consideration: the shares are bought outright.
    actions = [
        CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.MERGER,
                        new_symbol="BBB", ratio=0.0, amount=120.0)
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars([100.0, 100.0, 100.0, 100.0]), BuyFirstBar())

    assert "AAA" not in broker.state.positions
    assert "BBB" not in broker.state.positions
    assert _equity(result)[-1] == pytest.approx(102_000.0)


# --- ordering within one ex-date ------------------------------------------

def test_a_split_settles_before_a_cash_dividend_on_the_same_date():
    # The dividend is quoted per post-split share, so 200 shares must be paid,
    # not 100. Paying first would understate the dividend by half.
    actions = [
        CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.CASH_DIVIDEND, amount=1.0),
        CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.SPLIT, ratio=2.0),
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars([100.0, 100.0, 50.0, 50.0]), BuyFirstBar())

    assert _cash(result)[-1] == pytest.approx(90_200.0)


def test_a_dividend_owed_on_the_delisting_date_is_paid_before_the_shares_vanish():
    actions = [
        CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.DELISTING, amount=0.0),
        CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.CASH_DIVIDEND, amount=2.0),
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars([100.0, 100.0, 100.0, 100.0]), BuyFirstBar())

    # 90,000 after the purchase, plus the 200 dividend, and the shares expire
    # worthless. Ordering the delisting first would silently drop the payment.
    assert _cash(result)[-1] == pytest.approx(90_200.0)
    assert "AAA" not in broker.state.positions


def _fill(symbol: str, quantity: float, price: float):
    from cortexbacktest import Fill

    return Fill(symbol, datetime(2024, 1, 1), Side.BUY, quantity, price)
