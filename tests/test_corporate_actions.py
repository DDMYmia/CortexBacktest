from cortexbacktest import FillPrice
from datetime import date, datetime

import pytest

from cortexbacktest import (
    BacktestEngine,
    Bar,
    BrokerSimulator,
    CorporateAction,
    CorporateActionSchedule,
    CorporateActionType,
    Order,
    Side,
    Strategy,
)


class BuyFirstBar(Strategy):
    def __init__(self, quantity: float = 100.0, side: Side = Side.BUY) -> None:
        self.done = False
        self.quantity = quantity
        self.side = side

    def on_bars(self, bars: list[Bar]) -> list[Order]:
        if self.done:
            return []
        self.done = True
        return [Order(bar.symbol, bar.timestamp, self.side, self.quantity) for bar in bars]


def _bars(prices: list[float], *, symbol: str = "AAA", adjusted: bool = False) -> list[Bar]:
    return [
        Bar(symbol, datetime(2024, 1, day + 1), price, price, price, price, adj_close=price if adjusted else None)
        for day, price in enumerate(prices)
    ]


def _equity(result) -> list[float]:
    return result.performance.equity_curve()["equity"].to_list()


def test_split_scales_position_and_leaves_equity_unchanged():
    actions = [CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.SPLIT, ratio=2.0)]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    # Unadjusted prices: the raw quote halves on the ex-date.
    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars([100.0, 100.0, 50.0, 50.0]), BuyFirstBar())

    position = broker.state.positions["AAA"]
    assert position.quantity == pytest.approx(200.0)
    assert position.last_price == pytest.approx(50.0)
    assert _equity(result) == pytest.approx([100_000.0] * 4)


def test_reverse_split_scales_position_down():
    actions = [CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.SPLIT, ratio=0.2)]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars([100.0, 100.0, 500.0, 500.0]), BuyFirstBar())

    position = broker.state.positions["AAA"]
    assert position.quantity == pytest.approx(20.0)
    assert position.last_price == pytest.approx(500.0)
    assert _equity(result) == pytest.approx([100_000.0] * 4)


def test_cash_dividend_pays_holder_and_offsets_the_ex_date_price_drop():
    actions = [CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.CASH_DIVIDEND, amount=2.0)]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    result = BacktestEngine(trade_on_close=True, broker=broker).run(_bars([100.0, 100.0, 98.0, 98.0]), BuyFirstBar())

    assert result.performance.equity_curve()["cash"].to_list() == pytest.approx([90_000.0, 90_000.0, 90_200.0, 90_200.0])
    assert _equity(result) == pytest.approx([100_000.0] * 4)


def test_cash_dividend_charges_a_short_position():
    actions = [CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.CASH_DIVIDEND, amount=2.0)]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([100.0, 100.0, 98.0, 98.0]), BuyFirstBar(side=Side.SELL))

    event = broker.corporate_action_events[0]
    assert event.cash_amount == pytest.approx(-200.0)
    assert broker.state.cash == pytest.approx(109_800.0)


def test_corporate_action_records_what_it_did():
    actions = [CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.SPLIT, ratio=2.0)]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([100.0, 100.0, 50.0]), BuyFirstBar())

    assert len(broker.corporate_action_events) == 1
    event = broker.corporate_action_events[0]
    assert event.symbol == "AAA"
    assert event.action_type == CorporateActionType.SPLIT
    assert (event.quantity_before, event.quantity_after) == (100.0, 200.0)
    assert event.cash_amount == 0.0


def test_action_without_a_position_is_a_no_op():
    actions = [CorporateAction("ZZZ", date(2024, 1, 2), CorporateActionType.CASH_DIVIDEND, amount=5.0)]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([100.0, 100.0]), BuyFirstBar())

    assert broker.corporate_action_events == []
    assert broker.state.cash == pytest.approx(90_000.0)


def test_engine_rejects_corporate_actions_on_adjusted_prices():
    actions = [CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.SPLIT, ratio=2.0)]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, corporate_actions=actions)

    with pytest.raises(ValueError, match="adj_close"):
        BacktestEngine(trade_on_close=True, broker=broker).run(_bars([100.0, 100.0, 50.0], adjusted=True), BuyFirstBar())


def test_adjusted_prices_are_fine_without_corporate_actions():
    result = BacktestEngine(trade_on_close=True).run(_bars([100.0, 100.0, 50.0], adjusted=True), BuyFirstBar())

    assert result.performance.equity_curve().height == 3


def test_split_applies_before_dividend_on_a_shared_ex_date():
    # The dividend is quoted per post-split share, so 200 shares receive it.
    actions = [
        CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.CASH_DIVIDEND, amount=1.0),
        CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.SPLIT, ratio=2.0),
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=100_000.0, corporate_actions=actions)

    BacktestEngine(trade_on_close=True, broker=broker).run(_bars([100.0, 100.0, 49.0]), BuyFirstBar())

    kinds = [event.action_type for event in broker.corporate_action_events]
    assert kinds == [CorporateActionType.SPLIT, CorporateActionType.CASH_DIVIDEND]
    assert broker.corporate_action_events[1].cash_amount == pytest.approx(200.0)


def test_schedule_groups_by_ex_date():
    actions = [
        CorporateAction("AAA", date(2024, 1, 3), CorporateActionType.SPLIT, ratio=2.0),
        CorporateAction("BBB", date(2024, 1, 3), CorporateActionType.CASH_DIVIDEND, amount=1.0),
        CorporateAction("AAA", date(2024, 1, 9), CorporateActionType.CASH_DIVIDEND, amount=1.0),
    ]
    schedule = CorporateActionSchedule(actions)

    assert len(schedule) == 3
    assert bool(schedule)
    assert len(schedule.actions_on(date(2024, 1, 3))) == 2
    assert schedule.actions_on(date(2024, 1, 4)) == ()
    assert schedule.symbols == frozenset({"AAA", "BBB"})


def test_empty_schedule_is_falsy():
    assert not CorporateActionSchedule()
    assert len(CorporateActionSchedule()) == 0


def test_symbol_is_normalized_to_upper_case():
    action = CorporateAction("aaa", date(2024, 1, 3), CorporateActionType.SPLIT, ratio=2.0)

    assert action.symbol == "AAA"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"action_type": CorporateActionType.SPLIT, "ratio": 0.0}, "split ratio"),
        ({"action_type": CorporateActionType.SPLIT, "ratio": -1.0}, "split ratio"),
        ({"action_type": CorporateActionType.SPLIT, "ratio": 2.0, "amount": 1.0}, "must not carry a dividend"),
        ({"action_type": CorporateActionType.CASH_DIVIDEND, "amount": 0.0}, "dividend amount"),
        ({"action_type": CorporateActionType.CASH_DIVIDEND, "amount": -1.0}, "dividend amount"),
        ({"action_type": CorporateActionType.CASH_DIVIDEND, "amount": 1.0, "ratio": 2.0}, "must not carry a split"),
    ],
)
def test_corporate_action_rejects_incoherent_input(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CorporateAction("AAA", date(2024, 1, 3), **kwargs)


def test_corporate_action_requires_a_symbol():
    with pytest.raises(ValueError, match="symbol"):
        CorporateAction("", date(2024, 1, 3), CorporateActionType.SPLIT, ratio=2.0)
