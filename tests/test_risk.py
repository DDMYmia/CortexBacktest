from datetime import UTC, datetime

import pytest

from cortexbacktest import BacktestEngine, Bar, FillPrice, Order, Side, Strategy
from cortexbacktest.contracts import Position
from cortexbacktest.event import (
    BrokerSimulator,
    LongOnlyRule,
    MaxGroupWeightRule,
    MaxGrossExposureRule,
    MaxLeverageRule,
    MaxNetExposureRule,
    MaxOrderValueRule,
    MaxPositionValueRule,
    MaxPositionWeightRule,
)


def test_long_only_rule_rejects_short_sale_on_submit():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, risk_rules=[LongOnlyRule()])
    order = Order("AAPL", datetime(2024, 1, 1, tzinfo=UTC), Side.SELL, 1)

    record = broker.submit_order(order)

    assert record.status.value == "rejected"
    assert "long-only" in record.reason


def test_max_order_value_rule_rejects_at_fill_time():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, risk_rules=[MaxOrderValueRule(max_value=50)])
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100)
    order = Order("AAPL", bar.timestamp, Side.BUY, 1)

    with pytest.raises(ValueError, match="order value"):
        broker.execute(order, bar)


def test_max_position_value_rule_checks_projected_position():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, risk_rules=[MaxPositionValueRule(max_value=150)])
    broker.state.positions["AAPL"] = Position("AAPL", 1, 100)
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100)

    with pytest.raises(ValueError, match="position value"):
        broker.execute(Order("AAPL", bar.timestamp, Side.BUY, 1), bar)


def test_max_leverage_rule_checks_projected_gross_exposure():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=1_000, risk_rules=[MaxLeverageRule(max_leverage=0.5)])
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100)

    with pytest.raises(ValueError, match="leverage"):
        broker.execute(Order("AAPL", bar.timestamp, Side.BUY, 6), bar)


def test_max_position_weight_rule_checks_projected_weight():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=1_000, risk_rules=[MaxPositionWeightRule(max_weight=0.4)])
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100)

    with pytest.raises(ValueError, match="position weight"):
        broker.execute(Order("AAPL", bar.timestamp, Side.BUY, 5), bar)


def test_max_gross_exposure_rule_checks_projected_book():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=1_000, risk_rules=[MaxGrossExposureRule(max_exposure=300)])
    broker.state.positions["MSFT"] = Position("MSFT", 2, 100)
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100)

    with pytest.raises(ValueError, match="gross exposure"):
        broker.execute(Order("AAPL", bar.timestamp, Side.BUY, 2), bar)


def test_max_net_exposure_rule_allows_offsets_and_rejects_directional_book():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE,
        initial_cash=1_000,
        allow_short=True,
        risk_rules=[MaxNetExposureRule(max_abs_exposure=100)],
    )
    broker.state.positions["MSFT"] = Position("MSFT", -2, 100)
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100)

    fill = broker.execute(Order("AAPL", bar.timestamp, Side.BUY, 1), bar)

    assert fill.quantity == 1
    with pytest.raises(ValueError, match="net exposure"):
        broker.execute(Order("AAPL", bar.timestamp, Side.BUY, 3), bar)


def test_max_group_weight_rule_checks_projected_group_exposure():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE,
        initial_cash=1_000,
        risk_rules=[
            MaxGroupWeightRule(
                groups={"aapl": "technology", "MSFT": "technology"},
                max_weights={"technology": 0.5},
            )
        ],
    )
    broker.state.positions["MSFT"] = Position("MSFT", 2, 100)
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100)

    with pytest.raises(ValueError, match="group technology weight"):
        broker.execute(Order("AAPL", bar.timestamp, Side.BUY, 5), bar)


def test_max_group_weight_rule_ignores_unconfigured_groups():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE,
        initial_cash=1_000,
        risk_rules=[
            MaxGroupWeightRule(
                groups={"XOM": "energy"},
                max_weights={"technology": 0.5},
            )
        ],
    )
    bar = Bar("XOM", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100)

    fill = broker.execute(Order("XOM", bar.timestamp, Side.BUY, 8), bar)

    assert fill.quantity == 8


class TooLargeOrderStrategy(Strategy):
    def __init__(self) -> None:
        self.submitted = False

    def on_bar(self, bar: Bar):
        if self.submitted:
            return []
        self.submitted = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 10)]


def test_event_loop_records_risk_rejection_without_crashing():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, risk_rules=[MaxOrderValueRule(max_value=50)])
    bars = [Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100)]

    result = BacktestEngine(trade_on_close=True, broker=broker).run(bars, TooLargeOrderStrategy())

    assert result.summary()["fills"] == 0
    assert "order value" in result.records.orders()["reason"].drop_nulls().to_list()[-1]


def test_contract_multiplier_is_applied_to_cash_and_risk_checks():
    bar = Bar("FUT", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10)
    cash_limited_broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=1_000, contract_multipliers={"FUT": 100})

    with pytest.raises(ValueError, match="insufficient cash"):
        cash_limited_broker.execute(Order("FUT", bar.timestamp, Side.BUY, 10), bar)

    risk_limited_broker = BrokerSimulator(fill_price=FillPrice.CLOSE,
        initial_cash=20_000,
        contract_multipliers={"FUT": 100},
        risk_rules=[MaxPositionValueRule(max_value=1_000)],
    )
    with pytest.raises(ValueError, match="position value"):
        risk_limited_broker.execute(Order("FUT", bar.timestamp, Side.BUY, 10), bar)


class _MarkRecordingRule:
    """Records the ledger mark each risk check sees at fill time."""

    def __init__(self) -> None:
        self.marks: list[float | None] = []

    def check(self, context):
        position = context.state.positions.get("AAPL")
        self.marks.append(position.last_price if position else None)
        return None


class BuyOneEachBarStrategy(Strategy):
    def on_bar(self, bar):
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 1)]


def test_risk_checks_see_the_previous_close_when_filling_at_the_open():
    # With trade_on_close=False a bar fills at its open, so the equity the risk
    # engine evaluates must still be marked to the previous bar's close. Marking
    # the current close first would let a fill be sized against a price that has
    # not happened yet.
    rule = _MarkRecordingRule()
    broker = BrokerSimulator(initial_cash=100_000, fill_price=FillPrice.OPEN, risk_rules=[rule])
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100, volume=1_000),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 100, 200, 100, 200, volume=1_000),
        Bar("AAPL", datetime(2024, 1, 3, tzinfo=UTC), 300, 500, 300, 500, volume=1_000),
    ]

    BacktestEngine(broker=broker, trade_on_close=False).run(bars, BuyOneEachBarStrategy())

    # Checks run at submit time and at fill time. The bar-3 fill check (index 3)
    # must see bar 2's close of 200; before the fix it saw bar 3's close of 500,
    # a price the fill at bar 3's open could not have known.
    assert rule.marks == [None, None, 200, 200, 500]
