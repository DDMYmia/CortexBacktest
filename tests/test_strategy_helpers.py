from datetime import UTC, datetime

import pytest

from cortexbacktest import BacktestEngine, Bar, OrderType, Side, Strategy


class SingleAssetHelperStrategy(Strategy):
    def __init__(self) -> None:
        self.step = 0
        self.seen_positions = []

    def on_bar(self, bar: Bar):
        if self.step == 0:
            self.step += 1
            self.buy(size=3, tag="entry")
            return []
        if self.step == 1:
            position = self.position()
            self.seen_positions.append(position.quantity if position else 0)
            self.step += 1
            self.close(tag="exit")
        return []


def test_strategy_helpers_buy_position_and_close_single_asset():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 11, 11, 11, 11),
    ]

    result = BacktestEngine(trade_on_close=True).run(bars, SingleAssetHelperStrategy())

    assert [(fill.side, fill.quantity) for fill in result.fills] == [(Side.BUY, 3), (Side.SELL, 3)]
    assert result.positions == []
    assert result.orders[0].order.tag == "entry"
    assert result.orders[3].order.tag == "exit"
    submitted = [record.order for record in result.orders if record.status.value == "submitted"]
    assert [order.quantity for order in submitted] == [3, 3]


class MultiAssetHelperStrategy(Strategy):
    def __init__(self) -> None:
        self.submitted = False

    def on_bars(self, bars: list[Bar]):
        if self.submitted:
            return []
        self.submitted = True
        self.buy("aapl", quantity=2, limit_price=10.5)
        self.sell("msft", quantity=1, stop_price=19.5)
        return []


def test_strategy_helpers_support_multi_asset_orders():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("MSFT", datetime(2024, 1, 1, tzinfo=UTC), 20, 20, 20, 20),
    ]

    result = BacktestEngine(trade_on_close=True).run(bars, MultiAssetHelperStrategy())
    submitted = [record.order for record in result.orders if record.status.value == "submitted"]

    assert [(order.symbol, order.side, order.order_type) for order in submitted] == [
        ("AAPL", Side.BUY, OrderType.LIMIT),
        ("MSFT", Side.SELL, OrderType.STOP),
    ]


class AmbiguousSymbolStrategy(Strategy):
    def on_bars(self, bars: list[Bar]):
        self.buy(quantity=1)
        return []


def test_strategy_helpers_require_symbol_for_multi_asset_slice():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("MSFT", datetime(2024, 1, 1, tzinfo=UTC), 20, 20, 20, 20),
    ]

    with pytest.raises(ValueError, match="symbol is required"):
        BacktestEngine(trade_on_close=True).run(bars, AmbiguousSymbolStrategy())


class AmbiguousPositionStrategy(Strategy):
    def on_bars(self, bars: list[Bar]):
        self.position()
        return []


def test_strategy_position_requires_symbol_for_multi_asset_slice():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("MSFT", datetime(2024, 1, 1, tzinfo=UTC), 20, 20, 20, 20),
    ]

    with pytest.raises(ValueError, match="symbol is required"):
        BacktestEngine(trade_on_close=True).run(bars, AmbiguousPositionStrategy())


class ConflictingSizeStrategy(Strategy):
    def on_bar(self, bar: Bar):
        self.buy(quantity=1, size=2)
        return []


def test_strategy_helpers_reject_quantity_and_size_together():
    bars = [Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10)]

    with pytest.raises(ValueError, match="quantity or size"):
        BacktestEngine(trade_on_close=True).run(bars, ConflictingSizeStrategy())


class PartialCloseStrategy(Strategy):
    def __init__(self) -> None:
        self.step = 0

    def on_bar(self, bar: Bar):
        if self.step == 0:
            self.step += 1
            self.buy(size=10)
        elif self.step == 1:
            self.step += 1
            self.close(portion=0.25, tag="close-quarter")
        elif self.step == 2:
            self.step += 1
            self.close(size=3, tag="close-size")
        return []


def test_strategy_close_supports_portion_and_size():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 3, tzinfo=UTC), 10, 10, 10, 10),
    ]

    result = BacktestEngine(trade_on_close=True).run(bars, PartialCloseStrategy())

    assert [(fill.side, fill.quantity) for fill in result.fills] == [
        (Side.BUY, 10),
        (Side.SELL, 2.5),
        (Side.SELL, 3),
    ]
    assert [(position.symbol, position.quantity) for position in result.positions] == [("AAPL", 4.5)]
    submitted = [record.order for record in result.orders if record.status.value == "submitted"]
    assert [(order.quantity, order.tag) for order in submitted] == [
        (10, None),
        (2.5, "close-quarter"),
        (3, "close-size"),
    ]


class EmptyCloseStrategy(Strategy):
    def __init__(self) -> None:
        self.result = "unset"

    def on_bar(self, bar: Bar):
        self.result = self.close()
        return []


def test_strategy_close_returns_none_without_position():
    strategy = EmptyCloseStrategy()
    bars = [Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10)]

    result = BacktestEngine(trade_on_close=True).run(bars, strategy)

    assert strategy.result is None
    assert result.orders == []


def test_strategy_close_rejects_invalid_portion_after_position_exists():
    class InvalidAfterEntryStrategy(Strategy):
        def __init__(self) -> None:
            self.step = 0

        def on_bar(self, bar: Bar):
            if self.step == 0:
                self.step += 1
                self.buy(size=1)
            else:
                self.close(portion=1.5)
            return []

    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10),
    ]

    with pytest.raises(ValueError, match="portion"):
        BacktestEngine(trade_on_close=True).run(bars, InvalidAfterEntryStrategy())


def test_strategy_close_rejects_quantity_and_portion_after_position_exists():
    class ConflictingAfterEntryStrategy(Strategy):
        def __init__(self) -> None:
            self.step = 0

        def on_bar(self, bar: Bar):
            if self.step == 0:
                self.step += 1
                self.buy(size=1)
            else:
                self.close(size=1, portion=0.5)
            return []

    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10),
    ]

    with pytest.raises(ValueError, match="quantity/size or portion"):
        BacktestEngine(trade_on_close=True).run(bars, ConflictingAfterEntryStrategy())
