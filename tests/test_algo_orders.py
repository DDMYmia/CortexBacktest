from datetime import UTC, datetime, timedelta

from cortexbacktest import BacktestEngine, Bar, Order, Side, Strategy
from cortexbacktest.contracts import OrderType


def _bars(*, volumes: list[float] | None = None) -> list[Bar]:
    volumes = volumes or [100, 100, 100]
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Bar("AAPL", start + timedelta(days=index), 10, 10, 10, 10, volume=volume)
        for index, volume in enumerate(volumes)
    ]


class TwapStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.started = False

    def on_bar(self, bar: Bar):
        if not self.started:
            self.started = True
            self.twap_order(total_quantity=6, slices=3)
        return []


class VwapStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.started = False

    def on_bar(self, bar: Bar):
        if not self.started:
            self.started = True
            self.vwap_order(total_quantity=10, volume_share=0.1)
        return []


class IcebergStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.started = False

    def on_bar(self, bar: Bar):
        if not self.started:
            self.started = True
            self.iceberg_order(total_quantity=5, display_quantity=2)
        return []


def test_twap_order_emits_equal_child_orders():
    result = BacktestEngine(trade_on_close=True).run(_bars(), TwapStrategy())

    assert [fill.quantity for fill in result.fills] == [2, 2, 2]
    assert [record.order.tag for record in result.orders if record.status_label == "submitted"] == [
        "twap",
        "twap",
        "twap",
    ]


def test_vwap_order_participates_in_bar_volume():
    result = BacktestEngine(trade_on_close=True).run(_bars(volumes=[10, 20, 100]), VwapStrategy())

    assert [fill.quantity for fill in result.fills] == [1, 2, 7]


def test_iceberg_order_exposes_display_quantity_one_child_at_a_time():
    result = BacktestEngine(trade_on_close=True).run(_bars(), IcebergStrategy())

    assert [fill.quantity for fill in result.fills] == [2, 2, 1]
    assert [fill.side for fill in result.fills] == [Side.BUY, Side.BUY, Side.BUY]


class DeferredBracketStrategy(Strategy):
    """Parent limit order that only fills on a later bar, with a held child."""

    def __init__(self) -> None:
        super().__init__()
        self.started = False

    def on_bars(self, bars):
        if self.started:
            return []
        self.started = True
        bar = bars[0]
        parent = Order(
            bar.symbol, bar.timestamp, Side.BUY, 10,
            order_type=OrderType.LIMIT, limit_price=95.0, id="P1",
        )
        child = Order(
            bar.symbol, bar.timestamp, Side.SELL, 10,
            order_type=OrderType.LIMIT, limit_price=200.0, id="C1", parent_id="P1",
        )
        return [parent, child]


def test_held_bracket_child_survives_bars_before_parent_fill():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [
        Bar("AAPL", start, 100, 101, 99, 100),
        Bar("AAPL", start + timedelta(days=1), 100, 101, 90, 95),
        Bar("AAPL", start + timedelta(days=2), 100, 250, 95, 240),
    ]
    result = BacktestEngine(trade_on_close=True).run(bars, DeferredBracketStrategy())

    filled_ids = [record.order.id for record in result.orders if record.status_label == "filled"]
    assert filled_ids == ["P1", "C1"]


class WideBarOcoStrategy(Strategy):
    """OCO exit pair where one bar triggers both legs; only one may fill."""

    def __init__(self, *, padding_orders: int) -> None:
        super().__init__()
        self.started = False
        self.padding_orders = padding_orders

    def on_bars(self, bars):
        if self.started:
            return []
        self.started = True
        bar = bars[0]
        orders = [
            Order(
                "ZZZ", bar.timestamp, Side.BUY, 1,
                order_type=OrderType.LIMIT, limit_price=0.01, id=f"D{index}",
            )
            for index in range(self.padding_orders)
        ]
        orders.append(Order(
            "AAPL", bar.timestamp, Side.SELL, 10,
            order_type=OrderType.LIMIT, limit_price=105.0, id="TP", oco_group="EXIT",
        ))
        orders.append(Order(
            "AAPL", bar.timestamp, Side.SELL, 10,
            order_type=OrderType.STOP, stop_price=95.0, id="SL", oco_group="EXIT",
        ))
        return orders


def _wide_bar_oco_fills(padding_orders: int):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [
        Bar("AAPL", start, 100, 101, 99, 100),
        Bar("ZZZ", start, 100, 101, 99, 100),
        Bar("AAPL", start + timedelta(days=1), 100, 120, 90, 100),
        Bar("ZZZ", start + timedelta(days=1), 100, 101, 99, 100),
    ]
    strategy = WideBarOcoStrategy(padding_orders=padding_orders)
    result = BacktestEngine(trade_on_close=True).run(bars, strategy)
    return [fill for fill in result.fills if fill.symbol == "AAPL"]


def test_oco_fills_exactly_one_leg_linear_path():
    fills = _wide_bar_oco_fills(padding_orders=0)
    assert len(fills) == 1


def test_oco_fills_exactly_one_leg_indexed_path():
    # More than 32 open orders forces the symbol-indexed matching path.
    fills = _wide_bar_oco_fills(padding_orders=40)
    assert len(fills) == 1


def test_oco_linear_and_indexed_paths_agree():
    linear = _wide_bar_oco_fills(padding_orders=0)
    indexed = _wide_bar_oco_fills(padding_orders=40)
    assert [(f.order_id, f.side, f.quantity, f.price) for f in linear] == [
        (f.order_id, f.side, f.quantity, f.price) for f in indexed
    ]
