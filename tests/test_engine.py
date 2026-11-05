from datetime import UTC, datetime

from cortexbacktest import BacktestEngine, Bar, FillPrice, Order, OrderStatus, OrderType, Side, Strategy, TimeInForce
from cortexbacktest.broker import BrokerSimulator


class BuyOnceStrategy(Strategy):
    def __init__(self, quantity: int = 10) -> None:
        self.quantity = quantity
        self.bought = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if self.bought:
            return []
        self.bought = True
        return [Order(symbol=bar.symbol, timestamp=bar.timestamp, side=Side.BUY, quantity=self.quantity)]


class CountingBroker(BrokerSimulator):
    def __init__(self) -> None:
        super().__init__(fill_price=FillPrice.CLOSE)
        self.processed_bars = 0

    def process_bar(self, bar: Bar, *, expire_orders: bool = True, mark: bool = True):
        self.processed_bars += 1
        return super().process_bar(bar, expire_orders=expire_orders, mark=mark)


def test_backtest_engine_executes_market_order_and_marks_equity():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 12, 12, 12, 12),
    ]

    result = BacktestEngine(trade_on_close=True).run(bars, BuyOnceStrategy())

    assert len(result.fills) == 1
    assert result.performance.equity_curve()["equity"][-1] == 100_020
    assert result.records.fills().row(0, named=True)["symbol"] == "AAPL"
    assert result.trades.fill_summary()["fill_count"] == 1
    assert result.records.orders()["status"].is_in(["submitted", "accepted", "filled"]).all()


def test_trade_on_close_skips_fill_pass_when_no_open_orders():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 12, 12, 12, 12),
    ]
    broker = CountingBroker()

    result = BacktestEngine(trade_on_close=True, broker=broker).run(bars, BuyOnceStrategy())

    assert broker.processed_bars == 1
    assert result.summary()["fills"] == 1
    assert result.performance.equity_curve()["equity"][-1] == 100_020


class MarketSliceStrategy(Strategy):
    def __init__(self) -> None:
        self.days = []
        self.slices = []
        self.ordered = False

    def before_trading_day(self, trading_day, bars: list[Bar]) -> None:
        self.days.append((trading_day, [bar.symbol for bar in bars]))

    def on_bars(self, bars: list[Bar]) -> list[Order]:
        self.slices.append([bar.symbol for bar in bars])
        if self.ordered:
            return []
        self.ordered = True
        return [
            Order("AAPL", bars[0].timestamp, Side.BUY, 1),
            Order("MSFT", bars[0].timestamp, Side.BUY, 2),
        ]


def test_backtest_engine_passes_full_market_slice_to_strategy():
    bars = [
        Bar("MSFT", datetime(2024, 1, 1, tzinfo=UTC), 20, 20, 20, 20),
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 11, 11, 11, 11),
        Bar("MSFT", datetime(2024, 1, 2, tzinfo=UTC), 21, 21, 21, 21),
    ]
    strategy = MarketSliceStrategy()

    result = BacktestEngine(trade_on_close=True).run(bars, strategy)

    assert strategy.days == [
        (datetime(2024, 1, 1, tzinfo=UTC).date(), ["AAPL", "MSFT"]),
        (datetime(2024, 1, 2, tzinfo=UTC).date(), ["AAPL", "MSFT"]),
    ]
    assert strategy.slices == [["AAPL", "MSFT"], ["AAPL", "MSFT"]]
    assert len(result.fills) == 2
    assert result.performance.equity_curve()["timestamp"].to_list() == [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
    ]
    assert result.performance.equity_curve()["equity"][-1] == 100_003


def test_backtest_engine_can_delay_orders_until_next_bar():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 12, 12, 12, 12),
    ]

    result = BacktestEngine(trade_on_close=False).run(bars, BuyOnceStrategy())

    assert len(result.fills) == 1
    assert result.fills[0].timestamp == datetime(2024, 1, 2, tzinfo=UTC)
    assert result.fills[0].price == 12


class BuyLimitStrategy(Strategy):
    def __init__(self) -> None:
        self.submitted = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if self.submitted:
            return []
        self.submitted = True
        return [
            Order(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                side=Side.BUY,
                quantity=5,
                order_type=OrderType.LIMIT,
                limit_price=9.5,
            )
        ]


def test_backtest_engine_fills_limit_order_when_bar_trades_through_price():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 11, 9, 10),
    ]

    result = BacktestEngine(trade_on_close=False).run(bars, BuyLimitStrategy())

    assert len(result.fills) == 1
    assert result.fills[0].price == 9.5


class BuyStopStrategy(Strategy):
    def __init__(self) -> None:
        self.submitted = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if self.submitted:
            return []
        self.submitted = True
        return [
            Order(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                side=Side.BUY,
                quantity=5,
                order_type=OrderType.STOP,
                stop_price=11.0,
            )
        ]


def test_backtest_engine_fills_buy_stop_order_when_bar_triggers_stop():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10.5, 9.5, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 12, 10, 10.5),
    ]

    result = BacktestEngine(trade_on_close=False).run(bars, BuyStopStrategy())
    orders = result.records.orders()

    assert len(result.fills) == 1
    assert result.fills[0].price == 11.0
    assert result.fills[0].timestamp == datetime(2024, 1, 2, tzinfo=UTC)
    assert orders.row(0, named=True)["stop_price"] == 11.0


class SellStopLimitStrategy(Strategy):
    def __init__(self) -> None:
        self.submitted = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if self.submitted:
            return []
        self.submitted = True
        return [
            Order(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                side=Side.SELL,
                quantity=5,
                order_type=OrderType.STOP_LIMIT,
                stop_price=9.0,
                limit_price=8.8,
            )
        ]


def test_backtest_engine_fills_sell_stop_limit_after_stop_and_limit_are_reached():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10.5, 9.5, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10.2, 8.7, 9.5),
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, allow_short=True)

    result = BacktestEngine(broker=broker, trade_on_close=False).run(bars, SellStopLimitStrategy())

    assert len(result.fills) == 1
    assert result.fills[0].price == 9.0
    assert result.open_orders == []


def test_stop_limit_order_stays_open_after_stop_trigger_without_limit_fill():
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE)
    record = broker.submit_order(
        Order(
            "AAPL",
            timestamp,
            Side.BUY,
            1,
            order_type=OrderType.STOP_LIMIT,
            stop_price=11.0,
            limit_price=10.5,
        )
    )

    fills = broker.process_bar(Bar("AAPL", timestamp, 10, 12, 10.8, 11.5))

    assert record.status == OrderStatus.SUBMITTED
    assert fills == []
    assert len(broker.open_orders) == 1
    assert broker.open_orders[0].order_type == OrderType.STOP_LIMIT


def test_broker_rejects_stop_orders_without_stop_price():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE)
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)

    record = broker.submit_order(
        Order("AAPL", timestamp, Side.BUY, 1, order_type=OrderType.STOP)
    )

    assert record.status == OrderStatus.REJECTED
    assert record.reason == "stop orders require stop_price"


class UnfilledLimitStrategy(Strategy):
    def __init__(self) -> None:
        self.submitted = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if self.submitted:
            return []
        self.submitted = True
        return [
            Order(
                symbol=bar.symbol.lower(),
                timestamp=bar.timestamp,
                side=Side.BUY,
                quantity=5,
                order_type=OrderType.LIMIT,
                limit_price=1.0,
            )
        ]


def test_backtest_result_exposes_open_orders():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10),
    ]

    result = BacktestEngine(trade_on_close=False).run(bars, UnfilledLimitStrategy())
    open_orders = result.records.open_orders()

    assert len(result.fills) == 0
    assert len(result.open_orders) == 1
    assert result.open_orders[0].id == "O00000001"
    assert open_orders.row(0, named=True)["symbol"] == "AAPL"
    assert open_orders.row(0, named=True)["order_id"] == "O00000001"


def test_broker_can_cancel_open_order_by_id():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE)
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    record = broker.submit_order(
        Order(
            symbol="aapl",
            timestamp=timestamp,
            side=Side.BUY,
            quantity=1,
            order_type=OrderType.LIMIT,
            limit_price=1.0,
        )
    )

    cancelled = broker.cancel_order(record.order.id)
    broker.process_bar(Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10))

    assert cancelled is not None
    assert cancelled.status == OrderStatus.CANCELLED
    assert broker.open_orders == []
    assert broker.fills == []


class ExpiringLimitStrategy(Strategy):
    def __init__(self) -> None:
        self.submitted = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if self.submitted:
            return []
        self.submitted = True
        return [
            Order(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                side=Side.BUY,
                quantity=1,
                order_type=OrderType.LIMIT,
                limit_price=1.0,
                expires_at=bar.timestamp,
                time_in_force=TimeInForce.GTD,
            )
        ]


def test_backtest_engine_expires_stale_open_orders():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10),
    ]

    result = BacktestEngine(trade_on_close=False).run(bars, ExpiringLimitStrategy())
    statuses = result.records.orders()["status"].to_list()

    assert len(result.fills) == 0
    assert result.open_orders == []
    assert statuses == ["submitted", "accepted", "expired"]


class DayLimitStrategy(Strategy):
    def __init__(self) -> None:
        self.submitted = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if self.submitted:
            return []
        self.submitted = True
        return [
            Order(
                symbol=bar.symbol,
                timestamp=bar.timestamp,
                side=Side.BUY,
                quantity=1,
                order_type=OrderType.LIMIT,
                limit_price=1.0,
                time_in_force=TimeInForce.DAY,
            )
        ]


def test_day_order_expires_on_next_session():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10),
    ]

    result = BacktestEngine(trade_on_close=False).run(bars, DayLimitStrategy())

    assert result.fills == []
    assert result.open_orders == []
    assert [record.status for record in result.orders] == [
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.EXPIRED,
    ]
    assert result.orders[-1].order.time_in_force == TimeInForce.DAY


def test_broker_applies_commission_and_slippage():
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10)
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, commission_bps=10, slippage_bps=10)
    fill = broker.execute(Order("AAPL", bar.timestamp, Side.BUY, 10), bar)

    assert fill.price == 10.01
    assert round(fill.commission, 4) == 0.1001
    assert round(broker.state.cash, 4) == 99_899.7999


def test_broker_supports_multiple_fill_price_models():
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 16, 8, 14, vwap=12.5)
    expected = {
        FillPrice.OPEN: 10,
        FillPrice.CLOSE: 14,
        FillPrice.MID: 12,
        FillPrice.HLC3: (16 + 8 + 14) / 3,
        FillPrice.OHLC4: (10 + 16 + 8 + 14) / 4,
        FillPrice.VWAP: 12.5,
    }

    for fill_price, expected_price in expected.items():
        broker = BrokerSimulator(fill_price=fill_price)
        fill = broker.execute(Order("AAPL", bar.timestamp, Side.BUY, 1), bar)
        assert fill.price == expected_price


def test_vwap_fill_price_requires_vwap_data():
    broker = BrokerSimulator(fill_price=FillPrice.VWAP)
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 16, 8, 14)

    record = broker.submit_order(Order("AAPL", bar.timestamp, Side.BUY, 1))
    fills = broker.process_bar(bar)

    assert record.status == OrderStatus.SUBMITTED
    assert fills == []
    assert broker.order_records[-1].status == OrderStatus.REJECTED
    assert broker.order_records[-1].reason == "VWAP fill price requires bar.vwap"


class TrailingStopStrategy(Strategy):
    def __init__(self) -> None:
        self.bought = False
        self.stop_submitted = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if not self.bought:
            self.bought = True
            return [Order(symbol=bar.symbol, timestamp=bar.timestamp, side=Side.BUY, quantity=1)]
        if not self.stop_submitted:
            self.stop_submitted = True
            return [
                Order(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    side=Side.SELL,
                    quantity=1,
                    order_type=OrderType.TRAILING_STOP,
                    trail_amount=2.0,
                )
            ]
        return []


class OcoExitStrategy(Strategy):
    def __init__(self) -> None:
        self.step = 0

    def on_bar(self, bar: Bar) -> list[Order]:
        self.step += 1
        if self.step == 1:
            return [Order(symbol=bar.symbol, timestamp=bar.timestamp, side=Side.BUY, quantity=1)]
        if self.step == 2:
            return [
                Order(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    side=Side.SELL,
                    quantity=1,
                    order_type=OrderType.LIMIT,
                    limit_price=12.0,
                    id="TP",
                    oco_group="EXIT",
                ),
                Order(
                    symbol=bar.symbol,
                    timestamp=bar.timestamp,
                    side=Side.SELL,
                    quantity=1,
                    order_type=OrderType.STOP,
                    stop_price=9.0,
                    id="SL",
                    oco_group="EXIT",
                ),
            ]
        return []


class BracketStrategy(Strategy):
    def __init__(self) -> None:
        self.submitted = False

    def on_bar(self, bar: Bar) -> list[Order]:
        if self.submitted:
            return []
        self.submitted = True
        self.bracket(quantity=1, take_profit_price=12.0, stop_loss_price=9.0)
        return []


def test_broker_partially_fills_orders_against_bar_volume():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, max_volume_share=0.5)
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10, volume=100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10, volume=100),
    ]

    result = BacktestEngine(trade_on_close=True, broker=broker).run(bars, BuyOnceStrategy(quantity=80))
    orders = result.records.orders()
    fills = result.records.fills()

    assert [fill.quantity for fill in result.fills] == [50, 30]
    assert result.open_orders == []
    assert orders["status"].to_list() == ["submitted", "accepted", "partially_filled", "filled"]
    assert orders["quantity"].to_list() == [80, 80, 50, 30]
    assert fills["order_id"].to_list() == ["O00000001", "O00000001"]


def test_partial_fill_keeps_remaining_order_open_when_volume_is_exhausted():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, max_volume_share=0.25)
    bars = [Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10, volume=100)]

    result = BacktestEngine(trade_on_close=True, broker=broker).run(bars, BuyOnceStrategy(quantity=80))

    assert [fill.quantity for fill in result.fills] == [25]
    assert len(result.open_orders) == 1
    assert result.open_orders[0].quantity == 55
    assert result.open_orders[0].id == "O00000001"


def test_trailing_stop_updates_stop_price_and_fills_on_reversal():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10, volume=100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 13, 13, 12, 13, volume=100),
        Bar("AAPL", datetime(2024, 1, 3, tzinfo=UTC), 15, 15, 14, 15, volume=100),
        Bar("AAPL", datetime(2024, 1, 4, tzinfo=UTC), 12, 14, 12, 12, volume=100),
    ]

    result = BacktestEngine(trade_on_close=True).run(bars, TrailingStopStrategy())

    assert [fill.side for fill in result.fills] == [Side.BUY, Side.SELL]
    assert result.fills[-1].price == 12
    trailing_records = [record for record in result.orders if record.order.order_type == OrderType.TRAILING_STOP]
    assert trailing_records[-1].status == OrderStatus.FILLED
    assert trailing_records[-1].order.stop_price == 13


def test_trailing_stop_does_not_trigger_on_the_bar_that_set_it():
    # Bar 3 spikes to 20, which ratchets the sell stop to 18. That level must
    # protect bar 4 onwards: testing bar 3's own low against it would be
    # look-ahead and would sell at 10 on the spike bar itself.
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10, volume=100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10, volume=100),
        Bar("AAPL", datetime(2024, 1, 3, tzinfo=UTC), 10, 20, 10, 10, volume=100),
        Bar("AAPL", datetime(2024, 1, 4, tzinfo=UTC), 19, 19, 19, 19, volume=100),
    ]

    result = BacktestEngine(trade_on_close=True).run(bars, TrailingStopStrategy())

    assert [fill.side for fill in result.fills] == [Side.BUY]
    assert len(result.open_orders) == 1
    assert result.open_orders[0].order_type == OrderType.TRAILING_STOP
    assert result.open_orders[0].stop_price == 18


def test_oco_orders_cancel_sibling_after_fill():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10, volume=100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 12, 12, 10, 12, volume=100),
    ]

    result = BacktestEngine(trade_on_close=True).run(bars, OcoExitStrategy())

    assert [fill.side for fill in result.fills] == [Side.BUY, Side.SELL]
    statuses = {record.order.id: record.status for record in result.orders if record.order.id in {"TP", "SL"}}
    assert statuses["TP"] == OrderStatus.FILLED
    assert statuses["SL"] == OrderStatus.CANCELLED
    assert result.open_orders == []


def test_bracket_order_activates_children_and_cancels_oco_sibling():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10, volume=100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 12, 12, 10, 12, volume=100),
    ]

    result = BacktestEngine(trade_on_close=True).run(bars, BracketStrategy())

    assert [fill.side for fill in result.fills] == [Side.BUY, Side.SELL]
    filled = [record for record in result.orders if record.status == OrderStatus.FILLED]
    cancelled = [record for record in result.orders if record.status == OrderStatus.CANCELLED]
    assert len(filled) == 2
    assert len(cancelled) == 1
    assert cancelled[0].reason == "oco sibling filled"
