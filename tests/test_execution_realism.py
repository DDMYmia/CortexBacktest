from cortexbacktest import FillPrice
from datetime import UTC, datetime

from cortexbacktest import (
    BacktestEngine,
    Bar,
    BidAskSpreadMatching,
    BrokerSimulator,
    CashInterestModel,
    MarketStateMatching,
    Order,
    OrderType,
    SettlementModel,
    ShortBorrowFeeModel,
    Side,
    Strategy,
    event_tear_sheet,
    event_tear_sheet_html,
)


class BuyOnce(Strategy):
    def __init__(self, quantity=100):
        super().__init__()
        self.quantity = quantity
        self.done = False

    def on_bar(self, bar):
        if self.done:
            return []
        self.done = True
        self.buy(bar.symbol, quantity=self.quantity)
        return []


def test_bid_ask_matching_applies_spread_and_queue_participation():
    bars = [Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100, volume=100)]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, matching_model=BidAskSpreadMatching(spread_bps=100, participation=0.5))

    result = BacktestEngine(trade_on_close=True, broker=broker).run(bars, BuyOnce(quantity=100))

    assert result.fills[0].quantity == 50
    assert result.fills[0].price == 100.5
    assert result.open_order_count == 1


def test_bid_ask_matching_fills_limits_the_bar_range_traded_through():
    # The bar closes at 100 (ask 100.5), so a buy limit at 96 is not marketable
    # against the close-derived quote — but the bar traded down to 95, so the
    # limit did trade. It must fill, capped at the limit price.
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    bar = Bar("AAPL", timestamp, 100, 105, 95, 100, volume=1_000)
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, matching_model=BidAskSpreadMatching(spread_bps=100))
    broker.submit_order(Order("AAPL", timestamp, Side.BUY, 10, order_type=OrderType.LIMIT, limit_price=96))
    unreachable = BrokerSimulator(fill_price=FillPrice.CLOSE, matching_model=BidAskSpreadMatching(spread_bps=100))
    unreachable.submit_order(
        Order("AAPL", timestamp, Side.BUY, 10, order_type=OrderType.LIMIT, limit_price=94)
    )

    fills = broker.process_bar(bar)

    assert [fill.price for fill in fills] == [96]
    # A limit the bar never reached still does not fill.
    assert unreachable.process_bar(bar) == []


def test_market_state_matching_defers_orders_during_halts():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100, halted=True),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 101, 101, 101, 101),
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, matching_model=MarketStateMatching())

    result = BacktestEngine(trade_on_close=True, broker=broker).run(bars, BuyOnce(quantity=10))

    assert len(result.fills) == 1
    assert result.fills[0].timestamp == bars[1].timestamp
    assert result.fills[0].price == 101


def test_market_state_matching_respects_locked_limit_direction():
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    locked_up = Bar("AAPL", timestamp, 110, 110, 110, 110, limit_up=110, limit_down=90)
    buy_broker = BrokerSimulator(fill_price=FillPrice.CLOSE, matching_model=MarketStateMatching())
    sell_broker = BrokerSimulator(fill_price=FillPrice.CLOSE, matching_model=MarketStateMatching())
    buy_broker.submit_order(Order("AAPL", timestamp, Side.BUY, 1))
    sell_broker.submit_order(Order("AAPL", timestamp, Side.SELL, 1))

    assert buy_broker.process_bar(locked_up) == []
    assert len(sell_broker.process_bar(locked_up)) == 1
    assert len(buy_broker.open_orders) == 1


class BuyThenSell(Strategy):
    def __init__(self):
        super().__init__()
        self.step = 0

    def on_bar(self, bar):
        if self.step == 0:
            self.step += 1
            self.buy(bar.symbol, quantity=10)
        elif self.step == 1:
            self.step += 1
            self.sell(bar.symbol, quantity=10)
        return []


def test_t_plus_one_settlement_keeps_sale_proceeds_pending_until_next_day():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 11, 11, 11, 11),
        Bar("AAPL", datetime(2024, 1, 3, tzinfo=UTC), 11, 11, 11, 11),
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=1_000, settlement_model=SettlementModel(delay_days=1))

    result = BacktestEngine(trade_on_close=True, broker=broker).run(bars, BuyThenSell())
    rows = result.performance.equity_curve().to_dicts()

    assert rows[1]["cash"] == 900
    assert rows[1]["pending_cash"] == 110
    assert rows[2]["cash"] == 1010
    assert rows[2]["pending_cash"] == 0


class ShortOnce(Strategy):
    def __init__(self):
        super().__init__()
        self.done = False

    def on_bar(self, bar):
        if not self.done:
            self.done = True
            self.sell(bar.symbol, quantity=10)
        return []


def test_short_borrow_fee_accrues_on_next_trading_day():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 10, 10, 10, 10),
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=1_000, short_borrow_fee_model=ShortBorrowFeeModel(annual_rate=0.365))

    result = BacktestEngine(trade_on_close=True, broker=broker).run(bars, ShortOnce())

    assert round(broker.financing_state.charges[0].amount, 4) == -0.1
    assert round(result.performance.equity_curve()["equity"][-1], 4) == 999.9


def test_cash_interest_model_accrues_credit_interest():
    model = CashInterestModel(credit_rate=0.365)

    assert model.charge(1_000, 1) == 1.0


def test_event_tear_sheet_builds_compact_report_and_html():
    bars = [Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10)]
    result = BacktestEngine(trade_on_close=True).run(bars, BuyOnce(quantity=1))

    sheet = event_tear_sheet(result)
    html = event_tear_sheet_html(result)

    assert sheet["counts"]["fills"] == 1
    assert "CortexBacktest Event Tear Sheet" in html
