from datetime import datetime, timedelta, timezone

import pytest

from cortexbacktest import Bar, FillPrice, Order, OrderType, Side
from cortexbacktest.engine import BacktestEngine
from cortexbacktest.event.broker import BrokerSimulator
from cortexbacktest.event.strategy import Strategy


class OpeningOrders(Strategy):
    def __init__(self):
        super().__init__()
        self.seen = []

    def on_before_bars(self, bars):
        bar = bars[0]
        assert bar.volume is None and bar.open_interest is None
        self.seen.append(("open", self.timestamp, bar.close, self.bars[bar.symbol].high))
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 1),
                Order(bar.symbol, bar.timestamp, Side.BUY, 1,
                      order_type=OrderType.LIMIT, limit_price=95)]

    def on_bar(self, bar):
        self.seen.append(("close", self.timestamp, bar.close, self.bars[bar.symbol].high))
        return []


def test_opening_print_execution_preserves_close_marks_and_hides_later_range():
    close = datetime(2024, 1, 2, tzinfo=timezone.utc)
    opening = close - timedelta(days=1)
    bar = Bar("A", close, 100, 120, 90, 110, 1000)
    strategy = OpeningOrders()
    result = BacktestEngine(broker=BrokerSimulator(initial_cash=1000, fill_price=FillPrice.OPEN),
                           trade_on_close=False, opening_times={close: opening},
                           build_manifest=True).run([bar], strategy)
    assert strategy.seen == [("open", opening, 100, 100), ("close", close, 110, 120)]
    # The 95 limit only crossed later in the day: never label it an opening fill.
    assert len(result.fills) == 1
    assert result.fills[0].timestamp == opening
    assert result.fills[0].price == 100
    assert result.performance.equity_curve()[0, "timestamp"] == close
    assert result.performance.equity_curve()[0, "equity"] == 1010
    assert result.run_manifest["engine"]["opening_times"] == [[close.isoformat(), opening.isoformat()]]

    shifted = BacktestEngine(broker=BrokerSimulator(initial_cash=1000, fill_price=FillPrice.OPEN),
                            trade_on_close=False, opening_times={close: opening + timedelta(hours=1)},
                            build_manifest=True).run([bar], OpeningOrders())
    assert shifted.run_manifest["run_id"] != result.run_manifest["run_id"]
    assert shifted.run_manifest["data_sha256"] == result.run_manifest["data_sha256"]


@pytest.mark.parametrize("case", ["missing", "overlap", "naive", "close_fill", "trade_close"])
def test_invalid_opening_clock_never_produces_a_completed_backtest(case):
    close = datetime(2024, 1, 2, tzinfo=timezone.utc)
    second = close + timedelta(days=1)
    times = {close: close - timedelta(days=1), second: close}
    if case == "missing":
        del times[second]
    elif case == "overlap":
        times[second] -= timedelta(seconds=1)
    elif case == "naive":
        times[close] = times[close].replace(tzinfo=None)
    with pytest.raises(ValueError, match="opening|open|Session"):
        BacktestEngine(broker=BrokerSimulator(fill_price=FillPrice.CLOSE if case == "close_fill" else FillPrice.OPEN),
                       trade_on_close=case == "trade_close", opening_times=times).run(
                           [Bar("A", day, 100, 120, 90, 110) for day in (close, second)], OpeningOrders())
