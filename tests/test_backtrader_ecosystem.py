from cortexbacktest import FillPrice
import polars as pl
from datetime import UTC, datetime, timedelta

import pytest

from cortexbacktest import (
    BacktestEngine,
    BacktraderStrategyMixin,
    Bar,
    BrokerSimulator,
    BuyAndHoldStrategy,
    CommissionScheme,
    DrawdownAnalyzer,
    FillStatsAnalyzer,
    FixedCommission,
    FixedSizeSizer,
    INDICATORS,
    MarginRequirementRule,
    Order,
    PercentCashSizer,
    PerShareCommission,
    Side,
    SignalStrategy,
    Strategy,
    TargetPercentSizer,
    TieredCommission,
    replay_bars,
    resample_bars,
)


def _bars():
    start = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    return [
        Bar("AAPL", start, 10, 12, 9, 11, volume=100, vwap=10.5),
        Bar("AAPL", start + timedelta(minutes=1), 11, 13, 10, 12, volume=150, vwap=12.0),
        Bar("MSFT", start + timedelta(minutes=1), 20, 21, 19, 20, volume=50),
        Bar("AAPL", start + timedelta(minutes=5), 12, 14, 11, 13, volume=200, vwap=13.0),
    ]


def test_indicator_registry_runs_indicators():
    prices = pl.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-03"], "A": [10.0, 11.0, 12.0]})

    result = INDICATORS.run("momentum", prices, window=1)

    assert "momentum" in INDICATORS.names()
    assert round(result["A"][-1], 4) == round(1 / 11, 4)


def test_resample_bars_and_replay_bars_aggregate_ohlcv():
    resampled = resample_bars(_bars(), "5m")
    replayed = list(replay_bars(_bars()[:2], "5m"))

    aapl = [bar for bar in resampled if bar.symbol == "AAPL"]
    assert len(aapl) == 2
    assert aapl[0].open == 10
    assert aapl[0].high == 13
    assert aapl[0].low == 9
    assert aapl[0].close == 12
    assert aapl[0].volume == 250
    assert round(aapl[0].vwap, 4) == round((10.5 * 100 + 12.0 * 150) / 250, 4)
    assert replayed[-1].close == 12
    assert replayed[-1].volume == 250


def test_resample_bars_keeps_latest_open_interest_and_market_limits():
    timestamp = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    bars = [
        Bar("IF", timestamp, 10, 11, 9, 10, open_interest=100, halted=True),
        Bar(
            "IF",
            timestamp + timedelta(minutes=1),
            10,
            12,
            10,
            11,
            open_interest=120,
            limit_up=13,
            limit_down=8,
        ),
    ]

    result = resample_bars(bars, "5m")[0]

    assert result.open_interest == 120
    assert result.halted is False
    assert result.limit_up == 13
    assert result.limit_down == 8


def test_new_analyzers_are_opt_in_and_report_results():
    result = BacktestEngine(trade_on_close=True, 
        analyzers=[DrawdownAnalyzer(), FillStatsAnalyzer()],
    ).run(_bars(), BuyAndHoldStrategy(quantity=2))

    assert "drawdown" in result.analyzer_results
    assert result.analyzer_results["fill_stats"]["fills"] == 2
    assert result.analyzer_results["fill_stats"]["by_symbol"]["AAPL"]["quantity"] == 2


def test_commission_models_and_margin_scheme():
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10)
    order = Order("AAPL", bar.timestamp, Side.BUY, 10)

    assert FixedCommission(2.5).calculate(order, 10) == 2.5
    assert PerShareCommission(0.01, minimum=1.0).calculate(order, 10) == 1.0
    assert TieredCommission(((100, 0.02), (1_000, 0.01))).calculate(order, 10) == 0.2

    broker = BrokerSimulator(fill_price=FillPrice.CLOSE,
        initial_cash=100,
        commission_scheme=CommissionScheme(FixedCommission(0), initial_margin=1.0),
        allow_short=True,
    )
    with pytest.raises(ValueError, match="margin requirement"):
        broker.execute(Order("AAPL", bar.timestamp, Side.SELL, 20), bar)

    assert MarginRequirementRule(initial_margin=0.5)


def test_sizers_and_target_order_helpers():
    bar = Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10)
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=1_000)

    assert FixedSizeSizer(5).size(broker=broker, bar=bar) == 5
    assert PercentCashSizer(0.5).size(broker=broker, bar=bar) == 50
    assert TargetPercentSizer(0.2).size(broker=broker, bar=bar) == 20

    class TargetStrategy(Strategy):
        def on_bar(self, bar):
            order = self.order_target_percent(bar.symbol, target_percent=0.5)
            return [order] if order else []

    result = BacktestEngine(trade_on_close=True, broker=broker).run([bar], TargetStrategy())

    assert result.fills[0].quantity == 50


def test_signal_strategy_and_backtrader_migration_mixin():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, 10, 10, 10),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 11, 11, 11, 11),
    ]

    signal_result = BacktestEngine(trade_on_close=True).run(
        bars,
        SignalStrategy(lambda bar, strategy: 1 if bar.close == 10 else 0, quantity=3),
    )

    class MigratedStrategy(BacktraderStrategyMixin):
        def __init__(self):
            super().__init__()
            self.called = 0

        def next(self):
            self.called += 1
            if self.called == 1:
                self.buy("AAPL", quantity=1)
            return []

    migrated = MigratedStrategy()
    migrated_result = BacktestEngine(trade_on_close=True).run(bars, migrated)

    assert signal_result.fills[0].quantity == 3
    assert migrated.called == 2
    assert migrated_result.fills[0].quantity == 1
