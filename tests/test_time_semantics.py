from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from cortexbacktest import BacktestEngine, Bar, BrokerSimulator, Order, Side, Strategy
from cortexbacktest.data.calendar import TradingCalendar
from cortexbacktest.event.financing import SettlementModel
from cortexbacktest.event.timeframes import replay_bars, resample_bars
from cortexbacktest import FillPrice


def minutes():
    return [Bar("X", datetime(2026, 9, 4, 9, 30, tzinfo=UTC) + timedelta(minutes=i),
                100+i, 100+i, 100+i, 100+i) for i in range(5)]


def test_completed_high_timeframe_only_reaches_strategy_at_bucket_end():
    seen = []

    class Observe(Strategy):
        def on_bar(self, bar):
            seen.append((bar.timestamp, bar.close))
            return []

    feed = resample_bars(minutes(), "5m", source_timeframe="1m")
    BacktestEngine(trade_on_close=True).run(feed, Observe())
    assert seen == [(datetime(2026, 9, 4, 9, 35, tzinfo=UTC), 104)]
    assert resample_bars(minutes(), "5m", timestamp_label="start")[0].timestamp.minute == 30


def test_replay_preserves_source_availability_and_never_rewrites_history():
    output = list(replay_bars(minutes(), "5m", source_timeframe="1m"))
    assert [bar.timestamp.minute for bar in output] == [31, 32, 33, 34, 35]
    assert [bar.close for bar in output] == [100, 101, 102, 103, 104]
    assert output[:3] == list(replay_bars(minutes()[:3], "5m", source_timeframe="1m"))


def test_straddling_source_bars_are_rejected():
    with pytest.raises(ValueError, match="straddles"):
        resample_bars(minutes(), "3m", source_timeframe="5m")
    with pytest.raises(ValueError, match="align"):
        resample_bars(minutes()[1:], "5m", source_timeframe="5m")


def test_dst_repeated_hour_is_not_merged():
    zone = ZoneInfo("America/New_York")
    bars = [Bar("X", datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=fold),
                100+fold, 100+fold, 100+fold, 100+fold) for fold in [0, 1]]
    output = resample_bars(bars, "1h")
    assert len(output) == 2
    assert [bar.timestamp.astimezone(UTC).hour for bar in output] == [6, 7]
    assert [bar.close for bar in output] == [100, 101]


def calendar():
    # Monday is an explicitly excluded holiday.
    return TradingCalendar.from_sessions(["2026-09-04", "2026-09-08", "2026-09-09"])


def test_trading_settlement_skips_weekends_and_supplied_holidays():
    start = datetime(2026, 9, 4, 15, tzinfo=UTC)
    model = SettlementModel.trading_days(calendar(), delay_days=2)
    assert model.settlement_time(start) == start.replace(day=9)
    assert SettlementModel(2).settlement_time(start) == start.replace(day=6)


def test_market_timezone_settlement_preserves_wall_time_across_dst():
    sessions = TradingCalendar.from_sessions(["2026-03-06", "2026-03-09"])
    model = SettlementModel.trading_days(sessions, timezone="America/New_York")
    assert model.settlement_time(datetime(2026, 3, 6, 20, tzinfo=UTC)) == datetime(2026, 3, 9, 19, tzinfo=UTC)
    with pytest.raises(ValueError, match="timezone-aware"):
        model.settlement_time(datetime(2026, 3, 6, 15))


def test_calendar_exhaustion_does_not_partially_mutate_ledger():
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, settlement_model=SettlementModel.trading_days(calendar(), delay_days=2))
    timestamp = datetime(2026, 9, 9, tzinfo=UTC)
    bar = Bar("X", timestamp, 100, 100, 100, 100)
    with pytest.raises(ValueError, match="cover settlement"):
        broker.execute(Order("X", timestamp, Side.SELL, 1), bar)
    assert broker.state.cash == 100_000
    assert broker.state.positions == {}
    assert broker.ledger.fill_count == 0
    assert broker.ledger.total_commission == 0


def test_trading_calendar_is_in_manifest_and_cash_remains_pending():
    class Sell(Strategy):
        def on_bar(self, bar):
            return [Order("X", bar.timestamp, Side.SELL, 1)] if bar.timestamp.day == 4 else []

    bars = [Bar("X", datetime(2026, 9, day, tzinfo=UTC), 100, 100, 100, 100) for day in [4, 8, 9]]
    result = BacktestEngine(trade_on_close=True, build_manifest=True, broker=BrokerSimulator(fill_price=FillPrice.CLOSE,
        settlement_model=SettlementModel.trading_days(calendar(), delay_days=2))).run(bars, Sell())
    assert result.performance.equity_curve()["pending_cash"].to_list() == [100, 100, 0]
    assert result.run_manifest["broker"]["settlement_model"]["settings"]["sessions"] == ["2026-09-04", "2026-09-08", "2026-09-09"]
