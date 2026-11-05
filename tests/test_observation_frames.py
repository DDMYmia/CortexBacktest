"""Observation files become engine inputs that join, and are stamped when known."""

from datetime import UTC, datetime

import pytest

from cortexbacktest import BacktestEngine, BrokerSimulator, Order, Side, Strategy
from cortexbacktest.adapters import instrument_symbol, observation_bars, observation_factors


def _bars():
    return {"recordVersion": "market-bars-v2", "records": [
        {"instrumentId": "7", "sessionDate": f"2026-09-1{day}", "availableAt": f"2026-09-1{day}T21:00:00+00:00",
         "sessionClose": f"2026-09-1{day}T20:00:00+00:00",
         "open": 10.0, "high": 20.0, "low": 9.0, "close": 10.0 + day, "volume": 100.0, "adjClose": 10.0 + day}
        for day in (6, 5)  # out of order on purpose
    ]}


def _factors():
    return {"recordVersion": "factor-observations-v1", "records": [
        {"instrument_id": "7", "factor_name": "news_sentiment", "known_at": "2026-09-15T22:00:00+00:00",
         "event_time": "2026-09-15T13:00:00+00:00", "value": 0.5},
    ]}


def test_bars_are_stamped_when_available_and_sorted():
    bars = observation_bars(_bars())
    assert [bar.timestamp for bar in bars] == [datetime(2026, 9, 15, 21, tzinfo=UTC), datetime(2026, 9, 16, 21, tzinfo=UTC)]
    assert {bar.symbol for bar in bars} == {instrument_symbol("7")}
    closes = observation_bars(_bars(), stamp="sessionClose")
    assert [bar.timestamp.hour for bar in closes] == [20, 20] and [bar.adj_close for bar in closes] == [15.0, 16.0]
    with pytest.raises(ValueError):
        observation_bars(_bars(), stamp="sessionDate")


def test_factors_are_keyed_by_known_at_not_event_time():
    frame = observation_factors(_factors())
    assert frame["available_at"].to_list() == [datetime(2026, 9, 15, 22, tzinfo=UTC)]
    assert frame["symbol"].to_list() == [instrument_symbol("7")]


def test_wrong_record_version_is_rejected():
    with pytest.raises(ValueError):
        observation_bars(_factors())
    with pytest.raises(ValueError):
        observation_factors(_bars())


def test_factor_joins_the_symbol_the_engine_trades():
    """The engine upper-cases symbols; a lower-case join key silently never trades."""
    factors = observation_factors(_factors())

    class Follow(Strategy):
        def on_bar(self, bar):
            known = factors.filter((factors["symbol"] == bar.symbol) & (factors["available_at"] <= bar.timestamp))
            return [] if known.is_empty() or getattr(self, "done", False) else [
                setattr(self, "done", True) or Order(bar.symbol, bar.timestamp, Side.BUY, 1)]

    result = BacktestEngine(broker=BrokerSimulator(initial_cash=1_000)).run(observation_bars(_bars()), Follow())
    assert result.summary()["orders"] == 1
