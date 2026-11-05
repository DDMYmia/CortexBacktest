from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from cortexbacktest.broker import BrokerSimulator
from cortexbacktest.contracts import Bar
from cortexbacktest.engine import BacktestEngine, BacktestResult
from cortexbacktest.event.feed import ColumnarBarFeed
from cortexbacktest.event.loop import BacktestEngine as CoreEngine
from cortexbacktest.event.result import BacktestResult as ResultData
from cortexbacktest.strategy import Strategy
from cortexbacktest import FillPrice


class Observe(Strategy):
    def on_bar(self, bar):
        return []


def bars():
    return [Bar("X", datetime(2024, 1, day, tzinfo=UTC), 10, 10, 10, 10) for day in range(1, 6)]


@pytest.mark.parametrize("empty", [False, True])
def test_engine_cannot_be_reused_even_after_empty_run(empty):
    engine = BacktestEngine(trade_on_close=True)
    result = engine.run([] if empty else bars(), Observe())
    before = result.performance.equity_curve()
    with pytest.raises(RuntimeError, match="engine_factory"):
        engine.run(bars(), Observe())
    assert result.performance.equity_curve().equals(before)


def test_engine_cannot_be_reused_after_partial_failure():
    class Fail(Observe):
        def on_bar(self, bar):
            raise ValueError("strategy failed")

    engine = BacktestEngine(trade_on_close=True)
    with pytest.raises(ValueError, match="strategy failed"):
        engine.run(bars(), Fail())
    with pytest.raises(RuntimeError, match="including failed runs"):
        engine.run(bars(), Observe())


@pytest.mark.parametrize("trade_on_close", [True, False])
@pytest.mark.parametrize("hook", ["mark", "equity_snapshot_row"])
def test_instance_broker_hooks_are_observed_in_both_execution_modes(trade_on_close, hook):
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE)
    spy = Mock(wraps=getattr(broker, hook))
    setattr(broker, hook, spy)
    result = BacktestEngine(broker=broker, trade_on_close=trade_on_close).run(bars(), Observe())
    assert spy.call_count == 5
    assert result.performance.equity_curve()["equity"].to_list() == [100_000] * 5


@pytest.mark.parametrize("trade_on_close", [True, False])
def test_subclass_broker_hooks_are_observed(trade_on_close):
    class CustomBroker(BrokerSimulator):
        marks = 0
        snapshots = 0

        def mark(self, bar):
            self.marks += 1
            return super().mark(bar)

        def equity_snapshot_row(self, timestamp):
            self.snapshots += 1
            return super().equity_snapshot_row(timestamp)

    broker = CustomBroker(fill_price=FillPrice.CLOSE)
    BacktestEngine(broker=broker, trade_on_close=trade_on_close).run(bars(), Observe())
    assert (broker.marks, broker.snapshots) == (5, 5)


def test_default_manifest_does_not_hash_data(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("default engine must not hash bars")

    monkeypatch.setattr(ColumnarBarFeed, "data_fingerprint", forbidden)
    assert BacktestEngine(trade_on_close=True).run(bars(), Observe()).run_manifest is None


def test_environment_probe_is_deferred_until_manifest_serialization(monkeypatch):
    from cortexbacktest.event import manifest

    platform_spy = Mock(return_value="test-platform")
    python_spy = Mock(return_value="test-python")
    monkeypatch.setattr(manifest.platform, "platform", platform_spy)
    monkeypatch.setattr(manifest.platform, "python_version", python_spy)
    result = BacktestEngine(trade_on_close=True, build_manifest=True).run(bars(), Observe())
    platform_spy.assert_not_called()
    python_spy.assert_not_called()
    record = result.run_manifest
    assert record["platform"] == "test-platform"
    assert record["python_version"] == "test-python"
    assert result.run_manifest["run_id"] == record["run_id"]


def test_core_result_is_data_and_public_result_adds_reporting():
    core = CoreEngine().run(bars(), Observe())
    public = BacktestEngine(trade_on_close=True).run(bars(), Observe())
    assert type(core) is ResultData
    assert not hasattr(core, "performance")
    assert type(public) is BacktestResult
    assert isinstance(public, ResultData)
    assert public.performance.equity_curve().equals(core._equity_curve_pl)
