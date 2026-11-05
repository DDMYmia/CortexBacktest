"""Regression tests for the runtime paths touched while clearing the basedpyright baseline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from cortexbacktest import Bar, Strategy
from cortexbacktest.api import BacktestConfig, BrokerConfig, run_backtest
from cortexbacktest.event.strategy_loader import _instantiate_strategy
from cortexbacktest.reporting.summary import _json_value


class _Noop(Strategy):
    def on_bar(self, bar):
        return []


def _bars(count: int = 3) -> list[Bar]:
    start = datetime(2024, 1, 2, tzinfo=UTC)
    return [Bar("SPY", start + timedelta(days=i), 100.0, 101.0, 99.0, 100.5, 1e6) for i in range(count)]


def test_strategy_factory_must_return_a_strategy() -> None:
    with pytest.raises(TypeError, match="must return a Strategy"):
        _instantiate_strategy(lambda: object(), kwargs={}, broker=None, dataset=None)


def test_strategy_factory_returning_a_strategy_is_accepted() -> None:
    strategy = _instantiate_strategy(_Noop, kwargs={}, broker=None, dataset=None)
    assert isinstance(strategy, _Noop)


def test_run_backtest_accepts_a_generator_of_bars() -> None:
    config = BacktestConfig(broker=BrokerConfig(initial_cash=10_000))
    from_list = run_backtest(_bars(), _Noop(), config=config)
    from_generator = run_backtest((bar for bar in _bars()), _Noop(), config=config)
    assert from_generator.summary()["final_equity"] == from_list.summary()["final_equity"]


def test_json_value_handles_plain_and_numpy_scalars() -> None:
    assert _json_value(1.5) == 1.5
    assert _json_value(float("nan")) is None
    assert _json_value(np.float64(2.0)) == 2.0
    assert _json_value(np.int64(3)) == 3
