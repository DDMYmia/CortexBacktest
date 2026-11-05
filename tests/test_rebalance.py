from datetime import UTC, datetime

import pandas as pd
import polars as pl
import pytest

from cortexbacktest import Bar
from cortexbacktest.contracts import Position, Side
from cortexbacktest.event import (
    BacktestEngine,
    BrokerSimulator,
    RebalanceConfig,
    TargetWeightStrategy,
    orders_for_target_weights,
)
from cortexbacktest.research import run_factor_research
from cortexbacktest import FillPrice


def test_pinned_step_quantizes_trade_delta_without_overselling_fractional_dust():
    from cortexbacktest.event.rebalance import quantity_changes_for_target_weights

    config = RebalanceConfig(allow_fractional=True, quantity_steps=(("A", "0.01"),))
    args = dict(prices={"A": 100.}, equity=1000., config=config)
    assert quantity_changes_for_target_weights(target_weights={"A": 1.}, quantities={"A": 0.}, **args) == {"A": 10.}
    assert quantity_changes_for_target_weights(target_weights={"A": 0.}, quantities={"A": .0155}, **args) == {"A": -.01}
    assert quantity_changes_for_target_weights(target_weights={"A": 0.}, quantities={"A": .0055}, **args) == {}


@pytest.mark.parametrize("steps", [(("A", "0"),), (("A", "-0.01"),),
                                  (("A", "NaN"),), (("A", "0.01"), ("A", "0.02"))])
def test_invalid_or_duplicate_steps_cannot_become_a_sizing_policy(steps):
    with pytest.raises(ValueError, match="Quantity steps"):
        RebalanceConfig(allow_fractional=True, quantity_steps=steps)


def test_orders_for_target_weights_generates_buy_and_sell_orders():
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    orders = orders_for_target_weights(
        timestamp=timestamp,
        target_weights={"AAPL": 0.5, "MSFT": 0.0},
        prices={"AAPL": 100.0, "MSFT": 50.0},
        positions={"MSFT": Position("MSFT", 10, 50.0)},
        equity=10_000.0,
        config=RebalanceConfig(allow_fractional=False),
    )

    by_symbol = {order.symbol: order for order in orders}

    assert by_symbol["AAPL"].side == Side.BUY
    assert by_symbol["AAPL"].quantity == 50
    assert by_symbol["MSFT"].side == Side.SELL
    assert by_symbol["MSFT"].quantity == 10


def test_orders_for_target_weights_respects_cash_buffer_and_min_trade_value():
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    orders = orders_for_target_weights(
        timestamp=timestamp,
        target_weights={"AAPL": 0.5},
        prices={"AAPL": 100.0},
        positions={},
        equity=10_000.0,
        config=RebalanceConfig(cash_buffer=0.1, min_trade_value=5_000.0, allow_fractional=False),
    )

    assert orders == []


def test_target_weight_strategy_executes_through_event_engine():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    weights = pd.DataFrame({"AAPL": [1.0, 1.0]}, index=dates)
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110),
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=10_000)
    strategy = TargetWeightStrategy(
        weights,
        broker=broker,
        config=RebalanceConfig(allow_fractional=False),
    )

    result = BacktestEngine(broker=broker, trade_on_close=True).run(bars, strategy)

    assert result.summary()["fills"] == 1
    assert broker.state.positions["AAPL"].quantity == 100
    assert result.summary()["final_equity"] == 11_000


def test_target_weight_strategy_accepts_polars_target_weights():
    weights = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "aapl": [1.0, 1.0],
        }
    )
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110),
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=10_000)
    strategy = TargetWeightStrategy(
        weights,
        broker=broker,
        config=RebalanceConfig(allow_fractional=False),
    )

    result = BacktestEngine(broker=broker, trade_on_close=True).run(bars, strategy)

    assert isinstance(strategy.target_weights, pl.DataFrame)
    assert result.summary()["fills"] == 1
    assert broker.state.positions["AAPL"].quantity == 100
    assert result.summary()["final_equity"] == 11_000


def test_target_weight_strategy_exits_positions_missing_from_next_target_row():
    weights = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "AAPL": [1.0, 0.0],
            "MSFT": [0.0, 1.0],
        }
    )
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("MSFT", datetime(2024, 1, 1, tzinfo=UTC), 50, 50, 50, 50),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 100, 100, 100, 100),
        Bar("MSFT", datetime(2024, 1, 2, tzinfo=UTC), 50, 50, 50, 50),
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=10_000)
    strategy = TargetWeightStrategy(
        weights,
        broker=broker,
        config=RebalanceConfig(allow_fractional=False),
    )

    result = BacktestEngine(broker=broker, trade_on_close=True).run(bars, strategy)

    fills_by_symbol = [(fill.symbol, fill.side) for fill in result.fills]
    assert fills_by_symbol == [("AAPL", Side.BUY), ("AAPL", Side.SELL), ("MSFT", Side.BUY)]
    assert "AAPL" not in broker.state.positions
    assert broker.state.positions["MSFT"].quantity == 200


def test_signal_lag_sessions_counts_trading_sessions_not_matrix_rows():
    """A sparse (e.g. monthly) target-weight matrix must still lag by real
    trading sessions read off the bar feed, not by the matrix's own row
    positions — otherwise a lag of N sessions silently becomes N matrix rows,
    which can be many more sessions than intended (or, as here, unreachable
    entirely when the matrix has too few rows to index into)."""
    weights = pl.DataFrame({"date": [datetime(2024, 1, 1)], "AAPL": [1.0]})
    bars = [
        Bar("AAPL", datetime(2024, 1, day, tzinfo=UTC), 100, 100, 100, 100)
        for day in (1, 2, 3, 4, 5)
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=10_000)
    strategy = TargetWeightStrategy(
        weights,
        broker=broker,
        config=RebalanceConfig(allow_fractional=False),
        signal_lag_sessions=2,
    )

    result = BacktestEngine(broker=broker, trade_on_close=False).run(bars, strategy)

    # Session index 2 (2024-01-03) lagged by 2 sessions asof-matches the
    # single row at 2024-01-01, so the rebalance fires there.
    assert result.summary()["fills"] == 1
    assert broker.state.positions["AAPL"].quantity == 100


def test_orders_for_target_weights_accepts_polars_row_dict():
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    row = {"aapl": 0.5, "MSFT": None}

    orders = orders_for_target_weights(
        timestamp=timestamp,
        target_weights=row,
        prices={"AAPL": 100.0},
        positions={},
        equity=10_000.0,
        config=RebalanceConfig(allow_fractional=False),
    )

    assert len(orders) == 1
    assert orders[0].symbol == "AAPL"
    assert orders[0].quantity == 50


def test_polars_research_target_weights_feed_event_engine_directly():
    prices = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "AAPL": [100.0, 105.0, 110.0],
            "MSFT": [100.0, 99.0, 98.0],
        }
    )
    factors = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "AAPL": [1.0, 1.0, 1.0],
            "MSFT": [-1.0, -1.0, -1.0],
        }
    )
    report = run_factor_research(
        prices,
        factors,
        horizons=(1,),
        quantiles=2,
        top_quantile=0.5,
        bottom_quantile=None,
    )
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("MSFT", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 105, 105, 105, 105),
        Bar("MSFT", datetime(2024, 1, 2, tzinfo=UTC), 99, 99, 99, 99),
        Bar("AAPL", datetime(2024, 1, 3, tzinfo=UTC), 110, 110, 110, 110),
        Bar("MSFT", datetime(2024, 1, 3, tzinfo=UTC), 98, 98, 98, 98),
    ]
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=10_000)
    strategy = TargetWeightStrategy(
        report.target_weights_frame(),
        broker=broker,
        config=RebalanceConfig(allow_fractional=True),
    )

    result = BacktestEngine(broker=broker, trade_on_close=True).run(bars, strategy)

    assert result.summary()["fills"] > 0
    assert "AAPL" in broker.state.positions


def test_decision_times_are_strict_and_allow_multiple_intraday_decisions():
    from datetime import timedelta

    start = datetime(2024, 1, 1, 9, tzinfo=UTC)
    targets = pl.DataFrame({"date": [start, start + timedelta(hours=1)], "AAPL": [1., 0.]})
    broker = BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=1000)
    strategy = TargetWeightStrategy(targets, broker=broker, trade_at_open=True, decision_times=True)
    def bars(when):
        return [Bar("AAPL", when, 100, 100, 100, 100, 1000)]
    assert strategy.on_before_bars(bars(start)) == []
    orders = strategy.on_before_bars(bars(start + timedelta(minutes=1)))
    assert len(orders) == 1 and orders[0].side == Side.BUY
    broker.state.positions["AAPL"] = Position("AAPL", 10, 100)
    assert strategy.on_before_bars(bars(start + timedelta(hours=1))) == []
    orders = strategy.on_before_bars(bars(start + timedelta(hours=1, minutes=1)))
    assert len(orders) == 1 and orders[0].side == Side.SELL
