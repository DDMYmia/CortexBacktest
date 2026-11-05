from datetime import UTC, datetime
import glob
import tempfile

import polars as pl
import pytest

from cortexbacktest import (
    Bar,
    BrokerSimulator,
    FactorSweepCase,
    RebalanceConfig,
    momentum,
    run_execution_sweep,
    run_factor_sweep,
)
from cortexbacktest.event.feed import ColumnarBarFeed
from cortexbacktest.event.feed_storage import read_feed_ipc, write_feed_ipc


_DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]


def _prices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": _DATES,
            "AAPL": [100.0, 102.0, 103.0, 104.0, 106.0, 108.0],
            "MSFT": [100.0, 99.0, 98.0, 97.0, 98.0, 99.0],
            "NVDA": [100.0, 101.0, 104.0, 108.0, 111.0, 115.0],
        }
    )


def _bars(prices: pl.DataFrame) -> list[Bar]:
    symbols = [column for column in prices.columns if column != "date"]
    prices = prices.with_columns(pl.col("date").str.to_datetime())
    bars = []
    for row in prices.iter_rows(named=True):
        day = row["date"]
        timestamp = datetime(day.year, day.month, day.day, tzinfo=UTC)
        for symbol in symbols:
            price = float(row[symbol])
            bars.append(
                Bar(symbol=symbol, timestamp=timestamp, open=price, high=price, low=price, close=price, volume=10_000)
            )
    return bars


def test_run_execution_sweep_executes_factor_sweep_target_weights():
    prices = _prices()
    factor_sweep = run_factor_sweep(
        prices,
        [
            FactorSweepCase("mom_1", lambda frame: momentum(frame, window=1), top_quantile=1 / 3, bottom_quantile=None),
            FactorSweepCase("mom_2", lambda frame: momentum(frame, window=2), top_quantile=1 / 3, bottom_quantile=None),
        ],
        horizons=(1,),
        quantiles=3,
    )

    execution = run_execution_sweep(
        factor_sweep,
        _bars(prices),
        initial_cash=100_000,
        commission_bps=1.0,
        slippage_bps=1.0,
        rebalance_config=RebalanceConfig(cash_buffer=0.01, min_trade_value=500.0, allow_fractional=False),
    )
    summary = execution.summary()

    assert set(execution.event_results) == {"mom_1", "mom_2"}
    assert all(not result.records_retained for result in execution.event_results.values())
    assert all(result.run_manifest is None for result in execution.event_results.values())
    assert isinstance(summary, pl.DataFrame)
    assert {"case", "event_final_equity", "research_avg_rank_ic", "commission", "slippage"}.issubset(summary.columns)
    assert set(summary["case"].to_list()) == {"mom_1", "mom_2"}
    assert (summary["event_final_equity"] > 0).all()
    assert (summary["fills"] > 0).any()
    assert (summary["commission"] > 0).any()
    assert (summary["slippage"] >= 0).all()
    assert summary.filter(pl.col("case") == "mom_1")["fills"].item() > 0
    assert execution.best_case() in {"mom_1", "mom_2"}


def test_top_level_exports_broker_simulator():
    assert BrokerSimulator(initial_cash=100_000).state.cash == 100_000


def test_run_execution_sweep_parallel_process_matches_serial_results():
    prices = _prices()
    factor_sweep = run_factor_sweep(
        _prices(),
        [
            FactorSweepCase(
                "mom_1",
                lambda frame: momentum(frame, window=1),
                top_quantile=1 / 3,
                bottom_quantile=None,
            ),
            FactorSweepCase(
                "mom_2",
                lambda frame: momentum(frame, window=2),
                top_quantile=1 / 3,
                bottom_quantile=None,
            ),
        ],
        horizons=(1,),
        quantiles=3,
    )
    kwargs = {
        "initial_cash": 100_000,
        "commission_bps": 1.0,
        "slippage_bps": 1.0,
        "rebalance_config": RebalanceConfig(cash_buffer=0.01, min_trade_value=500.0, allow_fractional=False),
    }

    serial = run_execution_sweep(factor_sweep, _bars(prices), **kwargs)
    parallel = run_execution_sweep(factor_sweep, _bars(prices), max_workers=2, parallel_backend="process", **kwargs)

    assert list(parallel.event_results) == list(serial.event_results)
    serial_summary = serial.summary().sort("case")
    parallel_summary = parallel.summary().sort("case")
    assert parallel_summary.select(["case", "event_final_equity", "orders", "fills", "order_records"]).rows() == (
        serial_summary.select(["case", "event_final_equity", "orders", "fills", "order_records"]).rows()
    )


def test_run_execution_sweep_parallel_thread_allows_custom_broker_factory():
    prices = _prices()
    factor_sweep = run_factor_sweep(
        prices,
        [
            FactorSweepCase("mom_1", lambda frame: momentum(frame, window=1), top_quantile=1 / 3, bottom_quantile=None),
            FactorSweepCase("mom_2", lambda frame: momentum(frame, window=2), top_quantile=1 / 3, bottom_quantile=None),
        ],
        horizons=(1,),
        quantiles=3,
    )

    def broker_factory():
        from cortexbacktest.event import BrokerSimulator

        return BrokerSimulator(initial_cash=100_000, commission_bps=1.0)

    execution = run_execution_sweep(
        factor_sweep,
        _bars(prices),
        broker_factory=broker_factory,
        max_workers=2,
        parallel_backend="thread",
    )

    assert set(execution.event_results) == {"mom_1", "mom_2"}
    assert execution.summary()["event_final_equity"].min() > 0


def test_run_execution_sweep_process_rejects_custom_broker_factory():
    prices = _prices()
    factor_sweep = run_factor_sweep(
        prices,
        [FactorSweepCase("mom_1", lambda frame: momentum(frame, window=1), top_quantile=1 / 3, bottom_quantile=None)],
        horizons=(1,),
        quantiles=3,
    )

    with pytest.raises(ValueError, match="process execution sweeps require default broker construction"):
        run_execution_sweep(
            factor_sweep,
            _bars(prices),
            broker_factory=lambda: None,
            max_workers=2,
            parallel_backend="process",
        )


def test_execution_sweep_rejects_unknown_best_case_metric():
    prices = _prices()
    factor_sweep = run_factor_sweep(
        prices,
        [FactorSweepCase("mom_1", lambda frame: momentum(frame, window=1), top_quantile=1 / 3, bottom_quantile=None)],
        horizons=(1,),
        quantiles=3,
    )
    execution = run_execution_sweep(factor_sweep, _bars(prices))

    with pytest.raises(ValueError, match="Unknown metric"):
        execution.best_case(metric="missing")


def test_bars_ipc_round_trip_preserves_every_field(tmp_path):
    bars = [
        Bar(
            "AAPL",
            datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1_000.0,
            adj_close=100.4,
            vwap=100.2,
            open_interest=None,
            halted=False,
            limit_up=110.0,
            limit_down=90.0,
        ),
        Bar("MSFT", datetime(2024, 1, 3, 9, 30, tzinfo=UTC), 200.0, 201.0, 199.0, 200.5),
    ]

    path = tmp_path / "bars.arrow"
    write_feed_ipc(ColumnarBarFeed.from_bars(bars), path)
    rebuilt = read_feed_ipc(path).bars

    assert rebuilt == bars


def test_execution_sweep_process_backend_cleans_up_its_ipc_temp_file():
    prices = _prices()
    factor_sweep = run_factor_sweep(
        prices,
        [FactorSweepCase("mom_1", lambda frame: momentum(frame, window=1), top_quantile=1 / 3, bottom_quantile=None),
         FactorSweepCase("mom_2", lambda frame: momentum(frame, window=2), top_quantile=1 / 3, bottom_quantile=None)],
        horizons=(1,),
        quantiles=3,
    )
    before = set(glob.glob(f"{tempfile.gettempdir()}/cortexbacktest-feed-*"))

    run_execution_sweep(factor_sweep, _bars(prices), max_workers=2, parallel_backend="process")

    after = set(glob.glob(f"{tempfile.gettempdir()}/cortexbacktest-feed-*"))
    assert after == before  # no leaked temp file after successful process execution
