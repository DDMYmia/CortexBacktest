from datetime import UTC, datetime

import polars as pl
import pytest

from cortexbacktest import BacktestEngine, Bar, Order, Side, Strategy
from cortexbacktest.reporting import (
    align_returns,
    alpha,
    benchmark_report,
    beta,
    drawdown_series,
    drawdown_table,
    information_ratio,
    performance_report,
    period_returns,
    returns_from_equity,
    rolling_beta,
    rolling_sharpe,
)


def _equity(values: list[float], dates: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime.fromisoformat(date) for date in dates],
            "equity": values,
        }
    )


def _returns(values: list[float], dates: list[str], name: str = "return") -> pl.DataFrame:
    return pl.DataFrame({"timestamp": [datetime.fromisoformat(date) for date in dates], name: values})


DATES = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]


def test_explicit_starting_capital_keeps_entry_cost_and_reconciles_total_return():
    from cortexbacktest import BacktestResult

    equity = _equity([99.0, 110.0, 105.0], DATES[:3])
    returns = returns_from_equity(equity, initial_equity=100)["return"]
    assert returns[0] == pytest.approx(-0.01)
    assert (1 + returns).product() - 1 == pytest.approx(0.05)
    result = BacktestResult(equity, fills=[], orders=[], initial_equity=100)
    assert result.performance.returns()["return"].to_list() == pytest.approx(returns.to_list())


@pytest.mark.parametrize("initial", [0, -1, float("nan"), float("inf")])
def test_invalid_explicit_starting_capital_rejected(initial):
    with pytest.raises(ValueError, match="initial_equity"):
        returns_from_equity(_equity([99.0], DATES[:1]), initial_equity=initial)


def test_drawdown_series_and_table_identify_episode():
    equity = _equity([100.0, 120.0, 90.0, 130.0], DATES)

    series = drawdown_series(equity)
    table = drawdown_table(equity)

    # Peak 120 -> valley 90 is a 25% drawdown, recovered when equity exceeds 120.
    assert series["drawdown"].min() == pytest.approx(-0.25)
    assert table.height == 1
    row = table.row(0, named=True)
    assert row["depth"] == pytest.approx(-0.25)
    # Reporting normalizes equity timestamps to UTC.
    assert row["start"] == datetime(2024, 1, 3, tzinfo=UTC)
    assert row["valley"] == datetime(2024, 1, 3, tzinfo=UTC)
    assert row["end"] == datetime(2024, 1, 4, tzinfo=UTC)


def test_returns_from_equity_leaves_the_first_period_null():
    returns = returns_from_equity(_equity([100.0, 110.0, 121.0], ["2024-01-01", "2024-01-31", "2024-02-29"]))

    values = returns["return"].to_list()
    assert values[0] is None
    assert values[1] == pytest.approx(0.10)
    assert values[2] == pytest.approx(0.10)


def test_period_returns_compute_monthly_and_annual_returns():
    equity = _equity([100.0, 110.0, 121.0], ["2024-01-01", "2024-01-31", "2024-02-29"])

    monthly = period_returns(equity, "ME")
    annual = period_returns(equity, "YE")

    assert [round(value, 4) for value in monthly["return"].to_list()] == [0.10, 0.10]
    assert annual["return"].to_list()[0] == pytest.approx(0.21)


def test_performance_report_contains_summary_and_tables():
    report = performance_report(_equity([100.0, 110.0, 105.0, 120.0], DATES))

    assert report.summary["total_return"] == pytest.approx(0.20)
    assert report.summary["max_drawdown"] < 0
    assert isinstance(report.drawdowns, pl.DataFrame)
    assert isinstance(report.monthly_returns, pl.DataFrame)
    assert isinstance(report.annual_returns, pl.DataFrame)
    assert not report.drawdowns.is_empty()


def test_align_returns_keeps_only_shared_timestamps():
    strategy = _returns([0.01, 0.02, 0.03], ["2024-01-01", "2024-01-02", "2024-01-03"])
    benchmark = _returns([0.01, 0.02], ["2024-01-02", "2024-01-03"])

    aligned = align_returns(strategy, benchmark)

    assert aligned.columns == ["timestamp", "strategy", "benchmark"]
    assert aligned.height == 2
    assert aligned["strategy"].to_list() == [0.02, 0.03]


def test_alpha_beta_and_information_ratio_from_return_series():
    # strategy = 2 * benchmark + 0.001 makes beta and alpha analytically exact.
    benchmark_values = [0.01, -0.01, 0.02, -0.02]
    benchmark = _returns(benchmark_values, DATES, name="benchmark")
    strategy = _returns([value * 2 + 0.001 for value in benchmark_values], DATES, name="strategy")

    assert beta(strategy, benchmark) == pytest.approx(2.0)
    assert alpha(strategy, benchmark, periods_per_year=252) == pytest.approx(0.252)
    assert information_ratio(strategy, benchmark) > 0


def test_beta_is_zero_when_the_benchmark_never_moves():
    flat = _returns([0.0, 0.0, 0.0, 0.0], DATES, name="benchmark")
    strategy = _returns([0.01, 0.02, -0.01, 0.03], DATES, name="strategy")

    assert beta(strategy, flat) == 0.0


def test_rolling_metrics_validate_window():
    returns = _returns([0.01, 0.02, 0.03], DATES[:3])

    with pytest.raises(ValueError, match="window"):
        rolling_sharpe(returns, window=1)
    with pytest.raises(ValueError, match="window"):
        rolling_beta(returns, returns, window=1)


def test_rolling_metrics_return_timestamped_frames():
    benchmark_values = [0.01, -0.01, 0.02, -0.02]
    benchmark = _returns(benchmark_values, DATES, name="benchmark")
    strategy = _returns([value * 2 for value in benchmark_values], DATES, name="strategy")

    beta_frame = rolling_beta(strategy, benchmark, window=2)
    sharpe_frame = rolling_sharpe(strategy, window=2)

    assert beta_frame.columns == ["timestamp", "rolling_beta"]
    assert sharpe_frame.columns == ["timestamp", "rolling_sharpe"]
    # A window of 2 leaves the first row undefined, and beta stays 2 thereafter.
    assert beta_frame["rolling_beta"].to_list()[0] is None
    assert beta_frame["rolling_beta"].to_list()[1:] == pytest.approx([2.0, 2.0, 2.0])


def test_benchmark_report_computes_relative_metrics():
    strategy = _equity([100.0, 104.0, 107.0, 112.0], DATES)
    benchmark = _equity([100.0, 101.0, 103.0, 104.0], DATES)

    report = benchmark_report(strategy, benchmark, periods_per_year=252, rolling_window=2)

    assert report.summary["strategy_total_return"] == pytest.approx(0.12)
    assert report.summary["benchmark_total_return"] == pytest.approx(0.04)
    assert report.summary["active_total_return"] == pytest.approx(0.08)
    assert report.summary["beta"] != 0
    assert report.summary["information_ratio"] > 0
    assert isinstance(report.aligned_returns, pl.DataFrame)
    # The strategy beat the benchmark every period.
    assert all(value > 0 for value in report.active_returns["active_return"].to_list())
    assert report.rolling_beta.height == report.aligned_returns.height
    assert not report.rolling_sharpe.drop_nulls().is_empty()


def test_benchmark_report_on_disjoint_timestamps_is_empty_but_valid():
    strategy = _equity([100.0, 104.0], ["2024-01-01", "2024-01-02"])
    benchmark = _equity([100.0, 101.0], ["2024-03-01", "2024-03-02"])

    report = benchmark_report(strategy, benchmark)

    assert report.summary["beta"] == 0.0
    assert report.summary["information_ratio"] == 0.0
    assert report.aligned_returns.is_empty()
    assert report.active_returns.is_empty()


class BuySellStrategy(Strategy):
    def on_bar(self, bar: Bar):
        if bar.timestamp.day == 1:
            return [Order(bar.symbol, bar.timestamp, Side.BUY, 10)]
        if bar.timestamp.day == 2:
            return [Order(bar.symbol, bar.timestamp, Side.SELL, 10)]
        return []


def test_backtest_result_exposes_performance_report():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110),
    ]

    report = BacktestEngine(trade_on_close=True).run(bars, BuySellStrategy()).performance.report()

    assert report.summary["total_return"] > 0
    assert "sharpe" in report.summary
    assert isinstance(report.drawdowns, pl.DataFrame)


def test_backtest_result_exposes_polars_performance_tables():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110),
        Bar("AAPL", datetime(2024, 1, 3, tzinfo=UTC), 105, 105, 105, 105),
    ]

    result = BacktestEngine(trade_on_close=True).run(bars, BuySellStrategy())
    equity_curve = result.performance.equity_curve()

    assert isinstance(equity_curve, pl.DataFrame)
    assert {"timestamp", "equity", "cash", "gross_exposure", "leverage"}.issubset(equity_curve.columns)
    assert equity_curve.height == len(bars)
    assert isinstance(result.performance.returns(), pl.DataFrame)
    assert isinstance(result.performance.drawdowns(), pl.DataFrame)
    assert isinstance(result.performance.monthly_returns(), pl.DataFrame)
    assert isinstance(result.performance.annual_returns(), pl.DataFrame)


def test_backtest_result_exposes_benchmark_report():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110),
        Bar("AAPL", datetime(2024, 1, 3, tzinfo=UTC), 115, 115, 115, 115),
    ]
    result = BacktestEngine(trade_on_close=True).run(bars, BuySellStrategy())
    benchmark = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, day, tzinfo=UTC) for day in (1, 2, 3)],
            "equity": [100_000.0, 100_500.0, 101_000.0],
        }
    )

    report = result.performance.benchmark(benchmark, rolling_window=2)

    assert "active_total_return" in report.summary
    assert not report.aligned_returns.is_empty()
