from datetime import UTC, datetime

import pytest
import polars as pl

from cortexbacktest import BacktestEngine, Bar, Order, Side, Strategy
from cortexbacktest.reporting import exposure_summary, turnover_summary


class BuyHoldStrategy(Strategy):
    def __init__(self) -> None:
        self.submitted = False

    def on_bar(self, bar: Bar):
        if self.submitted:
            return []
        self.submitted = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 10)]


def test_backtest_result_exposes_positions_and_exposure_reports():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110),
    ]

    result = BacktestEngine(trade_on_close=True).run(bars, BuyHoldStrategy())
    positions = result.records.positions()
    exposure = result.exposure.frame()
    summary = result.exposure.summary()
    groups = result.exposure.by_group({"AAPL": "technology"})
    turnover = result.exposure.turnover()
    turnover_totals = result.exposure.turnover_summary()

    # 10 shares bought at 100 and marked at 110.
    assert positions.row(0, named=True)["symbol"] == "AAPL"
    assert positions.row(0, named=True)["quantity"] == 10
    assert positions.row(0, named=True)["market_value"] == 1_100
    assert isinstance(exposure, pl.DataFrame)
    assert {"gross_exposure", "net_exposure", "leverage"}.issubset(exposure.columns)
    assert summary["max_gross_exposure"] == pytest.approx(1_100)
    assert summary["max_leverage"] > 0
    assert groups.row(0, named=True)["group"] == "technology"
    assert groups.row(0, named=True)["market_value"] == pytest.approx(1_100)
    assert turnover.row(0, named=True)["traded_notional"] == pytest.approx(1_000)
    assert turnover_totals["total_traded_notional"] == pytest.approx(1_000)

    # The namespace and the free functions must agree.
    assert exposure_summary(exposure) == summary
    assert turnover_summary(result.records.fills(), result.performance.equity_curve()) == turnover_totals


def test_event_report_includes_positions_exposure_and_turnover_records():
    bars = [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 100, 100, 100, 100),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 110, 110, 110, 110),
    ]

    result = BacktestEngine(trade_on_close=True).run(bars, BuyHoldStrategy())
    report = result.report()

    assert "positions" in report["records"]
    assert "exposure" in report["records"]
    assert "turnover" in report["records"]
    assert report["records"]["positions"][0]["symbol"] == "AAPL"
