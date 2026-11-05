import json
from datetime import UTC, datetime

import polars as pl

from cortexbacktest import (
    Bar,
    FactorSweepCase,
    RebalanceConfig,
    execution_sweep_report,
    factor_sweep_report,
    momentum,
    research_report,
    run_execution_sweep,
    run_factor_research,
    run_factor_sweep,
    write_execution_sweep_report_html,
    write_execution_sweep_report_json,
    write_factor_sweep_report_html,
    write_factor_sweep_report_json,
    write_research_report_html,
    write_research_report_json,
)


_DATES = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]


def _prices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": _DATES,
            "A": [10.0, 11.0, 12.0, 13.0],
            "B": [10.0, 9.0, 8.0, 7.0],
            "C": [10.0, 10.5, 11.0, 11.5],
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


def test_research_report_json_and_html(tmp_path):
    prices = _prices()
    factors = pl.DataFrame({"date": _DATES, "A": [1.0] * 4, "B": [-1.0] * 4, "C": [0.2] * 4})
    report = run_factor_research(prices, factors, horizons=(1,), quantiles=3)

    payload = research_report(report)
    json_path = write_research_report_json(report, tmp_path / "research.json")
    html_path = write_research_report_html(report, tmp_path / "research.html")

    assert "summary" in payload
    assert "records" in payload
    assert json.loads(json_path.read_text())["summary"]
    assert "Research Report" in html_path.read_text()


def test_factor_sweep_and_execution_sweep_report_exports(tmp_path):
    prices = _prices()
    sweep = run_factor_sweep(
        prices,
        [
            FactorSweepCase("mom_1", lambda frame: momentum(frame, window=1), top_quantile=1 / 3, bottom_quantile=None),
        ],
        horizons=(1,),
        quantiles=3,
    )
    execution = run_execution_sweep(
        sweep,
        _bars(prices),
        rebalance_config=RebalanceConfig(allow_fractional=False, min_trade_value=1.0),
    )

    factor_payload = factor_sweep_report(sweep)
    execution_payload = execution_sweep_report(execution)
    factor_json = write_factor_sweep_report_json(sweep, tmp_path / "factor_sweep.json")
    factor_html = write_factor_sweep_report_html(sweep, tmp_path / "factor_sweep.html")
    execution_json = write_execution_sweep_report_json(execution, tmp_path / "execution_sweep.json")
    execution_html = write_execution_sweep_report_html(execution, tmp_path / "execution_sweep.html")

    assert factor_payload["best_case"] == "mom_1"
    assert execution_payload["best_case"] == "mom_1"
    assert json.loads(factor_json.read_text())["best_case"] == "mom_1"
    assert "Factor Sweep Report" in factor_html.read_text()
    assert json.loads(execution_json.read_text())["best_case"] == "mom_1"
    assert "Execution Sweep Report" in execution_html.read_text()
