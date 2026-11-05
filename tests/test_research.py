import warnings

import polars as pl
import pytest

from cortexbacktest import (
    FactorResearchReport,
    evaluate_factor,
    research_report,
    run_factor_research,
)
from cortexbacktest.data import Universe

DATES4 = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
DATES3 = ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_evaluate_empty_panel_retains_empty_metric_schemas():
    frame = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.String,
                                 "factor": pl.Float64, "forward_return": pl.Float64})
    result = evaluate_factor(frame)
    for attribute, columns in [("ic_by_date", ["date", "ic"]),
                               ("rank_ic_by_date", ["date", "rank_ic"]),
                               ("quantile_returns", ["date"]),
                               ("spread_by_date", ["date", "spread"])]:
        output = getattr(result, attribute)
        assert output.is_empty()
        assert output.columns == columns


def test_empty_top_bottom_weights_retains_sorted_symbol_schema():
    from cortexbacktest import top_bottom_weights

    output = top_bottom_weights(pl.DataFrame(schema={"date": pl.Date, "Z": pl.Float64, "A": pl.Float64}))
    assert output.is_empty()
    assert output.columns == ["date", "A", "Z"]
    assert output.schema["A"] == output.schema["Z"] == pl.Float64


def test_evaluate_factor_reports_positive_rank_ic_for_monotonic_factor():
    frame = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 5 + ["2024-01-02"] * 5,
            "symbol": list("ABCDE") * 2,
            "factor": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05, 0.00, 0.01, 0.02, 0.03, 0.04],
        }
    )

    result = evaluate_factor(frame, quantiles=5)

    assert result.mean_rank_ic == pytest.approx(1.0)
    assert result.mean_spread > 0
    assert result.ic_by_date.columns == ["date", "ic"]
    assert result.rank_ic_by_date.columns == ["date", "rank_ic"]
    assert "date" in result.quantile_returns.columns
    assert result.spread_by_date.columns == ["date", "spread"]


def test_evaluate_factor_orders_quantile_columns_numerically():
    # With quantiles >= 10 the bucket labels sort lexicographically as
    # "1", "10", "2", ... so the spread would compute Q9 - Q1 instead of
    # Q10 - Q1 unless the columns are ordered numerically.
    symbols = [f"S{index:02d}" for index in range(10)]
    factors = [float(index) for index in range(10)]
    forward = [0.01 * index for index in range(10)]
    frame = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 10 + ["2024-01-02"] * 10,
            "symbol": symbols * 2,
            "factor": factors * 2,
            "forward_return": forward * 2,
        }
    )

    result = evaluate_factor(frame, quantiles=10)

    assert result.quantile_returns.columns == ["date", *[str(bucket) for bucket in range(1, 11)]]
    row = result.spread_by_date.row(0, named=True)
    assert row["spread"] == pytest.approx(0.09)


def test_evaluate_factor_constant_inputs_return_empty_metrics_without_warning():
    frame = pl.DataFrame(
        {
            "date": ["2024-01-01"] * 4 + ["2024-01-02"] * 4,
            "symbol": list("ABCD") * 2,
            "factor": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0],
            "forward_return": [0.01, 0.02, 0.03, 0.04, 0.01, 0.01, 0.01, 0.01],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = evaluate_factor(frame, quantiles=2)

    # Day one has a constant factor; day two has constant forward returns. Neither
    # yields a defined correlation, so every metric frame is empty.
    assert result.ic_by_date.is_empty()
    assert result.rank_ic_by_date.is_empty()
    assert result.quantile_returns.is_empty()


def test_run_factor_research_returns_multi_horizon_report():
    prices = pl.DataFrame(
        {
            "date": DATES4,
            "A": [10.0, 11.0, 12.0, 13.0],
            "B": [10.0, 9.0, 8.0, 7.0],
            "C": [10.0, 10.0, 10.0, 10.0],
        }
    )
    factors = pl.DataFrame({"date": DATES4, "A": [1.0] * 4, "B": [-1.0] * 4, "C": [0.0] * 4})

    report = run_factor_research(
        prices,
        factors,
        horizons=(1, 2),
        quantiles=3,
        top_quantile=1 / 3,
        bottom_quantile=1 / 3,
    )

    assert set(report.horizon_results) == {1, 2}
    assert report.summary().filter(pl.col("horizon") == 1)["mean_rank_ic"].item() > 0
    assert report.coverage_by_date["coverage"][0] == 1.0
    assert report.turnover_by_date["turnover"][0] == 1.0
    assert report.equity_curve["equity"][-1] > 1.0


def test_run_factor_research_accepts_polars_matrices_and_exports_records():
    prices = pl.DataFrame({"date": DATES3, "A": [10.0, 11.0, 12.0], "B": [10.0, 9.0, 8.0]})
    factors = pl.DataFrame({"date": DATES3, "A": [1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0]})

    report = run_factor_research(prices, factors, horizons=(1,), quantiles=2, top_quantile=0.5, bottom_quantile=0.5)
    payload = research_report(report)

    assert isinstance(report, FactorResearchReport)
    assert report.summary()["horizon"].to_list() == [1]
    assert payload["summary"][0]["horizon"] == 1
    assert payload["records"]["target_weights"][0]["date"] == "2024-01-01T00:00:00"


def test_run_factor_research_respects_universe_mask():
    prices = pl.DataFrame(
        {"date": DATES3, "A": [10.0, 11.0, 12.0], "B": [10.0, 9.0, 8.0], "C": [10.0, 20.0, 30.0]}
    )
    factors = pl.DataFrame(
        {"date": DATES3, "A": [1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0], "C": [10.0, 10.0, 10.0]}
    )
    universe = Universe.from_symbols(DATES3, ["A", "B"])

    report = run_factor_research(
        prices,
        factors,
        universe=universe,
        horizons=(1,),
        quantiles=2,
        top_quantile=0.5,
        bottom_quantile=0.5,
    )

    # C is masked out: it never gets weight and coverage is 2 of 3 symbols.
    assert report.target_weights_frame()["C"].to_list() == [0.0, 0.0, 0.0]
    assert report.coverage()["coverage"].to_list() == [2 / 3, 2 / 3, 2 / 3]


def test_factor_research_result_exposes_ic_and_rank_ic_summaries():
    prices = pl.DataFrame(
        {"date": DATES4, "A": [10.0, 11.0, 12.0, 13.0], "B": [10.0, 9.0, 8.0, 7.0]}
    )
    factors = pl.DataFrame({"date": DATES4, "A": [1.0] * 4, "B": [-1.0] * 4})

    report = run_factor_research(prices, factors, horizons=(1,), quantiles=2, top_quantile=0.5, bottom_quantile=0.5)
    result = report.horizon_results[1]

    ic_summary = result.ic_summary()
    rank_ic_summary = result.rank_ic_summary()

    assert ic_summary.height == 1
    assert rank_ic_summary.height == 1
    assert rank_ic_summary.row(0, named=True)["mean"] == pytest.approx(result.mean_rank_ic)
    assert ic_summary.row(0, named=True)["mean"] == pytest.approx(result.mean_ic)
