from datetime import date

import polars as pl
import pytest

from cortexbacktest.research import (
    FactorResearchReport,
    FactorResearchResult,
    cumulative_quantile_returns,
    factor_decay_summary,
    factor_information_coefficient,
    factor_rank_autocorrelation,
    ic_summary,
    quantile_turnover,
)


def test_ic_summary_pl_reports_stability_and_significance():
    summary = ic_summary(pl.DataFrame({"rank_ic": [0.1, 0.2, -0.1, 0.3]}))
    row = summary.row(0, named=True)

    assert row["count"] == 4
    assert row["mean"] == pytest.approx(0.125)
    assert row["information_ratio"] > 0
    assert row["t_stat"] == pytest.approx(row["information_ratio"] * 2)
    assert row["positive_ratio"] == pytest.approx(0.75)


def test_cumulative_quantile_returns_pl_compounds_each_quantile():
    result = cumulative_quantile_returns(
        pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "1": [0.01, -0.01],
                "5": [0.10, 0.10],
            }
        )
    )

    assert result["1"].to_list()[-1] == pytest.approx(-0.0001)
    assert result["5"].to_list()[-1] == pytest.approx(0.21)


def test_factor_rank_autocorrelation_pl_tracks_cross_sectional_persistence():
    factors = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "A": [1.0, 2.0, 3.0],
            "B": [2.0, 3.0, 2.0],
            "C": [3.0, 1.0, 1.0],
        }
    )

    result = factor_rank_autocorrelation(factors)

    assert result.height == 2
    assert result["rank_autocorrelation"].to_list()[0] == pytest.approx(-0.5)


def test_factor_decay_summary_pl_summarizes_each_horizon():
    empty = pl.DataFrame()
    report = FactorResearchReport(
        horizon_results={
            1: FactorResearchResult(
                ic_by_date=pl.DataFrame({"date": [date(2024, 1, 1)], "ic": [0.5]}),
                rank_ic_by_date=pl.DataFrame({"date": [date(2024, 1, 1)], "rank_ic": [0.4]}),
                quantile_returns=empty,
                spread_by_date=empty,
            ),
            5: FactorResearchResult(
                ic_by_date=pl.DataFrame({"date": [date(2024, 1, 1)], "ic": [0.2]}),
                rank_ic_by_date=pl.DataFrame({"date": [date(2024, 1, 1)], "rank_ic": [0.1]}),
                quantile_returns=empty,
                spread_by_date=empty,
            ),
        },
        coverage_by_date=empty,
        turnover_by_date=empty,
        target_weights=empty,
        equity_curve=empty,
    )

    result = factor_decay_summary(report)

    assert result["horizon"].to_list() == [1, 5]
    assert result["mean"].to_list() == pytest.approx([0.4, 0.1])
    assert report.factor_decay().equals(result)


def test_factor_information_coefficient_pl_supports_groups_and_neutralization():
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 1)] * 6,
            "symbol": ["A", "B", "C", "D", "E", "F"],
            "group": ["tech", "tech", "tech", "bank", "bank", "bank"],
            "factor": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "forward_return": [0.01, 0.02, 0.03, 0.30, 0.20, 0.10],
        }
    )

    by_group = factor_information_coefficient(frame, by_group=True)
    neutral = factor_information_coefficient(frame, group_adjust=True)

    assert by_group["group"].to_list() == ["bank", "tech"]
    assert by_group["rank_ic"].to_list() == pytest.approx([-1.0, 1.0])
    assert neutral["rank_ic"].to_list() == pytest.approx([-0.23904572186687872])


def test_quantile_turnover_pl_tracks_membership_changes():
    factors = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "A": [1.0, 4.0, 4.0],
            "B": [2.0, 3.0, 3.0],
            "C": [3.0, 2.0, 2.0],
            "D": [4.0, 1.0, 1.0],
        }
    )

    result = quantile_turnover(factors, quantiles=2)

    assert result.height == 4
    assert result.filter(pl.col("date") == date(2024, 1, 2))["turnover"].to_list() == pytest.approx(
        [1.0, 1.0]
    )
    assert result.filter(pl.col("date") == date(2024, 1, 3))["turnover"].to_list() == pytest.approx(
        [0.0, 0.0]
    )
