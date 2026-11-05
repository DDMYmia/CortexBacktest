import polars as pl
import pytest

from cortexbacktest import (
    FactorPreprocessConfig,
    demean_cross_section,
    neutralize_by_group,
    prepare_factor,
    rank_cross_section,
    run_factor_research,
    winsorize_cross_section,
    zscore_cross_section,
)

GROUPS = {"A": "tech", "B": "tech", "C": "financials", "D": "financials"}


@pytest.mark.parametrize("transform", [
    winsorize_cross_section, demean_cross_section, zscore_cross_section, rank_cross_section,
    lambda frame: neutralize_by_group(frame, GROUPS), prepare_factor,
    lambda frame: prepare_factor(frame, FactorPreprocessConfig(
        winsorize_quantiles=(0.1, 0.9), neutralize_groups=GROUPS, rank=True, zscore=True, demean=True)),
])
@pytest.mark.parametrize("date_only", [True, False])
def test_preprocessing_preserves_degenerate_matrix_shape(transform, date_only):
    from datetime import datetime

    frame = (pl.DataFrame({"date": ["2024-01-02", "2024-01-01"]}) if date_only else
             pl.DataFrame(schema={"date": pl.Date, "B": pl.Float64, "A": pl.Float64}))
    result = transform(frame)
    assert result.columns == frame.columns
    assert result.height == frame.height
    assert result.schema["date"] == pl.Datetime("us")
    if date_only:
        assert result["date"].to_list() == [datetime(2024, 1, 1), datetime(2024, 1, 2)]
    else:
        assert result.schema["A"] == result.schema["B"] == pl.Float64


def _factors() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "A": [1.0, 2.0],
            "B": [3.0, 2.0],
            "C": [2.0, 2.0],
            "D": [6.0, None],
        }
    )


def _row(frame: pl.DataFrame, index: int) -> dict:
    return frame.row(index, named=True)


def test_winsorize_cross_section_clips_by_date():
    factors = pl.DataFrame({"date": ["2024-01-01"], "A": [1.0], "B": [2.0], "C": [100.0]})

    row = _row(winsorize_cross_section(factors, lower_quantile=0.0, upper_quantile=0.5), 0)

    assert row["A"] == 1.0
    assert row["B"] == 2.0
    assert row["C"] == 2.0


def test_winsorize_uses_linear_quantile_interpolation():
    # Polars defaults to "nearest", which would clip to 1.0 here. Linear is the
    # convention the rest of the research stack assumes.
    factors = pl.DataFrame({"date": ["2024-01-01"], **{f"S{i}": [float(i + 1)] for i in range(10)}})

    row = _row(winsorize_cross_section(factors, lower_quantile=0.01, upper_quantile=0.99), 0)

    assert row["S0"] == pytest.approx(1.09)
    assert row["S9"] == pytest.approx(9.91)


def test_zscore_cross_section_handles_constant_rows_and_nulls():
    result = zscore_cross_section(_factors())

    first = _row(result, 0)
    values = [first[symbol] for symbol in ("A", "B", "C", "D")]
    assert sum(values) / len(values) == pytest.approx(0.0)

    # 2024-01-02 has zero dispersion: every present value sits at the mean.
    second = _row(result, 1)
    assert [second[symbol] for symbol in ("A", "B", "C")] == [0.0, 0.0, 0.0]
    assert second["D"] is None


def test_zscore_cross_section_rejects_negative_ddof():
    with pytest.raises(ValueError, match="ddof"):
        zscore_cross_section(_factors(), ddof=-1)


def test_rank_cross_section_averages_ties_and_keeps_nulls():
    factors = pl.DataFrame({"date": ["2024-01-01"], "A": [1.0], "B": [2.0], "C": [2.0], "D": [None]})

    row = _row(rank_cross_section(factors), 0)

    # Three ranked values; B and C tie for ranks 2 and 3 -> average 2.5.
    assert row["A"] == pytest.approx(1 / 3)
    assert row["B"] == pytest.approx(2.5 / 3)
    assert row["C"] == pytest.approx(2.5 / 3)
    assert row["D"] is None


def test_neutralize_by_group_subtracts_datewise_group_means():
    row = _row(neutralize_by_group(_factors(), GROUPS), 0)

    assert row["A"] == pytest.approx(-1.0)
    assert row["B"] == pytest.approx(1.0)
    assert row["C"] == pytest.approx(-2.0)
    assert row["D"] == pytest.approx(2.0)


def test_neutralize_by_group_keeps_ungrouped_symbols_by_default():
    row = _row(neutralize_by_group(_factors(), {"A": "tech", "B": "tech"}), 0)

    assert row["A"] == pytest.approx(-1.0)
    assert row["B"] == pytest.approx(1.0)
    # C and D have no group; incomplete metadata must not silently drop them.
    assert row["C"] == 2.0
    assert row["D"] == 6.0


def test_neutralize_by_group_can_blank_ungrouped_symbols():
    row = _row(neutralize_by_group(_factors(), {"A": "tech", "B": "tech"}, missing="nan"), 0)

    assert row["C"] is None
    assert row["D"] is None


def test_neutralize_by_group_rejects_unknown_missing_policy():
    with pytest.raises(ValueError, match="missing"):
        neutralize_by_group(_factors(), GROUPS, missing="drop")


def test_prepare_factor_applies_configured_pipeline():
    config = FactorPreprocessConfig(
        winsorize_quantiles=(0.0, 1.0),
        neutralize_groups=GROUPS,
        zscore=True,
    )

    row = _row(prepare_factor(_factors(), config), 0)
    values = [row[symbol] for symbol in ("A", "B", "C", "D")]

    assert sum(values) / len(values) == pytest.approx(0.0)
    variance = sum(value**2 for value in values) / len(values)
    assert variance**0.5 == pytest.approx(1.0)


def test_prepare_factor_rejects_invalid_quantiles():
    with pytest.raises(ValueError, match="lower_quantile"):
        prepare_factor(_factors(), FactorPreprocessConfig(winsorize_quantiles=(0.9, 0.1)))


def test_prepare_factor_requires_a_date_column():
    with pytest.raises(ValueError, match="date column"):
        prepare_factor(pl.DataFrame({"A": [1.0], "B": [2.0]}))


def test_run_factor_research_pl_accepts_preprocessing_config():
    prices = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "A": [10.0, 11.0, 12.0, 13.0],
            "B": [10.0, 9.0, 8.0, 7.0],
            "C": [10.0, 10.0, 10.0, 10.0],
            "D": [10.0, 10.0, 11.0, 12.0],
        }
    )
    factors = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "A": [1.0, 1.0, 1.0, 1.0],
            "B": [-1.0, -1.0, -1.0, -1.0],
            "C": [0.0, 0.0, 0.0, 0.0],
            "D": [0.5, 0.5, 0.5, 0.5],
        }
    )

    report = run_factor_research(
        prices,
        factors,
        preprocessing=FactorPreprocessConfig(zscore=True),
        horizons=(1,),
        quantiles=2,
    )

    assert report.summary()["horizon"].to_list() == [1]
    assert "equity" in report.equity_curve_frame().columns
