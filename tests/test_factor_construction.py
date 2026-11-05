import math

import polars as pl
import pytest

from cortexbacktest import (
    FactorPreprocessConfig,
    FactorSpec,
    align_factor_matrices,
    combine_factor_scores,
    factor_decay_exponential,
    factor_decay_linear,
    run_factor_research,
)

DATES = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]


def _row(frame: pl.DataFrame, index: int) -> dict:
    return frame.row(index, named=True)


def test_combine_factor_scores_weighted_and_normalized():
    momentum = pl.DataFrame({"date": DATES[:2], "A": [1.0, 2.0], "B": [3.0, 4.0]})
    quality = pl.DataFrame({"date": DATES[:2], "A": [2.0, 2.0], "B": [4.0, 4.0]})

    result = combine_factor_scores(
        {"momentum": momentum, "quality": quality},
        weights={"momentum": 2.0, "quality": 1.0},
    )

    # Weighted mean: (value*2 + other*1) / 3.
    assert _row(result, 0)["A"] == pytest.approx((1.0 * 2.0 + 2.0) / 3.0)
    assert _row(result, 1)["B"] == pytest.approx((4.0 * 2.0 + 4.0) / 3.0)


def test_combine_factor_scores_min_count_keeps_stricter_cells_null():
    factor_a = pl.DataFrame({"date": DATES[:2], "A": [1.0, None], "B": [3.0, 4.0]})
    factor_b = pl.DataFrame({"date": DATES[:2], "A": [2.0, 2.0], "B": [None, 4.0]})

    result = combine_factor_scores({"a": factor_a, "b": factor_b}, min_count=2)

    assert _row(result, 0)["A"] == pytest.approx(1.5)
    # min_count=2 blanks cells with fewer than two contributing factors.
    assert _row(result, 1)["A"] is None
    assert _row(result, 0)["B"] is None
    assert _row(result, 1)["B"] == pytest.approx(4.0)


def test_combine_factor_scores_accepts_per_factor_preprocessing():
    raw = pl.DataFrame({"date": DATES[:2], "A": [1.0, 1.0], "B": [3.0, 3.0], "C": [5.0, 5.0]})
    inverse = raw.with_columns([pl.col(column) * -1 for column in ("A", "B", "C")])

    result = combine_factor_scores(
        [
            FactorSpec("raw", raw, preprocessing=FactorPreprocessConfig(zscore=True)),
            FactorSpec("inverse", inverse, weight=-1.0, preprocessing=FactorPreprocessConfig(zscore=True)),
        ]
    )

    row = _row(result, 0)
    # Both legs rank A < B < C after z-scoring, and the -1 weight re-aligns them.
    assert row["A"] < row["B"] < row["C"]


def test_factor_decay_linear_weights_recent_values_more_heavily():
    factors = pl.DataFrame({"date": DATES, "A": [1.0, 2.0, 3.0, 4.0]})

    result = factor_decay_linear(factors, window=3)

    assert _row(result, 1)["A"] is None
    # Linear weights 1,2,3 over the trailing window, normalized by weight mass 6.
    assert _row(result, 2)["A"] == pytest.approx((1.0 * 1 + 2.0 * 2 + 3.0 * 3) / 6)
    assert _row(result, 3)["A"] == pytest.approx((2.0 * 1 + 3.0 * 2 + 4.0 * 3) / 6)


def test_factor_decay_exponential_matches_ewm_recurrence():
    factors = pl.DataFrame({"date": DATES[:3], "A": [1.0, 2.0, 3.0]})

    result = factor_decay_exponential(factors, span=2)

    # adjust=False EWM with span=2 has alpha = 2/3.
    alpha = 2.0 / 3.0
    expected = [1.0]
    for value in (2.0, 3.0):
        expected.append(alpha * value + (1 - alpha) * expected[-1])
    assert result["A"].to_list() == pytest.approx(expected)


def test_align_factor_matrices_outer_join_adds_missing_symbols():
    factor_a = pl.DataFrame({"date": DATES[:2], "A": [1.0, 2.0]})
    factor_b = pl.DataFrame({"date": DATES[:2], "B": [3.0, 4.0]})

    aligned = align_factor_matrices({"a": factor_a, "b": factor_b}, join="outer")

    assert set(aligned) == {"a", "b"}
    assert aligned["a"].columns == ["date", "A", "B"]
    assert aligned["b"].columns == ["date", "A", "B"]
    assert aligned["a"]["B"].null_count() == 2
    assert aligned["b"]["A"].null_count() == 2


def test_composite_factor_runs_through_research_workflow():
    prices = pl.DataFrame(
        {
            "date": DATES,
            "A": [10.0, 11.0, 12.0, 13.0],
            "B": [10.0, 9.0, 8.0, 7.0],
            "C": [10.0, 10.5, 11.0, 11.5],
        }
    )
    momentum = pl.DataFrame(
        {"date": DATES, "A": [1.0] * 4, "B": [-1.0] * 4, "C": [0.2] * 4}
    )
    quality = pl.DataFrame(
        {"date": DATES, "A": [0.5] * 4, "B": [-0.5] * 4, "C": [0.1] * 4}
    )
    composite = combine_factor_scores({"momentum": momentum, "quality": quality})

    report = run_factor_research(prices, composite, horizons=(1,), quantiles=3)

    mean_rank_ic = report.summary().filter(pl.col("horizon") == 1)["mean_rank_ic"].item()
    assert not math.isnan(mean_rank_ic)
    assert mean_rank_ic > 0


def test_linear_decay_keeps_column_masks_independent():
    frame = pl.DataFrame({
        "date": list(range(5)),
        "Z": [1.0, None, 3.0, float("inf"), 5.0],
        "A": [2.0, 4.0, 6.0, 8.0, 10.0],
        "empty": pl.Series([None] * 5, dtype=pl.Float64),
    })
    result = factor_decay_linear(frame, window=3, min_periods=2)
    assert result.columns == ["date", "A", "Z", "empty"]
    assert result["Z"].to_list()[2:] == [2.5, None, 4.5]
    assert result["A"].to_list()[2:] == pytest.approx([28 / 6, 40 / 6, 52 / 6])
    assert result["empty"].null_count() == 5


@pytest.mark.parametrize("decay,options", [
    (factor_decay_linear, {"window": 2}),
    (factor_decay_exponential, {"span": 2}),
])
def test_decay_accepts_date_only_matrix(decay, options):
    result = decay(pl.DataFrame({"date": [2, 0, 1]}), **options)
    assert result.columns == ["date"]
    assert result.height == 3


def test_spec_weight_override_preserves_preprocessing():
    values = pl.DataFrame({"date": DATES[:1], "A": [1.0], "B": [3.0], "C": [5.0]})
    config = FactorPreprocessConfig(zscore=True)
    actual = combine_factor_scores([FactorSpec("value", values, preprocessing=config)],
                                   weights={"value": -2.0})
    expected = combine_factor_scores([FactorSpec("value", values, weight=-2.0, preprocessing=config)])
    assert actual.equals(expected)
    assert actual["A"][0] > actual["B"][0] > actual["C"][0]
