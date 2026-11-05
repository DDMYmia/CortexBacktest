import polars as pl
import pytest

from cortexbacktest import (
    apply_volatility_target,
    cap_weights,
    equal_weight,
    inverse_volatility_weights,
    normalize_weights,
    select_top_n,
    simulate_target_weights,
    target_volatility_scale,
    weights_from_signals,
)

DATES = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]


@pytest.mark.parametrize("rows", [0, 2])
def test_selection_retains_entirely_missing_symbols(rows):
    scores = pl.DataFrame({"date": DATES[:rows], "A": [None] * rows,
                           "B": [1.0, None][:rows]},
                          schema={"date": pl.String, "A": pl.Float64, "B": pl.Float64})
    selected = select_top_n(scores, n=1)
    assert selected.columns == ["date", "A", "B"]
    assert selected["A"].to_list() == [False] * rows
    assert selected["B"].to_list() == [True, False][:rows]
    assert weights_from_signals(selected)["A"].to_list() == [0.0] * rows


def _row(frame: pl.DataFrame, index: int) -> dict:
    return frame.row(index, named=True)


def test_equal_weight_selects_positive_scores_for_long_only():
    scores = pl.DataFrame({"date": ["2024-01-01"], "A": [1.0], "B": [-1.0], "C": [2.0]})

    row = _row(equal_weight(scores, gross_weight=1.0), 0)

    # A and C are positive; each gets half the gross weight.
    assert row["A"] == pytest.approx(0.5)
    assert row["B"] == 0.0
    assert row["C"] == pytest.approx(0.5)


def test_select_top_n_and_weights_from_signals():
    scores = pl.DataFrame({"date": ["2024-01-01"], "A": [1.0], "B": [3.0], "C": [2.0]})

    selected = select_top_n(scores, n=2)
    weights = weights_from_signals(selected, gross_weight=1.0)

    selected_row = _row(selected, 0)
    assert (selected_row["A"], selected_row["B"], selected_row["C"]) == (False, True, True)
    weights_row = _row(weights, 0)
    assert weights_row["A"] == 0.0
    assert weights_row["B"] == pytest.approx(0.5)
    assert weights_row["C"] == pytest.approx(0.5)


def test_normalize_weights_scales_row_gross_exposure():
    weights = pl.DataFrame({"date": ["2024-01-01"], "A": [0.8], "B": [0.1], "C": [0.1]})

    row = _row(normalize_weights(weights, gross_weight=2.0), 0)

    assert abs(row["A"]) + abs(row["B"]) + abs(row["C"]) == pytest.approx(2.0)
    assert row["A"] == pytest.approx(1.6)


def test_cap_weights_bounds_each_position_and_keeps_gross():
    weights = pl.DataFrame({"date": ["2024-01-01"], "A": [0.8], "B": [0.1], "C": [0.1]})

    row = _row(cap_weights(weights, max_weight=0.5, gross_weight=1.0), 0)
    values = [row["A"], row["B"], row["C"]]

    assert sum(abs(value) for value in values) == pytest.approx(1.0)
    assert max(abs(value) for value in values) <= 0.5 + 1e-12
    # A is pinned at the cap; the freed exposure is redistributed to B and C.
    assert row["A"] == pytest.approx(0.5)
    assert row["B"] == pytest.approx(0.25)
    assert row["C"] == pytest.approx(0.25)


def test_inverse_volatility_weights_favor_lower_volatility_symbol():
    returns = pl.DataFrame(
        {
            "date": DATES,
            "LOW": [0.010, 0.011, 0.009, 0.010],
            "HIGH": [0.10, -0.10, 0.10, -0.10],
        }
    )

    weights = inverse_volatility_weights(returns, window=3, min_periods=2)

    last = _row(weights, len(DATES) - 1)
    assert last["LOW"] > last["HIGH"]
    assert abs(last["LOW"]) + abs(last["HIGH"]) == pytest.approx(1.0)


def test_target_volatility_scale_and_apply_volatility_target():
    returns = pl.DataFrame({"date": DATES, "return": [0.01, -0.01, 0.01, -0.01]})
    weights = pl.DataFrame({"date": DATES, "A": [1.0, 1.0, 1.0, 1.0]})

    scale = target_volatility_scale(returns, target_volatility=0.1, max_leverage=0.5)
    scaled = apply_volatility_target(weights, returns, target_volatility=0.1, window=2, max_leverage=0.5)

    assert 0 < scale <= 0.5
    assert _row(scaled, len(DATES) - 1)["A"] <= 0.5


def test_optimized_weights_can_feed_portfolio_simulation():
    returns = pl.DataFrame(
        {"date": DATES, "A": [0.0, 0.02, 0.01, 0.03], "B": [0.0, -0.01, -0.02, -0.01]}
    )
    scores = pl.DataFrame(
        {"date": DATES, "A": [1.0, 1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0, -1.0]}
    )

    weights = equal_weight(scores, gross_weight=1.0)
    equity = simulate_target_weights(returns, weights)

    assert equity["equity"][-1] > 1.0
