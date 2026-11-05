import polars as pl
import pytest

from cortexbacktest.research import simulate_target_weights, top_bottom_weights


def test_simulate_target_weights_lags_weights_one_period():
    returns = pl.DataFrame({"date": ["2024-01-01", "2024-01-02"], "A": [0.10, 0.20]})
    weights = pl.DataFrame({"date": ["2024-01-01", "2024-01-02"], "A": [1.0, 1.0]})

    equity = simulate_target_weights(returns, weights)

    # Weights apply from the next period, so day one is flat and day two earns 20%.
    assert equity["equity"][0] == 1.0
    assert round(equity["equity"][1], 6) == 1.2


def test_top_bottom_weights_matches_long_short_selection():
    factors = pl.DataFrame(
        {"date": ["2024-01-01", "2024-01-02"], "A": [3.0, 3.0], "B": [1.0, 1.0], "C": [2.0, 2.0]}
    )

    weights = top_bottom_weights(factors, top_quantile=1 / 3, bottom_quantile=1 / 3)

    assert weights.columns == ["date", "A", "B", "C"]
    assert weights.select(["A", "B", "C"]).to_dict(as_series=False) == {
        "A": [1.0, 1.0],
        "B": [-1.0, -1.0],
        "C": [0.0, 0.0],
    }


def test_top_bottom_weights_never_overwrites_a_long_with_an_overlapping_short():
    # A single symbol per side: the top selection wins, so no symbol goes short.
    factors = pl.DataFrame({"date": ["2024-01-01"], "A": [2.0], "B": [1.0]})

    weights = top_bottom_weights(factors, top_quantile=1.0, bottom_quantile=1.0)

    assert weights.select(["A", "B"]).row(0, named=True) == {"A": 0.5, "B": 0.5}


def test_portfolio_helpers_validate_inputs():
    frame = pl.DataFrame({"date": ["2024-01-01"], "A": [1.0]})

    with pytest.raises(ValueError, match="initial_equity"):
        simulate_target_weights(frame, frame, initial_equity=0)
    with pytest.raises(ValueError, match="top_quantile"):
        top_bottom_weights(frame, top_quantile=0)


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf")])
def test_missing_held_return_does_not_become_flat_or_resume_known_nav(missing):
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    returns = pl.DataFrame({"date": dates, "A": [0., .1, missing, .2]})
    weights = pl.DataFrame({"date": dates, "A": [1., 1., 0., 0.]})
    equity = simulate_target_weights(returns, weights, transaction_cost_bps=10)
    assert equity["equity"].to_list() == [1., pytest.approx(1.099), None, None]


def test_missing_unheld_return_does_not_poison_known_portfolio():
    returns = pl.DataFrame({"date": ["2024-01-01", "2024-01-02"], "A": [None, .1], "B": [float("nan"), None]})
    weights = pl.DataFrame({"date": returns["date"], "A": [1., 1.], "B": [0., 0.]})
    assert simulate_target_weights(returns, weights)["equity"].to_list() == [1., 1.1]


def test_report_does_not_reuse_stale_nav_after_missing_valuation():
    import math
    from cortexbacktest.research.workflow import FactorResearchReport
    from cortexbacktest.reporting import research_report

    report = FactorResearchReport({}, pl.DataFrame(), pl.DataFrame(), pl.DataFrame(),
                                  pl.DataFrame({"date": ["2024-01-01", "2024-01-02"], "equity": [1.2, None]}))
    assert math.isnan(report.final_equity)
    assert research_report(report, include_records=False)["final_equity"] is None
