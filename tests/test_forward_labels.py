import polars as pl
import pytest

from cortexbacktest.research.labels import attach_forward_labels


def attach(decisions, prices, **kwargs):
    return attach_forward_labels(decisions, prices, pl.Series([0, 1, 2, 3]),
                                  price_basis="total_return_adjusted", price_data_id="verified-fixture", **kwargs)


def test_exiting_signal_universe_still_receives_holding_period_prices():
    decisions = pl.DataFrame({"date": [0], "instrument_id": ["exited"], "factor": [1.0]})
    prices = pl.DataFrame({"date": [1, 2, 1, 2], "instrument_id": ["exited", "exited", "other", "other"],
                           "open": [100.0, 70.0, 2.0, 3.0]})
    result = attach(decisions, prices)
    assert result.panel.height == decisions.height
    assert result.panel["forward_return"][0] == pytest.approx(-0.3)
    assert result.coverage["valid_rows"] == 1


def test_gap_does_not_shift_exit_to_next_available_quote():
    decisions = pl.DataFrame({"date": [0, 3], "instrument_id": ["A", "A"]})
    prices = pl.DataFrame({"date": [1, 3], "instrument_id": ["A", "A"], "open": [100.0, 120.0]})
    result = attach(decisions, prices)
    assert result.panel["forward_return"].to_list() == [None, None]
    assert result.coverage["unresolved_rows"] == 1
    assert result.coverage["right_censored_rows"] == 1


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_exit_quote_is_not_a_terminal_outcome(bad):
    decisions = pl.DataFrame({"date": [0], "instrument_id": ["A"]})
    prices = pl.DataFrame({"date": [1, 2], "instrument_id": ["A", "A"], "open": [100.0, bad]})
    assert attach(decisions, prices).coverage["unresolved_rows"] == 1


def test_unadjusted_prices_and_duplicate_identities_fail():
    decisions = pl.DataFrame({"date": [0], "instrument_id": ["A"]})
    prices = pl.DataFrame({"date": [1, 1], "instrument_id": ["A", "A"], "open": [100.0, 110.0]})
    with pytest.raises(ValueError, match="duplicate"):
        attach(decisions, prices)
    with pytest.raises(ValueError, match="total_return_adjusted"):
        attach_forward_labels(decisions, prices, pl.Series([0, 1, 2]),
                               price_basis="raw", price_data_id="raw-tape")


def test_overflow_is_counted_as_unresolved():
    decisions = pl.DataFrame({"date": [0], "instrument_id": ["A"]})
    prices = pl.DataFrame({"date": [1, 2], "instrument_id": ["A", "A"], "open": [1e-300, 1e300]})
    result = attach(decisions, prices)
    assert result.coverage["unresolved_rows"] == 1
    assert result.panel["forward_return"][0] is None
