"""Delisted symbols must cost the research path what they cost the broker.

A delisted symbol simply stops having prices, and `forward_returns_matrix` reads
the gap as `null` — the position leaves the sample instead of losing its value.
IC, quantile spreads, and long-short portfolios are then computed over survivors
only, which flatters exactly the factors that load on distress.

The correction is applied to the price matrix rather than to any consumer: the
cash-out value is carried forward from the ex-date, so a worthless delisting is
a price of 0 and the terminal return is -100%. Everything downstream is then
right without knowing delisting exists, which is what these tests check.
"""

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from cortexbacktest import CorporateAction, CorporateActionType, load_backtest_dataset
from cortexbacktest.research.delisting import apply_delisting_prices, delisted_symbols
from cortexbacktest.research.frames import forward_returns_matrix

DATES = [datetime(2024, 1, 1) + timedelta(days=day) for day in range(5)]


def _prices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": DATES,
            "GOOD": [100.0, 101.0, 102.0, 103.0, 104.0],
            # Stops trading after the third day.
            "DEAD": [100.0, 100.0, 100.0, None, None],
        }
    )


def _delisting(amount: float, day: int = 4) -> CorporateAction:
    return CorporateAction("DEAD", date(2024, 1, day), CorporateActionType.DELISTING, amount=amount)


# --- the bias itself -------------------------------------------------------

def test_without_the_correction_the_terminal_loss_is_missing():
    # Pinning the broken behaviour so the fix below is measured against
    # something, not asserted into existence.
    forward = forward_returns_matrix(_prices(), 1)

    last_traded = forward.filter(pl.col("date") == DATES[2])["DEAD"].item()
    assert last_traded is None, "the day it died contributes no return at all"


# A worthless delisting is a total loss; a cash-out of 30 after a last trade of 100
# loses 70%, not everything; a 2-day return from the day before the ex-date has to
# include the loss, not stop at the last quote.
@pytest.mark.parametrize(("cash_out", "horizon", "day", "loss"), [
    (0.0, 1, 2, -1.0), (30.0, 1, 2, -0.7), (0.0, 2, 1, -1.0),
], ids=["worthless", "cash-out", "longer-horizon"])
def test_a_delisting_return_is_the_cash_out_against_the_last_trade(cash_out, horizon, day, loss):
    forward = forward_returns_matrix(apply_delisting_prices(_prices(), [_delisting(cash_out)]), horizon)
    assert forward.filter(pl.col("date") == DATES[day])["DEAD"].item() == pytest.approx(loss)


def test_proceeds_sit_flat_afterwards():
    # After the cash-out the holder has cash, so the series must not keep
    # producing returns — nor go back to null and re-open the same hole.
    prices = apply_delisting_prices(_prices(), [_delisting(30.0)])
    forward = forward_returns_matrix(prices, 1)

    assert forward.filter(pl.col("date") == DATES[3])["DEAD"].item() == pytest.approx(0.0)


# --- leaving everything else alone ----------------------------------------

def test_surviving_symbols_are_untouched():
    prices = apply_delisting_prices(_prices(), [_delisting(0.0)])

    assert prices["GOOD"].to_list() == _prices()["GOOD"].to_list()


def test_prices_before_the_ex_date_are_untouched():
    prices = apply_delisting_prices(_prices(), [_delisting(0.0)])

    assert prices["DEAD"].to_list()[:3] == [100.0, 100.0, 100.0]


@pytest.mark.parametrize(
    "action",
    [
        CorporateAction("DEAD", date(2024, 1, 4), CorporateActionType.SPLIT, ratio=2.0),
        CorporateAction("DEAD", date(2024, 1, 4), CorporateActionType.CASH_DIVIDEND, amount=1.0),
        CorporateAction("DEAD", date(2024, 1, 4), CorporateActionType.SYMBOL_CHANGE, new_symbol="NEW"),
    ],
)
def test_non_delisting_actions_do_not_touch_the_price_matrix(action):
    # Splits and dividends belong to the price basis, which is the data
    # provider's job; a rename has no price effect at all.
    assert apply_delisting_prices(_prices(), [action]).equals(_prices())


def test_a_delisting_for_an_absent_symbol_is_ignored():
    absent = CorporateAction("OTHER", date(2024, 1, 4), CorporateActionType.DELISTING, amount=0.0)

    assert apply_delisting_prices(_prices(), [absent]).equals(_prices())


def test_no_actions_returns_the_same_frame():
    assert apply_delisting_prices(_prices(), []).equals(_prices())


def test_delisted_symbols_reports_only_delistings():
    actions = [
        _delisting(0.0),
        CorporateAction("GOOD", date(2024, 1, 4), CorporateActionType.SPLIT, ratio=2.0),
    ]

    assert delisted_symbols(actions) == frozenset({"DEAD"})


# --- through the dataset ---------------------------------------------------

def _long_form():
    bars, factors = [], []
    for symbol, series in (("GOOD", [100.0, 101.0, 102.0, 103.0, 104.0]),
                           ("DEAD", [100.0, 100.0, 100.0])):
        for day, price in enumerate(series):
            bars.append({"date": DATES[day], "symbol": symbol, "adj_close": price})
            # The factor likes DEAD, so ignoring its delisting flatters it.
            factors.append({"date": DATES[day], "symbol": symbol,
                            "factor": 1.0 if symbol == "DEAD" else 0.0})
    return pl.DataFrame(bars), pl.DataFrame(factors)


def test_the_dataset_applies_delistings_when_given_them():
    bars, factors = _long_form()

    dataset = load_backtest_dataset(bars=bars, factors=factors, corporate_actions=[_delisting(0.0)])

    dead = dataset.prices.filter(pl.col("date") == DATES[3])["DEAD"].item()
    assert dead == pytest.approx(0.0), "the cash-out value is carried into the matrix"


def test_the_dataset_is_unchanged_when_no_actions_are_supplied():
    # The correction is opt-in: a caller without corporate action data gets the
    # same dataset it always did rather than a silently different one.
    bars, factors = _long_form()

    with_none = load_backtest_dataset(bars=bars, factors=factors)
    dead = with_none.prices.filter(pl.col("date") == DATES[3])["DEAD"].item()

    assert dead is None


def test_the_terminal_loss_reaches_the_returns_matrix():
    bars, factors = _long_form()

    dataset = load_backtest_dataset(bars=bars, factors=factors, corporate_actions=[_delisting(0.0)])

    # `returns` is derived from the corrected prices, so the -100% is there for
    # every consumer that reads it.
    assert dataset.returns.filter(pl.col("date") == DATES[3])["DEAD"].item() == pytest.approx(-1.0)
