from datetime import datetime, timedelta, timezone

import pytest

from cortexbacktest.options import (
    OptionBacktestConfig, OptionContract, OptionMarketFrame, OptionQuote, RegTOptionMargin,
    ResearchOptionMargin, run_option_backtest,
)

T = datetime(2026, 1, 5, 20, tzinfo=timezone.utc)
EXPIRY = T + timedelta(days=30)
SPOT = 400.0


def contract(symbol, right, strike, *, expiry=EXPIRY, underlying="SPY", multiplier=100):
    return OptionContract(symbol, underlying, right, strike, expiry, multiplier)


BOOK = {c.symbol: c for c in (
    contract("P390", "PUT", 390), contract("P380", "PUT", 380),
    contract("C410", "CALL", 410), contract("C420", "CALL", 420),
    contract("P390-FEB", "PUT", 390, expiry=EXPIRY + timedelta(days=28)),
)}
MARKS = {"P390": 5.0, "P380": 3.0, "C410": 4.0, "C420": 2.0, "P390-FEB": 8.0, "SPY": SPOT}


def requirement(positions, model=None, book=None, marks=None):
    return (model or RegTOptionMargin()).requirement(positions, book or BOOK, marks or MARKS)


def test_a_naked_short_is_unchanged_by_netting():
    """The netting model must not quietly become cheaper where nothing is netted."""
    # 5.00 premium + max(20% of 400 less the 10 out-of-the-money, 10% of 390) = 75/share.
    assert requirement({"P390": -1}) == pytest.approx(7_500)
    assert requirement({"P390": -1}) == pytest.approx(
        requirement({"P390": -1}, ResearchOptionMargin()))
    assert requirement({"C410": -1}) == pytest.approx(7_400)
    assert requirement({"P390": 1}) == pytest.approx(500)      # long premium, no loan value


def test_a_credit_spread_costs_its_width_not_a_naked_short():
    """The reason this model exists: the protective leg has to cap the loss.

    Under the unnetted model the long leg makes a 390/380 put spread cost *more*
    than the naked 390 put inside it, which is roughly ten times its real
    maximum loss and makes credit-spread research meaningless.
    """
    spread = requirement({"P390": -1, "P380": 1})
    assert spread == pytest.approx(1_000)                      # (390-380) x 100
    assert requirement({"P390": -1, "P380": 1}, ResearchOptionMargin()) == pytest.approx(7_800)
    assert spread < requirement({"P390": -1})
    assert requirement({"C410": -1, "C420": 1}) == pytest.approx(1_000)
    assert requirement({"P390": -3, "P380": 3}) == pytest.approx(3_000)


def test_a_debit_spread_costs_the_premium_it_paid():
    """The long leg is deeper than the short, so the short cannot lose; but a long
    option has no loan value and must not become free collateral."""
    assert requirement({"P380": -1, "P390": 1}) == pytest.approx(500)
    assert requirement({"C420": -1, "C410": 1}) == pytest.approx(400)


def test_an_iron_condor_charges_only_the_wider_wing():
    """Both wings cannot finish in the money, so they do not both need collateral."""
    condor = requirement({"P390": -1, "P380": 1, "C410": -1, "C420": 1})
    assert condor == pytest.approx(1_000)                      # not 2,000, and not 15,400
    assert requirement({"P390": -1, "P380": 1, "C410": -1, "C420": 1},
                       ResearchOptionMargin()) == pytest.approx(15_400)

    wide_puts = {c.symbol: c for c in (*BOOK.values(), contract("P360", "PUT", 360))}
    marks = {**MARKS, "P360": 1.0}
    uneven = requirement({"P390": -1, "P360": 1, "C410": -1, "C420": 1},
                         book=wide_puts, marks=marks)
    assert uneven == pytest.approx(3_000)                      # the 30-wide put wing wins


def test_stock_covers_a_short_call_and_a_partial_lot_leaves_a_fraction_naked():
    covered = requirement({"C410": -1, "SPY": 100})
    assert covered == pytest.approx(100 * SPOT * 0.5)          # the stock's own margin only

    half = requirement({"C410": -1, "SPY": 50})
    assert half == pytest.approx(50 * SPOT * 0.5 + 0.5 * 7_400)
    # Short stock covers a short put the same way; long stock does not.
    assert requirement({"P390": -1, "SPY": -100}) == pytest.approx(100 * SPOT * 0.5)
    assert requirement({"P390": -1, "SPY": 100}) == pytest.approx(100 * SPOT * 0.5 + 7_500)


def test_a_short_combination_is_the_greater_side_plus_the_other_premium():
    strangle = requirement({"P390": -1, "C410": -1})
    assert strangle == pytest.approx(7_500 + 400)              # put is the greater side
    assert strangle < requirement({"P390": -1}) + requirement({"C410": -1})


def test_a_ratio_spread_nets_one_pair_and_leaves_the_rest_naked():
    assert requirement({"P390": -2, "P380": 1}) == pytest.approx(1_000 + 7_500)


def test_legs_that_cannot_protect_each_other_do_not_net():
    """A later expiry does not cap this expiry's assignment, and nothing nets
    across underlyings or across a non-standard multiplier."""
    calendar = requirement({"P390": -1, "P390-FEB": 1})
    assert calendar == pytest.approx(7_500 + 800)              # naked short + long premium

    other = {**BOOK, "QQQ-P": contract("QQQ-P", "PUT", 380, underlying="QQQ")}
    marks = {**MARKS, "QQQ-P": 3.0, "QQQ": 380.0}
    assert requirement({"P390": -1, "QQQ-P": 1}, book=other, marks=marks) == pytest.approx(
        7_500 + 300)

    adjusted = {**BOOK, "P380-ADJ": contract("P380-ADJ", "PUT", 380, multiplier=107)}
    marks = {**MARKS, "P380-ADJ": 3.0}
    assert requirement({"P390": -1, "P380-ADJ": 1}, book=adjusted, marks=marks) == pytest.approx(
        7_500 + 3.0 * 107)


def test_netting_does_not_depend_on_position_ordering():
    legs = {"P390": -1, "P380": 1, "C410": -1, "C420": 1}
    orders = [dict(sorted(legs.items())), dict(sorted(legs.items(), reverse=True)),
              {"C420": 1, "P390": -1, "C410": -1, "P380": 1}]
    values = {round(requirement(order), 9) for order in orders}
    assert len(values) == 1


def test_fractions_are_validated():
    with pytest.raises(ValueError, match="fractions"):
        RegTOptionMargin(equity_fraction=1.5)
    with pytest.raises(ValueError, match="fractions"):
        RegTOptionMargin(stock_fraction=-0.1)
    index_style = RegTOptionMargin(equity_fraction=0.15, minimum_fraction=0.10)
    assert index_style.requirement({"P390": -1}, BOOK, MARKS) < requirement({"P390": -1})


def test_the_backtest_reports_the_netted_requirement():
    """End to end: the same credit spread, run under both models."""
    quotes = {symbol: OptionQuote(symbol, T, T, MARKS[symbol] - 0.05, MARKS[symbol] + 0.05,
                                  bid_size=50, ask_size=50, delta=-0.3)
              for symbol in ("P390", "P380")}
    quotes["SPY"] = OptionQuote("SPY", T, T, SPOT - 0.01, SPOT + 0.01, bid_size=1e6, ask_size=1e6)
    frames = [OptionMarketFrame(T + timedelta(days=day), {
        symbol: OptionQuote(symbol, T + timedelta(days=day), T + timedelta(days=day),
                            quote.bid, quote.ask, bid_size=quote.bid_size,
                            ask_size=quote.ask_size, delta=quote.delta)
        for symbol, quote in quotes.items()}) for day in range(3)]
    contracts = [BOOK["P390"], BOOK["P380"]]
    targets = {frames[0].timestamp: {"P390": -1, "P380": 1}}

    def strategy(frame, account):
        return targets.get(frame.timestamp, {})

    netted = run_option_backtest(contracts, frames, strategy,
                                 config=OptionBacktestConfig(margin=RegTOptionMargin()))
    unnetted = run_option_backtest(contracts, frames, strategy,
                                   config=OptionBacktestConfig(margin=ResearchOptionMargin()))
    assert netted.summary()["max_margin"] == pytest.approx(1_000, abs=20)
    assert unnetted.summary()["max_margin"] > 7_000
    assert netted.summary()["net_pnl"] == pytest.approx(unnetted.summary()["net_pnl"])
    assert netted.manifest["margin_model"] == "RegTOptionMargin"
