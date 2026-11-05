from datetime import date, datetime, timedelta, timezone

import pytest

from cortexbacktest.options.margin import RegTOptionMargin
from cortexbacktest.options.portfolio_risk import (
    Leg, analyse, coverage, fit_margin, liquidation_shock, render, scenarios, settle,
)

FRIDAY, NEXT = date(2026, 10, 23), date(2026, 10, 30)
NOW = datetime(2026, 10, 23, 15, 0, tzinfo=timezone.utc)  # 11:00 New York, before the close
SPOTS = {"AAA": 100.0, "QQQ": 700.0}


def book():
    return [Leg("AAA", 300, average_cost=110.0),
            Leg("AAA", -2, FRIDAY, 105.0, "PUT", 0.4, 0.9, 0.01),       # in the money: assigned
            Leg("AAA", -1, FRIDAY, 99.5, "PUT", 0.4, 0.45, 0.05),       # 0.5% out: possible
            Leg("AAA", -1, FRIDAY, 120.0, "CALL", 0.4, 0.01, 0.001),    # far out: expires
            Leg("AAA", -3, NEXT, 110.0, "CALL", 0.4, -0.2, 0.02)]


def test_settlement_turns_in_the_money_shorts_into_shares_and_flags_the_near_ones():
    after, flagged, cash = settle(book(), SPOTS, FRIDAY, 0.015)
    assert cash == -2 * 100 * 105.0
    assert [(f["strike"], f["state"]) for f in flagged] == [(105.0, "certain"), (99.5, "possible")]
    stock = [leg for leg in after if not leg.option]
    assert stock == [Leg("AAA", 500)] and all(leg.expiry == NEXT for leg in after if leg.option)
    rows = coverage(after, flagged, NEXT)
    assert rows == [{"symbol": "AAA", "shares": 500, "shares_if_possible": 600, "short_calls": 300,
                     "uncovered": 200, "uncovered_if_possible": 300}]


def test_leveraged_funds_carry_their_factor_in_stock_and_naked_requirements():
    plain, triple = RegTOptionMargin(stock_fraction=0.25), RegTOptionMargin(stock_fraction=0.25, leverage={"AAA": 3})
    assert plain.requirement({"AAA": 100}, {}, {"AAA": 100.0}) == pytest.approx(2_500)
    assert triple.requirement({"AAA": 100}, {}, {"AAA": 100.0}) == pytest.approx(7_500)
    assert RegTOptionMargin(stock_fraction=0.5, leverage={"AAA": 3}).requirement({"AAA": 1}, {}, {"AAA": 100.0}) == 100
    with pytest.raises(ValueError):
        RegTOptionMargin(leverage={"AAA": 0.5})


def test_the_margin_fit_reproduces_the_broker_and_scenarios_start_from_it():
    legs = book()
    unlevered, _ = fit_margin(legs, SPOTS, NOW, 0.04, 1.0, {})
    reported = 1.2 * unlevered.requirement(*_book_inputs(legs))
    margin, fit = fit_margin(legs, SPOTS, NOW, 0.04, reported, {"AAA": 3})
    assert abs(fit["residual"]) < 1e-6 and 1 <= fit["factors"]["AAA"] <= 3
    rows = scenarios(legs, SPOTS, {"AAA": 2.0}, [0.0, -0.1], NOW, NEXT, 0.04, 1.3, margin, 100_000)
    assert rows[0]["pnl_instant"] == pytest.approx(0) and rows[0]["maintenance"] == pytest.approx(reported)
    assert rows[1]["pnl_instant"] < 0
    shock = liquidation_shock(legs, SPOTS, {"AAA": 2.0}, NOW, NEXT, 0.04, 1.3, margin, reported + 5_000)
    assert shock is not None and -0.5 < shock < 0


def _book_inputs(legs):
    from cortexbacktest.options.portfolio_risk import _contracts
    return _contracts(legs, SPOTS, NOW, 0.04, 1.0)


def test_analyse_reports_every_section_and_renders_them():
    days = [date(2025, 10, 1) + timedelta(days=i) for i in range(300)]
    history = {"QQQ": [[str(d), 700 * (1 + 0.01 * ((i % 7) - 3) / 3)] for i, d in enumerate(days)],
               "AAA": [[str(d), 100 * (1 + 0.02 * ((i % 7) - 3) / 3)] for i, d in enumerate(days)]}
    legs = book()
    collected = {"as_of": NOW.isoformat(), "spots": SPOTS, "history": history,
                 "account": {"NetLiquidation": 60_000, "MaintMarginReq": 20_000, "TotalCashValue": 30_000,
                             "ExcessLiquidity": 40_000},
                 "positions": [{"symbol": x.symbol, "quantity": x.quantity, "average_cost": x.average_cost,
                                "expiry": x.expiry.isoformat() if x.expiry else None, "strike": x.strike,
                                "right": x.right, "iv": x.iv, "delta": x.delta, "gamma": x.gamma}
                               for x in legs]}
    report = analyse(collected, {"macro": [{"date": "2026-10-28", "title": "Fed decision"}]},
                     {"leveraged": {"AAA": 3}, "hedges": ["QQQ"], "shocks": [-0.05, 0.0]})
    assert report["horizon"] == "2026-10-30" and report["settlement"]["expiry"] == "2026-10-23"
    assert report["betas"]["AAA"] == pytest.approx(2.0, rel=0.05)
    assert {"put_hedges", "covered_calls", "scenarios", "liquidation_shock"} <= set(report)
    text = render(report)
    for heading in ("Settling today", "Covered-call cover", "Scenarios", "QQQ puts", "AAA calls", "Fed decision"):
        assert heading in text


def test_reference_tables_use_the_market_mid_when_the_chain_quotes_it():
    from cortexbacktest.options.portfolio_risk import covered_calls, listed_quotes, put_hedges
    chain = {"QQQ": [{"expiry": "2026-10-30", "right": "PUT", "strike": 680.0, "bid": 2.0, "ask": 2.2,
                      "implied_volatility": 0.2, "delta": -0.2},
                     {"expiry": "2026-10-30", "right": "PUT", "strike": 700.0, "bid": None, "ask": None,
                      "implied_volatility": None, "delta": None}]}
    listed = listed_quotes(chain)[("QQQ", "2026-10-30", "PUT")]
    rows = put_hedges(700.0, 0.25, 1.0, NEXT, NOW, 0.04, [0.0, 0.03], listed)
    assert [(r["strike"], r["source"]) for r in rows] == [(700.0, "model"), (680.0, "market")]
    assert rows[1]["premium"] == pytest.approx(210) and rows[1]["delta"] == -0.2
    calls = covered_calls(100.0, 0.4, 110.0, NEXT, NOW, 0.04, [100.0, 105.0])
    assert {r["source"] for r in calls} == {"model"}


def test_the_hedge_plan_sizes_puts_to_each_cut_and_reg_t_gives_them_no_margin_credit():
    from cortexbacktest.options.portfolio_risk import hedge_plan
    legs = [Leg("AAA", 1000, average_cost=100.0)]
    spots = {"AAA": 100.0, "QQQ": 700.0}
    beta = {"AAA": 2.0, "QQQ": 1.0}
    margin = RegTOptionMargin(stock_fraction=0.25)
    puts = {"QQQ": [{"strike": 690.0, "offset": 0.014, "premium": 900.0, "delta": -0.4,
                     "benchmark_dollar_delta_offset": 28_000.0, "source": "market"}]}
    rows = hedge_plan(legs, spots, beta, NOW, NEXT, 0.04, 1.3, margin, 40_000, 200_000, puts, [0.25, 0.5], {"QQQ": 0.2})
    assert [(r["cut"], r["contracts"]) for r in rows] == [(0.25, 2), (0.5, 4)]  # ceil(50k/28k), ceil(100k/28k)
    assert rows[1]["cost"] == 3_600 and rows[1]["remaining_benchmark_dollar_delta"] == pytest.approx(88_000)
    unhedged = liquidation_shock(legs, spots, beta, NOW, NEXT, 0.04, 1.3, margin, 40_000)
    # A long put has no loan value under Reg-T: its gains are charged back in full and its premium
    # leaves the account, so on another underlying it cannot push the forced-liquidation line out.
    assert unhedged is not None and rows[1]["liquidation_shock"] >= unhedged
