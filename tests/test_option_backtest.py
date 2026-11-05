from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json

import pytest

from cortexbacktest.event.financing import RateSchedule, SettlementModel
from cortexbacktest.options import (
    OptionBacktestConfig, OptionContract, OptionLifecycleEvent, OptionMarketFrame,
    OptionQuote, QuotePolicy, ResearchOptionMargin, ScheduledOptionTargets, run_option_backtest,
)

T = datetime(2026, 1, 5, 21, tzinfo=timezone.utc)

def contract(**kw):
    return replace(OptionContract("C", "S", "CALL", 100, T+timedelta(days=3)), **kw)

def quote(symbol, at, bid, ask, **kw):
    return OptionQuote(symbol, at, at, bid, ask, bid_size=10, ask_size=10, **kw)

def frame(day, bid=2, ask=3, spot=100, lifecycle=(), **kw):
    at = T+timedelta(days=day)
    return OptionMarketFrame(at, {"C":quote("C",at,bid,ask,delta=.5),
                                  "S":quote("S",at,spot-.1,spot+.1)}, lifecycle, **kw)

def run(frames, targets, *, c=None, config=None):
    return run_option_backtest([c or contract()],frames,
        lambda f,a: targets.get(f.timestamp, {}),config=config)

def identity(result):
    a = result.attribution
    assert a["net_pnl"] == pytest.approx(a["option_market_pnl"]+a["stock_market_pnl"]
        -a["option_spread"]-a["stock_spread"]-a["option_fees"]-a["stock_fees"]
        +a["financing"]+a["dividends"], abs=1e-8)


def test_next_frame_bid_ask_costs_and_replay():
    frames=[frame(0),frame(1),frame(2,bid=4,ask=5)]
    targets={T:{"C":2},T+timedelta(days=1):{"C":0}}
    result=run(frames,targets)
    assert result.fills["price"].to_list()==[3,4]
    assert result.fills["timestamp"][0]==T+timedelta(days=1)
    assert result.summary()["net_pnl"]==pytest.approx(200-2.6)
    assert result.attribution["option_spread"]==200
    identity(result)
    again=run(frames,targets)
    assert again.snapshots.equals(result.snapshots)
    assert again.manifest==result.manifest
    json.dumps(dict(result.manifest))


@pytest.mark.parametrize("quantity,right,settlement",[(1,"CALL","cash"),(-1,"CALL","cash"),
    (1,"PUT","cash"),(-1,"PUT","physical"),(1,"CALL","physical"),(-1,"CALL","physical")])
def test_expiry_cash_and_physical_accounting(quantity,right,settlement):
    c=contract(right=right,settlement=settlement)
    price=110 if right=="CALL" else 90
    at=T+timedelta(days=3)
    event=OptionLifecycleEvent("C",at,at,price)
    result=run([frame(0),frame(1),frame(3,spot=price,lifecycle=(event,))],
               {T:{"C":quantity}},c=c)
    initial_premium=3 if quantity>0 else 2
    assert result.summary()["net_pnl"]==pytest.approx(quantity*100*(10-initial_premium)-.65)
    positions=json.loads(result.snapshots["positions"][-1])
    # Settled contracts leave the book rather than sitting in it as zeros: the
    # lifecycle row is what records that the expiry happened.
    assert "C" not in positions
    assert result.lifecycle["symbol"].to_list()==["C"]
    if settlement=="physical":
        assert positions["S"]==quantity*100*(1 if right=="CALL" else -1)
    identity(result)


def test_missing_fixing_and_no_theoretical_fill():
    with pytest.raises(ValueError,match="missing expiry"):
        run([frame(0),frame(1),frame(3)],{T:{"C":1}})
    at=T+timedelta(days=1)
    missing=OptionMarketFrame(at,{"S":quote("S",at,99,101)})
    result=run([frame(0),missing],{T:{"C":1}})
    assert result.fills.height==0
    assert result.orders["reason"].to_list()==["missing_quote"]


def test_future_stale_and_same_observation_quotes():
    with pytest.raises(ValueError,match="future"):
        OptionMarketFrame(T,{"C":quote("C",T+timedelta(seconds=1),1,2)})
    with pytest.raises(ValueError,match="timezone"):
        replace(contract(),expiry=datetime(2026,1,8))
    old=frame(0).quotes
    result=run([frame(0),OptionMarketFrame(T+timedelta(days=1),old)],{T:{"C":1}})
    assert result.orders["reason"].to_list()==["quote_not_after_decision"]
    with pytest.raises(ValueError,match="stale valuation"):
        run([frame(0),frame(1),OptionMarketFrame(T+timedelta(days=2),{})],
            {T:{"C":1}},config=OptionBacktestConfig(quote_policy=QuotePolicy(max_age=timedelta(hours=1))))


def test_partial_ioc_and_last_signal_reported():
    result=run([frame(0),frame(1)],{T:{"C":20},T+timedelta(days=1):{"C":0}},
               config=OptionBacktestConfig(participation=.5))
    assert result.fills["quantity"].to_list()==[5]
    assert result.orders["reason"].to_list()==["partial_ioc","end_of_data"]


def test_initial_margin_rejection_and_forced_liquidation():
    reject=run([frame(0),frame(1)],{T:{"C":-1}},config=OptionBacktestConfig(initial_cash=1000))
    assert reject.orders["reason"].to_list()==["initial_margin"]
    result=run([frame(0),frame(1),frame(2,49,51,150)],{T:{"C":-1}},
               config=OptionBacktestConfig(initial_cash=3000))
    assert result.fills["forced"].to_list()==[False,True]
    assert json.loads(result.snapshots["positions"][-1])["C"]==0
    assert result.summary()["margin_breach_frames"]==1
    identity(result)


def test_early_assignment_and_nonexercise_override():
    c=contract(exercise_style="american",settlement="physical")
    at=T+timedelta(days=2)
    early=OptionLifecycleEvent("C",at,at,105,quantity=-1)
    result=run([frame(0),frame(1),frame(2,spot=105,lifecycle=(early,))],{T:{"C":-1}},c=c)
    assert json.loads(result.snapshots["positions"][-1])["S"]==-100
    identity(result)
    with pytest.raises(ValueError,match="American"):
        run([frame(0),frame(1),frame(2,lifecycle=(early,))],{T:{"C":-1}})
    expiry=T+timedelta(days=3)
    event=OptionLifecycleEvent("C",expiry,expiry,110,exercise=False)
    result=run([frame(0),frame(1),frame(3,lifecycle=(event,))],{T:{"C":1}})
    assert result.summary()["net_pnl"]==pytest.approx(-300.65)


def test_settlement_and_financing_and_dividend():
    cfg=OptionBacktestConfig(credit_rate=.05,settlement_model=SettlementModel(delay_days=2))
    result=run([frame(0),frame(1),frame(2,dividends={"S":1})],{T:{"S":-2,"C":-1}},config=cfg)
    assert result.snapshots["unsettled_cash"][1] > 0
    assert result.attribution["dividends"]==-2
    identity(result)


def test_inputs_and_view_immutable():
    with pytest.raises(ValueError):
        OptionBacktestConfig(spread_fraction=2)
    with pytest.raises(ValueError):
        quote("C",T,3,2)
    with pytest.raises(ValueError):
        ResearchOptionMargin(alpha=-1)
    with pytest.raises(ValueError,match="strictly increasing"):
        run([frame(0),frame(0)],{})
    with pytest.raises(ValueError,match="integer"):
        run([frame(0)],{T:{"C":.5}})
    with pytest.raises(ValueError,match="multi-currency"):
        run([frame(0)],{},c=contract(currency="EUR"))
    def bad(f,a):
        a.positions["C"]=100
        return {}
    with pytest.raises(TypeError):
        run_option_backtest([contract()],[frame(0)],bad)


def test_zero_bid_is_valid_for_closing_and_no_size_is_not_liquidity():
    result=run([frame(0),frame(1),frame(2,0,.5)],{T:{"C":1},T+timedelta(days=1):{"C":0}})
    assert result.fills["price"].to_list()==[3,0]
    at=T+timedelta(days=1)
    no_size=OptionQuote("C",at,at,2,3)
    f=OptionMarketFrame(at,{**frame(1).quotes,"C":no_size})
    result=run([frame(0),f],{T:{"C":1}})
    assert result.orders["reason"].to_list()==["missing_size"]


def test_hedge_uses_actual_filled_contracts_and_next_quote():
    template=ScheduledOptionTargets((contract(),),{T:{"C":1}},hedge_every=timedelta(days=1))
    result=run_option_backtest([contract()],[frame(0),frame(1),frame(2)],template.strategy())
    # Only displayed 10 shares can fill, not 50: partial hedge remains visible.
    assert result.fills["symbol"].to_list()==["C","S"]
    assert result.fills["quantity"].to_list()==[1,-10]
    identity(result)


def test_last_trade_cutoff_separate_from_expiry():
    c=contract(last_trade_at=T+timedelta(days=1))
    result=run([frame(0),frame(1)],{T:{"C":1}},c=c)
    assert result.orders["reason"].to_list()==["expired"]
    with pytest.raises(ValueError,match="last trading"):
        contract(last_trade_at=T+timedelta(days=5))


def test_liquidation_liquidity_shortfall_is_not_hidden():
    cfg=OptionBacktestConfig(initial_cash=3000)
    stressed=frame(2,49,51,150)
    q=replace(stressed.quotes["C"],ask_size=0)
    result=run([frame(0),frame(1),OptionMarketFrame(stressed.timestamp,{**stressed.quotes,"C":q})],
               {T:{"C":-1}},config=cfg)
    assert result.orders["reason"].to_list()==["filled","no_liquidity"]
    assert json.loads(result.snapshots["positions"][-1])["C"]==-1
    assert result.snapshots["margin_breach"][-1]


def test_duplicate_assignment_and_wrong_sign_fail():
    c=contract(exercise_style="american")
    at=T+timedelta(days=2)
    wrong=OptionLifecycleEvent("C",at,at,105,quantity=1)
    with pytest.raises(ValueError,match="signed position"):
        run([frame(0),frame(1),frame(2,lifecycle=(wrong,))],{T:{"C":-1}},c=c)
    event=replace(wrong,quantity=-1)
    with pytest.raises(ValueError,match="duplicate"):
        run([frame(0),frame(1),frame(2,lifecycle=(event,event))],{T:{"C":-1}},c=c)


def test_unsettled_proceeds_do_not_earn_early_interest():
    cfg=OptionBacktestConfig(initial_cash=10000,credit_rate=.365,option_fee=0,
                            settlement_model=SettlementModel(delay_days=2))
    c=contract(expiry=T+timedelta(days=20))
    result=run([frame(0),frame(1),frame(2)],{T:{"C":-1}},c=c,config=cfg)
    # First day earns 10; second earns 10.01 on pre-premium settled cash.
    assert result.attribution["financing"]==pytest.approx(20.01)
    identity(result)


def test_cash_settlement_does_not_apply_equity_exercise_threshold():
    at=T+timedelta(days=3)
    event=OptionLifecycleEvent("C",at,at,100.005)
    result=run([frame(0),frame(1),frame(3,spot=100.005,lifecycle=(event,))],{T:{"C":1}})
    assert result.summary()["net_pnl"]==pytest.approx(-300.15)
    assert result.lifecycle["exercise"].to_list()==[True]
    empty=run([frame(0)],{})
    assert "quantity" in empty.fills.columns
    assert "reason" in empty.orders.columns


class TestDatedFinancingRates:
    """A run financed from a published curve rather than one number.

    Financing is not a rounding error: over 2010-2023 a flat 5% produced 105,219
    of a 114,319 net P&L on a SPY short-put run, none of it the strategy's.
    """

    @staticmethod
    def _schedule(**kw):
        return RateSchedule.from_annual_percent(
            [(date(2026, 1, 5), 4.0), (date(2026, 1, 7), 8.0)], source="test", **kw)

    def test_the_rate_changes_with_the_date(self):
        schedule = self._schedule()
        assert schedule.rates(T)[0] == pytest.approx(0.04)
        assert schedule.rates(T + timedelta(days=1))[0] == pytest.approx(0.04)
        assert schedule.rates(T + timedelta(days=2))[0] == pytest.approx(0.08)

    def test_a_date_before_the_curve_is_refused_rather_than_financed_at_zero(self):
        with pytest.raises(ValueError, match="after the"):
            self._schedule().rates(T - timedelta(days=1))

    def test_a_debit_spread_separates_what_is_paid_from_what_is_earned(self):
        credit, debit, borrow = self._schedule(debit_spread=0.015, borrow=0.003).rates(T)
        assert (credit, debit, borrow) == pytest.approx((0.04, 0.055, 0.003))

    @pytest.mark.parametrize("bad", [
        {"credit": (0.04, -0.01)},                       # negative rate
        {"credit": (0.04,)},                             # wrong length
        {"dates": (date(2026, 1, 7), date(2026, 1, 5))},  # unsorted
    ])
    def test_an_unusable_curve_is_refused_at_construction(self, bad):
        base = {"dates": (date(2026, 1, 5), date(2026, 1, 7)), "credit": (0.04, 0.08),
                "debit": (0.04, 0.08), "borrow": (0.0, 0.0)}
        with pytest.raises(ValueError):
            RateSchedule(**{**base, **bad})

    def test_two_rates_at_once_is_refused(self):
        # The flat fields and a curve would both be applied; which one financed
        # the run would not be visible in the result.
        with pytest.raises(ValueError, match="must stay at 0"):
            OptionBacktestConfig(financing=self._schedule(), credit_rate=0.05)

    def test_the_curve_is_what_actually_accrues(self):
        # Two days at 4% then one at 8%, on cash only, with nothing traded.
        frames = [frame(0), frame(1), frame(2), frame(3)]
        result = run(frames, {}, config=OptionBacktestConfig(
            initial_cash=100_000, financing=self._schedule()))
        # Compounded, not simple: each interval accrues on the balance the
        # previous one left behind.
        cash = 100_000.0
        for rate in (0.04, 0.04, 0.08):
            cash *= 1 + rate / 365
        expected = cash - 100_000
        assert result.attribution["financing"] == pytest.approx(expected, rel=1e-9)
        identity(result)

    def test_the_manifest_names_the_curve_without_carrying_it(self):
        result = run([frame(0), frame(1)], {}, config=OptionBacktestConfig(
            financing=self._schedule()))
        assert result.manifest["financing_model"] == "RateSchedule"
        assert self._schedule().digest in result.manifest["financing_parameters"]
        assert "financing" not in result.manifest["config"]
        json.dumps(dict(result.manifest))
