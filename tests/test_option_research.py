from datetime import datetime, timedelta, timezone
from math import log

import numpy as np
import polars as pl
import pytest

from cortexbacktest.options import (
    OptionContract, OptionMarketFrame, OptionQuote, ScheduledOptionTargets,
    delta_hedge_targets, goyal_saretto_signal, historical_volatility,
    option_deflated_sharpe, option_frames_from_polars, option_scenario_summary,
    probability_of_backtest_overfitting, purged_option_split, run_option_scenarios,
    select_option,
)
T=datetime(2026,1,1,tzinfo=timezone.utc)


def fixture():
    contracts=[OptionContract("C","S","CALL",100,T+timedelta(days=30)),
               OptionContract("P","S","PUT",100,T+timedelta(days=30))]
    frames=[]
    for day in range(3):
        at=T+timedelta(days=day)
        quotes={s:OptionQuote(s,at,at,b,a,1000,1000,implied_volatility=.2,delta=d)
                for s,b,a,d in [("S",99,101,1),("C",2+day,3+day,.5),("P",2,3,-.5)]}
        frames.append(OptionMarketFrame(at,quotes))
    return contracts,frames


def test_mechanical_selection_signals_and_delta_units():
    contracts,frames=fixture()
    assert select_option(contracts,frames[0],underlying="S",right="PUT",target_delta=-.5).symbol=="P"
    assert goyal_saretto_signal(.3,.2,.4)==pytest.approx(0)
    assert goyal_saretto_signal(.4,.2,.2)==pytest.approx(log(2))
    prices=[100,101,98,102]
    assert historical_volatility(prices)==pytest.approx(np.std(np.diff(prices)/prices[:-1],ddof=1)*np.sqrt(252))
    assert delta_hedge_targets({c.symbol:c for c in contracts},frames[0],{"C":2,"P":-1})=={"S":-150}
    with pytest.raises(ValueError):
        goyal_saretto_signal(.2,0,.3)


def test_purged_cross_section_and_test_boundary_isolation():
    decisions=[T,T,T+timedelta(days=1),T+timedelta(days=2),T+timedelta(days=3)]
    ends=[T,T+timedelta(days=2),T+timedelta(days=1),T+timedelta(days=3),T+timedelta(days=4)]
    train,test=purged_option_split(decisions,ends,test_start=T+timedelta(days=2),test_end=T+timedelta(days=3))
    assert train.tolist()==[2]  # Both rows at T purged because one overlapping label.
    assert test.tolist()==[3]
    embargo,_=purged_option_split(decisions,ends,test_start=T+timedelta(days=2),test_end=T+timedelta(days=3),embargo=timedelta(days=1))
    assert embargo.tolist()==[]
    # Future labels do not affect training membership.
    changed=ends[:-1]+[T+timedelta(days=999)]
    assert purged_option_split(decisions,changed,test_start=T+timedelta(days=2),test_end=T+timedelta(days=3))[0].tolist()==[2]


def test_cscv_known_rank_reversal_and_stability():
    # Candidate A wins first half, loses second half; B does the opposite.
    a=np.array([.1,.3,.2,.4,-.1,-.3,-.2,-.4])
    result=probability_of_backtest_overfitting(np.c_[a,-a],blocks=2)
    assert result["pbo"]==1
    assert result["combinations"]==2
    assert result["logits"]==pytest.approx([log(.5),log(.5)])
    stable=np.c_[a+1,a-1]
    assert probability_of_backtest_overfitting(stable,blocks=2)["pbo"]==0
    with pytest.raises(ValueError,match="constant"):
        probability_of_backtest_overfitting(np.ones((8,2)),blocks=2)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.ones((9,2)),blocks=4)


def test_dsr_selection_penalty_and_no_silent_defaults():
    r=np.array([-.03,.02,.01,.04,-.01,.02,.01,-.005]*10)
    one=option_deflated_sharpe(r,independent_trials=1,trial_sharpe_variance=.01)
    many=option_deflated_sharpe(r,independent_trials=100,trial_sharpe_variance=.01)
    assert 0 < many < one < 1
    with pytest.raises(ValueError):
        option_deflated_sharpe([1]*10,independent_trials=2,trial_sharpe_variance=.1)


def test_parallel_scenarios_isolated_and_cost_reconciliation():
    contracts,frames=fixture()
    template=ScheduledOptionTargets(tuple(contracts),{T:{"C":1,"P":-1},T+timedelta(days=1):{"C":0,"P":0}})
    serial=run_option_scenarios(contracts,frames,template.strategy)
    parallel=run_option_scenarios(contracts,frames,template.strategy,workers=2)
    for value in serial:
        assert serial[value].snapshots.equals(parallel[value].snapshots)
    summary=option_scenario_summary(serial)
    assert summary.height==5
    assert serial[0].summary()["net_pnl"]-serial[1].summary()["net_pnl"]==pytest.approx(200)


def test_strict_tabular_adapter_and_future_rejection():
    at=T+timedelta(days=1)
    df=pl.DataFrame({"timestamp":[T],"event_time":[T],"available_at":[T],
                     "symbol":["C"],"bid":[1.],"ask":[2.]})
    assert option_frames_from_polars(df)[0].quotes["C"].mid==1.5
    with pytest.raises(ValueError,match="duplicate"):
        option_frames_from_polars(pl.concat([df,df]))
    with pytest.raises(ValueError,match="future"):
        option_frames_from_polars(df.with_columns(pl.lit(at).alias("available_at")))


def test_tabular_adapter_groups_by_instant_not_by_wall_clock():
    """The adapter sorts once and groups contiguously, so equal instants in
    different zone representations must land in the same frame."""
    eastern=timezone(timedelta(hours=-5))
    rows=pl.DataFrame({"timestamp":[T,T,T+timedelta(days=1)],
                       "event_time":[T,T,T+timedelta(days=1)],
                       "available_at":[T,T,T+timedelta(days=1)],
                       "symbol":["B","A","A"],"bid":[1.,2.,3.],"ask":[2.,3.,4.]})
    frames=option_frames_from_polars(rows.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/New_York")))
    assert len(frames)==2
    assert sorted(frames[0].quotes)==["A","B"]      # one frame, both symbols
    assert frames[0].timestamp==T and frames[1].timestamp==T+timedelta(days=1)
    assert frames[0].quotes["A"].event_time.utcoffset()==timedelta(0)
    assert T.astimezone(eastern)==frames[0].timestamp


def test_tabular_adapter_fills_absent_optional_columns_once():
    rows=pl.DataFrame({"timestamp":[T],"event_time":[T],"available_at":[T],
                       "symbol":["C"],"bid":[1.],"ask":[2.],"delta":[0.4]})
    quote=option_frames_from_polars(rows)[0].quotes["C"]
    assert quote.delta==0.4
    assert (quote.bid_size,quote.ask_size,quote.implied_volatility,
            quote.open_interest)==(None,None,None,None)
    with pytest.raises(ValueError,match="timezone-aware"):
        option_frames_from_polars(rows.with_columns(
            pl.col("timestamp").dt.replace_time_zone(None)))
