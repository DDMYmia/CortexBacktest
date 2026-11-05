from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from cortexbacktest.research import run_factor_research


def example():
    dates = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(5)]
    prices = pl.DataFrame({"date": dates, "A": [100., 200., 200., 220., 300.],
                           "B": [100., 100., 100., 100., 100.], "C": [100., 50., 50., 45., 30.]})
    source_dates = [dates[0] + timedelta(hours=16), dates[2] + timedelta(hours=16)]
    factors = pl.DataFrame({"date": source_dates, "A": [3., None], "B": [2., None], "C": [1., None]})
    decisions = pl.DataFrame({"date": source_dates,
                              "decision_time": [dates[1] + timedelta(minutes=30), dates[2] + timedelta(hours=17)]})
    return prices, factors, decisions


def test_delayed_decision_excludes_pre_entry_gap_and_null_decision_closes_exposure():
    prices, factors, decisions = example()
    report = run_factor_research(prices, factors, decision_times=decisions, horizons=(1,))
    assert report.target_weights["date"].equals(prices["date"])
    assert report.target_weights.select(pl.exclude("date")).head(2).sum().row(0) == (0., 0., 0.)
    assert report.equity_curve["equity"].head(3).to_list() == [1., 1., 1.]
    assert report.final_equity > 1
    assert report.equity_curve[3, "equity"] == report.equity_curve[4, "equity"]
    assert report.horizon_results[1].rank_ic_by_date[0, "date"] == prices[2, "date"]
    changed = prices.with_columns(pl.when(pl.col("date") < prices[2, "date"])
                                  .then(pl.col("A") * 10).otherwise(pl.col("A")).alias("A"))
    other = run_factor_research(changed, factors, decision_times=decisions, horizons=(1,))
    assert other.equity_curve.equals(report.equity_curve)


def test_exact_print_decision_waits_and_timezones_are_instants():
    prices, factors, decisions = example()
    decisions = decisions.head(1).with_columns(pl.lit(prices[2, "date"]).alias("decision_time"))
    report = run_factor_research(prices, factors.head(1), decision_times=decisions, horizons=(1,))
    assert report.equity_curve["equity"].head(4).to_list() == [1., 1., 1., 1.]
    zoned = prices.with_columns(pl.col("date").dt.convert_time_zone("America/New_York"))
    assert run_factor_research(zoned, factors.head(1), decision_times=decisions,
                               horizons=(1,)).equity_curve.equals(report.equity_curve)


def test_execution_aligned_research_matches_opening_print_event_replay():
    from cortexbacktest import Bar, FillPrice
    from cortexbacktest.engine import BacktestEngine
    from cortexbacktest.event.broker import BrokerSimulator
    from cortexbacktest.event.rebalance import RebalanceConfig, TargetWeightStrategy

    prices, factors, decisions = example()
    report = run_factor_research(prices, factors, decision_times=decisions, horizons=(1,))
    # Replay actual executable prints, not full daily OHLC at a guessed time.
    bars = [Bar(symbol, row["date"], row[symbol], row[symbol], row[symbol], row[symbol])
            for row in prices.iter_rows(named=True) for symbol in "ABC"]
    # Each bar is one print (O=H=L=C), so its close is the executable price.
    broker = BrokerSimulator(initial_cash=100_000., fill_price=FillPrice.CLOSE)
    strategy = TargetWeightStrategy(report.target_weights, broker=broker,
                                     config=RebalanceConfig(allow_fractional=True))
    event = BacktestEngine(broker=broker, trade_on_close=True).run(bars, strategy)
    assert min(fill.timestamp for fill in event.fills) == prices[2, "date"]
    assert all(fill.timestamp > decisions[0, "decision_time"] for fill in event.fills)
    assert event.performance.equity_curve()["equity"].to_list() == pytest.approx(
        [value * 100_000 for value in report.equity_curve["equity"]])


@pytest.mark.parametrize("failure", ["missing", "duplicate", "early", "naive", "assets", "price"])
def test_ambiguous_execution_clock_rejected(failure):
    prices, factors, decisions = example()
    if failure == "missing":
        decisions = decisions.head(1)
    elif failure == "duplicate":
        decisions = decisions.with_columns(pl.lit(decisions[1, "decision_time"]).alias("decision_time"))
    elif failure == "early":
        decisions = decisions.with_columns((pl.col("date") - pl.duration(minutes=1)).alias("decision_time"))
    elif failure == "naive":
        prices = prices.with_columns(pl.col("date").dt.replace_time_zone(None))
    elif failure == "assets":
        prices = prices.drop("C")
    else:
        prices = prices.with_columns(pl.lit(0.).alias("A"))
    with pytest.raises(ValueError):
        run_factor_research(prices, factors, decision_times=decisions, horizons=(1,))


def test_missing_held_execution_price_keeps_nav_unknown():
    prices, factors, decisions = example()
    prices = prices.with_columns(pl.when(pl.col("date") == prices[3, "date"])
                                 .then(None).otherwise(pl.col("A")).alias("A"))
    report = run_factor_research(prices, factors, decision_times=decisions, horizons=(1,))
    assert report.equity_curve["equity"].tail(2).to_list() == [None, None]
