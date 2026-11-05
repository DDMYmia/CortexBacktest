from dataclasses import replace
from datetime import datetime, timedelta
import json

import polars as pl
import pytest

from cortexbacktest import CorporateAction, CorporateActionType, load_backtest_dataset
from cortexbacktest.pipelines import BrokerExecutionConfig, run_mining_execution_pipeline
from cortexbacktest.pipelines.mining_store import load_mining_result, save_mining_result
from cortexbacktest.research import MiningFactorSpec, mine_factors


@pytest.fixture
def campaign(tmp_path):
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(180)]
    dates = [day for day in dates if day.weekday() < 5][:120]
    prices = pl.DataFrame({"date": dates, **{
        name: [100 * (1 + rate) ** i for i in range(len(dates))]
        for name, rate in zip("ABCD", (.001, .002, .003, .004), strict=True)}})
    raw = prices.unpivot(index="date", variable_name="symbol", value_name="close").with_columns(
        (pl.col("close") * .995).alias("open"), (pl.col("close") * 1.01).alias("high"),
        (pl.col("close") * .99).alias("low"), pl.lit(1_000_000.).alias("volume"),
    )
    data = load_backtest_dataset(bars=raw, factors=raw.select("date", "symbol").with_columns(
        pl.lit(0.).alias("factor")))
    selection = mine_factors(data.prices.head(90), [MiningFactorSpec("momentum", 2)])
    receipt = save_mining_result(selection, tmp_path / "selection")
    return load_mining_result(tmp_path / "selection", manifest_sha256=receipt), data, raw


def test_frozen_selection_executes_only_outer_targets_at_next_open_with_costs(campaign):
    selection, data, raw = campaign
    before = selection.trials.clone()
    results = run_mining_execution_pipeline(selection, data, execution_bars=raw)
    assert list(results) == [spec.name for spec in selection.selected]
    result = next(iter(results.values()))
    outer = data.prices.filter(pl.col("date") > selection.inner_end)
    assert result.target_weights.equals(result.research.target_weights)
    assert result.event.fills
    assert min(fill.timestamp for fill in result.event.fills) == outer[1, "date"]
    for fill in result.event.fills:
        price = raw.filter((pl.col("date") == fill.timestamp) & (pl.col("symbol") == fill.symbol))["open"].item()
        assert fill.price == pytest.approx(price)
    charged = run_mining_execution_pipeline(selection, data, execution_bars=raw,
                                           broker=BrokerExecutionConfig(commission_bps=10, slippage_bps=5))
    charged_result = next(iter(charged.values()))
    assert charged_result.event.summary()["final_equity"] < result.event.summary()["final_equity"]
    assert selection.trials.equals(before)
    assert charged_result.research.equity_curve.equals(result.research.equity_curve)


def test_frozen_execution_validates_inner_hash_and_raw_feed(campaign):
    selection, data, raw = campaign
    with pytest.raises(ValueError, match="Inner data changed"):
        run_mining_execution_pipeline(selection, replace(data, prices=data.prices.with_columns(pl.col("A") * 2)),
                                      execution_bars=raw)
    with pytest.raises(ValueError, match="raw OHLCV"):
        run_mining_execution_pipeline(selection, data, execution_bars=data.bars)
    with pytest.raises(ValueError, match="explicit raw OHLCV"):
        run_mining_execution_pipeline(selection, data, execution_bars=None)
    with pytest.raises(ValueError, match="identities and dates"):
        run_mining_execution_pipeline(selection, data, execution_bars=raw.head(1))


def test_frozen_execution_keeps_outer_corporate_actions(campaign):
    selection, data, raw = campaign
    action = CorporateAction("D", data.prices[100, "date"].date(), CorporateActionType.CASH_DIVIDEND, amount=5.)
    base = next(iter(run_mining_execution_pipeline(selection, data, execution_bars=raw).values()))
    paid = next(iter(run_mining_execution_pipeline(
        selection, replace(data, corporate_actions=(action,)), execution_bars=raw,
        explain_execution=True,
    ).values()))
    assert paid.event.summary()["final_equity"] > base.event.summary()["final_equity"]
    assert paid.attribution is not None


def test_mining_cli_persists_order_and_fill_results(campaign, tmp_path, capsys):
    from cortexbacktest import save_backtest_dataset
    from cortexbacktest.cli import main

    _, data, raw = campaign
    saved = save_backtest_dataset(data, tmp_path / "data")
    raw_path = tmp_path / "raw.parquet"
    raw.write_parquet(raw_path)
    out = tmp_path / "event-campaign"
    assert main(["factor-mine", "--dataset", str(saved), "--output", str(out),
                 "--direction", "momentum", "--budget", "4", "--train-sessions", "25",
                 "--test-sessions", "25", "--execution-bars", str(raw_path),
                 "--commission-bps", "10", "--slippage-bps", "5"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execution"].keys() == payload["holdout"].keys()
    assert payload["execution"]
    for result in payload["execution"].values():
        assert result["summary"]["orders"] > 0
        assert result["summary"]["fills"] > 0
    assert payload["execution_settings"]["commission_bps"] == 10
    assert json.loads((out / "holdout.json").read_text()) == payload


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf")])
def test_pipeline_summary_does_not_reuse_stale_research_nav(campaign, missing):
    selection, data, raw = campaign
    result = next(iter(run_mining_execution_pipeline(selection, data, execution_bars=raw).values()))
    report = replace(result.research, equity_curve=pl.DataFrame({"equity": [1.1, missing]}))
    assert replace(result, research=report).summary()["research_final_equity"] is None


def test_crypto_close_labels_execute_on_actual_next_open_and_keep_attribution(campaign):
    _, data, raw = campaign
    # Crypto daily bars close on the following UTC day. The target row must
    # retain its session association when execution moves to the previous day.
    raw = raw.with_columns(pl.col("date").dt.replace_time_zone("UTC"))
    data = load_backtest_dataset(bars=raw, factors=raw.select("date", "symbol").with_columns(
        pl.lit(0.).alias("factor")))
    selection = mine_factors(data.prices.head(90), [MiningFactorSpec("momentum", 2)])
    times = {day: day - timedelta(days=1) for day in data.prices["date"]}
    result = next(iter(run_mining_execution_pipeline(
        selection, data, execution_bars=raw, broker=BrokerExecutionConfig(opening_times=times),
        explain_execution=True,
    ).values()))
    outer = data.prices.filter(pl.col("date") > selection.inner_end)
    assert min(fill.timestamp for fill in result.event.fills) == times[outer[1, "date"]]
    assert result.event.performance.equity_curve()[0, "timestamp"] == outer[0, "date"]
    assert result.attribution.to_dict()["reconciliation_residual"] == pytest.approx(0.)
    for fill in result.event.fills:
        assert fill.timestamp in times.values()
