from datetime import date, timedelta
from hashlib import sha256
import json

import polars as pl
import pytest

from cortexbacktest.research import (
    WalkForwardSplit, auto_mine_factors, evaluate_holdout,
)
from cortexbacktest.pipelines.mining_store import load_mining_result, save_mining_result


def panel():
    return pl.DataFrame({
        "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(120)],
        **{name: [100 * (1 + rate) ** i for i in range(120)]
           for name, rate in zip("ABCD", (.001, .002, .003, .004), strict=True)},
    })


@pytest.mark.parametrize("datetime_dates", [False, True])
def test_checkpoint_restores_full_trials_and_frozen_outer_evaluation(tmp_path, datetime_dates):
    prices = panel()
    if datetime_dates:
        prices = prices.with_columns(pl.col("date").cast(pl.Datetime("us", "UTC")))
    frozen = auto_mine_factors(prices.head(90), research_direction="momentum", windows=(2, 5),
                               validation=WalkForwardSplit(30, 30), budget=4)
    directory = tmp_path / "selection"
    receipt = save_mining_result(frozen, directory)
    restored = load_mining_result(directory, manifest_sha256=receipt)
    assert restored.selected == frozen.selected
    assert restored.trials.equals(frozen.trials)
    assert restored.settings == frozen.settings
    assert restored.inner_end == frozen.inner_end
    before, after = evaluate_holdout(frozen, prices), evaluate_holdout(restored, prices)
    assert before and before.keys() == after.keys()
    for name in before:
        assert before[name].equity_curve.equals(after[name].equity_curve)
    with pytest.raises(FileExistsError):
        save_mining_result(frozen, directory)


def test_checkpoint_rejects_modified_manifest_trials_and_engine(tmp_path, monkeypatch):
    import cortexbacktest.pipelines.mining_store as store

    frozen = auto_mine_factors(panel(), research_direction="momentum", budget=1)
    directory = tmp_path / "selection"
    receipt = save_mining_result(frozen, directory)
    with pytest.raises(ValueError, match="manifest_digest"):
        load_mining_result(directory, manifest_sha256="0" * 64)
    manifest_path = directory / "manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    manifest["selected"] = []
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest_digest"):
        load_mining_result(directory, manifest_sha256=receipt)
    manifest_path.write_bytes(original)
    with monkeypatch.context() as patch:
        patch.setattr(store, "engine_code_digest", lambda: "new-engine")
        with pytest.raises(ValueError, match="engine_changed"):
            load_mining_result(directory, manifest_sha256=receipt)
    trials = directory / "trials.parquet"
    trials.write_bytes(trials.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="trials_digest"):
        load_mining_result(directory, manifest_sha256=receipt)


def test_interrupted_selection_is_not_recoverable_as_complete(tmp_path):
    directory = tmp_path / "selection"
    directory.mkdir()
    (directory / "trials.parquet").write_bytes(b"interrupted")
    with pytest.raises(FileNotFoundError):
        load_mining_result(directory, manifest_sha256=sha256(b"").hexdigest())


def test_cli_mines_freezes_and_backtests_saved_dataset(tmp_path, capsys):
    from cortexbacktest import load_backtest_dataset, save_backtest_dataset
    from cortexbacktest.cli import main

    bars = panel().unpivot(index="date", variable_name="symbol", value_name="adj_close")
    dataset = load_backtest_dataset(bars=bars, factors=bars.select("date", "symbol").with_columns(pl.lit(0.0).alias("factor")))
    saved = save_backtest_dataset(dataset, tmp_path / "data")
    out = tmp_path / "campaign"
    args = ["factor-mine", "--dataset", str(saved), "--output", str(out),
            "--direction", "momentum", "--train-sessions", "25", "--test-sessions", "25",
            "--budget", "4"]
    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "evaluated"
    assert payload["holdout"] and payload["trials"] == 4
    assert payload["evaluation_mode"] == "retrospective_not_pit_certified"
    assert json.loads((out / "holdout.json").read_text()) == payload
    frozen = load_mining_result(out, manifest_sha256=payload["manifest_sha256"])
    assert frozen.selected
    before = (out / "manifest.json").read_bytes()
    assert main(args) == 1  # Never overwrite a completed campaign.
    assert (out / "manifest.json").read_bytes() == before


@pytest.mark.parametrize('mode', ['long_only', 'long_short'])
def test_frozen_portfolio_mode_survives_checkpoint_and_holdout(tmp_path, mode):
    from cortexbacktest.research.mining import frozen_factor_targets

    prices = panel()
    frozen = auto_mine_factors(prices.head(90), research_direction='momentum', windows=(2, 5),
        validation=WalkForwardSplit(30, 30), budget=4, portfolio_mode=mode)
    digest = save_mining_result(frozen, tmp_path / mode)
    restored = load_mining_result(tmp_path / mode, manifest_sha256=digest)
    assert restored.settings['portfolio_mode'] == mode
    targets = frozen_factor_targets(restored, prices)
    reports = evaluate_holdout(restored, prices)
    assert targets and targets.keys() == reports.keys()
    for name, weights in targets.items():
        values = weights.select(pl.exclude('date')).to_numpy()
        assert (values < 0).any() == (mode == 'long_short')
        assert weights.equals(reports[name].target_weights_frame())


def test_unknown_portfolio_mode_rejected():
    with pytest.raises(ValueError, match='portfolio mode'):
        auto_mine_factors(panel(), research_direction='momentum', portfolio_mode='clip_at_broker')
