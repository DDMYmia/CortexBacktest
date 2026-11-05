import hashlib
import json

import pytest

from test_qlib_predictions import bundle as bundle
from cortexbacktest.adapters.qlib_predictions import evaluate_qlib_predictions, load_qlib_predictions


def allocation(bundle):
    root, manifest, save = bundle
    rows = json.loads((root / "predictions.json").read_text())
    for row in rows:
        row.pop("score")
        row["weight"] = {"1": .2, "2": .3, "3": .5}[row["instrumentId"]]
    (root / "predictions.json").write_text(json.dumps(rows))
    (root / "model.zip").write_text("not deserialized by Cortex")
    manifest.update(contractVersion="model-predictions-v1", featureCount=5,
        versions={"stable-baselines3": "fixture"}, settings={"kind": "rl_portfolio",
        "splits": manifest["settings"]["splits"], "output": "weight", "seed": 7,
        "features": "causal_ohlcv_5", "labelHorizon": 2, "algorithm": "ppo",
        "trainingSteps": 4096, "trainingCostsBps": 15})
    from datetime import date, timedelta
    end = manifest["settings"]["splits"]["train"][1]
    training = dict(version=2, method="allocation_target_l1_v2", algorithm="ppo",
        requestedSteps=4096, actualSteps=4096, effectiveUpdates=320,
        reloadPredictionEqual=True, holdoutUsedForFit=False, evaluationOwner="CortexBacktest",
        rewardDefinition="next_open_to_following_open_minus_target_l1_cost",
        lastRewardSession=(date.fromisoformat(end)-timedelta(days=2)).isoformat(),
        lastRewardLabelEnd=end, trainingRewardCostsBps=15,
        initialParametersSha256="a"*64, finalParametersSha256="b"*64,
        normalizationMean=[0.0]*5, normalizationScale=[1.0]*5,
        learningOptions={"n_steps":128,"batch_size":64})
    (root / "training.json").write_text(json.dumps(training))
    manifest["files"] = {n: hashlib.sha256((root / n).read_bytes()).hexdigest()
                         for n in ("predictions.json", "model.zip", "training.json")}
    return root, manifest, save


def test_rl_weights_reach_native_execution_without_rank_conversion(bundle):
    _, _, save = allocation(bundle)
    report, runs = evaluate_qlib_predictions(**save(), initial_cash=10000,
        commission_bps=10, slippage_bps=5, cash_rate_bps=100)
    assert report["portfolioPolicy"] == "model_long_only_target_weights_daily"
    assert set(runs["qlib"].records.fills()["symbol"]) == {"INSTRUMENT:1", "INSTRUMENT:2", "INSTRUMENT:3"}
    assert report["qualificationIssued"] is False


@pytest.mark.parametrize("weight", [-.1, .9, float("nan")])
def test_invalid_rehashed_weights_are_rejected(bundle, weight):
    root, manifest, save = allocation(bundle)
    rows = json.loads((root / "predictions.json").read_text())
    rows[0]["weight"] = weight
    raw = json.dumps(rows).encode()
    (root / "predictions.json").write_bytes(raw)
    manifest["files"]["predictions.json"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        load_qlib_predictions(**save())
