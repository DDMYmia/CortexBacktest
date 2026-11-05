import copy
import hashlib
import json

import polars as pl
import pytest

from cortexbacktest.research.model_diagnostics import (
    FEATURES, input_diagnostics, population_stability, rank_displacement,
)
from cortexbacktest.adapters.qlib_predictions import evaluate_qlib_predictions
from test_qlib_predictions import bundle as bundle
from test_model_predictions import allocation


def test_psi_train_bins_constants_missing_and_nonfinite():
    same = population_stability(list(range(40)), list(range(40)))
    assert same["psi"] == 0
    moved = population_stability(list(range(40)), list(range(100, 140)))
    assert moved["edges"] == same["edges"] and moved["psi"] > 0
    assert population_stability([2.] * 30, [2.] * 30)["psi"] == 0
    assert population_stability([2.] * 30, [3.] * 30)["psi"] > 0
    assert population_stability([None, float("inf")], [1.] * 30)["reason"] == "insufficient_finite_samples"


def test_rank_identity_changes_not_common_scale():
    frame = pl.DataFrame([{"date": day, "instrument": str(i), "x": float(i)}
                          for day in range(5) for i in range(50)])
    assert rank_displacement(frame, frame.with_columns(pl.col("x") + 1000), "x")["value"] == 0
    assert rank_displacement(frame, frame.with_columns(-pl.col("x")), "x")["value"] > .4
    assert rank_displacement(frame, frame.filter(pl.col("instrument") == "0"), "x")["value"] is None


def test_test_future_does_not_change_drift():
    latest = {(str(i), f"2025-01-{d:02}"): {"close": 10. + d * (i + 1), "volume": 10. + d}
              for i in range(3) for d in range(1, 32)}
    splits = {"train": ["2025-01-01", "2025-01-15"], "valid": ["2025-01-18", "2025-01-25"]}
    first = input_diagnostics(latest, splits)
    changed = copy.deepcopy(latest)
    for (_, day), row in changed.items():
        if day > "2025-01-25":
            row["close"] *= 100
            row["volume"] *= 100
    assert input_diagnostics(changed, splits) == first
    assert len(first["features"]) == 5 and first["testUsed"] is False


def attach(bundle):
    root, manifest, save = allocation(bundle)
    explanation = {"version": 1, "inputSha256": manifest["inputSha256"],
        "modelSha256": manifest["files"]["model.zip"], "featureNames": list(FEATURES),
        "window": manifest["settings"]["splits"]["valid"], "testUsed": False, "seed": 7,
        "explanationOwner": "producer_reported_not_independently_replayed_model",
        "imputation": "existing_worker_nonfinite_to_zero", "method": "unavailable",
        "reason": "supervised_mse_not_applicable_to_rl"}
    return root, manifest, save, explanation


@pytest.mark.parametrize("tamper", [None, "modelSha256", "window", "testUsed"])
def test_bound_diagnostic_consumption_and_rehashed_tamper(bundle, tamper):
    root, manifest, save, explanation = attach(bundle)
    if tamper == "modelSha256":
        explanation[tamper] = "0" * 64
    elif tamper == "window":
        explanation[tamper] = manifest["settings"]["splits"]["test"]
    elif tamper == "testUsed":
        explanation[tamper] = True
    raw = json.dumps(explanation).encode()
    (root / "diagnostics.json").write_bytes(raw)
    manifest["files"]["diagnostics.json"] = hashlib.sha256(raw).hexdigest()
    args = {**save(), "initial_cash": 10000, "commission_bps": 10, "slippage_bps": 5, "cash_rate_bps": 100}
    if tamper:
        with pytest.raises(ValueError, match="explanation binding"):
            evaluate_qlib_predictions(**args)
    else:
        report, _ = evaluate_qlib_predictions(**args)
        diagnostic = report["modelDiagnostics"]
        assert diagnostic["status"] == "available" and report["qualificationIssued"] is False
        assert all(x["rankDisplacement"]["value"] is None for x in diagnostic["inputDrift"]["features"])
