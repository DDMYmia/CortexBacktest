import copy
import hashlib
import json
from math import fsum, sqrt

import numpy as np
import polars as pl
import pytest

from cortexbacktest.adapters.qlib_predictions import evaluate_qlib_predictions
from cortexbacktest.research.alpha_diagnostics import (
    ALPHA_NAMES,
    ALPHA_DEFINITION_SHA256,
    OWNER,
    POLICY,
    PREPROCESSING,
)
from test_qlib_predictions import bundle as bundle


@pytest.fixture
def diagnostic(bundle):
    root, manifest, save = bundle
    manifest["settings"]["explainShap"] = False
    bars = json.loads((root.parent / "bars.json").read_text())
    latest = {(r["instrumentId"], r["sessionDate"]): r for r in bars["records"]}
    dates = sorted({d for _, d in latest})
    splits = manifest["settings"]["splits"]
    keys = sorted(
        (d, i) for i, d in latest if any(a <= d <= b for a, b in [splits["train"], splits["valid"]])
    )
    values = np.array([float(dates.index(d) % 17) + int(i) / 10 for d, i in keys], dtype=np.float32)
    train = [
        v
        for (d, _), v in zip(keys, values, strict=True)
        if splits["train"][0] <= d <= splits["train"][1]
    ]
    mean = fsum(train) / len(train)
    std = sqrt(fsum((v - mean) ** 2 for v in train) / len(train))
    normalized = np.asarray([(v - mean) / std for v in values], dtype=np.float32)
    y = []
    for day in sorted({d for d, _ in keys}):
        n = dates.index(day)
        z = [
            float(
                np.float32(latest[i, dates[n + 2]]["close"])
                / np.float32(latest[i, dates[n + 1]]["close"])
                - np.float32(1)
            )
            for i in sorted(manifest["instruments"])
        ]
        center = fsum(z) / 3
        scale = sqrt(fsum((v - center) ** 2 for v in z) / 2)
        y.extend((v - center) / scale for v in z)
    frame = pl.DataFrame(
        {
            "date": [d for d, _ in keys],
            "instrument": [i for _, i in keys],
            **{
                "raw:" + n: np.full(len(keys), 2.0, dtype=np.float32) if j == 0 else values
                for j, n in enumerate(ALPHA_NAMES)
            },
            **{
                "processed:" + n: np.full(len(keys), 2.0, dtype=np.float32)
                if j == 0
                else normalized
                for j, n in enumerate(ALPHA_NAMES)
            },
            "label": np.asarray(y, dtype=np.float32),
            "prediction": np.asarray(y, dtype=np.float64) * 0.5,
        }
    )
    eligible = [d for d in dates if splits["valid"][0] <= d <= splits["valid"][1]][:-2]
    selected = [j for j, (d, _) in enumerate(keys) if d in eligible][:512]
    mse = float(
        np.mean(
            (frame["prediction"].to_numpy()[selected] - frame["label"].to_numpy()[selected]) ** 2
        )
    )
    report = {
        "version": 1,
        "inputSha256": manifest["inputSha256"],
        "modelSha256": manifest["files"]["model.txt"],
        "featureSnapshotSha256": "",
        "featureDefinitionSha256": ALPHA_DEFINITION_SHA256,
        "featureNames": list(ALPHA_NAMES),
        "windows": {k: splits[k] for k in ["train", "valid"]},
        "testUsed": False,
        "featureDtype": "float32",
        "seed": 7,
        "preprocessing": PREPROCESSING,
        "labelSpace": "qlib_cross_sectional_zscore_ddof1",
        "samplePolicy": POLICY,
        "explanationOwner": OWNER,
        "explanation": {
            "method": "validation_permutation_mse",
            "samples": len(selected),
            "baselineMse": mse,
            "labelLastSession": dates[dates.index(keys[selected[-1]][0]) + 2],
            "importance": [{"feature": n, "mseIncrease": 0.0} for n in ALPHA_NAMES],
            "interpretation": "validation_sensitivity_not_causal_attribution_or_test_selection",
        },
        "shap": {"status": "not_requested"},
    }

    def publish(panel=frame, doc=report):
        panel.write_parquet(root / "features.parquet")
        digest = hashlib.sha256((root / "features.parquet").read_bytes()).hexdigest()
        doc["featureSnapshotSha256"] = digest
        manifest["files"]["features.parquet"] = digest
        raw = json.dumps(doc, allow_nan=False).encode()
        (root / "diagnostics.json").write_bytes(raw)
        manifest["files"]["diagnostics.json"] = hashlib.sha256(raw).hexdigest()
        return {
            **save(),
            "initial_cash": 10000,
            "commission_bps": 10,
            "slippage_bps": 5,
            "cash_rate_bps": 100,
        }

    return root, manifest, frame, report, publish


def test_alpha_157_drift_processing_labels_and_legacy_evaluation(diagnostic):
    _, _, _, _, publish = diagnostic
    report, _ = evaluate_qlib_predictions(**publish())
    result = report["modelDiagnostics"]
    assert len(result["inputDrift"]["features"]) == 157
    assert result["inputDrift"]["features"][0]["populationStability"]["psi"] == 0
    assert all(r["rankDisplacement"]["value"] is None for r in result["inputDrift"]["features"])
    assert result["shap"] == {"status": "not_requested"}
    assert report["qualificationIssued"] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "kind",
    [
        "input",
        "model",
        "feature_digest",
        "definition",
        "order",
        "testUsed",
        "window",
        "baseline",
        "label_horizon",
        "importance",
        "shap_request",
    ],
)
def test_rehashed_diagnostic_claims_are_rejected(diagnostic, kind):
    root, manifest, frame, doc, publish = diagnostic
    doc = copy.deepcopy(doc)
    if kind == "input":
        doc["inputSha256"] = "0" * 64
    elif kind == "model":
        doc["modelSha256"] = "0" * 64
    elif kind == "feature_digest":
        args = publish()
        doc["featureSnapshotSha256"] = "0" * 64
        raw = json.dumps(doc).encode()
        (root / "diagnostics.json").write_bytes(raw)
        manifest["files"]["diagnostics.json"] = hashlib.sha256(raw).hexdigest()
        # Rehash the outer manifest while preserving the deliberately wrong inner binding.
        raw = json.dumps(manifest).encode()
        (root / "manifest.json").write_bytes(raw)
        args["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        with pytest.raises(ValueError):
            evaluate_qlib_predictions(**args)
        return
    elif kind == "definition":
        doc["featureDefinitionSha256"] = "0" * 64
    elif kind == "order":
        doc["featureNames"].reverse()
    elif kind == "testUsed":
        doc["testUsed"] = True
    elif kind == "window":
        doc["windows"]["valid"] = manifest["settings"]["splits"]["test"]
    elif kind == "baseline":
        doc["explanation"]["baselineMse"] += 1
    elif kind == "label_horizon":
        doc["explanation"]["labelLastSession"] = manifest["settings"]["splits"]["test"][0]
    elif kind == "importance":
        doc["explanation"]["importance"][0]["feature"] = "unknown"
    elif kind == "shap_request":
        doc["shap"] = {"status": "available"}
    with pytest.raises(ValueError):
        evaluate_qlib_predictions(**publish(doc=doc))


@pytest.mark.parametrize(
    "kind",
    [
        "normalization",
        "label",
        "prediction",
        "key",
        "duplicate",
        "missing",
        "column_order",
        "test_row",
    ],
)
def test_rehashed_feature_snapshot_semantics_are_rejected(diagnostic, kind):
    _, manifest, frame, doc, publish = diagnostic
    if kind == "normalization":
        frame = frame.with_columns(pl.col("processed:MA5") + 1)
    elif kind == "label":
        frame = frame.with_columns(pl.col("label") + 1)
    elif kind == "prediction":
        frame = frame.with_columns(pl.col("prediction") + 1)
    elif kind == "key":
        frame = frame.with_columns(pl.lit("999").alias("instrument"))
    elif kind == "duplicate":
        frame = pl.concat([frame, frame.head(1)])
    elif kind == "missing":
        frame = frame.slice(1)
    elif kind == "column_order":
        frame = frame.select(frame.columns[::-1])
    elif kind == "test_row":
        frame = frame.with_columns(pl.lit(manifest["settings"]["splits"]["test"][0]).alias("date"))
    with pytest.raises(ValueError):
        evaluate_qlib_predictions(**publish(panel=frame))


def test_synthetic_shap_summary_and_rehashed_bad_additivity(diagnostic):
    _, manifest, frame, doc, publish = diagnostic
    manifest["settings"]["explainShap"] = True
    selected = frame.filter(
        pl.col("date").is_between(
            pl.lit(manifest["settings"]["splits"]["valid"][0]),
            pl.lit(doc["explanation"]["labelLastSession"]),
        )
    ).head(doc["explanation"]["samples"])
    mean = float(selected["prediction"].mean())
    doc["shap"] = {
        "status": "available",
        "method": "lightgbm_pred_contrib_tree_shap",
        "samples": doc["explanation"]["samples"],
        "meanBias": 0.0,
        "meanPrediction": mean,
        "maxAdditivityError": 0.0,
        "importance": [
            {
                "feature": n,
                "meanContribution": mean if j == 0 else 0.0,
                "meanAbsContribution": abs(mean) if j == 0 else 0.0,
            }
            for j, n in enumerate(ALPHA_NAMES)
        ],
    }
    result, _ = evaluate_qlib_predictions(**publish())
    assert result["modelDiagnostics"]["shap"]["status"] == "available"
    doc["shap"]["meanBias"] = 1.0
    with pytest.raises(ValueError, match="additive"):
        evaluate_qlib_predictions(**publish())


def test_raw_null_features_are_reported_and_existing_imputation_is_checked(diagnostic):
    _, _, frame, doc, publish = diagnostic
    frame = frame.with_columns(
        pl.lit(None, dtype=pl.Float32).alias("raw:MA5"),
        pl.lit(0.0, dtype=pl.Float32).alias("processed:MA5"),
    )
    result, _ = evaluate_qlib_predictions(**publish(panel=frame))
    row = next(
        r for r in result["modelDiagnostics"]["inputDrift"]["features"] if r["feature"] == "MA5"
    )
    assert row["trainMissingOrNonfinite"] > 0 and row["populationStability"]["psi"] is None


@pytest.mark.parametrize("missing", ["features.parquet", "diagnostics.json"])
def test_diagnostic_manifest_requires_file_pair(diagnostic, missing):
    from cortexbacktest.adapters.qlib_predictions import load_qlib_predictions

    root, manifest, _, _, publish = diagnostic
    args = publish()
    del manifest["files"][missing]
    raw = json.dumps(manifest).encode()
    (root / "manifest.json").write_bytes(raw)
    args["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    for key in ["initial_cash", "commission_bps", "slippage_bps", "cash_rate_bps"]:
        args.pop(key)
    with pytest.raises(ValueError, match="manifest"):
        load_qlib_predictions(**args)
