"Validate bounded producer features and recompute Alpha158 diagnostics without Qlib."

from __future__ import annotations

from math import isfinite
import warnings

import numpy as np
import polars as pl

from .model_diagnostics import panel_diagnostics

ALPHA_NAMES = (
    "KMID",
    "KLEN",
    "KMID2",
    "KUP",
    "KUP2",
    "KLOW",
    "KLOW2",
    "KSFT",
    "KSFT2",
    "OPEN0",
    "HIGH0",
    "LOW0",
) + tuple(
    f"{group}{n}"
    for group in (
        "ROC",
        "MA",
        "STD",
        "BETA",
        "RSQR",
        "RESI",
        "MAX",
        "MIN",
        "QTLU",
        "QTLD",
        "RANK",
        "RSV",
        "IMAX",
        "IMIN",
        "IMXD",
        "CORR",
        "CORD",
        "CNTP",
        "CNTN",
        "CNTD",
        "SUMP",
        "SUMN",
        "SUMD",
        "VMA",
        "VSTD",
        "WVMA",
        "VSUMP",
        "VSUMN",
        "VSUMD",
    )
    for n in (5, 10, 20, 30, 60)
)
ALPHA_DEFINITION_SHA256 = "cfa6680a5b7395c244e4195fcf7204167cf03e1abb7a828bb213f6431db360c0"
PREPROCESSING = "qlib_processinf_cross_section_then_train_zscore_then_fillna_zero"
POLICY = "first_512_chronological_validation_rows_after_two_session_purge"
OWNER = "producer_reported_bound_model_not_independently_replayed"


def _number(x):
    return type(x) in (int, float) and isfinite(x)


def alpha_input_diagnostics(
    frame: pl.DataFrame, report: dict, manifest: dict, latest: dict
) -> dict:
    splits = manifest["settings"]["splits"]
    common = {
        "version",
        "inputSha256",
        "modelSha256",
        "featureSnapshotSha256",
        "featureDefinitionSha256",
        "featureNames",
        "windows",
        "testUsed",
        "featureDtype",
        "seed",
        "preprocessing",
        "labelSpace",
        "samplePolicy",
        "explanationOwner",
        "explanation",
        "shap",
    }
    if (
        not isinstance(report, dict)
        or set(report) != common
        or type(report["version"]) is not int
        or report["version"] != 1
        or report["inputSha256"] != manifest["inputSha256"]
        or report["modelSha256"] != manifest["files"]["model.txt"]
        or report["featureSnapshotSha256"] != manifest["files"]["features.parquet"]
        or report["featureNames"] != list(ALPHA_NAMES)
        or report["featureDefinitionSha256"] != ALPHA_DEFINITION_SHA256
        or report["windows"] != {k: splits[k] for k in ["train", "valid"]}
        or report["testUsed"] is not False
        or report["featureDtype"] != "float32"
        or report["seed"] != 7
        or report["preprocessing"] != PREPROCESSING
        or report["labelSpace"] != "qlib_cross_sectional_zscore_ddof1"
        or report["samplePolicy"] != POLICY
        or report["explanationOwner"] != OWNER
    ):
        raise ValueError("Alpha diagnostics binding mismatch")
    columns = [
        "date",
        "instrument",
        *["raw:" + n for n in ALPHA_NAMES],
        *["processed:" + n for n in ALPHA_NAMES],
        "label",
        "prediction",
    ]
    if (
        frame.columns != columns
        or frame.schema["date"] != pl.String
        or frame.schema["instrument"] != pl.String
    ):
        raise ValueError("Alpha feature order/schema mismatch")
    expected = {
        (d, i) for i, d in latest if any(a <= d <= b for a, b in [splits["train"], splits["valid"]])
    }
    keys = list(zip(frame["date"], frame["instrument"], strict=True))
    if len(keys) != len(expected) or set(keys) != expected or keys != sorted(keys):
        raise ValueError(
            "Alpha feature panel must cover exact ordered train/valid keys, never test"
        )
    for name in columns[2:]:
        if not frame.schema[name].is_numeric() or (
            frame[name].null_count() and not name.startswith("raw:")
        ):
            raise ValueError("Alpha numeric feature snapshot required; missing raw values use NaN")
    clean = frame.select(["raw:" + n for n in ALPHA_NAMES]).to_numpy().astype(np.float32)
    x = frame.select(["processed:" + n for n in ALPHA_NAMES]).to_numpy().astype(np.float32)
    label = frame["label"].to_numpy().astype(np.float32)
    prediction = frame["prediction"].to_numpy()
    if (
        not np.isfinite(x).all()
        or not np.isfinite(label).all()
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("Alpha processed features, labels and predictions must be finite")
    train = np.array([splits["train"][0] <= d <= splits["train"][1] for d, _ in keys])
    # Reproduce the declared processing from producer raw features; feature equations remain producer-reported.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for day in sorted(set(frame["date"])):
            mask = np.array([d == day for d, _ in keys])
            group = clean[mask]
            mean = np.nanmean(np.where(np.isinf(group), np.nan, group), axis=0)
            clean[mask] = np.where(np.isinf(group), mean, group)
        training = np.asfortranarray(clean[train])
        mean = np.nanmean(training, axis=0)
        scale = np.nanstd(training, axis=0)
        constant = scale == 0
        mean[constant] = 0
        scale[constant] = 1
        normalized = (clean - mean) / scale
    normalized = np.where(np.isnan(normalized), 0, normalized)
    if not np.allclose(x, normalized, rtol=2e-5, atol=2e-5):
        raise ValueError("Alpha processed features do not match training-only normalization")
    dates = sorted({d for _, d in latest})
    instruments = manifest["instruments"]
    # Qlib expressions consume the float32 dump. Labels are then CSZScoreNorm(ddof=1).
    expected_labels = []
    for day in sorted(set(frame["date"])):
        n = dates.index(day)
        values = np.array(
            [
                np.float32(latest[i, dates[n + 2]]["close"])
                / np.float32(latest[i, dates[n + 1]]["close"])
                - np.float32(1)
                for i in sorted(instruments)
            ],
            dtype=np.float32,
        )
        values = (values - values.mean()) / values.std(ddof=1)
        expected_labels.extend(values)
    if not np.allclose(label, expected_labels, rtol=2e-4, atol=2e-4):
        raise ValueError("Alpha validation labels do not match source horizon/cross-section")
    valid_days = [d for d in dates if splits["valid"][0] <= d <= splits["valid"][1]][:-2]
    selected = np.array([j for j, (d, _) in enumerate(keys) if d in valid_days][:512], dtype=int)
    explanation = report["explanation"]
    if not isinstance(explanation, dict):
        raise ValueError("Alpha explanation must be an object")
    if not len(selected):
        if explanation != {
            "method": "unavailable",
            "reason": "validation_label_horizon_unavailable",
        }:
            raise ValueError("Alpha unavailable explanation mismatch")
    else:
        fields = {
            "method",
            "samples",
            "baselineMse",
            "labelLastSession",
            "importance",
            "interpretation",
        }
        mse = float(np.mean((prediction[selected] - label[selected]) ** 2))
        if (
            set(explanation) != fields
            or explanation["method"] != "validation_permutation_mse"
            or type(explanation["samples"]) is not int
            or explanation["samples"] != len(selected)
            or explanation["labelLastSession"] != dates[dates.index(keys[selected[-1]][0]) + 2]
            or not _number(explanation["baselineMse"])
            or explanation["baselineMse"] < 0
            or not np.isclose(explanation["baselineMse"], mse, rtol=1e-10, atol=1e-12)
            or explanation["interpretation"]
            != "validation_sensitivity_not_causal_attribution_or_test_selection"
            or not isinstance(explanation["importance"], list)
            or len(explanation["importance"]) != 157
        ):
            raise ValueError("Alpha validation explanation mismatch")
        for name, row in zip(ALPHA_NAMES, explanation["importance"], strict=True):
            if (
                set(row) != {"feature", "mseIncrease"}
                or row["feature"] != name
                or not _number(row["mseIncrease"])
            ):
                raise ValueError("Alpha permutation feature mismatch")
    shap = report["shap"]
    if not isinstance(shap, dict):
        raise ValueError("Alpha SHAP summary must be an object")
    if not manifest["settings"].get("explainShap", False):
        if shap != {"status": "not_requested"}:
            raise ValueError("Alpha SHAP request binding mismatch")
    elif not len(selected):
        if shap != {"status": "unavailable", "reason": "validation_label_horizon_unavailable"}:
            raise ValueError("Alpha SHAP unavailable mismatch")
    else:
        fields = {
            "status",
            "method",
            "samples",
            "meanBias",
            "meanPrediction",
            "maxAdditivityError",
            "importance",
        }
        if (
            set(shap) != fields
            or shap["status"] != "available"
            or shap["method"] != "lightgbm_pred_contrib_tree_shap"
            or type(shap["samples"]) is not int
            or shap["samples"] != len(selected)
            or any(
                not _number(shap[k]) for k in ["meanBias", "meanPrediction", "maxAdditivityError"]
            )
            or not 0 <= shap["maxAdditivityError"] <= 1e-8
            or not np.isclose(
                shap["meanPrediction"], prediction[selected].mean(), rtol=1e-10, atol=1e-12
            )
            or not isinstance(shap["importance"], list)
            or len(shap["importance"]) != 157
        ):
            raise ValueError("Alpha SHAP summary mismatch")
        means = []
        for name, row in zip(ALPHA_NAMES, shap["importance"], strict=True):
            if (
                set(row) != {"feature", "meanContribution", "meanAbsContribution"}
                or row["feature"] != name
                or not _number(row["meanContribution"])
                or not _number(row["meanAbsContribution"])
                or row["meanAbsContribution"] < 0
                or row["meanAbsContribution"] + 1e-10 < abs(row["meanContribution"])
            ):
                raise ValueError("Alpha SHAP feature mismatch")
            means.append(row["meanContribution"])
        if (
            abs(sum(means) + shap["meanBias"] - shap["meanPrediction"])
            > shap["maxAdditivityError"] + 1e-10
        ):
            raise ValueError("Alpha SHAP additive summary mismatch")
    raw_frame = frame.select(
        "date", "instrument", *[pl.col("raw:" + n).alias(n) for n in ALPHA_NAMES]
    )
    return {
        "status": "available",
        "explanation": explanation,
        "shap": shap,
        "inputDrift": panel_diagnostics(raw_frame, splits, ALPHA_NAMES),
        "preprocessingCheck": "training_only_recomputed_from_bound_raw_features",
        "labelCheck": "source_close_horizon_and_cross_section_recomputed",
        "reproduction": "producer_feature_values_and_model_explanations_not_independently_replayed",
    }
