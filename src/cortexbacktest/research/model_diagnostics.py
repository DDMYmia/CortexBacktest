"Model-input diagnostics from owned bars; no model loading or trading verdict."
from __future__ import annotations

from math import isfinite, log
import polars as pl

FEATURES = ("return_1", "return_5", "return_20", "volatility_20", "volume_ratio_20")
MIN_SAMPLES = 20
MIN_RANK_INSTRUMENTS = 50


def feature_panel(latest: dict) -> pl.DataFrame:
    frame = pl.DataFrame([{"date": day, "instrument": instrument,
                          "close": row["close"], "volume": row["volume"]}
                         for (instrument, day), row in latest.items()]).sort("instrument", "date")
    frame = frame.with_columns(*[(pl.col("close") / pl.col("close").shift(n) - 1)
        .over("instrument").alias(f"return_{n}") for n in (1, 5, 20)])
    return frame.with_columns(
        pl.col("return_1").rolling_std(20, min_samples=20, ddof=1)
            .over("instrument").alias("volatility_20"),
        (pl.col("volume") / pl.col("volume").rolling_mean(20, min_samples=20)
            .over("instrument") - 1).alias("volume_ratio_20"),
    ).select("date", "instrument", *FEATURES)


def population_stability(train: list, recent: list) -> dict:
    a = [float(v) for v in train if v is not None and isfinite(v)]
    b = [float(v) for v in recent if v is not None and isfinite(v)]
    result = {"trainSamples": len(a), "comparisonSamples": len(b), "psi": None,
              "reason": None, "edges": []}
    if min(len(a), len(b)) < MIN_SAMPLES:
        return {**result, "reason": "insufficient_finite_samples"}
    if min(a) == max(a):
        center = a[0]
        def category(value):
            return 0 if value < center else (1 if value == center else 2)
        counts_a = [sum(category(v) == n for v in a) for n in range(3)]
        counts_b = [sum(category(v) == n for v in b) for n in range(3)]
        result["edges"] = [center]
        result["binPolicy"] = "constant_below_equal_above"
    else:
        series = pl.Series(a)
        quantiles = [series.quantile(n / 10, interpolation="linear") for n in range(1, 10)]
        edges = sorted({float(value) for value in quantiles if value is not None})
        def counts(values):
            bins = [0] * (len(edges) + 1)
            for value in values:
                bins[sum(value > edge for edge in edges)] += 1
            return bins
        counts_a, counts_b = counts(a), counts(b)
        result["edges"] = edges
        result["binPolicy"] = "training_deciles_open_tails"
    pa = [max(n / len(a), 1e-6) for n in counts_a]
    pb = [max(n / len(b), 1e-6) for n in counts_b]
    result["psi"] = sum((q - p) * log(q / p) for p, q in zip(pa, pb, strict=True))
    return result


def rank_displacement(train: pl.DataFrame, recent: pl.DataFrame, feature: str) -> dict:
    def means(frame):
        clean = frame.filter(pl.col(feature).is_finite())
        return (clean.with_columns((pl.col(feature).rank("average").over("date")
            / pl.len().over("date")).alias("rank"))
            .group_by("instrument").agg(pl.col("rank").mean(), pl.len().alias("observations"))
            .filter(pl.col("observations") >= 5).select("instrument", "rank"))
    common = means(train).join(means(recent), on="instrument", suffix="_recent")
    if common.height < MIN_RANK_INSTRUMENTS:
        return {"value": None, "reason": "insufficient_common_instruments",
                "commonInstruments": common.height, "minimumInstruments": MIN_RANK_INSTRUMENTS}
    return {"value": common.select((pl.col("rank") - pl.col("rank_recent")).abs().mean()).item(),
            "reason": None, "commonInstruments": common.height,
            "minimumInstruments": MIN_RANK_INSTRUMENTS}


def panel_diagnostics(frame: pl.DataFrame, splits: dict, features=FEATURES) -> dict:
    train = frame.filter(pl.col("date").is_between(pl.lit(splits["train"][0]), pl.lit(splits["train"][1])))
    valid = frame.filter(pl.col("date").is_between(pl.lit(splits["valid"][0]), pl.lit(splits["valid"][1])))
    rows = []
    for feature in features:
        rows.append({"feature": feature,
            "populationStability": population_stability(train[feature].to_list(), valid[feature].to_list()),
            "rankDisplacement": rank_displacement(train, valid, feature),
            "trainMissingOrNonfinite": train.filter(~pl.col(feature).is_finite().fill_null(False)).height,
            "validationMissingOrNonfinite": valid.filter(~pl.col(feature).is_finite().fill_null(False)).height})
    return {"baseline": "train", "comparison": "valid", "windows": {k: splits[k] for k in ("train", "valid")},
            "testUsed": False, "features": rows, "minimumSamples": MIN_SAMPLES,
            "interpretation": "input_distribution_diagnostic_not_performance_or_qualification"}


def input_diagnostics(latest: dict, splits: dict) -> dict:
    return panel_diagnostics(feature_panel(latest), splits)
