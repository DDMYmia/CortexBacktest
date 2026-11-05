"""Verify isolated Qlib predictions and evaluate them with the native simulator.

Historical model reconstruction is diagnostic, never received-PIT evidence or a
qualified strategy. The consumer never imports Qlib or deserializes its model.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from math import isfinite
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker, ValidationError
import polars as pl

from .observation_file import instrument_symbol, load_observation_file
from .. import strict_json

_VALIDATOR = Draft202012Validator(
    json.loads(Path(__file__).with_name("qlib-predictions-v1.schema.json").read_bytes()),
    format_checker=FormatChecker(),
)

_MODEL_VALIDATOR = Draft202012Validator(
    json.loads(Path(__file__).with_name("model-predictions-v1.schema.json").read_bytes()),
    format_checker=FormatChecker(),
)


def _read(path, sha256, *, maximum=100_000_000):
    raw = Path(path).read_bytes()
    if len(raw) > maximum or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Qlib artifact digest/size mismatch")
    return raw


def load_qlib_predictions(directory, *, manifest_sha256, bars_path, bars_sha256, producer_sha256):
    """Bind fixed file names, complete test scope, source receipts and label embargo."""
    root = Path(directory).resolve()
    manifest = strict_json.loads(_read(root / "manifest.json", manifest_sha256))
    try:
        validator = (
            _MODEL_VALIDATOR
            if manifest.get("contractVersion") == "model-predictions-v1"
            else _VALIDATOR
        )
        validator.validate(manifest)
    except ValidationError as error:
        raise ValueError("Invalid Qlib predictions manifest") from error
    if manifest["producerSha256"] != producer_sha256 or manifest["inputSha256"] != bars_sha256:
        raise ValueError("Qlib producer/input identity mismatch")
    generated = datetime.fromisoformat(manifest["generatedAt"])
    cutoff = datetime.fromisoformat(manifest["inputAsOf"])
    if (
        generated.tzinfo is None
        or cutoff.tzinfo is None
        or generated < cutoff
        or generated > datetime.now(timezone.utc)
    ):
        raise ValueError("Invalid Qlib artifact clocks")
    source = load_observation_file(bars_path, sha256=bars_sha256, as_of=cutoff)
    if (
        source["recordVersion"] != "market-bars-v4"
        or source["scope"]["source"] != "BINANCE"
        or source["scope"]["adjustment"] != "unadjusted"
    ):
        raise ValueError("Initial Qlib adapter requires unadjusted Binance receipts")
    latest = {}
    for row in source["records"]:
        key = (row["instrumentId"], row["sessionDate"])
        if key not in latest or datetime.fromisoformat(row["recordedAt"]) > datetime.fromisoformat(
            latest[key]["recordedAt"]
        ):
            latest[key] = row
    if not latest or any(row["invalidReason"] is not None for row in latest.values()):
        raise ValueError("Latest Qlib input has invalidated prices")
    instruments = sorted({key[0] for key in latest}, key=int)
    dates = sorted({key[1] for key in latest})
    if (
        manifest["instruments"] != instruments
        or len(instruments) < 2
        or len(latest) != len(instruments) * len(dates)
        or (datetime.fromisoformat(dates[-1]) - datetime.fromisoformat(dates[0])).days + 1
        != len(dates)
    ):
        raise ValueError("Qlib fixed daily universe is incomplete")
    settings = manifest["settings"]
    splits = settings["splits"]
    for start, end in splits.values():
        if start not in dates or end not in dates or start > end:
            raise ValueError("Qlib split outside fixed input")
    if (
        dates.index(splits["train"][0]) < 60
        or dates.index(splits["train"][1]) + 2 >= dates.index(splits["valid"][0])
        or dates.index(splits["valid"][1]) + 2 >= dates.index(splits["test"][0])
    ):
        raise ValueError("Qlib warmup/label embargo violated")
    for name, digest in manifest["files"].items():
        if (root / name).is_symlink():
            raise ValueError("Qlib components must be regular owned files")
        _read(root / name, digest)
    predictions = strict_json.loads(
        _read(root / "predictions.json", manifest["files"]["predictions.json"]))
    required = {
        (i, day)
        for i in instruments
        for day in dates
        if splits["test"][0] <= day <= splits["test"][1]
    }
    field = settings.get("output", "score")
    seen = set()
    totals = {}
    for row in predictions:
        if set(row) != {"sessionDate", "instrumentId", field}:
            raise ValueError("Unexpected Qlib prediction fields")
        key = (row["instrumentId"], row["sessionDate"])
        if (
            key in seen
            or key not in required
            or type(row[field]) not in (float, int)
            or not isfinite(row[field])
        ):
            raise ValueError("Duplicate, out-of-scope or nonfinite Qlib prediction")
        if field == "weight":
            if not 0 <= row[field] <= 1:
                raise ValueError("Model target weight outside long-only bounds")
            totals[key[1]] = totals.get(key[1], 0.0) + row[field]
        seen.add(key)
    if any(abs(value - 1.0) > 1e-6 for value in totals.values()):
        raise ValueError("Model target weights must sum to one")
    if seen != required or manifest["recordCount"] != len(predictions):
        raise ValueError("Incomplete Qlib prediction panel")
    return manifest, predictions, latest


def _model_diagnostics(directory, manifest, latest, manifest_sha256):
    from ..research.model_diagnostics import FEATURES, input_diagnostics

    digest = manifest["files"].get("diagnostics.json")
    if digest is None:
        return {"status": "not_generated_by_legacy_producer"}
    explanation = strict_json.loads(
        _read(Path(directory) / "diagnostics.json", digest))
    settings = manifest["settings"]
    splits = settings["splits"]
    artifact = "model.pkl" if settings["kind"] == "tree_oof_stacking" else "model.zip"
    common = {
        "version",
        "inputSha256",
        "modelSha256",
        "featureNames",
        "window",
        "testUsed",
        "explanationOwner",
        "imputation",
        "seed",
        "method",
    }
    if (
        explanation.get("version") != 1
        or explanation.get("inputSha256") != manifest["inputSha256"]
        or explanation.get("modelSha256") != manifest["files"][artifact]
        or explanation.get("featureNames") != list(FEATURES)
        or explanation.get("window") != splits["valid"]
        or explanation.get("testUsed") is not False
        or explanation.get("seed") != 7
        or explanation.get("explanationOwner")
        != "producer_reported_not_independently_replayed_model"
        or explanation.get("imputation") != "existing_worker_nonfinite_to_zero"
    ):
        raise ValueError("Model explanation binding mismatch")
    days = sorted({day for _, day in latest if splits["valid"][0] <= day <= splits["valid"][1]})
    if settings["kind"] == "rl_portfolio" or len(days) <= 2:
        reason = (
            "supervised_mse_not_applicable_to_rl"
            if settings["kind"] == "rl_portfolio"
            else "validation_label_horizon_unavailable"
        )
        if (
            set(explanation) != common | {"reason"}
            or explanation["method"] != "unavailable"
            or explanation["reason"] != reason
        ):
            raise ValueError("Invalid unavailable model explanation")
    else:
        if (
            set(explanation)
            != common
            | {"samples", "baselineMse", "labelLastSession", "importance", "interpretation"}
            or explanation["method"] != "validation_permutation_mse"
            or type(explanation["samples"]) is not int
            or explanation["samples"] != (len(days) - 2) * len(manifest["instruments"])
            or explanation["labelLastSession"] != days[-1]
            or type(explanation["baselineMse"]) not in (int, float)
            or not isfinite(explanation["baselineMse"])
            or explanation["baselineMse"] < 0
            or explanation["interpretation"]
            != "one_permutation_sensitivity_not_causal_attribution_or_selection"
            or not isinstance(explanation["importance"], list)
            or len(explanation["importance"]) != len(FEATURES)
        ):
            raise ValueError("Invalid validation model explanation")
        for feature, row in zip(FEATURES, explanation["importance"], strict=True):
            if (
                set(row) != {"feature", "mseIncrease"}
                or row["feature"] != feature
                or type(row["mseIncrease"]) not in (int, float)
                or not isfinite(row["mseIncrease"])
            ):
                raise ValueError("Invalid permutation importance")
    return {
        "status": "available",
        "inputSha256": manifest["inputSha256"],
        "modelSha256": manifest["files"][artifact],
        "producerSha256": manifest["producerSha256"],
        "manifestSha256": manifest_sha256,
        "explanation": explanation,
        "inputDrift": input_diagnostics(latest, splits),
        "reproduction": "drift_recomputed_from_bars_explanation_producer_reported",
    }


def _alpha_diagnostics(directory, manifest, latest, manifest_sha256):
    from io import BytesIO
    import pyarrow.parquet as pq
    from ..research.alpha_diagnostics import alpha_input_diagnostics

    if "diagnostics.json" not in manifest["files"]:
        return {"status": "not_generated_by_legacy_producer"}
    report = strict_json.loads(
        _read(Path(directory) / "diagnostics.json", manifest["files"]["diagnostics.json"]))
    raw = _read(Path(directory) / "features.parquet", manifest["files"]["features.parquet"])
    metadata = pq.ParquetFile(BytesIO(raw)).metadata
    splits = manifest["settings"]["splits"]
    expected = sum(
        any(a <= d <= b for a, b in [splits["train"], splits["valid"]]) for _, d in latest
    )
    if metadata.num_rows != expected or metadata.num_columns != 318:
        raise ValueError("Alpha feature snapshot size mismatch")
    frame = pl.read_parquet(BytesIO(raw))
    result = alpha_input_diagnostics(frame, report, manifest, latest)
    return {
        **result,
        "inputSha256": manifest["inputSha256"],
        "modelSha256": manifest["files"]["model.txt"],
        "featureSnapshotSha256": manifest["files"]["features.parquet"],
        "producerSha256": manifest["producerSha256"],
        "manifestSha256": manifest_sha256,
        "featureNames": report["featureNames"],
    }


def _allocation_prices(latest):
    return (
        pl.DataFrame(
            [
                {
                    "date": datetime.fromisoformat(r["sessionClose"]),
                    "symbol": instrument_symbol(r["instrumentId"]),
                    "close": r["close"],
                }
                for r in latest.values()
            ]
        )
        .pivot(on="symbol", index="date", values="close")
        .sort("date")
    )


def _constrained_targets(
    targets, latest, window, *, method, max_weight, max_turnover, risk_aversion
):
    "Allocate frozen assets; budget target-to-target changes after initial funding."
    from ..research.optimization import optimize_weights

    prices = _allocation_prices(latest)
    symbols = sorted(c for c in targets.columns if c != "date")
    previous = None
    rows, turnover = [], []
    for stamp in targets["date"]:
        history = prices.filter(pl.col("date") <= stamp).tail(window + 1)
        if history.height != window + 1:
            raise ValueError("Constrained allocation requires a complete trailing window")
        returns = history.select("date", *[pl.col(s).pct_change().alias(s) for s in symbols]).slice(
            1
        )
        expected = {s: float(returns[s].mean()) for s in symbols} if method == "mvo" else None
        weights = optimize_weights(
            returns,
            previous=previous,
            expected_returns=expected,
            risk_aversion=1.0 if risk_aversion is None else risk_aversion,
            max_weight=max_weight,
            max_turnover=None if previous is None else max_turnover,
        )
        change = sum(abs(weights[s] - (0.0 if previous is None else previous[s])) for s in symbols)
        turnover.append(
            {"date": stamp.isoformat(), "l1": change, "budgetApplied": previous is not None}
        )
        rows.append({"date": stamp, **weights})
        previous = weights
    return pl.DataFrame(rows).select("date", *symbols), turnover


def _hrp_targets(targets, latest, window):
    "Use only trailing closes through each decision; allocation failure rejects the run."
    from ..research.optimization import hierarchical_risk_parity

    prices = _allocation_prices(latest)
    symbols = sorted(c for c in targets.columns if c != "date")
    rows = []
    for row in targets.iter_rows(named=True):
        selected = [s for s in symbols if row[s] > 0]
        history = prices.filter(pl.col("date") <= row["date"]).tail(window + 1)
        if history.height != window + 1 or not selected:
            raise ValueError("HRP requires a complete trailing window and selected instruments")
        returns = history.select(
            "date", *[pl.col(s).pct_change().alias(s) for s in selected]
        ).slice(1)
        weights = hierarchical_risk_parity(returns)
        rows.append({"date": row["date"], **{s: weights.get(s, 0.0) for s in symbols}})
    return pl.DataFrame(rows).select("date", *symbols)


def _holding_targets(targets, holding):
    "Reuse the native state machine with binary membership and fixed entry/exit thresholds."
    from ..research.holding import HoldingConfig, apply_holding_recipe

    fields = {
        "enter_periods",
        "exit_periods",
        "min_active_periods",
        "cooldown_periods",
        "holding_periods",
        "formation_interval",
    }
    if (
        not isinstance(holding, dict)
        or set(holding) != fields
        or any(
            type(v) is not int or not (0 if k == "cooldown_periods" else 1) <= v <= 252
            for k, v in holding.items()
        )
    ):
        raise ValueError("Explicit bounded holding decision-grid counters required")
    config = HoldingConfig(enter_threshold=1.0, exit_threshold=0.0, **holding)
    scores = targets.select(
        "date", *[(pl.col(s) > 0).cast(pl.Float64).alias(s) for s in targets.columns if s != "date"]
    )
    return apply_holding_recipe(targets, scores, config=config)


def _event_study_report(latest, config, cutoff):
    "Bound caller-declared event clocks to the same frozen price panel; no received-news claim."
    from ..research.event_study import event_study

    fields = {
        "benchmark_instrument_id",
        "events",
        "event_window",
        "estimation_window",
        "estimation_gap",
        "model",
    }
    if not isinstance(config, dict) or set(config) not in (fields, fields | {"news_source"}):
        raise ValueError("Explicit complete event study settings required")
    window = config["event_window"]
    if (
        not isinstance(window, (list, tuple))
        or len(window) != 2
        or any(type(x) is not int or not -20 <= x <= 20 for x in window)
        or window[0] > window[1]
        or type(config["estimation_window"]) is not int
        or not 30 <= config["estimation_window"] <= 500
        or type(config["estimation_gap"]) is not int
        or not 0 <= config["estimation_gap"] <= 60
        or config["model"] not in {"market", "market_adjusted", "mean_adjusted"}
        or not isinstance(config["events"], list)
        or len(config["events"]) > 50
    ):
        raise ValueError("Invalid bounded event study settings")
    instruments = {int(i) for i, _ in latest}
    benchmark = config["benchmark_instrument_id"]
    if type(benchmark) is not int or benchmark not in instruments:
        raise ValueError("Event benchmark must belong to the frozen universe")

    events, seen = [], set()
    for row in config["events"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"event_id", "instrument_id", "event_at", "available_at"}
            or not isinstance(row["event_id"], str)
            or not 1 <= len(row["event_id"]) <= 512
            or any(c.isspace() for c in row["event_id"])
            or row["event_id"] in seen
            or type(row["instrument_id"]) is not int
            or row["instrument_id"] not in instruments
            or row["instrument_id"] == benchmark
        ):
            raise ValueError("Unique event identity and distinct frozen target/benchmark required")
        seen.add(row["event_id"])
        events.append(
            {
                "event_id": row["event_id"],
                "symbol": instrument_symbol(str(row["instrument_id"])),
                "event_at": None if row["event_at"] is None else strict_json.clock(row["event_at"]),
                "available_at": strict_json.clock(row["available_at"]),
            }
        )
    news_metadata = None
    identity_config = dict(config)
    if "news_source" in config:
        from .news_events import load_news_replay_events

        source = config["news_source"]
        if not isinstance(source, dict) or set(source) != {"artifact_sha256", "event_ids", "news_as_of", "path"}:
            raise ValueError("Complete frozen news reference required")
        ids = source["event_ids"]
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 50
                or any(not isinstance(value, str) for value in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("One to fifty unique news event IDs required")
        received, news_metadata = load_news_replay_events(source["path"], sha256=source["artifact_sha256"])
        if not news_metadata["receivedRevisionsVerified"] or strict_json.clock(news_metadata["asOf"]) != strict_json.clock(source["news_as_of"]):
            raise ValueError("Received news cutoff binding required")
        if strict_json.clock(news_metadata["asOf"]) > cutoff:
            raise ValueError("News evidence cutoff exceeds frozen price input cutoff")
        indexed = {row["event_id"]: row for row in received.to_dicts()}
        if not set(ids).issubset(indexed):
            raise ValueError("Selected event missing from frozen news")
        selected = [indexed[event_id] for event_id in ids]
        if events != selected:
            raise ValueError("Requested events differ from frozen received news")
        identity_config["news_source"] = {key: value for key, value in source.items() if key != "path"}
    prices = _allocation_prices(latest)
    symbols = [s for s in prices.columns if s != "date"]
    returns = prices.select("date", *[pl.col(s).pct_change().alias(s) for s in symbols]).slice(1)
    market = returns.select("date", pl.col(instrument_symbol(str(benchmark))).alias("market"))
    frame = pl.DataFrame(
        events,
        schema={
            "event_id": pl.String,
            "symbol": pl.String,
            "event_at": pl.Datetime("us", "UTC"),
            "available_at": pl.Datetime("us", "UTC"),
        },
    )
    result = event_study(
        returns,
        market,
        frame,
        as_of=min(cutoff, prices["date"][-1]),
        event_window=tuple(window),
        estimation_window=config["estimation_window"],
        estimation_gap=config["estimation_gap"],
        model=config["model"],
    ).to_dict()
    return {
        **result,
        "status": "available" if result["events"] else "unavailable",
        "reason": None if result["events"] else "no_accepted_events",
        "requestedEvents": len(events),
        "benchmarkInstrumentId": benchmark,
        "eventEvidence": "received_news_receipts" if news_metadata else "request_supplied_clocks_not_received_news_receipts",
        "newsSource": news_metadata,
        "priceInputAsOf": cutoff.isoformat(),
        "returnBasis": "frozen_unadjusted_close_returns_not_executable_strategy_returns",
        "inputEventsSha256": hashlib.sha256(
            json.dumps(identity_config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
    }


def evaluate_qlib_predictions(
    directory,
    *,
    manifest_sha256,
    bars_path,
    bars_sha256,
    producer_sha256,
    initial_cash,
    commission_bps,
    slippage_bps,
    cash_rate_bps,
    allocation="equal_weight",
    hrp_window=None,
    allocation_window=None,
    max_weight=None,
    max_turnover=None,
    risk_aversion=None,
    holding=None,
    risk_calibration=None,
    event_study=None,
    group_attribution=None,
    style_risk=None,
    market_risk=None,
    market_quotes=None,
    market_model_known_at=None,
):
    """One declared baseline, native costs, equal-weight buy/hold and cash comparisons."""
    from ..datasets import load_backtest_dataset
    from ..event import RebalanceConfig
    from ..event.financing import CashInterestModel
    from ..pipelines.factor_workflow import FactorWorkflowConfig, evaluate_return_family
    from ..pipelines.research_execution import (
        BrokerExecutionConfig,
        ResearchExecutionPipelineConfig,
        _execute_targets,
    )

    if (
        any(
            type(v) not in (int, float) or not isfinite(v) or v < 0
            for v in (initial_cash, commission_bps, slippage_bps, cash_rate_bps)
        )
        or initial_cash <= 0
    ):
        raise ValueError("Explicit finite capital, costs and cash rate required")
    if (
        allocation not in {"equal_weight", "hrp", "gmv", "mvo"}
        or allocation == "hrp"
        and (type(hrp_window) is not int or not 20 <= hrp_window <= 1000)
        or allocation != "hrp"
        and hrp_window is not None
    ):
        raise ValueError("Explicit supported allocation and HRP lookback required")
    if allocation in {"gmv", "mvo"}:
        if (
            type(allocation_window) is not int
            or not 20 <= allocation_window <= 1000
            or (not isinstance(max_weight, (int, float)) or isinstance(max_weight, bool))
            or not isfinite(max_weight)
            or not 0 < max_weight <= 1
            or (not isinstance(max_turnover, (int, float)) or isinstance(max_turnover, bool))
            or not isfinite(max_turnover)
            or not 0 <= max_turnover <= 2
            or allocation == "gmv"
            and risk_aversion is not None
            or allocation == "mvo"
            and (
                (not isinstance(risk_aversion, (int, float)) or isinstance(risk_aversion, bool))
                or not isfinite(risk_aversion)
                or not 0 < risk_aversion <= 10000
            )
        ):
            raise ValueError(
                "Explicit valid optimizer window, cap, turnover and MVO risk aversion required"
            )
    elif any(v is not None for v in (allocation_window, max_weight, max_turnover, risk_aversion)):
        raise ValueError("Optimizer options require GMV or MVO")
    manifest, predictions, latest = load_qlib_predictions(
        directory,
        manifest_sha256=manifest_sha256,
        bars_path=bars_path,
        bars_sha256=bars_sha256,
        producer_sha256=producer_sha256,
    )
    alpha_diagnostic = (
        _alpha_diagnostics(directory, manifest, latest, manifest_sha256)
        if manifest["contractVersion"] == "qlib-predictions-v1"
        else None
    )
    start, end = manifest["settings"]["splits"]["test"]
    bars = pl.DataFrame(
        [
            {
                "date": datetime.fromisoformat(r["sessionClose"]),
                "symbol": instrument_symbol(r["instrumentId"]),
                **{f: r[f] for f in ["open", "high", "low", "close", "volume"]},
                "adj_close": r["adjClose"],
            }
            for r in latest.values()
            if start <= r["sessionDate"] <= end
        ]
    ).sort("date", "symbol")
    signals = pl.DataFrame(
        [
            {
                "date": datetime.fromisoformat(
                    latest[(p["instrumentId"], p["sessionDate"])]["sessionClose"]
                ),
                "symbol": instrument_symbol(p["instrumentId"]),
                "score": p[manifest["settings"].get("output", "score")],
            }
            for p in predictions
        ]
    )
    symbols = sorted(bars["symbol"].unique().to_list())
    # A single predeclared long-only rank baseline; no holdout selection or score flipping.
    count = max(1, len(symbols) // 2)
    targets = (
        signals.sort(["date", "score", "symbol"], descending=[False, True, False])
        .with_columns(pl.int_range(pl.len()).over("date").alias("rank"))
        .with_columns(
            pl.when(pl.col("rank") < count).then(1.0 / count).otherwise(0.0).alias("weight")
        )
        .pivot(on="symbol", index="date", values="weight")
        .sort("date")
        .select("date", *symbols)
    )
    if manifest["settings"].get("output") == "weight":
        targets = (
            signals.rename({"score": "weight"})
            .pivot(on="symbol", index="date", values="weight")
            .sort("date")
            .select("date", *symbols)
        )
    if allocation == "hrp":
        if manifest["settings"].get("output") == "weight":
            raise ValueError("HRP cannot replace already allocated model weights")
        targets = _hrp_targets(targets, latest, hrp_window)
    turnover = None
    if allocation in {"gmv", "mvo"}:
        if manifest["settings"].get("output") == "weight":
            raise ValueError(
                "Constrained allocation cannot replace already allocated model weights"
            )
        targets, turnover = _constrained_targets(
            targets,
            latest,
            allocation_window,
            method=allocation,
            max_weight=max_weight,
            max_turnover=max_turnover,
            risk_aversion=risk_aversion,
        )
    holding_report = None
    if holding is not None:
        if allocation in {"gmv", "mvo"} or manifest["settings"].get("output") == "weight":
            raise ValueError(
                "Holding cannot override allocated RL weights or constrained turnover budgets"
            )
        before_holding = hashlib.sha256(targets.write_json().encode()).hexdigest()
        targets, checkpoint = _holding_targets(targets, holding)
        holding_report = {
            "config": holding,
            "scoreBasis": "positive_pre_holding_target_membership_1_else_0",
            "enterThreshold": 1.0,
            "exitThreshold": 0.0,
            "decisionGrid": "explicit_complete_test_sessions",
            "cohortCapital": "fixed_equal_slots_unallocated_cash_not_redistributed",
            "exitPolicy": "state_exit_clips_all_cohorts_expiry_is_independent_of_minimum_active_state",
            "checkpoint": checkpoint,
            "decisions": targets.height,
            "inputTargetsSha256": before_holding,
            "outputTargetsSha256": hashlib.sha256(targets.write_json().encode()).hexdigest(),
            "executionBoundary": "target_controls_not_guaranteed_fills_or_position_liquidation",
        }
    factors = signals.select("date", "symbol", pl.col("score").alias("factor"))
    dataset = load_backtest_dataset(bars=bars, factors=factors)
    opening = {stamp: stamp - timedelta(days=1) for stamp in bars["date"].unique()}
    # Reserve the declared fee/slippage budget equally, with a numerical rounding margin.
    cash_buffer = 1 - (1 - 1e-9) / ((1 + commission_bps / 10000) * (1 + slippage_bps / 10000))
    cfg = ResearchExecutionPipelineConfig(
        rebalance=RebalanceConfig(allow_fractional=True, cash_buffer=cash_buffer),
        broker=BrokerExecutionConfig(
            initial_cash=initial_cash,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            allow_short=False,
            opening_times=opening,
            cash_interest_model=CashInterestModel(credit_rate=cash_rate_bps / 10000),
        ),
    )
    before_start = min(opening.values()) - timedelta(seconds=1)
    equal = pl.DataFrame({"date": [before_start], **{s: [1.0 / len(symbols)] for s in symbols}})
    cash = pl.DataFrame({"date": [before_start], **{s: [0.0] for s in symbols}})
    feed = dataset.to_event_feed()
    runs = {
        name: _execute_targets(
            weights,
            feed,
            dataset=dataset,
            cfg=cfg,
            decision_times=True,
            run_metadata={
                "scope": "qlib_historical_diagnostic",
                "inputSha256": bars_sha256,
                "predictionsSha256": manifest_sha256,
                "qualificationIssued": False,
            },
        )
        for name, weights in {"qlib": targets, "equal_weight_buy_hold": equal, "cash": cash}.items()
    }
    if set(runs["equal_weight_buy_hold"].records.fills()["symbol"]) != set(symbols):
        raise ValueError("Equal-weight benchmark did not fill every frozen instrument")
    curves = {
        name: run.performance.equity_curve().select("timestamp", "equity")
        for name, run in runs.items()
    }
    returns = {
        name: curve.with_columns(pl.col("equity").pct_change().fill_null(0.0).alias(name)).select(
            "timestamp", name
        )
        for name, curve in curves.items()
    }
    aligned = returns["qlib"].join(returns["equal_weight_buy_hold"], on="timestamp")
    differences = aligned.select(
        pl.col("timestamp").alias("date"),
        (pl.col("qlib") - pl.col("equal_weight_buy_hold")).alias("qlib_vs_equal_weight"),
    )
    family = evaluate_return_family(
        differences, config=FactorWorkflowConfig(trials=1, periods_per_year=365)
    )
    report = {
        "scope": "historical_reconstruction_not_received_pit",
        "manifestSha256": manifest_sha256,
        "inputSha256": bars_sha256,
        "test": [start, end],
        "family": family,
        "execution": {
            "initialCash": initial_cash,
            "commissionBps": commission_bps,
            "slippageBps": slippage_bps,
            "cashAnnualRateBps": cash_rate_bps,
            "cashBuffer": cash_buffer,
            "quantityPolicy": "continuous_fractional_not_exchange_filter_certification",
            "decisionPolicy": "close_scores_strictly_before_execution_open",
        },
        "qualificationIssued": False,
        "executionAuthorized": False,
        "portfolioPolicy": (
            "model_long_only_target_weights_daily"
            if manifest["settings"].get("output") == "weight"
            else "long_only_top_half_equal_weight_daily"
        ),
        "trialFamilyScope": "one_declared_baseline_not_certification_of_all_prior_exploratory_trials",
        "costEvidence": "configured_fee_and_slippage_only_spread_impact_unfilled_uncertified",
    }
    if allocation == "hrp":
        serialized = targets.write_json()
        report["portfolioPolicy"] = "long_only_top_half_hrp_daily"
        report["allocation"] = {
            "method": "hrp",
            "lookbackPeriods": hrp_window,
            "priceWindow": "through_decision_only",
            "onFailure": "reject",
            "targetWeightsSha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "targetWeights": json.loads(serialized),
        }
    if allocation in {"gmv", "mvo"}:
        serialized = targets.write_json()
        report["portfolioPolicy"] = f"long_only_frozen_universe_{allocation}_daily"
        report["allocation"] = {
            "method": allocation,
            "lookbackPeriods": allocation_window,
            "universe": "all_frozen_instruments_not_score_ranked",
            "priceWindow": "through_decision_only",
            "expectedReturns": "trailing_arithmetic_mean_daily"
            if allocation == "mvo"
            else "not_used",
            "maxWeight": max_weight,
            "maxTurnover": max_turnover,
            "riskAversion": risk_aversion,
            "turnoverBasis": "l1_target_to_previous_target_cash_excluded_not_realized_fills",
            "initialFunding": "from_cash_exempt_from_rebalance_budget",
            "onFailure": "reject",
            "turnover": turnover,
            "targetWeightsSha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "targetWeights": json.loads(serialized),
        }
    if risk_calibration is not None:
        from ..reporting.risk_calibration import historical_var_calibration

        if not isinstance(risk_calibration, dict) or set(risk_calibration) != {
            "window",
            "confidence",
            "significance",
        }:
            raise ValueError("Explicit risk calibration window/confidence/significance required")
        report["riskCalibration"] = {
            "config": risk_calibration,
            "forecastMethod": "rolling_empirical_positive_loss_linear_quantile",
            "returnBasis": "native_equity_intervals_including_configured_costs_and_cash_interest",
            "timing": "forecast_uses_only_completed_returns_at_or_before_period_start",
            "warmup": "within_test_equity_only_no_zero_padding_or_pretest_strategy_history",
            "interpretation": "passive_asymptotic_diagnostic_not_risk_or_trading_approval",
            "portfolios": {
                name: historical_var_calibration(curve, **risk_calibration)
                for name, curve in curves.items()
            },
        }
    if event_study is not None:
        report["eventStudy"] = _event_study_report(
            latest, event_study, datetime.fromisoformat(manifest["inputAsOf"])
        )
    if group_attribution is not None:
        from ..reporting.native_attribution import native_group_attribution

        report["groupAttribution"] = native_group_attribution(runs, latest, group_attribution)
    if market_risk is not None:
        from ..reporting.native_market_risk import native_market_risk

        report["marketRisk"] = native_market_risk(
            runs,
            latest,
            market_risk,
            as_of=datetime.fromisoformat(manifest["inputAsOf"]),
            input_sha256=bars_sha256,
            quotes=market_quotes,
            model_known_at=market_model_known_at,
        )
    elif market_quotes is not None:
        raise ValueError("Quote evidence requires native market risk")
    if style_risk is not None:
        from ..reporting.native_style_risk import native_style_risk

        report["styleRisk"] = native_style_risk(runs, latest, style_risk)
    if holding_report is not None:
        report["holding"] = holding_report
    if alpha_diagnostic is not None:
        report["modelDiagnostics"] = alpha_diagnostic
    if manifest["contractVersion"] == "model-predictions-v1":
        report["modelDiagnostics"] = _model_diagnostics(
            directory, manifest, latest, manifest_sha256
        )
        report["experimentKind"] = manifest["settings"]["kind"]
        report["trainingArtifactEvaluation"] = (
            "training rewards excluded; native Cortex fills and costs only"
        )
    return report, runs
