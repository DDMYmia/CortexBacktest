"""Observation receipt -> native mining -> frozen outer evaluation."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field, replace
from datetime import datetime
from hashlib import sha256
import json

import polars as pl

from ..contracts import CorporateAction, CorporateActionType
from ..reporting import research_report
from ..datasets import load_backtest_dataset
from ..event import RebalanceConfig
from ..research import MiningFactorSpec, RevisionPanel, WalkForwardSplit, auto_mine_factors, evaluate_holdout
from .factor_workflow import FactorWorkflowConfig, evaluate_return_family
from .mining_execution import run_frozen_factor_simulation
from .research_execution import BrokerExecutionConfig, ResearchExecutionPipelineConfig, ResearchExecutionPipelineResult
from .mining_store import load_mining_result, save_mining_result


def _write_action_events(event, directory):
    """Keep the applied account ledger, including unpaid dividend entitlements."""
    records = [{"timestamp": r.timestamp.isoformat(), "symbol": r.symbol,
                "action_type": r.action_type.value, "quantity_before": r.quantity_before,
                "quantity_after": r.quantity_after, "cash_amount": r.cash_amount,
                "tax_amount": r.tax_amount,
                "settles_on": r.settles_on.isoformat() if r.settles_on else None}
               for r in event.corporate_action_events or []]
    (directory / "corporate_action_events.ndjson").write_text(
        "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records))


@dataclass(frozen=True)
class _EvaluationPanel(RevisionPanel):
    """Share causal features only within this job's frozen outer evaluation.

    Training slices remain ordinary panels. Research labels and event fills
    keep their own clocks; only the identical input features are reused.
    """
    _features: dict[tuple[MiningFactorSpec, bool], pl.DataFrame] = field(default_factory=dict, init=False, repr=False, compare=False)

    def factor(self, spec: MiningFactorSpec, *, last_only: bool = False) -> pl.DataFrame:
        key = (spec, last_only)
        if key not in self._features:
            self._features[key] = super().factor(spec, last_only=last_only)
        return self._features[key].clone()


def load_membership(exported: dict) -> tuple[pl.DataFrame, dict] | None:
    """The point-in-time membership a universe study declares, verified against its digest.

    The producer writes ``membership.json`` beside the bars: one row per stretch an instrument
    was in the index, bounded by session closes. ``None`` when the study names fixed
    instruments instead.
    """
    universe = exported["settings"]["mining"].get("universe")
    descriptor = exported.get("membership")
    if universe is None and descriptor is None:
        return None
    if universe is None or descriptor is None:
        raise ValueError("A universe study requires its membership file, and only a universe study has one")
    content = Path(descriptor["path"]).read_bytes()
    if sha256(content).hexdigest() != descriptor["sha256"]:
        raise ValueError("Membership file digest mismatch")
    document = json.loads(content)
    if document.get("contractVersion") != "index-membership-v1" or document.get("universe") != universe:
        raise ValueError("Membership file does not match the requested universe")
    frame = pl.DataFrame(document["records"], schema={"instrumentId": pl.String, "start": pl.String, "end": pl.String})
    frame = frame.select((pl.lit("INSTRUMENT:") + pl.col("instrumentId")).alias("symbol"),
                         *(pl.col(c).str.to_datetime(time_zone="UTC") for c in ("start", "end")))
    return frame, {"universe": universe, "index": document["index"], "sha256": descriptor["sha256"],
                   "stretches": frame.height, "instruments": frame["symbol"].n_unique()}


def load_corporate_actions(exported: dict) -> tuple[list[CorporateAction], str] | None:
    """The splits and cash dividends an equity study declares, verified against their digest.

    The producer writes ``corporate-actions.json`` beside the bars (ex-dates on each instrument's own
    sessions). The order-level simulation applies them to raw prices; ``None`` when absent.
    """
    descriptor = exported.get("corporateActions")
    if descriptor is None:
        return None
    content = Path(descriptor["path"]).read_bytes()
    if sha256(content).hexdigest() != descriptor["sha256"]:
        raise ValueError("Corporate-action file digest mismatch")
    document = json.loads(content)
    if document.get("contractVersion") != "corporate-actions-v1":
        raise ValueError("Unsupported corporate-action file")
    actions = []
    for row in document["records"]:
        ex_date = datetime.fromisoformat(row["exDate"]).date()
        symbol = f"INSTRUMENT:{row['instrumentId']}"
        if row["actionType"] == "split":
            actions.append(CorporateAction(symbol, ex_date, CorporateActionType.SPLIT, ratio=float(row["ratio"])))
        elif row["actionType"] == "cash_dividend":
            pay = datetime.fromisoformat(row["payDate"]).date() if row.get("payDate") else None
            actions.append(CorporateAction(symbol, ex_date, CorporateActionType.CASH_DIVIDEND, ratio=1.0,
                                           amount=float(row["amount"]), pay_date=pay))
        else:
            raise ValueError("Unsupported corporate action type")
    return actions, descriptor["sha256"]


def mining_revision_panels(bars: dict, exported: dict, *, replay_as_of: datetime | None = None):
    """Shared receipt-to-panel conversion for research and frozen-strategy decisions.

    The caller verifies the observation contract and digest before conversion.
    Keep the decision delay and raw opening policy identical to research.
    """
    frame = pl.DataFrame(bars["records"], infer_schema_length=None).with_columns(
        (pl.lit("INSTRUMENT:") + pl.col("instrumentId").cast(pl.String)).alias("symbol"),
        pl.col("sessionClose").str.to_datetime(time_zone="UTC").alias("date"),
        pl.col("availableAt").str.to_datetime(time_zone="UTC").alias("available"),
        pl.col("recordedAt").str.to_datetime(time_zone="UTC").alias("recorded_at"),
        pl.col("sourceKnownAt").str.to_datetime(time_zone="UTC").alias("known_at"),
    )
    reason = ([pl.col("invalidReason").alias("invalid_reason")]
              if "invalidReason" in frame.columns else [])
    settings = exported["settings"]
    spec = settings["mining"]
    if spec.get("benchmark") is not None and (
        spec["benchmark"] != "initial_equal_weight_buy_hold" or spec.get("portfolioMode") != "long_only"
        or spec.get("rebalance") is None
    ):
        raise ValueError("Invalid declared long-only benchmark")
    policy = spec.get("knowledgePolicy", "received")
    delay = spec.get("decisionDelayMinutes", 0)
    if type(delay) is not int or not 0 <= delay <= 1440:
        raise ValueError("Invalid decision delay")
    if delay and (settings.get("calendar") != "crypto_utc" or set(frame["adjustment"]) != {"unadjusted"}
                  or frame.filter(pl.col("adjClose") != pl.col("close")).height):
        raise ValueError("Delayed mining requires unadjusted crypto prices")
    rebalance = spec.get("rebalanceFrequency", "daily")
    if rebalance not in ("daily", "weekly"):
        raise ValueError("Unsupported rebalance frequency")
    clock = spec.get("decisionClock", "session_close")
    if clock not in ("session_close", "after_bar_known"):
        raise ValueError("Unsupported decision clock")
    after_known = clock == "after_bar_known"
    if after_known and (settings.get("calendar") != "us_equity" or delay):
        raise ValueError("The after-bar-known clock is for undelayed US-equity sessions")
    # Keep event-time history and its receipt timestamps separate. A late old
    # observation can warm up a later signal, never an earlier decision.
    decisions = frame.select("date").unique().sort("date").with_columns(
        (pl.col("date") + pl.duration(minutes=delay)).alias("decision_time"))
    if after_known:
        # Decide once the session's bars are knowable: the producer stamps a daily bar known at
        # 21:00 UTC, an hour after the 20:00 UTC close under daylight saving time. The earliest
        # stamp of the session sets its clock, so one late bar cannot hold back the others.
        known = frame.group_by("date").agg(pl.col("known_at").min().alias("known"))
        decisions = decisions.join(known, on="date").with_columns(
            pl.max_horizontal("decision_time", "known").alias("decision_time")).drop("known").sort("date")
    if delay:
        cutoff = datetime.fromisoformat(exported["asOf"])
        if cutoff.tzinfo is None:
            raise ValueError("Export asOf must be timezone-aware")
        decisions = decisions.filter(pl.col("decision_time") <= cutoff)
        frame = frame.filter(pl.col("date").is_in(decisions["date"].implode()))
    membership = load_membership(exported)
    if membership is not None and (delay or spec.get("rebalance") is not None or spec.get("benchmark") is not None):
        raise ValueError("A universe study has no delayed, rebalanced or benchmarked execution yet")
    panel = RevisionPanel(frame.select("date", "symbol",
                                      pl.col("adjClose").cast(pl.Float64).alias("price"), "recorded_at", "known_at", *reason), decisions, policy, replay_as_of,
                          membership[0] if membership is not None else None, rebalance)
    execution_panel = None
    if after_known:
        # A decision after the close trades at the next session's open; labels are measured
        # from there. Every NYSE session opens at 09:30 New York time, half days included.
        # Opens are put on the research close's basis (open x adjClose / close).
        opening = (pl.col("date").dt.convert_time_zone("America/New_York").dt.date()
                   .dt.combine(pl.time(9, 30)).dt.replace_time_zone("America/New_York")
                   .dt.convert_time_zone("UTC"))
        execution_panel = RevisionPanel(frame.select(
            opening.alias("date"), "symbol",
            (pl.col("open") * pl.col("adjClose") / pl.col("close")).cast(pl.Float64).alias("price"),
            "recorded_at", "known_at", *reason),
            decisions.with_columns(opening), policy, replay_as_of)
    if delay:
        execution_panel = RevisionPanel(frame.select(
            (pl.col("date") - pl.duration(days=1)).alias("date"), "symbol",
            pl.col("open").cast(pl.Float64).alias("price"), "recorded_at", "known_at", *reason),
            decisions.with_columns(pl.col("date") - pl.duration(days=1)), policy, replay_as_of)
    return frame, panel, execution_panel, delay


def _return_verdict(returns, trials):
    verdict = evaluate_return_family(returns, config=FactorWorkflowConfig(trials=trials))
    # Candidate identities remain values at the HTTP boundary.
    verdict["corrections"] = [{"candidate": name, **stats} for name, stats in verdict["corrections"].items()]
    return verdict


def run_mining_job(bars: dict, exported: dict, out: Path) -> dict:
    """Caller verifies the observation-file contract/hash before this entry.

    Zero delay retains close-time research. Explicit crypto delays use received
    raw opening versions and the first opening after the decision. Neither
    policy infers historical vintages or index membership.
    """
    frame, panel, execution_panel, delay = mining_revision_panels(bars, exported)
    membership = load_membership(exported)
    ledger = load_corporate_actions(exported)
    equity = exported["settings"].get("calendar") == "us_equity"
    decisions = panel.decisions
    settings = exported["settings"]
    spec = settings["mining"]
    replay = panel.knowledge_policy == "source_publication_replay"
    # Each session's own decision time (close plus any delay, or once its bars are known).
    frame = frame.join(decisions.select("date", "decision_time"), on="date", how="left")
    eligible = pl.col("known_at" if replay else "available") <= pl.col("decision_time")
    usable = pl.col("invalidReason").is_null() if "invalidReason" in frame.columns else pl.lit(True)
    latest = frame.filter(pl.col("known_at" if replay else "available") <= decisions[-1, "decision_time"]).sort("recorded_at").unique(
        subset=["date", "instrumentId"], keep="last")
    sessions = frame.select("date", "instrumentId").unique().height
    study_start = datetime.fromisoformat(spec["studyStart"]) if spec.get("studyStart") else None
    warmup = 0
    if study_start is not None:
        if study_start.utcoffset() is None:
            raise ValueError("Study start must be timezone-aware")
        if not decisions[0, "date"] <= study_start <= decisions[-1, "date"]:
            raise ValueError("Study start must fall within the exported session history")
        warmup = decisions.filter(pl.col("date") < study_start).height
    study_sessions = decisions.height - warmup
    split = warmup + int(study_sessions * settings["inSample"])
    timely = frame.filter(eligible).sort("recorded_at").unique(
        subset=["date", "instrumentId"], keep="last").filter(usable)
    # Price coverage is input evidence, not a factor/label eligibility claim.
    # Preserve empty dates and count assets, never revisions, within each date.
    coverage = decisions.select("date").join(
        timely.group_by("date").len(name="assets"), on="date", how="left"
    ).with_columns(pl.col("assets").fill_null(0))
    assets = frame["instrumentId"].n_unique()
    price_coverage: dict[str, object] = {"requestedAssets": assets}
    for name, dates in (("inner", coverage.slice(warmup, split - warmup)), ("outer", coverage.slice(split))):
        price_coverage[name] = {"dates": dates.height,
                               "datesWithTwoPrices": dates.filter(pl.col("assets") >= 2).height,
                               "datesWithAllPrices": dates.filter(pl.col("assets") == assets).height}
    result = auto_mine_factors(
        panel.head(split), research_direction=spec["direction"], budget=spec["budget"],
        seed=spec["seed"], horizon=spec["horizon"], cost_bps=settings["costBps"],
        validation=WalkForwardSplit(spec["trainSessions"], spec["testSessions"]),
        execution=execution_panel.head(split) if execution_panel is not None else None,
        study_start=study_start, portfolio_mode=spec.get("portfolioMode", "long_short"),
    )
    checkpoint = out / "selection"
    # Timely input and internal selection are research evidence, not formal
    # strategy qualification. No native mining job grants trading eligibility.
    result = replace(result, settings={**result.settings, "research_only": True})
    rebalance = RebalanceConfig.from_document(spec.get("rebalance", {}))
    if "rebalance" in spec:
        result = replace(result, settings={**result.settings, "rebalance": rebalance.to_document()})
    if "benchmark" in spec:
        result = replace(result, settings={**result.settings, "benchmark": spec["benchmark"]})
    if replay:
        result = replace(result, settings={**result.settings, "knowledge_policy": panel.knowledge_policy,
                                          "replay_as_of": exported["asOf"]})
    receipt = save_mining_result(result, checkpoint)
    frozen = load_mining_result(checkpoint, manifest_sha256=receipt)
    panel = _EvaluationPanel(panel.revisions, panel.decisions, panel.knowledge_policy, panel.replay_as_of,
                             panel.membership, panel.rebalance)
    execution = {"status": "no_selected_factors", "results": []}
    return_family = None
    reports = evaluate_holdout(frozen, panel, execution=execution_panel) if frozen.selected else {}
    if frozen.selected:
        # Market observation files carry no corporate-action ledger. Do not silently
        # simulate adjusted equity research without dividends and split events.
        if equity and ledger is None:
            execution["status"] = "requires_corporate_actions"
        elif not equity and (set(frame["adjustment"]) != {"unadjusted"}
                             or frame.filter(pl.col("adjClose") != pl.col("close")).height):
            execution["status"] = "requires_unadjusted_prices"
        elif latest.height != sessions or latest.filter(~usable).height:
            execution["status"] = "requires_complete_execution_prices"
        else:
            raw = latest.select("date", "symbol",
                               "open", "high", "low", "close", "volume")
            # Equities: research ranks on the adjusted close, orders trade the raw prices, and the
            # ledger's splits and dividends (applied at each ex-date's open) bring them to total return.
            dataset = load_backtest_dataset(
                bars=raw.with_columns(latest["adjClose"].cast(pl.Float64).alias("adj_close")) if equity else raw,
                factors=raw.select("date", "symbol").with_columns(pl.lit(0.).alias("factor")),
                corporate_actions=ledger[0] if equity and ledger else None,
            )
            # A crypto day opens where the previous closes; a NYSE session opens at 09:30 New York.
            opening = ((pl.col("date").dt.convert_time_zone("America/New_York").dt.date()
                        .dt.combine(pl.time(9, 30)).dt.replace_time_zone("America/New_York")
                        .dt.convert_time_zone("UTC")) if equity else pl.col("date") - pl.duration(days=1))
            broker = BrokerExecutionConfig(commission_bps=settings["costBps"], opening_times=dict(
                latest.select("date").unique().with_columns(opening.alias("opening")).iter_rows()))
            cfg = ResearchExecutionPipelineConfig(horizons=(frozen.horizon,),
                                                   transaction_cost_bps=frozen.cost_bps, broker=broker, rebalance=rebalance)
            simulation_results = []
            events = run_frozen_factor_simulation(checkpoint, manifest_sha256=receipt,
                dataset=dataset, execution_bars=raw, revision_panel=panel,
                inner_execution=execution_panel.head(split) if execution_panel is not None else None,
                broker=broker, rebalance=rebalance, trial_family=True)
            benchmark_event = events.pop("__benchmark__", None)
            event_items = iter(events.items())
            name, event = next(event_items)
            returns = event.performance.returns().rename({"timestamp": "date", "return": name})
            for name, event in event_items:
                values = event.performance.returns().rename({"timestamp": "date", "return": name})
                if not returns["date"].equals(values["date"]):
                    raise ValueError("Trial execution dates differ")
                returns = returns.with_columns(values[name])
            path = out / "trial-returns.parquet"
            returns.write_parquet(path)
            return_family = {"sha256": sha256(path.read_bytes()).hexdigest(),
                "basis": "absolute_net_execution_returns", "initialCash": broker.initial_cash,
                "candidates": list(events), "verdict": _return_verdict(returns, frozen.trials.height)}
            if benchmark_event is not None:
                benchmark_returns = benchmark_event.performance.returns().rename({"timestamp": "date"})
                if not returns["date"].equals(benchmark_returns["date"]):
                    raise ValueError("Benchmark execution dates differ")
                directory = out / "benchmark"
                benchmark_event.records.write_dir(directory, format="ndjson")
                _write_action_events(benchmark_event, directory)
                path = out / "benchmark-returns.parquet"
                benchmark_returns.write_parquet(path)
                excess = returns.with_columns((pl.col(name) - benchmark_returns["return"]).alias(name)
                                              for name in events)
                return_family["benchmark"] = {"policy": spec["benchmark"],
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                    "artifacts": {file.name: sha256(file.read_bytes()).hexdigest() for file in sorted(directory.iterdir())},
                    "excessVerdict": _return_verdict(excess, frozen.trials.height)}
            for selected_factor in frozen.selected:
                name = selected_factor.name
                event = events[name]
                directory = out / "simulation" / sha256(name.encode()).hexdigest()
                event.records.write_dir(directory, format="ndjson")
                _write_action_events(event, directory)
                simulation_results.append({"candidate": name,
                    "artifacts": {file.name: sha256(file.read_bytes()).hexdigest() for file in sorted(directory.iterdir())},
                    "report": ResearchExecutionPipelineResult(
                        research=reports[name], event=event, target_weights=reports[name].target_weights_frame(),
                        config=cfg).report(include_records=False)})
            execution = {"status": "evaluated", "commissionBps": settings["costBps"], "slippageBps": 0.,
                         "targetPolicy": "decision_time_v1", "fillPolicy": "first_open_after_decision",
                         "initialCash": 100_000.,
                         "results": simulation_results}
    return {
        "kind": "factor_mining", "inputs": {"bars": exported["bars"]["sha256"],
                                            **({"membership": membership[1]["sha256"]} if membership else {}),
                                            **({"corporateActions": ledger[1]} if ledger else {})},
        "settings": settings, "manifestSha256": receipt,
        "status": "evaluated" if reports else "no_selected_factors",
        "pit": {"decisionPolicy": ("session_close_plus_delay" if delay else
                                   "after_bar_known_next_open" if execution_panel is not None else "session_close"),
                **({"knowledgePolicy": panel.knowledge_policy, "formalTradingEligible": False,
                    "versionPolicy": "exported_versions_not_historical_vintages"} if replay else {}),
                "decisionDelayMinutes": delay, "records": sessions, "revisionRecords": frame.height,
                "eligibleRecords": timely.height, "priceCoverage": price_coverage,
                "invalidRevisionRecords": frame.filter(~usable).height,
                **({"studyStart": study_start.isoformat(), "warmupSessions": warmup,
                    "studySessions": study_sessions, "innerSessions": split - warmup,
                    "outerSessions": decisions.height - split} if study_start is not None else {}),
                **({"universePolicy": "point_in_time_index_membership", "membership": membership[1]} if membership
                   else {"universePolicy": "fixed_requested_instruments_not_historical_membership"})},
        "selected": [s.name for s in frozen.selected],
        "execution": execution,
        **({"returnFamily": return_family} if return_family is not None else {}),
        "trials": json.loads(frozen.trials.write_json()),
        # Candidate identity is a value: HTTP casing must never rename formulas.
        "holdout": [{"candidate": name, "report": research_report(report, include_records=False)}
                    for name, report in reports.items()],
    }
