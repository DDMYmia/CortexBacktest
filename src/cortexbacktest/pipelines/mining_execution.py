"""Execute frozen mined factors through the existing event backtest engine."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from dataclasses import replace
from datetime import datetime
from math import isfinite
from pathlib import Path

import polars as pl

if TYPE_CHECKING:
    from .factor_strategy import FrozenFactorStrategy

from ..data import apply_universe_mask
from ..datasets import BacktestDataset
from ..contracts import FillPrice
from ..event import RebalanceConfig, quantity_changes_for_target_weights
from ..event.feed import ColumnarBarFeed
from ..research import MiningResult, RevisionPanel, evaluate_holdout
from .research_execution import (
    BrokerExecutionConfig,
    ResearchExecutionPipelineConfig,
    ResearchExecutionPipelineResult,
    _execute_research,
    _execution_feed,
)


def _frozen_rebalance(selection, supplied):
    frozen = RebalanceConfig.from_document(selection.settings.get("rebalance", {}))
    if "rebalance" in selection.settings and supplied is not None and supplied != frozen:
        raise ValueError("Rebalance differs from frozen research execution policy")
    return supplied if supplied is not None else frozen


def _frozen_execution_feed(dataset, execution_bars, selection, revision_panel=None, opening_times=None):
    feed = _execution_feed(dataset, execution_bars, start_after=selection.inner_end)
    if revision_panel is not None:
        if opening_times is None:
            raise ValueError("Revision execution requires explicit opening times")
        cutoff = revision_panel.through(selection.inner_end).decisions[-1, "decision_time"]
        feed = ColumnarBarFeed.from_bars(
            bar for group in feed for bar in group.bars if opening_times[bar.timestamp] > cutoff)
    return feed


def run_mining_execution_pipeline(
    selection: MiningResult,
    dataset: BacktestDataset,
    *,
    execution_bars: pl.DataFrame,
    availability: pl.DataFrame | None = None,
    revision_panel: RevisionPanel | None = None,
    execution: RevisionPanel | None = None,
    broker: BrokerExecutionConfig | None = None,
    rebalance: RebalanceConfig | None = None,
    explain_execution: bool = False,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, ResearchExecutionPipelineResult]:
    """Backtest every frozen candidate, without reselecting on the outer period.

    Load ``selection`` from its verified checkpoint first. Explicit raw OHLCV
    must cover the full research dataset; only bars after the selection cutoff
    are executed, starting in cash. Dataset corporate actions remain attached.
    Close-derived targets trade at the next session's open by default. The
    matrix cost assumption and broker commission/slippage remain separate.
    With an explicit execution revision panel, reports already use executable
    instants: replay starts after the frozen decision cutoff without another
    signal lag, and raw opening prices must match that evaluation snapshot.
    """
    if execution_bars is None:
        raise ValueError("Frozen factor execution requires explicit raw OHLCV execution_bars")
    rebalance = _frozen_rebalance(selection, rebalance)
    prices = apply_universe_mask(dataset.prices, dataset.universe).sort("date")
    if revision_panel is not None and not prices.equals(revision_panel.prices()):
        raise ValueError("Execution dataset research prices must match the revision evaluation snapshot")
    reports = evaluate_holdout(selection, revision_panel if revision_panel is not None else prices,
                               availability=availability, execution=execution)
    opening_times = None
    if execution is not None:
        assert revision_panel is not None  # evaluate_holdout validates this binding
        opening_times = dict(zip(revision_panel.decisions["date"], execution.decisions["date"], strict=True))
        expected = execution.prices()
        supplied = execution_bars.select(pl.col("date").replace_strict(opening_times), "symbol", "open").pivot(
            on="symbol", index="date", values="open").select(expected.columns).sort("date")
        if not supplied.equals(expected):
            raise ValueError("Raw opens must match the revision execution snapshot")
        if broker is not None and broker.opening_times is not None and broker.opening_times != opening_times:
            raise ValueError("Broker opening times differ from frozen execution schedule")
        broker = replace(broker or BrokerExecutionConfig(), opening_times=opening_times)
    feed = _frozen_execution_feed(dataset, execution_bars, selection,
                                  revision_panel if execution is not None else None, opening_times)
    cfg = ResearchExecutionPipelineConfig(
        horizons=(selection.horizon,), transaction_cost_bps=selection.cost_bps,
        broker=broker or BrokerExecutionConfig(),
        rebalance=rebalance or RebalanceConfig(), explain_execution=explain_execution,
    )
    return {name: _execute_research(report, feed, dataset=dataset, cfg=cfg,
                                   run_metadata=run_metadata, targets_at_open=execution is not None)
            for name, report in reports.items()}


def run_frozen_factor_simulation(
    checkpoint: str | Path,
    *,
    manifest_sha256: str,
    dataset: BacktestDataset,
    revision_panel: RevisionPanel,
    execution_bars: pl.DataFrame,
    inner_execution: RevisionPanel | None = None,
    broker: BrokerExecutionConfig | None = None,
    rebalance: RebalanceConfig | None = None,
    factor: str | None = None,
    strategy: FrozenFactorStrategy | None = None,
    trial_family: bool = False,
):
    """Replay frozen decision targets without requiring future research labels.

    Explicit opening_times must map the close-labelled feed to executable opens.
    Orders use only targets strictly
    older than each opening instant. This uses the existing simulator, not a
    broker connector, and never authorizes live execution.
    """
    from ..research import frozen_factor_targets
    from .mining_store import load_mining_result
    from .research_execution import _execute_targets

    if trial_family and (strategy is not None or factor is not None):
        raise ValueError("Trial diagnostics cannot replace a pinned strategy")
    if strategy is not None:
        strategy.verify_runtime()
        if (strategy.selection_sha256 != manifest_sha256
                or factor is not None and factor != strategy.factor
                or rebalance is not None and rebalance != strategy.rebalance):
            raise ValueError("Simulation parameters conflict with frozen strategy")
        if set(dataset.prices.columns) - {"date"} != set(strategy.symbols):
            raise ValueError("Simulation outside frozen strategy scope")
        factor, rebalance = strategy.factor, strategy.rebalance
    selection = load_mining_result(checkpoint, manifest_sha256=manifest_sha256)
    rebalance = _frozen_rebalance(selection, rebalance)
    selected_names = {spec.name for spec in selection.selected}
    if trial_family:
        from ..research import MiningFactorSpec

        specs = []
        for name in selection.trials["candidate"].unique(maintain_order=True):
            indicator, window, direction = name.rsplit(":", 2)
            specs.append(MiningFactorSpec(indicator, int(window), int(direction)))
        selection = replace(selection, selected=tuple(specs))
    prices = apply_universe_mask(dataset.prices, dataset.universe).sort("date")
    if not prices.equals(revision_panel.prices()):
        raise ValueError("Simulation dataset must match the revision evaluation snapshot")
    if factor is not None:
        chosen = tuple(spec for spec in selection.selected if spec.name == factor)
        if not chosen:
            raise ValueError("Factor is not part of the frozen selection")
        selection = replace(selection, selected=chosen)
    targets = frozen_factor_targets(selection, revision_panel, inner_execution=inner_execution)
    execution_config = broker or BrokerExecutionConfig()
    if execution_config.opening_times is None:
        raise ValueError("Frozen decision simulation requires explicit opening times")
    if execution_config.trade_on_close or execution_config.fill_price != FillPrice.OPEN:
        raise ValueError("Frozen decision simulation requires open fills")
    # Keep training history for factor warm-up, but never count its idle cash
    # snapshots as observations of the frozen strategy's execution returns.
    feed = _frozen_execution_feed(dataset, execution_bars, selection,
                                  revision_panel, execution_config.opening_times)
    cfg = ResearchExecutionPipelineConfig(broker=execution_config, rebalance=rebalance or RebalanceConfig())
    metadata: dict[str, Any] = {"selectionSha256": manifest_sha256, "revisionSha256": revision_panel.fingerprint(),
                "mode": "frozen-factor-simulation"}
    if strategy is not None:
        metadata["strategySha256"] = strategy.sha256
        metadata["strategyArtifact"] = strategy.to_bytes().decode()
    events = {}
    for name, weights in targets.items():
        provenance = {**metadata, "factor": name}
        if trial_family:
            if name in selected_names:
                from .factor_strategy import FrozenFactorStrategy
                from ..event.manifest import engine_code_digest

                artifact = FrozenFactorStrategy(engine_code_digest(), manifest_sha256, name,
                    tuple(sorted(set(dataset.prices.columns) - {"date"})), cfg.rebalance)
                provenance.update(strategySha256=artifact.sha256, strategyArtifact=artifact.to_bytes().decode())
            else:
                provenance["mode"] = "frozen-trial-diagnostic"
        events[name] = _execute_targets(weights, feed, dataset=dataset, cfg=cfg, decision_times=True,
                                       run_metadata=provenance)
    if trial_family and selection.settings.get("benchmark") == "initial_equal_weight_buy_hold":
        symbols = sorted(set(dataset.prices.columns) - {"date"})
        decision = revision_panel.decisions.filter(pl.col("date") > selection.inner_end)[0, "decision_time"]
        weights = pl.DataFrame({"date": [decision], **{s: [1 / len(symbols)] for s in symbols}})
        events["__benchmark__"] = _execute_targets(weights, feed, dataset=dataset, cfg=cfg, decision_times=True,
            run_metadata={**metadata, "mode": "frozen-benchmark-simulation",
                          "benchmarkPolicy": "initial_equal_weight_buy_hold",
                          "rebalance": cfg.rebalance.to_document()})
    return events


def frozen_factor_quantity_changes(
    checkpoint: str | Path, *, manifest_sha256: str, factor: str, revision_panel: RevisionPanel,
    as_of: datetime, prices: dict[str, float], quantities: dict[str, float], equity: float,
    inner_execution: RevisionPanel | None = None, rebalance: RebalanceConfig | None = None,
) -> dict[str, float]:
    """Size one frozen factor against a caller-verified current portfolio snapshot.

    The caller must establish account identity, common valuation currency, raw
    executable prices and snapshot freshness. This computes diagnostic changes,
    not orders or execution authority. Every study asset requires an explicit
    holding (including zero) and a price; missing observations cannot liquidate
    positions by implication. Use the same rebalance policy as the simulation.
    """
    from ..research import frozen_factor_values, top_bottom_weights
    from ..research.mining import _frozen_inputs, _bottom_quantile
    from .mining_store import load_mining_result

    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("Portfolio planning requires an aware cutoff")
    if revision_panel.decisions[-1, "decision_time"] > as_of:
        raise ValueError("Factor decisions follow the portfolio planning cutoff")
    selection = load_mining_result(checkpoint, manifest_sha256=manifest_sha256)
    rebalance = _frozen_rebalance(selection, rebalance)
    if selection.settings.get("knowledge_policy", "received") != "received":
        anchor = selection.settings.get("replay_as_of")
        if (anchor is None or revision_panel.replay_as_of != datetime.fromisoformat(anchor)
                or revision_panel.decisions[-1, "decision_time"] <= revision_panel.replay_as_of):
            raise ValueError("historical_replay_not_admitted_to_shadow")
    if factor not in {spec.name for spec in selection.selected}:
        raise ValueError("Factor is not part of the frozen selection")
    if inner_execution is not None:
        inner_execution = inner_execution.head(revision_panel.through(selection.inner_end).decisions.height)
    if equity == 0:
        symbols = set(revision_panel.revisions["symbol"])
        if (set(prices) != symbols or set(quantities) != symbols
                or any(quantity != 0 for quantity in quantities.values())
                or any(not isfinite(price) or price <= 0 for price in prices.values())):
            raise ValueError("Zero capital requires explicit zero holdings and positive scoped prices")
        _frozen_inputs(selection, revision_panel, None, inner_execution)
        return {}
    values = frozen_factor_values(selection, revision_panel, inner_execution=inner_execution,
                                 factor=factor, last_only=True)[factor].tail(1)
    if values.is_empty():
        raise ValueError("No post-selection factor decision available")
    if any(value is None or not isfinite(value) for value in values.drop("date").row(0)):
        raise ValueError("Incomplete factor decision cannot imply liquidation")
    weights = top_bottom_weights(values, bottom_quantile=_bottom_quantile(selection.settings.get("portfolio_mode", "long_short")))
    row = {symbol: weight for symbol, weight in weights.row(-1, named=True).items() if symbol != "date"}
    if set(prices) != set(row) or set(quantities) != set(row):
        raise ValueError("Explicit prices and holdings must exactly cover the frozen universe")
    if (not isfinite(equity) or equity <= 0
            or any(not isfinite(v) or v <= 0 for v in prices.values())
            or any(not isfinite(v) for v in quantities.values())):
        raise ValueError("Portfolio sizing requires finite equity, holdings and positive prices")
    return quantity_changes_for_target_weights(target_weights=row, prices=prices, quantities=quantities,
                                               equity=equity, config=rebalance)
