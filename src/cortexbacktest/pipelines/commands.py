"""Command workflow orchestration over datasets, research, and event APIs.

The CLI supplies parsed options; these functions return payloads without owning
argument parsing or terminal output.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, cast

import polars as pl

from ..contracts import FillPrice
from ..data import apply_universe_mask
from ..data.backend import (
    normalize_date_column,
    read_table,
)
from ..datasets import (
    load_backtest_dataset,
    load_backtest_dataset_dir,
    load_target_weights,
    normalize_bars,
    save_backtest_dataset,
    validate_bars,
)
from ..event import (
    BacktestEngine,
    BrokerSimulator,
    EventRecordSink,
    LongOnlyRule,
    MaxGroupWeightRule,
    MaxLeverageRule,
    MaxOrderValueRule,
    MaxPositionValueRule,
    MaxPositionWeightRule,
    RebalanceConfig,
    TargetWeightStrategy,
    load_strategy,
    parse_strategy_kwargs,
)
from ..reporting import (
    event_report,
    execution_sweep_report,
    factor_sweep_report,
    research_report,
    write_event_report_html,
    write_event_report_json,
    write_execution_sweep_report_html,
    write_execution_sweep_report_json,
    write_factor_sweep_report_html,
    write_factor_sweep_report_json,
    write_report_html,
    write_report_json,
    write_research_report_html,
    write_research_report_json,
)
from ..execution import run_execution_sweep
from .mining_store import load_mining_result, save_mining_result
from ..pipelines import (
    BrokerExecutionConfig,
    ResearchExecutionPipelineConfig,
    run_research_execution_pipeline,
    run_mining_execution_pipeline,
)
from ..research import (
    WalkForwardSplit,
    auto_mine_factors,
    evaluate_holdout,
    FactorPreprocessConfig,
    FactorSweepCase,
    run_factor_sweep,
    top_bottom_weights,
)



def _bars_summary(bars: pl.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(bars)),
        "symbols": bars["symbol"].n_unique(),
        "start": _maybe_iso(bars["date"].min()),
        "end": _maybe_iso(bars["date"].max()),
    }


def _load_dataset(args: argparse.Namespace):
    return load_backtest_dataset(
        bars=args.bars, factors=args.factors, membership=args.membership,
        date_col=args.date_col, symbol_col=args.symbol_col,
        price_col=args.price_col, factor_col=args.factor_col,
    )


def _dataset_status(dataset) -> dict[str, Any]:
    """Saved-dataset universe and quality evidence shared by inspect and build."""
    return {
        "universe": {
            "enabled": dataset.universe is not None,
            "active_cells": int(
                dataset.universe.mask.select(pl.exclude("date")).sum().sum_horizontal().item()
            ) if dataset.universe is not None else None,
        },
        "quality": {
            "ok": dataset.quality.ok,
            "errors": list(dataset.quality.errors),
            "warnings": list(dataset.quality.warnings),
        },
    }


def _dataset_inspect(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_backtest_dataset_dir(args.dataset)
    bars = dataset.bars
    factors = dataset.factors
    return {
        # Reported as "path" regardless of which spelling was used, so the
        # output shape does not change with the option name.
        "path": str(args.dataset),
        "bars": {"columns": list(bars.columns), **_bars_summary(bars)},
        "factors": {
            "rows": int(len(factors)),
            "columns": list(factors.columns),
        },
        "matrices": {
            "prices": _wide_polars_shape(dataset.prices),
            "factor_values": _wide_polars_shape(dataset.factor_values),
            "returns": _wide_polars_shape(dataset.returns),
        },
        "calendar": {
            "sessions": int(len(dataset.calendar.sessions)),
            "start": _maybe_iso(dataset.calendar.sessions.min()),
            "end": _maybe_iso(dataset.calendar.sessions.max()),
        },
        **_dataset_status(dataset),
    }


def _dataset_build(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _load_dataset(args)
    output_dir = save_backtest_dataset(
        dataset,
        args.output,
        format=args.format,
    )
    bars = dataset.bars
    factors = dataset.factors
    return {
        "output": str(output_dir),
        "format": args.format,
        "bars": _bars_summary(bars),
        "factors": {
            "rows": int(len(factors)),
        },
        "matrices": {
            "prices": _wide_polars_shape(dataset.prices),
            "factor_values": _wide_polars_shape(dataset.factor_values),
        },
        **_dataset_status(dataset),
    }


def _dataset_validate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    normalized_bars = normalize_bars(
        read_table(args.bars),
        date_col=args.date_col,
        symbol_col=args.symbol_col,
        price_col=args.price_col,
    )
    quality = validate_bars(normalized_bars)
    errors = list(quality.errors)
    warnings = list(quality.warnings)
    dataset_summary = None
    if args.factors is not None and quality.ok:
        try:
            dataset = _load_dataset(args)
            # Shapes report (sessions, symbols); the date column is excluded.
            prices_shape = _wide_polars_shape(dataset.prices)
            factor_values_shape = _wide_polars_shape(dataset.factor_values)
            dataset_summary = {
                "factors_rows": int(len(dataset.factors)),
                "prices_shape": prices_shape,
                "factor_values_shape": factor_values_shape,
                "universe_enabled": dataset.universe is not None,
            }
        except Exception as exc:
            errors.append(str(exc))
    payload = {
        "ok": not errors,
        "bars": _bars_summary(normalized_bars),
        "dataset": dataset_summary,
        "errors": errors,
        "warnings": warnings,
    }
    return payload, 0 if payload["ok"] else 1


def _write_outputs(result, args, json_writer, html_writer, **kwargs) -> dict[str, str | None]:
    json_path = json_writer(result, args.output_json, **kwargs)
    html_path = html_writer(result, args.output_html, **kwargs) if args.output_html is not None else None
    return {"json": str(json_path), "html": str(html_path) if html_path else None}


def _factor_mining(args: argparse.Namespace) -> dict[str, Any]:
    """Use one dataset/evaluator and freeze proposals before outer backtesting.

    This command is retrospective research, not historical PIT certification.
    """
    if not 0.5 <= args.inner_fraction <= 0.9:
        raise ValueError("inner_fraction must be between 0.5 and 0.9")
    dataset = load_backtest_dataset_dir(args.dataset)
    if not dataset.quality.ok:
        raise ValueError(f"Invalid dataset: {dataset.quality.errors}")
    prices = apply_universe_mask(dataset.prices, dataset.universe).sort("date")
    split = int(prices.height * args.inner_fraction)
    if prices.height - split <= args.horizon:
        raise ValueError("Insufficient later holdout data")
    result = auto_mine_factors(
        prices.head(split), research_direction=args.direction,
        budget=args.budget, seed=args.seed, horizon=args.horizon, cost_bps=args.cost_bps,
        validation=WalkForwardSplit(args.train_sessions, args.test_sessions),
    )
    directory = Path(args.output)
    receipt = save_mining_result(result, directory)
    restored = load_mining_result(directory, manifest_sha256=receipt)
    executions = run_mining_execution_pipeline(
        restored, dataset, execution_bars=read_table(args.execution_bars),
        broker=BrokerExecutionConfig(**{**_broker_options(args), "fill_price": FillPrice(args.fill_price)}),
    ) if args.execution_bars is not None else {}
    reports = ({name: result.research for name, result in executions.items()} if args.execution_bars is not None
               else evaluate_holdout(restored, prices) if restored.selected else {})
    payload = {"evaluation_mode": "retrospective_not_pit_certified",
               "prices_sha256": hashlib.sha256(prices.write_csv().encode()).hexdigest(),
               "dataset_provenance": dataset.provenance.to_dict() if dataset.provenance else None,
               "manifest_sha256": receipt, "trials": result.trials.height,
               "selected": [spec.name for spec in restored.selected],
               "holdout": {name: research_report(report) for name, report in reports.items()},
               "status": "evaluated" if reports else "no_selected_factors"}
    if args.execution_bars is not None:
        payload["execution_bars_sha256"] = hashlib.sha256(args.execution_bars.read_bytes()).hexdigest()
        payload["execution"] = {name: result.report() for name, result in executions.items()}
        payload["execution_settings"] = _broker_options(args)
    write_report_json(payload, directory / "holdout.json")
    return payload


def _factor_research(args: argparse.Namespace) -> dict[str, Any]:
    horizons = _parse_horizons(args.horizons)
    preprocessing, preprocessing_settings = _build_preprocessing(args)
    dataset = load_backtest_dataset_dir(args.dataset)
    report = dataset.run_factor_research(
        horizons=horizons,
        quantiles=args.quantiles,
        preprocessing=preprocessing,
    )
    include_records = not args.no_records
    outputs = _write_outputs(
        report, args, write_research_report_json, write_research_report_html,
        include_records=include_records,
    )
    payload = research_report(report, include_records=False)
    payload["outputs"] = outputs
    payload["settings"] = {
        "dataset": str(args.dataset),
        "horizons": list(horizons),
        "quantiles": args.quantiles,
        "preprocessing": preprocessing_settings,
        "include_records": include_records,
    }
    return payload


def _broker_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "initial_cash": args.initial_cash,
        "commission_bps": args.commission_bps,
        "slippage_bps": args.slippage_bps,
        "fill_price": args.fill_price,
        "allow_short": args.allow_short,
        "max_volume_share": args.max_volume_share,
    }


def _rebalance_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cash_buffer": args.cash_buffer,
        "min_trade_value": args.min_trade_value,
        "allow_fractional": args.allow_fractional,
    }


def _prepare_factor_sweep(args: argparse.Namespace):
    horizons = _parse_horizons(args.horizons)
    preprocessing, preprocessing_settings = _build_preprocessing(args)
    bottom_quantile = None if args.bottom_quantile <= 0 else args.bottom_quantile
    cases = [
        _parse_sweep_case(
            value,
            preprocessing=preprocessing,
            top_quantile=args.top_quantile,
            bottom_quantile=bottom_quantile,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        for value in args.case
    ]
    dataset = load_backtest_dataset_dir(args.dataset)
    factor_result = run_factor_sweep(
        dataset.prices,
        cases,
        universe=dataset.universe,
        horizons=horizons,
        quantiles=args.quantiles,
    )
    return dataset, factor_result, {
        "dataset": str(args.dataset),
        "cases": [case.name for case in cases],
        "horizons": list(horizons),
        "quantiles": args.quantiles,
        "top_quantile": args.top_quantile,
        "bottom_quantile": bottom_quantile,
        "transaction_cost_bps": args.transaction_cost_bps,
        "preprocessing": preprocessing_settings,
    }


def _factor_sweep(args: argparse.Namespace) -> dict[str, Any]:
    _, result, settings = _prepare_factor_sweep(args)
    include_records = not args.no_records
    outputs = _write_outputs(
        result, args, write_factor_sweep_report_json, write_factor_sweep_report_html,
        include_records=include_records,
    )
    payload = factor_sweep_report(result, include_records=False)
    payload["outputs"] = outputs
    payload["settings"] = {
        **settings,
        "include_records": include_records,
    }
    return payload


def _execution_sweep(args: argparse.Namespace) -> dict[str, Any]:
    dataset, factor_result, settings = _prepare_factor_sweep(args)
    risk_rules, risk_settings = _build_risk_rules(args)

    result = run_execution_sweep(
        factor_result,
        dataset.to_bars(),
        **{**_broker_options(args), "fill_price": FillPrice(args.fill_price)},
        risk_rules=risk_rules,
        rebalance_config=RebalanceConfig(
            **_rebalance_options(args),
        ),
        trade_on_close=args.trade_on_close,
        max_workers=args.max_workers,
        parallel_backend=args.parallel_backend,
    )
    include_records = not args.no_records
    outputs = _write_outputs(
        result, args, write_execution_sweep_report_json, write_execution_sweep_report_html,
        include_records=include_records,
    )
    payload = execution_sweep_report(result, include_records=False)
    payload["outputs"] = outputs
    payload["settings"] = {
        **settings,
        **_broker_options(args),
        "risk_rules": risk_settings,
        **_rebalance_options(args),
        "trade_on_close": args.trade_on_close,
        "max_workers": args.max_workers,
        "parallel_backend": args.parallel_backend,
        "include_records": include_records,
    }
    return payload


def _build_preprocessing(
    args: argparse.Namespace,
) -> tuple[FactorPreprocessConfig | None, dict[str, Any]]:
    settings: dict[str, Any] = {}
    winsorize_quantiles = None
    if args.winsorize is not None:
        winsorize_quantiles = _parse_float_tuple(args.winsorize, label="winsorize", size=2)
        settings["winsorize"] = list(winsorize_quantiles)
    neutralize_groups = None
    if args.neutralize_groups_file is not None:
        neutralize_groups = _load_groups_file(
            args.neutralize_groups_file,
            symbol_col=args.neutralize_symbol_col,
            group_col=args.neutralize_group_col,
        )
        settings["neutralize_groups_file"] = str(args.neutralize_groups_file)
    if args.rank:
        settings["rank"] = True
    if args.zscore:
        settings["zscore"] = True
    if args.demean:
        settings["demean"] = True
    if not settings:
        return None, {}
    return (
        FactorPreprocessConfig(
            winsorize_quantiles=cast(tuple[float, float] | None, winsorize_quantiles),
            neutralize_groups=neutralize_groups,
            rank=args.rank,
            zscore=args.zscore,
            demean=args.demean,
        ),
        settings,
    )


def _parse_sweep_case(
    value: str,
    *,
    preprocessing: FactorPreprocessConfig | None,
    top_quantile: float,
    bottom_quantile: float | None,
    transaction_cost_bps: float,
) -> FactorSweepCase:
    if "=" not in value:
        raise ValueError("factor sweep case must use name=indicator:param=value")
    name, spec = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("factor sweep case name must not be empty")
    parts = [part.strip() for part in spec.split(":") if part.strip()]
    if not parts:
        raise ValueError("factor sweep case requires an indicator name")
    if parts[0] in {"builtin", "tulip", "pandas_ta"}:
        if len(parts) < 2 or "=" in parts[1]:
            raise ValueError("namespaced factor case requires provider:indicator")
        indicator_name = ":".join(parts[:2])
        params = _parse_case_params(parts[2:])
    else:
        indicator_name = parts[0]
        params = _parse_case_params(parts[1:])
    from ..research.indicator_registry import INDICATORS

    factory = _indicator_factory(indicator_name, params)
    return FactorSweepCase(
        name,
        factory,
        params={"indicator": indicator_name, **params},
        indicator_metadata=INDICATORS.describe(indicator_name, **params),
        preprocessing=preprocessing,
        top_quantile=top_quantile,
        bottom_quantile=bottom_quantile,
        transaction_cost_bps=transaction_cost_bps,
    )


def _indicator_factory(indicator_name: str, params: dict[str, Any]):
    from ..research.indicator_registry import INDICATORS

    if ":" in indicator_name:
        requirements = INDICATORS.describe(indicator_name, **params)
        if not requirements.get("requirements_known", True) or requirements["input_kind"] != "prices":
            raise ValueError("factor CLI requires a reviewed wide-price indicator")
    elif indicator_name not in {"momentum", "sma_ratio", "ema_ratio", "rolling_volatility", "rsi"}:
        raise ValueError(f"Unsupported factor indicator: {indicator_name}")

    def build(prices):
        return INDICATORS.run(indicator_name, prices, **params)

    return build


def _parse_case_params(values: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("factor sweep case params must use key=value")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("factor sweep case param name must not be empty")
        params[key] = _parse_scalar(raw.strip())
    return params


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _research_execution(args: argparse.Namespace) -> dict[str, Any]:
    horizons = _parse_horizons(args.horizons)
    preprocessing, preprocessing_settings = _build_preprocessing(args)
    bottom_quantile = None if args.bottom_quantile <= 0 else args.bottom_quantile
    risk_rules, risk_settings = _build_risk_rules(args)
    dataset = load_backtest_dataset_dir(args.dataset)
    config = ResearchExecutionPipelineConfig(
        horizons=horizons,
        quantiles=args.quantiles,
        top_quantile=args.top_quantile,
        bottom_quantile=bottom_quantile,
        transaction_cost_bps=args.transaction_cost_bps,
        preprocessing=preprocessing,
        rebalance=RebalanceConfig(
            **_rebalance_options(args),
        ),
        broker=BrokerExecutionConfig(
            **{**_broker_options(args), "fill_price": FillPrice(args.fill_price)},
            trade_on_close=False,
            snapshot_frequency=args.snapshot_frequency,
            risk_rules=tuple(risk_rules),
        ),
    )
    result = run_research_execution_pipeline(dataset, config=config)
    include_records = not args.no_records
    report = result.report(include_records=include_records)
    json_path = write_report_json(report, args.output_json)
    html_path = None
    if args.output_html is not None:
        html_path = write_report_html(
            "CortexBacktest - Research Execution Pipeline",
            report,
            args.output_html,
        )
    payload = result.report(include_records=False)
    payload["outputs"] = {"json": str(json_path), "html": str(html_path) if html_path else None}
    payload["settings"] = {
        "dataset": str(args.dataset),
        "horizons": list(horizons),
        "quantiles": args.quantiles,
        "top_quantile": args.top_quantile,
        "bottom_quantile": bottom_quantile,
        "transaction_cost_bps": args.transaction_cost_bps,
        "preprocessing": preprocessing_settings,
        **_broker_options(args),
        "risk_rules": risk_settings,
        **_rebalance_options(args),
        "trade_on_close": config.broker.trade_on_close,
        "snapshot_frequency": args.snapshot_frequency,
        "include_records": include_records,
    }
    return payload


def _event_backtest(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_backtest_dataset_dir(args.dataset)
    risk_rules, risk_settings = _build_risk_rules(args)
    broker = BrokerSimulator(
        **{**_broker_options(args), "fill_price": FillPrice(args.fill_price)},
        risk_rules=risk_rules,
    )
    bottom_quantile = None
    target_source = None
    if args.strategy is not None:
        if args.target_weights is not None:
            raise ValueError("--strategy and --target-weights cannot be used together")
        strategy = load_strategy(
            args.strategy,
            kwargs=parse_strategy_kwargs(args.strategy_kwargs),
            broker=broker,
            dataset=dataset,
        )
        strategy_source = args.strategy
    else:
        if args.target_weights is None:
            if args.bottom_quantile <= 0:
                bottom_quantile = None
            else:
                bottom_quantile = args.bottom_quantile
            target_weights = top_bottom_weights(
                dataset.factor_values,
                top_quantile=args.top_quantile,
                bottom_quantile=bottom_quantile,
                long_weight=args.long_weight,
                short_weight=args.short_weight,
            )
            target_source = "factor_values"
        else:
            target_weights = load_target_weights(
                args.target_weights,
                date_col=args.date_col,
                symbol_col=args.symbol_col,
                weight_col=args.weight_col,
            )
            target_source = str(args.target_weights)
        strategy = TargetWeightStrategy(
            target_weights,
            broker=broker,
            config=RebalanceConfig(
                **_rebalance_options(args),
            ),
        )
        strategy_source = "target_weights"
    record_sink = None
    if args.records_dir is not None:
        record_sink = EventRecordSink(args.records_dir, format=args.records_format)
    result = BacktestEngine(
        broker=broker,
        trade_on_close=args.trade_on_close,
        snapshot_frequency=args.snapshot_frequency,
        result_retention=args.result_retention,
        record_sink=record_sink,
        build_manifest=args.build_manifest,
    ).run(
        dataset.to_event_feed(),
        strategy,
    )
    benchmark_equity = None
    if args.benchmark is not None:
        benchmark_equity = _load_equity_series(
            args.benchmark,
            date_col=args.benchmark_date_col,
            equity_col=args.benchmark_equity_col,
        )
    include_records = not args.no_records and result.records_retained
    outputs = _write_outputs(
        result, args, write_event_report_json, write_event_report_html,
        include_records=include_records,
        benchmark_equity=benchmark_equity,
        benchmark_rolling_window=args.benchmark_rolling_window,
    )
    payload = event_report(
        result,
        include_records=False,
        benchmark_equity=benchmark_equity,
        benchmark_rolling_window=args.benchmark_rolling_window,
    )
    payload["outputs"] = {
        **outputs,
        "records_dir": str(args.records_dir) if args.records_dir is not None else None,
    }
    payload["settings"] = {
        "dataset": str(args.dataset),
        **_broker_options(args),
        "risk_rules": risk_settings,
        "top_quantile": args.top_quantile,
        "bottom_quantile": bottom_quantile,
        "long_weight": args.long_weight,
        "short_weight": args.short_weight,
        "target_weights": target_source,
        "strategy": strategy_source,
        **_rebalance_options(args),
        "trade_on_close": args.trade_on_close,
        "snapshot_frequency": args.snapshot_frequency,
        "result_retention": args.result_retention,
        "benchmark": str(args.benchmark) if args.benchmark is not None else None,
        "benchmark_rolling_window": args.benchmark_rolling_window
        if args.benchmark is not None
        else None,
        "include_records": include_records,
        "records_format": args.records_format if args.records_dir is not None else None,
    }
    return payload


def _build_risk_rules(args: argparse.Namespace):
    rules = []
    settings: dict[str, Any] = {}
    if args.long_only:
        rules.append(LongOnlyRule())
        settings["long_only"] = True
    for name, rule in (
        ("max_order_value", MaxOrderValueRule),
        ("max_position_value", MaxPositionValueRule),
        ("max_leverage", MaxLeverageRule),
        ("max_position_weight", MaxPositionWeightRule),
    ):
        value = getattr(args, name)
        if value is not None:
            rules.append(rule(value))
            settings[name] = value
    if args.max_group_weight:
        if args.groups_file is None:
            raise ValueError("--max-group-weight requires --groups-file")
        groups = _load_groups_file(
            args.groups_file,
            symbol_col=args.group_symbol_col,
            group_col=args.group_col,
        )
        max_weights = _parse_key_float_pairs(args.max_group_weight, label="max_group_weight")
        rules.append(MaxGroupWeightRule(groups=groups, max_weights=max_weights))
        settings["groups_file"] = str(args.groups_file)
        settings["max_group_weight"] = max_weights
    return rules, settings


def _load_groups_file(
    path: Path,
    *,
    symbol_col: str = "symbol",
    group_col: str = "group",
) -> dict[str, str]:
    frame = read_table(path, backend="polars")
    missing = {symbol_col, group_col}.difference(frame.columns)
    if missing:
        raise ValueError(f"groups file missing required columns: {sorted(missing)}")
    groups = (
        frame.select(symbol_col, group_col)
        .drop_nulls()
        .with_columns(
            pl.col(symbol_col).cast(pl.String).str.to_uppercase(),
            pl.col(group_col).cast(pl.String),
        )
    )
    if groups.is_empty():
        raise ValueError("groups file contains no usable rows")
    return dict(groups.iter_rows())


def _load_equity_series(
    path: Path,
    *,
    date_col: str = "date",
    equity_col: str = "equity",
) -> pl.DataFrame:
    """Load a benchmark equity curve as a timestamp/equity Polars frame."""
    frame = read_table(path)
    missing = {date_col, equity_col}.difference(frame.columns)
    if missing:
        raise ValueError(f"equity file missing required columns: {sorted(missing)}")
    data = (
        normalize_date_column(frame, date_col)
        .select(
            pl.col(date_col).alias("timestamp"),
            pl.col(equity_col).cast(pl.Float64, strict=False).alias("equity"),
        )
        .drop_nulls()
        .sort("timestamp")
    )
    if data.is_empty():
        raise ValueError("equity file contains no numeric equity values")
    return data


def _parse_key_float_pairs(values: list[str], *, label: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} entries must use GROUP=WEIGHT")
        key, raw_weight = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{label} group name must not be empty")
        try:
            parsed[key] = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"{label} weight must be numeric: {value}") from exc
    return parsed


def _parse_float_tuple(value: str, *, label: str, size: int) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be a comma-separated list of numbers") from exc
    if len(parsed) != size:
        raise ValueError(f"{label} requires {size} comma-separated values")
    return parsed


def _parse_horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("horizons must be a comma-separated list of integers") from exc
    if not horizons:
        raise ValueError("horizons must not be empty")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive integers")
    return horizons


def _maybe_iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _wide_polars_shape(frame) -> list[int]:
    return [frame.height, sum(column != "date" for column in frame.columns)]
