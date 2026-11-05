"""Command-line parsing and presentation."""

from __future__ import annotations
import argparse
import json
import sys
from datetime import time
from pathlib import Path
from typing import Sequence
from . import __version__
from .contracts import FillPrice
from .options.calculator import analyse
from .options.chain_view import DEFAULT_ROOT as CHAIN_ROOT, chain_view
from .options.chains import ChainTimePolicy, OptionChainStore, clean_option_chain, normalize_option_chain
from .pipelines.commands import (
    _dataset_inspect,
    _dataset_build,
    _dataset_validate,
    _factor_research,
    _factor_mining,
    _factor_sweep,
    _execution_sweep,
    _research_execution,
    _event_backtest,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cortexbacktest command line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        payload = args.func(args)
    except Exception as exc:  # pragma: no cover - exact argparse stderr path is enough
        print(f"cortexbacktest: error: {exc}", file=sys.stderr)
        return 1
    exit_code = 0
    if isinstance(payload, tuple):
        payload, exit_code = payload
    if payload is not None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortexbacktest",
        description="CortexBacktest dataset and research utilities.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    from .pipelines.recipe_commands import add_recipe_commands
    add_recipe_commands(subparsers)
    from .pipelines.dynamic_style_commands import add_dynamic_style_commands
    add_dynamic_style_commands(subparsers)
    from .pipelines.named_recipe_commands import add_named_recipe_commands
    add_named_recipe_commands(subparsers)
    from .pipelines.news_event_commands import add_news_event_commands
    add_news_event_commands(subparsers)

    inspect_parser = subparsers.add_parser(
        "dataset-inspect",
        help="Inspect a saved BacktestDataset directory.",
    )
    _add_dataset_arg(inspect_parser)
    inspect_parser.set_defaults(func=_dataset_inspect)

    build_parser = subparsers.add_parser(
        "dataset-build",
        help="Build and save a standard BacktestDataset directory from bars/factors files.",
    )
    build_parser.add_argument(
        "--bars", required=True, type=Path, help="Input bars CSV/parquet path."
    )
    build_parser.add_argument(
        "--factors", required=True, type=Path, help="Input factors CSV/parquet path."
    )
    build_parser.add_argument(
        "--output", required=True, type=Path, help="Output dataset directory."
    )
    build_parser.add_argument(
        "--membership", type=Path, help="Optional point-in-time membership CSV/parquet path."
    )
    _add_column_args(build_parser)
    build_parser.add_argument(
        "--format",
        default="csv",
        choices=("csv", "parquet"),
        help="Saved dataset file format.",
    )
    build_parser.set_defaults(func=_dataset_build)

    validate_parser = subparsers.add_parser(
        "dataset-validate",
        help="Validate raw bars/factors files before building a dataset.",
    )
    validate_parser.add_argument(
        "--bars", required=True, type=Path, help="Input bars CSV/parquet path."
    )
    validate_parser.add_argument("--factors", type=Path, help="Optional factors CSV/parquet path.")
    validate_parser.add_argument(
        "--membership", type=Path, help="Optional point-in-time membership CSV/parquet path."
    )
    _add_column_args(validate_parser)
    validate_parser.set_defaults(func=_dataset_validate)

    research_parser = subparsers.add_parser(
        "factor-research",
        help="Run factor research from a saved BacktestDataset directory.",
    )
    _add_dataset_arg(research_parser)
    _add_report_output_args(research_parser, "research")
    _add_horizon_args(research_parser)
    _add_preprocessing_args(research_parser)
    _add_no_records_arg(research_parser, "time-series record tables")
    research_parser.set_defaults(func=_factor_research)

    mining_parser = subparsers.add_parser("factor-mine", help="Mine, freeze, then backtest on a reserved outer period.")
    _add_dataset_arg(mining_parser)
    mining_parser.add_argument("--output", type=Path, required=True, help="New immutable campaign directory.")
    mining_parser.add_argument("--direction", choices=("momentum", "mean_reversion", "low_volatility"), required=True)
    mining_parser.add_argument("--inner-fraction", type=float, default=0.7)
    mining_parser.add_argument("--train-sessions", type=int, default=252)
    mining_parser.add_argument("--test-sessions", type=int, default=126)
    mining_parser.add_argument("--budget", type=int, default=12)
    mining_parser.add_argument("--seed", type=int, default=7)
    mining_parser.add_argument("--horizon", type=int, default=1)
    mining_parser.add_argument("--cost-bps", type=float, default=10.0)
    mining_parser.add_argument("--execution-bars", type=Path,
                               help="Explicit raw OHLCV table for frozen outer-period event execution.")
    _add_broker_args(mining_parser, fill_price=FillPrice.OPEN)
    mining_parser.set_defaults(func=_factor_mining)

    sweep_parser = subparsers.add_parser(
        "factor-sweep",
        help="Compare multiple built-in factor configurations from a saved dataset.",
    )
    _add_dataset_arg(sweep_parser)
    sweep_parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Sweep case as name=indicator:param=value, for example mom_1=momentum:window=1.",
    )
    _add_report_output_args(sweep_parser, "factor sweep")
    _add_horizon_args(sweep_parser)
    _add_quantile_portfolio_args(sweep_parser)
    _add_preprocessing_args(sweep_parser)
    _add_no_records_arg(sweep_parser, "nested case reports")
    sweep_parser.set_defaults(func=_factor_sweep)

    execution_sweep_parser = subparsers.add_parser(
        "execution-sweep",
        help="Run factor sweep target weights through the event execution engine.",
    )
    _add_dataset_arg(execution_sweep_parser)
    execution_sweep_parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Sweep case as name=indicator:param=value, for example mom_1=momentum:window=1.",
    )
    _add_report_output_args(execution_sweep_parser, "execution sweep")
    _add_horizon_args(execution_sweep_parser)
    _add_quantile_portfolio_args(execution_sweep_parser, cost_help="Research portfolio cost in bps.")
    _add_preprocessing_args(execution_sweep_parser)
    _add_broker_args(execution_sweep_parser)
    _add_rebalance_args(execution_sweep_parser)
    execution_sweep_parser.add_argument(
        "--trade-on-close",
        action="store_true",
        help="Fill orders on the decision bar's close (same-bar); the default fills at the next bar's open.",
    )
    execution_sweep_parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel event execution workers for sweep cases. Use 1 for serial execution.",
    )
    execution_sweep_parser.add_argument(
        "--parallel-backend",
        choices=("process", "thread"),
        default="process",
        help="Parallel backend for execution sweeps.",
    )
    _add_risk_args(execution_sweep_parser)
    _add_no_records_arg(execution_sweep_parser, "nested event result reports")
    execution_sweep_parser.set_defaults(func=_execution_sweep)

    pipeline_parser = subparsers.add_parser(
        "research-execution",
        help="Run the standard dataset -> factor research -> target weights -> event execution pipeline.",
    )
    _add_dataset_arg(pipeline_parser)
    _add_report_output_args(pipeline_parser, "pipeline")
    _add_horizon_args(pipeline_parser)
    _add_quantile_portfolio_args(pipeline_parser, cost_help="Research portfolio cost in bps.")
    _add_preprocessing_args(pipeline_parser)
    _add_broker_args(pipeline_parser, fill_price=FillPrice.OPEN)
    _add_risk_args(pipeline_parser)
    _add_rebalance_args(pipeline_parser)
    pipeline_parser.add_argument(
        "--trade-on-open",
        action="store_true",
        help="Compatibility flag; research-execution already uses next-session open fills by default.",
    )
    pipeline_parser.add_argument(
        "--snapshot-frequency",
        choices=("bar", "session", "final", "none"),
        default="bar",
        help="Equity snapshot frequency for event execution.",
    )
    _add_no_records_arg(pipeline_parser, "nested record tables")
    pipeline_parser.set_defaults(func=_research_execution)

    option_parser = subparsers.add_parser(
        "option-price", help="Price one vanilla option every way; reads a JSON request on stdin.")
    option_parser.set_defaults(func=lambda _args: analyse(json.load(sys.stdin)))
    chain_parser = subparsers.add_parser(
        "option-chain", help="One session of a stored option chain; reads a JSON request on stdin.")
    chain_parser.set_defaults(func=lambda _args: chain_view(json.load(sys.stdin)))
    import_parser = subparsers.add_parser(
        "option-chain-import", help="Add daily chain files in the canonical long shape to the chain store.")
    import_parser.add_argument("files", nargs="+", type=Path)
    import_parser.add_argument("--root", type=Path, default=CHAIN_ROOT, help="Chain store root.")
    import_parser.set_defaults(func=_option_chain_import)
    risk_parser = subparsers.add_parser(
        "portfolio-risk", help="Risk review of a collected account: exposures, expiry, cover, scenarios, margin.")
    risk_parser.add_argument("--collected", type=Path, required=True, help="Collected portfolio snapshot (JSON).")
    risk_parser.add_argument("--events", type=Path, help="Earnings and macro events before the horizon.")
    risk_parser.add_argument("--whatif", type=Path, help="Broker margin what-if result (JSON).")
    risk_parser.add_argument("--config", type=Path, help="Shocks, hedges, leveraged ETFs, rate (JSON).")
    risk_parser.add_argument("--output", type=Path, required=True, help="Directory for report.json and report.md.")
    risk_parser.set_defaults(func=_portfolio_risk)

    event_parser = subparsers.add_parser(
        "event-backtest",
        help="Run an event-driven rebalance backtest from dataset factor target weights.",
    )
    _add_dataset_arg(event_parser)
    event_parser.add_argument(
        "--target-weights",
        type=Path,
        help="Optional CSV/parquet target weights file. Supports wide date x symbol or long date/symbol/weight.",
    )
    event_parser.add_argument(
        "--strategy",
        help="Optional Strategy spec as module:Class or /path/to/strategy.py:Class.",
    )
    event_parser.add_argument(
        "--strategy-kwargs",
        help="Optional JSON object passed to the dynamic strategy constructor.",
    )
    _add_report_output_args(event_parser, "event backtest")
    event_parser.add_argument(
        "--benchmark",
        type=Path,
        help="Optional benchmark equity CSV/parquet with date and equity columns.",
    )
    event_parser.add_argument(
        "--benchmark-date-col", default="date", help="Benchmark date column name."
    )
    event_parser.add_argument(
        "--benchmark-equity-col", default="equity", help="Benchmark equity column name."
    )
    event_parser.add_argument(
        "--benchmark-rolling-window",
        default=20,
        type=int,
        help="Rolling window for benchmark beta and sharpe metrics.",
    )
    _add_broker_args(event_parser)
    _add_risk_args(event_parser)
    _add_quantile_args(event_parser)
    event_parser.add_argument(
        "--long-weight", default=1.0, type=float, help="Total long-side target weight."
    )
    event_parser.add_argument(
        "--short-weight", default=-1.0, type=float, help="Total short-side target weight."
    )
    event_parser.add_argument(
        "--date-col", default="date", help="Date column for target weights files."
    )
    event_parser.add_argument(
        "--symbol-col", default="symbol", help="Symbol column for long target weights files."
    )
    event_parser.add_argument(
        "--weight-col", default="weight", help="Weight column for long target weights files."
    )
    _add_rebalance_args(event_parser)
    event_parser.add_argument(
        "--trade-on-close",
        action="store_true",
        help="Fill orders on the decision bar's close (same-bar); the default fills at the next bar's open.",
    )
    event_parser.add_argument(
        "--snapshot-frequency",
        choices=("bar", "session", "final", "none"),
        default="bar",
        help="Equity snapshot frequency.",
    )
    event_parser.add_argument(
        "--result-retention",
        choices=("full", "summary"),
        default="full",
        help="Retain full event records or only compact summary counts in the returned result.",
    )
    _add_no_records_arg(event_parser, "time-series record tables")
    event_parser.add_argument(
        "--records-dir",
        type=Path,
        help="Optional directory for streaming event records outside the JSON report.",
    )
    event_parser.add_argument(
        "--records-format",
        choices=("csv", "ndjson"),
        default="csv",
        help="Format for --records-dir outputs.",
    )
    event_parser.add_argument(
        "--build-manifest",
        action="store_true",
        help="Hash input bars and include reproducibility metadata (adds per-bar work).",
    )
    event_parser.set_defaults(func=_event_backtest)
    return parser


def _add_broker_args(parser: argparse.ArgumentParser, *, fill_price: FillPrice = FillPrice.OPEN) -> None:
    parser.add_argument(
        "--initial-cash", default=100_000.0, type=float, help="Starting cash."
    )
    parser.add_argument(
        "--commission-bps", default=0.0, type=float, help="Commission in basis points."
    )
    parser.add_argument(
        "--slippage-bps", default=0.0, type=float, help="Slippage in basis points."
    )
    parser.add_argument(
        "--fill-price",
        default=fill_price.value,
        choices=tuple(item.value for item in FillPrice),
        help="Broker fill price model.",
    )
    parser.add_argument(
        "--allow-short",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow short positions in the broker ledger.",
    )
    parser.add_argument(
        "--max-volume-share",
        default=1.0,
        type=float,
        help="Maximum fraction of bar volume available for fills.",
    )


def _add_risk_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--long-only", action="store_true", help="Reject orders that would create shorts."
    )
    parser.add_argument(
        "--max-order-value", type=float, help="Reject fills above this order notional."
    )
    parser.add_argument(
        "--max-position-value", type=float, help="Reject fills above this position notional."
    )
    parser.add_argument(
        "--max-leverage", type=float, help="Reject fills above this projected leverage."
    )
    parser.add_argument(
        "--max-position-weight", type=float, help="Reject fills above this projected weight."
    )
    parser.add_argument(
        "--groups-file",
        type=Path,
        help="Optional CSV/parquet file with symbol and group columns for group risk limits.",
    )
    parser.add_argument(
        "--group-symbol-col", default="symbol", help="Symbol column in --groups-file."
    )
    parser.add_argument(
        "--group-col", default="group", help="Group column in --groups-file."
    )
    parser.add_argument(
        "--max-group-weight",
        action="append",
        default=[],
        metavar="GROUP=WEIGHT",
        help="Projected max group weight. Can be supplied multiple times.",
    )


def _add_rebalance_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cash-buffer", default=0.0, type=float, help="Fraction of equity left as cash."
    )
    parser.add_argument(
        "--min-trade-value", default=0.0, type=float, help="Minimum notional trade value."
    )
    parser.add_argument(
        "--allow-fractional",
        action="store_true",
        help="Allow fractional position quantities.",
    )


def _add_dataset_arg(parser: argparse.ArgumentParser) -> None:
    """The saved-dataset directory every research and execution command reads."""
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset directory path.")


def _add_column_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--date-col", default="date", help="Date column name.")
    parser.add_argument("--symbol-col", default="symbol", help="Symbol column name.")
    parser.add_argument(
        "--price-col", default="adj_close", help="Price column used for research prices."
    )
    parser.add_argument("--factor-col", default="factor", help="Factor value column name.")


def _add_report_output_args(parser: argparse.ArgumentParser, subject: str) -> None:
    """The paired JSON/HTML report destinations.

    Kept separate from `_add_no_records_arg` because `--no-records` does not sit
    next to these in every subcommand, and merging them would silently reorder
    `--help`.
    """
    parser.add_argument(
        "--output-json", required=True, type=Path, help=f"Path for the JSON {subject} report."
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        help=f"Optional path for the standalone HTML {subject} report.",
    )


def _add_no_records_arg(parser: argparse.ArgumentParser, records: str) -> None:
    parser.add_argument(
        "--no-records", action="store_true", help=f"Write compact reports without {records}."
    )


def _add_horizon_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--horizons",
        default="1,5,20",
        help="Comma-separated forward-return horizons, for example 1,5,20.",
    )
    parser.add_argument("--quantiles", default=5, type=int, help="Number of factor quantiles.")


def _add_quantile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--top-quantile", default=0.2, type=float, help="Long-side factor quantile."
    )
    parser.add_argument(
        "--bottom-quantile",
        default=0.2,
        type=float,
        help="Short-side factor quantile. Use 0 to disable shorts.",
    )


def _add_quantile_portfolio_args(
    parser: argparse.ArgumentParser, *, cost_help: str = "Portfolio cost in bps."
) -> None:
    _add_quantile_args(parser)
    parser.add_argument("--transaction-cost-bps", default=0.0, type=float, help=cost_help)


def _add_preprocessing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--winsorize",
        help="Optional lower,upper cross-sectional winsorization quantiles, for example 0.01,0.99.",
    )
    parser.add_argument(
        "--rank", action="store_true", help="Rank factor values by date before research."
    )
    parser.add_argument(
        "--zscore", action="store_true", help="Z-score factor values by date before research."
    )
    parser.add_argument(
        "--demean", action="store_true", help="Demean factor values by date before research."
    )
    parser.add_argument(
        "--neutralize-groups-file",
        type=Path,
        help="Optional CSV/parquet file with symbol and group columns for factor neutralization.",
    )
    parser.add_argument(
        "--neutralize-symbol-col",
        default="symbol",
        help="Symbol column in --neutralize-groups-file.",
    )
    parser.add_argument(
        "--neutralize-group-col",
        default="group",
        help="Group column in --neutralize-groups-file.",
    )



def _option_chain_import(args: argparse.Namespace) -> dict[str, object]:
    """Each file is one underlying's session, quoted at the US close (16:00 New York, either season)."""
    import polars as pl

    store, policy = OptionChainStore(args.root), ChainTimePolicy(time(16), time(16), zone="America/New_York")
    written: dict[str, object] = {}
    for path in args.files:
        chain, removed = clean_option_chain(normalize_option_chain(pl.read_csv(path), "long", time_policy=policy))
        written[path.name] = {"stored": store.ingest(chain), "dropped": removed}
    return written


def _portfolio_risk(args: argparse.Namespace) -> dict[str, object]:
    from .options.portfolio_risk import analyse, render

    def read(path: Path | None) -> dict:
        return json.loads(path.read_text()) if path else {}

    report = analyse(read(args.collected), read(args.events), read(args.config))
    if args.whatif:
        report["whatif"] = read(args.whatif).get("orders", [])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=1, default=str))
    (args.output / "report.md").write_text(render(report))
    return {"report": str(args.output / "report.md"), "liquidation_shock": report.get("liquidation_shock"),
            "margin_residual": report["margin_fit"]["residual"]}

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
