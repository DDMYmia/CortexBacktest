"""JSON-safe report builders for backtest results."""

from __future__ import annotations

import json
import math
from datetime import timedelta
from html import escape
from pathlib import Path
from typing import Any


import polars as pl
import polars.selectors as cs

from .analyzers import trade_summary
from .records import fills_frame
from .trades import PNL_RECONCILIATION_COLUMNS, trade_stats, trades_frame


def event_report(
    result,
    *,
    include_records: bool = True,
    benchmark_equity: pl.DataFrame | None = None,
    benchmark_rolling_window: int = 20,
) -> dict[str, Any]:
    """Build a JSON-safe report dictionary from an event backtest result."""
    performance = result.performance.report()
    trades = result.trades.frame() if result.records_retained else trades_frame([])
    fills = result.records.fills() if result.records_retained else fills_frame([])
    pnl = result.trades.pnl_summary()
    report: dict[str, Any] = {
        "summary": _json_value(result.summary()),
        "performance": _json_value(performance.summary),
        "trades": _json_value(trade_stats(trades)),
        "fills": _json_value(trade_summary(fills)),
        "pnl_reconciliation": _json_value(pnl),
        "counts": {
            "equity_points": int(result.performance.equity_curve().height),
            "orders": int(result.order_record_count),
            "open_orders": int(result.open_order_count),
            "fills": int(result.fill_count),
            "round_trips": trades.height,
        },
        "records_retained": bool(result.records_retained),
    }
    if getattr(result, "algo_orders", None):
        report["algo_orders"] = _json_value(result.algo_orders)
    run_manifest = getattr(result, "run_manifest", None)
    if run_manifest is not None:
        report["run_manifest"] = _json_value(run_manifest)
    if include_records:
        report["records"] = {
            "equity_curve": frame_records(result.performance.equity_curve()),
            "orders": frame_records(result.records.orders()),
            "fills": frame_records(fills),
            "open_orders": frame_records(result.records.open_orders()),
            "positions": frame_records(result.records.positions()),
            "exposure": frame_records(result.exposure.frame()),
            "turnover": frame_records(result.exposure.turnover()),
            "trades": frame_records(trades),
            "pnl_reconciliation": frame_records(pl.DataFrame([pnl]).select(PNL_RECONCILIATION_COLUMNS)),
            "drawdowns": frame_records(result.performance.drawdowns()),
        }
    if benchmark_equity is not None:
        benchmark = result.performance.benchmark(
            benchmark_equity,
            rolling_window=benchmark_rolling_window,
        )
        report["benchmark"] = {
            "summary": _json_value(benchmark.summary),
        }
        if include_records:
            report["benchmark"]["records"] = {
                "aligned_returns": frame_records(benchmark.aligned_returns),
                "active_returns": frame_records(benchmark.active_returns),
                "rolling_beta": frame_records(benchmark.rolling_beta),
                "rolling_sharpe": frame_records(benchmark.rolling_sharpe),
            }
    return report


def write_event_report_json(
    result,
    path: str | Path,
    *,
    include_records: bool = True,
    indent: int = 2,
    benchmark_equity: pl.DataFrame | None = None,
    benchmark_rolling_window: int = 20,
) -> Path:
    """Write a JSON report and return the output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = event_report(
        result,
        include_records=include_records,
        benchmark_equity=benchmark_equity,
        benchmark_rolling_window=benchmark_rolling_window,
    )
    output_path.write_text(json.dumps(report, indent=indent, sort_keys=True), encoding="utf-8")
    return output_path


def event_report_html(
    result,
    *,
    include_records: bool = True,
    max_rows: int = 50,
    benchmark_equity: pl.DataFrame | None = None,
    benchmark_rolling_window: int = 20,
) -> str:
    """Render a standalone HTML report for an event backtest result."""
    report = event_report(
        result,
        include_records=include_records,
        benchmark_equity=benchmark_equity,
        benchmark_rolling_window=benchmark_rolling_window,
    )
    sections = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>CortexBacktest Report</title>",
        f"<style>{_REPORT_CSS}</style>",
        "</head>",
        "<body>",
        "<main>",
        "<header><h1>CortexBacktest Report</h1></header>",
        _metric_cards(report),
        _section("Summary", _mapping_table(report["summary"])),
        _section("Performance", _mapping_table(report["performance"])),
        _section("Trade Statistics", _mapping_table(report["trades"])),
        _section("Fill Summary", _mapping_table(report["fills"])),
        _section("PnL Reconciliation", _mapping_table(report["pnl_reconciliation"])),
    ]
    if "run_manifest" in report:
        sections.append(_section("Run Manifest", _mapping_table(report["run_manifest"])))
    if "benchmark" in report:
        sections.append(_section("Benchmark", _mapping_table(report["benchmark"]["summary"])))
    if include_records and "records" in report:
        records = report["records"]
        sections.extend(
            [
                _section("Equity Curve", _records_table(records["equity_curve"], max_rows=max_rows)),
                _section("Orders", _records_table(records["orders"], max_rows=max_rows)),
                _section("Fills", _records_table(records["fills"], max_rows=max_rows)),
                _section("Positions", _records_table(records["positions"], max_rows=max_rows)),
                _section("Exposure", _records_table(records["exposure"], max_rows=max_rows)),
                _section("Turnover", _records_table(records["turnover"], max_rows=max_rows)),
                _section("Round Trips", _records_table(records["trades"], max_rows=max_rows)),
                _section("PnL Reconciliation Records", _records_table(records["pnl_reconciliation"], max_rows=max_rows)),
                _section("Drawdowns", _records_table(records["drawdowns"], max_rows=max_rows)),
            ]
        )
    if include_records and "benchmark" in report and "records" in report["benchmark"]:
        benchmark_records = report["benchmark"]["records"]
        sections.extend(
            [
                _section("Benchmark Aligned Returns", _records_table(benchmark_records["aligned_returns"], max_rows=max_rows)),
                _section("Benchmark Active Returns", _records_table(benchmark_records["active_returns"], max_rows=max_rows)),
                _section("Benchmark Rolling Beta", _records_table(benchmark_records["rolling_beta"], max_rows=max_rows)),
                _section("Benchmark Rolling Sharpe", _records_table(benchmark_records["rolling_sharpe"], max_rows=max_rows)),
            ]
        )
    sections.extend(["</main>", "</body>", "</html>"])
    return "\n".join(sections)


def write_event_report_html(
    result,
    path: str | Path,
    *,
    include_records: bool = True,
    max_rows: int = 50,
    benchmark_equity: pl.DataFrame | None = None,
    benchmark_rolling_window: int = 20,
) -> Path:
    """Write a standalone HTML report and return the output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        event_report_html(
            result,
            include_records=include_records,
            max_rows=max_rows,
            benchmark_equity=benchmark_equity,
            benchmark_rolling_window=benchmark_rolling_window,
        ),
        encoding="utf-8",
    )
    return output_path


def frame_records(frame: pl.DataFrame | pl.Series) -> list[dict[str, Any]]:
    """Convert a Polars DataFrame/Series to JSON-safe row records."""
    if isinstance(frame, pl.Series):
        frame = pl.DataFrame({frame.name or "value": frame})
    if frame.is_empty():
        return []
    # Numeric columns need only a non-finite mask. Dates, durations and nested
    # values keep the same Python conversion without visiting every scalar.
    frame = frame.with_columns(pl.when(cs.float().is_finite()).then(cs.float()).otherwise(None))
    records = frame.to_dicts()
    for column in frame.select(cs.temporal() | cs.nested() | cs.object()).columns:
        for row in records:
            row[column] = _json_value(row[column])
    return records


def _json_value(value):
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, timedelta):
        return _iso_duration(value)
    if hasattr(value, "isoformat") and value.__class__.__module__ == "datetime":
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value


def _iso_duration(delta: timedelta) -> str:
    """Render a timedelta as an ISO 8601 duration (e.g. P1DT2H)."""
    total_seconds = int(delta.total_seconds())
    sign = "-" if total_seconds < 0 else ""
    total_seconds = abs(total_seconds)
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    date_part = f"{days}D" if days else ""
    time_part = ""
    if hours or minutes or seconds:
        time_part = "T" + (f"{hours}H" if hours else "") + (f"{minutes}M" if minutes else "") + (f"{seconds}S" if seconds else "")
    body = date_part + time_part or "T0S"
    return f"{sign}P{body}"


def _metric_cards(report: dict[str, Any]) -> str:
    metrics = [
        ("Final Equity", report["summary"].get("final_equity")),
        ("Total Return", report["summary"].get("total_return")),
        ("Sharpe", report["summary"].get("sharpe")),
        ("Max Drawdown", report["summary"].get("max_drawdown")),
        ("Fills", report["counts"].get("fills")),
        ("Round Trips", report["counts"].get("round_trips")),
    ]
    cards = "\n".join(
        f"<article class=\"metric\"><span>{escape(label)}</span><strong>{escape(_format_value(value))}</strong></article>"
        for label, value in metrics
    )
    return f"<section class=\"metrics\">{cards}</section>"


def _section(title: str, body: str) -> str:
    return f"<section><h2>{escape(title)}</h2>{body}</section>"


def _mapping_table(mapping: dict[str, Any]) -> str:
    rows = "\n".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(_format_value(value))}</td></tr>"
        for key, value in mapping.items()
    )
    return f"<table class=\"mapping\"><tbody>{rows}</tbody></table>"


def _records_table(rows: list[dict[str, Any]], *, max_rows: int) -> str:
    if not rows:
        return "<p class=\"empty\">No records.</p>"
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    visible_rows = rows[:max_rows]
    columns = list(visible_rows[0])
    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body = "\n".join(
        "<tr>"
        + "".join(f"<td>{escape(_format_value(row.get(column)))}</td>" for column in columns)
        + "</tr>"
        for row in visible_rows
    )
    note = "" if len(rows) <= max_rows else f"<p class=\"note\">Showing {max_rows} of {len(rows)} rows.</p>"
    return f"{note}<div class=\"table-wrap\"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if abs(value) < 1 and value != 0:
            return f"{value:.4%}"
        return f"{value:,.4f}"
    return str(value)


_REPORT_CSS = """
:root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; background: #f6f7f9; color: #1f2933; }
main { max-width: 1180px; margin: 0 auto; padding: 32px; }
header { margin-bottom: 24px; }
h1 { margin: 0; font-size: 28px; }
h2 { margin: 0 0 12px; font-size: 18px; }
section { margin: 18px 0; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.metric { background: white; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; }
.metric span { display: block; color: #64748b; font-size: 12px; margin-bottom: 8px; }
.metric strong { font-size: 18px; }
table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9dee7; }
th, td { padding: 9px 10px; border-bottom: 1px solid #e5e9f0; text-align: left; font-size: 13px; }
th { background: #eef2f7; font-weight: 600; }
.mapping th { width: 260px; }
.table-wrap { overflow-x: auto; }
.empty, .note { color: #64748b; font-size: 13px; }
""".strip()
