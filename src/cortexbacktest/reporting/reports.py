"""JSON and HTML exports for research and sweep reports."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import polars as pl

from .charts import chart_styles, drawdown_chart, equity_chart, exposure_chart, return_histogram
from .summary import _format_value, _mapping_table, _records_table, _section, frame_records


def research_report(report, *, include_records: bool = True) -> dict[str, Any]:
    """Build a JSON-safe report dictionary from a FactorResearchReport."""
    result: dict[str, Any] = {
        "summary": frame_records(report.summary()),
        "coverage_mean": _json_value(report.coverage_mean),
        "turnover_mean": _json_value(report.turnover_mean),
        "final_equity": _json_value(report.final_equity),
    }
    if include_records:
        result["records"] = {
            "coverage": frame_records(report.coverage()),
            "turnover": frame_records(report.turnover()),
            "target_weights": frame_records(report.target_weights_frame()),
            "equity_curve": frame_records(report.equity_curve_frame()),
        }
    return result


def factor_sweep_report(result, *, include_records: bool = True) -> dict[str, Any]:
    """Build a JSON-safe report dictionary from a FactorSweepResult."""
    report: dict[str, Any] = {
        "summary": frame_records(result.summary()),
        "stability": frame_records(result.stability()),
        "best_case": result.best_case(),
    }
    if include_records:
        report["cases"] = {
            name: research_report(item, include_records=False)
            for name, item in result.reports.items()
        }
    return report


def execution_sweep_report(result, *, include_records: bool = True) -> dict[str, Any]:
    """Build a JSON-safe report dictionary from an ExecutionSweepResult."""
    report: dict[str, Any] = {
        "summary": frame_records(result.summary()),
        "best_case": result.best_case(),
    }
    if include_records:
        report["event_results"] = {
            name: event_result.report(include_records=False)
            for name, event_result in result.event_results.items()
        }
    return report


def equity_curve_charts(curve) -> list[str]:
    """Equity, drawdown, exposure and return-distribution SVGs from one curve.

    Takes anything with the equity-curve columns -- a Polars frame, or a mapping
    of column name to sequence -- so a caller that already has the frame does not
    have to convert it, and a caller that only has lists does not need Polars.
    Missing mapping keys omit charts; missing Polars columns may still raise.
    """
    def column(name: str):
        try:
            values = curve[name]
        except (KeyError, TypeError, IndexError):
            return None
        return list(values) if values is not None else None

    stamps = column("timestamp") or []
    equity = column("equity")
    if not equity:
        return []
    charts = [equity_chart(stamps, equity), drawdown_chart(stamps, equity)]
    long_exposure, short_exposure = column("long_exposure"), column("short_exposure")
    if long_exposure:
        charts.append(exposure_chart(stamps, long_exposure, short_exposure or []))
    returns = [
        (equity[index] - equity[index - 1]) / equity[index - 1]
        for index in range(1, len(equity))
        if equity[index - 1]
    ]
    if returns:
        charts.append(return_histogram(returns))
    return charts


def report_html(title: str, report: dict[str, Any], *, max_rows: int = 50,
                charts: Sequence[str] | None = None) -> str:
    """Render a standalone HTML report from a report dictionary.

    `charts` are SVG fragments placed above the tables; `equity_curve_charts`
    builds the usual set. They are inline SVG with no plotting dependency and no
    script, so the file renders anywhere and hashes identically for identical
    input.
    """
    sections = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>{escape(title)}</title>",
        f"<style>{_REPORT_CSS}\n{_CHART_LAYOUT_CSS}\n{chart_styles()}</style>",
        "</head>",
        "<body><main>",
        f"<header><h1>{escape(title)}</h1></header>",
    ]
    if charts:
        sections.append('<section class="cb-charts">' + "".join(charts) + "</section>")
    for key, value in report.items():
        sections.append(_render_section(str(key), value, max_rows=max_rows))
    sections.extend(["</main></body>", "</html>"])
    return "\n".join(sections)


def write_report_json(report: dict[str, Any], path: str | Path, *, indent: int = 2) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=indent, sort_keys=True), encoding="utf-8")
    return output_path


def write_report_html(title: str, report: dict[str, Any], path: str | Path, *,
                      max_rows: int = 50, charts: Sequence[str] | None = None) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_html(title, report, max_rows=max_rows, charts=charts), encoding="utf-8")
    return output_path


def write_research_report_json(report, path: str | Path, *, include_records: bool = True) -> Path:
    return write_report_json(research_report(report, include_records=include_records), path)


def write_research_report_html(report, path: str | Path, *, include_records: bool = True) -> Path:
    return write_report_html(
        "CortexBacktest - Research Report",
        research_report(report, include_records=include_records),
        path,
    )


def write_factor_sweep_report_json(result, path: str | Path, *, include_records: bool = True) -> Path:
    return write_report_json(factor_sweep_report(result, include_records=include_records), path)


def write_factor_sweep_report_html(result, path: str | Path, *, include_records: bool = True) -> Path:
    return write_report_html(
        "CortexBacktest - Factor Sweep Report",
        factor_sweep_report(result, include_records=include_records),
        path,
    )


def write_execution_sweep_report_json(result, path: str | Path, *, include_records: bool = True) -> Path:
    return write_report_json(execution_sweep_report(result, include_records=include_records), path)


def write_execution_sweep_report_html(result, path: str | Path, *, include_records: bool = True) -> Path:
    return write_report_html(
        "CortexBacktest - Execution Sweep Report",
        execution_sweep_report(result, include_records=include_records),
        path,
    )


def _render_section(title: str, value: Any, *, max_rows: int) -> str:
    if isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            return _section(title, _records_table(value, max_rows=max_rows))
        return _section(title, f"<pre>{escape(json.dumps(value, indent=2, default=str))}</pre>")
    if isinstance(value, dict):
        if all(not isinstance(item, (list, dict)) for item in value.values()):
            return _section(title, _mapping_table(value))
        nested = "\n".join(_render_section(f"{title}.{key}", item, max_rows=max_rows) for key, item in value.items())
        return nested
    return _section(title, f"<p>{escape(_format_value(value))}</p>")


def _json_value(value):
    return frame_records(pl.DataFrame({"value": [value]}))[0]["value"]


_CHART_LAYOUT_CSS = (
    ".cb-charts{display:grid;gap:18px;margin:0 0 26px;}"
)
_REPORT_CSS = """
:root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; background: #f6f7f9; color: #1f2933; }
main { max-width: 1180px; margin: 0 auto; padding: 32px; }
header { margin-bottom: 24px; }
h1 { margin: 0; font-size: 28px; }
h2 { margin: 20px 0 12px; font-size: 18px; }
table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9dee7; }
th, td { padding: 9px 10px; border-bottom: 1px solid #e5e9f0; text-align: left; font-size: 13px; }
th { background: #eef2f7; font-weight: 600; }
.mapping th { width: 260px; }
.table-wrap { overflow-x: auto; }
.empty, .note { color: #64748b; font-size: 13px; }
""".strip()
