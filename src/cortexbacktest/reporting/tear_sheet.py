"""Compact tear sheet builders."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from .summary import event_report


def event_tear_sheet(result, *, include_records: bool = False) -> dict[str, Any]:
    """Build a compact event tear sheet from a BacktestResult."""
    base = event_report(result, include_records=include_records)
    sheet = {
        key: base[key]
        for key in (
            "summary", "performance", "trades", "fills", "counts", "records_retained"
        )
    }
    if getattr(result, "algo_orders", None):
        sheet["algo_orders"] = result.algo_orders
    analyzer_results = getattr(result, "analyzer_results", None)
    if analyzer_results:
        sheet["analyzers"] = analyzer_results
    if include_records and "records" in base:
        sheet["records"] = {
            "equity_curve": base["records"]["equity_curve"],
            "drawdowns": base["records"]["drawdowns"],
            "trades": base["records"]["trades"],
        }
    return sheet


def event_tear_sheet_html(result, *, include_records: bool = False, max_rows: int = 25) -> str:
    sheet = event_tear_sheet(result, include_records=include_records)
    sections = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>CortexBacktest Event Tear Sheet</title>",
        f"<style>{_CSS}</style>",
        "</head><body><main>",
        "<header><h1>CortexBacktest Event Tear Sheet</h1></header>",
        _cards(sheet),
        _mapping_section("Performance", sheet["performance"]),
        _mapping_section("Trades", sheet["trades"]),
        _mapping_section("Fills", sheet["fills"]),
        _mapping_section("Counts", sheet["counts"]),
    ]
    if "algo_orders" in sheet:
        sections.append(_table_section("Algorithm order plans", sheet["algo_orders"], max_rows=max_rows))
    if "analyzers" in sheet:
        sections.append(_mapping_section("Analyzers", sheet["analyzers"]))
    if include_records and "records" in sheet:
        for name, rows in sheet["records"].items():
            sections.append(_table_section(name.replace("_", " ").title(), rows, max_rows=max_rows))
    sections.append("</main></body></html>")
    return "\n".join(sections)


def write_event_tear_sheet_html(
    result,
    path: str | Path,
    *,
    include_records: bool = False,
    max_rows: int = 25,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        event_tear_sheet_html(result, include_records=include_records, max_rows=max_rows),
        encoding="utf-8",
    )
    return output_path


def _cards(sheet: dict[str, Any]) -> str:
    summary = sheet["summary"]
    performance = sheet["performance"]
    items = [
        ("Final Equity", summary.get("final_equity")),
        ("Total Return", summary.get("total_return")),
        ("Sharpe", summary.get("sharpe")),
        ("Max Drawdown", performance.get("max_drawdown")),
        ("Fills", sheet["counts"].get("fills")),
        ("Trades", sheet["counts"].get("round_trips")),
    ]
    return "<section class=\"cards\">" + "".join(
        f"<article><span>{escape(label)}</span><strong>{escape(_format(value))}</strong></article>"
        for label, value in items
    ) + "</section>"


def _mapping_section(title: str, mapping: dict[str, Any]) -> str:
    rows = []
    for key, value in mapping.items():
        rows.append(f"<tr><th>{escape(str(key))}</th><td>{escape(_format(value))}</td></tr>")
    return f"<section><h2>{escape(title)}</h2><table><tbody>{''.join(rows)}</tbody></table></section>"


def _table_section(title: str, rows: list[dict[str, Any]], *, max_rows: int) -> str:
    if not rows:
        return f"<section><h2>{escape(title)}</h2><p>No records.</p></section>"
    visible = rows[:max_rows]
    columns = list(rows[0])
    head = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(_format(row.get(column)))}</td>" for column in columns) + "</tr>"
        for row in visible
    )
    return f"<section><h2>{escape(title)}</h2><div class=\"table\"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></section>"


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return ", ".join(f"{key}: {_format(item)}" for key, item in value.items())
    if isinstance(value, list):
        return f"{len(value)} rows"
    if isinstance(value, float):
        if abs(value) < 1 and value != 0:
            return f"{value:.4%}"
        return f"{value:,.4f}"
    return str(value)


_CSS = """
:root { color-scheme: light; }
body { margin: 0; background: #f7f8fa; color: #17202a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 28px; }
h1 { margin: 0 0 18px; font-size: 28px; }
h2 { margin: 22px 0 10px; font-size: 17px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
article { background: white; border: 1px solid #dbe1ea; padding: 12px; }
article span { display: block; color: #667085; font-size: 12px; }
article strong { display: block; margin-top: 5px; font-size: 18px; }
table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #dbe1ea; }
th, td { border-bottom: 1px solid #e7ebf1; padding: 8px 9px; text-align: left; font-size: 13px; vertical-align: top; }
th { background: #eef2f6; width: 230px; }
.table { overflow-x: auto; }
""".strip()
