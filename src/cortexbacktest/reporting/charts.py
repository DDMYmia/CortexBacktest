"""Deterministic inline-SVG charts for backtest reports.

No plotting dependency. That is a decision, not an omission: the core stack is
fixed by ADR-0003 (polars, numpy, pyarrow, orjson), matplotlib and plotly are
declared only for the competitor benchmarks, and a report that renders in any
browser without a runtime is worth more here than one that needs a toolkit.

The output is also byte-deterministic for identical input, which matters in a
repository that hashes its artifacts: two runs of the same backtest produce the
same report file.

Every builder returns an SVG fragment. `chart_styles()` returns the one CSS
block they all reference, so a document embedding several charts carries it
once. Reports are light-only: one light palette, never switched by the
viewer's colour scheme.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from html import escape
from typing import Iterable, Sequence

__all__ = [
    "chart_styles",
    "drawdown_chart",
    "equity_chart",
    "exposure_chart",
    "return_histogram",
]

_WIDTH = 720
_HEIGHT = 220
_PAD_LEFT = 64
_PAD_RIGHT = 12
_PAD_TOP = 16
_PAD_BOTTOM = 28


def chart_styles() -> str:
    """CSS for every chart in this module; embed once per document. Light only."""
    return """
.cb-chart { display:block; width:100%; height:auto; font:12px system-ui,-apple-system,sans-serif; }
.cb-chart .cb-axis { stroke:#c9ced6; stroke-width:1; }
.cb-chart .cb-grid { stroke:#eceef2; stroke-width:1; }
.cb-chart .cb-label { fill:#5b6472; }
.cb-chart .cb-line { fill:none; stroke:#2f6feb; stroke-width:1.5;
                     stroke-linejoin:round; stroke-linecap:round; }
.cb-chart .cb-area { fill:rgba(47,111,235,.14); stroke:none; }
.cb-chart .cb-loss { fill:rgba(203,54,54,.16); stroke:none; }
.cb-chart .cb-loss-line { fill:none; stroke:#cb3636; stroke-width:1.4; }
.cb-chart .cb-bar { fill:#2f6feb; }
.cb-chart .cb-bar-neg { fill:#cb3636; }
.cb-chart .cb-short { fill:none; stroke:#cb3636; stroke-width:1.4; stroke-dasharray:4 3; }
.cb-chart .cb-title { fill:#1d2530; font-weight:600; }
""".strip()


def _finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def _nice_bounds(low: float, high: float) -> tuple[float, float]:
    """Pad a range so the line never touches the frame, and never collapse it."""
    if not math.isfinite(low) or not math.isfinite(high):
        return (0.0, 1.0)
    if high - low < 1e-12:
        magnitude = abs(high) if high else 1.0
        return (low - magnitude * 0.05, high + magnitude * 0.05)
    pad = (high - low) * 0.06
    return (low - pad, high + pad)


def _format_value(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"{value / 1e9:.2f}B"
    if magnitude >= 1e6:
        return f"{value / 1e6:.2f}M"
    if magnitude >= 1e3:
        return f"{value / 1e3:.1f}k"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.4g}"


def _format_stamp(value) -> str:
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


def _x_positions(count: int, width: int) -> list[float]:
    inner = width - _PAD_LEFT - _PAD_RIGHT
    if count <= 1:
        return [_PAD_LEFT + inner / 2] * max(count, 0)
    step = inner / (count - 1)
    return [_PAD_LEFT + index * step for index in range(count)]


def _y_scaler(low: float, high: float, height: int):
    inner = height - _PAD_TOP - _PAD_BOTTOM
    span = high - low or 1.0
    return lambda value: _PAD_TOP + inner - (value - low) / span * inner


def _frame(title: str, low: float, high: float, stamps: Sequence, width: int, height: int,
           fmt=_format_value) -> list[str]:
    """Axis, three gridlines with value labels, and first/last date labels."""
    scale = _y_scaler(low, high, height)
    parts = [f'<text class="cb-title" x="{_PAD_LEFT}" y="11">{escape(title)}</text>']
    for fraction in (0.0, 0.5, 1.0):
        value = low + (high - low) * fraction
        y = round(scale(value), 2)
        parts.append(f'<line class="cb-grid" x1="{_PAD_LEFT}" y1="{y}" x2="{width - _PAD_RIGHT}" y2="{y}"/>')
        parts.append(
            f'<text class="cb-label" x="{_PAD_LEFT - 6}" y="{y + 4}" text-anchor="end">'
            f"{escape(fmt(value))}</text>"
        )
    baseline = height - _PAD_BOTTOM
    parts.append(f'<line class="cb-axis" x1="{_PAD_LEFT}" y1="{baseline}" x2="{width - _PAD_RIGHT}" y2="{baseline}"/>')
    if stamps:
        parts.append(
            f'<text class="cb-label" x="{_PAD_LEFT}" y="{height - 8}">{escape(_format_stamp(stamps[0]))}</text>'
        )
        parts.append(
            f'<text class="cb-label" x="{width - _PAD_RIGHT}" y="{height - 8}" text-anchor="end">'
            f"{escape(_format_stamp(stamps[-1]))}</text>"
        )
    return parts


def _svg(body: Sequence[str], title: str, width: int, height: int) -> str:
    return (
        f'<svg class="cb-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(title)}" xmlns="http://www.w3.org/2000/svg">'
        + "".join(body)
        + "</svg>"
    )


def _path(xs: Sequence[float], ys: Sequence[float]) -> str:
    return "M" + " L".join(f"{round(x, 2)},{round(y, 2)}" for x, y in zip(xs, ys))


def equity_chart(timestamps: Sequence, equity: Sequence[float], *,
                 title: str = "Equity", width: int = _WIDTH, height: int = _HEIGHT) -> str:
    """Equity over time, filled to the starting level."""
    values = _finite(equity)
    if len(values) < 2:
        return _empty(title, width, height)
    low, high = _nice_bounds(min(values), max(values))
    scale = _y_scaler(low, high, height)
    xs = _x_positions(len(values), width)
    ys = [scale(value) for value in values]
    body = _frame(title, low, high, timestamps, width, height)
    base = round(scale(values[0]), 2)
    body.append(
        f'<path class="cb-area" d="{_path(xs, ys)} L{round(xs[-1], 2)},{base} L{round(xs[0], 2)},{base} Z"/>'
    )
    body.append(f'<path class="cb-line" d="{_path(xs, ys)}"/>')
    return _svg(body, title, width, height)


def drawdown_chart(timestamps: Sequence, equity: Sequence[float], *,
                   title: str = "Drawdown", width: int = _WIDTH, height: int = _HEIGHT) -> str:
    """Underwater plot: distance below the running peak, as a fraction."""
    values = _finite(equity)
    if len(values) < 2:
        return _empty(title, width, height)
    peak = values[0]
    drawdown = []
    for value in values:
        peak = value if value > peak else peak
        drawdown.append((value - peak) / peak if peak else 0.0)
    # Pad below only: a drawdown is never positive, so the axis tops out at zero.
    low, high = _nice_bounds(min(drawdown), 0.0)[0], 0.0
    scale = _y_scaler(low, high, height)
    xs = _x_positions(len(drawdown), width)
    ys = [scale(value) for value in drawdown]
    body = _frame(title, low, high, timestamps, width, height, fmt=lambda v: f"{v:.1%}")
    zero = round(scale(0.0), 2)
    body.append(
        f'<path class="cb-loss" d="{_path(xs, ys)} L{round(xs[-1], 2)},{zero} L{round(xs[0], 2)},{zero} Z"/>'
    )
    body.append(f'<path class="cb-loss-line" d="{_path(xs, ys)}"/>')
    return _svg(body, title, width, height)


def exposure_chart(timestamps: Sequence, long_exposure: Sequence[float],
                   short_exposure: Sequence[float], *, title: str = "Exposure",
                   width: int = _WIDTH, height: int = _HEIGHT) -> str:
    """Long and short exposure on one axis; short is drawn dashed."""
    longs = _finite(long_exposure)
    shorts = _finite(short_exposure)
    if len(longs) < 2:
        return _empty(title, width, height)
    combined = longs + shorts + [0.0]
    low, high = _nice_bounds(min(combined), max(combined))
    scale = _y_scaler(low, high, height)
    xs = _x_positions(len(longs), width)
    body = _frame(title, low, high, timestamps, width, height)
    body.append(f'<path class="cb-line" d="{_path(xs, [scale(v) for v in longs])}"/>')
    if shorts and any(value for value in shorts):
        body.append(f'<path class="cb-short" d="{_path(_x_positions(len(shorts), width), [scale(v) for v in shorts])}"/>')
    return _svg(body, title, width, height)


def return_histogram(returns: Sequence[float], *, bins: int = 41,
                     title: str = "Return distribution",
                     width: int = _WIDTH, height: int = _HEIGHT) -> str:
    """Histogram of per-period returns; bars below zero are drawn in the loss colour."""
    values = _finite(returns)
    if len(values) < 2:
        return _empty(title, width, height)
    low_value, high_value = min(values), max(values)
    if high_value - low_value < 1e-15:
        return _empty(title, width, height)
    bins = max(5, bins)
    step = (high_value - low_value) / bins
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - low_value) / step))
        counts[index] += 1
    peak = max(counts) or 1
    scale = _y_scaler(0.0, peak * 1.06, height)
    body = _frame(title, 0.0, peak * 1.06, (), width, height)
    inner = width - _PAD_LEFT - _PAD_RIGHT
    bar_width = inner / bins
    baseline = round(scale(0.0), 2)
    for index, count in enumerate(counts):
        if not count:
            continue
        x = _PAD_LEFT + index * bar_width
        y = round(scale(count), 2)
        edge = low_value + index * step
        css = "cb-bar-neg" if edge + step / 2 < 0 else "cb-bar"
        body.append(
            f'<rect class="{css}" x="{round(x + 0.5, 2)}" y="{y}" '
            f'width="{round(max(bar_width - 1.0, 0.5), 2)}" height="{round(baseline - y, 2)}"/>'
        )
    body.append(
        f'<text class="cb-label" x="{_PAD_LEFT}" y="{height - 8}">{escape(_format_value(low_value))}</text>'
    )
    body.append(
        f'<text class="cb-label" x="{width - _PAD_RIGHT}" y="{height - 8}" text-anchor="end">'
        f"{escape(_format_value(high_value))}</text>"
    )
    return _svg(body, title, width, height)


def _empty(title: str, width: int, height: int) -> str:
    """A chart with nothing to draw says so rather than rendering an empty frame."""
    body = [
        f'<text class="cb-title" x="{_PAD_LEFT}" y="11">{escape(title)}</text>',
        f'<text class="cb-label" x="{width / 2}" y="{height / 2}" text-anchor="middle">'
        "not enough data</text>",
    ]
    return _svg(body, title, width, height)
