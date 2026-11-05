"""Draw the README benchmark charts from a driver results.csv.

    python benchmarks/real_data_comparison/plot_results.py \
        benchmark-results/competitors-2026-11-05/results.csv docs/assets
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ACCENT, CONTEXT = "#2a78d6", "#b8b6ad"
NAMES = {"fusion": "CortexBacktest", "vectorbt": "vectorbt (vectorized)", "bt": "bt (weight-based)"}


def _rows(path: Path, dataset: str, symbols: str, scenario: str, field: str) -> list[tuple[str, float]]:
    out = []
    for row in csv.DictReader(path.open()):
        if (row["dataset"], row["symbols"], row["scenario"]) != (dataset, symbols, scenario) or not row[field]:
            continue
        if row["engine"].startswith("fusion ("):  # alternative conventions; the default row represents it
            continue
        out.append((NAMES.get(row["engine"], row["engine"]), float(row[field])))
    return out


def _human(value: float) -> str:
    for unit, scale in (("M", 1e6), ("k", 1e3)):
        if value >= scale:
            return f"{value / scale:.1f}{unit}" if value < 10 * scale else f"{value / scale:.0f}{unit}"
    return f"{value:.2f}" if value < 10 else f"{value:.0f}"


def bar_chart(rows, *, title: str, subtitle: str, unit: str, out: Path, higher_is_better: bool) -> None:
    rows = sorted(rows, key=lambda item: item[1], reverse=not higher_is_better)[::-1]
    names, values = [r[0] for r in rows], [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8.0, 0.42 * len(rows) + 1.3), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    colors = [ACCENT if n == "CortexBacktest" else CONTEXT for n in names]
    ax.barh(names, values, color=colors, height=0.62, edgecolor=SURFACE, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(min(values) / 2, max(values) * 4)
    for y, (name, value) in enumerate(zip(names, values)):
        ax.text(value * 1.12, y, f"{_human(value)} {unit}", va="center", fontsize=8.5,
                color=INK if name == "CortexBacktest" else INK_2,
                fontweight="bold" if name == "CortexBacktest" else "normal")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _human(v)))
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="y", colors=INK, labelsize=9, length=0)
    for label in ax.get_yticklabels():
        if label.get_text() == "CortexBacktest":
            label.set_fontweight("bold")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    fig.text(0.015, 0.97, title, fontsize=12, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 0.97 - 0.32 / fig.get_size_inches()[1], subtitle, fontsize=8.5, color=INK_2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_size_inches()[1]))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(out)


def main() -> None:
    results, out = Path(sys.argv[1]), Path(sys.argv[2])
    bar_chart(_rows(results, "deep.npz", "596", "sma", "observations_per_second"),
              title="Throughput: 596 real tickers x 2,932 daily bars",
              subtitle="SMA(20/50), 100 shares, zero costs; bar observations per second, log scale. Higher is better.",
              unit="bars/s", out=out / "benchmark-throughput.png", higher_is_better=True)
    bar_chart(_rows(results, "recent.npz", "34", "resting", "seconds"),
              title="Resting orders: 19,142 GTC limit orders, 34 tickers",
              subtitle="A buy limit at 0.97 x close on every bar; run time in seconds, log scale. Lower is better.",
              unit="s", out=out / "benchmark-resting.png", higher_is_better=False)
    bar_chart(_rows(results, "deep.npz", "596", "mtf", "seconds"),
              title="Multi-timeframe strategy: 596 tickers x 2,932 daily bars",
              subtitle="Weekly regime + daily RSI/Bollinger/ATR computed in each engine's callback; seconds, log scale. Lower is better.",
              unit="s", out=out / "benchmark-mtf.png", higher_is_better=False)


if __name__ == "__main__":
    main()
