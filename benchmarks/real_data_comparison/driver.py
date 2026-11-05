"""Run every engine over every scenario and shape; write results.csv and report.md.

    python driver.py --out ../../benchmark-results/competitors-2026-11-05

Rules, applied to all engines alike:
  * Each run is a fresh subprocess under `/usr/bin/time -l`, so no engine benefits
    from another's warm caches or JIT state, and peak RSS is per run (it includes
    loading the dataset).
  * Shapes run smallest-first per engine. If the last measured rate projects
    the next shape past BUDGET seconds, the run is skipped and the projection
    is recorded instead of a number. A skip is reported, never hidden.
  * Every SMA run is checked against the order count the signal implies; more
    than 1% off and the timing is thrown away as INVALID (an engine that aborts
    partway looks fast).
  * Every SMA run that states a fill convention is checked against the plain
    ledger for that convention (`common.truth_pnl`). A difference over $0.01 is
    a mismatch unless KNOWN_DIFFERENCES explains it; any unexplained mismatch
    makes the driver exit non-zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import common

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / ".venv/bin/python"
BUDGET = 300.0
FUSION_CONVENTIONS = {"fusion": common.NEXT_OPEN, "fusion (next-close)": common.NEXT_CLOSE,
                      "fusion (same-bar close)": common.SAME_CLOSE}


def _venv(name: str) -> Path:
    return HERE / f"venv_{name}/bin/python"


ENGINES = {  # name -> (interpreter, script, extra args)
    **{name: (PROJECT, "run_fusion.py", ("--convention", conv)) for name, conv in FUSION_CONVENTIONS.items()},
    "backtrader": (_venv("backtrader"), "run_backtrader.py", ()),
    "backtesting.py": (_venv("backtestingpy"), "run_backtesting_py.py", ()),
    "vectorbt": (_venv("vectorbt"), "run_vectorbt.py", ()),
    "pybroker": (_venv("pybroker"), "run_pybroker.py", ()),
    "zipline-reloaded": (_venv("zipline"), "run_zipline.py", ()),
    "vnpy": (_venv("vnpy"), "run_vnpy.py", ()),
    "lumibot": (_venv("lumibot"), "run_lumibot.py", ()),
    "nautilus_trader": (_venv("nautilus"), "run_nautilus.py", ()),
    "bt": (_venv("bt"), "run_bt.py", ()),
    "qstrader": (_venv("qstrader"), "run_qstrader.py", ()),
    "pyalgotrade": (_venv("pyalgotrade"), "run_pyalgotrade.py", ()),
    "cipher-bt": (_venv("cipher"), "run_cipher.py", ()),
    "freqtrade": (_venv("freqtrade"), "run_freqtrade.py", ()),
    "qlib": (_venv("qlib"), "run_qlib.py", ()),
    "jesse": (_venv("jesse"), "run_jesse.py", ()),
}
SINGLE_ASSET_ONLY = {"backtesting.py", "vnpy", "cipher-bt"}
# Measured, explained differences from the ledger of the engine's convention.
KNOWN_DIFFERENCES = {
    "nautilus_trader": "instrument prices have 2-decimal precision; bars with finer prices are rounded",
    "zipline-reloaded": "the daily bundle stores prices at 0.001 precision; finer prices are rounded",
    "qlib": "its binary data format stores prices as float32",
}

# (title, scenario, dataset, symbols, bars)
PLAN = [
    ("Recent real data, single asset", "sma", "recent.npz", 1, 563),
    ("Recent real data, multi asset", "sma", "recent.npz", 34, 563),
    ("Resting limit orders, single asset", "resting", "recent.npz", 1, 563),
    ("Resting limit orders, multi asset", "resting", "recent.npz", 34, 563),
    ("Multi-timeframe strategy", "mtf", "recent.npz", 1, 563),
    ("Multi-timeframe strategy", "mtf", "recent.npz", 34, 563),
    ("Multi-timeframe strategy", "mtf", "wide.npz", 100, 1000),
    ("Multi-timeframe strategy", "mtf", "deep.npz", 596, 2932),
    ("Vertical scale", "sma", "deep.npz", 1, 2932),
    ("Vertical scale", "sma", "wide.npz", 100, 1000),
    ("Vertical scale", "sma", "deep.npz", 100, 2932),
    ("Vertical scale", "sma", "wide.npz", 500, 1000),
    ("Vertical scale", "sma", "wide.npz", 643, 1000),
    ("Vertical scale", "sma", "deep.npz", 596, 2932),
]
# Sections whose shapes grow; a slow engine skips the larger ones by projection.
SCALED = {"Vertical scale", "Multi-timeframe strategy"}
FIELDS = ["section", "scenario", "dataset", "symbols", "bars_per_symbol", "bar_observations", "engine",
          "convention", "seconds", "observations_per_second", "peak_rss_mb", "orders", "fills", "pnl",
          "truth_pnl", "pnl_diff", "note"]


def run(engine: str, scenario: str, dataset: str, symbols: int, bars: int) -> dict:
    python, script, extra = ENGINES[engine]
    proc = subprocess.run(
        ["/usr/bin/time", "-l", str(python), str(HERE / script), "--dataset", dataset,
         "--symbols", str(symbols), "--bars", str(bars), "--scenario", scenario, *extra],
        capture_output=True, text=True, cwd=HERE,
    )
    rss = re.search(r"(\d+)\s+maximum resident set size", proc.stderr)
    peak = round(int(rss.group(1)) / 2**20, 1) if rss else None  # macOS reports bytes
    if proc.returncode == 0:
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return {**json.loads(line), "peak_rss_mb": peak}
                except json.JSONDecodeError:
                    continue
    errors = [line for line in proc.stderr.strip().splitlines() if "maximum resident" not in line
              and not re.match(r"\s+\d+\s", line)]
    return {"error": (errors or [f"no result (exit {proc.returncode})"])[-1][:200]}


_TARGETS: dict[tuple, object] = {}


def truth(scenario: str, dataset: str, symbols: int, bars: int, convention: str) -> float | None:
    if scenario not in ("sma", "mtf") or convention not in common.CONVENTIONS:
        return None
    _, ts, cols = common.load(dataset, symbols, bars)
    if scenario == "sma":
        return common.truth_pnl(cols, common.signals(cols["close"]), convention)
    key = (dataset, symbols, bars)
    if key not in _TARGETS:  # seconds of pure Python per shape; compute once per run
        _TARGETS[key] = common.mtf_targets(ts, cols)
    return common.truth_pnl(cols, _TARGETS[key], convention)


def measure(engine: str, scenarios: set[str] | None = None) -> list[dict]:
    rows, rate = [], {}
    run(engine, "sma", "recent.npz", 1, 100)  # discard: first interpreter start pages in .pyc
    for section, scenario, dataset, symbols, bars in PLAN:
        if scenarios and scenario not in scenarios:
            continue
        obs = symbols * bars
        row = {"section": section, "scenario": scenario, "dataset": dataset, "symbols": symbols,
               "bars_per_symbol": bars, "bar_observations": obs, "engine": engine}
        if engine in SINGLE_ASSET_ONLY and symbols != 1:
            rows.append({**row, "note": "not run: single-asset engine"})
            continue
        if section in SCALED and rate.get(section) and obs / rate[section] > BUDGET:
            rows.append({**row, "note": f"skipped: projected {obs / rate[section]:.0f}s over {BUDGET:.0f}s budget"})
            print(f"  {engine:24} {symbols:4}x{bars:<5} SKIP (proj {obs / rate[section]:.0f}s)", flush=True)
            continue
        out = run(engine, scenario, dataset, symbols, bars)
        if "unsupported" in out:
            rows.append({**row, "note": f"unsupported: {out['unsupported']}"})
            continue
        if "error" in out:
            rows.append({**row, "note": f"failed: {out['error']}"})
            print(f"  {engine:24} {scenario} {symbols}x{bars} FAIL {out['error'][:80]}", flush=True)
            continue
        got, exp = out["orders"], out.get("expected_orders")
        if exp is not None and abs(got - exp) > max(1, 0.01 * exp):
            rows.append({**row, "orders": got, "note": f"INVALID: {got} orders vs {exp} expected"})
            continue
        if section in SCALED:
            rate[section] = out["observations_per_second"]
        ledger = truth(scenario, dataset, symbols, bars, out.get("convention", ""))
        diff = None if ledger is None or out.get("pnl") is None else round(out["pnl"] - ledger, 4)
        note = out["note"]
        if diff is not None and abs(diff) > 0.01:
            reason = KNOWN_DIFFERENCES.get(engine)
            note = f"{note}; P&L {diff:+,.2f} vs ledger: {reason}" if reason else f"{note}; PNL MISMATCH {diff:+,.2f}"
        rows.append({**row, "convention": out.get("convention", ""), "seconds": out["seconds"],
                     "observations_per_second": out["observations_per_second"],
                     "peak_rss_mb": out["peak_rss_mb"], "orders": got, "fills": out.get("fills"),
                     "pnl": out.get("pnl"), "truth_pnl": ledger, "pnl_diff": diff, "note": note})
        print(f"  {engine:24} {scenario:7} {symbols:4}x{bars:<5} {out['seconds']:9.3f}s pnl={out.get('pnl')} "
              f"diff={diff}", flush=True)
    return rows


def _fmt(value, digits=0):
    if value in (None, ""):
        return ""
    return f"{value:,.{digits}f}" if isinstance(value, (int, float)) else str(value)


def report(rows: list[dict]) -> str:
    lines = ["# Engine comparison", "",
             "Generated by `benchmarks/real_data_comparison/driver.py`. SMA(20/50), 100 shares, zero costs; "
             "the resting scenario submits a GTC buy limit at 0.97 x close on every bar. "
             "`pnl_diff` is the engine's P&L minus the plain ledger of its own fill convention.", ""]
    keys = []
    for row in rows:
        key = (row["section"], row["scenario"], row["dataset"], row["symbols"], row["bars_per_symbol"])
        if key not in keys:
            keys.append(key)
    for key in keys:
        section, scenario, dataset, symbols, bars = key
        lines += [f"## {section}: {dataset}, {symbols} x {bars} ({scenario})", "",
                  "| Engine | Convention | Seconds | Obs/s | Peak RSS MB | Fills | P&L | vs ledger | Note |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for row in rows:
            if (row["section"], row["scenario"], row["dataset"], row["symbols"], row["bars_per_symbol"]) != key:
                continue
            lines.append(f"| {row['engine']} | {row.get('convention', '')} | {_fmt(row.get('seconds'), 3)} | "
                         f"{_fmt(row.get('observations_per_second'))} | {_fmt(row.get('peak_rss_mb'), 1)} | "
                         f"{_fmt(row.get('fills'))} | {_fmt(row.get('pnl'), 2)} | {_fmt(row.get('pnl_diff'), 2)} | "
                         f"{row.get('note', '')} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--engines", default="", help="Comma-separated subset.")
    parser.add_argument("--scenarios", default="", help="Comma-separated subset: sma, resting, mtf.")
    args = parser.parse_args()
    wanted = [e for e in (args.engines.split(",") if args.engines else ENGINES) if e]
    rows = []
    for engine in wanted:
        if not ENGINES[engine][0].exists():
            print(f"  {engine:24} SKIP (no interpreter at {ENGINES[engine][0]})", flush=True)
            continue
        rows += measure(engine, set(filter(None, args.scenarios.split(","))))
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "report.md").write_text(report(rows))
    mismatches = [r for r in rows if "PNL MISMATCH" in str(r.get("note", ""))]
    print(f"WROTE {args.out / 'results.csv'} and report.md ({len(rows)} rows, {len(mismatches)} mismatches)")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
