"""Bit-exact output fingerprints, to compare two interpreters or Polars versions.

`dump` runs a fixed set of workloads and writes every result table as canonical
JSON: floats are written with `float.hex()` so two values compare equal only if
they are the same IEEE-754 double. A `manifest.json` records a SHA-256 per table.

`compare` loads two dump directories and reports, per table, whether it is
identical and, if not, the schema difference, the number of differing cells and
the largest absolute and relative error among float columns.

    python benchmarks/output_fingerprint.py dump --output-dir out/polars144
    python benchmarks/output_fingerprint.py dump --output-dir out/polars21   # in the other venv
    python benchmarks/output_fingerprint.py compare --left out/polars144 --right out/polars21

The Polars thread count is pinned to one inside this script (set before import),
so repeated runs of one build are bit-stable. Run-manifest fields that describe the environment rather than the result
(`python_version`, `platform`, `package_version`) and the engine-source hash that
derives from it (`engine_code_sha256`, `run_id`) are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

# One worker thread: multi-threaded float reductions sum in a run-dependent order,
# which changes the last bit of some results between two runs of the same build.
os.environ["POLARS_MAX_THREADS"] = "1"

import polars as pl  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# `engine_code_sha256` hashes this package's own sources, and `run_id` is derived
# from it, so an edit to the engine (such as the Polars upgrade itself) changes
# both by design. Their inputs (data hash, broker, strategy, metadata) are still
# compared through the remaining manifest fields.
ENVIRONMENT_MANIFEST_KEYS = {"python_version", "platform", "package_version", "engine_code_sha256", "run_id"}


def _encode(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value.hex()
    if isinstance(value, int):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    return repr(value)


def frame_payload(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
        "height": frame.height,
        "columns": {name: [_encode(v) for v in frame[name].to_list()] for name in frame.columns},
    }


def _sha(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _write(directory: Path, name: str, payload: Any, index: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    index[name] = _sha(payload)


def _frames_of(obj: Any, prefix: str) -> dict[str, pl.DataFrame]:
    out: dict[str, pl.DataFrame] = {}
    for attr in sorted(dir(obj)):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(obj, attr)
        except Exception:  # noqa: BLE001 - some properties may need a non-empty state
            continue
        if isinstance(value, pl.DataFrame):
            out[f"{prefix}.{attr}"] = value
    return out


def dump_factor_workload(directory: Path, index: dict[str, str]) -> None:
    from cortexbacktest import run_factor_research
    from factor_pipeline_perf import FACTOR_NAMES, build_panel

    prices, factors = build_panel(symbols=300, dates=800, seed=7)
    for name in FACTOR_NAMES:
        report = run_factor_research(
            prices,
            factors[name],
            horizons=(1, 5, 20),
            quantiles=5,
            top_quantile=0.2,
            bottom_quantile=0.2,
            transaction_cost_bps=5.0,
        )
        for attr, frame in _frames_of(report, f"factor.{name}").items():
            _write(directory, attr, frame_payload(frame), index)
        for horizon, result in sorted(report.horizon_results.items()):
            for attr, frame in _frames_of(result, f"factor.{name}.h{horizon}").items():
                _write(directory, attr, frame_payload(frame), index)
        _write(directory, f"factor.{name}.summary", frame_payload(report.summary()), index)


def dump_event_workload(directory: Path, index: dict[str, str]) -> None:
    from cortexbacktest import Bar
    from cortexbacktest.api import BacktestConfig, BrokerConfig, run_backtest
    from cortexbacktest.strategy_library import SmaCross

    from factor_pipeline_perf import build_panel

    prices, _ = build_panel(symbols=40, dates=500, seed=11)
    symbols = [c for c in prices.columns if c != "date"]
    bars: list[Bar] = []
    for row in prices.iter_rows(named=True):
        ts = datetime(row["date"].year, row["date"].month, row["date"].day, tzinfo=_utc())
        for symbol in symbols:
            close = row[symbol]
            if close is None or math.isnan(close):
                continue
            bars.append(Bar(symbol, ts, close, close * 1.01, close * 0.99, close, 1e6))
    config = BacktestConfig(broker=BrokerConfig(initial_cash=1_000_000.0, commission_bps=1.0, slippage_bps=2.0))
    result = run_backtest(bars, SmaCross(fast=10, slow=40), config=config, metadata={"workload": "fingerprint"})
    for attr, frame in _frames_of(result, "event").items():
        _write(directory, attr, frame_payload(frame), index)
    summary = result.summary()
    _write(directory, "event.summary", _encode(summary), index)
    manifest = dict(result.run_manifest or {})
    for key in ENVIRONMENT_MANIFEST_KEYS:
        manifest.pop(key, None)
    _write(directory, "event.run_manifest", _encode(manifest), index)


def _utc():
    from datetime import UTC

    return UTC


def cmd_dump(args: argparse.Namespace) -> int:
    directory: Path = args.output_dir
    index: dict[str, str] = {}
    dump_factor_workload(directory, index)
    dump_event_workload(directory, index)
    (directory / "manifest.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(index)} tables to {directory}")
    return 0


def _decode(value: Any) -> Any:
    """Invert `_encode` for floats; other values pass through unchanged."""
    if isinstance(value, str):
        if value in ("nan", "inf", "-inf"):
            return float(value)
        if value.startswith(("0x", "-0x")):
            return float.fromhex(value)
    return value


def cmd_compare(args: argparse.Namespace) -> int:
    """Exit 0 only if every table matches within the given tolerances.

    A cell matches when |x - y| <= abs_tol + rel_tol * max(|x|, |y|). With both
    tolerances at zero the comparison is bit-exact (NaN matches NaN).
    """
    left = json.loads((args.left / "manifest.json").read_text())
    right = json.loads((args.right / "manifest.json").read_text())
    names = sorted(set(left) | set(right))
    identical = 0
    within = 0
    problems = 0
    for name in names:
        if name not in left or name not in right:
            print(f"MISSING  {name}  ({'right' if name not in left else 'left'} only)")
            problems += 1
            continue
        if left[name] == right[name]:
            identical += 1
            continue
        a = json.loads((args.left / f"{name}.json").read_text())
        b = json.loads((args.right / f"{name}.json").read_text())
        if "columns" not in a or "columns" not in b:
            print(f"DIFFERS  {name}: non-table payload")
            print(f"           left : {json.dumps(a, sort_keys=True)[:400]}")
            print(f"           right: {json.dumps(b, sort_keys=True)[:400]}")
            continue
        if a["schema"] != b["schema"]:
            print(f"SCHEMA   {name}: {a['schema']} vs {b['schema']}")
            continue
        differing = 0
        beyond = 0
        max_abs = 0.0
        max_rel = 0.0
        for col in a["columns"]:
            for x_raw, y_raw in zip(a["columns"][col], b["columns"][col]):
                x, y = _decode(x_raw), _decode(y_raw)
                if x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)):
                    continue
                differing += 1
                if isinstance(x, float) and isinstance(y, float):
                    diff = abs(x - y)
                    max_abs = max(max_abs, diff)
                    if x != 0:
                        max_rel = max(max_rel, diff / abs(x))
                    if diff > args.abs_tol + args.rel_tol * max(abs(x), abs(y)):
                        beyond += 1
                else:
                    beyond += 1
        if beyond:
            problems += 1
            print(f"FAIL     {name}: {beyond} of {differing} differing cells beyond tolerance, "
                  f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
        else:
            within += 1
            print(f"WITHIN   {name}: {differing} cells differ, all within tolerance, "
                  f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
    print(f"\n{identical} bit-identical, {within} within tolerance, {problems} failing, "
          f"of {len(names)} tables (abs_tol={args.abs_tol}, rel_tol={args.rel_tol})")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    dump = sub.add_parser("dump")
    dump.add_argument("--output-dir", type=Path, required=True)
    dump.set_defaults(func=cmd_dump)
    compare = sub.add_parser("compare")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--abs-tol", type=float, default=0.0)
    compare.add_argument("--rel-tol", type=float, default=0.0)
    compare.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
