"""Extreme-scale throughput and peak memory, one engine per subprocess.

`steady_state_compare.py` answers "how fast at 100k bars". This answers "what
happens as the bar count grows", which is a different question: an engine can
hold a flat bars/s rate and still fall over on memory, and an engine that wins
at 100k can lose at 10M if its per-bar retention is linear.

Two things force the subprocess split:

- **Peak RSS is per-process.** Running every engine in one process reports the
  high-water mark of whichever ran worst, attributed to all of them.
- **Extreme sizes kill processes.** A 10M-bar run that exhausts memory must
  cost one row, not the whole sweep.

Engine definitions come from `steady_state_compare.ENGINES` so the two scripts
cannot drift on what "the same work" means. Same method otherwise: build the
payload, warm up, time the run only, best of N.

    python benchmarks/extreme_scaling_compare.py --bars 100000,1000000 --output results.csv

Memory is reported two ways: `peak_rss_mb` is the child's high-water mark
(payload plus engine, since the payload has to exist for the run to happen),
and `peak_rss_over_import_mb` subtracts the RSS measured after imports but
before the payload is built. This difference includes warmup and allocator
history; it is not an isolated live-allocation measurement.

Engines are skipped at a size when the previous size projects a runtime past
`--budget-seconds`, so a slow engine cannot stall the sweep. Skips are recorded
as rows with a reason rather than dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.steady_state_compare import ENGINES, NOT_COMPARABLE  # noqa: E402


def _peak_rss_mb() -> float:
    """Process high-water mark in MB.

    ru_maxrss is bytes on macOS and kilobytes on Linux — the one portability
    trap in this measurement.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def _run_child(engine: str, bars: int, repeats: int) -> int:
    """Time one engine at one size and print a JSON result line."""
    setup, run = ENGINES[engine](bars)
    baseline_rss = _peak_rss_mb()

    run(setup())  # warm up imports, JIT, and any lazy engine state
    best = float("inf")
    for _ in range(repeats):
        payload = setup()
        start = time.perf_counter()
        run(payload)
        best = min(best, time.perf_counter() - start)
        del payload

    peak = _peak_rss_mb()
    print(
        json.dumps(
            {
                "engine": engine,
                "bars": bars,
                "seconds": best,
                "bars_per_second": bars / best,
                "peak_rss_mb": peak,
                "peak_rss_over_import_mb": peak - baseline_rss,
            }
        ),
        flush=True,
    )
    return 0


def _measure(engine: str, bars: int, repeats: int, timeout: float) -> dict:
    """Run one child and parse its result, turning any failure into a row."""
    command = [sys.executable, str(Path(__file__).resolve()),
               "--child", engine, "--bars", str(bars), "--repeats", str(repeats)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"engine": engine, "bars": bars, "status": f"timeout after {timeout:.0f}s"}

    if completed.returncode != 0:
        # A killed child (OOM) reports a negative return code and no stdout.
        detail = (completed.stderr or "").strip().splitlines()
        reason = detail[-1] if detail else f"exit {completed.returncode}"
        return {"engine": engine, "bars": bars, "status": f"failed: {reason[:160]}"}

    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("{"):
            return {**json.loads(line), "status": "ok"}
    return {"engine": engine, "bars": bars, "status": "failed: no result line"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bars", default="100000,1000000,5000000",
                        help="Comma-separated bar counts to sweep.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--engines", default="", help="Comma-separated subset of engine names.")
    parser.add_argument("--budget-seconds", type=float, default=600.0,
                        help="Skip an engine at a size when the previous size projects past this.")
    parser.add_argument("--output", type=Path)
    # Internal: the parent re-invokes this file to isolate one measurement.
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    sizes = [int(value) for value in args.bars.split(",") if value.strip()]
    if args.child:
        return _run_child(args.child, sizes[0], args.repeats)

    selected = [name for name in args.engines.split(",") if name.strip()] or list(ENGINES)
    unknown = [name for name in selected if name not in ENGINES]
    if unknown:
        parser.error(f"unknown engines: {unknown}. available: {list(ENGINES)}")

    print(f"extreme scaling, best of {args.repeats}, data construction excluded from timing")
    print(f"{platform.python_implementation()} {platform.python_version()} on {platform.machine()}\n")

    rows: list[dict] = []
    rates: dict[str, float] = {}
    for bars in sizes:
        print(f"--- {bars:,} bars ---")
        for engine in selected:
            projected = bars / rates[engine] if engine in rates else 0.0
            if projected > args.budget_seconds:
                status = f"skipped: projected {projected:.0f}s over {args.budget_seconds:.0f}s budget"
                print(f"{engine:<24} {status}")
                rows.append({"engine": engine, "bars": bars, "status": status})
                continue

            result = _measure(engine, bars, args.repeats, timeout=args.budget_seconds * 4)
            rows.append(result)
            if result["status"] != "ok":
                print(f"{engine:<24} {result['status']}")
                continue

            rates[engine] = result["bars_per_second"]
            note = "  <- not comparable: ingestion hoisted out of the timed run" if engine in NOT_COMPARABLE else ""
            print(f"{engine:<24} {result['seconds']:9.4f}s  {result['bars_per_second']:>12,.0f} bars/s  "
                  f"peak RSS {result['peak_rss_mb']:7.1f} MB ({result['peak_rss_over_import_mb']:+7.1f} MB){note}")
        print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fields = ["engine", "bars", "seconds", "bars_per_second", "peak_rss_mb",
                  "peak_rss_over_import_mb", "status", "comparable"]
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "comparable": row["engine"] not in NOT_COMPARABLE})
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
