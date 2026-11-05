"""Compare a `perf_gate.py` result against the committed baseline.

Tolerance is wide (25% by default) on purpose: this runs on shared CI
hardware, not a dedicated benchmark box, and the point is to catch a change
that actually made something slower or fatter, not to chase noise. A metric
only fails when it crosses the tolerance in the *bad* direction — a
throughput number climbing, or memory dropping, never fails.

    python benchmarks/compare_perf_gate.py \\
        --baseline benchmarks/perf_baseline.json \\
        --current benchmark-results/perf-gate/latest.json \\
        --tolerance 0.25

Exits 0 if nothing regressed beyond tolerance, 1 otherwise. A metric present
in --current but missing from --baseline is reported and does not fail the
run — it has nothing to regress against yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Mirrors benchmarks/perf_gate.py's METRIC_KINDS; duplicated rather than
# imported so this script can compare an old result file against a newer
# perf_gate.py that has renamed or added metrics without an import error.
LOWER_IS_BETTER_METRICS = {
    "extreme_scaling.fusion_100000.peak_rss_mb",
    "extreme_scaling.fusion_1000000.peak_rss_mb",
}


def compare(baseline: dict[str, float], current: dict[str, float], tolerance: float) -> tuple[list[str], list[str]]:
    """Returns (report_lines, regression_lines)."""
    report: list[str] = []
    regressions: list[str] = []
    width = max((len(name) for name in current), default=0)
    for name in sorted(current):
        value = current[name]
        base = baseline.get(name)
        if base is None:
            report.append(f"{name:<{width}}  {value:>14,.1f}  (no baseline yet)")
            continue
        delta = (value - base) / base if base else 0.0
        higher_better = name not in LOWER_IS_BETTER_METRICS
        bad_direction = delta < 0 if higher_better else delta > 0
        regressed = bad_direction and abs(delta) > tolerance
        flag = "REGRESSION" if regressed else "ok"
        report.append(f"{name:<{width}}  {value:>14,.1f}  baseline {base:>14,.1f}  {delta:+7.1%}  {flag}")
        if regressed:
            regressions.append(
                f"{name}: {value:,.1f} vs baseline {base:,.1f} ({delta:+.1%}, tolerance {tolerance:.0%})"
            )
    return report, regressions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.25)
    args = parser.parse_args()

    baseline_payload = json.loads(args.baseline.read_text())
    current_payload = json.loads(args.current.read_text())
    baseline_metrics = baseline_payload.get("metrics", baseline_payload)
    current_metrics = current_payload.get("metrics", current_payload)

    report, regressions = compare(baseline_metrics, current_metrics, args.tolerance)
    print("\n".join(report))

    if regressions:
        print(f"\n{len(regressions)} metric(s) regressed beyond {args.tolerance:.0%} tolerance:")
        for line in regressions:
            print(f"  - {line}")
        return 1

    print(f"\nall metrics within {args.tolerance:.0%} of baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
