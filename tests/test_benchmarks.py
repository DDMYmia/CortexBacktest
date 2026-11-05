from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.full_benchmark_suite import _csv_rows

def test_event_engine_compare_all_scenarios_runs_fusion_only() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/event_engine_compare.py",
            "--bars",
            "10",
            "--symbols",
            "2",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--scenario",
            "all",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "scenario,engine,available,seconds,bars_per_second,reason" in result.stdout
    for scenario in ("buy_once", "multi_asset", "multi_order", "order_types", "partial_fill", "open_order_scan"):
        assert f"{scenario},fusion_event,True," in result.stdout


def test_event_engine_compare_open_order_scan_large_pool_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/event_engine_compare.py",
            "--bars",
            "20",
            "--symbols",
            "5",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--scenario",
            "open_order_scan",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "scenario,engine,available,seconds,bars_per_second,reason"
    assert lines[1].startswith("open_order_scan,fusion_event,True,")


def test_factor_research_compare_matches_alphalens_rank_ic() -> None:
    pytest.importorskip("alphalens.performance")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/factor_research_compare.py",
            "--dates",
            "4",
            "--symbols",
            "10",
            "--repeat",
            "1",
            "--warmup",
            "0",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0].startswith("engine,available,seconds")
    fusion = lines[1].split(",")
    alphalens = lines[2].split(",")
    assert fusion[1] == "True"
    assert alphalens[1] == "True"
    assert float(fusion[4]) == pytest.approx(float(alphalens[4]))


def test_execution_sweep_parallel_compare_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/execution_sweep_parallel_compare.py",
            "--dates",
            "20",
            "--symbols",
            "5",
            "--cases",
            "2",
            "--workers",
            "2",
            "--repeat",
            "1",
            "--warmup",
            "0",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "mode,workers,cases,seconds,cases_per_second,final_equity_sum"
    assert lines[1].startswith("serial,1,2,")
    assert lines[2].startswith("process,2,2,")
    assert float(lines[1].split(",")[5]) == pytest.approx(float(lines[2].split(",")[5]))


def test_event_engine_modes_compare_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/event_engine_modes_compare.py",
            "--bars",
            "20",
            "--symbols",
            "3",
            "--repeat",
            "1",
            "--warmup",
            "0",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "scenario,mode,seconds,bars_per_second,equity_points"
    assert any("feed_final_snapshot" in line for line in lines)
    assert any("feed_final_summary_retention" in line for line in lines)
    assert any(line.endswith(",1") for line in lines if "feed_final_snapshot" in line)


def test_event_record_export_compare_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/event_record_export_compare.py",
            "--bars",
            "20",
            "--symbols",
            "3",
            "--repeat",
            "1",
            "--warmup",
            "0",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "mode,seconds,orders,fills,equity_points"
    assert any(line.startswith("json_report_records,") for line in lines)
    assert any(line.startswith("stream_csv_records,") for line in lines)
    assert any(line.startswith("run_stream_ndjson_summary,") for line in lines)


def test_rebalance_execution_compare_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/rebalance_execution_compare.py",
            "--dates",
            "5",
            "--symbols",
            "4",
            "--top",
            "2",
            "--repeat",
            "1",
            "--warmup",
            "0",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "mode,seconds,bars_per_second,fills,orders,final_equity"
    assert any(line.startswith("list_bar_snapshots,") for line in lines)
    assert any(line.startswith("feed_final_snapshot,") for line in lines)


def test_large_scale_stress_compare_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/large_scale_stress_compare.py",
            "--symbols", "5",
            "--bars", "20",
            "--trade-fraction", "0.7",
            "--bracket-fraction", "0.1",
            "--trailing-fraction", "0.1",
            "--long-lived-limit-fraction", "0.1",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["symbols"] == 5
    assert payload["bar_observations"] > 0
    assert payload["orders"] > 0
    # A bracket is entry + 2 OCO legs, so orders should exceed fills whenever
    # at least one bracket or long-lived limit fired.
    assert payload["orders"] >= payload["fills"]


def test_full_benchmark_suite_parses_csv_rows():
    rows = _csv_rows("a,b\n1,2\n")

    assert rows == [{"a": "1", "b": "2"}]
