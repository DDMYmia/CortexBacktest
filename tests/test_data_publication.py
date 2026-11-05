from datetime import UTC, datetime, time, timedelta

import polars as pl
import pytest

from cortexbacktest.adapters.export_files import build_export_dataset, normalize_bar_export
from cortexbacktest.data.calendar import TradingCalendar
from cortexbacktest.datasets import (
    DatasetProvenance,
    FactorAvailability,
    load_backtest_dataset_dir,
    publish_backtest_snapshot,
    save_backtest_dataset,
)
from cortexbacktest.event.parquet_feed import ParquetBarFeed


def dataset(available_at=None):
    dates = [datetime(2024, 1, d) for d in (2, 3, 4)]
    calendar = TradingCalendar(pl.Series("date", dates)).with_hours(
        time(9, 30), time(16), zone="America/New_York"
    )
    provenance = DatasetProvenance(
        "fixture",
        "bars",
        datetime(2024, 1, 5, tzinfo=UTC),
        "v1",
        adjustment_mode="split_adjusted",
        calendar_version="XNYS-v1",
        universe_version="membership-v1",
        corporate_action_version="actions-v1",
    )
    bars = pl.DataFrame({"date": dates, "symbol": ["A"] * 3, "adj_close": [10.0, 11.0, 12.0]})
    factors = pl.DataFrame(
        {
            "date": [dates[0]],
            "symbol": ["A"],
            "factor": [1.0],
            "available_at": [available_at if available_at is not None else datetime(2024, 1, 2, 22, tzinfo=UTC)],
        }
    )
    return build_export_dataset(
        bars,
        factors,
        calendar=calendar,
        availability=FactorAvailability(),
        provenance=provenance,
        membership=pl.DataFrame({"symbol": ["A"], "start": [dates[0]], "end": [dates[-1]]}),
        corporate_actions=[],
    )


def test_strict_ingress_and_unknown_adjustment():
    with pytest.raises(ValueError, match="Strict research"):
        build_export_dataset(pl.DataFrame(), pl.DataFrame())
    with pytest.raises(ValueError, match="raw close"):
        normalize_bar_export(
            pl.DataFrame({"date": ["2024-01-01"], "symbol": ["A"], "close": [1.0]})
        )
    data = dataset()
    assert data.research_verified and data.factors["date"][0].day == 3


def test_immutable_publish_and_failure_keeps_previous(tmp_path, monkeypatch):
    import cortexbacktest.datasets.loaders as module

    data = dataset()
    previous = publish_backtest_snapshot(data, tmp_path)
    pointer = (tmp_path / "current.json").read_bytes()
    assert load_backtest_dataset_dir(tmp_path).calendar.zone == "America/New_York"
    with pytest.raises(FileExistsError):
        save_backtest_dataset(data, previous)
    original = module.write_table

    def fail_factors(frame, path, **kwargs):
        if path.name.startswith("factors"):
            raise OSError("injected disk failure")
        return original(frame, path, **kwargs)

    monkeypatch.setattr(module, "write_table", fail_factors)
    with pytest.raises(OSError):
        publish_backtest_snapshot(data, tmp_path)
    assert (tmp_path / "current.json").read_bytes() == pointer
    assert load_backtest_dataset_dir(tmp_path).bars.equals(data.bars)
    assert not list((tmp_path / "snapshots").glob(".snapshot-*"))


def test_filtered_symbol_partition_merge_and_pruning(tmp_path):
    dates = [datetime(2024, 1, 2, tzinfo=UTC) + timedelta(minutes=i) for i in range(12)]
    paths = []
    for symbol in ["B", "A"]:
        frame = pl.DataFrame(
            {
                "date": dates,
                "symbol": [symbol] * 12,
                "open": [10.0] * 12,
                "high": [11.0] * 12,
                "low": [9.0] * 12,
                "close": [10.0] * 12,
                "volume": [1.0] * 12,
            }
        )
        path = tmp_path / f"{symbol}.parquet"
        frame.write_parquet(path, row_group_size=3)
        paths.append(path)
    feed = ParquetBarFeed(
        paths, merge_sorted=True, start=dates[3], end=dates[6], symbols=["a"], batch_rows=2
    )
    groups = list(feed)
    assert len(groups) == 3 and all([b.symbol for b in g.bars] == ["A"] for g in groups)
    assert feed.scan_stats["row_groups_skipped"] == 6
    assert feed.scan_stats["rows_decoded"] == 6


def test_snapshot_status_requires_external_manifest_binding(tmp_path):
    import hashlib
    import json

    path = save_backtest_dataset(dataset(), tmp_path / "dataset")
    manifest = path / "manifest.json"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert not load_backtest_dataset_dir(path).research_verified
    assert load_backtest_dataset_dir(path, manifest_sha256=digest).research_verified
    assert not load_backtest_dataset_dir(
        path, manifest_sha256=digest, verify_hashes=False
    ).research_verified
    doc = json.loads(manifest.read_bytes())
    doc["calendar"]["zone"] = "UTC"
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="manifest hash"):
        load_backtest_dataset_dir(path, manifest_sha256=digest)


@pytest.mark.parametrize("version", [6, 100, True, "5"])
def test_unknown_snapshot_versions_are_not_guessed(tmp_path, version):
    import json

    path = save_backtest_dataset(dataset(), tmp_path / "dataset")
    manifest = path / "manifest.json"
    doc = json.loads(manifest.read_bytes())
    doc["version"] = version
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Unsupported"):
        load_backtest_dataset_dir(path)


def test_component_paths_are_checked_even_without_hash_verification(tmp_path):
    import json

    path = save_backtest_dataset(dataset(), tmp_path / "dataset")
    manifest = path / "manifest.json"
    doc = json.loads(manifest.read_bytes())
    doc["bars"] = "../outside.parquet"
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="escapes"):
        load_backtest_dataset_dir(path, verify_hashes=False)


def test_component_counts_cannot_be_silently_reduced(tmp_path):
    import json

    path = save_backtest_dataset(dataset(), tmp_path / "dataset")
    manifest = path / "manifest.json"
    doc = json.loads(manifest.read_bytes())
    doc["rows"]["factors"] += 1
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="record counts"):
        load_backtest_dataset_dir(path)


def test_external_digest_does_not_turn_unknown_provenance_into_verified(tmp_path):
    import hashlib
    import json

    path = save_backtest_dataset(dataset(), tmp_path / "dataset")
    manifest = path / "manifest.json"
    doc = json.loads(manifest.read_bytes())
    doc["provenance"]["universe_version"] = "unknown"
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="verified adjustment"):
        load_backtest_dataset_dir(
            path, manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()
        )


@pytest.mark.parametrize("missing", ["membership", "corporate_actions"])
def test_strict_build_needs_actual_inputs_not_just_version_labels(missing):
    data = dataset()
    evidence = {
        "membership": pl.DataFrame(
            {"symbol": ["A"], "start": [datetime(2024, 1, 2)], "end": [datetime(2024, 1, 4)]}
        ),
        "corporate_actions": [],
    }
    evidence.pop(missing)
    with pytest.raises(ValueError, match="explicit membership and corporate actions"):
        build_export_dataset(
            data.bars,
            data.factors,
            calendar=data.calendar,
            availability=FactorAvailability(),
            provenance=data.provenance,
            **evidence,
        )


@pytest.mark.parametrize("timestamp", ["2024-01-02T22:00:00Z", "2024-01-03T11:00:00+13:00"])
def test_strict_json_availability_preserves_instant_across_offsets(timestamp):
    data = dataset(timestamp)
    assert data.research_verified
    assert data.factors["date"].to_list() == [datetime(2024, 1, 3)]
    assert data.factors["event_time"].to_list() == [datetime(2024, 1, 2, 22, tzinfo=UTC)]


def test_research_layer_uses_prior_month_and_inclusive_membership_end(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    from cortexbacktest.data.universe import Universe

    script = Path(__file__).parents[1] / "scripts/build_research_layer.py"
    if not script.exists():  # a data-server script; the public export leaves it out
        pytest.skip("scripts/build_research_layer.py is not part of this checkout")
    spec = importlib.util.spec_from_file_location("research_layer", script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    source = tmp_path / "daily.parquet"
    pl.DataFrame({"ticker": ["AAA", "BBB", "AAA", "BBB"],
                  "day": [datetime(2020, 1, 2)] * 2 + [datetime(2020, 2, 3)] * 2,
                  "close": [10.0] * 4, "volume": [1.0, 100.0, 1000.0, 1.0]}).write_parquet(source)
    monkeypatch.setattr(builder, "DAILY", source)
    assert builder.monthly_universes(1) == {"2020-02": ["BBB"]}
    spans = builder.membership_intervals({"2020-01": ["AAA"], "2020-02": ["BBB"]})
    mask = Universe.from_membership(spans, [datetime(2020, 1, 31), datetime(2020, 2, 1)]).mask
    assert mask["AAA"].to_list() == [True, False]
    assert mask["BBB"].to_list() == [False, True]
