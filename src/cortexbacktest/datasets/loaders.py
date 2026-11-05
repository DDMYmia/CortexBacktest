"""Dataset loaders for research and backtesting workflows."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time
from pathlib import Path
from typing import Iterable, Literal
from uuid import uuid4

import polars as pl

from ..contracts import CorporateAction, CorporateActionType
from ..data.backend import (
    normalize_date_column,
    read_table,
    to_backend,
    write_table,
)
from ..data.calendar import SessionWindow, TradingCalendar, infer_calendar_from_index
from ..data.universe import Universe
from ..research.delisting import apply_delisting_prices
from ..research.frames import factor_matrix, price_matrix, returns_matrix
from ..research.workflow import FactorResearchReport, run_factor_research
from .diagnostics import BarDiagnostics, diagnose_bars
from .provenance import DatasetProvenance, sha256_file, verify_file_hashes
from .schema import BarResolution, DataQualityReport, bars_to_events, normalize_bars, validate_bars


@dataclass(frozen=True, slots=True)
class BacktestDataset:
    bars: pl.DataFrame
    factors: pl.DataFrame
    prices: pl.DataFrame
    factor_values: pl.DataFrame
    returns: pl.DataFrame
    calendar: TradingCalendar
    universe: Universe | None = None
    quality: DataQualityReport = DataQualityReport()
    provenance: DatasetProvenance | None = None
    # Bars are normalized to the canonical schema and factors are keyed on
    # date/symbol, but the factor value column keeps its source name. A saved
    # dataset records it so a directory round trip does not have to guess.
    factor_col: str = "factor"
    resolution: BarResolution = "session"
    research_verified: bool = False
    corporate_actions: tuple[CorporateAction, ...] = ()

    def run_factor_research(self, **kwargs) -> FactorResearchReport:
        return run_factor_research(
            self.prices,
            self.factor_values,
            universe=self.universe,
            **kwargs,
        )

    def diagnose(self, **kwargs) -> "BarDiagnostics":
        """Describe this dataset's bars: gaps, stale runs, outliers, churn.

        Separate from loading on purpose. `validate_bars` decides whether the
        data can be used and runs on the way in; this describes what is in it,
        costs a few passes, and is asked for when a result is being written up.
        """
        return diagnose_bars(self.bars, calendar=self.calendar, **kwargs)

    def to_bars(self):
        """Convert the dataset bars to event-engine Bar objects."""
        return bars_to_events(self.bars)

    def to_event_feed(self):
        """Convert dataset bars to a pre-grouped event feed."""
        from ..event.feed import ColumnarBarFeed

        return ColumnarBarFeed.from_polars(self.bars)

    def run_event_backtest(
        self, strategy, *, broker=None, trade_on_close: bool = False, snapshot_frequency: Literal["bar", "session", "final", "none"] = "bar"
    ):
        """Run an event-driven backtest over this dataset's bars."""
        from ..event.engine import BacktestEngine

        engine = BacktestEngine(
            broker=broker, trade_on_close=trade_on_close, snapshot_frequency=snapshot_frequency
        )
        return engine.run(self.to_event_feed(), strategy)

    def run_research_execution_pipeline(self, *, config=None, run_metadata=None, execution_bars=None):
        """Run the standard factor-research to event-execution workflow."""
        from ..pipelines import run_research_execution_pipeline

        return run_research_execution_pipeline(self, config=config, run_metadata=run_metadata, execution_bars=execution_bars)

    def save(self, path: str | Path, *, format: str = "parquet") -> Path:
        """Persist dataset bars/factors and optional universe membership to a directory."""
        return save_backtest_dataset(self, path, format=format)


def load_backtest_dataset(
    *,
    bars,
    factors,
    membership=None,
    calendar: TradingCalendar | None = None,
    date_col: str = "date",
    symbol_col: str = "symbol",
    price_col: str = "adj_close",
    factor_col: str = "factor",
    provenance: DatasetProvenance | None = None,
    corporate_actions: Iterable[CorporateAction] | None = None,
    resolution: BarResolution = "session",
) -> BacktestDataset:
    """Load standard long-form bars/factors into a Polars BacktestDataset.

    ``resolution`` says what one bar row means. The default, ``"session"``,
    is the cross-sectional contract: one observation per symbol per session, and
    intraday input is refused at the door. ``"intraday"`` opens the same path to
    minute or hourly bars, and then a session-aware ``calendar`` is required --
    every return below is between adjacent rows, and without knowing where a
    session ends each overnight gap would be folded in as an intraday move.


    Pass ``corporate_actions`` to make delistings cost what they cost. Without
    them a delisted symbol's prices simply stop, and every downstream
    calculation reads that as the position leaving the sample rather than losing
    its value — survivorship bias, applied hardest to the factors that load on
    distress. See `research/delisting.py`.
    """
    bars_frame = normalize_bars(
        _load_frame(bars),
        date_col=date_col,
        symbol_col=symbol_col,
        price_col=price_col,
    )
    # normalize_bars already keys bars on date/symbol; key the factors the same
    # way so both frames of a dataset share one column contract.
    factors_frame = _load_frame(factors)
    renames = {
        source: target
        for source, target in ((date_col, "date"), (symbol_col, "symbol"))
        if source != target and source in factors_frame.columns
    }
    if renames:
        factors_frame = factors_frame.rename(renames)
    sessions = calendar or infer_calendar_from_index(bars_frame["date"])
    if resolution == "intraday" and not sessions.has_hours:
        raise ValueError(
            "intraday resolution needs a calendar with session hours: returns here are "
            "between adjacent rows, and the session boundary is what separates an "
            "intraday move from an overnight gap. Build one with "
            "TradingCalendar.from_exchange() or infer_session_hours()."
        )
    # Everything below -- the price matrix pivot, adjacent-row returns, the
    # adj_close price semantics -- is defined per session by default. Intraday
    # bars pass every other check and then quietly produce wrong returns, so at
    # session resolution they are refused here rather than downstream.
    quality = validate_bars(bars_frame, calendar=sessions, resolution=resolution)
    quality.raise_if_failed()
    prices = price_matrix(bars_frame)
    actions = tuple(corporate_actions or ())
    if actions:
        # Applied to the price matrix rather than to each consumer, so returns,
        # IC, quantiles, and portfolio construction all pay for a delisting
        # without knowing it happened.
        prices = apply_delisting_prices(prices, actions)
    factor_values = factor_matrix(
        factors_frame,
        date_col="date",
        symbol_col="symbol",
        factor_col=factor_col,
    )
    # The calendar reaches the return calculation, so an intraday dataset's
    # first bar of each session returns null instead of an overnight gap.
    returns = returns_matrix(prices, calendar=sessions if resolution == "intraday" else None)
    sessions.validate_timestamps(factors_frame["date"], label="factor timestamps")
    universe = None
    if membership is not None:
        membership_frame = _load_frame(membership)
        universe = Universe.from_membership(membership_frame, prices["date"])
    return BacktestDataset(
        bars=bars_frame,
        factors=factors_frame,
        prices=prices,
        factor_values=factor_values,
        returns=returns,
        calendar=sessions,
        universe=universe,
        quality=quality,
        provenance=provenance,
        factor_col=factor_col,
        resolution=resolution,
        corporate_actions=actions,
    )


def _write_backtest_dataset(
    dataset: BacktestDataset,
    path: str | Path,
    *,
    format: str = "parquet",
) -> None:
    """Save a Polars BacktestDataset directory with a small JSON manifest."""
    suffix = _format_suffix(format)
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    bars_name = f"bars{suffix}"
    factors_name = f"factors{suffix}"
    manifest = {
        "version": 5,
        "resolution": dataset.resolution,
        "rows": {"bars": dataset.bars.height, "factors": dataset.factors.height},
        "time_bounds": {
            "first": str(dataset.bars["date"].min()),
            "last": str(dataset.bars["date"].max()),
        },
        "schema": {key: str(value) for key, value in dataset.bars.schema.items()},
        "research_status": "pit_verified" if dataset.research_verified else "unverified",
        "calendar": {
            "sessions": [value.isoformat() for value in dataset.calendar.sessions],
            "name": dataset.calendar.name,
            "zone": dataset.calendar.zone,
            "hours": [
                dataset.calendar.hours.open.isoformat(),
                dataset.calendar.hours.close.isoformat(),
            ]
            if dataset.calendar.hours
            else None,
            "overrides": {
                str(day): [hours.open.isoformat(), hours.close.isoformat()]
                for day, hours in dataset.calendar.session_overrides.items()
            },
        },
        "format": format,
        "bars": bars_name,
        "factors": factors_name,
        "membership": None,
        "corporate_actions": "corporate_actions.json",
        # Without this a reload guesses the default and fails on any dataset
        # whose factor column carries its source name.
        "columns": {"factor": dataset.factor_col},
        "provenance": dataset.provenance.to_dict() if dataset.provenance else None,
    }
    write_table(dataset.bars, output_dir / bars_name)
    write_table(dataset.factors, output_dir / factors_name)
    actions = [
        {**asdict(action), "action_type": action.action_type.value,
         "ex_date": action.ex_date.isoformat(),
         "pay_date": action.pay_date.isoformat() if action.pay_date else None}
        for action in dataset.corporate_actions
    ]
    (output_dir / "corporate_actions.json").write_text(
        json.dumps(actions, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    if dataset.universe is not None:
        membership_name = f"membership{suffix}"
        write_table(dataset.universe.mask, output_dir / membership_name)
        manifest["membership"] = membership_name
    manifest["sha256"] = {
        name: sha256_file(output_dir / name)
        for name in (
            bars_name,
            factors_name,
            "corporate_actions.json",
            *([manifest["membership"]] if manifest["membership"] else []),
        )
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_backtest_dataset_dir(
    path: str | Path, *, verify_hashes: bool = True,
    manifest_sha256: str | None = None,
) -> BacktestDataset:
    """Load a saved dataset; restoring verified status needs an external manifest hash.

    Legacy loads remain supported but are unverified. The external digest binds
    calendar, provenance and component names as well as the component hashes.
    It does not independently prove the producer's market-data assertions.
    """
    dataset_dir = Path(path)
    if (dataset_dir / "current.json").exists():
        pointer = json.loads((dataset_dir / "current.json").read_text())
        candidate = (dataset_dir / pointer["path"]).resolve()
        if not candidate.is_relative_to(dataset_dir.resolve()):
            raise ValueError("Snapshot pointer escapes repository")
        dataset_dir = candidate
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    content = manifest_path.read_bytes()
    if manifest_sha256 is not None and hashlib.sha256(content).hexdigest() != manifest_sha256:
        raise ValueError("Snapshot manifest hash mismatch")
    manifest = json.loads(content)
    version = manifest.get("version", 1)
    if type(version) is not int or version not in range(1, 6):
        raise ValueError("Unsupported dataset manifest version")
    components = [manifest[key] for key in ("bars", "factors", "membership", "corporate_actions") if manifest.get(key)]
    for name in components:
        if not (dataset_dir / name).resolve().is_relative_to(dataset_dir.resolve()):
            raise ValueError("Dataset component escapes snapshot directory")
    if version == 5:
        if not manifest.get("corporate_actions"):
            raise ValueError("Snapshot is missing corporate actions component")
        if any(name not in manifest.get("sha256", {}) for name in components):
            raise ValueError("Snapshot is missing component hashes")
    if verify_hashes and manifest.get("sha256"):
        verify_file_hashes(dataset_dir, manifest["sha256"])
    provenance = (
        DatasetProvenance.from_dict(manifest["provenance"]) if manifest.get("provenance") else None
    )
    membership = manifest.get("membership")
    actions = []
    if manifest.get("corporate_actions"):
        action_path = (dataset_dir / manifest["corporate_actions"]).resolve()
        for row in json.loads(action_path.read_text(encoding="utf-8")):
            actions.append(CorporateAction(
                **{**row, "action_type": CorporateActionType(row["action_type"]),
                   "ex_date": date.fromisoformat(row["ex_date"]),
                   "pay_date": date.fromisoformat(row["pay_date"]) if row.get("pay_date") else None}
            ))
    # Manifests before version 3 carry no column names; they were only ever
    # readable when the source columns matched these defaults anyway.
    columns = manifest.get("columns") or {}
    calendar_data = manifest.get("calendar")
    calendar = None
    if calendar_data:
        hours = calendar_data.get("hours")
        calendar = TradingCalendar(
            sessions=pl.Series(
                "date", [datetime.fromisoformat(v) for v in calendar_data["sessions"]]
            ),
            name=calendar_data["name"],
            zone=calendar_data.get("zone"),
            hours=SessionWindow(*map(time.fromisoformat, hours)) if hours else None,
            session_overrides={
                date.fromisoformat(day): SessionWindow(*map(time.fromisoformat, window))
                for day, window in calendar_data.get("overrides", {}).items()
            },
        )
    dataset = load_backtest_dataset(
        calendar=calendar,
        resolution=manifest.get("resolution", "session"),
        bars=dataset_dir / manifest["bars"],
        factors=dataset_dir / manifest["factors"],
        provenance=provenance,
        factor_col=columns.get("factor", "factor"),
        corporate_actions=actions,
    )
    if version == 5 and manifest.get("rows") != {"bars": dataset.bars.height, "factors": dataset.factors.height}:
        raise ValueError("Snapshot record counts do not match manifest")
    verified = (manifest_sha256 is not None and verify_hashes and version == 5
                and manifest.get("research_status") == "pit_verified")
    if verified:
        if provenance is None or calendar is None or not calendar.has_hours or not membership:
            raise ValueError("Verified snapshot requires provenance, membership and explicit calendar hours")
        provenance.require_verified_versions()
    if not membership:
        return replace(dataset, research_verified=verified)
    mask_frame = _load_frame(dataset_dir / membership)
    if "date" not in mask_frame.columns:
        raise ValueError("Saved membership mask requires a date column")
    mask = normalize_date_column(mask_frame).sort("date")
    mask = mask.rename({column: str(column).upper() for column in mask.columns if column != "date"})
    boolean = [
        pl.col(column).cast(pl.Boolean).alias(column) for column in mask.columns if column != "date"
    ]
    universe = Universe(mask.with_columns(boolean), name="saved")
    return replace(dataset, universe=universe, research_verified=verified)


def load_target_weights(
    weights,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    weight_col: str = "weight",
) -> pl.DataFrame:
    """Load a date x symbol target-weight matrix as Polars."""
    frame = _load_frame(weights)
    if date_col in frame.columns and {symbol_col, weight_col}.issubset(frame.columns):
        result = (
            frame.select(
                [
                    pl.col(date_col).alias("date"),
                    pl.col(symbol_col).cast(pl.Utf8).str.to_uppercase().alias("symbol"),
                    pl.col(weight_col).cast(pl.Float64, strict=False).alias("weight"),
                ]
            )
            .drop_nulls(["date", "symbol", "weight"])
            .group_by(["date", "symbol"])
            .agg(pl.col("weight").last())
            .pivot(index="date", on="symbol", values="weight", aggregate_function="last")
            .sort("date")
        )
        result = result.select(
            ["date", *sorted(column for column in result.columns if column != "date")]
        )
    elif date_col in frame.columns:
        symbols = sorted(str(column).upper() for column in frame.columns if column != date_col)
        rename_map = {
            column: str(column).upper()
            for column in frame.columns
            if column != date_col and str(column).upper() != column
        }
        result = frame.rename(rename_map) if rename_map else frame
        result = (
            result.with_columns(pl.col(date_col).alias("date"))
            .select(
                [
                    pl.col("date"),
                    *[
                        pl.col(symbol).cast(pl.Float64, strict=False).alias(symbol)
                        for symbol in symbols
                    ],
                ]
            )
            .sort("date")
        )
    else:
        raise ValueError("target weights require a date column for Polars input")
    if result.is_empty():
        raise ValueError("target weights are empty")
    if result["date"].is_duplicated().any():
        raise ValueError("target weights require unique dates")
    return result


def _load_frame(frame_or_path) -> pl.DataFrame:
    """Read files directly as Polars; convert external in-memory frames once."""
    if isinstance(frame_or_path, (str, os.PathLike)):
        return read_table(Path(frame_or_path))
    return to_backend(frame_or_path)


def _format_suffix(format: str) -> str:
    normalized = format.lower()
    if normalized == "csv":
        return ".csv"
    if normalized in {"parquet", "pq"}:
        return ".parquet"
    raise ValueError("format must be 'csv' or 'parquet'")


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_backtest_dataset(
    dataset: BacktestDataset,
    path: str | Path,
    *,
    format: str = "parquet",
) -> Path:
    """Publish a new immutable directory; an existing snapshot is never overwritten."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Immutable snapshot already exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=target.parent))
    try:
        dataset.quality.raise_if_failed()
        _write_backtest_dataset(dataset, staging, format=format)
        for component in staging.iterdir():
            with component.open("rb") as handle:
                os.fsync(handle.fileno())
        _sync_directory(staging)
        # Renaming a completed nonempty directory is atomic on a local filesystem.
        # A concurrently published nonempty target causes rename to fail.
        os.rename(staging, target)
        _sync_directory(target.parent)
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def publish_backtest_snapshot(dataset: BacktestDataset, repository: str | Path) -> Path:
    """Create an immutable version, then atomically move the repository pointer."""
    if dataset.provenance is None or not dataset.research_verified:
        raise ValueError(
            "Published research snapshots require strict PIT verification and provenance"
        )
    root = Path(repository)
    snapshot_id = uuid4().hex
    output = save_backtest_dataset(dataset, root / "snapshots" / snapshot_id)
    pointer = root / f".current-{snapshot_id}.json"
    try:
        with pointer.open("w") as handle:
            json.dump({"snapshot_id": snapshot_id, "path": f"snapshots/{snapshot_id}"}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pointer, root / "current.json")
        _sync_directory(root)
    finally:
        pointer.unlink(missing_ok=True)
    return output
