"""Adapter for plain bar and factor exports from an upstream application.

The producing application exports plain files with stable columns. This
module reads those files without importing `backend.*`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl

from ..contracts import CorporateAction
from ..data import read_table, to_backend
from ..data.backend import normalize_date_column
from ..data.calendar import TradingCalendar
from ..datasets import BacktestDataset, DatasetProvenance, load_backtest_dataset
from ..datasets.availability import FactorAvailability, align_factor_availability
from ..research.frames import (
    forward_returns_matrix,
    stack_factor_and_forward_returns,
)

BAR_COLUMN_ALIASES = {
    "timestamp": "date",
    "time": "date",
    "ticker": "symbol",
    "adjClose": "adj_close",
    "adj_close": "adj_close",
    "adjusted_close": "adj_close",
    "close": "close",
}

# `timestamp` and `analysis_timestamp` become `date`, which claims the value was
# usable that day. For an analysis timestamp that is a claim about the exporter,
# not a fact: it says when the analysis ran, and an analysis that ran at 18:00
# cannot be traded on at that morning's open. Pass `calendar` and `availability`
# to `normalize_factor_export` to have the column read as an observation and
# realigned onto the first session that could act on it.
FACTOR_COLUMN_ALIASES = {
    "timestamp": "date",
    "analysis_timestamp": "date",
    "ticker": "symbol",
    "factor_value": "factor",
    "score": "factor",
    "value": "factor",
}


@dataclass(frozen=True, slots=True)
class ExportBundle:
    bars: pl.DataFrame
    factors: pl.DataFrame
    dataset: BacktestDataset
    prices: pl.DataFrame
    returns: pl.DataFrame
    forward_returns: pl.DataFrame
    factor_values: pl.DataFrame
    research_frame: pl.DataFrame


def load_table(path: str | Path) -> pl.DataFrame:
    """Load a CSV or parquet table as Polars."""
    return read_table(path, backend="polars")


def normalize_bar_export(frame) -> pl.DataFrame:
    """Normalize exported bars to date/symbol/adj_close columns as Polars."""
    data = _rename_known_columns(frame, BAR_COLUMN_ALIASES)
    if "adj_close" not in data.columns and "close" in data.columns:
        raise ValueError(
            "adj_close is missing; raw close cannot be silently relabelled as adjusted"
        )
    required = {"date", "symbol", "adj_close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Bar export missing required columns: {sorted(missing)}")
    return _normalize_values(data, "adj_close")


def normalize_factor_export(
    frame,
    *,
    factor_name: str | None = None,
    calendar: TradingCalendar | None = None,
    availability: FactorAvailability | None = None,
) -> pl.DataFrame:
    """Normalize exported factors to date/symbol/factor columns as Polars.

    Without `calendar`, the export's timestamp is taken at face value: the
    factor is assumed usable on the day it is stamped. That holds for a factor
    computed from a session's own closing data and does not hold for anything
    with a publication time.

    With `calendar` (and optionally `availability`), the timestamp is read as an
    *observation* and the row is restamped onto the first session that could
    have traded it. The original is kept in `event_time`, so a dataset built
    this way still shows where each value came from.
    """
    data = _rename_known_columns(frame, FACTOR_COLUMN_ALIASES)
    if factor_name is not None:
        if factor_name not in data.columns:
            raise ValueError(f"Factor column not found: {factor_name}")
        data = data.rename({factor_name: "factor"})

    required = {"date", "symbol", "factor"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Factor export missing required columns: {sorted(missing)}")
    if calendar is not None:
        data = align_factor_availability(
            data.rename({"date": "event_time"}), calendar, availability
        )
    elif availability is not None:
        raise ValueError(
            "availability needs a calendar: the session a value becomes tradable in "
            "cannot be found without one"
        )
    return _normalize_values(data, "factor")


def build_research_bundle(
    bars,
    factors,
    *,
    forward_periods: int = 5,
    factor_name: str | None = None,
    calendar: TradingCalendar | None = None,
    availability: FactorAvailability | None = None,
    provenance: DatasetProvenance | None = None,
    research_mode: str = "strict",
    membership=None,
    corporate_actions: Iterable[CorporateAction] | None = None,
) -> ExportBundle:
    """Build normalized Polars matrices and research rows from exports."""
    dataset = build_export_dataset(
        bars,
        factors,
        factor_name=factor_name,
        calendar=calendar,
        availability=availability,
        provenance=provenance,
        research_mode=research_mode,
        membership=membership,
        corporate_actions=corporate_actions,
    )
    normalized_bars, normalized_factors = dataset.bars, dataset.factors
    prices = dataset.prices
    returns = dataset.returns
    factor_values = dataset.factor_values
    forward_returns = forward_returns_matrix(prices, periods=forward_periods)
    research_frame = stack_factor_and_forward_returns(factor_values, forward_returns)

    return ExportBundle(
        bars=normalized_bars,
        factors=normalized_factors,
        dataset=dataset,
        prices=prices,
        returns=returns,
        forward_returns=forward_returns,
        factor_values=factor_values,
        research_frame=research_frame,
    )


def build_export_dataset(
    bars,
    factors,
    *,
    factor_name: str | None = None,
    calendar: TradingCalendar | None = None,
    availability: FactorAvailability | None = None,
    provenance: DatasetProvenance | None = None,
    research_mode: str = "strict",
    membership=None,
    corporate_actions: Iterable[CorporateAction] | None = None,
) -> BacktestDataset:
    """Normalize exports and return a Polars BacktestDataset.

    Pass `calendar` for exports whose factor timestamp is an observation rather
    than a tradable session -- anything derived from news, filings, analyst
    output or a nightly analysis run.
    """
    bar_frame = load_table(bars) if isinstance(bars, str | Path) else bars
    factor_frame = load_table(factors) if isinstance(factors, str | Path) else factors
    if research_mode not in {"strict", "exploration"}:
        raise ValueError("research_mode must be strict or exploration")
    if research_mode == "strict":
        if provenance is None or calendar is None or availability is None:
            raise ValueError(
                "Strict research requires provenance, calendar and explicit availability policy"
            )
        if not calendar.has_hours:
            raise ValueError("Strict research requires exchange session hours")
        provenance.require_verified_versions()
        if membership is None or corporate_actions is None:
            raise ValueError("Strict research requires explicit membership and corporate actions")
        if "available_at" not in factor_frame.columns:
            raise ValueError(
                "Strict factors require available_at; observation dates are not publication times"
            )
        from datetime import UTC, datetime

        stamps = []
        for value in factor_frame["available_at"]:
            stamp = (
                value
                if isinstance(value, datetime)
                else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            )
            if stamp.tzinfo is None or stamp > provenance.as_of:
                raise ValueError("Factor availability must be timezone-aware and <= snapshot as_of")
            stamps.append(stamp.astimezone(UTC))
        factor_frame = _rename_known_columns(factor_frame, FACTOR_COLUMN_ALIASES)
        factor_frame = factor_frame.with_columns(pl.Series("date", stamps, dtype=pl.Datetime("us", "UTC")))
    dataset = load_backtest_dataset(
        provenance=provenance,
        membership=membership,
        corporate_actions=corporate_actions,
        calendar=calendar,
        bars=normalize_bar_export(bar_frame),
        factors=normalize_factor_export(
            factor_frame,
            factor_name=factor_name,
            calendar=calendar,
            availability=availability,
        ),
    )

    return replace(dataset, research_verified=research_mode == "strict")


def _rename_known_columns(frame, aliases: dict[str, str]) -> pl.DataFrame:
    data = to_backend(frame, backend="polars")
    rename_map = {}
    for column in data.columns:
        normalized = str(column).strip()
        rename_map[column] = aliases.get(normalized, normalized)
    return data.rename(rename_map)


def _normalize_values(data: pl.DataFrame, value_column: str) -> pl.DataFrame:
    """Common typed table boundary after bar/factor-specific semantic checks."""
    return (
        normalize_date_column(data)
        .with_columns(
            pl.col("symbol").cast(pl.Utf8).str.to_uppercase(),
            pl.col(value_column).cast(pl.Float64, strict=False),
        )
        .drop_nulls(["date", "symbol", value_column])
        .sort(["date", "symbol"])
    )
