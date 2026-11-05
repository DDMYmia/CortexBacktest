"""Standard dataset schemas and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import polars as pl

from ..contracts import Bar
from ..data.backend import normalize_date_column, to_backend
from ..data.calendar import TradingCalendar

STANDARD_BAR_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume", "adj_close")
OPTIONAL_BAR_COLUMNS = ("vwap",)

# What one row means. The bar columns are identical either way -- the `date`
# column has always been a Datetime, so it holds 09:31:00 as readily as a
# midnight session label -- and that is exactly why the difference has to be
# named rather than inferred. "session" is the daily contract: one observation
# per symbol per session, timestamped with a label rather than the instant it
# was observed. "intraday" is the event contract: every row is an instant, many
# per session. "any" checks only what both share, and is what a feed that
# carries its own availability semantics wants.
BarResolution = Literal["any", "session", "intraday"]


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """Validation result for loaded market data."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise ValueError("; ".join(self.errors))


def normalize_bars(
    frame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    price_col: str = "adj_close",
) -> pl.DataFrame:
    """Normalize long-form bar data to the engine OHLCV schema using Polars."""
    data = to_backend(frame, backend="polars")
    rename_map = {
        date_col: "date",
        symbol_col: "symbol",
        price_col: "adj_close",
    }
    data = data.rename({old: new for old, new in rename_map.items() if old in data.columns and old != new})

    required = {"date", "symbol"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Bar data missing required columns: {sorted(missing)}")

    data = normalize_date_column(data)
    fallback_price = _fallback_price_column(data)
    expressions = [
        pl.col("date"),
        pl.col("symbol").cast(pl.Utf8).str.to_uppercase().alias("symbol"),
    ]
    for column in ("open", "high", "low", "close", "adj_close"):
        source = pl.col(column) if column in data.columns else pl.col(fallback_price)
        expressions.append(source.cast(pl.Float64, strict=False).alias(column))
    volume = pl.col("volume") if "volume" in data.columns else pl.lit(None)
    expressions.append(volume.cast(pl.Float64, strict=False).alias("volume"))
    for column in OPTIONAL_BAR_COLUMNS:
        if column in data.columns:
            expressions.append(pl.col(column).cast(pl.Float64, strict=False).alias(column))

    output_columns = list(STANDARD_BAR_COLUMNS) + [column for column in OPTIONAL_BAR_COLUMNS if column in data.columns]
    return data.select(expressions).select(output_columns).sort(["date", "symbol"])


def validate_bars(
    frame,
    *,
    calendar: TradingCalendar | None = None,
    resolution: BarResolution = "any",
) -> DataQualityReport:
    """Validate normalized Polars bar data at a declared resolution.

    ``resolution="session"`` rejects intraday observations outright. That is not
    pedantry: the cross-sectional layer builds its price matrix by pivoting on
    `date` and takes returns between adjacent rows, so minute bars fed into it
    silently turn every overnight gap into an intraday return and leave the
    dividend- and split-adjusted `adj_close` semantics meaningless. The session
    resolution check rejects those inputs at the door.

    ``resolution="intraday"`` asserts the opposite: timestamps are instants, and
    with a session-aware calendar they must land inside a session window.
    """
    data = to_backend(frame, backend="polars")
    errors: list[str] = []
    warnings: list[str] = []

    missing = set(STANDARD_BAR_COLUMNS).difference(data.columns)
    if missing:
        errors.append(f"missing standard bar columns: {sorted(missing)}")
        return DataQualityReport(tuple(errors), tuple(warnings))

    null_keys = data.select((pl.col("date").is_null() | pl.col("symbol").is_null()).sum()).item()
    if int(null_keys):
        errors.append("date and symbol must be non-null")

    duplicate_count = data.select(pl.struct(["date", "symbol"]).is_duplicated().sum()).item()
    if int(duplicate_count):
        errors.append(f"duplicate date/symbol rows: {int(duplicate_count)}")

    for column in ("open", "high", "low", "close", "adj_close"):
        invalid_count = data.select((pl.col(column).is_null() | ~pl.col(column).is_finite() | (pl.col(column) <= 0)).sum()).item()
        if int(invalid_count):
            errors.append(f"{column} contains {int(invalid_count)} missing, nonfinite or non-positive values")

    high_low_invalid = data.select((pl.col("high") < pl.col("low")).sum()).item()
    if int(high_low_invalid):
        errors.append(f"high is below low on {int(high_low_invalid)} rows")

    close_outside_range = data.select(((pl.col("close") > pl.col("high")) | (pl.col("close") < pl.col("low"))).sum()).item()
    if int(close_outside_range):
        warnings.append(f"close is outside high/low range on {int(close_outside_range)} rows")

    open_outside_range = data.select(((pl.col("open") > pl.col("high")) | (pl.col("open") < pl.col("low"))).sum()).item()
    if int(open_outside_range):
        warnings.append(f"open is outside high/low range on {int(open_outside_range)} rows")

    negative_volume = data.select((pl.col("volume").is_not_null() & ((pl.col("volume") < 0) | ~pl.col("volume").is_finite())).sum()).item()
    if int(negative_volume):
        errors.append(f"volume contains {int(negative_volume)} negative or nonfinite values")

    if resolution == "session":
        per_session = (
            data.group_by([pl.col("date").dt.truncate("1d").alias("session"), "symbol"])
            .len()
            .filter(pl.col("len") > 1)
            .sort(["session", "symbol"])
        )
        if per_session.height:
            examples = [
                f"{row['symbol']}@{row['session'].date().isoformat()}x{row['len']}"
                for row in per_session.head(3).iter_rows(named=True)
            ]
            errors.append(
                f"session-resolution bars carry {per_session.height} symbol-sessions with "
                f"more than one observation: {examples}. Intraday data does not belong in the "
                "cross-sectional contract -- aggregate it with event.timeframes.resample_bars "
                "(timeframe='1d'), or run it through the event feed instead."
            )

    if calendar is not None:
        try:
            calendar.validate_timestamps(
                data["date"],
                label="bar timestamps",
                instants=resolution == "intraday" and calendar.has_hours,
            )
        except ValueError as exc:
            errors.append(str(exc))

    return DataQualityReport(tuple(errors), tuple(warnings))


def bars_to_events(frame, *, resolution: BarResolution = "any", calendar: TradingCalendar | None = None) -> list[Bar]:
    """Convert Polars-compatible long-form bars to event-engine Bar objects.

    The default resolution is deliberately permissive: `Bar.timestamp` is an
    instant by contract, so the event path accepts session labels and intraday
    instants alike. Pass ``resolution="intraday"`` with a session-aware calendar
    when the feed is meant to be intraday and you want pre-market prints and
    bars past an early close rejected at load time.
    """
    data = to_backend(frame, backend="polars")
    if set(STANDARD_BAR_COLUMNS).difference(data.columns):
        data = normalize_bars(data)
    else:
        data = data.sort(["date", "symbol"])
    report = validate_bars(data, calendar=calendar, resolution=resolution)
    report.raise_if_failed()
    bars = []
    has_vwap = "vwap" in data.columns
    for row in data.iter_rows(named=True):
        bars.append(
            Bar(
                symbol=str(row["symbol"]),
                timestamp=_to_datetime(row["date"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=None if row["volume"] is None else float(row["volume"]),
                adj_close=None if row["adj_close"] is None else float(row["adj_close"]),
                vwap=None if not has_vwap or row["vwap"] is None else float(row["vwap"]),
            )
        )
    return bars


def _fallback_price_column(data: pl.DataFrame) -> str:
    for column in ("adj_close", "close", "price"):
        if column in data.columns:
            return column
    raise ValueError("Bar data requires at least one price column: adj_close, close, or price")




def _to_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    parsed = pl.Series([value])
    if parsed.dtype == pl.String:
        parsed = parsed.str.to_datetime(strict=False)
    return parsed.cast(pl.Datetime("us"))[0]
