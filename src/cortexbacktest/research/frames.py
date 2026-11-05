"""DataFrame builders for prices, factors, and forward returns.

Returns here are differences between *adjacent rows*. For one-row-per-session
data that is a close-to-close return and needs no qualification. For intraday
rows it is not: the step from a session's last bar to the next session's first
bar is an overnight move, and folding it into a series of intraday returns
overstates the volatility of every minute at the open and attributes an
overnight gap to a strategy that was flat overnight.

So these builders ask which one they are looking at instead of assuming.
Intraday input without a calendar is refused rather than silently mixed; with
one, the overnight step is separated out into `overnight_returns_matrix` rather
than dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ..data.backend import normalize_date_column, to_backend

if TYPE_CHECKING:
    from ..data.calendar import TradingCalendar


def price_matrix(
    frame: pl.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    price_col: str = "adj_close",
) -> pl.DataFrame:
    """Build a date x symbol price matrix as a Polars DataFrame."""
    return _matrix_pl(frame, date_col=date_col, symbol_col=symbol_col, value_col=price_col)


def returns_matrix(prices: pl.DataFrame, *, calendar: "TradingCalendar | None" = None) -> pl.DataFrame:
    """Returns between adjacent rows; on intraday data, within a session only.

    One row per session gives close-to-close returns, unchanged. Intraday rows
    need a ``calendar`` so the session each row belongs to is knowable; the
    first observation of every session then returns null, because its
    predecessor is on the other side of an overnight gap. That move is not
    discarded -- `overnight_returns_matrix` returns exactly those steps.
    """
    data = _to_polars_matrix(prices).sort("date")
    sessions = _sessions(data, calendar, caller="returns_matrix")
    value_columns = [column for column in data.columns if column != "date"]
    returns = data.select("date", pl.col(value_columns).pct_change())
    if sessions is None:
        return returns
    return _mask(returns, ~_starts_session(sessions), value_columns)


def overnight_returns_matrix(prices: pl.DataFrame, *, calendar: "TradingCalendar") -> pl.DataFrame:
    """The steps `returns_matrix` leaves out: last bar of a session to the next first.

    Rows inside a session are null, so this composes with `returns_matrix` --
    every adjacent-row move appears in exactly one of the two.
    """
    data = _to_polars_matrix(prices).sort("date")
    sessions = _sessions(data, calendar, caller="overnight_returns_matrix", require=True)
    assert sessions is not None
    value_columns = [column for column in data.columns if column != "date"]
    returns = data.select("date", pl.col(value_columns).pct_change())
    return _mask(returns, _starts_session(sessions), value_columns)


def forward_returns_matrix(
    prices: pl.DataFrame,
    periods: int = 1,
    *,
    calendar: "TradingCalendar | None" = None,
) -> pl.DataFrame:
    """Forward returns from row T to row T + periods, within one session intraday.

    On intraday data a horizon that runs past the close would otherwise reach
    into the next session and quietly become an overnight-plus-intraday return.
    Those rows are null: the horizon does not exist inside the session.
    """
    if periods < 1:
        raise ValueError("periods must be >= 1")
    data = _to_polars_matrix(prices).sort("date")
    sessions = _sessions(data, calendar, caller="forward_returns_matrix")
    value_columns = [column for column in data.columns if column != "date"]
    forward = data.select("date", pl.col(value_columns).shift(-periods) / pl.col(value_columns) - 1.0)
    if sessions is None:
        return forward
    same_session = sessions == sessions.shift(-periods)
    return _mask(forward, same_session.fill_null(False), value_columns)


def _sessions(data: pl.DataFrame, calendar: "TradingCalendar | None", *,
              caller: str, require: bool = False) -> pl.Series | None:
    """Session key per row, or None when the data is already one row per session.

    Sessions do not span midnight in this calendar, so the local date is the
    session key. Tz-aware timestamps are converted first; naive ones are wall
    clock in the calendar zone, the convention `data.calendar` states.
    """
    column = data["date"]
    if calendar is not None and calendar.zone is not None and getattr(column.dtype, "time_zone", None):
        column = column.dt.convert_time_zone(calendar.zone)
    keys = column.dt.date()
    intraday = keys.n_unique() < len(keys)
    if not intraday and not require:
        return None
    if calendar is None:
        raise ValueError(
            f"{caller} received intraday rows without a calendar. Adjacent-row returns would "
            "fold each overnight gap into an intraday return. Pass a session-aware calendar, "
            "or aggregate to one row per session first."
        )
    return keys


def _starts_session(sessions: pl.Series) -> pl.Series:
    return (sessions != sessions.shift(1)).fill_null(True)


def _mask(returns: pl.DataFrame, keep: pl.Series, value_columns: list[str]) -> pl.DataFrame:
    return returns.with_columns(pl.when(keep).then(pl.col(value_columns)).otherwise(None).name.keep())


def factor_matrix(
    frame: pl.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    factor_col: str = "factor",
) -> pl.DataFrame:
    """Build a date x symbol factor matrix as a Polars DataFrame."""
    return _matrix_pl(frame, date_col=date_col, symbol_col=symbol_col, value_col=factor_col)


def stack_factor_and_forward_returns(
    factors: pl.DataFrame,
    forward_returns,
    *,
    factor_name: str = "factor",
) -> pl.DataFrame:
    """Stack Polars factor and forward-return matrices into research long form."""
    factor_frame = _to_polars_matrix(factors)
    forward_frame = _to_polars_matrix(forward_returns)
    symbols = sorted((set(factor_frame.columns) & set(forward_frame.columns)) - {"date"})
    if not symbols:
        return pl.DataFrame({"date": [], "symbol": [], factor_name: [], "forward_return": []})
    factor_long = (
        factor_frame.select(["date", *symbols])
        .unpivot(index="date", variable_name="symbol", value_name=factor_name)
        .drop_nulls(["date", "symbol", factor_name])
    )
    forward_long = (
        forward_frame.select(["date", *symbols])
        .unpivot(index="date", variable_name="symbol", value_name="forward_return")
        .drop_nulls(["date", "symbol", "forward_return"])
    )
    return (
        factor_long.join(forward_long, on=["date", "symbol"], how="inner")
        .select(["date", "symbol", factor_name, "forward_return"])
        .sort(["date", "symbol"])
    )


def _matrix_pl(frame: pl.DataFrame, *, date_col: str, symbol_col: str, value_col: str) -> pl.DataFrame:
    data = to_backend(frame, backend="polars")
    missing = {date_col, symbol_col, value_col}.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = normalize_date_column(data, date_col)
    pivoted = (
        data.select(
            [
                pl.col(date_col).alias("date"),
                pl.col(symbol_col).cast(pl.Utf8).str.to_uppercase().alias("symbol"),
                pl.col(value_col).cast(pl.Float64, strict=False).alias("value"),
            ]
        )
        .drop_nulls(["date", "symbol", "value"])
        .pivot(index="date", on="symbol", values="value", aggregate_function="last")
        .sort("date")
    )
    return pivoted.select(["date", *sorted(column for column in pivoted.columns if column != "date")])


def _to_polars_matrix(frame: pl.DataFrame) -> pl.DataFrame:
    data = to_backend(frame, backend="polars")
    if "date" not in data.columns:
        raise ValueError("wide matrices require a date column")
    return normalize_date_column(data)
