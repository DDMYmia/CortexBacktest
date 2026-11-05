"""Performance reporting from equity curves."""

from __future__ import annotations

from dataclasses import dataclass
import math


import polars as pl

from ..data.backend import to_backend


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    summary: dict[str, float]
    drawdowns: pl.DataFrame
    monthly_returns: pl.DataFrame
    annual_returns: pl.DataFrame


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Strategy-versus-benchmark comparison.

    Every frame carries a ``timestamp`` column plus its value column, so the
    records exporter treats them like any other Polars table.
    """

    summary: dict[str, float]
    aligned_returns: pl.DataFrame
    active_returns: pl.DataFrame
    rolling_beta: pl.DataFrame
    rolling_sharpe: pl.DataFrame


def returns_from_equity(equity, *, initial_equity: float | None = None) -> pl.DataFrame:
    """Period returns; an explicit starting capital includes the first period.

    Without a starting capital, the first period remains unknown. Never infer
    it from the first snapshot, which may already include entry costs.
    """
    if initial_equity is not None and (not math.isfinite(initial_equity) or initial_equity <= 0):
        raise ValueError("initial_equity must be finite and positive")
    frame = _equity_frame(equity)
    if frame.is_empty():
        return _empty_polars(["timestamp", "return"])
    return frame.sort("timestamp").select(
        [
            pl.col("timestamp"),
            (pl.col("equity").pct_change() if initial_equity is None else
             pl.col("equity") / pl.col("equity").shift(1, fill_value=initial_equity) - 1).alias("return"),
        ]
    )


def align_returns(
    strategy_returns,
    benchmark_returns,
    *,
    strategy_name: str = "strategy",
    benchmark_name: str = "benchmark",
) -> pl.DataFrame:
    """Inner-join strategy and benchmark returns on their common timestamps.

    Accepts return frames as produced by :func:`returns_from_equity`.
    """
    left = _returns_frame(strategy_returns, strategy_name)
    right = _returns_frame(benchmark_returns, benchmark_name)
    if left.is_empty() or right.is_empty():
        return _empty_polars(["timestamp", strategy_name, benchmark_name])
    return (
        left.join(right, on="timestamp", how="inner")
        .drop_nulls()
        .sort("timestamp")
    )


def _active_from_aligned(aligned: pl.DataFrame) -> pl.DataFrame:
    if aligned.is_empty():
        return _empty_polars(["timestamp", "active_return"])
    return aligned.select(
        pl.col("timestamp"),
        (pl.col("strategy") - pl.col("benchmark")).alias("active_return"),
    )


def active_returns(strategy_returns, benchmark_returns) -> pl.DataFrame:
    return _active_from_aligned(align_returns(strategy_returns, benchmark_returns))


def _beta_from_aligned(aligned: pl.DataFrame) -> float:
    if aligned.height < 2:
        return 0.0
    stats = aligned.select(
        pl.cov("strategy", "benchmark", ddof=1).alias("covariance"),
        pl.col("benchmark").var(ddof=1).alias("variance"),
    ).row(0, named=True)
    variance = stats["variance"]
    if variance is None or variance == 0 or math.isnan(variance):
        return 0.0
    covariance = stats["covariance"]
    if covariance is None or math.isnan(covariance):
        return 0.0
    return float(covariance / variance)


def beta(strategy_returns, benchmark_returns) -> float:
    return _beta_from_aligned(align_returns(strategy_returns, benchmark_returns))


def _alpha_from_aligned(aligned: pl.DataFrame, *, periods_per_year: int, risk_free_rate: float) -> float:
    if aligned.is_empty():
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    exposure = _beta_from_aligned(aligned)
    means = aligned.select(
        pl.col("strategy").mean().alias("strategy"),
        pl.col("benchmark").mean().alias("benchmark"),
    ).row(0, named=True)
    period_alpha = (means["strategy"] - period_rf) - exposure * (means["benchmark"] - period_rf)
    return float(period_alpha * periods_per_year)


def alpha(
    strategy_returns,
    benchmark_returns,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    return _alpha_from_aligned(
        align_returns(strategy_returns, benchmark_returns),
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )


def _information_ratio_from_aligned(aligned: pl.DataFrame, *, periods_per_year: int) -> float:
    active = _active_from_aligned(aligned).drop_nulls()
    if active.is_empty():
        return 0.0
    stats = active.select(
        pl.col("active_return").mean().alias("mean"),
        pl.col("active_return").std(ddof=1).alias("std"),
    ).row(0, named=True)
    std = stats["std"]
    if std is None or std == 0 or math.isnan(std):
        return 0.0
    return float(stats["mean"] / std * periods_per_year**0.5)


def information_ratio(
    strategy_returns,
    benchmark_returns,
    *,
    periods_per_year: int = 252,
) -> float:
    return _information_ratio_from_aligned(
        align_returns(strategy_returns, benchmark_returns),
        periods_per_year=periods_per_year,
    )


def rolling_sharpe(
    returns,
    *,
    window: int,
    periods_per_year: int = 252,
) -> pl.DataFrame:
    if window < 2:
        raise ValueError("window must be >= 2")
    frame = _returns_frame(returns, "return").drop_nulls()
    if frame.is_empty():
        return _empty_polars(["timestamp", "rolling_sharpe"])
    rolling_std = pl.col("return").rolling_std(window_size=window, ddof=1)
    return frame.select(
        pl.col("timestamp"),
        pl.when(rolling_std == 0)
        .then(None)
        .otherwise(pl.col("return").rolling_mean(window_size=window) / rolling_std * periods_per_year**0.5)
        .alias("rolling_sharpe"),
    )


def _rolling_beta_from_aligned(aligned: pl.DataFrame, *, window: int) -> pl.DataFrame:
    if window < 2:
        raise ValueError("window must be >= 2")
    if aligned.is_empty():
        return _empty_polars(["timestamp", "rolling_beta"])
    variance = pl.col("benchmark").rolling_var(window_size=window, ddof=1)
    return aligned.select(
        pl.col("timestamp"),
        pl.when(variance == 0)
        .then(None)
        .otherwise(pl.rolling_cov(pl.col("strategy"), pl.col("benchmark"), window_size=window) / variance)
        .alias("rolling_beta"),
    )


def rolling_beta(
    strategy_returns,
    benchmark_returns,
    *,
    window: int,
) -> pl.DataFrame:
    return _rolling_beta_from_aligned(align_returns(strategy_returns, benchmark_returns), window=window)


def _returns_frame(value, name: str) -> pl.DataFrame:
    """Normalize a returns input to a two-column ``timestamp``/``name`` frame."""
    if isinstance(value, pl.DataFrame):
        if value.is_empty():
            return _empty_polars(["timestamp", name])
        if name in value.columns:
            return value.select(pl.col("timestamp"), pl.col(name))
        # The value column is whichever non-timestamp column is present.
        columns = [column for column in value.columns if column != "timestamp"]
        if len(columns) != 1:
            raise ValueError(f"expected one value column beside timestamp, found {columns}")
        return value.select(pl.col("timestamp"), pl.col(columns[0]).alias(name))
    raise TypeError("returns must be a Polars DataFrame with timestamp and value columns")


def drawdown_series(equity) -> pl.DataFrame:
    """Return point-in-time drawdown values as a Polars DataFrame."""
    frame = _equity_frame(equity)
    if frame.is_empty():
        return _empty_polars(["timestamp", "drawdown"])
    return frame.sort("timestamp").select(
        [
            pl.col("timestamp"),
            (pl.col("equity") / pl.col("equity").cum_max() - 1.0).alias("drawdown"),
        ]
    )


def drawdown_table(equity) -> pl.DataFrame:
    """Return drawdown episodes with start, valley, end, and depth as Polars."""
    dd = drawdown_series(equity)
    if dd.is_empty():
        return _empty_polars(["start", "valley", "end", "depth", "duration"])

    rows = []
    in_drawdown = False
    start = None
    valley = None
    depth = 0.0
    values = dd.drop_nulls(["drawdown"]).iter_rows(named=True)
    last_timestamp = None
    for row in values:
        timestamp = row["timestamp"]
        value = row["drawdown"]
        last_timestamp = timestamp
        if value < 0 and not in_drawdown:
            in_drawdown = True
            start = timestamp
            valley = timestamp
            depth = float(value)
        elif value < 0 and in_drawdown:
            if value < depth:
                depth = float(value)
                valley = timestamp
        elif value >= 0 and in_drawdown:
            rows.append(_drawdown_row(start, valley, timestamp, depth))
            in_drawdown = False
            start = None
            valley = None
            depth = 0.0
    if in_drawdown:
        rows.append(_drawdown_row(start, valley, last_timestamp, depth))
    if not rows:
        return _empty_polars(["start", "valley", "end", "depth", "duration"])
    return pl.DataFrame(rows).select(["start", "valley", "end", "depth", "duration"])


def period_returns(equity, freq: str) -> pl.DataFrame:
    """Compute period returns as a Polars DataFrame."""
    frame = _equity_frame(equity)
    if frame.is_empty():
        return _empty_polars(["timestamp", "return"])
    every = _polars_period(freq)
    period_end = (
        frame.sort("timestamp")
        .group_by_dynamic("timestamp", every=every)
        .agg(pl.col("equity").last())
        .drop_nulls(["equity"])
    )
    if period_end.is_empty():
        return _empty_polars(["timestamp", "return"])
    baseline = frame.sort("timestamp").head(1)
    values = pl.concat([baseline, period_end]).select(["timestamp", "equity"])
    return values.with_columns(pl.col("equity").pct_change().alias("return")).slice(1).select(["timestamp", "return"])


def performance_report(equity, *, periods_per_year: int = 252) -> PerformanceReport:
    """Build a performance report using Polars tables and reductions."""
    frame = _equity_frame(equity)
    if frame.is_empty():
        return PerformanceReport(
            summary={
                "total_return": 0.0,
                "cagr": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
            },
            drawdowns=drawdown_table(frame),
            monthly_returns=_empty_polars(["timestamp", "return"]),
            annual_returns=_empty_polars(["timestamp", "return"]),
        )
    returns = returns_from_equity(frame)
    return PerformanceReport(
        summary={
            "total_return": _total_return(frame),
            "cagr": _cagr(frame, periods_per_year=periods_per_year),
            "sharpe": _sharpe_ratio(returns, periods_per_year=periods_per_year),
            "max_drawdown": _max_drawdown(frame),
            "volatility": _volatility(returns, periods_per_year=periods_per_year),
        },
        drawdowns=drawdown_table(frame),
        monthly_returns=period_returns(frame, "ME"),
        annual_returns=period_returns(frame, "YE"),
    )


def benchmark_report(
    strategy_equity,
    benchmark_equity,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    rolling_window: int = 20,
) -> BenchmarkReport:
    """Compare a strategy equity curve against a benchmark equity curve.

    Both inputs are equity curves as ``timestamp``/``equity`` frames; see
    :func:`_equity_frame` for the accepted shapes.
    """
    strategy_frame = _equity_frame(strategy_equity)
    benchmark_frame = _equity_frame(benchmark_equity)
    aligned = align_returns(
        returns_from_equity(strategy_frame),
        returns_from_equity(benchmark_frame),
    )
    if aligned.is_empty():
        return BenchmarkReport(
            summary={
                "strategy_total_return": 0.0,
                "benchmark_total_return": 0.0,
                "active_total_return": 0.0,
                "alpha": 0.0,
                "beta": 0.0,
                "information_ratio": 0.0,
                "correlation": 0.0,
                "tracking_error": 0.0,
            },
            aligned_returns=aligned,
            active_returns=_empty_polars(["timestamp", "active_return"]),
            rolling_beta=_empty_polars(["timestamp", "rolling_beta"]),
            rolling_sharpe=_empty_polars(["timestamp", "rolling_sharpe"]),
        )

    active = _active_from_aligned(aligned)
    stats = aligned.select(
        pl.corr("strategy", "benchmark").alias("correlation"),
        (pl.col("strategy") - pl.col("benchmark")).std(ddof=1).alias("tracking_error"),
    ).row(0, named=True)
    correlation = stats["correlation"] if aligned.height > 1 else 0.0
    tracking_error = stats["tracking_error"] if aligned.height > 1 else 0.0

    # Total returns come from the equity curves restricted to shared timestamps,
    # so a benchmark covering a wider window cannot flatter either side.
    common = strategy_frame.join(benchmark_frame, on="timestamp", how="inner").drop_nulls()
    strategy_total_return = _total_return(common.select(pl.col("timestamp"), pl.col("equity")))
    benchmark_total_return = _total_return(
        common.select(pl.col("timestamp"), pl.col("equity_right").alias("equity"))
    )
    return BenchmarkReport(
        summary={
            "strategy_total_return": strategy_total_return,
            "benchmark_total_return": benchmark_total_return,
            "active_total_return": strategy_total_return - benchmark_total_return,
            "alpha": _alpha_from_aligned(
                aligned,
                periods_per_year=periods_per_year,
                risk_free_rate=risk_free_rate,
            ),
            "beta": _beta_from_aligned(aligned),
            "information_ratio": _information_ratio_from_aligned(aligned, periods_per_year=periods_per_year),
            "correlation": _finite_or_zero(correlation),
            "tracking_error": _finite_or_zero(tracking_error) * periods_per_year**0.5,
        },
        aligned_returns=aligned,
        active_returns=active,
        rolling_beta=_rolling_beta_from_aligned(aligned, window=rolling_window),
        rolling_sharpe=rolling_sharpe(
            aligned.select(pl.col("timestamp"), pl.col("strategy").alias("return")),
            window=rolling_window,
            periods_per_year=periods_per_year,
        ),
    )


def _finite_or_zero(value) -> float:
    if value is None or math.isnan(value):
        return 0.0
    return float(value)


def _total_return(equity: pl.DataFrame) -> float:
    values = equity.select("equity").drop_nulls()
    if values.height < 2:
        return 0.0
    first = values.item(0, "equity")
    last = values.item(values.height - 1, "equity")
    return float(last / first - 1.0) if first else 0.0


def _cagr(equity: pl.DataFrame, *, periods_per_year: int = 252) -> float:
    values = equity.select("equity").drop_nulls()
    if values.height < 2:
        return 0.0
    first = values.item(0, "equity")
    last = values.item(values.height - 1, "equity")
    years = (values.height - 1) / periods_per_year
    if first <= 0 or years <= 0:
        return 0.0
    return float((last / first) ** (1 / years) - 1)


def _sharpe_ratio(returns: pl.DataFrame, *, periods_per_year: int = 252) -> float:
    clean = returns.select(pl.col("return").cast(pl.Float64, strict=False).alias("return")).drop_nulls()
    if clean.is_empty():
        return 0.0
    std = clean.select(pl.col("return").std(ddof=1)).item()
    if std is None or std == 0 or math.isnan(std):
        return 0.0
    mean = clean.select(pl.col("return").mean()).item()
    return float(mean / std * math.sqrt(periods_per_year))


def _max_drawdown(equity: pl.DataFrame) -> float:
    drawdowns = drawdown_series(equity)
    if drawdowns.is_empty():
        return 0.0
    value = drawdowns.select(pl.col("drawdown").min()).item()
    return 0.0 if value is None else float(value)


def _volatility(returns: pl.DataFrame, *, periods_per_year: int = 252) -> float:
    clean = returns.select(pl.col("return").cast(pl.Float64, strict=False).alias("return")).drop_nulls()
    if clean.height <= 1:
        return 0.0
    std = clean.select(pl.col("return").std(ddof=1)).item()
    return 0.0 if std is None or math.isnan(std) else float(std * math.sqrt(periods_per_year))


def _drawdown_row(start, valley, end, depth: float) -> dict:
    return {
        "start": start,
        "valley": valley,
        "end": end,
        "depth": depth,
        "duration": end - start,
    }


def _equity_frame(equity) -> pl.DataFrame:
    """Normalize an equity curve to a ``timestamp``/``equity`` Polars frame."""
    frame = to_backend(equity, backend="polars")
    if frame.is_empty():
        return _empty_polars(["timestamp", "equity"])
    if "timestamp" not in frame.columns:
        if "date" in frame.columns:
            frame = frame.rename({"date": "timestamp"})
        else:
            raise ValueError("equity data requires a timestamp or date column")
    if "equity" not in frame.columns:
        raise ValueError("equity data requires an equity column")
    return (
        frame.select(
            [
                pl.col("timestamp").cast(pl.Datetime("us", "UTC"), strict=False).alias("timestamp"),
                pl.col("equity").cast(pl.Float64, strict=False).alias("equity"),
            ]
        )
        .drop_nulls(["timestamp", "equity"])
        .sort("timestamp")
    )


def _polars_period(freq: str) -> str:
    normalized = freq.upper()
    if normalized in {"ME", "M", "MS"}:
        return "1mo"
    if normalized in {"YE", "Y", "YS", "A", "AE"}:
        return "1y"
    return freq


def _empty_polars(columns: list[str]) -> pl.DataFrame:
    return pl.DataFrame({column: [] for column in columns})
