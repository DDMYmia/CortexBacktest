"""Technical factor builders."""

from __future__ import annotations

import polars as pl

from .frames import _to_polars_matrix


def momentum(prices: pl.DataFrame, *, window: int = 20) -> pl.DataFrame:
    """Price momentum over `window` periods as a Polars wide matrix."""
    return _native_factor(prices, "momentum", window)


def sma_ratio(prices: pl.DataFrame, *, window: int = 20) -> pl.DataFrame:
    """Price divided by simple moving average minus one as a Polars wide matrix."""
    return _native_factor(prices, "sma_ratio", window)


def ema_ratio(prices: pl.DataFrame, *, span: int = 20) -> pl.DataFrame:
    """Price divided by exponential moving average minus one as a Polars wide matrix."""
    _validate_window(span)
    data = _price_matrix_pl(prices)
    column = pl.col(_symbols(data))
    return data.select([pl.col("date"), column / column.ewm_mean(span=span, adjust=False, min_samples=span) - 1.0])


def rolling_volatility(
    prices: pl.DataFrame,
    *,
    window: int = 20,
    annualization: float = 252**0.5,
) -> pl.DataFrame:
    """Rolling realized volatility from close-to-close returns as a Polars wide matrix."""
    return _native_factor(prices, "rolling_volatility", window, annualization=annualization)


def risk_adjusted_momentum(prices: pl.DataFrame, *, window: int = 20) -> pl.DataFrame:
    """Window price return divided by unannualized sample return volatility.

    Every price in the window is required; zero variance is undefined, not an
    arbitrarily large signal. The same terms as `momentum` and `rolling_volatility`.
    """
    return _native_factor(prices, "risk_adjusted_momentum", window)


def rsi(prices: pl.DataFrame, *, window: int = 14) -> pl.DataFrame:
    """Relative strength index, scaled to 0-100, as a Polars wide matrix."""
    _validate_window(window)
    data = _price_matrix_pl(prices)
    delta = pl.col(_symbols(data)).diff()
    gain = pl.when(delta > 0).then(delta).when(delta <= 0).then(0.0).otherwise(None)
    loss = pl.when(delta < 0).then(-delta).when(delta >= 0).then(0.0).otherwise(None)
    average_gain = gain.rolling_mean(window_size=window, min_samples=window)
    average_loss = loss.rolling_mean(window_size=window, min_samples=window)
    relative_strength = average_gain / pl.when(average_loss == 0).then(None).otherwise(average_loss)
    return data.select([pl.col("date"),
        pl.when((average_loss == 0) & (average_gain > 0)).then(100.0)
        .when((average_gain == 0) & (average_loss > 0)).then(0.0)
        .otherwise(100.0 - (100.0 / (1.0 + relative_strength))).name.keep()])


def macd(prices: pl.DataFrame, *, fast: int = 12, slow: int = 26, signal: int = 9) -> pl.DataFrame:
    """MACD line as a Polars wide matrix."""
    _validate_window(fast)
    _validate_window(slow)
    _validate_window(signal)
    if fast >= slow:
        raise ValueError("fast must be < slow")
    data = _price_matrix_pl(prices)
    column = pl.col(_symbols(data))
    return data.select([pl.col("date"), column.ewm_mean(span=fast, adjust=False, min_samples=fast)
                                 - column.ewm_mean(span=slow, adjust=False, min_samples=slow)])


def bollinger_band_width(prices: pl.DataFrame, *, window: int = 20, num_std: float = 2.0) -> pl.DataFrame:
    """Bollinger band width as a Polars wide matrix."""
    _validate_window(window)
    data = _price_matrix_pl(prices)
    column = pl.col(_symbols(data))
    average = column.rolling_mean(window_size=window, min_samples=window)
    std = column.rolling_std(window_size=window, min_samples=window, ddof=1)
    return data.select([pl.col("date"), ((num_std * 2.0 * std) / average).name.keep()])


def rolling_zscore(prices: pl.DataFrame, *, window: int = 20) -> pl.DataFrame:
    """Rolling z-score as a Polars wide matrix."""
    return _native_factor(prices, "rolling_zscore", window)


def _native_factor(prices: pl.DataFrame, indicator: str, window: int, **kwargs) -> pl.DataFrame:
    _validate_window(window)
    data = _price_matrix_pl(prices)
    return data.select([pl.col("date"), _native_expression(indicator, pl.col(_symbols(data)), window, **kwargs)])


def _native_expression(indicator: str, column: pl.Expr, window: int, *, annualization: float = 252**0.5) -> pl.Expr:
    """Shared kernels for public matrices and independent received-revision windows."""
    trend = column / column.shift(window) - 1.0
    average = column.rolling_mean(window_size=window, min_samples=window)
    std = column.rolling_std(window_size=window, min_samples=window, ddof=1)
    volatility = column.pct_change().rolling_std(window_size=window, min_samples=window, ddof=1) * 1.
    if indicator == "risk_adjusted_momentum":
        ratio = trend / volatility
        complete = column.is_not_null().cast(pl.UInt32).rolling_sum(
            window_size=window + 1, min_samples=window + 1) == window + 1
        return pl.when(complete & (volatility > 0) & ratio.is_finite()).then(ratio).otherwise(None)
    return {"momentum": trend, "sma_ratio": column / average - 1.0,
            "rolling_zscore": (column - average) / std,
            "rolling_volatility": volatility * annualization}[indicator]


def average_true_range(
    bars: pl.DataFrame,
    *,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.DataFrame:
    """Average true range for a single OHLCV Polars table."""
    _validate_window(window)
    data = _to_polars_matrix(bars).sort("date")
    high = pl.col(high_col)
    low = pl.col(low_col)
    previous_close = pl.col(close_col).shift(1)
    true_range = pl.max_horizontal(high - low, (high - previous_close).abs(), (low - previous_close).abs())
    return data.select(
        [
            pl.col("date"),
            true_range.rolling_mean(window_size=window, min_samples=window).alias("atr"),
        ]
    )


def stochastic_oscillator(
    bars: pl.DataFrame,
    *,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.DataFrame:
    """Stochastic oscillator %K as a Polars table."""
    _validate_window(window)
    data = _to_polars_matrix(bars).sort("date")
    lowest_low = pl.col(low_col).rolling_min(window_size=window, min_samples=window)
    highest_high = pl.col(high_col).rolling_max(window_size=window, min_samples=window)
    return data.select(
        [
            pl.col("date"),
            ((pl.col(close_col) - lowest_low) / (highest_high - lowest_low) * 100.0).alias("stochastic"),
        ]
    )


def factor_long_form(
    factor,
    *,
    factor_name: str = "factor",
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pl.DataFrame:
    """Convert a Polars date x symbol factor matrix to long-form factor rows."""
    data = _to_polars_matrix(factor)
    symbols = _symbols(data)
    if not symbols:
        return pl.DataFrame({date_col: [], symbol_col: [], factor_name: []})
    return (
        data.select(["date", *symbols])
        .unpivot(index="date", variable_name=symbol_col, value_name=factor_name)
        .drop_nulls(["date", symbol_col, factor_name])
        .rename({"date": date_col})
        .sort([date_col, symbol_col])
    )


def _validate_window(window: int) -> None:
    if window < 1:
        raise ValueError("window must be >= 1")


def _price_matrix_pl(prices: pl.DataFrame) -> pl.DataFrame:
    data = _to_polars_matrix(prices).sort("date")
    return data.select(["date", *sorted(_symbols(data))])


def _symbols(frame: pl.DataFrame) -> list[str]:
    return [column for column in frame.columns if column != "date"]
