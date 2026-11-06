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


def rsi(prices: pl.DataFrame, *, window: int = 14, smoothing: str = "simple") -> pl.DataFrame:
    """Relative strength index, scaled to 0-100, as a Polars wide matrix.

    `smoothing="simple"` averages gains and losses over the window (Cutler's RSI);
    `smoothing="wilder"` is Wilder's recursive average, the usual charting value.
    """
    _validate_window(window)
    _validate_smoothing(smoothing)
    data = _price_matrix_pl(prices)
    delta = pl.col(_symbols(data)).diff()
    gain = pl.when(delta > 0).then(delta).when(delta <= 0).then(0.0).otherwise(None)
    loss = pl.when(delta < 0).then(-delta).when(delta >= 0).then(0.0).otherwise(None)
    average_gain = _smooth(gain, window, smoothing)
    average_loss = _smooth(loss, window, smoothing)
    relative_strength = average_gain / pl.when(average_loss == 0).then(None).otherwise(average_loss)
    return data.select([pl.col("date"),
        pl.when((average_loss == 0) & (average_gain > 0)).then(100.0)
        .when((average_gain == 0) & (average_loss > 0)).then(0.0)
        .otherwise(100.0 - (100.0 / (1.0 + relative_strength))).name.keep()])


def macd(prices: pl.DataFrame, *, fast: int = 12, slow: int = 26, signal: int = 9) -> pl.DataFrame:
    """MACD line as a Polars wide matrix."""
    _validate_fast_slow(fast, slow)
    _validate_window(signal)
    data = _price_matrix_pl(prices)
    return data.select([pl.col("date"), _macd_line(pl.col(_symbols(data)), fast, slow)])


def macd_histogram(prices: pl.DataFrame, *, fast: int = 12, slow: int = 26, signal: int = 9) -> pl.DataFrame:
    """MACD line minus its signal line as a Polars wide matrix."""
    _validate_fast_slow(fast, slow)
    _validate_window(signal)
    data = _price_matrix_pl(prices)
    line = _macd_line(pl.col(_symbols(data)), fast, slow)
    return data.select([pl.col("date"), (line - _ema(line, signal)).name.keep()])


def percentage_price_oscillator(prices: pl.DataFrame, *, fast: int = 12, slow: int = 26) -> pl.DataFrame:
    """Fast EMA over slow EMA minus one, in percent: MACD comparable across price levels."""
    _validate_fast_slow(fast, slow)
    data = _price_matrix_pl(prices)
    column = pl.col(_symbols(data))
    return data.select([pl.col("date"), ((_ema(column, fast) / _ema(column, slow) - 1.0) * 100.0).name.keep()])


def bollinger_band_width(prices: pl.DataFrame, *, window: int = 20, num_std: float = 2.0) -> pl.DataFrame:
    """Bollinger band width as a Polars wide matrix."""
    _validate_window(window)
    data = _price_matrix_pl(prices)
    column = pl.col(_symbols(data))
    average = column.rolling_mean(window_size=window, min_samples=window)
    std = column.rolling_std(window_size=window, min_samples=window, ddof=1)
    return data.select([pl.col("date"), ((num_std * 2.0 * std) / average).name.keep()])


def bollinger_percent_b(prices: pl.DataFrame, *, window: int = 20, num_std: float = 2.0) -> pl.DataFrame:
    """Position inside the Bollinger bands: 0 at the lower band, 1 at the upper band.

    Uses the sample standard deviation, as `bollinger_band_width` does. A flat
    window has no bands and gives null.
    """
    _validate_window(window)
    data = _price_matrix_pl(prices)
    column = pl.col(_symbols(data))
    average = column.rolling_mean(window_size=window, min_samples=window)
    std = column.rolling_std(window_size=window, min_samples=window, ddof=1)
    return data.select([pl.col("date"), _ratio(column - average + num_std * std, 2.0 * num_std * std).name.keep()])


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
    smoothing: str = "simple",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.DataFrame:
    """Average true range for a single OHLCV Polars table.

    The first bar has no previous close, so its true range is high minus low.
    `smoothing="wilder"` replaces the rolling mean with Wilder's recursive average.
    """
    _validate_window(window)
    _validate_smoothing(smoothing)
    data = _to_polars_matrix(bars).sort("date")
    true_range = _true_range(pl.col(high_col), pl.col(low_col), pl.col(close_col))
    return data.select([pl.col("date"), _smooth(true_range, window, smoothing).alias("atr")])


def normalized_average_true_range(
    bars: pl.DataFrame,
    *,
    window: int = 14,
    smoothing: str = "simple",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.DataFrame:
    """Average true range as a percentage of the close, comparable across price levels."""
    atr = average_true_range(bars, window=window, smoothing=smoothing,
                             high_col=high_col, low_col=low_col, close_col=close_col)["atr"]
    data = _to_polars_matrix(bars).sort("date").with_columns(atr.alias("_atr"))
    return data.select([pl.col("date"), (_ratio(pl.col("_atr"), pl.col(close_col)) * 100.0).alias("natr")])


def directional_indicators(
    bars: pl.DataFrame,
    *,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.DataFrame:
    """Wilder's +DI and -DI, scaled to 0-100, for a single OHLCV table."""
    _validate_window(window)
    data = _to_polars_matrix(bars).sort("date")
    plus_di, minus_di = _directional(pl.col(high_col), pl.col(low_col), pl.col(close_col), window)
    return data.select([pl.col("date"), plus_di.alias("plus_di"), minus_di.alias("minus_di")])


def average_directional_index(
    bars: pl.DataFrame,
    *,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.DataFrame:
    """Wilder's ADX: trend strength on a 0-100 scale, whatever the trend's direction."""
    _validate_window(window)
    data = _to_polars_matrix(bars).sort("date")
    plus_di, minus_di = _directional(pl.col(high_col), pl.col(low_col), pl.col(close_col), window)
    directional_index = _ratio((plus_di - minus_di).abs(), plus_di + minus_di) * 100.0
    return data.select([pl.col("date"), _wilder_mean(directional_index, window).alias("adx")])


def aroon_oscillator(
    bars: pl.DataFrame,
    *,
    window: int = 25,
    high_col: str = "high",
    low_col: str = "low",
) -> pl.DataFrame:
    """Aroon up minus Aroon down over `window + 1` bars, from -100 to 100.

    A tied high or low counts from its most recent bar.
    """
    _validate_window(window)
    data = _to_polars_matrix(bars).sort("date")
    high, low = pl.col(high_col), pl.col(low_col)
    since_high = _bars_since(high, high.rolling_max(window_size=window + 1, min_samples=window + 1), window)
    since_low = _bars_since(low, low.rolling_min(window_size=window + 1, min_samples=window + 1), window)
    return data.select([pl.col("date"), ((since_low - since_high) * (100.0 / window)).alias("aroon_oscillator")])


def stochastic_oscillator(
    bars: pl.DataFrame,
    *,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.DataFrame:
    """Stochastic oscillator %K as a Polars table. A flat window gives null."""
    _validate_window(window)
    data = _to_polars_matrix(bars).sort("date")
    percent_k = _percent_k(pl.col(high_col), pl.col(low_col), pl.col(close_col), window)
    return data.select([pl.col("date"), percent_k.alias("stochastic")])


def stochastic_d(
    bars: pl.DataFrame,
    *,
    window: int = 14,
    smooth: int = 3,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.DataFrame:
    """Stochastic %D: the `smooth`-bar simple mean of %K."""
    _validate_window(window)
    _validate_window(smooth)
    data = _to_polars_matrix(bars).sort("date")
    percent_k = _percent_k(pl.col(high_col), pl.col(low_col), pl.col(close_col), window)
    return data.select([pl.col("date"), percent_k.rolling_mean(window_size=smooth, min_samples=smooth).alias("stochastic_d")])


def williams_r(
    bars: pl.DataFrame,
    *,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.DataFrame:
    """Williams %R, from -100 at the window low to 0 at the window high."""
    _validate_window(window)
    data = _to_polars_matrix(bars).sort("date")
    percent_k = _percent_k(pl.col(high_col), pl.col(low_col), pl.col(close_col), window)
    return data.select([pl.col("date"), (percent_k - 100.0).alias("williams_r")])


def commodity_channel_index(
    bars: pl.DataFrame,
    *,
    window: int = 20,
    constant: float = 0.015,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.DataFrame:
    """Typical price distance from its mean, in units of `constant` mean deviations."""
    _validate_window(window)
    if constant <= 0:
        raise ValueError("constant must be > 0")
    data = _to_polars_matrix(bars).sort("date")
    typical = _typical_price(pl.col(high_col), pl.col(low_col), pl.col(close_col))
    average = typical.rolling_mean(window_size=window, min_samples=window)
    deviation = pl.sum_horizontal([(typical.shift(lag) - average).abs() for lag in range(window)]) / window
    return data.select([pl.col("date"), _ratio(typical - average, deviation * constant).alias("cci")])


def on_balance_volume(bars: pl.DataFrame, *, close_col: str = "close", volume_col: str = "volume") -> pl.DataFrame:
    """Cumulative volume signed by the close's direction, starting at the first volume.

    The level depends on where the table starts. A missing value ends the series,
    because no later total can be correct.
    """
    data = _to_polars_matrix(bars).sort("date")
    close, volume = pl.col(close_col), pl.col(volume_col)
    signed = pl.when(close.diff().is_null()).then(volume).otherwise(close.diff().sign() * volume)
    complete = (close.is_null() | volume.is_null()).cum_sum() == 0
    return data.select([pl.col("date"), pl.when(complete).then(signed.cum_sum()).alias("obv")])


def money_flow_index(
    bars: pl.DataFrame,
    *,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pl.DataFrame:
    """Volume-weighted RSI of the typical price, scaled to 0-100."""
    _validate_window(window)
    data = _to_polars_matrix(bars).sort("date")
    typical = _typical_price(pl.col(high_col), pl.col(low_col), pl.col(close_col))
    flow = typical * pl.col(volume_col)
    change = typical.diff()
    positive = pl.when(change > 0).then(flow).when(change <= 0).then(0.0)
    negative = pl.when(change < 0).then(flow).when(change >= 0).then(0.0)
    positive_sum = positive.rolling_sum(window_size=window, min_samples=window)
    negative_sum = negative.rolling_sum(window_size=window, min_samples=window)
    return data.select([pl.col("date"), (_ratio(positive_sum, positive_sum + negative_sum) * 100.0).alias("mfi")])


def chaikin_money_flow(
    bars: pl.DataFrame,
    *,
    window: int = 20,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pl.DataFrame:
    """Volume-weighted close location in the bar's range, from -1 to 1.

    A bar with no range contributes zero money flow but keeps its volume.
    """
    _validate_window(window)
    data = _to_polars_matrix(bars).sort("date")
    high, low, close, volume = (pl.col(name) for name in (high_col, low_col, close_col, volume_col))
    location = pl.when(high > low).then((2.0 * close - high - low) / (high - low)).when(high <= low).then(0.0)
    flow = (location * volume).rolling_sum(window_size=window, min_samples=window)
    return data.select([pl.col("date"), _ratio(flow, volume.rolling_sum(window_size=window, min_samples=window)).alias("cmf")])


def vwap_ratio(
    bars: pl.DataFrame,
    *,
    window: int = 20,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pl.DataFrame:
    """Close divided by the rolling volume-weighted typical price, minus one."""
    _validate_window(window)
    data = _to_polars_matrix(bars).sort("date")
    close, volume = pl.col(close_col), pl.col(volume_col)
    traded = (_typical_price(pl.col(high_col), pl.col(low_col), close) * volume).rolling_sum(
        window_size=window, min_samples=window)
    vwap = _ratio(traded, volume.rolling_sum(window_size=window, min_samples=window))
    return data.select([pl.col("date"), (close / vwap - 1.0).alias("vwap_ratio")])


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


def _validate_fast_slow(fast: int, slow: int) -> None:
    _validate_window(fast)
    _validate_window(slow)
    if fast >= slow:
        raise ValueError("fast must be < slow")


def _validate_smoothing(smoothing: str) -> None:
    if smoothing not in {"simple", "wilder"}:
        raise ValueError("smoothing must be simple or wilder")


def _ema(column: pl.Expr, span: int) -> pl.Expr:
    return column.ewm_mean(span=span, adjust=False, min_samples=span)


def _macd_line(column: pl.Expr, fast: int, slow: int) -> pl.Expr:
    return _ema(column, fast) - _ema(column, slow)


def _ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """Division that gives null, not infinity or NaN, on a zero denominator."""
    return pl.when(denominator != 0).then(numerator / denominator)


def _smooth(column: pl.Expr, window: int, smoothing: str) -> pl.Expr:
    if smoothing == "wilder":
        return _wilder_mean(column, window)
    return column.rolling_mean(window_size=window, min_samples=window)


def _wilder_mean(column: pl.Expr, window: int) -> pl.Expr:
    """Wilder's recursive average, seeded with the first complete simple mean."""
    seed = column.rolling_mean(window_size=window, min_samples=window)
    seen = seed.is_not_null().cast(pl.UInt32).cum_sum()
    seeded = pl.when(seen == 0).then(None).when(seen == 1).then(seed).otherwise(column)
    return seeded.ewm_mean(alpha=1.0 / window, adjust=False, min_samples=1)


def _true_range(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    previous_close = close.shift(1)
    return pl.max_horizontal(high - low, (high - previous_close).abs(), (low - previous_close).abs())


def _typical_price(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    return (high + low + close) / 3.0


def _percent_k(high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int) -> pl.Expr:
    lowest = low.rolling_min(window_size=window, min_samples=window)
    highest = high.rolling_max(window_size=window, min_samples=window)
    return _ratio(close - lowest, highest - lowest) * 100.0


def _bars_since(column: pl.Expr, extreme: pl.Expr, window: int) -> pl.Expr:
    """Bars back to the most recent row equal to `extreme`."""
    return pl.coalesce([pl.when(column.shift(lag) == extreme).then(pl.lit(float(lag))) for lag in range(window + 1)])


def _directional(high: pl.Expr, low: pl.Expr, close: pl.Expr, window: int) -> tuple[pl.Expr, pl.Expr]:
    up, down = high.diff(), -low.diff()
    plus = pl.when((up > down) & (up > 0)).then(up).when(up.is_not_null() & down.is_not_null()).then(0.0)
    minus = pl.when((down > up) & (down > 0)).then(down).when(up.is_not_null() & down.is_not_null()).then(0.0)
    true_range = pl.when(close.shift(1).is_not_null()).then(_true_range(high, low, close))
    average_range = _wilder_mean(true_range, window)
    return (_ratio(_wilder_mean(plus, window), average_range) * 100.0,
            _ratio(_wilder_mean(minus, window), average_range) * 100.0)


def _price_matrix_pl(prices: pl.DataFrame) -> pl.DataFrame:
    data = _to_polars_matrix(prices).sort("date")
    return data.select(["date", *sorted(_symbols(data))])


def _symbols(frame: pl.DataFrame) -> list[str]:
    return [column for column in frame.columns if column != "date"]
