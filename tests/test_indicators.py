import math

import polars as pl
import pytest

from cortexbacktest import (
    average_true_range,
    bollinger_band_width,
    ema_ratio,
    factor_long_form,
    macd,
    momentum,
    rolling_volatility,
    rolling_zscore,
    rsi,
    sma_ratio,
    stochastic_oscillator,
)

DATES = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]


def _prices() -> pl.DataFrame:
    return pl.DataFrame({"date": DATES, "A": [10.0, 11.0, 12.0, 13.0, 14.0], "B": [20.0, 19.0, 18.0, 17.0, 16.0]})


def _cell(frame: pl.DataFrame, day: int, symbol: str):
    return frame.row(day - 1, named=True)[symbol]


def test_momentum_factor():
    result = momentum(_prices(), window=2)

    # 12/10 - 1 = 0.2 for A, 18/20 - 1 = -0.1 for B, on 2024-01-03 (row 3).
    assert round(_cell(result, 3, "A"), 4) == 0.2
    assert round(_cell(result, 3, "B"), 4) == -0.1


def test_sma_and_ema_ratio_factors():
    prices = _prices()

    sma = sma_ratio(prices, window=2)
    ema = ema_ratio(prices, span=2)

    assert round(_cell(sma, 2, "A"), 4) == round(11 / 10.5 - 1, 4)
    assert _cell(ema, 1, "A") is None
    assert _cell(ema, 5, "A") > 0


def test_rolling_volatility_and_rsi():
    prices = _prices()

    vol = rolling_volatility(prices, window=2, annualization=1)
    strength = rsi(prices, window=2)

    assert _cell(vol, 5, "A") >= 0
    # A rises monotonically, B falls monotonically.
    assert _cell(strength, 5, "A") > 90
    assert _cell(strength, 5, "B") < 10


def test_factor_long_form():
    factor = momentum(_prices(), window=1)

    rows = factor_long_form(factor, factor_name="mom_1")

    assert rows.columns == ["date", "symbol", "mom_1"]
    assert set(rows["symbol"].to_list()) == {"A", "B"}


def test_indicator_window_validation():
    with pytest.raises(ValueError, match="window"):
        momentum(_prices(), window=0)


def test_additional_price_indicators_are_polars_frames():
    prices = _prices()

    for frame in (
        macd(prices, fast=2, slow=3, signal=2),
        bollinger_band_width(prices, window=3),
        rolling_zscore(prices, window=3),
    ):
        assert isinstance(frame, pl.DataFrame)
        assert frame.columns == ["date", "A", "B"]
        assert frame.height == len(DATES)


def test_rolling_zscore_normalizes_a_trending_window():
    prices = pl.DataFrame({"date": DATES[:3], "A": [1.0, 2.0, 3.0]})

    result = rolling_zscore(prices, window=3)

    # z-score of the last point of [1, 2, 3] using sample std (ddof=1): (3-2)/1.
    assert _cell(result, 3, "A") == pytest.approx(1.0)


def test_ohlc_indicators_return_named_columns():
    bars = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "high": [11.0, 12.0, 13.0, 14.0],
            "low": [9.0, 10.0, 10.0, 12.0],
            "close": [10.0, 11.0, 12.0, 13.0],
        }
    )

    atr = average_true_range(bars, window=2)
    stoch = stochastic_oscillator(bars, window=2)

    assert "atr" in atr.columns
    assert "stochastic" in stoch.columns
    # ATR is a positive true-range average; the last %K sits in [0, 100].
    assert atr["atr"][-1] > 0
    assert 0.0 <= stoch["stochastic"][-1] <= 100.0
    assert not math.isnan(stoch["stochastic"][-1])
