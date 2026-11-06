"""New indicators against stored TA-Lib output, plus their edge behaviour."""

import json
from pathlib import Path

import polars as pl
import pytest

from cortexbacktest import INDICATORS, average_true_range, commodity_channel_index, on_balance_volume, rsi
from cortexbacktest.research.indicator_registry import IndicatorSpec

REFERENCE = json.loads((Path(__file__).parent / "fixtures/indicators/talib-reference.json").read_text())
ROWS = len(REFERENCE["close"])
BARS = pl.DataFrame({"date": list(range(ROWS)), **{k: REFERENCE[k] for k in ("high", "low", "close", "volume")}})
PRICES = BARS.select("date", pl.col("close").alias("A"))

# TA-Lib seeds these recursions differently, so only converged rows can match.
EXACT, CONVERGED = 1e-7, 1e-3
CASES = [
    ("rsi", PRICES, {"smoothing": "wilder"}, "A", "rsi_wilder", EXACT),
    ("bollinger_percent_b", PRICES, {}, "A", "percent_b", EXACT),
    ("williams_r", BARS, {}, "williams_r", "willr", EXACT),
    ("commodity_channel_index", BARS, {}, "cci", "cci", EXACT),
    ("on_balance_volume", BARS, {}, "obv", "obv", EXACT),
    ("money_flow_index", BARS, {}, "mfi", "mfi", EXACT),
    ("aroon_oscillator", BARS, {}, "aroon_oscillator", "aroon", EXACT),
    ("stochastic_d", BARS, {}, "stochastic_d", "stoch_d", EXACT),
    ("average_true_range", BARS, {"smoothing": "wilder"}, "atr", "atr_wilder", CONVERGED),
    ("normalized_average_true_range", BARS, {"smoothing": "wilder"}, "natr", "natr_wilder", CONVERGED),
    ("directional_indicators", BARS, {}, "plus_di", "plus_di", CONVERGED),
    ("directional_indicators", BARS, {}, "minus_di", "minus_di", CONVERGED),
    ("average_directional_index", BARS, {}, "adx", "adx", CONVERGED),
    ("macd_histogram", PRICES, {}, "A", "macd_hist", CONVERGED),
    ("percentage_price_oscillator", PRICES, {}, "A", "ppo_ema", CONVERGED),
]


@pytest.mark.parametrize(("name", "data", "params", "column", "reference", "tolerance"), CASES,
                         ids=[f"{case[0]}-{case[4]}" for case in CASES])
def test_matches_talib(name, data, params, column, reference, tolerance):
    actual = INDICATORS.run(name, data, **params)[column].to_list()
    start = 0 if tolerance == EXACT else ROWS - 10
    compared = [(a, b) for a, b in zip(actual[start:], REFERENCE[reference][start:]) if b is not None]
    assert len(compared) >= 10
    assert all(a == pytest.approx(b, abs=tolerance) for a, b in compared)


@pytest.mark.parametrize("name", INDICATORS.names())
def test_no_output_depends_on_a_later_bar(name):
    data = PRICES if INDICATORS.spec(name).input_kind == "prices" else BARS
    full = INDICATORS.run(name, data)
    cut = INDICATORS.run(name, data.head(120))
    assert full.head(120).equals(cut)


def test_chaikin_money_flow_and_vwap_ratio_by_hand():
    bars = pl.DataFrame({"date": [1, 2], "high": [10., 12.], "low": [8., 8.],
                         "close": [10., 9.], "volume": [100., 300.]})
    # Close locations are +1 and -0.5; typical prices are 28/3 and 29/3.
    assert INDICATORS.run("chaikin_money_flow", bars, window=2)["cmf"][-1] == pytest.approx(-50 / 400)
    vwap = (28 / 3 * 100 + 29 / 3 * 300) / 400
    assert INDICATORS.run("vwap_ratio", bars, window=2)["vwap_ratio"][-1] == pytest.approx(9 / vwap - 1)


@pytest.mark.parametrize(("name", "column"), [("williams_r", "williams_r"), ("stochastic_oscillator", "stochastic"),
                                              ("commodity_channel_index", "cci")])
def test_flat_window_gives_null_not_nan(name, column):
    flat = pl.DataFrame({"date": [1, 2, 3], "high": [5.] * 3, "low": [5.] * 3, "close": [5.] * 3})
    assert INDICATORS.run(name, flat, window=2)[column].to_list() == [None] * 3


def test_flat_prices_have_no_bollinger_position():
    flat = pl.DataFrame({"date": [1, 2, 3], "A": [5.] * 3})
    assert INDICATORS.run("bollinger_percent_b", flat, window=2)["A"].to_list() == [None] * 3


def test_zero_volume_window_gives_null():
    bars = pl.DataFrame({"date": [1, 2, 3], "high": [6., 7., 8.], "low": [4., 5., 6.],
                         "close": [5., 6., 7.], "volume": [0., 0., 0.]})
    assert INDICATORS.run("chaikin_money_flow", bars, window=2)["cmf"].to_list() == [None] * 3
    assert INDICATORS.run("vwap_ratio", bars, window=2)["vwap_ratio"].to_list() == [None] * 3
    assert INDICATORS.run("money_flow_index", bars, window=2)["mfi"].to_list() == [None] * 3


def test_a_missing_value_ends_on_balance_volume():
    bars = pl.DataFrame({"date": [1, 2, 3, 4], "close": [1., None, 3., 4.], "volume": [10., 10., 10., 10.]})
    assert on_balance_volume(bars)["obv"].to_list() == [10., None, None, None]


def test_aroon_counts_a_tie_from_its_latest_bar():
    bars = pl.DataFrame({"date": [1, 2, 3], "high": [9., 9., 8.], "low": [3., 2., 1.]})
    assert INDICATORS.run("aroon_oscillator", bars, window=2)["aroon_oscillator"][-1] == -50


@pytest.mark.parametrize("call", [lambda: rsi(PRICES, smoothing="ema"),
                                  lambda: average_true_range(BARS, smoothing="ema")])
def test_unknown_smoothing_is_refused(call):
    with pytest.raises(ValueError, match="smoothing"):
        call()


def test_cci_refuses_a_nonpositive_constant():
    with pytest.raises(ValueError, match="constant"):
        commodity_channel_index(BARS, constant=0)


def test_default_smoothing_keeps_the_rolling_mean():
    wilder = rsi(PRICES, smoothing="wilder")["A"][-1]
    assert rsi(PRICES)["A"][-1] != wilder
    assert INDICATORS.describe("rsi")["effective_parameters"]["smoothing"] == "simple"


def test_warmup_sums_several_parameters():
    assert INDICATORS.describe("macd_histogram", slow=10, signal=3)["warmup_bars"] == 12
    assert INDICATORS.describe("average_directional_index", window=5)["warmup_bars"] == 10
    spec = INDICATORS.spec("stochastic_d")
    assert isinstance(spec, IndicatorSpec) and spec.warmup_bars({"window": 4, "smooth": 2}) == 5
    with pytest.raises(ValueError, match="smooth must be a positive integer"):
        spec.warmup_bars({"smooth": 0})


def long_bars(symbol, rows=40):
    return BARS.head(rows).with_columns(pl.lit(symbol).alias("symbol"))


def test_run_panel_gives_one_matrix_column_per_symbol():
    bars = pl.concat([long_bars("aaa"), long_bars("bbb"), long_bars("short", 5)])
    matrix = INDICATORS.run_panel("williams_r", bars)
    assert matrix.columns == ["date", "AAA", "BBB"]
    expected = INDICATORS.run("williams_r", BARS.head(40))["williams_r"].drop_nulls()
    assert matrix["AAA"].to_list() == expected.to_list()


def test_run_panel_needs_a_column_for_several_outputs():
    with pytest.raises(ValueError, match="several outputs"):
        INDICATORS.run_panel("directional_indicators", long_bars("A"))
    assert "A" in INDICATORS.run_panel("directional_indicators", long_bars("A"), column="plus_di").columns


@pytest.mark.parametrize(("name", "bars", "message"), [
    ("rsi", long_bars("A"), "price matrix"),
    ("williams_r", BARS, "missing_column:symbol"),
    ("williams_r", long_bars("A", 3), "no symbol has enough"),
])
def test_run_panel_refusals(name, bars, message):
    with pytest.raises(ValueError, match=message):
        INDICATORS.run_panel(name, bars)
