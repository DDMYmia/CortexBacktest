import json

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cortexbacktest.research.indicator_registry import INDICATORS, IndicatorRegistry
from cortexbacktest.pipelines.commands import _indicator_factory
from cortexbacktest.research.sweeps import build_indicator_sweep_cases


def prices(n=80):
    return pl.DataFrame({"date": range(n), "A": [10. + i + (i % 3) for i in range(n)]})


@pytest.mark.parametrize("name", INDICATORS.names())
def test_builtin_contract_matches_kernel_and_first_valid_row(name):
    data = prices()
    if name in {"average_true_range", "stochastic_oscillator"}:
        data = data.rename({"A": "close"}).with_columns(
            (pl.col("close") + 2).alias("high"), (pl.col("close") - 2).alias("low"))
    spec = INDICATORS.spec(name)
    actual = INDICATORS.run(name, data)
    expected = spec.func(data)
    assert_frame_equal(actual, expected)
    bars = INDICATORS.describe(name)["warmup_bars"]
    assert actual.drop("date").row(bars - 1)[0] is not None
    assert actual.drop("date").head(bars - 1).null_count().row(0)[0] == bars - 1
    json.dumps(INDICATORS.assess(name, data))


def test_missing_fields_and_renamed_ohlc():
    with pytest.raises(ValueError, match="missing_column:high"):
        INDICATORS.run("average_true_range", prices(), window=2)
    bars = pl.DataFrame({"date": [1, 2], "h": [3., 4.], "l": [1., 2.], "c": [2., 3.]})
    assert INDICATORS.run("average_true_range", bars, window=2,
                          high_col="h", low_col="l", close_col="c")["atr"][-1] == 2


def test_gap_restarts_each_symbols_warmup_without_zero_fill():
    data = pl.DataFrame({"date": range(7), "A": [1., 2., None, 4., 5., 6., 7.], "B": [1.] * 7})
    result = INDICATORS.run("builtin:momentum", data, window=2)
    assert result["A"].to_list()[:5] == [None] * 5
    assert result["A"][5] == .5
    assert result["B"][2] == 0
    assert INDICATORS.assess("momentum", data, window=2)["usable_rows"] == {"A": 2, "B": 5}


@pytest.mark.parametrize("factory", [lambda: _indicator_factory("momentum", {"window": 3}),
    lambda: build_indicator_sweep_cases("momentum", {"window": [3]})[0].factor])
def test_cli_and_sweep_share_warmup_gate(factory):
    with pytest.raises(ValueError, match="insufficient_complete_history"):
        factory()(prices(3))
    assert factory()(prices(4))["A"][-1] is not None


def test_context_extra_inputs_and_legacy_registration():
    registry = IndicatorRegistry()
    registry.register("custom", lambda df: df.select("date", "close"),
                      required_columns=("close",), extra_columns=("sector",),
                      min_warmup_bars=2, markets=("equity_us",), frequencies=("1d",))
    data = pl.DataFrame({"date": [1, 2], "close": [1., 2.], "sector": ["a", "a"]})
    with pytest.raises(ValueError, match="unsupported_or_missing_market"):
        registry.run("custom", data)
    context = {"market": "equity_us", "frequency": "1d"}
    assert registry.run("custom", data, input_context=context)["close"].to_list() == [None, 2.]
    assert "missing_column:sector" in registry.assess("custom", data.drop("sector"), input_context=context)["reasons"]
    registry.register("legacy", lambda df: df)
    assert_frame_equal(registry.run("legacy", data), data)


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_invalid_window(value):
    with pytest.raises(ValueError, match="positive integer"):
        INDICATORS.run("momentum", prices(), window=value)


def test_nonfinite_and_duplicate_input_rejected():
    with pytest.raises(ValueError, match="insufficient_complete_history"):
        INDICATORS.run("momentum", pl.DataFrame({"date": [1, 2], "A": [1., float("inf")]}), window=1)
    with pytest.raises(ValueError, match="duplicate_dates"):
        INDICATORS.run("momentum", pl.DataFrame({"date": [1, 1], "A": [1., 2.]}), window=1)
