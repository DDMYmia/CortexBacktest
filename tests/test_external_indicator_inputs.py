import importlib.util
import json

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cortexbacktest.research.indicator_registry import INDICATORS, IndicatorRegistry
from cortexbacktest.pipelines.commands import _indicator_factory
from cortexbacktest.research.sweeps import build_indicator_sweep_cases


def prices(n=12):
    return pl.DataFrame({"date": range(n), "A": [10.0 + i + (i % 3) for i in range(n)]})


@pytest.mark.parametrize("provider,param", [("tulip", "period"), ("pandas_ta", "length")])
@pytest.mark.parametrize(
    "name,columns,offset",
    [
        ("sma", ["close"], 0),
        ("ema", ["close"], 0),
        ("rsi", ["close"], 1),
        ("atr", ["high", "low", "close"], 0),
    ],
)
def test_metadata_is_serializable_and_parameter_dependent_without_optional_install(
    provider, param, name, columns, offset
):
    actual = IndicatorRegistry().describe(provider + ":" + name, **{param: 7})
    assert actual["requirements_known"] and actual["required_columns"] == columns
    assert actual["warmup_bars"] == 7 + (1 if provider == "pandas_ta" and name == "atr" else offset)
    assert actual["dependencies"] and actual["parameters"] == {param: 7}
    json.dumps(actual)


@pytest.mark.parametrize(
    "params",
    [
        {"period": True},
        {"period": 0},
        {"period": 2.5},
        {"offset": -1},
        {"fillna": 0},
        {"lookahead": True},
        {"talib": True},
    ],
)
def test_unreviewed_parameters_are_rejected(params):
    with pytest.raises(ValueError):
        IndicatorRegistry().describe("tulip:sma", **params)


def test_unknown_metadata_is_not_guessed_and_unknown_factor_cli_refuses():
    registry = IndicatorRegistry()
    assert registry.describe("tulip:unreviewed")["requirements_known"] is False
    assert (
        "provider_requirements_unknown" in registry.assess("tulip:unreviewed", prices())["reasons"]
    )
    with pytest.raises(ValueError, match="reviewed"):
        _indicator_factory("pandas_ta:vwap", {})
    with pytest.raises(ValueError, match="wide-price"):
        _indicator_factory("tulip:atr", {"period": 2})


def stub(monkeypatch, registry):
    provider = registry.get_provider("tulip")
    calls = []
    monkeypatch.setattr(provider, "is_available", lambda: True)

    def compute(name, data, period):
        calls.append(data)
        if name == "atr":
            return data.select(
                "date", (pl.col("high") - pl.col("low")).rolling_mean(period).alias("atr")
            )
        return data.select(
            "date", *[pl.col(c).rolling_mean(period).alias(c) for c in data.columns if c != "date"]
        )

    monkeypatch.setattr(provider, "_compute", compute)
    return provider, calls


def test_wide_gaps_restart_kernel_per_asset_and_do_not_fill(monkeypatch):
    registry = IndicatorRegistry()
    _, calls = stub(monkeypatch, registry)
    data = pl.DataFrame(
        {
            "date": range(9),
            "A": [1.0, 2.0, 3.0, None, 10.0, 11.0, 12.0, 13.0, 14.0],
            "B": [None] * 8 + [2.0],
        },
        schema_overrides={"B": pl.Float64},
    )
    out = registry.run("tulip:sma", data, period=3)
    assert out["A"].to_list() == [None, None, 2.0, None, None, None, 11.0, 12.0, 13.0]
    assert out["B"].null_count() == 9
    assert [f.height for f in calls] == [3, 5]
    assert registry.assess("tulip:sma", data, period=3)["usable_rows"] == {"A": 4, "B": 0}


def test_ohlc_requires_only_actual_inputs_and_resets_joint_history(monkeypatch):
    registry = IndicatorRegistry()
    _, calls = stub(monkeypatch, registry)
    data = pl.DataFrame(
        {"date": range(5), "high": [3.0, 4.0, None, 6.0, 7.0], "low": [1.0] * 5, "close": [2.0] * 5}
    )
    assert registry.run("tulip:atr", data, period=2)["atr"].to_list() == [
        None,
        2.5,
        None,
        None,
        5.5,
    ]
    assert [f.height for f in calls] == [2, 2]
    assert registry.assess("tulip:atr", data.drop("high"), period=2)["reasons"] == [
        "missing_column:high"
    ]


def test_cli_sweep_and_direct_provider_share_history_refusal(monkeypatch):
    provider, _ = stub(monkeypatch, INDICATORS)

    def direct(df):
        return provider.compute("sma", df, period=3)

    cli = _indicator_factory("tulip:sma", {"period": 3})
    sweep = build_indicator_sweep_cases("tulip:sma", {"period": [3]})[0].factor
    for function in [direct, cli, sweep]:
        with pytest.raises(ValueError, match="insufficient_complete_history"):
            function(prices(2))
        assert function(prices(3))["A"][0] is None
        assert function(prices(3))["A"][-1] is not None


def test_dependency_unavailable_is_part_of_assessment(monkeypatch):
    registry = IndicatorRegistry()
    provider = registry.get_provider("tulip")
    monkeypatch.setattr(provider, "is_available", lambda: False)
    assert registry.assess("tulip:sma", prices(), period=3)["reasons"] == [
        "provider_dependency_unavailable"
    ]
    with pytest.raises(ImportError, match="tulipy"):
        registry.run("tulip:sma", prices(), period=3)


@pytest.mark.parametrize("provider,param", [("tulip", "period"), ("pandas_ta", "length")])
@pytest.mark.parametrize("name", ["sma", "ema", "rsi", "atr"])
def test_real_provider_outputs_match_clean_segments_and_are_prefix_causal(provider, param, name):
    module = "tulipy" if provider == "tulip" else "pandas_ta_classic"
    if importlib.util.find_spec(module) is None:
        pytest.skip("optional real-provider validation runs in dedicated environment")
    registry = IndicatorRegistry()
    p = registry.get_provider(provider)
    data = prices(20)
    if name == "atr":
        data = data.rename({"A": "close"}).with_columns(
            (pl.col("close") + 2).alias("high"), (pl.col("close") - 2).alias("low")
        )
    kwargs = {param: 3}
    warmup = registry.describe(provider + ":" + name, **kwargs)["warmup_bars"]
    output = registry.run(provider + ":" + name, data, **kwargs)
    raw = p._compute(name, data, **kwargs)
    for col in output.columns:
        if col == "date":
            continue
        assert output[col].head(warmup - 1).null_count() == warmup - 1
        np.testing.assert_allclose(
            output[col].slice(warmup - 1).to_numpy(),
            raw[col].slice(warmup - 1).to_numpy(),
            atol=1e-12,
            equal_nan=True,
        )
    prefix = registry.run(provider + ":" + name, data.head(10), **kwargs)
    assert_frame_equal(prefix, output.head(10))
    field = "close" if name == "atr" else "A"
    gapped = data.with_columns(
        pl.when(pl.col("date") == 7).then(None).otherwise(pl.col(field)).alias(field)
    )
    output = registry.run(provider + ":" + name, gapped, **kwargs)
    tail = registry.run(provider + ":" + name, gapped.slice(8), **kwargs)
    assert_frame_equal(output.slice(8), tail)
    assert output.drop("date").slice(7, warmup).null_count().row(0)[0] == warmup


def test_actual_cli_case_parser_preserves_provider_namespace(monkeypatch):
    from cortexbacktest.pipelines.commands import _parse_sweep_case

    stub(monkeypatch, INDICATORS)
    case = _parse_sweep_case(
        "trial=tulip:sma:period=3",
        preprocessing=None,
        top_quantile=0.8,
        bottom_quantile=None,
        transaction_cost_bps=10.0,
    )
    assert case.params == {"indicator": "tulip:sma", "period": 3}
    assert case.factor(prices(4))["A"][-1] is not None
    with pytest.raises(ValueError, match="provider:indicator"):
        _parse_sweep_case(
            "trial=tulip:period=3",
            preprocessing=None,
            top_quantile=0.8,
            bottom_quantile=None,
            transaction_cost_bps=10.0,
        )
