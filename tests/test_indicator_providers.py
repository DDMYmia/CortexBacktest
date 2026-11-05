from datetime import date
import polars as pl
import pytest

from cortexbacktest.research.indicator_providers import (
    BuiltinPolarsProvider,
)
from cortexbacktest.research.indicator_registry import IndicatorRegistry


def test_builtin_provider():
    provider = BuiltinPolarsProvider()
    assert provider.provider_name == "builtin"
    assert provider.is_available() is True

    provider.register("custom_diff", lambda df: df.select([pl.col("close") - 1.0]))
    df = pl.DataFrame({"date": [date(2026, 1, 1)], "close": [10.0]})
    res = provider.compute("custom_diff", df)
    assert res["close"][0] == 9.0


def test_indicator_registry_pluggable():
    registry = IndicatorRegistry()
    assert "builtin" in registry.available_providers()

    df = pl.DataFrame({"date": [date(2026, 1, 1), date(2026, 1, 2)], "close": [10.0, 12.0]})
    builtin_prov = registry.get_provider("builtin")
    assert isinstance(builtin_prov, BuiltinPolarsProvider)

    # Test unknown provider raises
    with pytest.raises(KeyError, match="Unknown indicator provider"):
        registry.get_provider("nonexistent")

    # Test unavailable tulip / pandas_ta gracefully raises descriptive error
    tulip_prov = registry.get_provider("tulip")
    if not tulip_prov.is_available():
        with pytest.raises(ImportError, match="tulipy"):
            tulip_prov.compute("rsi", df)

    pta_prov = registry.get_provider("pandas_ta")
    if not pta_prov.is_available():
        with pytest.raises(ImportError, match="pandas-ta"):
            pta_prov.compute("rsi", df)


def test_build_indicator_sweep_cases():
    from cortexbacktest.research.sweeps import build_indicator_sweep_cases

    cases = build_indicator_sweep_cases(
        "rsi",
        {"window": [7, 14, 21]},
        base_name="rsi_opt",
    )
    assert len(cases) == 3
    assert cases[0].name == "rsi_opt_window7"
    assert cases[0].params == {"window": 7}
    assert cases[1].name == "rsi_opt_window14"
    assert cases[2].name == "rsi_opt_window21"


def test_tulip_inputs_follow_the_indicator_declaration(monkeypatch):
    """Inputs come from tulipy's metadata, not from retrying on any exception."""
    import numpy as np

    from cortexbacktest.research.indicator_providers import TulipIndicatorProvider

    calls = []

    def atr(high, low, close, period):
        calls.append(("atr", high[0], low[0], close[0], period))
        return np.ones(len(close) - period + 1)

    def sma(real, period):
        calls.append(("sma", real[0], period))
        raise FloatingPointError("a real failure must surface, not trigger another signature")

    atr.inputs, sma.inputs = ["high", "low", "close"], ["real"]
    provider = TulipIndicatorProvider()
    monkeypatch.setattr(provider, "_tulipy", type("Tulip", (), {"atr": staticmethod(atr), "sma": staticmethod(sma)}))
    monkeypatch.setattr(provider, "is_available", lambda: True)
    df = pl.DataFrame({"date": [1, 2, 3], "open": [1.0] * 3, "high": [3.0] * 3, "low": [1.0] * 3, "close": [2.0] * 3})

    out = provider.compute("atr", df, period=2)
    assert calls == [("atr", 3.0, 1.0, 2.0, 2)]
    assert out["atr"].to_list()[1:] == [1.0, 1.0]
    with pytest.raises(FloatingPointError):
        provider.compute("sma", df, period=2)
