import pandas as pd
import pytest

from cortexbacktest import normalize_ccxt_ohlcv, normalize_yfinance_history
from cortexbacktest.data.sources import download_yfinance_bars, fetch_ccxt_ohlcv_bars


def test_normalize_yfinance_single_symbol_history():
    frame = pd.DataFrame(
        {
            "Open": [100, 101],
            "High": [102, 103],
            "Low": [99, 100],
            "Close": [101, 102],
            "Adj Close": [100.5, 101.5],
            "Volume": [1_000, 1_100],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    bars = normalize_yfinance_history(frame, symbols="AAPL")

    assert bars["symbol"].tolist() == ["AAPL", "AAPL"]
    assert bars.loc[0, "adj_close"] == 100.5
    assert list(bars.columns) == ["date", "symbol", "open", "high", "low", "close", "volume", "adj_close"]


def test_normalize_yfinance_multiindex_ticker_first():
    columns = pd.MultiIndex.from_product(
        [["AAPL", "MSFT"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    frame = pd.DataFrame(
        [[100, 101, 99, 100.5, 100.0, 1_000, 200, 202, 198, 201, 200.5, 2_000]],
        index=pd.to_datetime(["2024-01-01"]),
        columns=columns,
    )

    bars = normalize_yfinance_history(frame)

    assert bars["symbol"].tolist() == ["AAPL", "MSFT"]
    assert bars.loc[bars["symbol"] == "MSFT", "close"].iloc[0] == 201


def test_normalize_yfinance_multiindex_field_first():
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["AAPL", "MSFT"]]
    )
    frame = pd.DataFrame(
        [[100, 200, 101, 202, 99, 198, 100.5, 201, 100.0, 200.5, 1_000, 2_000]],
        index=pd.to_datetime(["2024-01-01"]),
        columns=columns,
    )

    bars = normalize_yfinance_history(frame, symbols=["MSFT"])

    assert bars["symbol"].tolist() == ["MSFT"]
    assert bars.loc[0, "volume"] == 2_000


def test_download_yfinance_bars_requires_optional_dependency_if_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yfinance":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"\[data\]"):
        download_yfinance_bars("AAPL")


def test_normalize_ccxt_ohlcv_rows():
    rows = [
        [1_704_067_200_000, 42_000, 43_000, 41_000, 42_500, 123.4],
        [1_704_153_600_000, 42_500, 44_000, 42_000, 43_500, 234.5],
    ]

    bars = normalize_ccxt_ohlcv(rows, symbol="BTC/USDT")

    assert bars["symbol"].tolist() == ["BTC/USDT", "BTC/USDT"]
    assert bars.loc[0, "close"] == 42_500
    assert bars.loc[0, "adj_close"] == 42_500
    assert str(bars.loc[0, "date"].tz) == "UTC"


def test_normalize_ccxt_ohlcv_empty_rows():
    bars = normalize_ccxt_ohlcv([], symbol="ETH/USDT")

    assert bars.empty
    assert list(bars.columns) == ["date", "symbol", "open", "high", "low", "close", "volume", "adj_close"]


def test_fetch_ccxt_ohlcv_bars_requires_optional_dependency_if_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ccxt":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"\[data\]"):
        fetch_ccxt_ohlcv_bars("binance", "BTC/USDT")


def test_yfinance_download_preserves_single_use_symbol_iterable(monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    frame = pd.DataFrame({"Open": [100], "High": [102], "Low": [99], "Close": [101],
                          "Adj Close": [100.5], "Volume": [1000]},
                         index=pd.to_datetime(["2024-01-01"]))
    download = Mock(return_value=frame)
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download))
    bars = download_yfinance_bars(symbol for symbol in ["aapl"])
    assert download.call_args.kwargs["tickers"] == ["AAPL"]
    assert bars["symbol"].tolist() == ["AAPL"]
    assert bars["adj_close"].tolist() == [100.5]
