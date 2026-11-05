"""Market data source adapters.

Adapters in this module normalize external data source outputs into the
standard long-form bar contract consumed by `datasets.load_backtest_dataset`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import pandas as pd

YFINANCE_COLUMN_ALIASES = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "AdjClose": "adj_close",
    "Volume": "volume",
}

STANDARD_BAR_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume", "adj_close")


def _empty_normalized_frame() -> pd.DataFrame:
    """Return an empty frame with the public normalized-bar columns."""
    import pandas as pd

    return pd.DataFrame({column: pd.Series(dtype="object") for column in STANDARD_BAR_COLUMNS})


def normalize_yfinance_history(frame: pd.DataFrame, *, symbols: str | Iterable[str] | None = None) -> pd.DataFrame:
    """Normalize yfinance history/download output to date/symbol/OHLCV rows."""
    import pandas as pd
    if frame.empty:
        return _empty_normalized_frame()
    symbol_list = _symbols(symbols)
    if isinstance(frame.columns, pd.MultiIndex):
        return _normalize_yfinance_multiindex(frame, symbols=symbol_list)
    return _normalize_yfinance_single(frame, symbol=symbol_list[0] if symbol_list else None)


def download_yfinance_bars(
    symbols: str | Iterable[str],
    *,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = False,
    progress: bool = False,
) -> pd.DataFrame:
    """Download yfinance bars and normalize them.

    This function requires installing the optional `data` extra.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install cortexbacktest[data] to use yfinance downloads") from exc

    symbol_list = _symbols(symbols)
    data = yf.download(
        tickers=symbol_list,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=progress,
        group_by="ticker",
    )
    if data is None:
        raise RuntimeError("yfinance returned no data frame")
    return normalize_yfinance_history(data, symbols=symbol_list)


def normalize_ccxt_ohlcv(rows, *, symbol: str) -> pd.DataFrame:
    """Normalize ccxt OHLCV rows to date/symbol/OHLCV bars.

    ccxt returns rows as [timestamp_ms, open, high, low, close, volume].
    """
    import pandas as pd
    # ccxt's untyped row payload is validated by _standardize_columns below.
    data = cast(
        Any,
        pd.DataFrame(cast(Any, rows), columns=cast(Any, ["timestamp", "open", "high", "low", "close", "volume"])),
    )
    if data.empty:
        return _empty_normalized_frame()
    data["date"] = pd.to_datetime(data["timestamp"], unit="ms", utc=True)
    data["symbol"] = str(symbol).upper()
    return _standardize_columns(data)


def fetch_ccxt_ohlcv_bars(
    exchange_id: str,
    symbol: str,
    *,
    timeframe: str = "1d",
    since: int | None = None,
    limit: int | None = None,
    exchange_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Fetch ccxt OHLCV bars and normalize them.

    This function requires installing the optional `data` extra.
    """
    try:
        import ccxt
    except ImportError as exc:
        raise ImportError("Install cortexbacktest[data] to use ccxt downloads") from exc

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class(exchange_kwargs or {})
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    return normalize_ccxt_ohlcv(rows, symbol=symbol)


def _normalize_yfinance_single(frame: pd.DataFrame, *, symbol: str | None) -> pd.DataFrame:
    import pandas as pd
    if symbol is None:
        name = getattr(frame, "name", None)
        symbol = str(name).upper() if name else "UNKNOWN"
    data = frame.copy()
    data = data.rename(columns=YFINANCE_COLUMN_ALIASES)
    data["date"] = pd.to_datetime(data.index)
    data["symbol"] = str(symbol).upper()
    return _standardize_columns(data)


def _normalize_yfinance_multiindex(frame: pd.DataFrame, *, symbols: list[str]) -> pd.DataFrame:
    import pandas as pd
    columns = frame.columns
    level_0 = [str(value) for value in columns.get_level_values(0)]
    first_level_fields = any(value in YFINANCE_COLUMN_ALIASES for value in level_0)
    if first_level_fields:
        field_level = 0
        symbol_level = 1
    else:
        field_level = 1
        symbol_level = 0

    rows = []
    discovered_symbols = list(dict.fromkeys(str(value).upper() for value in columns.get_level_values(symbol_level)))
    requested = symbols or discovered_symbols
    for symbol in requested:
        selected_columns = [
            column
            for column in columns
            if str(column[symbol_level]).upper() == symbol
        ]
        if not selected_columns:
            continue
        symbol_frame = frame.loc[:, selected_columns].copy()
        symbol_frame.columns = [column[field_level] for column in selected_columns]
        rows.append(_normalize_yfinance_single(symbol_frame, symbol=symbol))
    if not rows:
        return _empty_normalized_frame()
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def _standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    if "adj_close" not in data.columns and "close" in data.columns:
        data["adj_close"] = data["close"]
    required = set(STANDARD_BAR_COLUMNS)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"bar data missing required columns: {sorted(missing)}")
    output = data.loc[:, list(STANDARD_BAR_COLUMNS)].copy()
    output["date"] = pd.to_datetime(output["date"])
    output["symbol"] = output["symbol"].astype(str).str.upper()
    for column in ("open", "high", "low", "close", "volume", "adj_close"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output.dropna(subset=["date", "symbol", "open", "high", "low", "close", "adj_close"]).sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)


def _symbols(symbols: str | Iterable[str] | None) -> list[str]:
    if symbols is None:
        return []
    if isinstance(symbols, str):
        return [symbols.upper()]
    return [str(symbol).upper() for symbol in symbols]
