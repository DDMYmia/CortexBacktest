"""Data subsystem: calendars, universes, and point-in-time masks."""


from .backend import (
    DataFrameBackend as DataFrameBackend,
    read_table as read_table,
    scan_table as scan_table,
    to_backend as to_backend,
    to_pandas as to_pandas,
    write_table as write_table,
)
from .calendar import (
    SessionWindow as SessionWindow,
    TradingCalendar as TradingCalendar,
    infer_calendar_from_index as infer_calendar_from_index,
    infer_session_hours as infer_session_hours,
)
from .futures import (
    ContractRoll as ContractRoll,
    RollAdjustment as RollAdjustment,
    build_continuous_futures_bars as build_continuous_futures_bars,
)
from .sources import (
    download_yfinance_bars as download_yfinance_bars,
    fetch_ccxt_ohlcv_bars as fetch_ccxt_ohlcv_bars,
    normalize_ccxt_ohlcv as normalize_ccxt_ohlcv,
    normalize_yfinance_history as normalize_yfinance_history,
)
from .history import (
    HistoricalDataStore as HistoricalDataStore,
    OrderBookRecorder as OrderBookRecorder,
    OrderBookReplayer as OrderBookReplayer,
    TickRecorder as TickRecorder,
    TickReplayer as TickReplayer,
    order_book_snapshot_to_tick as order_book_snapshot_to_tick,
)
from .universe import (
    Universe as Universe,
    apply_universe_mask as apply_universe_mask,
)

# Keep the public export list statically visible to type checkers.
__all__ = [
    "DataFrameBackend", "read_table",
    "scan_table", "to_backend", "to_pandas", "write_table", "SessionWindow", "TradingCalendar",
    "infer_calendar_from_index", "infer_session_hours", "ContractRoll", "RollAdjustment",
    "build_continuous_futures_bars", "download_yfinance_bars", "fetch_ccxt_ohlcv_bars",
    "normalize_ccxt_ohlcv", "normalize_yfinance_history", "HistoricalDataStore", "OrderBookRecorder",
    "OrderBookReplayer", "TickRecorder", "TickReplayer", "order_book_snapshot_to_tick", "Universe",
    "apply_universe_mask",
]
