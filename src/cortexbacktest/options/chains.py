"""Vendor option-chain ingestion: raw files -> one long schema -> engine frames.

The research kernel in this package consumes `OptionContract`/`OptionMarketFrame`
objects; every public option-chain dataset ships something else. This module is
the boundary between the two: it normalises a vendor table into the canonical
long schema below, reports and filters the quote defects those datasets are known
to contain, stores the result as hive-partitioned Parquet, and adapts it back to
the strict engine objects.

Canonical schema (`CHAIN_SCHEMA`), one row per quoted contract per observation::

    quote_date underlying underlying_price expiry dte strike right
    option_symbol bid ask mid implied_volatility delta gamma theta vega rho
    volume open_interest

`quote_date` and `expiry` are timezone-aware UTC instants, not dates: an
end-of-day file carries a date, and the instant that date stands for is a policy
decision (`ChainTimePolicy`), not a fact in the file. Nothing here infers a
settlement fixing, a publication time, or displayed size from daily volume.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from math import isfinite
from pathlib import Path
import re
from typing import Iterable, Iterator, Literal, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from uuid import uuid4

import polars as pl

from ..datasets.schema import DataQualityReport
from .contracts import OptionContract, OptionLifecycleEvent, OptionMarketFrame, utc
from .experiments import option_frames_from_polars

CHAIN_SCHEMA: Mapping[str, pl.DataType] = {
    "quote_date": pl.Datetime("us", "UTC"),
    "underlying": pl.String(),
    "underlying_price": pl.Float64(),
    "expiry": pl.Datetime("us", "UTC"),
    "dte": pl.Int32(),
    "strike": pl.Float64(),
    "right": pl.String(),
    "option_symbol": pl.String(),
    "bid": pl.Float64(),
    "ask": pl.Float64(),
    "mid": pl.Float64(),
    "implied_volatility": pl.Float64(),
    "delta": pl.Float64(),
    "gamma": pl.Float64(),
    "theta": pl.Float64(),
    "vega": pl.Float64(),
    "rho": pl.Float64(),
    "volume": pl.Float64(),
    "open_interest": pl.Float64(),
}
CHAIN_COLUMNS: tuple[str, ...] = tuple(CHAIN_SCHEMA)
CHAIN_KEY: tuple[str, ...] = ("quote_date", "underlying", "expiry", "strike", "right")
# Fields a vendor may omit entirely; everything else must be resolvable.
_OPTIONAL_FIELDS = frozenset({
    "underlying_price", "implied_volatility", "delta", "gamma", "theta", "vega", "rho",
    "volume", "open_interest",
})
_GREEKS = ("delta", "gamma", "theta", "vega", "rho")
_CALL_WORDS = frozenset({"c", "call", "calls", "1"})
_PUT_WORDS = frozenset({"p", "put", "puts", "0", "-1"})


def _key(name: str) -> str:
    """Vendor headers differ only by case, spaces and brackets: `[C_BID]` == `c_bid`."""
    return re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")


@dataclass(frozen=True, slots=True)
class ChainTimePolicy:
    """Turns vendor dates into the instants the engine compares.

    `quote_time`/`expiry_time` are times of day attached to a date-only column;
    a column that already carries a real time of day keeps it. They are read in
    `zone`, which defaults to UTC.

    Set `zone` for any dataset spanning a daylight-saving change. A US market
    close is 16:00 America/New_York, which is 20:00 UTC in summer and 21:00 in
    winter; pinning one UTC time of day across fourteen years puts half the
    sample an hour off, and that is a real error whenever a chain is joined to
    anything else with its own timestamps. With `zone` set, each date is
    converted on its own, so the transition weeks come out right.

    `publication_lag` is the delay between an observation and the moment a
    caller could have acted on it; it becomes `available_at`, and the frame
    timestamp.
    """

    quote_time: time = time(20, 0)
    expiry_time: time = time(20, 0)
    publication_lag: timedelta = timedelta(0)
    zone: str | None = None

    def __post_init__(self) -> None:
        if self.quote_time.tzinfo is not None or self.expiry_time.tzinfo is not None:
            raise ValueError("times of day carry no tzinfo; name a zone in `zone` instead")
        if self.publication_lag < timedelta(0):
            raise ValueError("publication_lag must be nonnegative")
        if self.zone is not None:
            try:
                ZoneInfo(self.zone)
            except (ZoneInfoNotFoundError, ValueError) as error:
                raise ValueError(f"unknown time zone {self.zone!r}") from error


@dataclass(frozen=True, slots=True)
class OptionChainSource:
    """How one vendor's table maps onto the canonical schema.

    `layout="long"` is one row per contract; `layout="wide"` is one row per
    strike with call and put columns side by side (OptionsDX). `aliases` maps a
    canonical field to accepted header names, matched after `_key` normalisation
    and in order, so an earlier alias wins over a later one.
    """

    name: str
    layout: Literal["long", "wide"] = "long"
    aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    call_prefix: str = "c"
    put_prefix: str = "p"
    default_underlying: str | None = None
    requires_underlying_argument: bool = False

    def resolve(self, columns: Iterable[str], prefix: str = "") -> dict[str, str]:
        """Canonical field -> actual column name, for the fields present."""
        available = {_key(column): column for column in columns}
        resolved: dict[str, str] = {}
        for canonical, names in self.aliases.items():
            for alias in names:
                candidate = f"{prefix}_{alias}" if prefix else alias
                if candidate in available:
                    resolved[canonical] = available[candidate]
                    break
        return resolved


_SHARED_ALIASES: dict[str, tuple[str, ...]] = {
    "quote_date": ("quote_date", "date", "trade_date", "data_date", "quotedate", "asof_date"),
    "underlying": ("underlying", "act_symbol", "underlying_symbol", "root", "ticker", "symbol"),
    "underlying_price": ("underlying_price", "underlying_last", "stock_price", "spot",
                         "underlying_close"),
    "expiry": ("expiry", "expiration", "expire_date", "expiration_date", "exp_date"),
    "strike": ("strike", "strike_price"),
}
_LONG_ALIASES: dict[str, tuple[str, ...]] = {
    **_SHARED_ALIASES,
    "right": ("right", "option_type", "call_put", "put_call", "type", "cp_flag"),
    "bid": ("bid", "bid_price", "best_bid"),
    "ask": ("ask", "ask_price", "best_ask", "offer"),
    "implied_volatility": ("implied_volatility", "iv", "impl_volatility", "sigma"),
    **{greek: (greek,) for greek in _GREEKS},
    "volume": ("volume", "trade_volume"),
    "open_interest": ("open_interest", "openinterest", "oi"),
}
# Per-leg fields of a wide file; the shared ones stay unprefixed.
_WIDE_LEG_ALIASES: dict[str, tuple[str, ...]] = {
    "bid": ("bid",), "ask": ("ask",),
    "implied_volatility": ("iv", "implied_volatility"),
    **{greek: (greek,) for greek in _GREEKS},
    "volume": ("volume",), "open_interest": ("open_interest", "oi"),
}

SOURCES: Mapping[str, OptionChainSource] = {
    # DoltHub post-no-preference/options: long, one row per contract, IV in `vol`,
    # no underlying price column (join one in with `with_underlying_prices`).
    "dolthub": OptionChainSource(
        "dolthub", "long",
        {**_LONG_ALIASES,
         "implied_volatility": ("vol", *_LONG_ALIASES["implied_volatility"])}),
    # Kaggle dudesurfin/spy-options-eod-volatility-surface-2010-2023: one parquet
    # per year, and despite being a Kaggle export it carries OptionsDX's wide,
    # bracketed layout (`[QUOTE_DATE]`, `[C_IV]`, `[P_IV]`, ...). The ticker is
    # SPY for the whole dataset and appears in no column.
    "kaggle_spy_eod": OptionChainSource(
        "kaggle_spy_eod", "wide",
        {**{k: v for k, v in _SHARED_ALIASES.items() if k != "underlying"},
         **_WIDE_LEG_ALIASES},
        default_underlying="SPY"),
    # A generic long export. `vol` is deliberately not an IV alias here: outside
    # DoltHub it is as likely to mean volume.
    "long": OptionChainSource("long", "long", _LONG_ALIASES),
    # OptionsDX monthly CSV/TXT: `[QUOTE_DATE]`, `[C_BID]`, `[P_BID]`, ... and no
    # ticker column, because the ticker is the file name.
    "optionsdx": OptionChainSource(
        "optionsdx", "wide",
        {**{k: v for k, v in _SHARED_ALIASES.items() if k != "underlying"},
         **_WIDE_LEG_ALIASES},
        requires_underlying_argument=True),
}


def _resolve_source(source: str | OptionChainSource) -> OptionChainSource:
    if isinstance(source, OptionChainSource):
        return source
    if source not in SOURCES:
        raise ValueError(f"unknown chain source {source!r}; known: {sorted(SOURCES)}")
    return SOURCES[source]


def _frame(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return data.collect() if isinstance(data, pl.LazyFrame) else data


def _timestamps(data: pl.DataFrame, column: str, moment: time,
                zone: str | None = None) -> pl.Series:
    """Vendor date/datetime column -> aware UTC instants.

    A date, or a datetime whose every time of day is midnight, is a date-only
    column and gets `moment` (read in `zone`, default UTC); anything else
    already carries its own time.
    """
    values = data[column]
    if values.dtype == pl.String:
        parsed = values.str.strip_chars().str.to_datetime(time_unit="us", strict=False)
        if parsed.null_count() > values.null_count():
            raise ValueError(f"column {column!r} contains unparseable timestamps")
        values = parsed
    dated = False
    if values.dtype == pl.Date:
        values, dated = values.dt.combine(moment), True
    elif isinstance(values.dtype, pl.Datetime):
        if values.dtype.time_zone is not None:
            return values.dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC"))
        if bool(values.dt.time().eq(time(0, 0)).all()):
            values, dated = values.dt.date().dt.combine(moment), True
    else:
        raise ValueError(f"column {column!r} is not a date, datetime or parseable string")
    values = values.cast(pl.Datetime("us"))
    if zone is not None and dated:
        # Per-date conversion, so a sample spanning a daylight-saving change is
        # right on both sides of it.
        return values.dt.replace_time_zone(zone).dt.convert_time_zone("UTC")
    return values.dt.replace_time_zone("UTC")


_TYPE_WORDS = {**{word: "CALL" for word in _CALL_WORDS}, **{word: "PUT" for word in _PUT_WORDS}}


def _right(values: pl.Series) -> pl.Series:
    words = values.cast(pl.String, strict=False).str.strip_chars().str.to_lowercase()
    resolved = words.replace_strict(_TYPE_WORDS, default=None, return_dtype=pl.String)
    if resolved.null_count() > values.null_count():
        unknown = sorted({w for w, r in zip(words, resolved) if r is None and w is not None})[:5]
        raise ValueError(f"unrecognised option type values: {unknown}")
    return resolved


def occ_symbol(underlying: str, expiry: datetime | date, right: str, strike: float) -> str:
    """OCC 21-character contract symbol, e.g. `SPY   220715P00380000`."""
    day = expiry.date() if isinstance(expiry, datetime) else expiry
    thousandths = round(strike * 1000)
    if not isfinite(strike) or strike <= 0 or thousandths >= 10 ** 8:
        raise ValueError(f"strike {strike} is not representable in an OCC symbol")
    return (f"{underlying.upper():<6}{day:%y%m%d}"
            f"{'C' if right == 'CALL' else 'P'}{thousandths:08d}")


def _occ_expression() -> pl.Expr:
    """`occ_symbol` as one expression, so a million-row file is not built row by row."""
    thousandths = (pl.col("strike") * 1000).round(0).cast(pl.Int64)
    return pl.concat_str(
        pl.col("underlying").str.pad_end(6),
        pl.col("expiry").dt.strftime("%y%m%d"),
        pl.when(pl.col("right") == "CALL").then(pl.lit("C")).otherwise(pl.lit("P")),
        thousandths.cast(pl.String).str.pad_start(8, "0"),
    )


def _leg(data: pl.DataFrame, source: OptionChainSource, prefix: str, right: str,
         shared: dict[str, str], overrides: Mapping[str, str]) -> pl.DataFrame:
    resolved = {**shared, **source.resolve(data.columns, prefix), **overrides}
    selected = data.select([pl.col(name).alias(canonical)
                            for canonical, name in resolved.items()])
    return selected.with_columns(pl.lit(right).alias("right"))


def normalize_option_chain(
    data: pl.DataFrame | pl.LazyFrame,
    source: str | OptionChainSource = "long",
    *,
    underlying: str | None = None,
    time_policy: ChainTimePolicy | None = None,
    overrides: Mapping[str, str] | None = None,
) -> pl.DataFrame:
    """Vendor table -> the canonical long chain schema.

    `overrides` maps a canonical field to an exact column name and wins over the
    source's aliases — the escape hatch for a file whose headers this module has
    never seen. `underlying` supplies the ticker for a source that keeps it in the
    file name rather than a column, and overrides any column if given.

    Nothing is dropped or repaired here: the output can still contain crossed,
    zero-bid or stale rows. Run `validate_option_chain`/`clean_option_chain` next.
    """
    spec = _resolve_source(source)
    policy = time_policy or ChainTimePolicy()
    overrides = dict(overrides or {})
    frame = _frame(data)
    if frame.is_empty():
        return pl.DataFrame(schema=dict(CHAIN_SCHEMA))
    if unknown := set(overrides) - set(CHAIN_SCHEMA):
        raise ValueError(f"overrides name unknown fields: {sorted(unknown)}")
    underlying = underlying or spec.default_underlying
    if spec.requires_underlying_argument and underlying is None:
        raise ValueError(f"source {spec.name!r} carries no ticker column; pass underlying=")

    if spec.layout == "wide":
        if leg_overrides := set(overrides) & set(_WIDE_LEG_ALIASES):
            raise ValueError("a wide source stores these per leg; rename the columns or use a "
                             f"custom OptionChainSource instead of overrides: {sorted(leg_overrides)}")
        shared = {c: n for c, n in spec.resolve(frame.columns).items()
                  if c not in _WIDE_LEG_ALIASES}
        legs = [_leg(frame, spec, prefix, right, shared, overrides)
                for prefix, right in ((spec.call_prefix, "CALL"), (spec.put_prefix, "PUT"))]
        common = [c for c in legs[0].columns if c in legs[1].columns]
        frame = pl.concat([leg.select(common) for leg in legs], how="vertical")
        resolved = {canonical: canonical for canonical in frame.columns}
    else:
        resolved = {**spec.resolve(frame.columns), **overrides}
        frame = frame.select([pl.col(name).alias(canonical)
                              for canonical, name in resolved.items()])
    if underlying is not None:
        frame = frame.with_columns(pl.lit(underlying).alias("underlying"))
        resolved["underlying"] = "underlying"
    missing = [c for c in CHAIN_SCHEMA
               if c not in resolved and c not in _OPTIONAL_FIELDS
               and c not in ("dte", "mid", "option_symbol")]
    if missing:
        raise ValueError(f"source {spec.name!r} is missing required columns: {missing} "
                         f"(seen: {sorted(_key(c) for c in _frame(data).columns)[:20]})")

    # The two columns whose handling depends on the vendor's dtype, resolved
    # against the schema before the rest becomes one lazy pipeline.
    frame = frame.with_columns(
        _timestamps(frame, "quote_date", policy.quote_time, policy.zone).alias("quote_date"),
        _timestamps(frame, "expiry", policy.expiry_time, policy.zone).alias("expiry"),
        _right(frame["right"]).alias("right"),
    )
    present = set(frame.columns)
    casts = [pl.col("underlying").cast(pl.String).str.strip_chars().str.to_uppercase()]
    # Padded values are stripped before the cast, and only where the column
    # arrived as text. OptionsDX writes `", 323.590000"` -- a space after every
    # comma -- and `cast(Float64, strict=False)` turns ` 323.59` into null
    # without complaining, so an unstripped ingest of one of those files lands
    # every price, strike and Greek as null and fails somewhere else entirely.
    schema = frame.schema
    casts += [(pl.col(column).str.strip_chars() if schema[column] == pl.String()
               else pl.col(column)).cast(pl.Float64, strict=False)
              for column, dtype in CHAIN_SCHEMA.items()
              if column in present and dtype == pl.Float64()]
    casts += [pl.lit(None, CHAIN_SCHEMA[column]).alias(column)
              for column in CHAIN_SCHEMA if column not in present]
    # One pass over the data instead of a with_columns per column: each of those
    # is a separate pipeline the engine cannot fuse, which on a 600 MB Kaggle
    # year is a copy of the frame per column rather than per file.
    derived = (
        frame.lazy()
        .with_columns(casts)
        .with_columns(
            ((pl.col("expiry").dt.date() - pl.col("quote_date").dt.date())
             .dt.total_days().cast(pl.Int32)).alias("dte"),
            ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
            _occ_expression().alias("option_symbol"),
            ((pl.col("strike") * 1000).round(0) >= 10 ** 8).alias("_unrepresentable"),
        )
        .select(*CHAIN_COLUMNS, "_unrepresentable")
        .sort(list(CHAIN_KEY))
        .collect()
    )
    unrepresentable = int(derived["_unrepresentable"].sum())
    if unrepresentable:
        raise ValueError(f"{unrepresentable} strikes are too large for an OCC symbol")
    return derived.drop("_unrepresentable")


def scan_chain_file(path: str | Path) -> pl.LazyFrame:
    """Lazily open a vendor chain file. Parquet by extension, otherwise CSV/TXT."""
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pl.scan_parquet(path)
    return pl.scan_csv(path, infer_schema_length=None, try_parse_dates=False)


def iter_normalized_chunks(
    source_file: str | Path | pl.LazyFrame,
    source: str | OptionChainSource = "long",
    *,
    dates_per_chunk: int = 20,
    underlying: str | None = None,
    time_policy: ChainTimePolicy | None = None,
    overrides: Mapping[str, str] | None = None,
) -> Iterator[pl.DataFrame]:
    """Normalise a large file in bounded memory, one group of sessions at a time.

    A Kaggle year is ~600 MB on disk and several times that once materialised
    with Greeks, so reading the whole file to normalise it is the difference
    between an ingest that runs anywhere and one that needs the machine's whole
    memory. Each chunk is filtered out of the file lazily, so only the sessions
    being normalised are ever resident, and the caller can write each chunk to
    the store before the next is read — the store appends, so the result on disk
    is identical to a single-shot ingest.

    Chunking is by *session*, never by row count: a session's rows must stay
    together or the put-call parity screen and the frame adapter would see
    half-built cross-sections.
    """
    if dates_per_chunk < 1:
        raise ValueError("dates_per_chunk must be positive")
    spec = _resolve_source(source)
    lazy: pl.LazyFrame
    if isinstance(source_file, pl.LazyFrame):
        lazy = source_file
    else:
        lazy = scan_chain_file(source_file)
    resolved = {**spec.resolve(lazy.collect_schema().names()), **dict(overrides or {})}
    if "quote_date" not in resolved:
        raise ValueError(f"source {spec.name!r} has no quote-date column in this file")
    column = pl.col(resolved["quote_date"])
    sessions = lazy.select(column.unique()).collect().to_series().sort().to_list()
    for start in range(0, len(sessions), dates_per_chunk):
        window = sessions[start:start + dates_per_chunk]
        chunk = lazy.filter(column.is_in(window)).collect()
        if chunk.height:
            yield normalize_option_chain(chunk, spec, underlying=underlying,
                                         time_policy=time_policy, overrides=overrides)


def with_underlying_prices(chain: pl.DataFrame, prices: pl.DataFrame, *,
                           price_column: str = "close") -> pl.DataFrame:
    """Fill `underlying_price` from a bar table — DoltHub ships chains without one.

    `prices` needs `date`/`symbol`/`<price_column>`; the join is on the calendar
    date of `quote_date` in UTC and the ticker, so the bar must be the same
    session's close as the chain snapshot. Existing non-null prices are kept.
    """
    missing = {"date", "symbol", price_column} - set(prices.columns)
    if missing:
        raise ValueError(f"price table missing columns: {sorted(missing)}")
    right = prices.select(
        pl.col("date").cast(pl.Datetime("us"), strict=False).dt.date().alias("_date"),
        pl.col("symbol").cast(pl.String).str.to_uppercase().alias("underlying"),
        pl.col(price_column).cast(pl.Float64).alias("_price"),
    ).unique(subset=["_date", "underlying"], keep="first")
    joined = (chain.with_columns(pl.col("quote_date").dt.date().alias("_date"))
              .join(right, on=["_date", "underlying"], how="left"))
    return (joined.with_columns(pl.col("underlying_price").fill_null(pl.col("_price")))
            .drop("_date", "_price").select(list(CHAIN_COLUMNS)))


@dataclass(frozen=True, slots=True)
class ChainQualityPolicy:
    """Thresholds for the checks that are judgement calls rather than errors."""

    max_relative_spread: float = 1.0
    max_implied_volatility: float = 5.0
    min_implied_volatility: float = 0.01
    parity_tolerance: float = 0.05
    parity_rate: float = 0.0
    parity_dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if not all(isfinite(v) and v >= 0 for v in
                   (self.max_relative_spread, self.max_implied_volatility,
                    self.min_implied_volatility, self.parity_tolerance)):
            raise ValueError("quality thresholds must be finite and nonnegative")
        if self.min_implied_volatility > self.max_implied_volatility:
            raise ValueError("implied volatility bounds are inverted")


def _parity_violations(chain: pl.DataFrame, policy: ChainQualityPolicy) -> int:
    """Rows where put-call parity fails by more than a fraction of spot.

    Uses continuous rate/dividend yield from the policy; both default to zero,
    which makes this a coarse screen for vendor mistakes (mismatched legs, a
    stale side, a wrong underlying price), not a term-structure calibration.
    """
    priced = chain.filter(pl.col("underlying_price").is_not_null() & pl.col("mid").is_not_null())
    if priced.is_empty():
        return 0
    wide = (priced.pivot("right", index=["quote_date", "underlying", "expiry",
                                               "strike", "dte", "underlying_price"],
                         values="mid", aggregate_function="first")
            .filter(pl.col("CALL").is_not_null() & pl.col("PUT").is_not_null())
            if {"CALL", "PUT"} <= set(priced["right"].unique()) else None)
    if wide is None or wide.is_empty():
        return 0
    years = pl.col("dte").cast(pl.Float64) / 365
    residual = ((pl.col("CALL") - pl.col("PUT"))
                - (pl.col("underlying_price") * (-policy.parity_dividend_yield * years).exp()
                   - pl.col("strike") * (-policy.parity_rate * years).exp()))
    return int(wide.select(
        (residual.abs() > policy.parity_tolerance * pl.col("underlying_price")).sum()).item())


def validate_option_chain(chain: pl.DataFrame,
                          policy: ChainQualityPolicy | None = None) -> DataQualityReport:
    """Structural errors and quote-quality warnings for a normalised chain.

    Errors are rows the engine cannot represent at all: crossed or nonpositive
    quotes, negative DTE, duplicate keys, and the values `OptionQuote` rejects
    outright — a negative implied volatility or open interest, a delta outside
    [-1, 1]. Those last three are errors precisely because a chain carrying them
    cannot be turned into frames; catching them here names the count instead of
    failing one row at a time inside the adapter.

    Warnings are the defects public option datasets are known to carry — zero
    bids, very wide spreads, IV outliers within a plausible range, wrong-signed
    deltas, parity breaks on volatile days — which are legitimate data in some
    studies and noise in others, so filtering them is `clean_option_chain`'s job.
    """
    policy = policy or ChainQualityPolicy()
    errors: list[str] = []
    warnings: list[str] = []
    missing = set(CHAIN_COLUMNS) - set(chain.columns)
    if missing:
        return DataQualityReport((f"missing chain columns: {sorted(missing)}",), ())
    if chain.is_empty():
        return DataQualityReport((), ("chain is empty",))

    for column in (*CHAIN_KEY, "bid", "ask"):
        nulls = int(chain[column].null_count())
        if nulls:
            errors.append(f"{column} has {nulls} null values")
    duplicates = int(chain.select(pl.struct(list(CHAIN_KEY)).is_duplicated().sum()).item())
    if duplicates:
        errors.append(f"duplicate contract rows: {duplicates}")
    bad_type = int(chain.select((~pl.col("right").is_in(["CALL", "PUT"])).sum()).item())
    if bad_type:
        errors.append(f"right outside CALL/PUT on {bad_type} rows")
    counts = chain.select(
        (pl.col("strike") <= 0).sum().alias("strike"),
        (pl.col("bid") < 0).sum().alias("bid"),
        (pl.col("ask") <= 0).sum().alias("ask"),
        (pl.col("ask") < pl.col("bid")).sum().alias("crossed"),
        (pl.col("dte") < 0).sum().alias("dte"),
        (pl.col("bid") == 0).sum().alias("zero_bid"),
        ((pl.col("ask") - pl.col("bid")) > policy.max_relative_spread * pl.col("mid"))
        .sum().alias("wide"),
        (pl.col("underlying_price").is_null() | (pl.col("underlying_price") <= 0))
        .sum().alias("spot"),
        (pl.col("implied_volatility").is_not_null()
         & ((pl.col("implied_volatility") < policy.min_implied_volatility)
            | (pl.col("implied_volatility") > policy.max_implied_volatility))).sum().alias("iv"),
        (pl.col("delta").is_not_null()
         & (((pl.col("right") == "CALL") & (pl.col("delta") < 0))
            | ((pl.col("right") == "PUT") & (pl.col("delta") > 0)))).sum().alias("delta"),
        # Rejected by `OptionQuote` itself, so they are errors here rather than
        # warnings: a chain carrying them cannot become frames at all, and the
        # Kaggle SPY dataset carries 217,106 negative implied volatilities.
        (pl.col("implied_volatility") < 0).sum().alias("negative_iv"),
        (pl.col("open_interest") < 0).sum().alias("negative_oi"),
        (pl.col("delta").abs() > 1).sum().alias("impossible_delta"),
    ).row(0, named=True)
    for name, message in (("strike", "nonpositive strikes"), ("bid", "negative bids"),
                          ("ask", "nonpositive asks"), ("crossed", "crossed bid/ask"),
                          ("dte", "expirations before the quote date"),
                          ("negative_iv", "negative implied volatilities"),
                          ("negative_oi", "negative open interest"),
                          ("impossible_delta", "deltas outside [-1, 1]")):
        if counts[name]:
            errors.append(f"{message}: {counts[name]}")
    for column in ("open_interest", "volume"):
        if int(chain[column].null_count()) == chain.height:
            warnings.append(f"{column} is not published by this source (all null)")
    for name, message in (("zero_bid", "zero-bid quotes"),
                          ("wide", f"spreads wider than {policy.max_relative_spread:g}x mid"),
                          ("spot", "rows without a positive underlying price"),
                          ("iv", "implied volatilities outside "
                                 f"[{policy.min_implied_volatility:g}, "
                                 f"{policy.max_implied_volatility:g}]"),
                          ("delta", "deltas out of range or wrong-signed")):
        if counts[name]:
            warnings.append(f"{message}: {counts[name]}")
    if not counts["crossed"] and not counts["spot"]:
        violations = _parity_violations(chain, policy)
        if violations:
            warnings.append("put-call parity residual above "
                            f"{policy.parity_tolerance:g}x spot on {violations} strikes")
    return DataQualityReport(tuple(errors), tuple(warnings))


@dataclass(frozen=True, slots=True)
class ChainFilter:
    """Which rows `clean_option_chain` drops. Every default is a deletion."""

    drop_crossed: bool = True
    #: Removes what the engine cannot represent: nonpositive quotes or strikes,
    #: and negative implied volatility / open interest or an impossible delta.
    drop_nonpositive: bool = True
    drop_zero_bid: bool = True
    drop_missing_underlying: bool = True
    max_relative_spread: float | None = 1.0
    min_open_interest: float = 0.0
    min_volume: float = 0.0
    implied_volatility_bounds: tuple[float, float] | None = (0.01, 5.0)
    dte_range: tuple[int, int | None] = (0, None)
    deduplicate: bool = True

    def __post_init__(self) -> None:
        low, high = self.dte_range
        if high is not None and low > high:
            raise ValueError("dte_range is inverted")
        if self.max_relative_spread is not None and not (
                isfinite(self.max_relative_spread) and self.max_relative_spread >= 0):
            raise ValueError("max_relative_spread must be finite and nonnegative")
        if self.implied_volatility_bounds is not None:
            iv_low, iv_high = self.implied_volatility_bounds
            if not (isfinite(iv_low) and isfinite(iv_high)) or iv_low > iv_high:
                raise ValueError("implied_volatility_bounds are inverted or non-finite")


def clean_option_chain(
    chain: pl.DataFrame, chain_filter: ChainFilter | None = None,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Apply `chain_filter` and report how many rows each rule removed.

    Rules are applied in order and the count is rows removed *by that rule*, so
    the counts sum to the total removed and a row defective in two ways is
    attributed to the first rule that catches it.

    A threshold on a column the vendor never populated is refused rather than
    applied. `open_interest` and `volume` are optional in `CHAIN_SCHEMA`, and a
    null fails every comparison, so `min_open_interest=10` against an all-null
    column deletes the entire chain and reports it as an open-interest filter
    working normally. The Kaggle/OptionsDX SPY dataset publishes no open
    interest at all -- 1,927,717 of 1,927,717 nulls in 2023 -- so this is the
    default dataset's behaviour, not a corner case.
    """
    rules = chain_filter or ChainFilter()
    for threshold, column in ((rules.min_open_interest, "open_interest"),
                              (rules.min_volume, "volume")):
        if threshold and chain.height and chain[column].null_count() == chain.height:
            raise ValueError(
                f"{column} is null on all {chain.height} rows, so a threshold on it would "
                f"drop the whole chain: this vendor does not publish {column}")
    removed: dict[str, int] = {}
    frame = chain
    def drop(name: str, keep: pl.Expr) -> None:
        nonlocal frame
        before = frame.height
        frame = frame.filter(keep.fill_null(False))
        if before - frame.height:
            removed[name] = before - frame.height

    drop("null_key", pl.all_horizontal([pl.col(column).is_not_null()
                                        for column in (*CHAIN_KEY, "bid", "ask")]))
    if rules.drop_nonpositive:
        drop("nonpositive_quote",
             (pl.col("ask") > 0) & (pl.col("bid") >= 0) & (pl.col("strike") > 0))
        # Same rule, different column: values `OptionQuote` will not accept.
        drop("unrepresentable_feature",
             (pl.col("implied_volatility").is_null() | (pl.col("implied_volatility") >= 0))
             & (pl.col("open_interest").is_null() | (pl.col("open_interest") >= 0))
             & (pl.col("delta").is_null() | (pl.col("delta").abs() <= 1)))
    if rules.drop_crossed:
        drop("crossed", pl.col("ask") >= pl.col("bid"))
    if rules.drop_zero_bid:
        drop("zero_bid", pl.col("bid") > 0)
    if rules.drop_missing_underlying:
        drop("missing_underlying", pl.col("underlying_price") > 0)
    if rules.max_relative_spread is not None:
        drop("wide_spread",
             (pl.col("ask") - pl.col("bid")) <= rules.max_relative_spread * pl.col("mid"))
    if rules.min_open_interest:
        drop("open_interest", pl.col("open_interest") >= rules.min_open_interest)
    if rules.min_volume:
        drop("volume", pl.col("volume") >= rules.min_volume)
    if rules.implied_volatility_bounds is not None:
        low, high = rules.implied_volatility_bounds
        drop("implied_volatility", pl.col("implied_volatility").is_null()
             | pl.col("implied_volatility").is_between(low, high))
    low_dte, high_dte = rules.dte_range
    drop("dte", (pl.col("dte") >= low_dte)
         & (pl.lit(True) if high_dte is None else pl.col("dte") <= high_dte))
    if rules.deduplicate:
        before = frame.height
        frame = frame.unique(subset=list(CHAIN_KEY), keep="first", maintain_order=True)
        if before - frame.height:
            removed["duplicate"] = before - frame.height
    return frame.sort(list(CHAIN_KEY)), removed


class OptionChainStore:
    """Hive-partitioned Parquet store: `<root>/underlying=SPY/year=2019/part-<id>.parquet`.

    Same layout as `HistoricalDataStore`'s tick store, for the same reason: a
    scan filtered by ticker and date prunes whole directories instead of reading
    fourteen years of SPY to answer a question about one month. `write` appends a
    new part file per partition touched and never rewrites one already on disk,
    so ingesting a vendor file costs O(rows in that file) — deduplication is
    `clean_option_chain`'s job, before the write, not the store's.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, chain: pl.DataFrame) -> tuple[Path, ...]:
        if missing := set(CHAIN_COLUMNS) - set(chain.columns):
            raise ValueError(f"chain missing columns: {sorted(missing)}")
        report = validate_option_chain(chain)
        report.raise_if_failed()
        if chain.is_empty():
            return ()
        written: list[Path] = []
        keyed = chain.select(list(CHAIN_COLUMNS)).with_columns(
            pl.col("quote_date").dt.year().alias("_year"))
        for (ticker, year), group in keyed.group_by("underlying", "_year", maintain_order=True):
            directory = self.root / f"underlying={ticker}" / f"year={year}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"part-{uuid4().hex[:12]}.parquet"
            group.drop("_year", "underlying").write_parquet(path)
            written.append(path)
        return tuple(written)

    def ingest(self, chain: pl.DataFrame) -> int:
        """Write the sessions not already stored for their underlying; a re-run adds nothing."""
        if chain.is_empty():
            return 0
        first, last = chain.select(pl.col("quote_date").min().alias("a"), pl.col("quote_date").max().alias("b")).row(0)
        stored = (self.scan(underlyings=chain["underlying"].unique().to_list(), start=first, end=last)
                  .select("underlying", "quote_date").unique().collect())
        fresh = chain.join(stored, on=["underlying", "quote_date"], how="anti")
        self.write(fresh)
        return fresh.height

    def scan(self, *, underlyings: Sequence[str] | None = None,
             start: datetime | None = None, end: datetime | None = None) -> pl.LazyFrame:
        """Lazy scan with ticker/date pushdown; inclusive `start`/`end`."""
        if not any(self.root.rglob("*.parquet")):
            return pl.LazyFrame(schema=dict(CHAIN_SCHEMA))
        scanned = pl.scan_parquet(self.root, hive_partitioning=True)
        if underlyings is not None:
            tickers = [ticker.upper() for ticker in underlyings]
            scanned = scanned.filter(pl.col("underlying").is_in(tickers))
        if start is not None:
            scanned = scanned.filter((pl.col("year") >= utc(start).year)
                                     & (pl.col("quote_date") >= utc(start)))
        if end is not None:
            scanned = scanned.filter((pl.col("year") <= utc(end).year)
                                     & (pl.col("quote_date") <= utc(end)))
        return scanned.select(list(CHAIN_COLUMNS)).sort(list(CHAIN_KEY))


def option_contracts_from_chain(
    chain: pl.DataFrame, *, multiplier: float = 100,
    exercise_style: Literal["european", "american"] = "american",
    settlement: Literal["cash", "physical"] = "physical",
) -> tuple[OptionContract, ...]:
    """One `OptionContract` per distinct contract in the chain.

    The defaults describe a US listed equity/ETF option (American, physically
    settled, 100 shares). Index options (SPX, and therefore any XSP/VIX study)
    are European and cash settled, and adjusted contracts have a non-standard
    multiplier: neither is inferable from a chain file, so pass them.
    """
    unique = (chain.select("option_symbol", "underlying", "right", "strike", "expiry")
              .unique(maintain_order=True).sort("option_symbol"))
    contracts = [OptionContract(row["option_symbol"], row["underlying"], row["right"],
                                float(row["strike"]), row["expiry"], multiplier,
                                exercise_style, settlement)
                 for row in unique.iter_rows(named=True)]
    strikes = {c.symbol for c in contracts}
    if len(strikes) != len(contracts):
        raise ValueError("chain contains contracts that collide on their OCC symbol")
    return tuple(contracts)


def option_chain_to_frames(
    chain: pl.DataFrame, *, time_policy: ChainTimePolicy | None = None,
    underlying_half_spread: float = 0.0, include_underlying: bool = True,
) -> tuple[OptionMarketFrame, ...]:
    """Chain rows -> one `OptionMarketFrame` per observation instant.

    Each frame carries every option quote observed at that instant plus, unless
    `include_underlying=False`, one quote per underlying built from
    `underlying_price` ± `underlying_half_spread`. The engine needs the underlying
    mark to value and margin an option position, so leaving it out only works for
    a run that holds nothing.

    Sizes are left unset: an EOD chain publishes no displayed size, and daily
    volume is not one. Run with `OptionBacktestConfig(require_sizes=False)` and
    read the result as an explicit "assume enough liquidity" scenario.

    No lifecycle events are attached — an expiry needs an official settlement
    fixing, which is not in these files. See `expiry_events_from_underlying`.
    """
    policy = time_policy or ChainTimePolicy()
    report = validate_option_chain(chain)
    report.raise_if_failed()
    if chain.is_empty():
        return ()
    if not isfinite(underlying_half_spread) or underlying_half_spread < 0:
        raise ValueError("underlying_half_spread must be finite and nonnegative")
    lag = policy.publication_lag
    quotes = chain.select(
        (pl.col("quote_date") + lag).alias("timestamp"),
        pl.col("option_symbol").alias("symbol"),
        pl.col("quote_date").alias("event_time"),
        (pl.col("quote_date") + lag).alias("available_at"),
        "bid", "ask", "implied_volatility", "delta", "open_interest",
    )
    if include_underlying:
        spots = (chain.filter(pl.col("underlying_price").is_not_null())
                 .select("quote_date", "underlying", "underlying_price")
                 .unique(maintain_order=True))
        conflicting = spots.select(pl.struct("quote_date", "underlying").is_duplicated().sum())
        if int(conflicting.item()):
            raise ValueError("underlying_price disagrees between rows of the same snapshot")
        if spots.height != chain.select("quote_date", "underlying").unique().height:
            raise ValueError("every snapshot needs an underlying price; "
                             "use with_underlying_prices or include_underlying=False")
        underlying_rows = spots.select(
            (pl.col("quote_date") + lag).alias("timestamp"),
            pl.col("underlying").alias("symbol"),
            pl.col("quote_date").alias("event_time"),
            (pl.col("quote_date") + lag).alias("available_at"),
            (pl.col("underlying_price") - underlying_half_spread).alias("bid"),
            (pl.col("underlying_price") + underlying_half_spread).alias("ask"),
            pl.lit(None, pl.Float64).alias("implied_volatility"),
            pl.lit(None, pl.Float64).alias("delta"),
            pl.lit(None, pl.Float64).alias("open_interest"),
        )
        quotes = pl.concat([quotes, underlying_rows], how="vertical")
    return option_frames_from_polars(quotes.sort("timestamp", "symbol"))


def expiry_events_from_prices(
    prices: Mapping[tuple[str, date], float], contracts: Iterable[OptionContract], *,
    time_policy: ChainTimePolicy | None = None,
) -> tuple[OptionLifecycleEvent, ...]:
    """One expiry event per contract, priced from an explicit `(ticker, date) -> close`.

    Separated from `expiry_events_from_underlying` so the fixing can come from
    somewhere better than the chain file that is being backtested. A chain's own
    `underlying_price` is the same vendor marking both sides of the trade; an
    independently sourced close at least disagrees when one of them is wrong.

    Neither is the official settlement value -- see
    `expiry_events_from_underlying` for what that costs.

    Contracts whose expiry date is absent get no event, so the backtest still
    fails loudly on a position it cannot settle rather than silently expiring it
    at an invented price.
    """
    policy = time_policy or ChainTimePolicy()
    events = []
    for contract in sorted(contracts, key=lambda c: (c.expiry, c.symbol)):
        price = prices.get((contract.underlying, contract.expiry.date()))
        if price is None:
            continue
        if not isfinite(price) or price <= 0:
            raise ValueError(f"settlement fixing for {contract.symbol} is not a price: {price}")
        observed = contract.expiry
        events.append(OptionLifecycleEvent(contract.symbol, observed,
                                           observed + policy.publication_lag, float(price)))
    return tuple(events)


def session_closes(prices: pl.DataFrame, *, ticker: str = "ticker",
                   timestamp: str = "timestamp", close: str = "close",
                   zone: str = "America/New_York", at: time = time(16, 0)) -> dict[tuple[str, date], float]:
    """`(ticker, session) -> close` from a bar table, for `expiry_events_from_prices`.

    Takes the last bar at or before `at` in `zone`, so a minute tape that runs
    into the evening still yields the 16:00 close rather than the last
    after-hours print. Bars after `at` are ignored, not averaged in.
    """
    local = pl.col(timestamp).dt.convert_time_zone(zone)
    rows = (prices.lazy()
            .with_columns(local.alias("_local"))
            .filter(pl.col("_local").dt.time() <= at)
            .sort("_local")
            .group_by(pl.col(ticker).alias("_ticker"), pl.col("_local").dt.date().alias("_session"))
            .agg(pl.col(close).last().alias("_close"))
            .collect())
    return {(name, session): float(value) for name, session, value in rows.iter_rows()}


def expiry_events_from_underlying(
    chain: pl.DataFrame, contracts: Iterable[OptionContract], *,
    time_policy: ChainTimePolicy | None = None,
) -> tuple[OptionLifecycleEvent, ...]:
    """Expiry events priced off the chain's own underlying close. An approximation.

    The engine requires an explicit fixing at expiry, and a public EOD chain has
    no settlement price in it. This builds one event per expiring contract from
    the underlying price recorded on the expiry date, which is *not* the official
    settlement value: PM-settled equity options settle off the closing price but
    not necessarily this vendor's close, and AM-settled index options settle off
    the opening prints, which this number has nothing to do with. Use it for
    research runs and say so; do not report it as a settlement-accurate result.

    `expiry_events_from_prices` takes the fixing from anywhere else, which is
    the better input when an independent close is available.

    Contracts whose expiry date has no row in the chain get no event, so the
    backtest still fails loudly on a position it cannot settle rather than
    silently expiring it at an invented price.
    """
    # One row per (ticker, session) before crossing into Python: a year of SPY
    # is ~2 million chain rows and ~250 closes, and only the closes are needed.
    sessions = (
        chain.lazy()
        .filter(pl.col("underlying_price").is_not_null())
        .group_by("underlying", pl.col("quote_date").dt.date().alias("session"))
        .agg(pl.col("underlying_price").first())
        .collect()
    )
    closes = {(ticker, session): price
              for ticker, session, price in sessions.iter_rows()}
    return expiry_events_from_prices(closes, contracts, time_policy=time_policy)


def attach_lifecycle(frames: Sequence[OptionMarketFrame],
                     events: Iterable[OptionLifecycleEvent]) -> tuple[OptionMarketFrame, ...]:
    """Merge lifecycle events into the frame that makes them available.

    An event lands on the earliest frame at or after its `available_at`; an event
    with no such frame is an error, because dropping it would silently leave a
    position unsettled.
    """
    ordered = sorted(frames, key=lambda f: f.timestamp)
    timestamps = [frame.timestamp for frame in ordered]
    buckets: dict[datetime, list[OptionLifecycleEvent]] = {}
    for event in events:
        available = event.available_at
        index = bisect_left(timestamps, available)
        if index == len(timestamps):
            raise ValueError(f"no frame at or after {available.isoformat()} for {event.symbol}")
        buckets.setdefault(timestamps[index], []).append(event)
    return tuple(
        OptionMarketFrame(frame.timestamp, frame.quotes,
                          (*frame.lifecycle, *buckets.get(frame.timestamp, ())),
                          frame.dividends)
        for frame in ordered)
