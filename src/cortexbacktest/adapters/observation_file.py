"""Strict observation receipt files, independent of the producer's database and source tree."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, time, timedelta, timezone
from importlib.resources import files
from multiprocessing import get_context
from pathlib import Path
import math
import os

from jsonschema import Draft202012Validator, FormatChecker, ValidationError
import orjson
import polars as pl

from ..contracts import Bar


_SCHEMA = orjson.loads(files(__package__).joinpath("observation-file-v1.schema.json").read_bytes())
_VALIDATOR = Draft202012Validator(_SCHEMA, format_checker=FormatChecker())
# Above this many records and sessions the schema check runs in chunks across processes.
_PARALLEL_ITEMS = 200_000


def _check_schema(document: dict) -> str | None:
    error = next(_VALIDATOR.iter_errors(document), None)
    return None if error is None else error.message


def _validate_schema(document: dict) -> None:
    """The whole document against the schema, every record and session in its own envelope.

    A large file is split into pieces that each keep the full envelope and a slice of
    `records` and of `scope.sessions`, so every item is checked in exactly the context it has
    in the file. The split is sound because neither array has a cross-item rule beyond a
    nonempty `sessions` (uniqueness of sessions and records is checked below, in Python).
    """
    records = document.get("records")
    scope = document.get("scope")
    sessions = scope.get("sessions") if isinstance(scope, dict) else None
    if (not isinstance(scope, dict) or not isinstance(records, list) or not isinstance(sessions, list) or not sessions
            or len(records) + len(sessions) < _PARALLEL_ITEMS):
        _VALIDATOR.validate(document)
        return
    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    pieces = workers * 4
    record_size, session_size = -(-len(records) // pieces), -(-len(sessions) // pieces)
    documents = [{**document, "records": records[i * record_size:(i + 1) * record_size],
                  "scope": {**scope, "sessions": sessions[i * session_size:(i + 1) * session_size] or sessions[:1]}}
                 for i in range(pieces)]
    # spawn, never fork: forking after Polars started its threads can deadlock.
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        for message in pool.map(_check_schema, documents, chunksize=1):
            if message is not None:
                raise ValidationError(message)


def _finite(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Observation JSON numbers must be finite")


def _instant(value: str) -> datetime:
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Receipt timestamps must include a timezone")
    return stamp


def validate_observation_file(document: dict, *, as_of: datetime) -> dict:
    """Validate the entire declared observation scope; never silently drop bad rows.

    An empty set can be valid. Its scope and count are still mandatory. This is
    an observation snapshot, not a claim that prices or a tradable dataset exist.
    Macro tombstones remain records; callers must select vintage before values.
    """
    _finite(document)
    _validate_schema(document)
    if as_of.tzinfo is None or as_of.utcoffset() is None or _instant(document["asOf"]) != as_of:
        raise ValueError("Snapshot asOf must equal the requested aware cutoff")
    rows, scope = document["records"], document["scope"]
    if document["recordCount"] != len(rows) or scope["start"] > scope["end"]:
        raise ValueError("Incomplete record count or reversed scope")
    sessions = {}
    market = document["recordVersion"] in {"market-bars-v2", "market-bars-v4"}
    if market:
        for item in scope["sessions"]:
            key = (item["instrumentId"], item["sessionDate"])
            if key in sessions:
                raise ValueError("Duplicate requested market session")
            sessions[key] = _instant(item["sessionClose"])
    seen = set()
    covered = set()
    for row in rows:
        if document["recordVersion"] == "factor-observations-v1":
            stamps = {
                key: _instant(row[key])
                for key in (
                    "as_of",
                    "event_time",
                    "known_at",
                    "score_known_at",
                    "recorded_at",
                    "entity_known_at",
                    "identity_known_at",
                )
            }
            # A value computed from public filings is knowable when its last filing was public;
            # the day the store recorded it is not knowledge (the producer's factor store agrees).
            filing_dated = (row.get("metadata") or {}).get("knowledge") == "filing-date"
            receipts = (
                ("score_known_at", "identity_known_at")
                if filing_dated
                else ("score_known_at", "recorded_at", "entity_known_at", "identity_known_at")
            )
            available = max(stamps[key] for key in receipts)
            if stamps["event_time"] > stamps["score_known_at"] or stamps["known_at"] != available:
                raise ValueError(
                    "Factor availability must include all receipts and follow the event"
                )
            if (
                row["symbol"] not in scope["symbols"]
                or row["symbol_source"] != scope["symbolSource"]
                or row["factor_name"] != scope["factorName"]
            ):
                raise ValueError("Factor lies outside the declared identity scope")
            event_date = stamps["event_time"].astimezone(timezone.utc).date().isoformat()
            cutoff = stamps["as_of"]
            key = (row["instrument_id"], row["event_id"], stamps["event_time"], row["source"])
        elif market:
            key = (row["instrumentId"], row["sessionDate"])
            covered.add(key)
            session_close = _instant(row["sessionClose"])
            available = max(
                session_close, _instant(row["recordedAt"]), _instant(row["sourceKnownAt"])
            )
            if (
                sessions.get(key) != session_close
                or row["source"] != scope["source"]
                or row["adjustment"] != scope["adjustment"]
            ):
                raise ValueError(
                    "Market record lies outside the declared session/source/basis scope"
                )
            if _instant(row["availableAt"]) != available:
                raise ValueError("Market availability must include session close and both receipts")
            if row.get("invalidReason") is None and (
                not row["low"]
                <= min(row["open"], row["close"])
                <= max(row["open"], row["close"])
                <= row["high"]
                or row["adjustment"] == "unadjusted"
                and row["adjClose"] != row["close"]
            ):
                raise ValueError("Market OHLC or unadjusted research close is inconsistent")
            event_date, cutoff = row["sessionDate"], _instant(row["asOf"])
            if document["recordVersion"] != "market-bars-v2":
                key = (*key, _instant(row["recordedAt"]))
        else:
            vintage = datetime.fromisoformat(row["realtimeStart"]).date()
            source_bound = datetime.combine(vintage, time.min, timezone.utc) + timedelta(hours=36)
            available = max(_instant(row["recordedAt"]), source_bound)
            if (
                vintage.isoformat() == "1776-07-04"
                or row["realtimeEnd"] is not None
                and row["realtimeEnd"] < row["realtimeStart"]
                or _instant(row["availableAt"]) != available
            ):
                raise ValueError("Invalid macro vintage or receipt availability")
            if row["seriesId"] != scope["seriesId"]:
                raise ValueError("Macro record lies outside the declared series")
            event_date, cutoff = row["observationDate"], _instant(row["asOf"])
            key = (row["seriesId"], event_date, row["realtimeStart"])
        if cutoff != as_of or available > as_of or not scope["start"] <= event_date <= scope["end"]:
            raise ValueError("Observation exceeds the snapshot time or date scope")
        if key in seen:
            raise ValueError("Duplicate observation identity in snapshot")
        seen.add(key)
    if market and covered != sessions.keys():
        raise ValueError("Incomplete requested market session coverage")
    return document


def load_observation_file(path: str | Path, *, sha256: str, as_of: datetime) -> dict:
    """Read one complete file bound to a separately supplied SHA-256 and cutoff."""
    # Hash the bytes we actually parse, rather than reopening a mutable file.
    import hashlib

    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError("Observation file digest mismatch")
    return validate_observation_file(orjson.loads(content), as_of=as_of)


def instrument_symbol(instrument_id: str) -> str:
    """The engine symbol for a canonical instrument; bars and factors join on it.

    Upper case because the engine upper-cases every symbol it is given: a
    factor keyed `instrument:1` never matches the bar the engine calls
    `INSTRUMENT:1`.
    """
    return f"INSTRUMENT:{instrument_id}"


def observation_bars(document: dict, *, stamp: str = "availableAt") -> list[Bar]:
    """A validated market-bars-v2 file as engine bars.

    By default each bar is stamped with its `availableAt` -- session close and
    both receipts -- not the session date, so a strategy sees a bar only once the
    platform could have. Research measuring session-to-session returns stamps
    `sessionClose`, the price history of record. Symbols are
    `instrument_symbol(instrumentId)`.
    """
    if document["recordVersion"] != "market-bars-v2" or stamp not in ("availableAt", "sessionClose"):
        raise ValueError("observation_bars needs a market-bars-v2 file stamped at availableAt or sessionClose")
    bars = [
        Bar(instrument_symbol(row["instrumentId"]), _instant(row[stamp]), row["open"], row["high"],
            row["low"], row["close"], row["volume"], adj_close=row["adjClose"])
        for row in document["records"]
    ]
    return sorted(bars, key=lambda bar: (bar.timestamp, bar.symbol))


def observation_factors(document: dict) -> pl.DataFrame:
    """A validated factor-observations-v1 file as rows keyed by when each became known.

    `available_at` is the record's `known_at` (the latest of its score, record,
    entity and identity receipts); `event_time` is kept for reference only and
    must not be used to decide when a value could be traded.
    """
    if document["recordVersion"] != "factor-observations-v1":
        raise ValueError("observation_factors needs a factor-observations-v1 file")
    rows = document["records"]
    return pl.DataFrame({
        "symbol": [instrument_symbol(row["instrument_id"]) for row in rows],
        "factor_name": [row["factor_name"] for row in rows],
        "available_at": [_instant(row["known_at"]).astimezone(timezone.utc) for row in rows],
        "event_time": [_instant(row["event_time"]).astimezone(timezone.utc) for row in rows],
        "value": [float(row["value"]) for row in rows],
    }, schema={"symbol": pl.String, "factor_name": pl.String, "available_at": pl.Datetime("us", "UTC"),
               "event_time": pl.Datetime("us", "UTC"), "value": pl.Float64}).sort("available_at", "symbol")
