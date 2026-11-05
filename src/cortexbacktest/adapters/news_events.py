"Read a digest-bound news replay file without importing its service or credentials."

from __future__ import annotations

from datetime import datetime, date, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from .observation_file import instrument_symbol
from ..research.event_study import _aware
from .. import strict_json


def load_news_replay_events(path: str | Path, *, sha256: str) -> tuple[pl.DataFrame, dict]:
    with Path(path).open("rb") as stream:
        raw = stream.read(10_000_001)
    if len(raw) > 10_000_000 or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("news replay digest/size mismatch")

    doc = strict_json.loads(raw)
    if doc.get("executionAuthorized") is not False or doc.get("gaps") != []:
        raise ValueError("complete read-only news replay required")
    version = doc.get("contractVersion")
    if version not in {None, "received-news-replay-v1"}:
        raise ValueError("unknown news replay contract")
    received = version == "received-news-replay-v1"
    if received and (doc.get("qualificationIssued") is not False or len(doc["events"]) > 500):
        raise ValueError("bounded non-qualifying received news required")
    scope = doc["scope"]
    instrument = scope["instrument_id"]
    if type(instrument) is not int or instrument <= 0:
        raise ValueError("explicit replay instrument identity required")
    cutoff = _aware(datetime.fromisoformat(scope["as_of"]))
    if received:
        start, end = date.fromisoformat(scope["start"]), date.fromisoformat(scope["end"])
        if (
            start > end
            or (end - start).days > 90
            or cutoff > datetime.now(timezone.utc)
            or not isinstance(scope["symbol"], str)
            or not scope["symbol"]
            or scope["symbol"] != scope["symbol"].strip().upper()
        ):
            raise ValueError("invalid received news scope")
    symbol = instrument_symbol(str(instrument))
    rows = []
    for event in doc["events"]:
        if event["kind"] != "news":
            if received:
                raise ValueError("received news export cannot include fills")
            continue
        if received:
            try:
                _check_evidence(event, scope)
            except (KeyError, TypeError, AttributeError, OverflowError) as exc:
                raise ValueError("incomplete received news evidence") from exc
        fields = ["receivedAt", "entityReceivedAt", "identityReceivedAt", "contentReceivedAt"]
        instants = [_aware(datetime.fromisoformat(event[key])) for key in fields]
        if event.get("bodyReceivedAt"):
            instants.append(_aware(datetime.fromisoformat(event["bodyReceivedAt"])))
        publication = event.get("publishedAt")
        if publication is not None:
            published = _aware(datetime.fromisoformat(publication))
            instants.append(published)
        else:
            published = None
        available = _aware(datetime.fromisoformat(event["availableAt"]))
        if available != max(instants) or available > cutoff:
            raise ValueError("news replay availability clock mismatch")
        rows.append(
            {
                "event_id": f"{instrument}:{event['articleId']}",
                "symbol": symbol,
                "event_at": published,
                "available_at": available,
            }
        )
    frame = pl.DataFrame(
        rows,
        schema={
            "event_id": pl.String,
            "symbol": pl.String,
            "event_at": pl.Datetime("us", "UTC"),
            "available_at": pl.Datetime("us", "UTC"),
        },
    )
    if frame["event_id"].n_unique() != frame.height:
        raise ValueError("duplicate news event identity")
    return frame, {
        "inputSha256": sha256,
        "asOf": cutoff.isoformat(),
        "instrument": symbol,
        "upstreamLimitations": doc.get("limitations", []),
        "identityPolicy": "upstream_explicit_scope_not_ticker_guessing",
        "receivedRevisionsVerified": received,
    }


def _stamp(value):
    return _aware(datetime.fromisoformat(value))


def _check_evidence(event, scope):
    "Recompute a frozen DB receipt projection; this is not provider signature authentication."
    evidence = event["evidence"]
    raw = json.dumps(
        evidence, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != event["evidenceSha256"]:
        raise ValueError("news evidence digest mismatch")
    cutoff = _stamp(scope["as_of"])
    content, entity, body, identity = (
        evidence[k] for k in ["content", "entity", "body", "identity"]
    )
    row_id = content["articleRowId"]
    if type(row_id) is not int or row_id <= 0:
        raise ValueError("news article row identity required")
    for receipt, clock, deleted in [
        (content, "contentReceivedAt", False),
        (entity, "entityReceivedAt", False),
        (body, "bodyReceivedAt", True),
    ]:
        if receipt is None:
            if event.get(clock) is not None:
                raise ValueError("news receipt clock without evidence")
            continue
        if (
            type(receipt["receiptId"]) is not int
            or receipt["receiptId"] <= 0
            or receipt["articleRowId"] != row_id
            or receipt["operation"]
            not in (
                {"seed", "insert", "update", "delete"} if deleted else {"seed", "insert", "update"}
            )
            or _stamp(receipt["recordedAt"]) != _stamp(event[clock])
            or _stamp(receipt["recordedAt"]) > cutoff
        ):
            raise ValueError("news revision identity/clock mismatch")
    source = content["document"]
    if (
        str(source["article_id"]) != str(event["articleId"])
        or any(source[k] != event[k] for k in ["title", "url", "publisher"])
        or _stamp(source["stored_at"]) != _stamp(event["receivedAt"])
    ):
        raise ValueError("news content projection mismatch")
    published = source["published_at"]
    if (published is None) != (event["publishedAt"] is None) or (
        published is not None and _stamp(published) != _stamp(event["publishedAt"])
    ):
        raise ValueError("news publication projection mismatch")
    window_at = _stamp(published or source["stored_at"])
    start = datetime.combine(date.fromisoformat(scope["start"]), time.min, timezone.utc)
    end = datetime.combine(
        date.fromisoformat(scope["end"]) + timedelta(days=1), time.min, timezone.utc
    )
    if not start <= window_at < end or not 0 <= (end - start).days <= 91:
        raise ValueError("news event outside declared scope")
    association = entity["document"]
    if (
        association["eventId"] != "url:" + hashlib.sha256(event["url"].encode()).hexdigest()
        or scope["symbol"] not in association["symbols"]
    ):
        raise ValueError("news URL/entity binding mismatch")
    zone = timezone.utc if scope.get("market_mic") is None else ZoneInfo("America/New_York")
    effective = window_at.astimezone(zone).date().isoformat()
    if (
        identity["contractVersion"] != "instrument-resolution-v1"
        or identity["status"] != "resolved"
        or identity["windowDefault"] is not False
        or identity["symbol"] != scope["symbol"]
        or identity["symbolSource"] != "fmp"
        or identity["marketMic"] != scope.get("market_mic")
        or identity["instrumentId"] != str(scope["instrument_id"])
        or identity["candidateIds"] != [str(scope["instrument_id"])]
        or identity["effectiveOn"] != effective
        or identity["validFrom"] > effective
        or _stamp(identity["asOf"]) != cutoff
        or _stamp(identity["recordedAt"]) != _stamp(event["identityReceivedAt"])
    ):
        raise ValueError("news instrument resolution mismatch")
    text = None if body is None or body["operation"] == "delete" else body["document"].get("body")
    expected = hashlib.sha256(text.encode()).hexdigest() if text is not None else None
    if event.get("bodySha256") != expected:
        raise ValueError("news body digest mismatch")
