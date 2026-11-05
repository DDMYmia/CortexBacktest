"Received source revisions are reproducible and cannot silently change projection."

import hashlib
import json

import pytest

from cortexbacktest.adapters.news_events import load_news_replay_events


CLOCK = "2025-01-05T00:00:00+00:00"


def document():
    source = {
        "article_id": "article-1",
        "title": "Received title",
        "url": "https://example.test/news",
        "publisher": "Example",
        "published_at": "2025-01-02T15:00:00+00:00",
        "stored_at": CLOCK,
    }

    def receipt(number, doc):
        return {
            "receiptId": number,
            "articleRowId": 8,
            "recordedAt": CLOCK,
            "operation": "insert",
            "document": doc,
        }

    identity = {
        "contractVersion": "instrument-resolution-v1",
        "symbol": "AAPL",
        "symbolSource": "fmp",
        "marketMic": "XNAS",
        "effectiveOn": "2025-01-02",
        "asOf": CLOCK,
        "status": "resolved",
        "instrumentId": "6",
        "candidateIds": ["6"],
        "recordedAt": CLOCK,
        "windowDefault": False,
        "validFrom": "2024-01-01",
    }
    evidence = {
        "content": receipt(11, source),
        "entity": receipt(
            12,
            {
                "eventId": "url:" + hashlib.sha256(source["url"].encode()).hexdigest(),
                "symbols": ["AAPL"],
            },
        ),
        "body": receipt(13, {"body": "Original body"}),
        "identity": identity,
    }
    event = {
        "kind": "news",
        "articleId": "article-1",
        "title": source["title"],
        "url": source["url"],
        "publisher": source["publisher"],
        "publishedAt": source["published_at"],
        **{
            k: CLOCK
            for k in [
                "receivedAt",
                "entityReceivedAt",
                "identityReceivedAt",
                "contentReceivedAt",
                "bodyReceivedAt",
                "availableAt",
            ]
        },
        "bodySha256": hashlib.sha256(b"Original body").hexdigest(),
        "evidence": evidence,
    }
    return {
        "contractVersion": "received-news-replay-v1",
        "scope": {
            "instrument_id": 6,
            "symbol": "AAPL",
            "market_mic": "XNAS",
            "start": "2025-01-01",
            "end": "2025-01-04",
            "as_of": CLOCK,
        },
        "gaps": [],
        "executionAuthorized": False,
        "qualificationIssued": False,
        "events": [event],
    }


def load(tmp_path, doc):
    for event in doc["events"]:
        event["evidenceSha256"] = hashlib.sha256(
            json.dumps(
                event["evidence"],
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    raw = json.dumps(doc).encode()
    path = tmp_path / "news.json"
    path.write_bytes(raw)
    return load_news_replay_events(path, sha256=hashlib.sha256(raw).hexdigest())


def test_received_body_and_identity_replay(tmp_path):
    events, meta = load(tmp_path, document())
    assert events["event_id"].to_list() == ["6:article-1"]
    assert meta["receivedRevisionsVerified"] is True


@pytest.mark.parametrize(
    "case",
    [
        "title",
        "publication",
        "content-clock",
        "body",
        "entity",
        "row-id",
        "operation",
        "instrument",
        "window-default",
        "valid-from",
        "cutoff",
        "source",
        "market",
        "effective",
        "scope",
        "future",
        "authorized",
        "duplicate",
        "kind",
        "missing-receipt",
    ],
)
def test_rehashed_projection_changes_rejected(tmp_path, case):
    doc = document()
    event = doc["events"][0]
    proof = event["evidence"]
    if case == "title":
        event["title"] = "Changed"
    elif case == "publication":
        event["publishedAt"] = "2025-01-02T16:00:00Z"
    elif case == "content-clock":
        proof["content"]["recordedAt"] = "2025-01-04T00:00:00Z"
    elif case == "body":
        proof["body"]["document"]["body"] = "Changed body"
    elif case == "entity":
        proof["entity"]["document"]["symbols"] = ["MSFT"]
    elif case == "row-id":
        proof["entity"]["articleRowId"] = 9
    elif case == "operation":
        proof["content"]["operation"] = "delete"
    elif case == "instrument":
        proof["identity"]["instrumentId"] = "7"
    elif case == "window-default":
        proof["identity"]["windowDefault"] = True
    elif case == "valid-from":
        proof["identity"]["validFrom"] = "2025-01-03"
    elif case == "cutoff":
        doc["scope"]["as_of"] = "2025-01-04T00:00:00Z"
    elif case == "source":
        proof["identity"]["symbolSource"] = "other"
    elif case == "market":
        proof["identity"]["marketMic"] = "XNYS"
    elif case == "effective":
        proof["identity"]["effectiveOn"] = "2025-01-03"
    elif case == "scope":
        doc["scope"]["start"] = "2025-01-03"
    elif case == "future":
        doc["scope"]["as_of"] = "2999-01-01T00:00:00Z"
    elif case == "authorized":
        doc["qualificationIssued"] = True
    elif case == "duplicate":
        doc["events"] *= 2
    elif case == "kind":
        event["kind"] = "fill"
    else:
        proof["body"] = None
    with pytest.raises(ValueError):
        load(tmp_path, doc)


@pytest.mark.parametrize("state", ["missing", "withdrawn", "unpublished"])
def test_missing_or_withdrawn_body_and_unknown_publication_preserved(tmp_path, state):
    doc = document()
    event = doc["events"][0]
    proof = event["evidence"]
    event["bodySha256"] = None
    if state == "withdrawn":
        proof["body"]["operation"] = "delete"
    else:
        proof["body"] = None
        event["bodyReceivedAt"] = None
    if state == "unpublished":
        event["publishedAt"] = None
        proof["content"]["document"]["published_at"] = None
        doc["scope"]["end"] = "2025-01-05"
        proof["identity"]["effectiveOn"] = "2025-01-04"
    events, _ = load(tmp_path, doc)
    assert events.height == 1
    if state == "unpublished":
        assert events["event_at"].null_count() == 1
