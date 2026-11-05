"Same-USD received equity style/return bundle, for analytics only."

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import math
from pathlib import Path
import re

import polars as pl

from .observation_file import validate_observation_file
from .native_research_weights import native_research_weights
from .sec_identity import verify_sec_identities
from .. import strict_json

_POLICY = "sec-us-gaap-usd-date-bound-v1"
_VENUES = {
    "NASDAQ": "XNAS",
    "Nasdaq": "XNAS",
    "The Nasdaq Stock Market LLC": "XNAS",
    "Nasdaq Stock Market LLC": "XNAS",
    "The Nasdaq Global Select Market": "XNAS",
    "The New York Stock Exchange": "XNYS",
    "NASDAQ Global Market": "XNAS",
    "NASDAQ Global Select": "XNAS",
    "NASDAQ Capital Market": "XNAS",
    "NYSE": "XNYS",
    "New York Stock Exchange": "XNYS",
    "New York Stock Exchange Arca": "ARCX",
    "AMEX": "XASE",
    "CBOE": "XCBO",
}
_US = {"XNAS", "XNYS", "ARCX", "XASE", "BATS", "IEXG", "XCBO"}


@dataclass(frozen=True)
class EquityRiskInputs:
    manifest: dict
    characteristics: pl.DataFrame
    returns: pl.DataFrame
    execution_admission: dict
    style_expires_at: datetime
    identity_evidence: dict
    portfolio_weights: dict | None = None
    period_returns: pl.DataFrame | None = None


def _load(content):
    return strict_json.loads(content)


def _number(value, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or positive and value <= 0:
        raise ValueError("finite explicit equity input number required")
    return float(value)


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def load_equity_risk_inputs(path, *, sha256):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("equity input manifest symlink refused")
    raw = path.read_bytes()
    if _sha(raw) != sha256:
        raise ValueError("equity risk manifest digest mismatch")
    manifest = _load(raw)
    if (
        manifest["contractVersion"] != "equity-risk-inputs-v2"
        or manifest["currency"] != "USD"
        or manifest["calendar"] != "us_equity"
        or manifest["executionAdmitted"] is not False
        or manifest["qualificationIssued"] is not False
        or manifest["executionAuthorized"] is not False
        or type(manifest["returnWindow"]) is not int
        or not 60 <= manifest["returnWindow"] <= 252
    ):
        raise ValueError("USD equity analytics-only contract required")
    cutoff = strict_json.clock(manifest["asOf"])
    if cutoff > datetime.now(UTC):
        raise ValueError("future equity snapshot cutoff")
    ids = manifest["instruments"]
    if not 10 <= len(ids) <= 100 or len(set(ids)) != len(ids) or any(type(i) is not int or i <= 0 for i in ids):
        raise ValueError("10 to 100 distinct equity identities required")
    files = {}
    for name, expected in manifest["files"].items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", "..", "manifest.json"}:
            raise ValueError("plain equity component filename required")
        target = path.with_name(name)
        if target.is_symlink():
            raise ValueError("equity component symlink refused")
        value = target.read_bytes()
        if _sha(value) != expected:
            raise ValueError("equity component digest mismatch")
        files[name] = value
    style = _load(files["style-manifest.json"])
    if (
        _sha(files["style-manifest.json"]) != manifest["styleInputSha256"]
        or style["contractVersion"] != "style-inputs-v2"
        or style["currency"] != "USD"
        or style["equityPolicy"] != _POLICY
        or sorted(style["instruments"]) != sorted(ids)
        or style["minimumAssets"] != 10
        or style["definitions"]
        != {"size": {"log_market_cap": 1.0}, "value": {"book_to_market": 1.0}}
    ):
        raise ValueError("matching explicit current USD style definitions required")
    for name, expected in style["files"].items():
        if name not in files or _sha(files[name]) != expected:
            raise ValueError("style submanifest component mismatch")
    style_at = strict_json.clock(style["asOf"])
    if (
        manifest["styleSnapshotAt"] != style["asOf"]
        or not style_at <= cutoff <= style_at + timedelta(hours=24)
        or type(style["maxProfileAgeHours"]) is not int
        or not 1 <= style["maxProfileAgeHours"] <= 168
        or type(style["maxBookAgeDays"]) is not int
        or not 1 <= style["maxBookAgeDays"] <= 730
    ):
        raise ValueError("current bounded style input clocks required")
    identities = _load(files["identities.json"])
    if sorted(t["instrument_id"] for t in identities) != sorted(ids):
        raise ValueError("identity evidence must cover exact equity universe")
    targets = {t["instrument_id"]: t for t in identities}
    identity_sha = _sha(files["filing-identities.json"])
    if (
        style["identityPolicy"] != "received-sec-filing-snapshots-v1"
        or style["identityInputSha256"] != identity_sha
        or manifest["identityInputSha256"] != identity_sha
    ):
        raise ValueError("SEC identity submanifest binding required")
    identity_known, identity_summary = verify_sec_identities(
        _load(files["filing-identities.json"]), targets, as_of=style_at, venues=_VENUES
    )
    identity_summary["sha256"] = identity_sha
    bindings = _load(files["profile-bindings.json"])
    if len(bindings) != len(ids) or sorted(b["instrumentId"] for b in bindings) != sorted(ids):
        raise ValueError("profile bindings must cover exact equity universe")
    equities = _load(files["equity-observations.json"])
    for book in equities:
        target = targets[int(book["instrument_id"])]
        md = book["metadata"]
        stamps = [
            strict_json.clock(book[k])
            for k in [
                "event_time",
                "known_at",
                "recorded_at",
                "identity_known_at",
                "score_known_at",
                "entity_known_at",
            ]
        ]
        if (
            book["contractVersion"] != "issuer-facts-v1"
            or book["source"] != "sec-edgar"
            or book["factor_name"] != "sec.equity"
            or md.get("definition") != _POLICY
            or md.get("taxonomy") != "us-gaap"
            or md.get("unit") != "USD"
            or str(md.get("cik") or "").lstrip("0") != str(target["cik"]).lstrip("0")
            or book["symbol"] != target["primary_symbol"]
            or book["symbol_source"] != "fmp"
            or book["identity_scope"] != "issuer_at_cutoff_not_historical_security"
            or book["identity_effective_on"] != str(style_at.astimezone(UTC).date())
            or max(stamps) > style_at
            or stamps[0] > stamps[1]
        ):
            raise ValueError("received current-issuer USD SEC facts required")
        _number(book["value"])
    expected_records = []
    profile_expiries = []
    for index, binding in enumerate(bindings):
        identifier = binding["instrumentId"]
        target = targets[identifier]
        receipt_bytes, body = files[f"profile-{index}.json"], files[f"profile-{index}.data"]
        receipt = _load(receipt_bytes)
        payload = _load(body)
        if (
            binding["sha256"] != _sha(receipt_bytes)
            or receipt["contractVersion"] != "research-source-receipt-v1"
            or receipt["rawSha256"] != _sha(body)
            or receipt["rawBytes"] != len(body)
            or receipt["source"] != "fmp"
            or receipt["dataset"] != "profile"
            or receipt["status"] != "received"
            or receipt["httpStatus"] != 200
            or not isinstance(payload, list)
            or len(payload) != 1
        ):
            raise ValueError("received profile bytes and digest required")
        profile = payload[0]
        received = strict_json.clock(receipt["receivedAt"])
        mic = _VENUES.get(target["exchange_mic"], target["exchange_mic"])
        labels = [profile[k] for k in ["exchange", "exchangeShortName"] if profile.get(k)]
        cik = str(profile.get("cik") or "").lstrip("0")
        if (
            target["asset_class"] != "equity"
            or target["symbol_namespace"] != "fmp"
            or target["currency"] != "USD"
            or profile["currency"] != "USD"
            or mic not in _US
            or not labels
            or any(_VENUES.get(label, label) != mic for label in labels)
            or profile["symbol"] != target["primary_symbol"]
            or receipt["scope"]["symbol"] != target["primary_symbol"]
            or not target.get("isin")
            or profile.get("isin") != target["isin"]
            or not cik
            or cik != str(target["cik"]).lstrip("0")
            or not cutoff - timedelta(hours=style["maxProfileAgeHours"]) <= received <= style_at
        ):
            raise ValueError("current matched USD profile identity/currency/clock required")
        profile_expiries.append(received + timedelta(hours=style["maxProfileAgeHours"]))
        candidates = [b for b in equities if int(b["instrument_id"]) == identifier]
        if not candidates:
            raise ValueError("missing current SEC equity; no implicit zero")
        book = max(candidates, key=lambda b: (strict_json.clock(b["event_time"]), strict_json.clock(b["known_at"])))
        same = [
            b
            for b in candidates
            if b["event_time"] == book["event_time"] and b["known_at"] == book["known_at"]
        ]
        if len({b["value"] for b in same}) != 1:
            raise ValueError("ambiguous equity vintage")
        if (
            not style_at - timedelta(days=style["maxBookAgeDays"])
            <= strict_json.clock(book["event_time"])
            <= style_at
        ):
            raise ValueError("stale latest book equity")
        cap = _number(profile.get("marketCap"), positive=True)
        known = max(
            received,
            strict_json.clock(book["known_at"]),
            strict_json.clock(book["recorded_at"]),
            identity_known[identifier],
        )
        expected_records.append(
            {
                "timestamp": style_at.isoformat(),
                "symbol": str(identifier),
                "known_at": known.isoformat(),
                "market_cap": cap,
                "log_market_cap": math.log(cap),
                "book_to_market": _number(book["value"]) / cap,
            }
        )
    expected_records.sort(key=lambda r: int(r["symbol"]))
    saved = sorted(_load(files["characteristics.json"]), key=lambda r: int(r["symbol"]))
    if len(saved) != len(ids):
        raise ValueError("complete received characteristics required")
    for actual, expected in zip(saved, expected_records, strict=True):
        if (
            set(actual) != set(expected)
            or actual["symbol"] != expected["symbol"]
            or any(strict_json.clock(actual[k]) != strict_json.clock(expected[k]) for k in ["timestamp", "known_at"])
        ):
            raise ValueError("characteristic identity/clock mismatch")
        for field in ["market_cap", "log_market_cap", "book_to_market"]:
            if not math.isclose(
                _number(actual[field]), expected[field], rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError("characteristics do not reproduce received profile/SEC facts")
    if _sha(files["equity-prices.json"]) != manifest["priceInputSha256"]:
        raise ValueError("price input submanifest mismatch")
    prices = validate_observation_file(_load(files["equity-prices.json"]), as_of=cutoff)
    if (
        prices["recordVersion"] != "market-bars-v4"
        or prices["scope"]["source"] != "FMP"
        or prices["scope"]["adjustment"] != "total_return"
        or prices["scope"]["end"] != manifest["end"]
    ):
        raise ValueError("explicit received FMP total-return analytics prices required")
    latest = {}
    for row in prices["records"]:
        key = row["instrumentId"], row["sessionDate"]
        if key not in latest or strict_json.clock(row["recordedAt"]) > strict_json.clock(latest[key]["recordedAt"]):
            latest[key] = row
    days = sorted({day for _, day in latest})
    if (
        len(days) != manifest["returnWindow"] + 1
        or len(latest) != len(ids) * len(days)
        or {int(i) for i, _ in latest} != set(ids)
    ):
        raise ValueError("complete common equity price sessions required")
    rows, period_rows = [], []
    for identifier in map(str, sorted(ids)):
        for before, day in zip(days, days[1:]):
            a, b = latest[identifier, before], latest[identifier, day]
            if any(
                r.get("invalidReason")
                or r["ohlcBasis"] != "unadjusted"
                or not r["adjustmentProvenance"]
                for r in [a, b]
            ):
                raise ValueError("latest received price revision invalid; no older fallback")
            previous, current = (
                _number(a["adjClose"], positive=True),
                _number(b["adjClose"], positive=True),
            )
            rows.append(
                {
                    "symbol": identifier,
                    "timestamp": strict_json.clock(b["sessionClose"]),
                    "known_at": max(strict_json.clock(a["availableAt"]), strict_json.clock(b["availableAt"])),
                    "return": current / previous - 1,
                }
            )
            period_rows.append({'period_start': strict_json.clock(a['sessionClose']), **rows[-1]})
    characteristics = pl.DataFrame(expected_records).with_columns(
        pl.col("timestamp", "known_at").str.to_datetime(time_zone="UTC")
    )
    admission = _load(files["execution-admission.json"])
    if (
        admission["contractVersion"] != "equity-execution-admission-v1"
        or admission["executionAdmitted"] is not False
        or admission["qualificationIssued"] is not False
        or admission["executionAuthorized"] is not False
        or not admission["blockers"]
        or admission["requiredIdentityBindings"] != len(latest)
        or not 0
        <= admission["receivedByHistoricalOpen"]
        <= admission["resolvedAtExport"]
        <= len(latest)
    ):
        raise ValueError("unqualified analytics-only execution admission required")
    if "historicalInputs" in admission:
        from .historical_input_semantics import validate_historical_inputs, received_reference_claims
        frozen_bindings = _load(files["identity-bindings.json"])
        if len(frozen_bindings) != len(latest):
            raise ValueError("historical_identity_binding_count_mismatch")
        claims = (received_reference_claims(_load(files["historical-reference-receipts.json"]), instruments=ids, cutoff=cutoff)
                  if "historical-reference-receipts.json" in files else None)
        validate_historical_inputs(admission["historicalInputs"], frozen_bindings, dates=set(days), cutoff=cutoff, reference_receipts=claims)
    weight_names = {f"{role}-{name}" for role in ("portfolio", "benchmark")
                    for name in ("positions.ndjson", "equity_curve.ndjson", "run_manifest.json")} | {"weight-source.json"}
    present = weight_names.intersection(files)
    if present and present != weight_names:
        raise ValueError("complete native weight input bundle required")
    weights = native_research_weights(files, instruments=ids, cutoff=cutoff) if present else None
    return EquityRiskInputs(
        manifest,
        characteristics,
        pl.DataFrame(rows),
        admission,
        min(profile_expiries),
        identity_summary,
        weights,
        period_returns=pl.DataFrame(period_rows),
    )
