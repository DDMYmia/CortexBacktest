"""Version-pinned strategy decisions, not an execution or broker integration.

The account adapter must establish account/contract identity and complete scoped
positions before constructing PortfolioObservation. This module never infers a
zero position from a missing row. It has no broker client or approval capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from hashlib import sha256
from itertools import groupby
import json
from math import isfinite
from pathlib import Path
import re

from .adapters.observation_file import (
    instrument_symbol, load_observation_file, observation_bars,
)
from .contracts import Bar, Order
from .event.manifest import engine_code_digest
from .gates import GATE_RULES, SeriesGate
from .strategy_library import STRATEGIES


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA-256 digest")


def _aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamps must include timezones")


def _quantity(value: Decimal) -> None:
    if (not isinstance(value, Decimal) or not value.is_finite()
            or len(value.as_tuple().digits) > 28
            or not -12 <= int(value.as_tuple().exponent) <= 12):
        raise ValueError("Unsupported position quantity; no rounding permitted")


@dataclass(frozen=True, slots=True)
class StrategyArtifact:
    """Immutable local artifact: a library strategy, its parameters and optional gate.

    No dynamic imports or executable payloads: `family` names an entry of the
    strategy library, so the same decision code runs in a backtest and the shadow run.
    """

    engine_code_sha256: str
    symbols: tuple[str, ...]
    family: str = "sma_cross"
    parameters: dict = field(default_factory=lambda: {"fast": 15, "slow": 30})
    quantity: float = 1.0
    gate: dict | None = None
    max_signal_age_seconds: int = 300
    max_account_age_seconds: int = 60

    def __post_init__(self) -> None:
        _digest(self.engine_code_sha256)
        if (not isinstance(self.symbols, tuple) or not self.symbols
                or any(not isinstance(s, str) or re.fullmatch(r"INSTRUMENT:[1-9][0-9]*", s)
                       is None for s in self.symbols)
                or tuple(sorted(set(self.symbols))) != self.symbols):
            raise ValueError("Artifact symbols must be sorted unique instrument identities")
        if type(self.quantity) not in (int, float) or not isfinite(self.quantity):
            raise ValueError("quantity must be finite and numeric")
        object.__setattr__(self, "quantity", float(self.quantity))
        _quantity(Decimal(str(self.quantity)))
        if self.family not in STRATEGIES or not isinstance(self.parameters, dict) or "quantity" in self.parameters:
            raise ValueError("Artifact needs a library family and its parameters without quantity")
        self.model()
        if self.gate is not None and (set(self.gate) != {"source", "name", "rule", "lookback"}
                                      or self.gate["rule"] not in GATE_RULES):
            raise ValueError("Artifact gate needs source, name, a known rule and lookback")
        for age in (self.max_signal_age_seconds, self.max_account_age_seconds):
            if type(age) is not int or not 0 < age <= 86400:
                raise ValueError("Freshness limits must be integer seconds in 1..86400")

    def model(self):
        return STRATEGIES[self.family](**self.parameters, quantity=self.quantity)

    def document(self) -> dict:
        return {
            "artifactVersion": "cortex-strategy-v1",
            "engineCodeSha256": self.engine_code_sha256,
            "symbols": list(self.symbols),
            "family": self.family,
            "parameters": {**self.parameters, "quantity": self.quantity},
            "gate": self.gate,
            "maxSignalAgeSeconds": self.max_signal_age_seconds,
            "maxAccountAgeSeconds": self.max_account_age_seconds,
        }

    def to_bytes(self) -> bytes:
        return _json(self.document())

    @property
    def sha256(self) -> str:
        return sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, content: bytes, *, expected_sha256: str) -> StrategyArtifact:
        _digest(expected_sha256)
        if sha256(content).hexdigest() != expected_sha256:
            raise ValueError("Strategy artifact digest mismatch")
        try:
            doc = json.loads(content)
            parameters = dict(doc["parameters"])
            quantity = parameters.pop("quantity")
            artifact = cls(doc["engineCodeSha256"], tuple(doc["symbols"]), doc["family"], parameters, quantity,
                           doc["gate"], doc["maxSignalAgeSeconds"], doc["maxAccountAgeSeconds"])
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("Invalid strategy artifact") from exc
        # Exact canonical bytes reject extra fields, versions and duplicate JSON keys.
        if artifact.to_bytes() != content:
            raise ValueError("Expected the canonical cortex-strategy-v1 artifact")
        artifact.verify_runtime()
        return artifact

    def verify_runtime(self) -> None:
        if self.engine_code_sha256 != engine_code_digest():
            raise ValueError("Strategy artifact engine code mismatch")

    def backtest_strategy(self, warmup: WarmupHistory | None = None, gate: SeriesGate | None = None):
        """The library strategy, seeded and bound to this artifact's scope, for the event engine.

        Its constructor takes the artifact and warmup documents, so the run manifest
        records both and a different warmup is a different run.
        """
        self.verify_runtime()
        pinned = self

        class Pinned(STRATEGIES[self.family]):  # type: ignore[misc,valid-type]
            def __init__(self, artifact: dict, warmup: dict | None):
                super().__init__(**pinned.parameters, quantity=pinned.quantity)
                self.artifact, self.warmup, self.gate = artifact, warmup, gate

            def on_bars(self, bars: list[Bar]) -> list[Order]:
                if any(bar.symbol not in pinned.symbols for bar in bars):
                    raise ValueError("Bar outside strategy artifact scope")
                if warmup_history is not None and any(bar.timestamp <= warmup_history.as_of for bar in bars):
                    raise ValueError("Stream cannot run before or at the warmup boundary")
                return super().on_bars(bars)

        warmup_history = warmup
        model = Pinned({"sha256": self.sha256, **self.document()}, None if warmup is None else warmup.descriptor())
        if warmup is not None:
            warmup.seed(model, self)
        return model


@dataclass(frozen=True, slots=True)
class WarmupHistory:
    """Receipt-validated history from load_warmup_history, never a tradable feed."""

    artifact_sha256: str
    input_sha256: str
    as_of: datetime
    # Symbol, market event time, actual receipt availability, closing price.
    closes: tuple[tuple[str, datetime, datetime, float], ...]

    def __post_init__(self) -> None:
        _digest(self.artifact_sha256)
        _digest(self.input_sha256)
        _aware(self.as_of)
        if not isinstance(self.closes, tuple) or not self.closes:
            raise ValueError("Warmup closes must be an immutable nonempty sequence")
        latest = {}
        for row in self.closes:
            if not isinstance(row, tuple) or len(row) != 4:
                raise ValueError("Invalid warmup close")
            symbol, event, available, close = row
            _aware(event)
            _aware(available)
            if (not event <= available <= self.as_of or type(close) not in (int, float)
                    or not isfinite(close)):
                raise ValueError("Warmup must be finite and known by its boundary")
            if symbol in latest and event <= latest[symbol]:
                raise ValueError("Warmup market events must increase per instrument")
            latest[symbol] = event

    def descriptor(self) -> dict:
        return {"inputSha256": self.input_sha256, "asOf": self.as_of.isoformat(),
                "recordCount": len(self.closes), "ordering": "market-event-time"}

    def seed(self, model, artifact: StrategyArtifact) -> None:
        """Replay the boundary's closes through the model's own decision step; no orders."""
        if self.artifact_sha256 != artifact.sha256:
            raise ValueError("Warmup belongs to a different strategy artifact")
        if {symbol for symbol, _, _, _ in self.closes} != set(artifact.symbols):
            raise ValueError("Warmup must cover strategy artifact scope")
        for event, rows in groupby(sorted(self.closes, key=lambda r: (r[1], r[0])), key=lambda r: r[1]):
            model.observe_targets([Bar(symbol, event, close, close, close, close) for symbol, _, _, close in rows])


@dataclass(frozen=True, slots=True)
class PortfolioObservation:
    """Caller-verified scoped holdings; source digest is a reference, not authentication.

    Every strategy instrument must appear, explicitly including zero holdings.
    Pending scoped orders block proposals because their eventual fills are unknown.
    Quantities use at most 28 digits and 12 decimal places; unsupported precision
    is rejected, never rounded to an executable quantity.
    """

    account_ref: str
    source_sha256: str
    event_time: datetime
    positions: tuple[tuple[str, Decimal], ...]
    pending_symbols: tuple[str, ...]

    def validate(self, symbols: tuple[str, ...]) -> None:
        _digest(self.source_sha256)
        _aware(self.event_time)
        if not isinstance(self.account_ref, str) or not self.account_ref.strip():
            raise ValueError("Account identity is required")
        if tuple(symbol for symbol, _ in self.positions) != symbols:
            raise ValueError("Explicit complete scoped positions are required")
        for _, quantity in self.positions:
            _quantity(quantity)
            if Decimal(str(float(quantity))) != quantity:
                raise ValueError("Position cannot round-trip through the strategy numeric model")
        if any(symbol not in symbols for symbol in self.pending_symbols):
            raise ValueError("Pending order identity outside the scoped observation")
        if self.pending_symbols:
            raise ValueError("Pending scoped orders require reconciliation before shadow proposals")


def _load_market_document(
    artifact: StrategyArtifact, path: str | Path, *, digest: str, as_of: datetime,
) -> dict:
    artifact.verify_runtime()
    _digest(digest)
    document = load_observation_file(path, sha256=digest, as_of=as_of)
    if document["recordVersion"] != "market-bars-v2":
        raise ValueError("Shadow strategies require market-bars-v2 observations")
    rows = document["records"]
    if tuple(sorted({instrument_symbol(r["instrumentId"]) for r in rows})) != artifact.symbols:
        raise ValueError("Market data must exactly cover strategy artifact scope")
    return document


def load_warmup_history(
    artifact: StrategyArtifact, path: str | Path, *, digest: str, as_of: datetime,
) -> WarmupHistory:
    """Freeze historical knowledge at a boundary, without backdating its receipts.

    All rows must already be available at as_of. Event ordering is valid for model
    initialization at that boundary, not for orders at the historical event times.
    """
    document = _load_market_document(artifact, path, digest=digest, as_of=as_of)
    closes = tuple(sorted((
        (instrument_symbol(row["instrumentId"]), datetime.fromisoformat(row["sessionClose"]),
         datetime.fromisoformat(row["availableAt"]), float(row["close"]))
        for row in document["records"]
    ), key=lambda r: (r[1], r[0])))
    return WarmupHistory(artifact.sha256, digest, as_of, closes)


def load_artifact_bars(
    artifact: StrategyArtifact, path: str | Path, *, digest: str, as_of: datetime,
    warmup: WarmupHistory | None = None,
) -> tuple[dict, list[Bar]]:
    """Shared frozen stream path; warmup is explicitly separate from tradable bars."""
    document = _load_market_document(artifact, path, digest=digest, as_of=as_of)
    if warmup is not None and warmup.artifact_sha256 != artifact.sha256:
        raise ValueError("Warmup belongs to a different strategy artifact")
    latest = {}
    for row in sorted(document["records"], key=lambda r: (
        datetime.fromisoformat(r["availableAt"]), r["instrumentId"],
    )):
        symbol = instrument_symbol(row["instrumentId"])
        event = datetime.fromisoformat(row["sessionClose"])
        available = datetime.fromisoformat(row["availableAt"])
        if warmup is not None and event <= warmup.as_of:
            raise ValueError("Stream market event must follow the warmup boundary")
        if symbol in latest:
            prior_event, prior_available = latest[symbol]
            if event <= prior_event or available <= prior_available:
                raise ValueError("Market events and availability must increase per instrument")
        latest[symbol] = (event, available)
    return document, observation_bars(document)


def evaluate_shadow(
    artifact: StrategyArtifact,
    path: str | Path,
    *,
    digest: str,
    as_of: datetime,
    portfolio: PortfolioObservation,
    now: datetime,
    warmup: WarmupHistory | None = None,
    gate: SeriesGate | None = None,
) -> dict:
    """Replay causal targets and compare with observed holdings, without simulated fills.

    Output is diagnostic and cannot be submitted as an order intent. Stable logical
    signal IDs let a future durable consumer reject repeat or conflicting deliveries;
    this calculation alone does not implement cross-process deduplication.
    """
    _aware(now)
    _aware(as_of)
    if as_of > now:
        raise ValueError("Future data cutoff")
    portfolio.validate(artifact.symbols)
    account_expiry = portfolio.event_time + timedelta(seconds=artifact.max_account_age_seconds)
    if not portfolio.event_time <= now < account_expiry:
        raise ValueError("Stale or future account observation")
    document, bars = load_artifact_bars(artifact, path, digest=digest, as_of=as_of, warmup=warmup)
    latest = {}
    for row in document["records"]:
        symbol = instrument_symbol(row["instrumentId"])
        event = datetime.fromisoformat(row["sessionClose"])
        if symbol not in latest or event > latest[symbol]:
            latest[symbol] = event
    expiries = {symbol: min(account_expiry, event + timedelta(
        seconds=artifact.max_signal_age_seconds)) for symbol, event in latest.items()}
    if any(now >= expiry for expiry in expiries.values()):
        raise ValueError("Stale market event, regardless of its receipt time")

    if (artifact.gate is None) != (gate is None):
        raise ValueError("A gated artifact needs its gate series, and only a gated one")
    model = artifact.model()
    if warmup is not None:
        warmup.seed(model, artifact)
    targets: dict[str, float] = {}
    last: dict[str, Bar] = {}
    for _, batch in groupby(bars, key=lambda bar: bar.timestamp):
        rows = list(batch)
        targets = model.observe_targets(rows)
        last.update((bar.symbol, bar) for bar in rows)
    if set(targets) != set(artifact.symbols):
        raise ValueError("Insufficient strategy warmup")
    if gate is not None:
        targets = {symbol: target if gate(last[symbol]) else 0.0 for symbol, target in targets.items()}

    decisions = []
    with localcontext() as context:
        context.prec = 64
        for symbol, current in portfolio.positions:
            target = Decimal(str(targets[symbol]))
            delta = (target - current if model.position_delta(float(target), float(current))
                     else Decimal(0))
            event_time = latest[symbol].astimezone(UTC).isoformat()
            signal_id = sha256(_json([artifact.sha256, portfolio.account_ref,
                                      symbol, event_time])).hexdigest()
            decisions.append({
                "signalId": signal_id, "symbol": symbol, "eventTime": event_time,
                "expiresAt": expiries[symbol].astimezone(UTC).isoformat(),
                "currentQuantity": str(current), "targetQuantity": str(target),
                "side": "BUY" if delta > 0 else "SELL" if delta < 0 else "HOLD",
                "quantity": str(abs(delta)),
            })
    payload = {
        "contractVersion": "shadow-decision-v1",
        "mode": "shadow", "executionAuthorized": False,
        "strategySha256": artifact.sha256, "engineCodeSha256": artifact.engine_code_sha256,
        "inputSha256": digest, "snapshotAsOf": as_of.astimezone(UTC).isoformat(),
        "warmup": None if warmup is None else warmup.descriptor(),
        "accountRef": portfolio.account_ref, "accountSourceSha256": portfolio.source_sha256,
        "accountEventTime": portfolio.event_time.astimezone(UTC).isoformat(),
        "decisions": decisions,
    }
    return {**payload, "decisionSha256": sha256(_json(payload)).hexdigest()}
