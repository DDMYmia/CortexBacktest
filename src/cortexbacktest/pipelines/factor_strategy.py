"""Portable frozen factor identity shared by simulation and portfolio planning."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from hashlib import sha256
import json
from math import isfinite
import re

from ..event import RebalanceConfig
from ..event.manifest import engine_code_digest
from ..shadow import PortfolioObservation
from .. import strict_json
from .mining_store import load_mining_result


@dataclass(frozen=True, slots=True)
class FactorValuation:
    """Caller-verified valuation receipt; a digest is provenance, not authentication.

    Equity and raw prices must use the declared common currency. Broker/data
    adapters must verify that currency and the referenced receipt contents.
    """

    portfolio: PortfolioObservation
    source_sha256: str
    event_time: datetime
    currency: str
    equity: Decimal
    prices: tuple[tuple[str, Decimal], ...]

    @classmethod
    def from_bytes(cls, content: bytes, *, expected_sha256: str, account_ref: str,
                   symbols: tuple[str, ...], now: datetime, max_age_seconds: int):
        """Read a trusted adapter's delivery, not an authenticated broker receipt.

        The caller supplies the delivery digest and account independently. Raw
        receipt verification remains the producing adapter's responsibility.
        """
        if sha256(content).hexdigest() != expected_sha256:
            raise ValueError("Valuation delivery digest mismatch")

        def amounts(rows):
            if not isinstance(rows, list) or any(
                not isinstance(row, list) or len(row) != 2
                or not all(isinstance(value, str) for value in row) for row in rows
            ):
                raise ValueError("Valuation amounts require decimal strings")
            return tuple((symbol, Decimal(value)) for symbol, value in rows)

        try:
            doc = strict_json.loads(content)
            mark = doc["valuation"]
            if doc["accountRef"] != account_ref or mark["sourceSha256"] != doc["sourceSha256"]:
                raise ValueError("Valuation account or receipt mismatch")
            if (not isinstance(mark["equity"], str) or not isinstance(doc["pendingSymbols"], list)
                    or any(not isinstance(s, str) for s in doc["pendingSymbols"])):
                raise ValueError("Invalid valuation equity or pending symbols")
            portfolio = PortfolioObservation(
                account_ref, doc["sourceSha256"], datetime.fromisoformat(doc["eventTime"]),
                amounts(doc["positions"]), tuple(doc["pendingSymbols"]))
            valuation = cls(portfolio, mark["sourceSha256"], datetime.fromisoformat(mark["eventTime"]),
                            mark["currency"], Decimal(mark["equity"]), amounts(mark["prices"]))
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid valuation delivery") from error
        valuation.validate(symbols, now=now, max_age_seconds=max_age_seconds)
        return valuation

    def validate(self, symbols: tuple[str, ...], *, now: datetime, max_age_seconds: int) -> None:
        self.portfolio.validate(symbols)
        if (type(max_age_seconds) is not int or not 0 < max_age_seconds <= 86400
                or now.tzinfo is None or now.utcoffset() is None
                or self.event_time.tzinfo is None or self.event_time.utcoffset() is None):
            raise ValueError("Valuation requires aware clocks and a bounded freshness limit")
        for event in (self.event_time, self.portfolio.event_time):
            if not event <= now < event + timedelta(seconds=max_age_seconds):
                raise ValueError("Stale or future portfolio valuation")
        if (not isinstance(self.source_sha256, str) or re.fullmatch(r"[a-f0-9]{64}", self.source_sha256) is None
                or self.currency not in {"USD", "USDT"}):
            raise ValueError("Valuation requires source identity and supported common currency")
        if not isinstance(self.prices, tuple) or tuple(s for s, _ in self.prices) != symbols:
            raise ValueError("Explicit raw prices must exactly cover strategy scope")
        for value in (self.equity, *(p for _, p in self.prices)):
            if (not isinstance(value, Decimal) or not value.is_finite()
                    or not isfinite(float(value)) or Decimal(str(float(value))) != value):
                raise ValueError("Valuation cannot round-trip through the strategy numeric model")
        if self.equity < 0 or any(price <= 0 for _, price in self.prices):
            raise ValueError("Valuation requires nonnegative capital and positive prices")
        if self.equity == 0 and any(quantity != 0 for _, quantity in self.portfolio.positions):
            raise ValueError("Zero capital conflicts with nonzero holdings")


@dataclass(frozen=True, slots=True)
class FrozenFactorStrategy:
    """A selection and its sizing policy, never a broker execution authorization.

    Local checkpoint paths are supplied by the caller, not included in identity.
    Moving verified files between machines therefore preserves the strategy hash.
    """

    engine_code_sha256: str
    selection_sha256: str
    factor: str
    symbols: tuple[str, ...]
    rebalance: RebalanceConfig = RebalanceConfig()
    research_only: bool = True

    def __post_init__(self):
        if type(self.research_only) is not bool:
            raise ValueError("Research-only policy must be boolean")
        for digest in (self.engine_code_sha256, self.selection_sha256):
            if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
                raise ValueError("Factor strategy requires SHA-256 identities")
        if not isinstance(self.factor, str) or not self.factor:
            raise ValueError("Factor name is required")
        if (not isinstance(self.symbols, tuple) or not self.symbols
                or any(not isinstance(s, str) or re.fullmatch(r"INSTRUMENT:[1-9][0-9]*", s) is None
                       for s in self.symbols) or tuple(sorted(set(self.symbols))) != self.symbols):
            raise ValueError("Factor strategy requires sorted unique instrument identities")
        if (not isinstance(self.rebalance, RebalanceConfig)
                or not isfinite(self.rebalance.min_trade_value)
                or type(self.rebalance.allow_fractional) is not bool
                or not isinstance(self.rebalance.tag, str)):
            raise ValueError("Invalid frozen rebalance policy")
        if self.rebalance.quantity_steps and tuple(s for s, _ in self.rebalance.quantity_steps) != self.symbols:
            raise ValueError("Frozen quantity steps must exactly cover sorted strategy scope")

    @classmethod
    def from_checkpoint(cls, checkpoint, *, selection_sha256: str, factor: str,
                        symbols: tuple[str, ...], rebalance: RebalanceConfig | None = None):
        from .mining_execution import _frozen_rebalance

        selection = load_mining_result(checkpoint, manifest_sha256=selection_sha256)
        if factor not in {spec.name for spec in selection.selected}:
            raise ValueError("Factor is not part of the frozen selection")
        return cls(engine_code_digest(), selection_sha256, factor, symbols, _frozen_rebalance(selection, rebalance))

    def to_bytes(self) -> bytes:
        return json.dumps({"artifactVersion": "cortex-factor-strategy-v1",
                           "engineCodeSha256": self.engine_code_sha256,
                           "selectionSha256": self.selection_sha256, "factor": self.factor,
                           "symbols": self.symbols, "rebalance": self.rebalance.to_document(),
                           **({"researchOnly": True} if self.research_only else {})},
                          sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    @property
    def sha256(self) -> str:
        return sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, content: bytes, *, expected_sha256: str):
        if sha256(content).hexdigest() != expected_sha256:
            raise ValueError("Factor strategy digest mismatch")
        try:
            doc = json.loads(content)
            strategy = cls(doc["engineCodeSha256"], doc["selectionSha256"], doc["factor"],
                           tuple(doc["symbols"]), RebalanceConfig.from_document(doc["rebalance"]), doc.get("researchOnly", False))
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid factor strategy artifact") from error
        if strategy.to_bytes() != content:
            raise ValueError("Expected canonical cortex-factor-strategy-v1 bytes")
        strategy.verify_runtime()
        return strategy

    def verify_runtime(self):
        if self.engine_code_sha256 != engine_code_digest():
            raise ValueError("Factor strategy engine code mismatch")

    def quantity_changes(self, checkpoint, *, revision_panel, as_of, valuation: FactorValuation,
                         max_age_seconds: int, max_signal_age_seconds: int, inner_execution=None):
        from .mining_execution import frozen_factor_quantity_changes

        self.verify_runtime()
        if not self.research_only and revision_panel.knowledge_policy == "source_publication_replay":
            raise ValueError("Frozen strategy research policy mismatch")
        valuation.validate(self.symbols, now=as_of, max_age_seconds=max_age_seconds)
        if type(max_signal_age_seconds) is not int or not 0 < max_signal_age_seconds <= 86400:
            raise ValueError("Factor signal requires a bounded freshness limit")
        if revision_panel.decisions.is_empty():
            raise ValueError("No factor decision available")
        event = revision_panel.decisions[-1, "decision_time"]
        if not event <= as_of < event + timedelta(seconds=max_signal_age_seconds):
            raise ValueError("Stale or future factor decision")
        return frozen_factor_quantity_changes(
            checkpoint, manifest_sha256=self.selection_sha256, factor=self.factor,
            revision_panel=revision_panel, as_of=as_of,
            prices={symbol: float(price) for symbol, price in valuation.prices},
            quantities={symbol: float(quantity) for symbol, quantity in valuation.portfolio.positions},
            equity=float(valuation.equity), inner_execution=inner_execution, rebalance=self.rebalance)

    def simulate(self, checkpoint, *, dataset, revision_panel, execution_bars, broker, inner_execution=None):
        from .mining_execution import run_frozen_factor_simulation

        return run_frozen_factor_simulation(
            checkpoint, manifest_sha256=self.selection_sha256, dataset=dataset,
            revision_panel=revision_panel, execution_bars=execution_bars, broker=broker,
            inner_execution=inner_execution, strategy=self)[self.factor]

    def shadow_decision(self, checkpoint, *, revision_panel, as_of, valuation: FactorValuation,
                        max_age_seconds: int, max_signal_age_seconds: int, inner_execution=None):
        """Publish the simulated sizing policy as diagnostic signals, never orders.

        The existing delivery writer handles persistence. Admission must separately
        pin this factor artifact and validate venue capabilities and exposure.
        One receipt must cover holdings and valuation so source lineage is retained
        in the existing accountSourceSha256 field.
        """
        if valuation.source_sha256 != valuation.portfolio.source_sha256:
            raise ValueError("Factor shadow requires one holdings and valuation receipt")
        changes = self.quantity_changes(
            checkpoint, revision_panel=revision_panel, as_of=as_of, valuation=valuation,
            max_age_seconds=max_age_seconds, max_signal_age_seconds=max_signal_age_seconds,
            inner_execution=inner_execution)
        event = revision_panel.decisions[-1, "decision_time"].astimezone(UTC)
        expiry = min(event + timedelta(seconds=max_signal_age_seconds),
                     valuation.event_time + timedelta(seconds=max_age_seconds),
                     valuation.portfolio.event_time + timedelta(seconds=max_age_seconds))
        portfolio = valuation.portfolio
        strategy_sha256 = self.sha256
        signals = []
        with localcontext() as context:
            context.prec = 64
            for symbol, current in portfolio.positions:
                delta = Decimal(str(changes.get(symbol, 0)))
                identity = [strategy_sha256, portfolio.account_ref, symbol, event.isoformat()]
                signals.append({
                    "signalId": sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest(),
                    "symbol": symbol, "eventTime": event.isoformat(), "expiresAt": expiry.isoformat(),
                    "currentQuantity": str(current), "targetQuantity": str(current + delta),
                    "side": "BUY" if delta > 0 else "SELL" if delta < 0 else "HOLD",
                    "quantity": str(abs(delta)),
                })
        payload = {"contractVersion": "shadow-decision-v1", "mode": "shadow", "executionAuthorized": False,
                   "strategySha256": strategy_sha256, "engineCodeSha256": self.engine_code_sha256,
                   "inputSha256": revision_panel.fingerprint(), "snapshotAsOf": as_of.astimezone(UTC).isoformat(),
                   "warmup": None, "accountRef": portfolio.account_ref,
                   "accountSourceSha256": portfolio.source_sha256,
                   "accountEventTime": portfolio.event_time.astimezone(UTC).isoformat(), "decisions": signals}
        content = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return {**payload, "decisionSha256": sha256(content).hexdigest()}
