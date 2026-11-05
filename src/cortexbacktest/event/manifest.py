"""Deterministic provenance records for event-backtest runs."""

from __future__ import annotations

import hashlib
import json
import inspect
from functools import lru_cache
from pathlib import Path
import platform
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import date, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterable, Mapping

import orjson

from ..contracts import Bar

_MANIFEST_SCHEMA_VERSION = 5


@dataclass(frozen=True, slots=True)
class RunManifest:
    """JSON-safe run provenance; `run_id` changes whenever inputs or settings change."""

    schema_version: int
    run_id: str
    data_sha256: str
    bar_count: int
    symbols: tuple[str, ...]
    start: str | None
    end: str | None
    engine: dict[str, Any]
    broker: dict[str, Any]
    strategy: str
    metadata: dict[str, Any]
    package_version: str
    strategy_config: dict[str, Any] | None = None
    engine_code_sha256: str | None = None
    python_version: str | None = None
    platform: str | None = None

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["python_version"] = self.python_version or platform.python_version()
        record["platform"] = self.platform or platform.platform()
        return record


@dataclass(frozen=True, slots=True)
class BarDataFingerprint:
    data_sha256: str
    bar_count: int
    symbols: tuple[str, ...]
    start: datetime | None
    end: datetime | None


class _BarFingerprintAccumulator:
    """Incremental form of the schema-5 fingerprint; never retains Bars."""

    def __init__(self) -> None:
        self.hasher = hashlib.sha256()
        self.symbols: set[str] = set()
        self.count = 0
        self.start: datetime | None = None
        self.end: datetime | None = None

    def update(self, bars: Iterable[Bar]) -> None:
        update = self.hasher.update
        for bar in bars:
            update(orjson.dumps((
                bar.symbol, bar.timestamp, bar.open, bar.high, bar.low, bar.close,
                bar.volume, bar.adj_close, bar.vwap, bar.open_interest, bar.halted,
                bar.limit_up, bar.limit_down,
            ), option=orjson.OPT_SERIALIZE_NUMPY))
            update(b"\n")
            self.symbols.add(bar.symbol)
            self.count += 1
            if self.start is None or bar.timestamp < self.start:
                self.start = bar.timestamp
            if self.end is None or bar.timestamp > self.end:
                self.end = bar.timestamp

    def finish(self) -> BarDataFingerprint:
        return BarDataFingerprint(self.hasher.hexdigest(), self.count,
                                  tuple(sorted(self.symbols)), self.start, self.end)


def fingerprint_bars(bars: Iterable[Bar]) -> BarDataFingerprint:
    """Hash a bar universe without retaining its observations."""
    accumulator = _BarFingerprintAccumulator()
    accumulator.update(bars)
    return accumulator.finish()


def _manifest_with_fingerprint(template: RunManifest, data: BarDataFingerprint) -> RunManifest:
    # The template captures broker/strategy configuration BEFORE simulation.
    # Rebuilding from a mutated broker at the end would hash final cash instead.
    payload = {name: getattr(template, name) for name in (
        "schema_version", "engine", "broker", "strategy", "strategy_config",
        "engine_code_sha256", "metadata", "package_version",
    )}
    payload["data_sha256"] = data.data_sha256
    return replace(template, run_id=run_manifest_digest(payload),
                   data_sha256=data.data_sha256, bar_count=data.bar_count, symbols=data.symbols,
                   start=data.start.isoformat() if data.start is not None else None,
                   end=data.end.isoformat() if data.end is not None else None)


def build_run_manifest(
    bars: Iterable[Bar],
    *,
    broker,
    strategy,
    trade_on_close: bool,
    snapshot_frequency: str,
    result_retention: str,
    metadata: Mapping[str, Any] | None = None,
    data_fingerprint: BarDataFingerprint | None = None,
    strategy_config: dict[str, Any] | None = None,
    opening_times: Mapping[datetime, datetime] | None = None,
) -> RunManifest:
    """Build a stable manifest without retaining a duplicate copy of market data."""
    fingerprint = data_fingerprint or fingerprint_bars(bars)
    engine = {
        "trade_on_close": trade_on_close,
        "snapshot_frequency": snapshot_frequency,
        "result_retention": result_retention,
    }
    if opening_times is not None:
        engine["opening_times"] = [[close.isoformat(), opening.isoformat()]
                                   for close, opening in sorted(opening_times.items())]
    broker_settings = {
        "initial_cash": broker.state.cash,
        "futures_symbols": sorted(broker.state.futures_symbols),
        "maintenance_margin": broker.maintenance_margin,
        "dividend_withholding_tax": broker.dividend_withholding_tax,
        "intrabar_policy": getattr(broker, "intrabar_policy", "submission"),
        "stop_limit_same_bar": getattr(broker, "stop_limit_same_bar", True),
        "fill_price": broker.fill_price.value,
        "allow_short": broker.allow_short,
        "max_volume_share": broker.max_volume_share,
        "commission_model": _model_record(broker.commission_model),
        "slippage_model": _model_record(broker.slippage_model),
        "matching_model": _model_record(broker.execution_handler.matching_model),
        "settlement_model": _model_record(broker.ledger.settlement_model),
        # Financing and corporate actions move equity, so they belong in the
        # run identity: without them two runs with different borrow costs or
        # dividend schedules would share a run_id.
        "cash_interest_model": _model_record(broker.cash_interest_model),
        "short_borrow_fee_model": _model_record(broker.short_borrow_fee_model),
        "corporate_actions": _corporate_actions_record(broker.corporate_actions),
        "contract_multipliers": dict(sorted(broker.state.contract_multipliers.items())),
        "risk_rules": [_model_record(rule) for rule in broker.risk_engine.rules],
    }
    user_metadata = _json_value(dict(metadata or {}))
    payload = {
        # Version 5 adds capital, futures/margin rules, strategy parameters and source identity.
        # Version 4 adds financing models and corporate actions to the broker
        # record; both change equity, so they must change the run identity.
        # Version 3 hashes each bar as a fixed positional orjson array with
        # native datetime encoding. Bumping the schema keeps persisted older
        # hashes semantically unambiguous.
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "data_sha256": fingerprint.data_sha256,
        "engine": engine,
        "broker": broker_settings,
        "strategy": _strategy_name(strategy),
        "strategy_config": strategy_config or strategy_configuration(strategy),
        "engine_code_sha256": engine_code_digest(),
        "metadata": user_metadata,
        "package_version": _package_version(),
    }
    run_id = run_manifest_digest(payload)
    return RunManifest(
        schema_version=_MANIFEST_SCHEMA_VERSION,
        run_id=run_id,
        data_sha256=payload["data_sha256"],
        bar_count=fingerprint.bar_count,
        symbols=fingerprint.symbols,
        start=fingerprint.start.isoformat() if fingerprint.start is not None else None,
        end=fingerprint.end.isoformat() if fingerprint.end is not None else None,
        engine=engine,
        broker=broker_settings,
        strategy=payload["strategy"],
        metadata=user_metadata,
        package_version=payload["package_version"],
        strategy_config=payload["strategy_config"],
        engine_code_sha256=payload["engine_code_sha256"],
    )


def _corporate_actions_record(schedule) -> dict[str, Any] | None:
    """Digest the schedule rather than embed it; it can hold thousands of rows."""
    if not schedule:
        return None
    hasher = hashlib.sha256()
    for action in schedule:
        hasher.update(orjson.dumps(asdict(action)))
        hasher.update(b"\n")
    return {"count": len(schedule), "sha256": hasher.hexdigest()}


def _is_dataclass_instance(value: Any) -> bool:
    # is_dataclass() is also true for the class itself, and asdict() raises on a class.
    return is_dataclass(value) and not isinstance(value, type)


def _model_record(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    values = asdict(value) if _is_dataclass_instance(value) else getattr(value, "__dict__", {})
    return {"type": _strategy_name(value), "settings": _json_value(values)}


def _strategy_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _json_value(value: Any) -> Any:
    if _is_dataclass_instance(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "value"):
        return _json_value(value.value)
    try:
        json.dumps(value)
    except TypeError as exc:
        raise TypeError(f"run manifest contains a non-JSON value: {type(value).__name__}") from exc
    return value


def run_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Recompute the schema-5 run identity without changing its hash contract."""
    if manifest["schema_version"] != 5:
        raise ValueError("Unsupported run manifest version")
    fields = ("schema_version", "data_sha256", "engine", "broker", "strategy",
              "strategy_config", "engine_code_sha256", "metadata", "package_version")
    return hashlib.sha256(_canonical_json_bytes({key: manifest[key] for key in fields})).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    # OPT_SERIALIZE_NUMPY: bar feeds built from pandas/NumPy sources carry
    # numpy scalar floats, which are valid float subclasses for the engine.
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY)


def _package_version() -> str:
    try:
        return version("cortexbacktest")
    except PackageNotFoundError:
        return "unknown"


@lru_cache(maxsize=1)
def engine_code_digest() -> str:
    """Content identity of installed engine Python sources, once per process."""
    root = Path(__file__).resolve().parents[1]
    hasher = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        hasher.update(path.relative_to(root).as_posix().encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def strategy_configuration(strategy) -> dict[str, Any]:
    """Capture declared constructor attributes before initialize mutates state.

    Missing attributes are explicit; use StrategySpec/metadata for factories that
    do not retain their constructor arguments. Runtime account state is excluded.
    """
    parameters = {}
    missing = []
    try:
        signature = inspect.signature(type(strategy).__init__)
    except (TypeError, ValueError):
        signature = inspect.Signature()
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if hasattr(strategy, name):
            value = getattr(strategy, name)
            try:
                parameters[name] = _json_value(value)
            except TypeError:
                missing.append(name)
        else:
            missing.append(name)
    try:
        source = inspect.getsource(type(strategy))
    except (OSError, TypeError):
        source = None
    return {"parameters": parameters, "unrecorded_parameters": missing,
            "source_sha256": None if source is None else hashlib.sha256(source.encode()).hexdigest()}
