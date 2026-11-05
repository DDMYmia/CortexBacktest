"Causal persistence and overlapping target-weight cohorts for the existing execution bridge."

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math

import polars as pl


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _clock(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("aware decision timestamp required")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class HoldingConfig:
    enter_threshold: float = 0.0
    exit_threshold: float = -0.01
    enter_periods: int = 1
    exit_periods: int = 1
    min_active_periods: int = 1
    cooldown_periods: int = 0
    holding_periods: int = 1
    formation_interval: int = 1
    max_gross: float = 1.0

    def __post_init__(self):
        if (
            not _finite(self.enter_threshold)
            or not _finite(self.exit_threshold)
            or self.enter_threshold <= self.exit_threshold
        ):
            raise ValueError("finite enter threshold must exceed exit threshold")
        for name in (
            "enter_periods",
            "exit_periods",
            "min_active_periods",
            "holding_periods",
            "formation_interval",
        ):
            _integer(getattr(self, name), name, 1)
        _integer(self.cooldown_periods, "cooldown_periods")
        if not _finite(self.max_gross) or self.max_gross <= 0:
            raise ValueError("positive finite gross budget required")

    @property
    def slots(self):
        return (self.holding_periods + self.formation_interval - 1) // self.formation_interval


@dataclass(slots=True)
class _Signal:
    active: bool = False
    enter_count: int = 0
    exit_count: int = 0
    active_age: int = 0
    cooldown_left: int = 0

    def advance(self, score, cfg):
        if not self.active:
            if self.cooldown_left:
                self.cooldown_left -= 1
                self.enter_count = 0
                return
            self.enter_count = (
                self.enter_count + 1 if score is not None and score >= cfg.enter_threshold else 0
            )
            if self.enter_count >= cfg.enter_periods:
                self.active = True
                self.active_age = 1
                self.enter_count = 0
        else:
            self.active_age += 1
            self.exit_count = (
                min(cfg.exit_periods, self.exit_count + 1)
                if score is not None and score <= cfg.exit_threshold
                else 0
            )
            if score is None or (
                self.active_age >= cfg.min_active_periods and self.exit_count >= cfg.exit_periods
            ):
                self.active = False
                self.active_age = self.exit_count = self.enter_count = 0
                self.cooldown_left = cfg.cooldown_periods


class PersistentWeightRecipe:
    "Filter supplied scores, expire funded cohorts, then emit one net target basket."

    __slots__ = ("_symbols", "_config", "_signals", "_cohorts", "_index", "_last")

    def __init__(self, symbols: Sequence[str], *, config: HoldingConfig):
        if (
            not symbols
            or isinstance(symbols, str)
            or any(
                not isinstance(s, str) or not s or s != s.strip() or s != s.upper() for s in symbols
            )
            or len(set(symbols)) != len(symbols)
            or "date" in symbols
        ):
            raise ValueError("unique explicit uppercase symbols required")
        if not isinstance(config, HoldingConfig):
            raise ValueError("HoldingConfig required")
        self._symbols = tuple(sorted(symbols))
        self._config = config
        self._signals = {s: _Signal() for s in self.symbols}
        self._cohorts: list[dict] = []
        self._index = -1
        self._last: datetime | None = None

    @property
    def symbols(self):
        return self._symbols

    @property
    def config(self):
        return self._config

    def _weights(self, values, *, budget):
        if (
            not isinstance(values, Mapping)
            or set(values) != set(self.symbols)
            or any(not _finite(x) for x in values.values())
        ):
            raise ValueError("complete finite weights for the pinned universe required")
        if math.fsum(abs(x) for x in values.values()) > budget * (1 + 1e-12):
            raise ValueError("cohort target exceeds gross capital budget")
        return {s: float(values[s]) for s in self.symbols}

    def update(
        self, timestamp: datetime, scores: Mapping[str, float | None], targets: Mapping[str, float]
    ) -> dict[str, float]:
        stamp = _clock(timestamp)
        if self._last is not None and stamp <= self._last:
            raise ValueError(
                "decisions must be strictly increasing; duplicate/replayed inputs are rejected"
            )
        if (
            not isinstance(scores, Mapping)
            or set(scores) != set(self.symbols)
            or any(x is not None and not _finite(x) for x in scores.values())
        ):
            raise ValueError("explicit finite-or-null scores for the pinned universe required")
        weights = self._weights(targets, budget=self.config.max_gross)
        # Validate the complete event before changing state.
        self._index += 1
        self._last = stamp
        for symbol, state in self._signals.items():
            state.advance(scores[symbol], self.config)
        cohorts = []
        for row in self._cohorts:
            if row["formed_at"] + self.config.holding_periods <= self._index:
                continue
            values = {
                s: row["weights"][s] if self._signals[s].active else 0.0 for s in self.symbols
            }
            if any(values.values()):
                cohorts.append({"formed_at": row["formed_at"], "weights": values})
        if self._index % self.config.formation_interval == 0:
            values = {
                s: weights[s] / self.config.slots if self._signals[s].active else 0.0
                for s in self.symbols
            }
            if any(values.values()):
                cohorts.append({"formed_at": self._index, "weights": values})
        self._cohorts = cohorts
        return {s: math.fsum(row["weights"][s] for row in cohorts) for s in self.symbols}

    def to_document(self) -> dict:
        return {
            "version": 1,
            "config": asdict(self.config),
            "symbols": list(self.symbols),
            "index": self._index,
            "last_timestamp": self._last.isoformat() if self._last else None,
            "signals": {s: asdict(state) for s, state in self._signals.items()},
            "cohorts": [
                {"formed_at": row["formed_at"], "weights": dict(row["weights"])}
                for row in self._cohorts
            ],
        }

    @classmethod
    def from_document(cls, document: dict, *, symbols: Sequence[str], config: HoldingConfig):
        if not isinstance(config, HoldingConfig):
            raise ValueError("HoldingConfig required")
        expected = {"version", "config", "symbols", "index", "last_timestamp", "signals", "cohorts"}
        if (
            not isinstance(document, dict)
            or set(document) != expected
            or type(document["version"]) is not int
            or document["version"] != 1
            or not isinstance(document["config"], dict)
            or document["config"] != asdict(config)
        ):
            raise ValueError("checkpoint version and pinned configuration must match")
        try:
            checked = HoldingConfig(**document["config"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid checkpoint configuration") from error
        instance = cls(symbols, config=checked)
        if document["symbols"] != list(instance.symbols):
            raise ValueError("checkpoint universe mismatch")
        index = document["index"]
        _integer(index, "checkpoint index", -1)
        last = document["last_timestamp"]
        if index == -1:
            if last is not None:
                raise ValueError("empty checkpoint must not carry a timestamp")
        else:
            try:
                last = _clock(datetime.fromisoformat(last))
            except (TypeError, ValueError) as error:
                raise ValueError("invalid checkpoint timestamp") from error
            if last.isoformat() != document["last_timestamp"]:
                raise ValueError("checkpoint timestamp must be canonical UTC")
        signals = document["signals"]
        if not isinstance(signals, dict) or set(signals) != set(instance.symbols):
            raise ValueError("checkpoint signal universe mismatch")
        restored = {}
        for symbol, row in signals.items():
            if (
                not isinstance(row, dict)
                or set(row) != set(asdict(_Signal()))
                or type(row["active"]) is not bool
            ):
                raise ValueError("invalid checkpoint signal state")
            for name in ("enter_count", "exit_count", "active_age", "cooldown_left"):
                _integer(row[name], name)
            state = _Signal(**row)
            if (
                state.enter_count >= config.enter_periods
                or state.exit_count > config.exit_periods
                or state.active_age > index + 1
                or state.enter_count > index + 1
                or state.cooldown_left > config.cooldown_periods
                or (state.active and state.exit_count > state.active_age - 1)
                or (
                    state.active
                    and state.exit_count >= config.exit_periods
                    and state.active_age >= config.min_active_periods
                )
                or (
                    state.active
                    and (state.active_age < 1 or state.enter_count or state.cooldown_left)
                )
                or (not state.active and (state.active_age or state.exit_count))
                or (state.cooldown_left and state.enter_count)
                or (index == -1 and row != asdict(_Signal()))
            ):
                raise ValueError("inconsistent checkpoint signal counters")
            restored[symbol] = state
        cohorts = document["cohorts"]
        if not isinstance(cohorts, list) or len(cohorts) > config.slots:
            raise ValueError("invalid checkpoint cohort list")
        restored_cohorts, previous = [], -1
        for row in cohorts:
            if not isinstance(row, dict) or set(row) != {"formed_at", "weights"}:
                raise ValueError("invalid checkpoint cohort")
            formed = row["formed_at"]
            _integer(formed, "cohort formation")
            if (
                formed <= previous
                or formed > index
                or formed % config.formation_interval
                or formed + config.holding_periods <= index
            ):
                raise ValueError("invalid or expired checkpoint cohort")
            values = instance._weights(row["weights"], budget=config.max_gross / config.slots)
            if not any(values.values()) or any(
                values[s] and not restored[s].active for s in instance.symbols
            ):
                raise ValueError("inactive symbol cannot retain a checkpoint cohort")
            restored_cohorts.append({"formed_at": formed, "weights": values})
            previous = formed
        instance._index, instance._last = index, last
        instance._signals, instance._cohorts = restored, restored_cohorts
        return instance


def apply_holding_recipe(
    targets: pl.DataFrame,
    scores: pl.DataFrame,
    *,
    config: HoldingConfig,
    checkpoint: dict | None = None,
) -> tuple[pl.DataFrame, dict]:
    "Transform exactly aligned aware date-by-symbol matrices; null scores explicitly break persistence."
    symbols = sorted(set(targets.columns) - {"date"})
    if not symbols or set(scores.columns) != set(targets.columns) or "date" not in targets.columns:
        raise ValueError("target and score matrices require the same explicit date/symbol columns")
    frames = []
    for frame in (targets, scores):
        dtype = frame.schema["date"]
        if (
            not frame.height
            or not isinstance(dtype, pl.Datetime)
            or not dtype.time_zone
            or frame["date"].null_count()
            or frame["date"].n_unique() != frame.height
        ):
            raise ValueError("nonempty unique aware decision clocks required")
        if frame.filter(pl.col("date").dt.epoch("ns") % 1000 != 0).height:
            raise ValueError(
                "submicrosecond decision clocks are unsupported; do not silently round"
            )
        for s in symbols:
            if frame.schema[s] != pl.Null and not frame.schema[s].is_numeric():
                raise ValueError("numeric weight/score matrices required")
        frames.append(
            frame.with_columns(
                pl.col("date").dt.convert_time_zone("UTC").dt.cast_time_unit("us")
            ).sort("date")
        )
    left, right = frames
    if not left["date"].equals(right["date"]):
        raise ValueError(
            "target and score decisions must match exactly; no dropped or inferred sessions"
        )
    recipe = (
        PersistentWeightRecipe(symbols, config=config)
        if checkpoint is None
        else PersistentWeightRecipe.from_document(checkpoint, symbols=symbols, config=config)
    )
    rows = []
    for target, signal in zip(left.iter_rows(named=True), right.iter_rows(named=True), strict=True):
        stamp = target.pop("date")
        signal.pop("date")
        rows.append({"date": stamp, **recipe.update(stamp, signal, target)})
    return pl.DataFrame(rows), recipe.to_document()
