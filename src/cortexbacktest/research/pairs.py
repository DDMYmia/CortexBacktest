"Frozen-formation pair selection and existing persistent-weight execution targets."

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import combinations
import hashlib
import json
import math
import warnings

import numpy as np
import polars as pl

from .holding import HoldingConfig, PersistentWeightRecipe, _clock, _integer, _finite
from .recipes import _matrix


@dataclass(frozen=True, slots=True)
class PairConfig:
    formation_periods: int = 120
    max_pairs: int = 5
    enter_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    confirmation_periods: int = 1
    cooldown_periods: int = 2
    max_holding_periods: int = 20
    max_gross: float = 1.0

    def __post_init__(self):
        _integer(self.formation_periods, "formation_periods", 20)
        for name in ["max_pairs", "confirmation_periods", "max_holding_periods"]:
            _integer(getattr(self, name), name, 1)
        _integer(self.cooldown_periods, "cooldown_periods")
        if (
            any(not _finite(x) for x in [self.enter_z, self.exit_z, self.stop_z, self.max_gross])
            or not 0 <= self.exit_z < self.enter_z < self.stop_z
            or self.max_gross <= 0
        ):
            raise ValueError(
                "finite pair thresholds must satisfy 0<=exit<enter<stop with positive capital"
            )


@dataclass(frozen=True, slots=True)
class PairModel:
    left: str
    right: str
    intercept: float
    beta: float
    center: float
    scale: float
    distance: float
    formation_start: datetime
    formation_end: datetime
    known_at: datetime
    data_sha256: str

    def __post_init__(self):
        if (
            not self.left
            or self.left >= self.right
            or any(s != s.strip() or s != s.upper() for s in [self.left, self.right])
            or any(
                not _finite(x)
                for x in [self.intercept, self.beta, self.center, self.scale, self.distance]
            )
            or self.beta <= 0
            or self.scale <= 1e-12
            or self.distance < 0
            or not _clock(self.formation_start)
            <= _clock(self.formation_end)
            <= _clock(self.known_at)
            or len(self.data_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.data_sha256)
        ):
            raise ValueError("invalid frozen pair model")

    def to_document(self):
        return {
            **asdict(self),
            **{
                k: _clock(getattr(self, k)).isoformat()
                for k in ["formation_start", "formation_end", "known_at"]
            },
        }


@dataclass(frozen=True, slots=True)
class PairFormation:
    models: tuple[PairModel, ...]
    candidates: pl.DataFrame
    summary: dict


def fit_pairs(
    prices: pl.DataFrame,
    *,
    config: PairConfig,
    known_at: datetime,
    currency: str,
    cointegration: bool = False,
) -> PairFormation:
    "Use only the explicitly supplied formation block; distance is not a cointegration test."
    if (
        not isinstance(config, PairConfig)
        or type(cointegration) is not bool
        or not isinstance(currency, str)
        or not currency.strip()
        or currency != currency.strip()
    ):
        raise ValueError("explicit pair config, diagnostic option and common currency required")
    price_clocks = ("known_at",) if "known_at" in prices.columns else ()
    data, symbols = _matrix(prices, metadata=price_clocks)
    known = _clock(known_at)
    if data.height != config.formation_periods or len(symbols) < 2 or data["date"][-1] > known:
        raise ValueError(
            "pair formation must have exact history available by the frozen model time"
        )
    if (
        price_clocks
        and data.filter((pl.col("known_at") < pl.col("date")) | (pl.col("known_at") > known)).height
    ):
        raise ValueError("formation prices must be received by frozen model availability")
    values = data.select(symbols).cast(pl.Float64).to_numpy()
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("complete finite positive formation prices required")
    payload = {
        "dates": [x.isoformat() for x in data["date"]],
        "symbols": symbols,
        "prices": values.tolist(),
        "currency": currency,
    }
    if price_clocks:
        payload["priceKnownAt"] = [x.isoformat() for x in data["known_at"]]
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    fitted = []
    audit = []
    for a, b in combinations(range(len(symbols)), 2):
        left, right = symbols[a], symbols[b]
        y = np.log(values[:, a])
        x = np.log(values[:, b])
        design = np.column_stack([np.ones(data.height), x])
        coef, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
        row = {
            "left": left,
            "right": right,
            "distance": None,
            "beta": None,
            "reason": None,
            "ar1": None,
            "half_life_periods": None,
            "half_life_reason": None,
            "cointegration_status": "formation_rejected" if cointegration else "not_requested",
            "cointegration_statistic": None,
            "cointegration_pvalue": None,
        }
        if rank != 2 or singular[0] / singular[-1] > 1e10:
            row["reason"] = "rank_deficient_formation"
        elif coef[1] <= 0:
            row["reason"] = "nonpositive_hedge_beta"
        else:
            spread = y - design @ coef
            center = float(spread.mean())
            scale = float(np.std(spread, ddof=1))
            distance = float(
                np.sum((values[:, a] / values[0, a] - values[:, b] / values[0, b]) ** 2)
            )
            if not math.isfinite(scale) or scale <= 1e-12 or not math.isfinite(distance):
                row["reason"] = "degenerate_or_nonfinite_formation"
            else:
                ar_design = np.column_stack([np.ones(data.height - 1), spread[:-1]])
                ar, _, ar_rank, _ = np.linalg.lstsq(ar_design, spread[1:], rcond=None)
                phi = float(ar[1])
                row["ar1"] = phi if ar_rank == 2 and math.isfinite(phi) else None
                if ar_rank == 2 and 0 < phi < 1:
                    row["half_life_periods"] = math.log(0.5) / math.log(phi)
                else:
                    row["half_life_reason"] = "no_supported_decay_in_zero_one"
                if cointegration:
                    try:
                        from statsmodels.tsa.stattools import coint
                    except ModuleNotFoundError as error:
                        if error.name not in {"statsmodels", "statsmodels.tsa", "statsmodels.tsa.stattools"}:
                            raise
                        row["cointegration_status"] = "dependency_unavailable"
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            statistic, pvalue, _ = coint(y, x, trend="c", maxlag=1, autolag=None)
                        if math.isfinite(statistic) and math.isfinite(pvalue) and 0 <= pvalue <= 1:
                            row.update(
                                cointegration_status="available_diagnostic_only",
                                cointegration_statistic=float(statistic),
                                cointegration_pvalue=float(pvalue),
                            )
                        else:
                            row["cointegration_status"] = "undefined_degenerate_statistic"
                row.update(distance=distance, beta=float(coef[1]))
                fitted.append(
                    PairModel(
                        left,
                        right,
                        float(coef[0]),
                        float(coef[1]),
                        center,
                        scale,
                        distance,
                        data["date"][0],
                        data["date"][-1],
                        known,
                        digest,
                    )
                )
        audit.append(row)
    selected = tuple(
        sorted(fitted, key=lambda m: (m.distance, m.left, m.right))[: config.max_pairs]
    )
    return PairFormation(
        selected,
        pl.DataFrame(audit, infer_schema_length=None),
        {
            "method": "normalized_price_distance_selection_frozen_log_ols_hedge",
            "currency": currency,
            "priceAvailability": "explicit_known_at"
            if price_clocks
            else "caller_asserted_by_formation_time",
            "formationDataSha256": digest,
            "formationPeriods": config.formation_periods,
            "candidatePairs": len(audit),
            "selectedPairs": len(selected),
            "budgetSlots": config.max_pairs,
            "cointegrationPolicy": "Engle_Granger_constant_maxlag1_no_autolag_diagnostic_not_selection_or_qualification",
            "halfLifePolicy": "frozen_spread_AR1_log_decay_only_for_0_phi_1",
            "evidence": "caller_supplied_same_currency_prices_not_authenticated_total_return_or_PIT",
        },
    )


class PairRecipe:
    "Frozen pairs emit one net asset basket through PersistentWeightRecipe."

    def __init__(self, models: tuple[PairModel, ...], *, config: PairConfig):
        if (
            not isinstance(config, PairConfig)
            or not models
            or len(models) > config.max_pairs
            or any(not isinstance(m, PairModel) for m in models)
            or len({(m.left, m.right) for m in models}) != len(models)
            or len(
                {(m.formation_start, m.formation_end, m.known_at, m.data_sha256) for m in models}
            )
            != 1
        ):
            raise ValueError("consistent frozen pair models within configured slots required")
        self._models = tuple(models)
        self._config = config
        self._ids = tuple("PAIR" + str(i) for i in range(len(models)))
        self._symbols = tuple(sorted({s for m in models for s in [m.left, m.right]}))
        self._gate_config = HoldingConfig(
            enter_threshold=config.enter_z,
            exit_threshold=config.exit_z,
            enter_periods=config.confirmation_periods,
            cooldown_periods=config.cooldown_periods,
            max_gross=config.max_gross,
        )
        self._gate = PersistentWeightRecipe(self._ids, config=self._gate_config)

    @property
    def symbols(self):
        return self._symbols

    def update(self, timestamp: datetime, prices: dict[str, float | None]) -> dict[str, float]:
        stamp = _clock(timestamp)
        if stamp <= max(m.known_at for m in self._models):
            raise ValueError("pair evaluation must strictly follow frozen formation availability")
        if set(prices) != set(self.symbols) or any(
            v is not None and (not _finite(v) or v <= 0) for v in prices.values()
        ):
            raise ValueError("explicit complete finite-positive-or-null pair prices required")
        previous = self._gate.to_document()
        scores = {}
        targets = {}
        old = {s: math.fsum(row["weights"][s] for row in previous["cohorts"]) for s in self._ids}
        for key, m in zip(self._ids, self._models, strict=True):
            state = previous["signals"][key]
            a, b = prices[m.left], prices[m.right]
            z = (
                None
                if a is None or b is None
                else (math.log(a) - m.intercept - m.beta * math.log(b) - m.center) / m.scale
            )
            if z is not None and not math.isfinite(z):
                raise ValueError("nonfinite pair divergence")
            forced = (
                z is None
                or abs(z) >= self._config.stop_z
                or (
                    state["active"]
                    and (
                        state["active_age"] >= self._config.max_holding_periods or old[key] * z >= 0
                    )
                )
            )
            scores[key] = abs(z) if not forced and z is not None else None
            direction = (
                (1.0 if old[key] > 0 else -1.0)
                if state["active"]
                else (-1.0 if z is not None and z > 0 else 1.0)
            )
            targets[key] = (
                0.0 if forced else direction * self._config.max_gross / self._config.max_pairs
            )
        net_pairs = self._gate.update(stamp, scores, targets)
        weights = {s: 0.0 for s in self.symbols}
        for key, m in zip(self._ids, self._models, strict=True):
            budget = net_pairs[key] / (1 + m.beta)
            weights[m.left] += budget
            weights[m.right] -= m.beta * budget
        return weights

    def to_document(self):
        return {
            "version": 1,
            "models": [m.to_document() for m in self._models],
            "config": asdict(self._config),
            "gate": self._gate.to_document(),
        }

    @classmethod
    def from_document(cls, document, *, models, config):
        expected = cls(models, config=config)
        if (
            not isinstance(document, dict)
            or set(document) != {"version", "models", "config", "gate"}
            or type(document["version"]) is not int
            or document["version"] != 1
            or document["models"] != [m.to_document() for m in models]
            or document["config"] != asdict(config)
        ):
            raise ValueError("pair checkpoint model/config binding mismatch")
        expected._gate = PersistentWeightRecipe.from_document(
            document["gate"], symbols=expected._ids, config=expected._gate_config
        )
        return expected
