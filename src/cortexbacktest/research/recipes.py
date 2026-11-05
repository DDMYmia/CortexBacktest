"Causal residual-score and dated group-budget recipes; the existing engine owns execution."

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping

import numpy as np
import polars as pl

from .holding import _integer, _finite


def _dated(frame: pl.DataFrame, clocks=("date",)) -> pl.DataFrame:
    for name in clocks:
        dtype = frame.schema.get(name)
        if (
            name not in frame.columns
            or not isinstance(dtype, pl.Datetime)
            or not dtype.time_zone
            or frame[name].null_count()
        ):
            raise ValueError("explicit aware complete recipe clocks required")
        if frame.filter(pl.col(name).dt.epoch("ns") % 1000 != 0).height:
            raise ValueError("submicrosecond recipe clocks unsupported")
    return frame.with_columns(
        pl.col(name).dt.convert_time_zone("UTC").dt.cast_time_unit("us") for name in clocks
    )


def _matrix(frame: pl.DataFrame, *, metadata=()) -> tuple[pl.DataFrame, list[str]]:
    data = _dated(frame, ("date", *metadata)).sort("date")
    symbols = sorted(set(data.columns) - {"date", *metadata})
    if (
        not data.height
        or not symbols
        or data["date"].n_unique() != data.height
        or any(not s or s != s.strip() or s != s.upper() for s in symbols)
    ):
        raise ValueError("unique recipe dates and explicit uppercase columns required")
    if any(not data.schema[s].is_numeric() and data.schema[s] != pl.Null for s in symbols):
        raise ValueError("numeric recipe values required")
    return data, symbols


@dataclass(frozen=True, slots=True)
class RecipeResult:
    values: pl.DataFrame
    diagnostics: pl.DataFrame
    summary: dict

    def to_dict(self):
        def records(frame):
            return frame.with_columns(
                pl.col(n).dt.to_string("%+")
                for n, d in frame.schema.items()
                if isinstance(d, pl.Datetime)
            ).to_dicts()

        return {
            "summary": self.summary,
            "values": records(self.values),
            "diagnostics": records(self.diagnostics),
            "qualificationIssued": False,
            "executionAuthorized": False,
        }


@dataclass(frozen=True, slots=True)
class ResidualMomentumConfig:
    training_periods: int = 252
    scoring_periods: int = 126
    skip_periods: int = 21

    def __post_init__(self):
        _integer(self.training_periods, "training_periods", 3)
        _integer(self.scoring_periods, "scoring_periods", 2)
        _integer(self.skip_periods, "skip_periods")


def residual_momentum(
    returns: pl.DataFrame,
    factors: pl.DataFrame,
    *,
    config: ResidualMomentumConfig,
    return_basis: str,
    risk_free_column: str = "RF",
) -> RecipeResult:
    "Fit exclusively before each scoring window; only information available by the decision is used."
    if not isinstance(config, ResidualMomentumConfig) or return_basis not in {"raw", "excess"}:
        raise ValueError("explicit residual configuration and raw/excess return basis required")
    asset_clocks = ("known_at",) if "known_at" in returns.columns else ()
    assets, symbols = _matrix(returns, metadata=asset_clocks)
    if asset_clocks and assets.filter(pl.col("known_at") < pl.col("date")).height:
        raise ValueError("asset return availability cannot precede its period end")
    factors, columns = _matrix(factors, metadata=("known_at",))
    if not assets["date"].equals(factors["date"]):
        raise ValueError("asset and factor grids must match exactly, without inner-join losses")
    if factors.filter(pl.col("known_at") < pl.col("date")).height:
        raise ValueError("factor return availability cannot precede its period end")
    if return_basis == "raw" and risk_free_column not in columns:
        raise ValueError("raw returns require an explicit aligned risk-free series")
    if return_basis == "excess" and risk_free_column in columns:
        raise ValueError("already-excess returns must not silently include RF as a regressor")
    names = [n for n in columns if n != risk_free_column]
    if not names or config.training_periods <= len(names) + 1:
        raise ValueError("factor design requires observations beyond intercept and coefficients")
    X = factors.select(names).cast(pl.Float64).to_numpy()
    Y = assets.select(symbols).cast(pl.Float64).to_numpy()
    if return_basis == "raw":
        rf = factors[risk_free_column].cast(pl.Float64).to_numpy()
        if np.any(Y[np.isfinite(Y)] < -1) or np.any(rf[np.isfinite(rf)] < -1):
            raise ValueError("raw simple returns cannot be below -100%")
        Y = Y - rf[:, None]
    known = factors["known_at"].cast(pl.Int64).to_numpy()
    asset_known = assets["known_at" if asset_clocks else "date"].cast(pl.Int64).to_numpy()
    dates = assets["date"].to_list()
    clocks = assets["date"].cast(pl.Int64).to_numpy()
    output: dict[str, list[float | None]] = {s: [None] * len(dates) for s in symbols}
    audit = []
    for t, date in enumerate(dates):
        end = t + 1 - config.skip_periods
        score_start = end - config.scoring_periods
        start = score_start - config.training_periods
        for j, symbol in enumerate(symbols):
            row = {
                "date": date,
                "symbol": symbol,
                "reason": None,
                "fit_start": None,
                "fit_end": None,
                "score_start": None,
                "score_end": None,
                "alpha": None,
                "residual_sum": None,
                "residual_std": None,
                "score": None,
            }
            if start < 0:
                row["reason"] = "insufficient_complete_history"
            elif np.any(known[start:end] > clocks[t]):
                row["reason"] = "factor_not_available_by_decision"
            elif np.any(asset_known[start:end] > clocks[t]):
                row["reason"] = "asset_not_available_by_decision"
            elif not np.isfinite(X[start:end]).all() or not np.isfinite(Y[start:end, j]).all():
                row["reason"] = "missing_or_nonfinite_window"
            else:
                design = np.column_stack([np.ones(config.training_periods), X[start:score_start]])
                beta, _, rank, singular = np.linalg.lstsq(
                    design, Y[start:score_start, j], rcond=None
                )
                if rank != design.shape[1] or singular[0] / singular[-1] > 1e10:
                    row["reason"] = "rank_deficient_or_ill_conditioned_factors"
                else:
                    residual = (
                        Y[score_start:end, j]
                        - np.column_stack([np.ones(config.scoring_periods), X[score_start:end]])
                        @ beta
                    )
                    spread = float(np.std(residual, ddof=1))
                    total = float(np.sum(residual))
                    if not math.isfinite(spread) or spread <= 1e-12 or not math.isfinite(total):
                        row["reason"] = "undefined_residual_dispersion"
                    else:
                        score = total / spread
                        if not math.isfinite(score):
                            raise ValueError("nonfinite residual score")
                        output[symbol][t] = score
                        row.update(
                            fit_start=dates[start],
                            fit_end=dates[score_start - 1],
                            score_start=dates[score_start],
                            score_end=dates[end - 1],
                            alpha=float(beta[0]),
                            residual_sum=total,
                            residual_std=spread,
                            score=score,
                        )
            audit.append(row)
    result = pl.DataFrame(
        {
            "date": assets["date"],
            **{s: pl.Series(s, v, dtype=pl.Float64) for s, v in output.items()},
        }
    )
    diagnostics = pl.DataFrame(audit, infer_schema_length=None).with_columns(
        pl.col(n).cast(pl.Datetime("us", "UTC"))
        for n in ["fit_start", "fit_end", "score_start", "score_end"]
    )
    return RecipeResult(
        result,
        diagnostics,
        {
            "method": "frozen_ols_out_of_fit_residual_momentum",
            "output": "factor_scores",
            "factorNames": names,
            "returnBasis": return_basis,
            "assetAvailability": "explicit_known_at"
            if asset_clocks
            else "caller_asserted_period_end",
            "riskFreeColumn": risk_free_column if return_basis == "raw" else None,
            "trainingPeriods": config.training_periods,
            "scoringPeriods": config.scoring_periods,
            "skipPeriods": config.skip_periods,
            "interpretation": "provided_decision_grid_not_original_37_month_equity_factor_replication",
        },
    )


def group_budget_weights(
    scores: pl.DataFrame,
    membership: pl.DataFrame,
    budgets: Mapping[str, float],
    *,
    top_n: int = 1,
    long_short: bool = False,
    max_gross: float = 1.0,
) -> RecipeResult:
    "Allocate declared group budgets; unavailable groups leave cash, never redistribute implicitly."
    _integer(top_n, "top_n", 1)
    if (
        type(long_short) is not bool
        or not _finite(max_gross)
        or max_gross <= 0
        or not budgets
        or any(
            not isinstance(g, str) or not g.strip() or g != g.strip() or not _finite(v) or v < 0
            for g, v in budgets.items()
        )
        or math.fsum(budgets.values()) > max_gross * (1 + 1e-12)
    ):
        raise ValueError("explicit nonnegative group budgets within gross capital required")
    data, symbols = _matrix(scores)
    members = _dated(membership, ("date", "known_at"))
    if (
        set(members.columns) != {"date", "known_at", "symbol", "group"}
        or members.select("date", "symbol").unique().height != members.height
    ):
        raise ValueError("explicit unique dated membership required")
    for name in ["symbol", "group"]:
        if members.schema[name] != pl.String or members[name].null_count():
            raise ValueError("complete membership labels required")
    if (
        members.filter(pl.col("known_at") > pl.col("date")).height
        or set(members["symbol"]) - set(symbols)
        or set(members["group"]) - set(budgets)
        or set(members["date"]) - set(data["date"])
    ):
        raise ValueError("membership must match declared universe/budgets and be known by decision")
    lookup = {(r["date"], r["symbol"]): r["group"] for r in members.iter_rows(named=True)}
    rows = []
    audit = []
    for row in data.iter_rows(named=True):
        date = row["date"]
        values = {s: 0.0 for s in symbols}
        finite = {
            s: v for s, v in row.items() if s != "date" and v is not None and math.isfinite(v)
        }
        if any((date, s) not in lookup for s in finite):
            raise ValueError("finite candidate has no PIT group membership")
        for group, budget in sorted(budgets.items()):
            eligible = sorted([(v, s) for s, v in finite.items() if lookup[date, s] == group])
            reason = None
            if long_short:
                if len(eligible) < 2 * top_n or eligible[-1][0] == eligible[0][0]:
                    reason = "insufficient_group_dispersion_or_members"
                else:
                    for _, s in eligible[-top_n:]:
                        values[s] = budget / (2 * top_n)
                    for _, s in eligible[:top_n]:
                        values[s] = -budget / (2 * top_n)
            else:
                selected = [s for v, s in eligible if v > 0][-top_n:]
                if not selected:
                    reason = "no_positive_finite_candidates"
                else:
                    for s in selected:
                        values[s] = budget / len(selected)
            audit.append(
                {
                    "date": date,
                    "group": group,
                    "budget": float(budget),
                    "allocated_gross": math.fsum(
                        abs(values[s]) for s in symbols if lookup.get((date, s)) == group
                    ),
                    "reason": reason,
                }
            )
        rows.append({"date": date, **values})
    return RecipeResult(
        pl.DataFrame(rows),
        pl.DataFrame(audit),
        {
            "method": "dated_group_top_bottom_fixed_budget",
            "output": "target_weights",
            "longShort": long_short,
            "topN": top_n,
            "missingBudgetPolicy": "cash_no_cross_group_renormalization",
            "membershipEvidence": "caller_supplied_asof_membership_not_authenticated_historical_industry",
        },
    )
