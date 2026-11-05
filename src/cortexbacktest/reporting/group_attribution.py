"Brinson-Fachler group effects and Carino linking, with observed-return reconciliation."

from __future__ import annotations

from dataclasses import dataclass
import math

import polars as pl

_EFFECTS = ("allocation", "selection", "interaction")
_ADJUSTMENTS = ("trading", "costs", "cash")
_CONTRIBUTIONS = tuple(f"{side}_{name}" for side in ("portfolio", "benchmark") for name in _ADJUSTMENTS)
_RETURNS = ("portfolio_return", "benchmark_return")
_WEIGHTS = ("portfolio_weight", "benchmark_weight")
_CLOCKS = ("timestamp", "period_start")
_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class GroupAttribution:
    summary: dict
    periods: pl.DataFrame
    groups: pl.DataFrame
    linked_groups: pl.DataFrame

    def to_dict(self) -> dict:
        def records(frame):
            clocks = [
                name for name, dtype in frame.schema.items() if isinstance(dtype, pl.Datetime)
            ]
            return frame.with_columns(pl.col(name).dt.to_string("%+") for name in clocks).to_dicts()

        return {
            "summary": self.summary,
            "periods": records(self.periods),
            "groups": records(self.groups),
            "linkedGroups": records(self.linked_groups),
            "qualificationIssued": False,
            "executionAuthorized": False,
        }


def _frame(frame: pl.DataFrame, columns: set[str], clocks: tuple[str, ...]) -> pl.DataFrame:
    if set(frame.columns) != columns or not frame.height:
        raise ValueError("nonempty explicit attribution columns required")
    for name in clocks:
        dtype = frame.schema[name]
        if not isinstance(dtype, pl.Datetime) or not dtype.time_zone or frame[name].null_count():
            raise ValueError("aware complete attribution clocks required")
    for name in columns & {*_RETURNS, *_WEIGHTS, *_CONTRIBUTIONS}:
        if (
            not frame.schema[name].is_numeric()
            or frame[name].null_count()
            or not frame[name].is_finite().all()
        ):
            raise ValueError("finite complete attribution weights and returns required")
    return frame.with_columns(
        pl.col(name).dt.convert_time_zone("UTC").dt.cast_time_unit("ns") for name in clocks
    )


def _reconcile(actual: float, expected: float, message: str) -> float:
    residual = expected - actual
    if not math.isfinite(residual) or not math.isclose(
        actual, expected, rel_tol=1e-9, abs_tol=_TOLERANCE
    ):
        raise ValueError(message)
    return residual


def _carino_factor(portfolio: float, benchmark: float) -> float:
    low, high = sorted((portfolio, benchmark))
    if not math.isfinite(low) or not math.isfinite(high) or low <= -1:
        raise ValueError("Carino needs finite period and compounded returns above -100%")
    delta = high - low
    if not delta:
        return 1 / (1 + low)
    ratio = delta / (1 + low)
    # log1p avoids cancellation for almost identical returns, including near -100%.
    value = (
        math.log1p(ratio) if math.isfinite(ratio) else math.log1p(high) - math.log1p(low)
    ) / delta
    if not math.isfinite(value) or value <= 0:
        raise ValueError("unrepresentable Carino factor")
    return value


def group_attribution(
    groups: pl.DataFrame,
    observed_returns: pl.DataFrame,
    *,
    currency: str,
    benchmark_id: str,
    adjustments: pl.DataFrame | None = None,
) -> GroupAttribution:
    "Attribute complete, fully invested group returns; do not infer missing groups, cash or costs."
    if (
        not isinstance(currency, str)
        or not currency.strip()
        or currency != currency.strip()
        or not isinstance(benchmark_id, str)
        or not benchmark_id.strip()
        or benchmark_id != benchmark_id.strip()
    ):
        raise ValueError("explicit common currency and benchmark identity required")
    observed = _frame(observed_returns, {*_CLOCKS, *_RETURNS, "currency"}, _CLOCKS)
    data = _frame(
        groups,
        {*_CLOCKS, *_RETURNS, *_WEIGHTS, "currency", "group", "weights_at", "classification_at"},
        (*_CLOCKS, "weights_at", "classification_at"),
    )
    effects_names = _EFFECTS
    if adjustments is not None:
        extra = _frame(adjustments, {*_CLOCKS, "currency", "group", *_CONTRIBUTIONS}, _CLOCKS)
        keys = [*_CLOCKS, "currency", "group"]
        if (extra.select(keys).unique().height != extra.height
                or data.select(keys).join(extra.select(keys), on=keys, how="anti").height
                or extra.select(keys).join(data.select(keys), on=keys, how="anti").height):
            raise ValueError("adjustments must cover exactly every attribution period/group")
        data = data.join(extra, on=keys, how="left")
        effects_names += _ADJUSTMENTS
    for frame in (observed, data):
        if (
            frame.schema["currency"] != pl.String
            or frame["currency"].null_count()
            or set(frame["currency"]) != {currency}
        ):
            raise ValueError("all group and observed returns must share the declared currency")
    if (
        data.schema["group"] != pl.String
        or data["group"].null_count()
        or any(not label.strip() or label != label.strip() for label in data["group"])
    ):
        raise ValueError("explicit nonblank group labels required")
    if (
        observed["timestamp"].n_unique() != observed.height
        or data.select("timestamp", "group").unique().height != data.height
    ):
        raise ValueError("duplicate attribution period or group")
    observed = observed.sort("timestamp")
    if observed.filter(pl.col("period_start") >= pl.col("timestamp")).height:
        raise ValueError("positive attribution intervals required")
    if (
        observed.height > 1
        and observed.with_columns(pl.col("timestamp").shift(1).alias("prior_end"))
        .slice(1)
        .filter(pl.col("period_start") != pl.col("prior_end"))
        .height
    ):
        raise ValueError("attribution intervals must be contiguous and nonoverlapping")
    keys = observed.select(*_CLOCKS)
    group_keys = data.select(*_CLOCKS).unique()
    if (
        keys.join(group_keys, on=list(_CLOCKS), how="anti").height
        or group_keys.join(keys, on=list(_CLOCKS), how="anti").height
    ):
        raise ValueError("groups must cover exactly the observed return intervals")
    if data.filter(
        (pl.col("weights_at") > pl.col("period_start"))
        | (pl.col("classification_at") > pl.col("period_start"))
    ).height:
        raise ValueError("weights and classifications must be known by period start")
    # Keep integer nanosecond clocks across Python arithmetic; datetime would truncate them.
    observed = observed.with_columns(pl.col(name).cast(pl.Int64) for name in _CLOCKS)
    data = data.with_columns(
        pl.col(name).cast(pl.Int64) for name in (*_CLOCKS, "weights_at", "classification_at")
    )
    totals = {row["timestamp"]: row for row in observed.iter_rows(named=True)}
    periods, effects = [], []
    for piece in data.sort("timestamp", "group").partition_by("timestamp", maintain_order=True):
        stamp = piece["timestamp"][0]
        total = totals[stamp]
        for name in _WEIGHTS:
            _reconcile(
                math.fsum(piece[name]),
                1.0,
                "include all funded weights and explicit cash; weights must sum to one",
            )
        rp = math.fsum(
            w * r for w, r in zip(piece["portfolio_weight"], piece["portfolio_return"], strict=True)
        )
        rb = math.fsum(
            w * r for w, r in zip(piece["benchmark_weight"], piece["benchmark_return"], strict=True)
        )
        bop_rp, bop_rb = rp, rb
        if adjustments is not None:
            rp += math.fsum(piece[f"portfolio_{name}"].sum() for name in _ADJUSTMENTS)
            rb += math.fsum(piece[f"benchmark_{name}"].sum() for name in _ADJUSTMENTS)
        rp_residual = _reconcile(
            rp,
            total["portfolio_return"],
            "group portfolio return does not reconcile; include costs/cash",
        )
        rb_residual = _reconcile(
            rb, total["benchmark_return"], "group benchmark return does not reconcile"
        )
        rows = []
        for row in piece.iter_rows(named=True):
            active_weight = row["portfolio_weight"] - row["benchmark_weight"]
            spread = row["portfolio_return"] - row["benchmark_return"]
            row.update(
                allocation=active_weight * (row["benchmark_return"] - bop_rb),
                selection=row["benchmark_weight"] * spread,
                interaction=active_weight * spread,
            )
            if adjustments is not None:
                row.update({name: row[f"portfolio_{name}"] - row[f"benchmark_{name}"] for name in _ADJUSTMENTS})
            row["total_effect"] = math.fsum(row[name] for name in effects_names)
            rows.append(row)
        aggregate = {name: math.fsum(row[name] for row in rows) for name in effects_names}
        residual = _reconcile(
            math.fsum(aggregate.values()),
            rp - rb,
            "single-period attribution failed reconciliation",
        )
        periods.append(
            {
                "timestamp": stamp,
                "period_start": total["period_start"],
                "portfolio_return": rp,
                "benchmark_return": rb,
                "active_return": rp - rb,
                **({"bop_portfolio_return": bop_rp, "bop_benchmark_return": bop_rb} if adjustments is not None else {}),
                **aggregate,
                "reconciliation_residual": residual,
                "portfolio_observed_residual": rp_residual,
                "benchmark_observed_residual": rb_residual,
            }
        )
        effects.extend(rows)
    try:
        compounded = {
            name: math.expm1(math.fsum(math.log1p(row[name]) for row in periods))
            for name in _RETURNS
        }
    except (ValueError, OverflowError) as error:
        raise ValueError("Carino needs representable positive compounded wealth") from error
    factor = _carino_factor(compounded["portfolio_return"], compounded["benchmark_return"])
    scaling = {
        row["timestamp"]: _carino_factor(row["portfolio_return"], row["benchmark_return"]) / factor
        for row in periods
    }
    for row in periods:
        row["carino_scale"] = scaling[row["timestamp"]]
    for row in effects:
        for name in effects_names:
            row["linked_" + name] = row[name] * scaling[row["timestamp"]]
    linked = []
    for label in sorted(set(data["group"])):
        rows = [row for row in effects if row["group"] == label]
        result = {name: math.fsum(row["linked_" + name] for row in rows) for name in effects_names}
        linked.append({"group": label, **result, "total_effect": math.fsum(result.values())})
    aggregate = {name: math.fsum(row[name] for row in linked) for name in effects_names}
    active = compounded["portfolio_return"] - compounded["benchmark_return"]
    residual = _reconcile(
        math.fsum(aggregate.values()), active, "Carino attribution failed reconciliation"
    )
    if any(
        not math.isfinite(value)
        for row in [*periods, *effects, *linked]
        for value in row.values()
        if isinstance(value, float)
    ):
        raise ValueError("nonfinite derived attribution effect")

    def dated(rows, clocks):
        return pl.DataFrame(rows).with_columns(
            pl.col(name).cast(pl.Datetime("ns", "UTC")) for name in clocks
        )

    return GroupAttribution(
        {
            "currency": currency,
            "benchmarkId": benchmark_id,
            "periods": len(periods),
            "method": ("brinson_fachler_carino" if adjustments is None else
                       "brinson_fachler_explicit_trading_cost_cash_carino"),
            **({"adjustmentPolicy": "explicit_contributions_reconciled_to_observed_net_returns",
                "allocationReference": "bop_gross_benchmark_return"} if adjustments is not None else {}),
            **compounded,
            "active_return": active,
            **aggregate,
            "reconciliation_residual": residual,
            "scope": "supplied_fully_invested_groups_reconciled_to_observed_returns",
            "timing": "beginning_weights_and_classification_known_by_period_start",
            "missingGroupPolicy": "reject_no_implicit_cash_costs_or_zero_returns",
            "evidence": "caller_supplied_clocks_and_returns_not_authenticated_PIT_benchmark",
        },
        dated(periods, _CLOCKS),
        dated(effects, (*_CLOCKS, "weights_at", "classification_at")),
        pl.DataFrame(linked),
    )
