"Cash-flow adjusted research returns; not an account ledger or reconciliation issuer."

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import json
from typing import Any

import numpy as np
import polars as pl

EXTERNAL_IN = {"deposit", "external_transfer_in"}
EXTERNAL_OUT = {"withdrawal", "external_transfer_out"}
INTERNAL = {"trade", "dividend", "interest", "fee", "fx_conversion", "in_scope_transfer"}


@dataclass(frozen=True, slots=True)
class CashflowPerformance:
    summary: dict[str, Any]
    periods: pl.DataFrame

    def to_dict(self) -> dict[str, Any]:
        frame = self.periods
        if frame.height:
            frame = frame.with_columns(pl.col("start", "end").dt.to_string("%+"))
        return {
            "summary": self.summary,
            "periods": frame.to_dicts(),
            "qualificationIssued": False,
            "executionAuthorized": False,
        }


def _clock_frame(frame: pl.DataFrame, cutoff: datetime) -> pl.DataFrame:
    dtype = frame.schema.get("timestamp")
    if not isinstance(dtype, pl.Datetime) or not dtype.time_zone or frame["timestamp"].null_count():
        raise ValueError("cash-flow observations require non-null aware timestamps")
    if frame.filter((pl.col("timestamp").dt.nanosecond() % 1000) != 0).height:
        raise ValueError("cashflow timestamps must be representable at microsecond precision")
    result = frame.with_columns(
        pl.col("timestamp").dt.convert_time_zone("UTC").dt.cast_time_unit("us")
    )
    if result.filter(pl.col("timestamp") > cutoff).height:
        raise ValueError("cash-flow observations exceed as_of")
    return result.sort("timestamp")


def _annual(period_return: float, years: float) -> float | None:
    if years < 1 or period_return < -1:
        return None
    if period_return == -1:
        return -1.0
    value = math.expm1(math.log1p(period_return) / years)
    return value if math.isfinite(value) else None


def _growth_roots(times: np.ndarray, amounts: np.ndarray, low: float, high: float) -> list[float]:
    # Exponential-sum zeros are isolated by derivative stationary points. A single
    # sign variation permits at most one real root; no arbitrary initial guess.
    keep = amounts != 0
    times, amounts = times[keep], amounts[keep]
    if len(amounts) < 2:
        return []
    times = times - times[0]
    amounts = amounts / np.max(np.abs(amounts))
    variations = int(np.count_nonzero(np.sign(amounts[1:]) != np.sign(amounts[:-1])))
    if variations == 0:
        return []

    def value(x):
        exponent = -times * x
        weights = amounts * np.exp(exponent - np.max(exponent))
        return float(weights.sum() / np.abs(weights).sum())

    def bisect(a, b):
        fa = value(a)
        for _ in range(100):
            middle = (a + b) / 2
            fm = value(middle)
            if abs(fm) < 1e-14 or b - a < 1e-12:
                return middle
            if fa * fm < 0:
                b = middle
            else:
                a, fa = middle, fm
        return (a + b) / 2

    critical = (
        [] if variations == 1 else _growth_roots(times[1:], -times[1:] * amounts[1:], low, high)
    )
    points = [low, *critical, high]
    roots = [x for x in points if abs(value(x)) < 1e-12]
    for a, b in zip(points[:-1], points[1:], strict=True):
        if abs(value(a)) > 1e-12 and abs(value(b)) > 1e-12 and value(a) * value(b) < 0:
            roots.append(bisect(a, b))
    unique = []
    for root in sorted(roots):
        if not unique or abs(root - unique[-1]) > 1e-8:
            unique.append(root)
    return unique


def _money_weighted(start, end, opening, closing, flows) -> dict:
    dated = {start: -opening, end: closing}
    for stamp, amount in flows:
        dated[stamp] = dated.get(stamp, 0.0) - amount
    dated = sorted((stamp, amount) for stamp, amount in dated.items() if amount != 0)
    if not dated or len(dated) > 128:
        return {"status": "unavailable", "reason": "empty_or_excessive_cashflow_complexity"}
    duration = (end - start).total_seconds()
    times = np.array([(stamp - start).total_seconds() / duration for stamp, _ in dated])
    amounts = np.array([value for _, value in dated])
    variations = int(np.count_nonzero(np.sign(amounts[1:]) != np.sign(amounts[:-1])))
    roots = _growth_roots(times, amounts, -30.0, 30.0)
    years = duration / (365 * 86400)
    candidates = [
        {"periodReturn": math.expm1(root), "annualizedReturn": _annual(math.expm1(root), years)}
        for root in roots
    ]
    status = (
        "available"
        if len(roots) == 1 and variations == 1
        else "ambiguous"
        if len(roots) > 1
        else "unproven_uniqueness"
        if roots
        else "no_root_in_bounds"
    )
    return {
        "status": status,
        "periodReturn": candidates[0]["periodReturn"] if status == "available" else None,
        "annualizedReturn": candidates[0]["annualizedReturn"] if status == "available" else None,
        "candidates": candidates,
        "signChanges": variations,
        "globalUniquenessProven": status == "available",
        "logPeriodGrowthBounds": [-30.0, 30.0],
        "method": "stationary_point_isolation_and_bisection_act_365",
    }


def cashflow_performance(
    valuations: pl.DataFrame,
    flows: pl.DataFrame,
    *,
    currency: str,
    coverage_complete: bool,
    as_of: datetime,
    flow_timing: str = "end",
    allow_zero_capital: bool = False,
) -> CashflowPerformance:
    "Report exact boundary TWR, Modified Dietz and MWR without assuming missing flows are zero."
    periods_schema = {
        "start": pl.Datetime("us", "UTC"),
        "end": pl.Datetime("us", "UTC"),
        "opening": pl.Float64,
        "closing": pl.Float64,
        "externalFlow": pl.Float64,
        "return": pl.Float64,
    }
    empty = pl.DataFrame(schema=periods_schema)
    if type(allow_zero_capital) is not bool:
        raise ValueError("explicit boolean zero-capital policy required")
    if type(coverage_complete) is not bool:
        raise ValueError("explicit boolean flow coverage required")
    if not coverage_complete:
        return CashflowPerformance(
            {
                "status": "insufficient_evidence",
                "reason": "cashflow_coverage_incomplete",
                "currency": currency,
                "twr": None,
                "mwr": None,
                "modifiedDietz": None,
            },
            empty,
        )
    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("aware as_of required")
    if flow_timing not in {"start", "end"} or not isinstance(currency, str) or not currency:
        raise ValueError("explicit currency and start/end flow timing required")
    if set(valuations.columns) != {"timestamp", "value", "currency"} or valuations.height < 2:
        raise ValueError("opening and closing NAV observations required")
    if set(flows.columns) != {"timestamp", "flow_id", "amount", "currency", "kind"}:
        raise ValueError("explicit cashflow identity, amount, currency and classification required")
    cutoff = as_of.astimezone(timezone.utc)
    nav, activity = _clock_frame(valuations, cutoff), _clock_frame(flows, cutoff)
    if (
        nav["timestamp"].n_unique() != nav.height
        or activity["flow_id"].n_unique() != activity.height
    ):
        raise ValueError("duplicate valuation or flow identity")
    for frame, field in ((nav, "value"), (activity, "amount")):
        if (
            not frame.schema[field].is_numeric()
            or frame[field].null_count()
            or not frame[field].is_finite().all()
        ):
            raise ValueError("complete finite NAV and cash amounts required")
        if frame["currency"].null_count() or any(c != currency for c in frame["currency"]):
            raise ValueError("mixed currencies require an explicit verified upstream conversion")
    dates = nav["timestamp"].to_list()
    values = nav["value"].to_list()
    if (values[0] <= 0 and not (allow_zero_capital and values[0] == 0)) or any(
        v < 0 for v in values
    ):
        raise ValueError("positive opening and nonnegative subsequent NAV required")
    external = []
    for row in activity.iter_rows(named=True):
        kind, amount, stamp = row["kind"], row["amount"], row["timestamp"]
        if not isinstance(row["flow_id"], str) or not row["flow_id"].strip():
            raise ValueError("stable flow identity required")
        if kind not in EXTERNAL_IN | EXTERNAL_OUT | INTERNAL:
            raise ValueError("unclassified flow; transfer scope must be explicit")
        if not dates[0] <= stamp <= dates[-1]:
            raise ValueError("cashflow outside valuation scope")
        if kind in INTERNAL:
            continue
        if (kind in EXTERNAL_IN and amount <= 0) or (kind in EXTERNAL_OUT and amount >= 0):
            raise ValueError("external cashflow sign contradicts portfolio perspective")
        if (flow_timing == "end" and stamp == dates[0]) or (
            flow_timing == "start" and stamp == dates[-1]
        ):
            raise ValueError("boundary cashflow contradicts declared valuation timing")
        external.append((stamp, float(amount)))
    by_time = {}
    for stamp, amount in external:
        by_time[stamp] = by_time.get(stamp, 0.0) + amount
    twr = {"status": "unavailable", "reason": "interim_valuation_required", "totalReturn": None}
    rows = []
    if all(stamp in dates for stamp in by_time):
        growth = 1.0
        reason = None
        zero_periods = 0
        for i in range(1, len(dates)):
            flow = by_time.get(dates[i - 1] if flow_timing == "start" else dates[i], 0.0)
            capital = values[i - 1] + (flow if flow_timing == "start" else 0.0)
            ending = values[i] - (flow if flow_timing == "end" else 0.0)
            if allow_zero_capital and capital == 0 and ending == 0:
                period_return = None
                zero_periods += 1
            elif capital <= 0 or ending < 0:
                reason = "nonpositive_capital_or_negative_flow_adjusted_nav"
                rows = []
                break
            else:
                period_return = ending / capital - 1
                growth *= 1 + period_return
            rows.append(
                {
                    "start": dates[i - 1],
                    "end": dates[i],
                    "opening": values[i - 1],
                    "closing": values[i],
                    "externalFlow": flow,
                    "return": period_return,
                }
            )
        if reason is None and zero_periods == len(dates) - 1:
            reason = "no_positive_capital_periods"
        if reason is None:
            twr = {
                "status": "available",
                "totalReturn": growth - 1,
                "reason": None,
                "annualizedReturn": _annual(
                    growth - 1, (dates[-1] - dates[0]).total_seconds() / (365 * 86400)
                ),
            }
        else:
            twr["reason"] = reason
        if zero_periods:
            twr["zeroCapitalPeriods"] = zero_periods
    duration = (dates[-1] - dates[0]).total_seconds()
    weighted = sum(
        amount * (dates[-1] - stamp).total_seconds() / duration for stamp, amount in external
    )
    total_flow = sum(amount for _, amount in external)
    capital = values[0] + weighted
    dietz = (
        {
            "status": "available",
            "totalReturn": (values[-1] - values[0] - total_flow) / capital,
            "averageCapital": capital,
            "weightedExternalFlow": weighted,
            "interpretation": "money_weighted_approximation_not_true_twr",
        }
        if capital > 0
        else {"status": "unavailable", "reason": "nonpositive_average_capital"}
    )
    mwr = _money_weighted(dates[0], dates[-1], values[0], values[-1], external)
    summary = {
        "status": "available" if twr["status"] == mwr["status"] == "available" else "partial",
        "currency": currency,
        "flowTiming": flow_timing,
        "asOf": cutoff.isoformat(),
        "from": dates[0].isoformat(),
        "to": dates[-1].isoformat(),
        "netExternalFlow": total_flow,
        "externalEvents": len(external),
        "internalEvents": activity.height - len(external),
        "twr": twr,
        "mwr": mwr,
        "modifiedDietz": dietz,
        "sourceCoverage": "caller_declared_not_broker_certification",
        "annualizationPolicy": "withheld_for_windows_shorter_than_365_days",
    }
    try:
        json.dumps(summary, allow_nan=False)
    except ValueError as error:
        raise ValueError("cash-flow derived metrics must remain finite") from error
    return CashflowPerformance(summary, pl.DataFrame(rows, schema=periods_schema))
