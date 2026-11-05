"Point-in-time anchored event research on explicit UTC return sessions."
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite, sqrt
from typing import Any

import numpy as np
import polars as pl

from .validation import student_t_two_sided_p_value


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("event research requires aware timestamps")
    return value.astimezone(timezone.utc)


def _sessions(frame: pl.DataFrame) -> pl.DataFrame:
    dtype = frame.schema.get("date")
    if not isinstance(dtype, pl.Datetime):
        raise ValueError("returns require a Datetime date column")
    if not dtype.time_zone or frame["date"].null_count() or frame["date"].n_unique() != frame.height:
        raise ValueError("return sessions must be unique, non-null and timezone aware")
    return frame.with_columns(pl.col("date").dt.convert_time_zone("UTC")).sort("date")


@dataclass(frozen=True, slots=True)
class EventStudyResult:
    abnormal_returns: pl.DataFrame
    average_returns: pl.DataFrame
    events: tuple[dict[str, Any], ...]
    dropped: tuple[dict[str, Any], ...]
    clusters: tuple[dict[str, Any], ...]
    inference: dict[str, Any]
    settings: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"abnormalReturns": self.abnormal_returns.to_dicts(),
                "averageReturns": self.average_returns.to_dicts(), "events": list(self.events),
                "dropped": list(self.dropped), "clusters": list(self.clusters),
                "inference": self.inference, "settings": self.settings,
                "qualificationIssued": False, "executionAuthorized": False}


def event_study(
    returns: pl.DataFrame, market: pl.DataFrame, events: pl.DataFrame, *, as_of: datetime,
    event_window: tuple[int, int] = (-5, 5), estimation_window: int = 120,
    estimation_gap: int = 10, model: str = "market",
) -> EventStudyResult:
    "Measure AR/CAR/CAAR; dates are return-session closes, offsets count rows."
    cutoff = _aware(as_of)
    if (len(event_window) != 2 or any(type(x) is not int for x in event_window)
            or event_window[0] > event_window[1]
            or type(estimation_window) is not int or estimation_window < 30
            or type(estimation_gap) is not int or estimation_gap < 0):
        raise ValueError("invalid event/estimation windows; need 30+ estimation rows")
    if model not in {"market", "market_adjusted", "mean_adjusted"}:
        raise ValueError("unknown normal-return model")
    data = _sessions(returns).filter(pl.col("date") <= cutoff)
    benchmark = _sessions(market).filter(pl.col("date") <= cutoff)
    if set(benchmark.columns) != {"date", "market"}:
        raise ValueError("benchmark must contain date and market only")
    symbols = [c for c in data.columns if c != "date"]
    if not symbols or any(not data.schema[s].is_numeric() for s in symbols) or not benchmark.schema["market"].is_numeric():
        raise ValueError("numeric asset and market returns required")
    aligned = data.select("date").join(benchmark, on="date", how="left")
    if aligned["market"].null_count():
        raise ValueError("benchmark must cover every supplied return session")
    required = {"event_id", "symbol", "event_at", "available_at"}
    if set(events.columns) != required or any(events[c].null_count() for c in ("event_id", "symbol", "available_at")):
        raise ValueError("events require event_at and non-null event_id, symbol, available_at")
    if events["event_id"].n_unique() != events.height:
        raise ValueError("duplicate event_id")
    dates = data["date"].to_list()
    market_values = aligned["market"].to_numpy().astype(float)
    rows, outcomes, dropped, intervals = [], [], [], []
    from bisect import bisect_left
    for event in events.iter_rows(named=True):
        identity, symbol = event["event_id"], event["symbol"]
        if not isinstance(identity, str) or not identity.strip() or not isinstance(symbol, str):
            raise ValueError("nonempty string event identities required")
        published = _aware(event["event_at"]) if event["event_at"] is not None else None
        available = _aware(event["available_at"])
        if published is not None and available < published:
            raise ValueError("event availability precedes publication")
        def drop(reason):
            dropped.append({"eventId": identity, "symbol": symbol, "reason": reason})
        if available > cutoff:
            drop("not_available_at_cutoff")
            continue
        if symbol not in symbols:
            drop("unknown_instrument")
            continue
        position = bisect_left(dates, available)
        if position == len(dates):
            drop("no_return_session_at_or_after_availability")
            continue
        a, b = position + event_window[0], position + event_window[1]
        # Even positive event windows cannot move model fitting past the event anchor.
        fit_end = min(position, a) - estimation_gap
        fit_start = fit_end - estimation_window
        if fit_start < 0 or a < 0:
            drop("insufficient_estimation_history")
            continue
        if b >= len(dates):
            drop("event_window_incomplete_at_cutoff")
            continue
        values = data[symbol].to_numpy().astype(float)
        y, m = values[fit_start:fit_end], market_values[fit_start:fit_end]
        target, m_target = values[a:b+1], market_values[a:b+1]
        if not all(np.isfinite(x).all() for x in (y, m, target, m_target)):
            drop("nonfinite_estimation_or_event_returns")
            continue
        if model == "market":
            design = np.column_stack([np.ones(len(m)), m])
            if np.linalg.matrix_rank(design) < 2:
                drop("unidentified_market_beta")
                continue
            alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
            aggregate = np.array([len(target), m_target.sum()])
            correction = float(aggregate @ np.linalg.solve(design.T @ design, aggregate))
            degrees = len(y) - 2
        elif model == "mean_adjusted":
            alpha, beta = float(y.mean()), 0.
            correction, degrees = len(target) ** 2 / len(y), len(y) - 1
        else:
            alpha, beta = 0., 1.
            correction, degrees = 0., len(y)
        residual = y - alpha - beta * m
        sigma2 = float(residual @ residual / degrees)
        abnormal = target - alpha - beta * m_target
        car = float(abnormal.sum())
        error = sqrt(max(0., sigma2 * (len(target) + correction)))
        values_to_check = (float(alpha), float(beta), sigma2, car, error, *map(float, abnormal))
        if not all(isfinite(x) for x in values_to_check):
            drop("nonfinite_fit_or_result")
            continue
        outcome = {"eventId": identity, "symbol": symbol, "eventAt": published.isoformat() if published is not None else None,
            "availableAt": available.isoformat(), "anchor": dates[position].isoformat(),
            "estimationStart": dates[fit_start].isoformat(), "estimationEnd": dates[fit_end-1].isoformat(),
            "eventStart": dates[a].isoformat(), "eventEnd": dates[b].isoformat(),
            "alpha": float(alpha), "beta": float(beta), "car": car, "carStandardError": error,
            "standardizedCar": car / error if error > 1e-14 else None}
        outcomes.append(outcome)
        intervals.append((a, b, identity))
        cumulative = 0.
        for offset, value in enumerate(abnormal, event_window[0]):
            cumulative += float(value)
            rows.append({"eventId": identity, "relativeSession": offset,
                         "ar": float(value), "car": cumulative})
    abnormal_frame = pl.DataFrame(rows, schema={"eventId": pl.String, "relativeSession": pl.Int64,
                                                "ar": pl.Float64, "car": pl.Float64})
    averages = (abnormal_frame.group_by("relativeSession").agg(pl.col("ar").mean().alias("aar"),
        pl.len().alias("events")).sort("relativeSession").with_columns(pl.col("aar").cum_sum().alias("caar")))
    clusters = []
    for a, b, identity in sorted(intervals):
        if clusters and a <= clusters[-1]["last"]:
            clusters[-1]["last"] = max(b, clusters[-1]["last"])
            clusters[-1]["eventIds"].append(identity)
        else:
            clusters.append({"first": a, "last": b, "eventIds": [identity]})
    shared = tuple({"start": dates[c["first"]].isoformat(), "end": dates[c["last"]].isoformat(),
                    "eventIds": c["eventIds"]} for c in clusters if len(c["eventIds"]) > 1)
    inference = {"method": "cross_sectional_car_t_test", "t": None, "pValue": None,
                 "reason": None, "acceptedEvents": len(outcomes)}
    if shared:
        inference["reason"] = "overlapping_event_windows_independence_unverified"
    elif len(outcomes) < 2:
        inference["reason"] = "fewer_than_two_nonoverlapping_events"
    else:
        cars = np.array([x["car"] for x in outcomes])
        std = float(cars.std(ddof=1))
        if std <= 1e-14:
            inference["reason"] = "zero_cross_sectional_variance"
        else:
            statistic = float(cars.mean() / (std / sqrt(len(cars))))
            inference.update(t=statistic, pValue=student_t_two_sided_p_value(statistic, len(cars)-1))
    return EventStudyResult(abnormal_frame, averages, tuple(outcomes), tuple(dropped), shared,
        inference, {"asOf": cutoff.isoformat(), "model": model, "eventWindow": list(event_window),
        "estimationWindow": estimation_window, "estimationGap": estimation_gap,
        "anchorPolicy": "first_return_close_at_or_after_available_at",
        "calendarPolicy": "caller_supplied_return_sessions_no_calendar_inference",
        "inferenceAssumption": "nonoverlap_does_not_prove_independence_t_test_is_exploratory",
        "scope": "retrospective_association_not_causation_or_trade_signal"})
