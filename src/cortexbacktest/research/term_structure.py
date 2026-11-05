"Synchronous raw-contract curve scores and explicit schedules for the existing roll engine."

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
import math

import polars as pl

from ..data.futures import ContractRoll
from .holding import _clock, _integer
from .recipes import RecipeResult, _dated


_COLUMNS = {
    "date",
    "root",
    "symbol",
    "expiry",
    "last_trade_at",
    "quote_at",
    "known_at",
    "price",
    "price_scale",
    "currency",
    "price_unit",
    "multiplier",
}


def term_structure(
    chain: pl.DataFrame,
    *,
    max_quote_age: timedelta,
    min_days_to_last_trade: int = 2,
    annualized: bool = True,
) -> RecipeResult:
    "Score the first two eligible expiries; never infer a curve from adjusted continuous bars."
    _integer(min_days_to_last_trade, "min_days_to_last_trade")
    if (
        set(chain.columns) != _COLUMNS
        or not chain.height
        or not isinstance(max_quote_age, timedelta)
        or max_quote_age.total_seconds() < 0
        or type(annualized) is not bool
    ):
        raise ValueError("complete raw-contract snapshots and explicit quote age required")
    data = _dated(chain, ("date", "expiry", "last_trade_at", "quote_at", "known_at"))
    for name in ["root", "symbol", "currency", "price_unit"]:
        if (
            data.schema[name] != pl.String
            or data[name].null_count()
            or any(not s or s != s.strip() for s in data[name])
        ):
            raise ValueError("complete explicit contract labels required")
    if any(s != s.upper() for name in ["root", "symbol", "currency"] for s in data[name]):
        raise ValueError("normalized uppercase contract identities and currency required")
    for name in ["price", "price_scale", "multiplier"]:
        if not data.schema[name].is_numeric() and data.schema[name] != pl.Null:
            raise ValueError("numeric contract prices and units required")
    if data.select("date", "root", "symbol").unique().height != data.height:
        raise ValueError("duplicate snapshot contract; resolve received revisions upstream")
    if data.filter(
        (pl.col("last_trade_at") > pl.col("expiry")) | (pl.col("quote_at") > pl.col("known_at"))
    ).height:
        raise ValueError("contract or quote clocks contradict declared availability")
    if any(
        v is None or not math.isfinite(v) or v <= 0
        for name in ["price_scale", "multiplier"]
        for v in data[name]
    ):
        raise ValueError("positive explicit price scale and contract multiplier required")
    roots = sorted(set(data["root"]))
    dates = sorted(set(data["date"]))
    grouped = {(g["date"][0], g["root"][0]): g for g in data.partition_by("date", "root")}
    output, audit = [], []
    for date in dates:
        values = {root: None for root in roots}
        for root in roots:
            block = grouped.get((date, root))
            row = {
                "date": date,
                "root": root,
                "reason": None,
                "near_contract": None,
                "far_contract": None,
                "near_last_trade_at": None,
                "far_last_trade_at": None,
                "near_expiry": None,
                "far_expiry": None,
                "quote_at": None,
                "known_at": None,
                "roll_return": None,
                "annualized_log_carry": None,
                "score": None,
            }
            if block is None:
                row["reason"] = "missing_root_snapshot"
            elif block["currency"].n_unique() != 1 or block["price_unit"].n_unique() != 1:
                row["reason"] = "incomparable_currency_or_price_unit"
            elif block.filter(pl.col("known_at") > date).height:
                row["reason"] = "chain_not_available_by_decision"
            else:
                eligible = block.filter(
                    (pl.col("last_trade_at") > date)
                    & (pl.col("last_trade_at") >= date + timedelta(days=min_days_to_last_trade))
                )
                eligible = eligible.sort("expiry", "symbol")
                if eligible.height < 2:
                    row["reason"] = "fewer_than_two_unexpired_eligible_contracts"
                elif eligible["expiry"].n_unique() != eligible.height:
                    row["reason"] = "ambiguous_equal_expiry_contracts"
                else:
                    near, far = eligible.head(2).to_dicts()
                    row.update(
                        near_contract=near["symbol"],
                        far_contract=far["symbol"],
                        near_last_trade_at=near["last_trade_at"],
                        far_last_trade_at=far["last_trade_at"],
                        near_expiry=near["expiry"],
                        far_expiry=far["expiry"],
                        known_at=max(near["known_at"], far["known_at"]),
                    )
                    if near["quote_at"] != far["quote_at"]:
                        row["reason"] = "asynchronous_contract_quotes"
                    elif date - near["quote_at"] > max_quote_age:
                        row["reason"] = "stale_contract_quotes"
                    elif any(
                        x["price"] is None or not math.isfinite(x["price"]) or x["price"] <= 0
                        for x in [near, far]
                    ):
                        row["reason"] = "nonpositive_or_missing_price_ratio_undefined"
                    else:
                        p, q = (x["price"] * x["price_scale"] for x in [near, far])
                        if not math.isfinite(p) or not math.isfinite(q) or p <= 0 or q <= 0:
                            row["reason"] = "invalid_normalized_prices"
                        else:
                            gap = (far["expiry"] - near["expiry"]).total_seconds() / 86400
                            log_carry = (math.log(p) - math.log(q)) * 365 / gap
                            ratio = p / q - 1
                            if not math.isfinite(log_carry) or not math.isfinite(ratio):
                                row["reason"] = "nonfinite_curve_score"
                            else:
                                score = log_carry if annualized else ratio
                                row.update(
                                    quote_at=near["quote_at"],
                                    roll_return=ratio,
                                    annualized_log_carry=log_carry,
                                    score=score,
                                )
                                values[root] = score
            audit.append(row)
        output.append({"date": date, **values})
    diagnostics = pl.DataFrame(audit, infer_schema_length=None).with_columns(
        pl.col(n).cast(pl.Datetime("us", "UTC"))
        for n in [
            "near_last_trade_at",
            "far_last_trade_at",
            "near_expiry",
            "far_expiry",
            "quote_at",
            "known_at",
        ]
    )
    result = pl.DataFrame(output, infer_schema_length=None).with_columns(
        pl.col(roots).cast(pl.Float64)
    )
    return RecipeResult(
        result,
        diagnostics,
        {
            "method": "synchronous_first_two_eligible_expiries",
            "output": "factor_scores",
            "score": "log_near_over_far_ACT365" if annualized else "near_over_far_minus_one",
            "minCalendarDaysToLastTrade": min_days_to_last_trade,
            "maxQuoteAgeSeconds": max_quote_age.total_seconds(),
            "evidence": "caller_supplied_raw_contract_metadata_not_verified_chain_completeness",
            "interpretation": "curve_indicator_not_realized_roll_return_or_trading_permission",
        },
    )


def curve_roll_schedule(
    result: RecipeResult,
    *,
    root: str,
    execution_times: Mapping[datetime, datetime],
) -> tuple[ContractRoll, ...]:
    "Map selected-front changes to explicit later execution instants; no calendar guessing."
    rows = result.diagnostics.filter(pl.col("root") == root).sort("date").to_dicts()
    if not rows or any(r["reason"] is not None for r in rows):
        raise ValueError("complete available root curve history required for a roll schedule")
    if set(execution_times) != {r["date"] for r in rows}:
        raise ValueError("explicit execution time for every supplied decision required")
    instants = [_clock(execution_times[r["date"]]) for r in rows]
    if any(t <= r["date"] for t, r in zip(instants, rows, strict=True)) or any(
        a >= b for a, b in zip(instants, instants[1:])
    ):
        raise ValueError("execution must strictly follow decisions and increase")
    schedule = []
    old = rows[0]
    for row, stamp in zip(rows[1:], instants[1:], strict=True):
        if row["near_contract"] != old["near_contract"]:
            if stamp >= old["near_last_trade_at"] or stamp >= row["near_last_trade_at"]:
                raise ValueError("roll execution must precede both contracts' last trading instant")
            schedule.append(ContractRoll(stamp, old["near_contract"], row["near_contract"]))
        old = row
    return tuple(schedule)
