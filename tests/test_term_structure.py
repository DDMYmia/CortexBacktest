from datetime import datetime, timedelta, timezone
import json
import math

import polars as pl
import pytest

from cortexbacktest.research.term_structure import term_structure, curve_roll_schedule

T = datetime(2025, 1, 7, tzinfo=timezone.utc)


def chain(days=(0, 2)):
    rows = []
    for day in days:
        stamp = T + timedelta(days=day)
        for i, (expiry, price) in enumerate(
            [
                (T + timedelta(days=3), 100.0),
                (T + timedelta(days=45), 110.0),
                (T + timedelta(days=75), 115.0),
            ]
        ):
            rows.append(
                dict(
                    date=stamp,
                    root="CL",
                    symbol="CL" + str(i),
                    expiry=expiry,
                    last_trade_at=expiry,
                    quote_at=stamp,
                    known_at=stamp,
                    price=price,
                    price_scale=1.0,
                    currency="USD",
                    price_unit="BARREL",
                    multiplier=1000.0,
                )
            )
    return pl.DataFrame(rows)


def compute(data, **kwargs):
    return term_structure(data, max_quote_age=timedelta(hours=1), **kwargs)


def test_ratio_annualization_and_explicit_roll():
    data = chain()
    out = compute(data)
    assert out.values["CL"][0] == pytest.approx(math.log(100 / 110) * 365 / 42)
    assert out.diagnostics["roll_return"][0] == pytest.approx(100 / 110 - 1)
    assert out.diagnostics["near_contract"].to_list() == ["CL0", "CL1"]
    assert compute(data, annualized=False).values["CL"][0] == pytest.approx(100 / 110 - 1)
    times = {d: d + timedelta(hours=1) for d in out.values["date"]}
    rolls = curve_roll_schedule(out, root="CL", execution_times=times)
    assert len(rolls) == 1 and rolls[0].from_contract == "CL0" and rolls[0].to_contract == "CL1"
    assert rolls[0].timestamp == T + timedelta(days=2, hours=1)
    json.dumps(out.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("known_at", T + timedelta(minutes=1), "chain_not_available_by_decision"),
        ("quote_at", T - timedelta(hours=2), "asynchronous_contract_quotes"),
        ("price", 0.0, "nonpositive_or_missing_price_ratio_undefined"),
        ("price", -20.0, "nonpositive_or_missing_price_ratio_undefined"),
        ("price", None, "nonpositive_or_missing_price_ratio_undefined"),
        ("price", float("inf"), "nonpositive_or_missing_price_ratio_undefined"),
        ("currency", "EUR", "incomparable_currency_or_price_unit"),
        ("price_unit", "GALLON", "incomparable_currency_or_price_unit"),
    ],
)
def test_unavailable_curves_not_zero(field, value, reason):
    data = chain((0,)).with_columns(
        pl.when(pl.col("symbol") == "CL0").then(pl.lit(value)).otherwise(pl.col(field)).alias(field)
    )
    result = compute(data)
    assert result.values["CL"][0] is None
    assert result.diagnostics["reason"][0] == reason


def test_stale_quotes_and_missing_root_snapshots():
    data = chain((0,)).with_columns((pl.col("quote_at") - timedelta(hours=2)).alias("quote_at"))
    assert compute(data).diagnostics["reason"][0] == "stale_contract_quotes"
    missing = pl.concat([chain((0,)), chain((2,)).with_columns(pl.lit("GC").alias("root"))])
    result = compute(missing)
    assert result.values["GC"][0] is None and result.values["CL"][1] is None
    assert result.diagnostics.filter(pl.col("reason") == "missing_root_snapshot").height == 2


def test_expired_ambiguous_and_price_scale():
    expired = chain((0,)).with_columns(pl.lit(T).alias("last_trade_at"))
    assert (
        compute(expired).diagnostics["reason"][0] == "fewer_than_two_unexpired_eligible_contracts"
    )
    ambiguous = chain((0,)).with_columns(
        pl.lit(T + timedelta(days=45)).alias("expiry"),
        pl.lit(T + timedelta(days=45)).alias("last_trade_at"),
    )
    assert compute(ambiguous).diagnostics["reason"][0] == "ambiguous_equal_expiry_contracts"
    cents = chain((0,)).with_columns(
        pl.when(pl.col("symbol") == "CL0")
        .then(pl.col("price") * 100)
        .otherwise(pl.col("price"))
        .alias("price"),
        pl.when(pl.col("symbol") == "CL0")
        .then(0.01)
        .otherwise(pl.col("price_scale"))
        .alias("price_scale"),
    )
    assert compute(cents).values.equals(compute(chain((0,))).values)


def test_future_snapshot_cannot_change_earlier_curve():
    whole = chain()
    prefix = compute(whole.filter(pl.col("date") == T))
    changed = whole.with_columns(
        pl.when(pl.col("date") > T).then(999.0).otherwise(pl.col("price")).alias("price")
    )
    assert compute(changed).values.head(1).equals(prefix.values)


def test_invalid_snapshot_and_roll_clocks_reject():
    with pytest.raises(ValueError, match="duplicate"):
        compute(pl.concat([chain(), chain()]))
    with pytest.raises(ValueError, match="contradict"):
        compute(chain().with_columns((pl.col("quote_at") + timedelta(days=1)).alias("quote_at")))
    with pytest.raises(ValueError, match="positive explicit"):
        compute(chain().with_columns(pl.lit(0.0).alias("multiplier")))
    out = compute(chain())
    with pytest.raises(ValueError, match="strictly follow"):
        curve_roll_schedule(out, root="CL", execution_times={d: d for d in out.values["date"]})
    with pytest.raises(ValueError, match="last trading"):
        curve_roll_schedule(
            out, root="CL", execution_times={d: d + timedelta(days=10) for d in out.values["date"]}
        )
    with pytest.raises(ValueError, match="every supplied"):
        curve_roll_schedule(out, root="CL", execution_times={})
