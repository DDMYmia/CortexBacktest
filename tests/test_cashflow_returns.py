from datetime import datetime, timedelta, timezone
import json
import math

import polars as pl
import pytest

from cortexbacktest.reporting import cashflow_performance

START = datetime(2023, 1, 1, tzinfo=timezone.utc)
MID, END = START + timedelta(days=365), START + timedelta(days=730)
FLOW_SCHEMA = {
    "timestamp": pl.Datetime("us", "UTC"),
    "flow_id": pl.String,
    "amount": pl.Float64,
    "currency": pl.String,
    "kind": pl.String,
}


def nav(values, dates=None):
    dates = dates or [START, MID, END]
    return pl.DataFrame(
        {"timestamp": dates, "value": [float(v) for v in values], "currency": ["USD"] * len(values)}
    )


def flows(rows=()):
    return pl.DataFrame(
        [
            {
                "timestamp": stamp,
                "flow_id": str(i),
                "amount": float(amount),
                "currency": currency,
                "kind": kind,
            }
            for i, (stamp, amount, currency, kind) in enumerate(rows)
        ],
        schema=FLOW_SCHEMA,
    )


def report(values, activity=None, **kwargs):
    return cashflow_performance(
        values,
        flows() if activity is None else activity,
        currency="USD",
        coverage_complete=kwargs.pop("complete", True),
        as_of=END,
        **kwargs,
    )


def test_twr_removes_contribution_mwr_tracks_money_timing():
    activity = flows([(MID, 100.0, "USD", "deposit")])
    result = report(nav([100, 210, 189]), activity)
    assert result.summary["twr"]["totalReturn"] == pytest.approx(-0.01)
    assert result.summary["twr"]["totalReturn"] == pytest.approx(
        report(nav([100, 110, 99])).summary["twr"]["totalReturn"]
    )
    expected = ((-100 + math.sqrt(10000 + 4 * 100 * 189)) / 200) ** 2 - 1
    assert result.summary["mwr"]["periodReturn"] == pytest.approx(expected, abs=1e-10)
    assert result.summary["mwr"]["periodReturn"] != pytest.approx(-0.01)
    json.dumps(result.to_dict(), allow_nan=False)


def test_start_timing_and_full_withdrawal():
    start = report(
        nav([100, 220, 242]), flows([(START, 100.0, "USD", "deposit")]), flow_timing="start"
    )
    assert start.summary["twr"]["totalReturn"] == pytest.approx(0.21)
    end = report(nav([100, 0], [START, END]), flows([(END, -110.0, "USD", "withdrawal")]))
    assert end.summary["twr"]["totalReturn"] == pytest.approx(0.1)


def test_internal_dividend_fee_and_trade_are_not_external_cash():
    result = report(
        nav([100, 110, 108]),
        flows(
            [(MID, 5.0, "USD", "dividend"), (END, -2.0, "USD", "fee"), (MID, -10.0, "USD", "trade")]
        ),
    )
    assert result.summary["netExternalFlow"] == 0 and result.summary["internalEvents"] == 3
    assert result.summary["twr"]["totalReturn"] == pytest.approx(0.08)


def test_interim_flow_needs_valuation_for_true_twr_but_not_mwr_or_dietz():
    result = report(nav([100, 220], [START, END]), flows([(MID, 100.0, "USD", "deposit")]))
    assert result.summary["twr"]["totalReturn"] is None
    assert result.summary["twr"]["reason"] == "interim_valuation_required"
    assert result.summary["modifiedDietz"]["totalReturn"] == pytest.approx(20 / 150)
    assert result.summary["mwr"]["status"] == "available"


def test_multiple_irr_roots_are_reported_without_arbitrary_choice():
    result = report(
        nav([100, 0], [START, END]),
        flows([(MID, -230.0, "USD", "withdrawal"), (END, 132.0, "USD", "deposit")]),
    )
    mwr = result.summary["mwr"]
    assert mwr["status"] == "ambiguous" and mwr["periodReturn"] is None
    assert [x["annualizedReturn"] for x in mwr["candidates"]] == pytest.approx([0.1, 0.2], abs=1e-9)


def test_tangent_root_does_not_claim_global_uniqueness():
    result = report(
        nav([100, 0], [START, END]),
        flows([(MID, -220.0, "USD", "withdrawal"), (END, 121.0, "USD", "deposit")]),
    )
    assert result.summary["mwr"]["status"] == "unproven_uniqueness"
    assert result.summary["mwr"]["periodReturn"] is None


def test_short_window_annualization_withheld_and_empty_flows_explicit():
    result = report(nav([100, 110], [START, START + timedelta(days=30)]))
    assert result.summary["twr"]["annualizedReturn"] is None
    assert result.summary["mwr"]["annualizedReturn"] is None
    assert result.summary["mwr"]["periodReturn"] == pytest.approx(0.1)


def test_unknown_flow_coverage_yields_no_returns():
    result = report(nav([100, 110, 120]), complete=False)
    assert result.summary["status"] == "insufficient_evidence"
    assert result.summary["twr"] is None and result.summary["mwr"] is None


@pytest.mark.parametrize(
    "kind", ["currency", "classification", "duplicate", "negative_deposit", "future", "boundary"]
)
def test_invalid_cash_evidence_rejected(kind):
    activity = flows([(MID, 100.0, "USD", "deposit")])
    if kind == "currency":
        activity = activity.with_columns(pl.lit("EUR").alias("currency"))
    elif kind == "classification":
        activity = activity.with_columns(pl.lit("transfer").alias("kind"))
    elif kind == "duplicate":
        activity = pl.concat([activity, activity])
    elif kind == "negative_deposit":
        activity = activity.with_columns(-pl.col("amount"))
    elif kind == "future":
        activity = activity.with_columns(pl.lit(END + timedelta(days=1)).alias("timestamp"))
    else:
        activity = activity.with_columns(pl.lit(START).alias("timestamp"))
    with pytest.raises(ValueError):
        report(nav([100, 210, 220]), activity)


@pytest.mark.parametrize("values", [[0, 110, 120], [100, -1, 120], [100, float("nan"), 120]])
def test_invalid_nav_rejected(values):
    with pytest.raises(ValueError):
        report(nav(values))


def test_submicrosecond_flow_is_not_rounded_onto_a_nav_boundary():
    activity = flows([(MID, 100.0, "USD", "deposit")]).with_columns(
        pl.col("timestamp").dt.cast_time_unit("ns") + pl.duration(nanoseconds=1)
    )
    with pytest.raises(ValueError, match="microsecond"):
        report(nav([100, 210, 220]), activity)


def test_explicit_zero_capital_periods_preserve_null_returns_before_funding():
    dates = [START, START + timedelta(days=1), START + timedelta(days=2), START + timedelta(days=3)]
    result = report(
        nav([0, 0, 110, 121], dates),
        flows([(dates[1], 100.0, "USD", "deposit")]),
        flow_timing="start",
        allow_zero_capital=True,
    )
    assert result.periods["return"].to_list()[0] is None
    assert result.summary["twr"]["totalReturn"] == pytest.approx(0.21)
    assert result.summary["twr"]["zeroCapitalPeriods"] == 1
    assert result.summary["mwr"]["status"] == "available"
    json.dumps(result.to_dict(), allow_nan=False)


def test_all_zero_capital_is_unavailable_and_unexplained_gain_is_not_a_return():
    result = report(nav([0, 0, 0]), allow_zero_capital=True)
    assert result.summary["twr"]["reason"] == "no_positive_capital_periods"
    assert result.summary["twr"]["totalReturn"] is None
    assert result.periods["return"].null_count() == 2
    unexplained = report(nav([0, 10, 11]), allow_zero_capital=True)
    assert unexplained.summary["twr"]["status"] == "unavailable"
    assert unexplained.summary["twr"]["totalReturn"] is None
