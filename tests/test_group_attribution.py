from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
import json
import math

import numpy as np
import polars as pl
import pytest

from cortexbacktest.reporting import group_attribution
from cortexbacktest.reporting.group_attribution import _carino_factor


START = datetime(2025, 1, 1, tzinfo=timezone.utc)


def inputs(periods=1):
    groups, observed = [], []
    for i in range(periods):
        start = START + timedelta(days=i)
        end = start + timedelta(days=1)
        # Hand-calculable allocation=.002, selection=.007, interaction=.0026.
        groups.extend(
            {
                "timestamp": end,
                "period_start": start,
                "weights_at": start,
                "classification_at": start,
                "currency": "USD",
                "group": label,
                "portfolio_weight": wp,
                "benchmark_weight": wb,
                "portfolio_return": rp,
                "benchmark_return": rb,
            }
            for label, wp, wb, rp, rb in [("A", 0.6, 0.5, 0.05, 0.03), ("B", 0.4, 0.5, 0.004, 0.01)]
        )
        observed.append(
            {
                "timestamp": end,
                "period_start": start,
                "portfolio_return": 0.0316,
                "benchmark_return": 0.02,
                "currency": "USD",
            }
        )
    return pl.DataFrame(groups), pl.DataFrame(observed)


def run(groups, observed):
    return group_attribution(groups, observed, currency="USD", benchmark_id="frozen-test-benchmark")


def test_single_period_hand_calculation_and_observed_reconciliation():
    result = run(*inputs())
    assert result.summary["allocation"] == pytest.approx(0.002)
    assert result.summary["selection"] == pytest.approx(0.007)
    assert result.summary["interaction"] == pytest.approx(0.0026)
    assert result.summary["active_return"] == pytest.approx(0.0116)
    assert abs(result.summary["reconciliation_residual"]) < 1e-16
    assert result.periods["carino_scale"][0] == pytest.approx(1)
    document = result.to_dict()
    assert document["qualificationIssued"] is False
    json.dumps(document, allow_nan=False)


def test_multiperiod_matches_independent_high_precision_log_linking():
    groups, observed = inputs(3)
    groups = groups.with_columns(
        pl.when(pl.col("timestamp") == START + timedelta(days=2))
        .then(pl.col("portfolio_return") * -2)
        .otherwise(pl.col("portfolio_return"))
        .alias("portfolio_return")
    )
    observed = observed.with_columns(
        pl.when(pl.col("timestamp") == START + timedelta(days=2))
        .then(-0.0632)
        .otherwise(pl.col("portfolio_return"))
        .alias("portfolio_return")
    )
    result = run(groups, observed)
    with localcontext() as ctx:
        ctx.prec = 65

        def D(x):
            return Decimal(str(x))

        rp = [D(".0316"), D("-.0632"), D(".0316")]
        rb = [D(".02")] * 3
        p = math.prod(1 + x for x in rp) - 1
        b = math.prod(1 + x for x in rb) - 1
        k = ((1 + p).ln() - (1 + b).ln()) / (p - b)
        for effect in ["allocation", "selection", "interaction"]:
            expected = sum(
                D(row[effect]) * (((1 + rp[i]).ln() - (1 + rb[i]).ln()) / (rp[i] - rb[i])) / k
                for i, row in enumerate(result.periods.iter_rows(named=True))
            )
            assert result.summary[effect] == pytest.approx(float(expected), abs=1e-15)
        assert result.summary["active_return"] == pytest.approx(float(p - b), abs=1e-15)
    assert abs(result.summary["reconciliation_residual"]) < 1e-15
    assert math.fsum(result.periods["active_return"]) != pytest.approx(
        result.summary["active_return"]
    )


def test_identical_and_near_equal_returns_have_finite_carino_limit():
    for value in [0.0, 0.1, -0.99999999999]:
        assert _carino_factor(value, value) == pytest.approx(1 / (1 + value))
        higher = math.nextafter(value, math.inf)
        assert math.isfinite(_carino_factor(higher, value))
    groups, observed = inputs(2)
    groups = groups.with_columns(pl.col("benchmark_return").alias("portfolio_return"))
    # Active returns remain nonzero because the weights differ.
    observed = observed.with_columns(pl.lit(0.022).alias("portfolio_return"))
    result = run(groups, observed)
    assert result.summary["selection"] == 0
    assert result.summary["interaction"] == 0
    assert result.summary["allocation"] == pytest.approx(0.004084)
    groups = groups.with_columns(pl.col("benchmark_weight").alias("portfolio_weight"))
    observed = observed.with_columns(pl.col("benchmark_return").alias("portfolio_return"))
    assert run(groups, observed).summary["active_return"] == 0


def test_reordering_and_changing_group_universe_do_not_change_linked_totals():
    groups, observed = inputs(2)
    groups = groups.with_columns(
        pl.when((pl.col("group") == "B") & (pl.col("timestamp") == START + timedelta(days=2)))
        .then(pl.lit("Cash"))
        .otherwise(pl.col("group"))
        .alias("group")
    )
    first = run(groups, observed)
    second = run(groups.reverse(), observed.reverse())
    assert first.to_dict() == second.to_dict()
    assert set(first.linked_groups["group"]) == {"A", "B", "Cash"}


def test_signed_fully_funded_weights_are_supported():
    groups, observed = inputs()
    groups = groups.with_columns(pl.Series("portfolio_weight", [1.2, -0.2]))
    observed = observed.with_columns(pl.lit(0.0592).alias("portfolio_return"))
    result = run(groups, observed)
    assert result.summary["active_return"] == pytest.approx(0.0392)
    assert abs(result.summary["reconciliation_residual"]) < 1e-15


@pytest.mark.parametrize(
    "kind",
    [
        "missing_group",
        "missing_period",
        "duplicate",
        "late_weights",
        "late_classification",
        "wrong_start",
        "gap",
        "null",
        "nan",
        "wrong_currency",
        "observed_currency",
        "weight_sum",
        "portfolio_mismatch",
        "benchmark_mismatch",
        "naive",
        "blank_group",
        "bankrupt",
    ],
)
def test_incomplete_or_inconsistent_evidence_is_rejected(kind):
    groups, observed = inputs(2)
    if kind == "missing_group":
        groups = groups.slice(1)
    elif kind == "missing_period":
        groups = groups.head(2)
    elif kind == "duplicate":
        groups = pl.concat([groups, groups.head(1)])
    elif kind in {"late_weights", "late_classification"}:
        name = "weights_at" if kind == "late_weights" else "classification_at"
        groups = groups.with_columns((pl.col(name) + timedelta(microseconds=1)).alias(name))
    elif kind == "wrong_start":
        groups = groups.with_columns((pl.col("period_start") - timedelta(hours=1)))
    elif kind == "gap":
        observed = observed.with_columns((pl.col("period_start") - timedelta(hours=1)))
    elif kind in {"null", "nan"}:
        groups = groups.with_columns(
            pl.lit(None if kind == "null" else float("nan"), dtype=pl.Float64).alias(
                "benchmark_return"
            )
        )
    elif kind == "wrong_currency":
        groups = groups.with_columns(pl.lit("HKD").alias("currency"))
    elif kind == "observed_currency":
        observed = observed.with_columns(pl.lit("HKD").alias("currency"))
    elif kind == "weight_sum":
        groups = groups.with_columns((pl.col("portfolio_weight") * 0.9))
    elif kind in {"portfolio_mismatch", "benchmark_mismatch"}:
        name = kind.removesuffix("_mismatch") + "_return"
        observed = observed.with_columns((pl.col(name) + 0.0001).alias(name))
    elif kind == "naive":
        groups = groups.with_columns(pl.col("weights_at").dt.replace_time_zone(None))
    elif kind == "blank_group":
        groups = groups.with_columns(pl.lit(" ").alias("group"))
    elif kind == "bankrupt":
        groups = groups.with_columns(pl.lit(-1.0).alias("portfolio_return"))
        observed = observed.with_columns(pl.lit(-1.0).alias("portfolio_return"))
    with pytest.raises(ValueError):
        run(groups, observed)


def test_nanosecond_clocks_are_not_rounded_or_dropped():
    groups, observed = inputs(2)
    groups = groups.with_columns(
        pl.col("timestamp", "period_start", "weights_at", "classification_at").dt.cast_time_unit(
            "ns"
        )
        + pl.duration(nanoseconds=1)
    )
    observed = observed.with_columns(
        pl.col("timestamp", "period_start").dt.cast_time_unit("ns") + pl.duration(nanoseconds=1)
    )
    result = run(groups, observed)
    assert result.periods["timestamp"].dt.nanosecond().to_list() == [1, 1]
    assert ".000000001" in result.to_dict()["periods"][0]["timestamp"]
    groups = groups.with_columns(pl.col("weights_at") + pl.duration(nanoseconds=1))
    with pytest.raises(ValueError, match="known by"):
        run(groups, observed)


def test_randomized_multiperiod_identity_against_direct_compounding():
    rng = np.random.default_rng(719)
    for _ in range(20):
        groups, observed = inputs(5)
        rows, totals = [], []
        for i in range(5):
            wp, wb = rng.dirichlet(np.ones(2)), rng.dirichlet(np.ones(2))
            rp, rb = rng.uniform(-0.2, 0.2, 2), rng.uniform(-0.2, 0.2, 2)
            for j, base in enumerate(groups.slice(i * 2, 2).iter_rows(named=True)):
                rows.append(
                    {
                        **base,
                        "portfolio_weight": wp[j],
                        "benchmark_weight": wb[j],
                        "portfolio_return": rp[j],
                        "benchmark_return": rb[j],
                    }
                )
            totals.append(
                {
                    **observed.row(i, named=True),
                    "portfolio_return": float(wp @ rp),
                    "benchmark_return": float(wb @ rb),
                }
            )
        result = run(pl.DataFrame(rows), pl.DataFrame(totals))
        expected = math.prod(1 + r["portfolio_return"] for r in totals) - math.prod(
            1 + r["benchmark_return"] for r in totals
        )
        assert result.summary["active_return"] == pytest.approx(expected, abs=2e-15)
        assert result.linked_groups["total_effect"].sum() == pytest.approx(expected, abs=2e-15)


def test_explicit_cash_group_is_included_without_zero_fill():
    groups, observed = inputs()
    groups = groups.with_columns(
        pl.Series("group", ["Equity", "Cash"]),
        pl.Series("portfolio_return", [0.05, 0.001]),
        pl.Series("benchmark_return", [0.03, 0.001]),
    )
    observed = observed.with_columns(
        pl.lit(0.0304).alias("portfolio_return"), pl.lit(0.0155).alias("benchmark_return")
    )
    assert run(groups, observed).summary["active_return"] == pytest.approx(0.0149)
