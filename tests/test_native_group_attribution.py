from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import polars as pl
import pytest
from cortexbacktest.reporting.native_attribution import native_group_attribution
from cortexbacktest.reporting.group_attribution import group_attribution


def fixture():
    dates = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(3)]
    prices = [(100.0, 100.0), (110.0, 100.0), (100.0, 100.0)]
    latest = {
        (str(j + 1), str(i)): {
            "instrumentId": str(j + 1),
            "sessionClose": date.isoformat(),
            "open": 100.0 if i < 2 or j == 1 else 110.0,
            "close": values[j],
        }
        for i, (date, values) in enumerate(zip(dates, prices, strict=True))
        for j in range(2)
    }

    def run(equity, cash, fills):
        curve = pl.DataFrame(
            {"timestamp": dates, "equity": equity, "cash": cash, "pending_cash": [0.0] * 3}
        )
        return SimpleNamespace(
            performance=SimpleNamespace(equity_curve=lambda: curve),
            records=SimpleNamespace(fills=lambda: pl.DataFrame(fills)),
        )

    def fill(at, symbol, side, price, commission, slippage):
        return {
            "timestamp": at,
            "symbol": symbol,
            "side": side,
            "quantity": 0.5,
            "price": price,
            "commission": commission,
            "slippage": slippage,
            "notional": 0.5 * price,
        }

    model = run(
        [100.0, 103.48, 102.4948],
        [100.0, 48.48, 102.4948],
        [
            fill(dates[0], "INSTRUMENT:1", "buy", 102.0, 1.0, 1.0),
            fill(dates[1], "INSTRUMENT:1", "sell", 108.0, 1.0, 1.0),
        ],
    )
    benchmark = run(
        [100.0, 100.0, 100.0],
        [100.0, 50.0, 50.0],
        [fill(dates[0], "INSTRUMENT:2", "buy", 100.0, 0.0, 0.0)],
    )
    config = {
        "currency": "USDT",
        "classification_at": "2024-01-01T00:00:00Z",
        "groups": [{"instrument_id": 1, "group": "A"}, {"instrument_id": 2, "group": "B"}],
    }
    return {"qlib": model, "equal_weight_buy_hold": benchmark}, latest, config


def test_rebalance_from_zero_weight_cost_cash_and_carino_reconcile_to_actual_net_equity():
    result = native_group_attribution(*fixture())
    summary = result["summary"]
    assert summary["periods"] == 2
    assert summary["portfolio_return"] == pytest.approx(0.024948)
    assert summary["benchmark_return"] == pytest.approx(0.0)
    assert sum(r["total_effect"] for r in result["linkedGroups"]) == pytest.approx(0.024948)
    assert abs(summary["reconciliation_residual"]) < 1e-12
    first = result["periods"][0]
    assert first["trading"] == pytest.approx(0.05)
    assert first["costs"] == pytest.approx(-0.02)
    assert first["cash"] == pytest.approx(0.0048)
    assert first["active_return"] == pytest.approx(0.0348)
    assert sum(x["portfolio"]["commissions"] for x in result["ledger"]) == 2.0
    assert sum(x["portfolio"]["slippage"] for x in result["ledger"]) == 2.0
    assert [x["portfolio"]["cashEarned"] for x in result["ledger"]] == pytest.approx([0.48, 1.0148])
    assert result["classificationEvidence"].startswith("caller_declared")
    assert not result["qualificationIssued"] and not result["executionAuthorized"]


@pytest.mark.parametrize(
    "patch",
    [
        {"groups": [{"instrument_id": 1, "group": "A"}]},
        {"groups": [{"instrument_id": 1, "group": "A"}, {"instrument_id": 1, "group": "B"}]},
        {"classification_at": "2025-01-02T00:00:00Z"},
        {"classification_at": "2024-01-01T00:00:00"},
        {"groups": [{"instrument_id": 1, "group": "__cash__"}, {"instrument_id": 2, "group": "B"}]},
    ],
)
def test_missing_duplicate_future_or_reserved_classification_rejects(patch):
    runs, latest, config = fixture()
    with pytest.raises(ValueError):
        native_group_attribution(runs, latest, {**config, **patch})


@pytest.mark.parametrize("kind", ["mid_interval", "slippage_mismatch", "net_equity_mismatch"])
def test_native_ledger_gaps_do_not_become_an_attribution_residual_bucket(kind):
    runs, latest, config = fixture()
    run = runs["qlib"]
    if kind == "net_equity_mismatch":
        curve = run.performance.equity_curve().with_columns((pl.col("equity") + 1).alias("equity"))
        run.performance.equity_curve = lambda: curve
    else:
        fills = run.records.fills()
        if kind == "mid_interval":
            fills = fills.with_columns(pl.col("timestamp") + timedelta(hours=1))
        else:
            fills = fills.with_columns(pl.lit(0.0).alias("slippage"))
        run.records.fills = lambda: fills
    with pytest.raises(ValueError):
        native_group_attribution(runs, latest, config)


def test_optional_adjustments_use_actual_net_returns_and_reject_missing_rows():
    at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = at + timedelta(days=1)
    base = {"timestamp": end, "period_start": at, "currency": "USD"}
    groups = pl.DataFrame(
        [
            {
                **base,
                "group": "all",
                "weights_at": at,
                "classification_at": at,
                "portfolio_weight": 1.0,
                "benchmark_weight": 1.0,
                "portfolio_return": 0.01,
                "benchmark_return": 0.005,
            }
        ]
    )
    observed = pl.DataFrame([{**base, "portfolio_return": 0.012, "benchmark_return": 0.004}])
    extra = pl.DataFrame(
        [
            {
                **base,
                "group": "all",
                "portfolio_trading": 0.003,
                "benchmark_trading": 0.0,
                "portfolio_costs": -0.002,
                "benchmark_costs": -0.001,
                "portfolio_cash": 0.001,
                "benchmark_cash": 0.0,
            }
        ]
    )
    result = group_attribution(
        groups, observed, currency="USD", benchmark_id="b", adjustments=extra
    )
    assert result.summary["active_return"] == pytest.approx(0.008)
    assert result.summary["costs"] == pytest.approx(-0.001)
    with pytest.raises(ValueError, match="cover exactly"):
        group_attribution(
            groups, observed, currency="USD", benchmark_id="b", adjustments=extra.with_columns(pl.lit("missing").alias("group"))
        )
