"Native daily long-only attribution from beginning holdings, actual fills and net equity."

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import re

import polars as pl

from .group_attribution import _reconcile, group_attribution


def native_group_attribution(runs, latest, config):
    "Reconcile every native net interval before Brinson plus explicit adjustments and Carino."
    if (
        not isinstance(config, dict)
        or set(config) != {"currency", "classification_at", "groups"}
        or not isinstance(config["currency"], str)
        or not re.fullmatch(r"[A-Z0-9]{2,8}", config["currency"])
        or not isinstance(config["classification_at"], str)
        or not isinstance(config["groups"], list)
    ):
        raise ValueError(
            "Explicit attribution currency, frozen classification clock and groups required"
        )
    known = datetime.fromisoformat(config["classification_at"])
    if known.tzinfo is None or known.utcoffset() is None:
        raise ValueError("Aware classification clock required")
    known = known.astimezone(timezone.utc)
    symbols = sorted({"INSTRUMENT:" + i for i, _ in latest})
    mapping = {}
    for row in config["groups"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"instrument_id", "group"}
            or type(row["instrument_id"]) is not int
            or not isinstance(row["group"], str)
            or not 1 <= len(row["group"]) <= 64
            or row["group"] != row["group"].strip()
            or row["group"].lower() == "__cash__"
        ):
            raise ValueError("Explicit unique attribution labels required; cash is reserved")
        symbol = "INSTRUMENT:" + str(row["instrument_id"])
        if symbol in mapping:
            raise ValueError("Duplicate attribution instrument")
        mapping[symbol] = row["group"]
    if sorted(mapping) != symbols:
        raise ValueError("Classification must cover every frozen instrument exactly once")
    labels = sorted(set(mapping.values()) | {"__cash__"})
    prices = {
        (datetime.fromisoformat(r["sessionClose"]), "INSTRUMENT:" + r["instrumentId"]): r
        for r in latest.values()
    }
    curves = {
        name: runs[name].performance.equity_curve().sort("timestamp")
        for name in ("qlib", "equal_weight_buy_hold")
    }
    if curves["qlib"].height < 2 or not curves["qlib"]["timestamp"].equals(
        curves["equal_weight_buy_hold"]["timestamp"]
    ):
        raise ValueError("Attribution requires exactly aligned native equity intervals")
    dates = curves["qlib"]["timestamp"].to_list()
    if known > dates[0]:
        raise ValueError("Classification must be known before first attribution interval")
    records = {}
    for name, curve in curves.items():
        if (
            curve["timestamp"].n_unique() != curve.height
            or not curve["equity"].is_finite().all()
            or (curve["equity"] <= 0).any()
            or curve["pending_cash"].abs().max() > 1e-10
            or (curve["cash"] < -1e-8).any()
        ):
            raise ValueError(
                "Positive native equity, settled cash and long-only cash funding required"
            )
        fills = runs[name].records.fills().to_dicts()
        quantities = dict.fromkeys(symbols, 0.0)
        by_time = {}
        for fill in fills:
            if (
                fill["symbol"] not in quantities
                or fill["side"] not in {"buy", "sell"}
                or any(
                    not math.isfinite(fill[k])
                    for k in ("quantity", "price", "commission", "slippage", "notional")
                )
                or fill["quantity"] <= 0
                or fill["price"] <= 0
                or min(fill["commission"], fill["slippage"]) < 0
            ):
                raise ValueError("Complete native long-only fill evidence required")
            signed = fill["quantity"] * (1 if fill["side"] == "buy" else -1)
            if fill["timestamp"] < dates[0]:
                quantities[fill["symbol"]] += signed
            elif fill["timestamp"] not in dates[:-1]:
                raise ValueError("Daily attribution requires fills exactly at interval start")
            else:
                by_time.setdefault(fill["timestamp"], []).append(fill)
        rows = curve.to_dicts()
        series = []
        for start, end in zip(rows[:-1], rows[1:], strict=True):
            at, stamp = start["timestamp"], end["timestamp"]
            equity = start["equity"]
            if any(q < -1e-8 for q in quantities.values()):
                raise ValueError("Short or inconsistent reconstructed positions")
            previous = {s: prices[at, s]["close"] for s in symbols}
            closing = {s: prices[stamp, s]["close"] for s in symbols}
            _reconcile(
                (start["cash"] + math.fsum(quantities[s] * previous[s] for s in symbols)) / equity,
                1.0,
                "Beginning native holdings do not reconcile to equity",
            )
            weights = {g: 0.0 for g in labels}
            gross = {g: 0.0 for g in labels}
            trading = {g: 0.0 for g in labels}
            costs = {g: 0.0 for g in labels}
            cash_effect = {g: 0.0 for g in labels}
            weights["__cash__"] = start["cash"] / equity
            for s in symbols:
                g = mapping[s]
                weights[g] += quantities[s] * previous[s] / equity
                gross[g] += quantities[s] * (closing[s] - previous[s]) / equity
            flows, fees, slippage = [], [], []
            for fill in by_time.get(at, []):
                s = fill["symbol"]
                g = mapping[s]
                sign = 1 if fill["side"] == "buy" else -1
                quantity = fill["quantity"]
                opening = prices[stamp, s]["open"]
                _reconcile(
                    sign * quantity * (fill["price"] - opening),
                    fill["slippage"],
                    "Native fill slippage differs from frozen reference open",
                )
                _reconcile(quantity * fill["price"], fill["notional"], "Fill notional mismatch")
                trading[g] += sign * quantity * (closing[s] - opening) / equity
                costs[g] -= (fill["commission"] + fill["slippage"]) / equity
                flows.append(sign * fill["notional"] + fill["commission"])
                fees.append(fill["commission"])
                slippage.append(fill["slippage"])
                quantities[s] += sign * quantity
                if abs(quantities[s]) < 1e-12:
                    quantities[s] = 0.0
            cash_earned = end["cash"] - start["cash"] + math.fsum(flows)
            cash_effect["__cash__"] = cash_earned / equity
            _reconcile(
                (end["cash"] + math.fsum(quantities[s] * closing[s] for s in symbols))
                / end["equity"],
                1.0,
                "Ending native holdings do not reconcile to equity",
            )
            observed = end["equity"] / equity - 1
            residual = _reconcile(
                math.fsum(
                    [*gross.values(), *trading.values(), *costs.values(), *cash_effect.values()]
                ),
                observed,
                "Native net PnL contributions do not reconcile",
            )
            series.append(
                {
                    "timestamp": stamp,
                    "period_start": at,
                    "observed": observed,
                    "weights": weights,
                    "rates": {g: gross[g] / weights[g] if weights[g] else 0.0 for g in labels},
                    "trading": trading,
                    "costs": costs,
                    "cash": cash_effect,
                    "commissions": math.fsum(fees),
                    "slippage": math.fsum(slippage),
                    "cashEarned": cash_earned,
                    "residual": residual,
                }
            )
        records[name] = series
    groups, adjustments, observed, ledger = [], [], [], []
    for portfolio, benchmark in zip(records["qlib"], records["equal_weight_buy_hold"], strict=True):
        base = {
            "timestamp": portfolio["timestamp"],
            "period_start": portfolio["period_start"],
            "currency": config["currency"],
        }
        observed.append(
            {
                **base,
                "portfolio_return": portfolio["observed"],
                "benchmark_return": benchmark["observed"],
            }
        )
        for g in labels:
            groups.append(
                {
                    **base,
                    "group": g,
                    "weights_at": base["period_start"],
                    "classification_at": known,
                    **{
                        f"{side}_weight": data["weights"][g]
                        for side, data in (("portfolio", portfolio), ("benchmark", benchmark))
                    },
                    **{
                        f"{side}_return": data["rates"][g]
                        for side, data in (("portfolio", portfolio), ("benchmark", benchmark))
                    },
                }
            )
            adjustments.append(
                {
                    **base,
                    "group": g,
                    **{
                        f"{side}_{effect}": data[effect][g]
                        for side, data in (("portfolio", portfolio), ("benchmark", benchmark))
                        for effect in ("trading", "costs", "cash")
                    },
                }
            )
        ledger.append(
            {
                "timestamp": base["timestamp"].isoformat(),
                **{
                    side: {
                        k: data[k] for k in ("commissions", "slippage", "cashEarned", "residual")
                    }
                    for side, data in (("portfolio", portfolio), ("benchmark", benchmark))
                },
            }
        )
    result = group_attribution(
        pl.DataFrame(groups),
        pl.DataFrame(observed),
        currency=config["currency"],
        benchmark_id="native_equal_weight_buy_hold",
        adjustments=pl.DataFrame(adjustments),
    ).to_dict()
    return {
        **result,
        "config": config,
        "status": "available",
        "ledger": ledger,
        "classificationEvidence": "caller_declared_frozen_labels_not_authenticated_PIT_industry_or_FX",
        "zeroWeightGroupReturn": "explicit_zero_rate_only_for_zero_beginning_weight",
        "returnScope": "first_native_equity_snapshot_to_last_initial_deployment_before_first_snapshot_excluded",
        "cashPolicy": "explicit_cash_group_earnings_from_cash_ledger_and_fill_flows",
        "tradePolicy": "actual_fills_at_interval_start_reference_open_and_explicit_costs",
        "classificationSha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def native_snapshot_weights(run, latest):
    "Reconstruct actual closing spot holdings and explicit cash, without target-weight substitutes."
    symbols = sorted({"INSTRUMENT:" + i for i, _ in latest})
    prices = {
        (datetime.fromisoformat(row["sessionClose"]), "INSTRUMENT:" + row["instrumentId"]): row[
            "close"
        ]
        for row in latest.values()
    }
    curve = run.performance.equity_curve().sort("timestamp")
    if (
        not curve.height
        or curve["timestamp"].n_unique() != curve.height
        or curve["equity"].null_count()
        or not curve["equity"].is_finite().all()
        or (curve["equity"] <= 0).any()
        or curve["pending_cash"].abs().max() > 1e-10
        or (curve["cash"] < -1e-8).any()
    ):
        raise ValueError("Complete positive settled long-only equity snapshots required")
    fills = run.records.fills().sort("timestamp").to_dicts()
    quantities = dict.fromkeys(symbols, 0.0)
    index, rows = 0, []
    for snapshot in curve.iter_rows(named=True):
        stamp = snapshot["timestamp"]
        while index < len(fills) and fills[index]["timestamp"] < stamp:
            fill = fills[index]
            if (
                fill["symbol"] not in quantities
                or fill["side"] not in {"buy", "sell"}
                or not math.isfinite(fill["quantity"])
                or fill["quantity"] <= 0
            ):
                raise ValueError("Complete native fill quantities required")
            quantities[fill["symbol"]] += fill["quantity"] * (1 if fill["side"] == "buy" else -1)
            index += 1
        if any(q < -1e-8 for q in quantities.values()):
            raise ValueError("Short holdings unsupported by frozen spot style adapter")
        values = {s: quantities[s] * prices[stamp, s] for s in symbols}
        _reconcile(
            (snapshot["cash"] + math.fsum(values.values())) / snapshot["equity"],
            1.0,
            "Native exposure holdings fail equity reconciliation",
        )
        rows.extend(
            {
                "timestamp": stamp,
                "symbol": s,
                "known_at": stamp,
                "weight": value / snapshot["equity"],
            }
            for s, value in {**values, "__CASH__": snapshot["cash"]}.items()
        )
    if index != len(fills):
        raise ValueError("Fills at or beyond final close cannot define closing snapshot weights")
    return pl.DataFrame(rows).sort("timestamp", "symbol")
