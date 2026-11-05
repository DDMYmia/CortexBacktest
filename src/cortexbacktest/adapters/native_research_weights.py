"Reconcile frozen native portfolio/benchmark closing holdings; never renormalise a subset."
from datetime import date
import hashlib
import math

from .. import strict_json


def native_research_weights(files, *, instruments, cutoff):
    load = strict_json.loads

    def number(value, *, positive=False):
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0 or positive and value <= 0:
            raise ValueError("finite nonnegative native holdings required")
        return float(value)

    source_bytes = files["weight-source.json"]
    source = load(source_bytes)
    symbols = {f"INSTRUMENT:{i}" for i in instruments}
    source_ids = source["sourceInstruments"]
    if (not source_ids or any(type(i) is not int or i <= 0 for i in source_ids)
        or len(set(source_ids)) != len(source_ids) or not set(source_ids) <= set(instruments)):
        raise ValueError("native holdings universe must be covered by risk universe")
    source_symbols = {f"INSTRUMENT:{i}" for i in source_ids}
    if (source["contractVersion"] != "native-research-weights-v1" or source["currency"] != "USD"
        or source["accountPortfolio"] is not False or sorted(source["instruments"]) != sorted(instruments)
        or not strict_json.clock(source["snapshotAt"]) <= strict_json.clock(source["knownAt"]) <= cutoff):
        raise ValueError("native research weight scope or clock mismatch")
    names = {f"{role}-{name}" for role in ("portfolio", "benchmark")
             for name in ("run_manifest.json", "positions.ndjson", "equity_curve.ndjson")}
    action_names = {f"{role}-corporate_action_events.ndjson" for role in ("portfolio", "benchmark")}
    if not names <= set(source["artifacts"]) <= names | action_names:
        raise ValueError("complete native portfolio and benchmark artifacts required")
    for name, sha in source["artifacts"].items():
        if hashlib.sha256(files[name]).hexdigest() != sha:
            raise ValueError("native research weight submanifest mismatch")
    weights, manifests, balances = {}, {}, {}
    for role in ("portfolio", "benchmark"):
        manifest = load(files[f"{role}-run_manifest.json"])
        manifests[role] = manifest
        if set(manifest["symbols"]) != source_symbols or len(manifest["symbols"]) != len(source_symbols):
            raise ValueError("native weight universe must match risk universe exactly")
        metadata = manifest["metadata"]
        if ((role == "portfolio" and metadata.get("factor") != source["candidate"])
            or (role == "benchmark" and metadata.get("benchmarkPolicy") != "initial_equal_weight_buy_hold")):
            raise ValueError("native weight strategy or benchmark binding mismatch")
        curve = [load(line) for line in files[f"{role}-equity_curve.ndjson"].splitlines() if line.strip()]
        if not curve:
            raise ValueError("native weight equity curve missing")
        stamps = [strict_json.clock(row["timestamp"]) for row in curve]
        if stamps != sorted(set(stamps)) or stamps[-1] != strict_json.clock(source["snapshotAt"]):
            raise ValueError("native closing snapshot date mismatch")
        closing = curve[-1]
        equity = number(closing["equity"], positive=True)
        cash = number(closing["cash"])
        pending = closing["pending_cash"]
        if type(pending) not in (float, int) or not math.isfinite(pending) or pending < 0:
            raise ValueError("native weights require evidenced nonnegative pending dividends")
        receivable = 0.0
        action_name = f"{role}-corporate_action_events.ndjson"
        if action_name in source["artifacts"]:
            # Trade settlement and derivative cash need their own complete ledger.
            broker = manifest["broker"]
            if (broker.get("settlement_model", {}).get("settings", {}).get("delay_days") != 0
                or broker.get("futures_symbols")):
                raise ValueError("native dividend ledger requires immediate trade settlement and no futures")
            seen_actions = set()
            for line in files[action_name].splitlines():
                if not line.strip():
                    continue
                action = load(line)
                at = strict_json.clock(action["timestamp"])
                key = (action["symbol"], at, action["action_type"])
                if (action["symbol"] not in source_symbols or at > stamps[-1]
                    or key in seen_actions):
                    raise ValueError("native account action scope or clock mismatch")
                seen_actions.add(key)
                amount = number(action["cash_amount"])
                number(action["tax_amount"])
                number(action["quantity_before"])
                number(action["quantity_after"])
                settles = date.fromisoformat(action["settles_on"]) if action["settles_on"] else None
                if settles is not None and settles < at.date():
                    raise ValueError("native account action settlement precedes entitlement")
                if amount and settles is not None and settles > stamps[-1].date():
                    if action["action_type"] != "cash_dividend":
                        raise ValueError("only evidenced dividend receivables supported")
                    receivable += amount
        if not math.isclose(pending, receivable, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError("native pending cash must reconcile to applied unpaid dividend events")
        values = dict.fromkeys(symbols, 0.0)
        seen = set()
        positions = [load(line) for line in files[f"{role}-positions.ndjson"].splitlines() if line.strip()]
        for row in positions:
            symbol = row["symbol"]
            if symbol not in source_symbols or symbol in seen:
                raise ValueError("unknown or duplicate native closing position")
            seen.add(symbol)
            quantity = number(row["quantity"])
            price = number(row["last_price"], positive=True)
            value = number(row["market_value"])
            if not math.isclose(quantity * price, value, rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError("native closing position valuation mismatch")
            values[symbol] = value
        if not math.isclose(math.fsum(values.values()) + cash + receivable, equity, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError("native closing holdings do not reconcile to equity")
        amounts = {**values, "__CASH__": cash}
        if action_names.intersection(source["artifacts"]):
            amounts["__DIVIDEND_RECEIVABLE__"] = receivable
        weights[role] = {s: v / equity for s, v in amounts.items()}
        if action_names.intersection(source["artifacts"]):
            balances[role] = {"settledCash": cash, "dividendReceivable": receivable, "equity": equity,
                              "receivableRiskAssumption": "zero_factor_specific_and_return_research_assumption_not_spendable_cash"}
    for name in ("data_sha256", "start", "end", "symbols", "broker"):
        if manifests["portfolio"][name] != manifests["benchmark"][name]:
            raise ValueError("native portfolio and benchmark run scopes differ")
    result = {"source": source, "sourceSha256": hashlib.sha256(source_bytes).hexdigest(), "weights": weights}
    if balances:
        result["balances"] = balances
    return result
