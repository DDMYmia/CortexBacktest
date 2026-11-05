from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import json

import pytest
from jsonschema.exceptions import ValidationError

from cortexbacktest import BacktestEngine, BrokerSimulator
from cortexbacktest.event.manifest import engine_code_digest
from cortexbacktest.reporting.shadow_delivery import write_shadow_decision
from cortexbacktest.shadow import (
    PortfolioObservation, StrategyArtifact, evaluate_shadow, load_artifact_bars,
    load_warmup_history,
)
from cortexbacktest.strategy_library import SmaCross


SYMBOL = "INSTRUMENT:1"
START = datetime(2026, 9, 1, 20, tzinfo=UTC)


def artifact(**kwargs):
    return StrategyArtifact(engine_code_digest(), (SYMBOL,), "sma_cross", {"fast": 2, "slow": 3}, 2.25,
                            **kwargs)


def market_file(tmp_path, prices=(10, 11, 12, 11, 10, 9, 10, 11)):
    rows = []
    for i, price in enumerate(prices):
        event = START + timedelta(days=i)
        available = event + timedelta(seconds=1)
        rows.append({
            "contractVersion": "market-bars-v2", "instrumentId": "1",
            "sessionDate": event.date().isoformat(), "adjustment": "unadjusted",
            "ohlcBasis": "unadjusted", "source": "shadow-test",
            "adjustmentProvenance": "test-only", "sessionClose": event.isoformat(),
            "sourceKnownAt": available.isoformat(), "recordedAt": available.isoformat(),
            "availableAt": available.isoformat(), "open": price, "high": price + 1,
            "low": price - 1, "close": price, "adjClose": price, "volume": 1000,
        })
    return write_rows(tmp_path, rows)


def write_rows(tmp_path, rows):
    cutoff = max(datetime.fromisoformat(r["availableAt"]) for r in rows)
    doc = {
        "contractVersion": "observation-file-v1", "recordVersion": "market-bars-v2",
        "asOf": cutoff.isoformat(), "recordCount": len(rows),
        "records": [{**r, "asOf": cutoff.isoformat()} for r in rows],
        "scope": {"source": "shadow-test", "adjustment": "unadjusted",
                  "start": min(r["sessionDate"] for r in rows),
                  "end": max(r["sessionDate"] for r in rows),
                  "sessions": [{k: r[k] for k in ("instrumentId", "sessionDate", "sessionClose")}
                               for r in rows]},
    }
    raw = json.dumps(doc).encode()
    path = tmp_path / f"{sha256(raw).hexdigest()}.json"
    path.write_bytes(raw)
    return path, sha256(raw).hexdigest(), cutoff


def portfolio(now, quantity="0", **kwargs):
    return PortfolioObservation("test-account", "a" * 64, now,
                                ((SYMBOL, Decimal(quantity)),), (), **kwargs)


def evaluate(art, inputs, account=None, now=None, warmup=None):
    path, digest, cutoff = inputs
    return evaluate_shadow(art, path, digest=digest, as_of=cutoff,
                           portfolio=account or portfolio(cutoff), now=now or cutoff, warmup=warmup)


def test_artifact_roundtrip_immutable_and_manifest_binding(tmp_path):
    art = artifact()
    loaded = StrategyArtifact.from_bytes(art.to_bytes(), expected_sha256=art.sha256)
    assert loaded == art
    with pytest.raises(FrozenInstanceError):
        art.family = "rsi_reversion"
    inputs = market_file(tmp_path)
    _, bars = load_artifact_bars(loaded, inputs[0], digest=inputs[1], as_of=inputs[2])
    result = BacktestEngine(broker=BrokerSimulator(initial_cash=10000), trade_on_close=False,
                            build_manifest=True).run(bars, loaded.backtest_strategy())
    config = result.run_manifest["strategy_config"]
    assert config["parameters"]["artifact"] == {"sha256": art.sha256, **art.document()}
    assert result.fill_count > 0


@pytest.mark.parametrize("field,value", [
    ("engineCodeSha256", "0" * 64), ("artifactVersion", "arbitrary-python"),
    ("symbols", [SYMBOL, SYMBOL]), ("symbols", ["AAPL"]),
    ("maxSignalAgeSeconds", 0), ("maxAccountAgeSeconds", True),
    ("module", "os.system"),
])
def test_artifact_rejects_changed_runtime_versions_scope_or_extra_keys(field, value):
    doc = artifact().document()
    doc[field] = value
    content = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError):
        StrategyArtifact.from_bytes(content, expected_sha256=sha256(content).hexdigest())


@pytest.mark.parametrize("field,value", [
    ("parameters", {"fast": True, "slow": 3}), ("parameters", {"fast": 3, "slow": 3}),
    ("parameters", {"fast": 2, "slow": 3, "quantity": 1}), ("family", "unknown"),
    ("gate", {"source": "factor"}), ("quantity", True),
    ("quantity", "2"), ("quantity", float("nan")), ("quantity", float("inf")),
    ("quantity", 0), ("quantity", 1e-20), ("quantity", 1e100),
])
def test_invalid_parameters_rejected(field, value):
    with pytest.raises(ValueError):
        replace(artifact(), **{field: value})


def test_digest_and_duplicate_json_keys_rejected():
    content = artifact().to_bytes()
    with pytest.raises(ValueError, match="digest mismatch"):
        StrategyArtifact.from_bytes(content + b" ", expected_sha256=artifact().sha256)
    content = content.replace(b'"fast":2', b'"fast":2,"fast":2')
    with pytest.raises(ValueError, match="canonical"):
        StrategyArtifact.from_bytes(content, expected_sha256=sha256(content).hexdigest())


def test_shadow_uses_actual_fractional_holdings_without_a_broker(tmp_path, monkeypatch):
    art = artifact()
    inputs = market_file(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Shadow touched a broker boundary")

    monkeypatch.setattr(BrokerSimulator, "__init__", forbidden)
    monkeypatch.setattr(SmaCross, "broker", property(forbidden))
    result = evaluate(art, inputs, portfolio(inputs[2], "0.123456789012"))
    assert result["executionAuthorized"] is False
    assert result["decisions"][0]["quantity"] == "2.126543210988"
    assert result["decisions"][0]["side"] == "BUY"
    assert result == evaluate(art, inputs, portfolio(inputs[2], "0.123456789012"))
    assert result["inputSha256"] == inputs[1]
    assert result["strategySha256"] == art.sha256


def test_actual_backtest_order_decisions_match_shadow_at_every_warmed_prefix(tmp_path):
    art = artifact()
    inputs = market_file(tmp_path)
    doc, bars = load_artifact_bars(art, inputs[0], digest=inputs[1], as_of=inputs[2])
    comparisons = []

    strategy = art.backtest_strategy()
    inner = strategy.on_bars

    def compare(current):
        orders = inner(current)
        prefix = [r for r in doc["records"]
                  if datetime.fromisoformat(r["availableAt"]) <= current[0].timestamp]
        if len(prefix) >= art.parameters["slow"]:
            assert not strategy.broker.order_manager.open_orders_for_symbol(SYMBOL)
            position = strategy.broker.state.positions.get(SYMBOL)
            quantity = str(position.quantity if position else 0)
            frozen = write_rows(tmp_path, prefix)
            shadow = evaluate(art, frozen, portfolio(current[0].timestamp, quantity))
            proposed = [(d["side"], Decimal(d["quantity"])) for d in shadow["decisions"]
                        if d["side"] != "HOLD"]
            assert proposed == [(o.side.value.upper(), Decimal(str(o.quantity))) for o in orders]
            comparisons.append(proposed)
        return orders

    strategy.on_bars = compare
    result = BacktestEngine(broker=BrokerSimulator(initial_cash=10000),
                            trade_on_close=False).run(bars, strategy)
    assert len(comparisons) == len(bars) - art.parameters["slow"] + 1
    assert any(p and p[0][0] == "BUY" for p in comparisons)
    assert any(p and p[0][0] == "SELL" for p in comparisons)
    assert [] in comparisons
    assert result.fill_count > 0


@pytest.mark.parametrize("change", ["stale", "future", "naive", "missing", "pending",
                                    "nan", "precision", "digest"])
def test_invalid_account_observation_rejected(tmp_path, change):
    inputs = market_file(tmp_path)
    account = portfolio(inputs[2])
    changes = {
        "stale": {"event_time": inputs[2] - timedelta(seconds=60)},
        "future": {"event_time": inputs[2] + timedelta(seconds=1)},
        "naive": {"event_time": inputs[2].replace(tzinfo=None)},
        "missing": {"positions": ()}, "pending": {"pending_symbols": (SYMBOL,)},
        "nan": {"positions": ((SYMBOL, Decimal("NaN")),)},
        "precision": {"positions": ((SYMBOL, Decimal("0.0000000000001")),)},
        "digest": {"source_sha256": "wrong"},
    }
    with pytest.raises(ValueError):
        evaluate(artifact(), inputs, replace(account, **changes[change]))


def test_stale_market_rejected_even_with_fresh_receipt(tmp_path):
    inputs = market_file(tmp_path)
    rows = json.loads(inputs[0].read_bytes())["records"]
    received = inputs[2] + timedelta(seconds=300)
    rows[-1].update(recordedAt=received.isoformat(), availableAt=received.isoformat())
    with pytest.raises(ValueError, match="Stale market"):
        evaluate(artifact(), write_rows(tmp_path, rows))


def test_expiry_boundary_future_cutoff_and_warmup(tmp_path):
    inputs = market_file(tmp_path)
    with pytest.raises(ValueError, match="Future data cutoff"):
        evaluate(artifact(), inputs, now=inputs[2] - timedelta(seconds=1))
    now = inputs[2] + timedelta(seconds=299)
    with pytest.raises(ValueError, match="Stale market"):
        evaluate(artifact(), inputs, portfolio(now), now=now)
    with pytest.raises(ValueError, match="warmup"):
        evaluate(artifact(), market_file(tmp_path, (10, 11)))


def test_old_receipt_order_and_scope_rejected(tmp_path):
    inputs = market_file(tmp_path)
    rows = json.loads(inputs[0].read_bytes())["records"]
    rows[0].update(recordedAt=rows[-1]["availableAt"], availableAt=rows[-1]["availableAt"])
    with pytest.raises(ValueError, match="increase"):
        evaluate(artifact(), write_rows(tmp_path, rows))
    other = replace(artifact(), symbols=("INSTRUMENT:2",))
    with pytest.raises(ValueError, match="scope"):
        load_artifact_bars(other, inputs[0], digest=inputs[1], as_of=inputs[2])
    _, bars = load_artifact_bars(artifact(), inputs[0], digest=inputs[1], as_of=inputs[2])
    with pytest.raises(ValueError, match="scope"):
        other.backtest_strategy().on_bars(bars[:1])


def test_logical_signal_id_stable_across_receipts_but_payload_changes(tmp_path):
    inputs = market_file(tmp_path)
    art = artifact()
    first = evaluate(art, inputs)
    changed = evaluate(art, inputs, replace(portfolio(inputs[2], "1"), source_sha256="b" * 64))
    assert first["decisions"][0]["signalId"] == changed["decisions"][0]["signalId"]
    assert first["decisionSha256"] != changed["decisionSha256"]
    different_strategy = evaluate(replace(art, quantity=3), inputs)
    assert first["decisions"][0]["signalId"] != different_strategy["decisions"][0]["signalId"]


def test_modified_market_file_rejected(tmp_path):
    inputs = market_file(tmp_path)
    inputs[0].write_bytes(inputs[0].read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest"):
        evaluate(artifact(), inputs)


def test_same_tolerance_as_backtest_and_no_silent_float_rounding(tmp_path):
    inputs = market_file(tmp_path)
    result = evaluate(artifact(), inputs, portfolio(inputs[2], "2.249999999999"))
    assert result["decisions"][0]["side"] == "HOLD"
    assert result["decisions"][0]["quantity"] == "0"
    with pytest.raises(ValueError, match="round-trip"):
        evaluate(artifact(), inputs, portfolio(inputs[2], "1234567890123456789"))


def test_multi_instrument_scope_rejects_one_stale_or_unwarmed_instrument(tmp_path):
    inputs = market_file(tmp_path)
    rows = json.loads(inputs[0].read_bytes())["records"]
    other = [{**r, "instrumentId": "2"} for r in rows]
    art = replace(artifact(), symbols=(SYMBOL, "INSTRUMENT:2"))
    account = replace(portfolio(inputs[2]),
                      positions=((SYMBOL, Decimal(0)), ("INSTRUMENT:2", Decimal("2.25"))))
    full = evaluate(art, write_rows(tmp_path, rows + other), account)
    assert [d["side"] for d in full["decisions"]] == ["BUY", "HOLD"]
    with pytest.raises(ValueError, match="Stale market"):
        evaluate(art, write_rows(tmp_path, rows + other[:-1]), account)
    with pytest.raises(ValueError, match="warmup"):
        evaluate(art, write_rows(tmp_path, rows + other[-2:]), account)


def batched_warmup(tmp_path, art, equal_receipts=False):
    inputs = market_file(tmp_path)
    rows = json.loads(inputs[0].read_bytes())["records"]
    history = rows[:3]
    boundary = datetime.fromisoformat(history[-1]["sessionClose"]) + timedelta(seconds=10)
    for i, row in enumerate(history):
        # Receipt order is the reverse of market time, as in a batch backfill.
        receipt = boundary - timedelta(seconds=0 if equal_receipts else i)
        row.update(recordedAt=receipt.isoformat(), availableAt=receipt.isoformat())
    frozen = write_rows(tmp_path, history)
    warmup = load_warmup_history(art, frozen[0], digest=frozen[1], as_of=frozen[2])
    return warmup, rows[3:], frozen


@pytest.mark.parametrize("equal_receipts", [False, True])
def test_batch_warmup_uses_event_order_preserves_receipts_and_never_trades(tmp_path, equal_receipts):
    art = artifact()
    warmup, _, frozen = batched_warmup(tmp_path, art, equal_receipts)
    original = frozen[0].read_bytes()
    strategy = art.backtest_strategy(warmup)
    assert strategy.observe_closes([]) == {SYMBOL: 2.25}
    assert [c[3] for c in warmup.closes] == [10, 11, 12]
    assert warmup.closes[0][2] >= warmup.closes[1][2] >= warmup.closes[2][2]
    assert all(c[1] <= c[2] <= warmup.as_of for c in warmup.closes)
    broker = BrokerSimulator(initial_cash=10000)
    result = BacktestEngine(broker=broker, build_manifest=True).run([], strategy)
    assert result.fill_count == broker.submitted_order_count == 0
    assert not broker.state.positions
    assert result.run_manifest["strategy_config"]["parameters"]["warmup"] == warmup.descriptor()
    assert frozen[0].read_bytes() == original


def test_warmed_backtest_and_shadow_match_for_each_new_market_event(tmp_path):
    art = artifact()
    warmup, rows, _ = batched_warmup(tmp_path, art)
    stream = write_rows(tmp_path, rows)
    _, bars = load_artifact_bars(art, stream[0], digest=stream[1], as_of=stream[2], warmup=warmup)
    comparisons = []

    strategy = art.backtest_strategy(warmup)
    inner = strategy.on_bars

    def compare(current):
        orders = inner(current)
        prefix = [r for r in rows if datetime.fromisoformat(r["availableAt"]) <= current[0].timestamp]
        position = strategy.broker.state.positions.get(SYMBOL)
        account = portfolio(current[0].timestamp, str(position.quantity if position else 0))
        shadow = evaluate(art, write_rows(tmp_path, prefix), account, warmup=warmup)
        proposed = [(d["side"], Decimal(d["quantity"])) for d in shadow["decisions"]
                    if d["side"] != "HOLD"]
        assert proposed == [(o.side.value.upper(), Decimal(str(o.quantity))) for o in orders]
        assert shadow["warmup"] == warmup.descriptor()
        comparisons.append(proposed)
        return orders

    strategy.on_bars = compare
    result = BacktestEngine(broker=BrokerSimulator(initial_cash=10000), trade_on_close=False,
                            build_manifest=True).run(bars, strategy)
    assert len(comparisons) == len(rows)
    assert all(fill.timestamp > warmup.as_of for fill in result.fills)
    assert result.fill_count > 0
    assert any(p and p[0][0] == "BUY" for p in comparisons)
    assert any(p and p[0][0] == "SELL" for p in comparisons)


def test_future_stream_changes_do_not_change_earlier_fills(tmp_path):
    art = artifact()
    warmup, rows, _ = batched_warmup(tmp_path, art)

    def run(source):
        path, digest, as_of = write_rows(tmp_path, source)
        _, bars = load_artifact_bars(art, path, digest=digest, as_of=as_of, warmup=warmup)
        return BacktestEngine(broker=BrokerSimulator(initial_cash=10000),
                              trade_on_close=False).run(bars, art.backtest_strategy(warmup))

    first = run(rows)
    cutoff = datetime.fromisoformat(rows[-2]["availableAt"])
    changed = [*rows[:-1], {**rows[-1], "open": 100, "close": 100,
                          "high": 101, "low": 99, "adjClose": 100}]
    second = run(changed)
    assert first.fills
    assert [f for f in first.fills if f.timestamp <= cutoff] == [
        f for f in second.fills if f.timestamp <= cutoff]


def test_stream_cannot_reuse_warmup_or_trade_before_boundary(tmp_path):
    art = artifact()
    warmup, rows, frozen = batched_warmup(tmp_path, art)
    with pytest.raises(ValueError, match="warmup boundary"):
        load_artifact_bars(art, frozen[0], digest=frozen[1], as_of=frozen[2], warmup=warmup)
    history_row = json.loads(frozen[0].read_bytes())["records"][-1]
    later = datetime.fromisoformat(rows[0]["availableAt"])
    history_row.update(recordedAt=later.isoformat(), availableAt=later.isoformat())
    with pytest.raises(ValueError, match="warmup boundary"):
        evaluate(art, write_rows(tmp_path, [history_row]), warmup=warmup)
    source = write_rows(tmp_path, rows)
    _, bars = load_artifact_bars(art, source[0], digest=source[1], as_of=source[2])
    with pytest.raises(ValueError, match="warmup boundary"):
        art.backtest_strategy(warmup).on_bars([bars[0]._replace(timestamp=warmup.as_of)])


def test_warmup_tampering_future_receipts_scope_and_artifact_mismatch(tmp_path):
    art = artifact()
    warmup, rows, frozen = batched_warmup(tmp_path, art)
    with pytest.raises(ValueError, match="different strategy"):
        replace(art, quantity=3).backtest_strategy(warmup)
    with pytest.raises(ValueError, match="different strategy"):
        evaluate(replace(art, quantity=3), write_rows(tmp_path, rows), warmup=warmup)
    with pytest.raises(ValueError, match="known by"):
        replace(warmup, as_of=warmup.as_of - timedelta(seconds=1))
    with pytest.raises(ValueError, match="scope"):
        bad = replace(warmup, closes=tuple(("INSTRUMENT:2", *c[1:]) for c in warmup.closes))
        art.backtest_strategy(bad)
    with pytest.raises(ValueError, match="increase"):
        replace(warmup, closes=warmup.closes[::-1])
    frozen[0].write_bytes(frozen[0].read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest"):
        load_warmup_history(art, frozen[0], digest=frozen[1], as_of=frozen[2])


def test_warmup_provenance_changes_run_and_decision_identity(tmp_path):
    art = artifact()
    warmup, rows, frozen = batched_warmup(tmp_path, art)
    original = json.loads(frozen[0].read_bytes())["records"]
    original[0].update(open=9, high=10, low=8, close=9, adjClose=9)
    alternate_file = write_rows(tmp_path, original)
    alternate = load_warmup_history(art, alternate_file[0], digest=alternate_file[1],
                                    as_of=alternate_file[2])
    stream = write_rows(tmp_path, rows)
    a = evaluate(art, stream, warmup=warmup)
    b = evaluate(art, stream, warmup=alternate)
    assert a["decisions"][0]["signalId"] == b["decisions"][0]["signalId"]
    assert a["decisionSha256"] != b["decisionSha256"]
    results = [BacktestEngine(build_manifest=True).run([], art.backtest_strategy(w))
               for w in (warmup, alternate)]
    assert results[0].run_manifest["run_id"] != results[1].run_manifest["run_id"]


def test_shadow_delivery_is_immutable_diagnostic_contract(tmp_path):
    decision = evaluate(artifact(), market_file(tmp_path))
    target = tmp_path / "decision.json"
    receipt = write_shadow_decision(decision, target)
    assert receipt["sha256"] == sha256(target.read_bytes()).hexdigest()
    assert json.loads(target.read_bytes())["executionAuthorized"] is False
    assert target.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        write_shadow_decision(decision, target)
    decision["decisionSha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        write_shadow_decision(decision, tmp_path / "bad.json")
    decision["executionAuthorized"] = True
    with pytest.raises(ValidationError):
        write_shadow_decision(decision, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


def test_any_library_family_decides_in_the_shadow_run(tmp_path):
    art = StrategyArtifact(engine_code_digest(), (SYMBOL,), "buy_hold", {}, 1.5)
    decision = evaluate(art, market_file(tmp_path))["decisions"][0]
    assert (decision["side"], decision["targetQuantity"]) == ("BUY", "1.5")


def test_a_closed_gate_holds_the_target_flat(tmp_path):
    from cortexbacktest.gates import SeriesGate
    gate = {"source": "factor", "name": "news_sentiment:fmp:X", "rule": "positive", "lookback": 1}
    art = StrategyArtifact(engine_code_digest(), (SYMBOL,), "buy_hold", {}, 1.5, gate)
    inputs = market_file(tmp_path)
    negative = SeriesGate([(START, -1.0)], "positive", 1)
    positive = SeriesGate([(START, 1.0)], "positive", 1)
    held = portfolio(inputs[2], "1.5")
    closed = evaluate_shadow(art, inputs[0], digest=inputs[1], as_of=inputs[2], portfolio=held, now=inputs[2], gate=negative)
    opened = evaluate_shadow(art, inputs[0], digest=inputs[1], as_of=inputs[2], portfolio=held, now=inputs[2], gate=positive)
    assert closed["decisions"][0]["side"] == "SELL" and opened["decisions"][0]["side"] == "HOLD"
    with pytest.raises(ValueError, match="gate series"):
        evaluate(art, inputs)
