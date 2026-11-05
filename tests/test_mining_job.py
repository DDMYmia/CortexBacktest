from datetime import datetime, timedelta, timezone
import json

import pytest
import polars as pl

from cortexbacktest.pipelines.mining_job import run_mining_job


@pytest.mark.parametrize("portfolio_mode", ["long_short", "long_only"])
@pytest.mark.parametrize("declared_execution", [False, True])
def test_job_splits_only_declared_study_period_and_keeps_warmup(tmp_path, monkeypatch, portfolio_mode, declared_execution):
    from collections import Counter
    from cortexbacktest.research import RevisionPanel

    built = Counter()
    original = RevisionPanel.factor

    def counted(panel, spec, *, last_only=False):
        if panel.decisions.height == 160:
            built[spec.name] += 1
        return original(panel, spec, last_only=last_only)

    monkeypatch.setattr(RevisionPanel, "factor", counted)
    dates = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(160)]
    start = dates[40]
    records = []
    for index, day in enumerate(dates):
        for asset, rate in enumerate((.001, .002, .003, .004)):
            price = 100 * (1 + rate) ** index
            receipt = max(day, start).isoformat()
            records.append({"instrumentId": str(asset + 1), "sessionClose": day.isoformat(),
                            "availableAt": receipt, "recordedAt": receipt, "sourceKnownAt": day.isoformat(),
                            "adjClose": price, "open": price, "high": price, "low": price, "close": price,
                            "volume": 1_000_000., "adjustment": "unadjusted"})
    settings = {"calendar": "crypto_utc", "inSample": .75, "costBps": 10, "mining": {
        "direction": "momentum", "budget": 2, "seed": 7, "horizon": 1,
        "trainSessions": 30, "testSessions": 30, "studyStart": start.isoformat(), "portfolioMode": portfolio_mode}}
    exported = {"settings": settings, "bars": {"sha256": "a" * 64}, "asOf": dates[-1].isoformat()}
    if declared_execution:
        settings["mining"]["rebalance"] = {"cash_buffer": .002, "min_trade_value": 0.,
            "allow_fractional": True, "tag": "rebalance",
            "quantity_steps": [[f"INSTRUMENT:{i}", "0.03"] for i in range(1, 5)]}
        if portfolio_mode == 'long_only':
            settings['mining']['benchmark'] = 'initial_equal_weight_buy_hold'
    result = run_mining_job({"records": records}, exported, tmp_path / "study")
    assert set(built) == {trial['candidate'] for trial in result['trials']}
    assert all(count == 1 for count in built.values())
    manifest = json.loads((tmp_path / "study/selection/manifest.json").read_text())
    assert manifest["inner_end"] == dates[129].isoformat()  # 40 warm-up + 90 study sessions.
    assert manifest["settings"]["study_start"] == start.isoformat()
    assert manifest["settings"]["portfolio_mode"] == portfolio_mode
    if declared_execution:
        from cortexbacktest.pipelines.factor_strategy import FrozenFactorStrategy
        from cortexbacktest.event import RebalanceConfig

        assert manifest['settings']['rebalance'] == settings['mining']['rebalance']
        strategy = FrozenFactorStrategy.from_checkpoint(tmp_path / 'study/selection',
            selection_sha256=result['manifestSha256'], factor=result['selected'][0],
            symbols=tuple(f'INSTRUMENT:{i}' for i in range(1, 5)))
        assert strategy.rebalance.allow_fractional is True
        assert strategy.rebalance.cash_buffer == .002
        with pytest.raises(ValueError, match='frozen research execution policy'):
            FrozenFactorStrategy.from_checkpoint(tmp_path / 'study/selection',
                selection_sha256=result['manifestSha256'], factor=result['selected'][0],
                symbols=strategy.symbols, rebalance=RebalanceConfig())
    assert {key: result["pit"][key] for key in ("warmupSessions", "studySessions", "innerSessions", "outerSessions")} == {
        "warmupSessions": 40, "studySessions": 120, "innerSessions": 90, "outerSessions": 30}
    assert result["selected"] and result["execution"]["status"] == "evaluated"
    family = result["returnFamily"]
    assert family["candidates"] == list(dict.fromkeys(row["candidate"] for row in result["trials"]))
    assert family["verdict"]["family"]["complete"] is True
    assert (tmp_path / "study/trial-returns.parquet").exists()
    if declared_execution and portfolio_mode == 'long_only':
        benchmark = family['benchmark']
        assert manifest['settings']['benchmark'] == settings['mining']['benchmark']
        assert benchmark['policy'] == settings['mining']['benchmark']
        assert benchmark['excessVerdict']['family']['trials'] == len(result['trials'])
        returns = pl.read_parquet(tmp_path / 'study/trial-returns.parquet')
        baseline = pl.read_parquet(tmp_path / 'study/benchmark-returns.parquet')
        assert returns['date'].equals(baseline['date'])
    else:
        assert 'benchmark' not in family
    assert result["pit"]["priceCoverage"] == {"requestedAssets": 4,
        "inner": {"dates": 90, "datesWithTwoPrices": 90, "datesWithAllPrices": 90},
        "outer": {"dates": 30, "datesWithTwoPrices": 30, "datesWithAllPrices": 30}}
    for trial in result["trials"]:
        assert [datetime.fromisoformat(fold["start"]) for fold in trial["folds"]] == [dates[70], dates[100]]
    # A timely invalidation masks the old version. A later invalidation must
    # not rewrite own-decision coverage, and extra versions are not extra assets.
    for row in records:
        row["invalidReason"] = None
    for index, receipt in [(280, dates[70]), (281, dates[70]), (282, dates[70]), (560, dates[159])]:
        invalid = {**records[index], "invalidReason": "unverified_price_basis",
                   "recordedAt": (receipt + timedelta(minutes=1)).isoformat(),
                   "availableAt": (receipt + timedelta(minutes=1)).isoformat()}
        for field in ("adjClose", "open", "high", "low", "close", "volume"):
            invalid[field] = None
        records.append(invalid)
    records.append({**records[280], "recordedAt": (dates[70] + timedelta(seconds=1)).isoformat(),
                    "availableAt": (dates[70] + timedelta(seconds=1)).isoformat()})
    settings["mining"]["decisionDelayMinutes"] = 30
    exported["asOf"] = (dates[-1] + timedelta(hours=1)).isoformat()
    changed = run_mining_job({"records": records}, exported, tmp_path / "invalidated")
    assert changed["pit"]["priceCoverage"] == {"requestedAssets": 4,
        "inner": {"dates": 90, "datesWithTwoPrices": 89, "datesWithAllPrices": 89},
        "outer": {"dates": 30, "datesWithTwoPrices": 30, "datesWithAllPrices": 30}}
    # A start leaving insufficient inner windows must fail before freezing.
    settings["mining"]["studyStart"] = dates[150].isoformat()
    with pytest.raises(ValueError, match="Inner validation"):
        run_mining_job({"records": records}, exported, tmp_path / "too_short")
    assert not (tmp_path / "too_short/selection/manifest.json").exists()


@pytest.mark.parametrize("late", [False, True])
# Delayed equity execution needs a corporate-action price basis, so only crypto is delayed.
@pytest.mark.parametrize(("calendar", "delay"), [("us_equity", 0), ("crypto_utc", 0), ("crypto_utc", 30)])
@pytest.mark.parametrize("revised", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_receipt_eligibility_controls_actual_mining_and_outer_backtest(tmp_path, late, calendar, revised, delay, invalid):
    records = []
    for day in range(120):
        close = datetime(2024, 1, 1, 0 if calendar == "crypto_utc" else 21, tzinfo=timezone.utc) + timedelta(days=day)
        for asset, rate in enumerate((.001, .002, .003, .004)):
            price = 100 * (1 + rate) ** day
            records.append({"instrumentId": str(asset + 1), "sessionClose": close.isoformat(),
                            "availableAt": (close + timedelta(days=365) if late else close + timedelta(minutes=15 if delay else 0)).isoformat(),
                            "recordedAt": (close + timedelta(days=365) if late else close + timedelta(minutes=15 if delay else 0)).isoformat(),
                            "sourceKnownAt": close.isoformat(),
                            "adjClose": price, "close": price, "open": price * .995, "high": price * 1.01,
                            "low": price * .99, "volume": 1_000_000., "adjustment": "unadjusted"})
    if revised:
        correction = {**records[40], "recordedAt": records[-1]["recordedAt"], "availableAt": records[-1]["availableAt"]}
        for field in ("adjClose", "close", "open", "high", "low"):
            correction[field] *= 1.1
        records.append(correction)
    if invalid:
        for row in records:
            row["invalidReason"] = None
        withdrawal = {**records[44], "invalidReason": "unverified_price_basis",
                      "recordedAt": records[479]["recordedAt"], "availableAt": records[479]["availableAt"]}
        for field in ("adjClose", "close", "open", "high", "low", "volume"):
            withdrawal[field] = None
        records.append(withdrawal)
    settings = {"calendar": calendar, "inSample": .75, "costBps": 10, "mining": {
        "direction": "momentum", "budget": 4, "seed": 7, "horizon": 1,
        "trainSessions": 30, "testSessions": 30, "decisionDelayMinutes": delay}}
    result = run_mining_job({"records": records}, {"settings": settings, "bars": {"sha256": "a" * 64},
                           "asOf": (close + timedelta(hours=1)).isoformat()}, tmp_path)
    assert len(result["trials"]) == 4
    assert result["pit"]["eligibleRecords"] == (0 if late else 480)
    assert result["pit"]["records"] == 480
    assert result["pit"]["priceCoverage"] == {"requestedAssets": 4,
        "inner": {"dates": 90, "datesWithTwoPrices": 0 if late else 90, "datesWithAllPrices": 0 if late else 90},
        "outer": {"dates": 30, "datesWithTwoPrices": 0 if late else 30, "datesWithAllPrices": 0 if late else 30}}
    assert result["pit"]["revisionRecords"] == 480 + revised + invalid
    assert result["pit"]["invalidRevisionRecords"] == int(invalid)
    assert bool(result["holdout"]) is not late
    assert bool(result["selected"]) is not late
    assert [entry["candidate"] for entry in result["holdout"]] == result["selected"]
    if late:
        assert {r["status"] for r in result["trials"]} == {"insufficient_evidence"}
        assert result["execution"] == {"status": "no_selected_factors", "results": []}
    elif calendar == "us_equity":
        assert result["execution"] == {"status": "requires_corporate_actions", "results": []}
    elif invalid:
        assert result["execution"] == {"status": "requires_complete_execution_prices", "results": []}
    else:
        assert result["execution"]["status"] == "evaluated"
        assert [entry["candidate"] for entry in result["execution"]["results"]] == result["selected"]
        assert all(entry["report"]["summary"]["fills"] > 0 for entry in result["execution"]["results"])
        assert result["execution"]["commissionBps"] == settings["costBps"]
        assert result["execution"]["fillPolicy"] == "first_open_after_decision"
        assert result["execution"]["targetPolicy"] == "decision_time_v1"
        import hashlib
        import json

        for entry in result["execution"]["results"]:
            directory = tmp_path / "simulation" / hashlib.sha256(entry["candidate"].encode()).hexdigest()
            assert set(entry["artifacts"]) == {"equity_curve.ndjson", "orders.ndjson", "fills.ndjson",
                "open_orders.ndjson", "positions.ndjson", "run_manifest.json", "corporate_action_events.ndjson"}
            for filename, digest in entry["artifacts"].items():
                assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == digest
            manifest = json.loads((directory / "run_manifest.json").read_bytes())
            assert manifest["metadata"]["selectionSha256"] == result["manifestSha256"]
            from cortexbacktest.pipelines import FrozenFactorStrategy

            strategy = FrozenFactorStrategy.from_bytes(manifest["metadata"]["strategyArtifact"].encode(),
                                                       expected_sha256=manifest["metadata"]["strategySha256"])
            assert strategy.factor == entry["candidate"]
            assert strategy.selection_sha256 == result["manifestSha256"]
            assert strategy.symbols == tuple(f"INSTRUMENT:{i}" for i in range(1, 5))
            assert (directory / "fills.ndjson").read_text().strip()
    assert (tmp_path / "selection/manifest.json").exists()
