from datetime import UTC, date, datetime
import json

from cortexbacktest import (
    BacktestEngine,
    Bar,
    BrokerSimulator,
    CorporateAction,
    CorporateActionType,
    Order,
    Side,
    Strategy,
    event_report,
)
from cortexbacktest.event.financing import CashInterestModel, ShortBorrowFeeModel
from cortexbacktest import FillPrice


class BuyOnce(Strategy):
    def __init__(self) -> None:
        self.submitted = False

    def on_bar(self, bar):
        if self.submitted:
            return []
        self.submitted = True
        return [Order(bar.symbol, bar.timestamp, Side.BUY, 1)]


def _bars(close: float = 10.0):
    high = max(10.0, close)
    return [
        Bar("AAPL", datetime(2024, 1, 1, tzinfo=UTC), 10, high, 10, close),
        Bar("AAPL", datetime(2024, 1, 2, tzinfo=UTC), 11, 11, 11, 11),
    ]


def test_run_manifest_is_stable_and_captures_data_settings_and_metadata():
    engine = BacktestEngine(trade_on_close=True, build_manifest=True,
        broker=BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=1_000),
        run_metadata={"dataset_version": "vendor-2024-01", "seed": 7},
    )

    first = engine.run(_bars(), BuyOnce()).run_manifest
    second = BacktestEngine(trade_on_close=True, build_manifest=True,
        broker=BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=1_000),
        run_metadata={"seed": 7, "dataset_version": "vendor-2024-01"},
    ).run(_bars(), BuyOnce()).run_manifest
    changed = BacktestEngine(trade_on_close=True, build_manifest=True, broker=BrokerSimulator(fill_price=FillPrice.CLOSE, initial_cash=1_000)).run(_bars(10.5), BuyOnce()).run_manifest

    assert first is not None
    assert first["run_id"] == second["run_id"]
    assert first["data_sha256"] != changed["data_sha256"]
    assert first["metadata"] == {"dataset_version": "vendor-2024-01", "seed": 7}
    assert first["engine"]["trade_on_close"] is True
    assert first["broker"]["fill_price"] == "close"
    json.dumps(first)


def test_run_manifest_is_embedded_in_reports_and_record_exports(tmp_path):
    result = BacktestEngine(trade_on_close=True, build_manifest=True, ).run(_bars(), BuyOnce())

    report = event_report(result, include_records=False)
    output = result.records.write_dir(tmp_path)

    assert report["run_manifest"]["run_id"] == result.run_manifest["run_id"]
    written = json.loads((output / "run_manifest.json").read_text())
    assert written["data_sha256"] == result.run_manifest["data_sha256"]


def test_manifest_can_be_disabled_for_throughput_sensitive_runs():
    result = BacktestEngine(trade_on_close=True, build_manifest=False).run(_bars(), BuyOnce())

    assert result.run_manifest is None


class SellOnce(Strategy):
    def __init__(self) -> None:
        self.submitted = False

    def on_bar(self, bar):
        if self.submitted:
            return []
        self.submitted = True
        return [Order(bar.symbol, bar.timestamp, Side.SELL, 100)]


def _flat_bars(days: int = 20) -> list[Bar]:
    return [Bar("AAA", datetime(2024, 1, day + 1, tzinfo=UTC), 100.0, 100.0, 100.0, 100.0) for day in range(days)]


def _run_id(broker: BrokerSimulator, strategy: Strategy) -> str:
    return BacktestEngine(trade_on_close=True, build_manifest=True, broker=broker).run(_flat_bars(), strategy).run_manifest["run_id"]


def test_short_borrow_fee_changes_the_run_id():
    # The fee moves final equity, so it must move the run identity with it.
    baseline = _run_id(BrokerSimulator(fill_price=FillPrice.CLOSE), SellOnce())
    with_fee = _run_id(BrokerSimulator(fill_price=FillPrice.CLOSE, short_borrow_fee_model=ShortBorrowFeeModel(annual_rate=0.5)), SellOnce())

    assert baseline != with_fee


def test_cash_interest_changes_the_run_id():
    baseline = _run_id(BrokerSimulator(fill_price=FillPrice.CLOSE), SellOnce())
    with_interest = _run_id(BrokerSimulator(fill_price=FillPrice.CLOSE, cash_interest_model=CashInterestModel(credit_rate=0.05)), SellOnce())

    assert baseline != with_interest


def test_corporate_actions_change_the_run_id():
    def dividend(amount: float) -> list[CorporateAction]:
        return [CorporateAction("AAA", date(2024, 1, 5), CorporateActionType.CASH_DIVIDEND, amount=amount)]

    baseline = _run_id(BrokerSimulator(fill_price=FillPrice.CLOSE), BuyOnce())
    one = _run_id(BrokerSimulator(fill_price=FillPrice.CLOSE, corporate_actions=dividend(1.0)), BuyOnce())
    two = _run_id(BrokerSimulator(fill_price=FillPrice.CLOSE, corporate_actions=dividend(2.0)), BuyOnce())

    assert len({baseline, one, two}) == 3


def test_corporate_actions_are_recorded_as_a_digest():
    actions = [CorporateAction("AAA", date(2024, 1, 5), CorporateActionType.SPLIT, ratio=2.0)]
    result = BacktestEngine(trade_on_close=True, build_manifest=True, broker=BrokerSimulator(fill_price=FillPrice.CLOSE, corporate_actions=actions)).run(_flat_bars(), BuyOnce())

    record = result.run_manifest["broker"]["corporate_actions"]
    assert record["count"] == 1
    assert len(record["sha256"]) == 64


def test_manifest_omits_corporate_actions_when_unused():
    result = BacktestEngine(trade_on_close=True, build_manifest=True, broker=BrokerSimulator(fill_price=FillPrice.CLOSE)).run(_flat_bars(), BuyOnce())

    broker_record = result.run_manifest["broker"]
    assert broker_record["corporate_actions"] is None
    assert broker_record["cash_interest_model"] is None
    assert broker_record["short_borrow_fee_model"] is None


def test_identical_configuration_reproduces_the_run_id():
    actions = [CorporateAction("AAA", date(2024, 1, 5), CorporateActionType.CASH_DIVIDEND, amount=1.0)]

    first = _run_id(BrokerSimulator(fill_price=FillPrice.CLOSE, corporate_actions=actions), BuyOnce())
    second = _run_id(BrokerSimulator(fill_price=FillPrice.CLOSE, corporate_actions=list(actions)), BuyOnce())

    assert first == second
