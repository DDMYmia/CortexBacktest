from copy import replace
from datetime import UTC, datetime, timedelta

import pytest

from cortexbacktest import Bar, ContractRoll
from cortexbacktest.pipelines import run_futures_roll_backtest


START = datetime(2026, 1, 1, tzinfo=UTC)


def data(*, volume=100):
    return [Bar(symbol, START + timedelta(days=i), price, price, price, price, volume=volume)
            for i in range(6) for symbol, price in [("F1", 100+i), ("F2", 110+i)]]


def run(bars=None, **kwargs):
    return run_futures_roll_backtest(data() if bars is None else bars,
        [ContractRoll(START + timedelta(days=2), "F1", "F2")],
        contract_multipliers={"F1": 10, "F2": 10}, initial_cash=10000, **kwargs)


@pytest.mark.parametrize("quantity", [1, -1])
def test_roll_creates_real_close_and_open_orders_without_booking_price_gap(quantity):
    result = run(quantity=quantity)
    fills = result.event.fills
    assert [(fill.symbol, fill.price) for fill in fills] == [("F1", 100), ("F1", 102), ("F2", 113)]
    assert fills[1].timestamp < fills[2].timestamp
    assert result.event.summary()["final_equity"] == pytest.approx(10000 + quantity * 40)
    assert result.rolls["status"].to_list() == ["complete"]
    assert result.event.run_manifest["metadata"]["roll_policy"] == "close_first_constant_contracts"


def test_roll_waits_for_partial_close_before_opening_new_contract():
    bars = [replace(bar, volume=1 if bar.symbol == "F1" and bar.timestamp >= START + timedelta(days=2) else 100) for bar in data()]
    result = run(bars, quantity=2)
    fills = result.event.fills
    closes = [fill for fill in fills if fill.symbol == "F1" and fill.side.value == "sell"]
    opens = [fill for fill in fills if fill.symbol == "F2"]
    assert [fill.quantity for fill in closes] == [1, 1]
    assert opens[0].timestamp > closes[-1].timestamp
    assert opens[0].quantity == 2


def test_missing_new_contract_keeps_roll_explicitly_incomplete():
    result = run([bar for bar in data() if bar.symbol == "F1"])
    assert result.rolls["status"].to_list() == ["waiting_open"]
    assert all(fill.symbol == "F1" for fill in result.event.fills)
    assert result.event.positions == []


def test_roll_halt_defers_close_and_charges_all_legs():
    bars = [replace(bar, halted=True) if bar.symbol == "F1" and bar.timestamp == START + timedelta(days=2) else bar for bar in data()]
    result = run(bars, commission_bps=10)
    fills = result.event.fills
    assert fills[1].timestamp == START + timedelta(days=3)
    assert fills[2].timestamp == START + timedelta(days=4)
    assert result.event.total_commission == pytest.approx(sum(fill.price * 10 * 0.001 for fill in fills))


def test_adjusted_contract_prices_are_rejected():
    with pytest.raises(ValueError, match="unadjusted"):
        run([replace(bar, adj_close=bar.close + 10) for bar in data()])


def test_rejected_initial_order_cannot_be_reported_as_successful_roll():
    with pytest.raises(RuntimeError, match="rejected"):
        run(quantity=1000)


def test_initial_partial_order_is_cancelled_at_roll_and_only_held_quantity_moves():
    bars = [replace(bar, volume=1 if bar.timestamp == START else 0)
            if bar.symbol == "F1" and bar.timestamp < START + timedelta(days=2) else bar for bar in data()]
    result = run(bars, quantity=2)
    assert [(fill.symbol, fill.quantity) for fill in result.event.fills] == [("F1", 1), ("F1", 1), ("F2", 1)]
    assert result.rolls["quantity"].to_list() == [1]


def test_chained_rolls_keep_only_the_latest_contract():
    bars = data() + [Bar("F3", START + timedelta(days=i), 120+i, 120+i, 120+i, 120+i, 100) for i in range(6)]
    result = run_futures_roll_backtest(bars,
        [ContractRoll(START + timedelta(days=2), "F1", "F2"),
         ContractRoll(START + timedelta(days=4), "F2", "F3")],
        contract_multipliers={"F1": 10, "F2": 10, "F3": 10})
    assert result.rolls["status"].to_list() == ["complete", "complete"]
    assert [position.symbol for position in result.event.positions] == ["F3"]
    assert len(result.event.fills) == 5
