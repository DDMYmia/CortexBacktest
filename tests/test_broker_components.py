from datetime import UTC, datetime

from cortexbacktest import Bar, FillPrice, Order, OrderStatus, OrderType, Side
from cortexbacktest.event import BrokerSimulator, ExecutionHandler, Ledger, OrderManager, RiskEngine


def test_order_manager_assigns_ids_and_records_lifecycle():
    manager = OrderManager()
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)

    order = manager.prepare_order(Order("aapl", timestamp, Side.BUY, 1))
    submitted = manager.submit(order)
    cancelled = manager.cancel_order(order.id, reason="replace")

    assert order.id == "O00000001"
    assert order.symbol == "AAPL"
    assert submitted.status == OrderStatus.SUBMITTED
    assert cancelled.status == OrderStatus.CANCELLED
    assert [record.status for record in manager.order_records] == [
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.CANCELLED,
    ]
    assert manager.open_orders == []


def test_order_manager_replaces_open_order_without_duplicate_index_entries():
    manager = OrderManager()
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    order = manager.prepare_order(
        Order("aapl", timestamp, Side.BUY, 1, order_type=OrderType.LIMIT, limit_price=9.0)
    )
    manager.submit(order)

    replacement = Order(
        "AAPL",
        timestamp,
        Side.BUY,
        2,
        order_type=OrderType.LIMIT,
        limit_price=10.0,
        id=order.id,
    )
    record = manager.replace_order(order.id, replacement)

    assert record is not None
    assert record.status == OrderStatus.REPLACED
    assert manager.open_order_count == 1
    assert [open_order.quantity for open_order in manager.open_orders_for_symbol("AAPL")] == [2]
    assert [open_order.limit_price for open_order in manager.open_orders] == [10.0]


def test_broker_replace_order_amends_limit_and_preserves_submitted_count():
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    broker = BrokerSimulator()
    submitted = broker.submit_order(
        Order("AAPL", timestamp, Side.BUY, 1, order_type=OrderType.LIMIT, limit_price=9.0)
    )

    record = broker.replace_order(submitted.order.id, limit_price=10.5, quantity=2)
    fills = broker.process_bar(Bar("AAPL", timestamp, 10, 11, 9, 10))

    assert record is not None
    assert record.status == OrderStatus.REPLACED
    assert broker.submitted_order_count == 1
    assert broker.order_record_count == 4
    assert fills[0].quantity == 2
    assert fills[0].order_id == submitted.order.id


def test_order_manager_replaces_held_child_and_preserves_parent_activation():
    manager = OrderManager()
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    parent = manager.prepare_order(Order("AAPL", timestamp, Side.BUY, 1, id="P1"))
    child = manager.prepare_order(
        Order(
            "AAPL",
            timestamp,
            Side.SELL,
            1,
            order_type=OrderType.LIMIT,
            limit_price=12,
            id="C1",
            parent_id="P1",
        )
    )
    manager.submit(parent)
    manager.submit(child)

    replacement = Order(
        "AAPL",
        timestamp,
        Side.SELL,
        2,
        order_type=OrderType.LIMIT,
        limit_price=13,
        id="C1",
        parent_id="P1",
    )
    record = manager.replace_order("C1", replacement)
    activated = manager.activate_children("P1")

    assert record is not None
    assert record.status == OrderStatus.REPLACED
    assert [(order.id, order.quantity, order.limit_price) for order in activated] == [("C1", 2, 13)]


def test_order_manager_closes_orders_after_partial_and_full_fill():
    manager = OrderManager()
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    order = manager.prepare_order(Order("aapl", timestamp, Side.BUY, 80))
    manager.submit(order)

    remainder = manager.close_order_after_fill(order, 25)
    final_remainder = manager.close_order_after_fill(remainder, 55)

    assert remainder is not None
    assert remainder.quantity == 55
    assert final_remainder is None
    assert [record.status for record in manager.order_records] == [
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
    ]
    assert [record.order.quantity for record in manager.order_records] == [80, 80, 25, 55]


def test_order_manager_symbol_index_keeps_partial_remainder_once():
    manager = OrderManager()
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    aapl = manager.prepare_order(Order("aapl", timestamp, Side.BUY, 80))
    msft = manager.prepare_order(Order("msft", timestamp, Side.BUY, 10))
    manager.submit(aapl)
    manager.submit(msft)

    remainder = manager.close_order_after_fill(aapl, 25)
    manager.remove_open_order(aapl.id)
    manager.add_open_order(remainder)

    assert [order.quantity for order in manager.open_orders_for_symbol("AAPL")] == [55]
    assert [order.quantity for order in manager.iter_open_orders_for_symbol("AAPL")] == [55]
    assert [order.quantity for order in manager.open_orders_for_symbol("MSFT")] == [10]
    # remove + add is the *replacement* path (OrderManager.replace_open_order),
    # and a replaced order going to the back of the queue is the correct
    # exchange semantic. The partial-fill path production actually uses is
    # update_open_order, which keeps its position — pinned in the next test.
    assert [order.id for order in manager.open_orders] == [msft.id, aapl.id]
    assert manager.open_order_count == 2


def test_update_open_order_keeps_queue_position_for_a_partial_remainder():
    """The path broker.py takes for a partial fill (update_open_order, not
    remove + add) must not cost the remainder its price-time priority."""
    manager = OrderManager()
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    aapl = manager.prepare_order(Order("AAPL", timestamp, Side.BUY, 80))
    msft = manager.prepare_order(Order("MSFT", timestamp, Side.BUY, 10))
    manager.submit(aapl)
    manager.submit(msft)

    remainder = manager.close_order_after_fill(aapl, 25)
    manager.update_open_order(remainder)

    assert [order.id for order in manager.open_orders] == [aapl.id, msft.id]
    assert [order.quantity for order in manager.open_orders_for_symbol("AAPL")] == [55]


def test_order_manager_records_expiration():
    manager = OrderManager()
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    order = manager.prepare_order(Order("aapl", timestamp, Side.BUY, 1))

    record = manager.expire(order)

    assert record.status == OrderStatus.EXPIRED
    assert record.reason == "order expired"


def test_broker_process_bar_fast_returns_without_open_orders():
    broker = BrokerSimulator()
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)

    fills = broker.process_bar(Bar("AAPL", timestamp, 10, 10, 10, 10))

    assert fills == []
    assert broker.fill_count == 0
    assert broker.order_record_count == 0
    assert broker.open_orders == []


def test_execution_handler_fills_limit_order_and_updates_ledger():
    ledger = Ledger(initial_cash=1_000)
    handler = ExecutionHandler(
        ledger=ledger,
        risk_engine=RiskEngine(),
        fill_price=FillPrice.CLOSE,
    )
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    order = Order(
        "AAPL",
        timestamp,
        Side.BUY,
        2,
        order_type=OrderType.LIMIT,
        limit_price=10.5,
        id="O00000001",
    )
    bar = Bar("AAPL", timestamp, 10, 11, 9, 10)

    fill = handler.try_fill(order, bar)

    assert fill.price == 10
    assert fill.quantity == 2
    assert fill.order_id == "O00000001"
    assert ledger.state.cash == 980
    assert ledger.state.positions["AAPL"].quantity == 2


def test_execution_handler_fast_market_fill_matches_default_semantics():
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    bar = Bar("AAPL", timestamp, 10, 11, 9, 10)
    order = Order("AAPL", timestamp, Side.BUY, 2, id="O00000001")
    fast_ledger = Ledger(initial_cash=1_000)
    fast_handler = ExecutionHandler(
        ledger=fast_ledger,
        risk_engine=RiskEngine(),
        fill_price=FillPrice.CLOSE,
    )
    slow_ledger = Ledger(initial_cash=1_000)
    slow_handler = ExecutionHandler(
        ledger=slow_ledger,
        risk_engine=RiskEngine(),
        fill_price=FillPrice.MID,
    )

    fast_fill = fast_handler.try_fill(order, bar)
    slow_fill = slow_handler.try_fill(order, Bar("AAPL", timestamp, 10, 10, 10, 10))

    assert fast_fill == slow_fill
    assert fast_ledger.state.cash == slow_ledger.state.cash == 980
    assert fast_ledger.state.positions["AAPL"].quantity == slow_ledger.state.positions["AAPL"].quantity == 2


def test_cancel_oco_siblings_only_touches_its_own_group_at_scale():
    """cancel_oco_siblings looks the filled order's group up by index
    (open_orders_for_oco_group) rather than scanning every open order — this
    pins that behavior by surrounding the pair under test with many other
    open orders, in unrelated groups and with no group at all, across
    several symbols, and checking none of them are touched."""
    manager = OrderManager()
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)

    def _open(symbol: str, oco_group: str | None) -> Order:
        order = manager.prepare_order(Order(symbol, timestamp, Side.SELL, 1, oco_group=oco_group))
        manager.submit(order)
        return order

    noise = [_open(f"NOISE{i}", f"OTHER-{i}") for i in range(50)] + [_open(f"BARE{i}", None) for i in range(50)]
    take_profit = _open("AAPL", "EXIT")
    stop_loss = _open("AAPL", "EXIT")
    filled = _open("AAPL", "EXIT")  # stands in for the order that just filled elsewhere

    records = manager.cancel_oco_siblings(filled)

    assert {record.order.id for record in records} == {take_profit.id, stop_loss.id}
    assert manager.get_order(take_profit.id) is None
    assert manager.get_order(stop_loss.id) is None
    # Every noise order is untouched: still open, under its own id.
    for order in noise:
        assert manager.get_order(order.id) is not None


def test_trigger_envelope_releases_an_always_candidate_after_ratcheting():
    """A trailing stop re-enters update_open_order on every ratchet. If each
    of those re-counted it as an always-candidate without releasing the
    previous version, the count would never fall back to zero and the symbol
    could never be gated again — the gate would quietly stop working for it."""
    manager = OrderManager()
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    order = manager.prepare_order(
        Order("AAPL", timestamp, Side.SELL, 1, order_type=OrderType.TRAILING_STOP, trail_amount=2.0)
    )
    manager.submit(order)

    # A trailing stop is a candidate on any bar, however far from its price.
    assert manager.symbol_can_trigger("AAPL", 1_000.0, 1_001.0) is True

    for high in (110.0, 120.0, 130.0):
        live = manager.get_order(order.id)
        manager.update_trailing_stop(live, Bar("AAPL", timestamp, high, high, high, high))

    manager.remove_open_order(order.id)
    assert manager.symbol_can_trigger("AAPL", 1_000.0, 1_001.0) is False


def test_trigger_envelope_rebuilds_when_the_extreme_order_leaves():
    """The envelope may only over-report, but it is useless if it never
    narrows: the departing order is usually the one pinning an extreme."""
    manager = OrderManager()
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    near = manager.prepare_order(
        Order("AAPL", timestamp, Side.BUY, 1, order_type=OrderType.LIMIT, limit_price=100.0)
    )
    far = manager.prepare_order(
        Order("AAPL", timestamp, Side.BUY, 1, order_type=OrderType.LIMIT, limit_price=10.0)
    )
    manager.submit(near)
    manager.submit(far)

    # A bar down to 99 reaches the 100 limit.
    assert manager.symbol_can_trigger("AAPL", 99.0, 101.0) is True
    manager.remove_open_order(near.id)
    # Only the 10 limit is left, so the same bar can no longer trigger anything.
    assert manager.symbol_can_trigger("AAPL", 99.0, 101.0) is False
    assert manager.symbol_can_trigger("AAPL", 9.0, 11.0) is True


def test_cancel_oco_siblings_across_different_symbols():
    """The contract does not require an OCO group to share one symbol (see
    tests/test_matching_path_equivalence.py's fuzz cases), so the index must
    not assume it does."""
    manager = OrderManager()
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)

    def _open(symbol: str) -> Order:
        order = manager.prepare_order(Order(symbol, timestamp, Side.SELL, 1, oco_group="CROSS"))
        manager.submit(order)
        return order

    filled = _open("AAPL")
    sibling = _open("MSFT")

    records = manager.cancel_oco_siblings(filled)

    assert [record.order.id for record in records] == [sibling.id]
    assert manager.get_order(sibling.id) is None
