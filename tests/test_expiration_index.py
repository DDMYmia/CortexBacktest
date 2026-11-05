from dataclasses import replace
from datetime import UTC, datetime, timedelta
import random

from cortexbacktest import Order, OrderType, Side, TimeInForce
from cortexbacktest.event.orders import OrderManager


class FullScan(OrderManager):
    def expire_orders(self, timestamp, *, reason='order expired'):
        return [self.expire(order, reason=reason) for order in self.open_orders
                if self.is_expired(order, timestamp)]


def test_expiration_matches_full_scan_after_amend_cancel_replace_and_reset():
    now = datetime(2024, 1, 1, tzinfo=UTC)
    actual, reference = OrderManager(), FullScan()
    rng = random.Random(73)
    for i in range(100):
        order = Order('S', now, Side.BUY, 1, OrderType.LIMIT, limit_price=1,
            id=str(i), time_in_force=TimeInForce.DAY if i % 7 == 0 else TimeInForce.GTC)
        actual.add_open_order(order)
        reference.add_open_order(order)
    for tick in range(200):
        identity = str(rng.randrange(100))
        old = actual.get_order(identity)
        if old is not None:
            op = rng.randrange(4)
            if op == 0:
                actual.cancel_order(identity)
                reference.cancel_order(identity)
            else:
                amended = replace(old, quantity=2, time_in_force=TimeInForce.GTD if op == 1 else TimeInForce.GTC,
                                  expires_at=now + timedelta(minutes=tick + 1) if op == 1 else None)
                actual.update_open_order(amended)
                reference.update_open_order(amended)
        at = now + timedelta(minutes=tick)
        assert actual.expire_orders(at) == reference.expire_orders(at)
        assert actual.open_orders == reference.open_orders
    assert actual.expire_orders(now + timedelta(days=1)) == reference.expire_orders(now + timedelta(days=1))
    # Public replacement must not leave stale candidates from the old book.
    actual.open_orders = reference.open_orders = []
    assert actual.expire_orders(now + timedelta(days=2)) == []


def test_deadline_equality_and_amendment_preserve_submission_order():
    now = datetime(2024, 1, 1, tzinfo=UTC)
    manager = OrderManager()
    for identity in ('first', 'second'):
        manager.add_open_order(Order('S', now, Side.BUY, 1, id=identity))
    for identity in ('second', 'first'):
        manager.update_open_order(replace(manager.get_order(identity), expires_at=now))
    assert manager.expire_orders(now) == []
    records = manager.expire_orders(now + timedelta(microseconds=1))
    assert [r.order.id for r in records] == ['first', 'second']
    assert manager.open_order_count == 0
