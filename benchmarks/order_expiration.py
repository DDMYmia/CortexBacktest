"""Compare same-day no-expiry checks on a mixed GTC/DAY book.

No orders are due: this measures candidate scanning, not expiration mutation.
"""
from datetime import UTC, datetime
import json
from pathlib import Path
from time import perf_counter

from cortexbacktest import Order, OrderType, Side, TimeInForce
from cortexbacktest.event.orders import OrderManager


def main():
    now = datetime(2024, 1, 1, 12, tzinfo=UTC)
    manager = OrderManager(retain_records=False)
    manager.track_pending_records = False
    for i in range(10100):
        manager.add_open_order(Order('S', now, Side.BUY, 1, OrderType.LIMIT,
            limit_price=1, id=str(i), time_in_force=TimeInForce.DAY if i % 101 == 0 else TimeInForce.GTC))
    def reference():
        return [manager.expire(order) for order in manager.open_orders if manager.is_expired(order, now)]
    results = {}
    for name, fn in [('full_book_reference', reference), ('production', lambda: manager.expire_orders(now))]:
        times = []
        for _ in range(3):
            before = perf_counter()
            for _ in range(500):
                assert fn() == []
            times.append(perf_counter() - before)
        results[name] = times
    result = {'orders': 10100, 'expirable': 100, 'checks': 500, 'seconds': results}
    path = Path('benchmark-results/strategy-suite-2026-09-09/expiration.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
