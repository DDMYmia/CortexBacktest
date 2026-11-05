from datetime import UTC, datetime, timedelta

import polars as pl

from cortexbacktest import Order, OrderType, Side
from cortexbacktest.contracts import OrderRecord, OrderStatus
from cortexbacktest.event.record_sink import EventRecordSink


def test_parquet_schema_survives_initial_nulls_and_fractional_quantities(tmp_path):
    now = datetime(2024, 1, 1, tzinfo=UTC)
    with EventRecordSink(tmp_path, format='parquet', buffer_rows=1) as sink:
        sink.write_order_record(OrderRecord(Order('S', now, Side.BUY, 1, id='first'), OrderStatus.ACCEPTED))
        sink.write_order_record(OrderRecord(
            Order('S', now, Side.BUY, .25, OrderType.LIMIT, limit_price=10.5,
                  expires_at=now + timedelta(days=1), id='second', tag='later'),
            OrderStatus.PARTIALLY_FILLED, 'volume limited'))
    orders = pl.read_parquet(tmp_path / 'orders.parquet')
    assert orders['reason'].to_list() == [None, 'volume limited']
    assert orders['limit_price'].to_list() == [None, 10.5]
    assert orders['quantity'].to_list() == [1.0, .25]
    assert orders['expires_at'][1] == (now + timedelta(days=1)).isoformat()
    assert pl.read_parquet(tmp_path / 'fills.parquet').schema['price'] == pl.Float64
