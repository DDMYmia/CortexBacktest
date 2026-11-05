from datetime import UTC, datetime, timedelta

from cortexbacktest import (
    HistoricalDataStore,
    OrderBookEvent,
    OrderBookRecorder,
    OrderBookReplayer,
    Side,
    order_book_snapshot_to_tick,
)


def test_order_book_recorder_replayer_snapshots_and_tick_conversion(tmp_path):
    store = HistoricalDataStore(tmp_path)
    start = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    events = [
        OrderBookEvent("AAPL", start, Side.BUY, price=99.9, quantity=100, level=0),
        OrderBookEvent("AAPL", start + timedelta(milliseconds=1), Side.SELL, price=100.1, quantity=120, level=0),
        OrderBookEvent("AAPL", start + timedelta(milliseconds=2), Side.BUY, price=99.8, quantity=80, level=1),
        OrderBookEvent("AAPL", start + timedelta(milliseconds=3), Side.BUY, price=99.8, quantity=0, level=1, action="delete"),
    ]
    recorder = OrderBookRecorder(store, "aapl_l2")
    for event in reversed(events):
        recorder.record(event)
    recorder.flush()

    replayed_events = list(OrderBookReplayer(store, "aapl_l2").replay_events())
    snapshots = list(OrderBookReplayer(store, "aapl_l2").replay_snapshots())
    tick = order_book_snapshot_to_tick(snapshots[1])

    assert [event.timestamp for event in replayed_events] == sorted(event.timestamp for event in events)
    assert snapshots[1].best_bid.price == 99.9
    assert snapshots[1].best_ask.price == 100.1
    assert snapshots[1].mid_price == 100.0
    assert snapshots[-1].bids == snapshots[1].bids
    assert tick.last_price == 100.0
    assert tick.bid_price == 99.9
    assert tick.ask_price == 100.1
    assert store.list_datasets("order_books") == ["aapl_l2"]
