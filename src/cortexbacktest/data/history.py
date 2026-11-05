"""Local historical data store for offline backtesting."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
import shutil
from uuid import uuid4

import polars as pl

from ..contracts import Bar, OrderBookEvent, OrderBookLevel, OrderBookSnapshot, Side, Tick


class HistoricalDataStore:
    """Parquet-backed store for bars, ticks, and order-book events.

    Ticks and order-book events — the two kinds a live/paper session appends
    to over and over, potentially for months — are stored hive-partitioned by
    date:

        <root>/ticks/<name>/date=2026-09-05/part-<id>.parquet
        <root>/order_books/<name>/date=2026-09-05/part-<id>.parquet

    Appending writes one new part file per date touched and never reads or
    rewrites a file already on disk, so a flush costs O(new rows) rather than
    O(rows written so far) — the old single-file layout read the whole
    dataset back, concatenated, and rewrote it on every append. Bars use a
    single file by default; `write_bars(..., partitioned=True)` opts into the
    same date partitions. Bar reads support both layouts.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_bars(self, name: str, bars: list[Bar], *, partitioned: bool = False) -> Path:
        if partitioned:
            return self._write_partitioned("bars", name, _frame_from_records(bars))
        dataset_dir = self._dataset_dir("bars", name)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        path = self._file_path("bars", name)
        _frame_from_records(bars).write_parquet(path)
        return path

    def read_bars(self, name: str) -> list[Bar]:
        frame = self.scan_bars(name).collect()
        if frame.is_empty():
            return []
        return [
            Bar(
                symbol=row["symbol"],
                timestamp=row["timestamp"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row.get("volume"),
                adj_close=row.get("adj_close"),
                vwap=row.get("vwap"),
                open_interest=row.get("open_interest"),
                halted=bool(row.get("halted", False)),
                limit_up=row.get("limit_up"),
                limit_down=row.get("limit_down"),
            )
            for row in frame.iter_rows(named=True)
        ]

    def scan_bars(self, name: str, *, start_date: date | None = None,
                  end_date: date | None = None, symbols: list[str] | None = None,
                  columns: list[str] | None = None) -> pl.LazyFrame:
        """Lazy date/symbol/projection pushdown for single-file and partitioned bars."""
        self._validate_name(name)
        if start_date is not None and end_date is not None and start_date > end_date:
            raise ValueError("start_date must not exceed end_date")
        frame = self._scan_frame("bars", name, start_date=start_date, end_date=end_date)
        if not frame.collect_schema().names():
            return frame
        if symbols is not None:
            frame = frame.filter(pl.col("symbol").str.to_uppercase().is_in([s.upper() for s in symbols]))
        return frame.select(columns) if columns is not None else frame

    def write_ticks(self, name: str, ticks: list[Tick]) -> Path:
        return self._write_partitioned("ticks", name, _frame_from_records(ticks))

    def read_ticks(self, name: str, *, start_date: date | None = None, end_date: date | None = None) -> list[Tick]:
        frame = self._scan_frame("ticks", name, start_date=start_date, end_date=end_date).collect()
        return [
            Tick(
                symbol=row["symbol"],
                timestamp=row["timestamp"],
                last_price=row["last_price"],
                volume=row.get("volume"),
                bid_price=row.get("bid_price"),
                ask_price=row.get("ask_price"),
                bid_volume=row.get("bid_volume"),
                ask_volume=row.get("ask_volume"),
                open_interest=row.get("open_interest"),
            )
            for row in frame.iter_rows(named=True)
        ]

    def scan_ticks(self, name: str, *, start_date: date | None = None, end_date: date | None = None) -> pl.LazyFrame:
        """Lazily scan a tick dataset as raw columns, without building Tick objects.

        Filter pushdown prunes whole `date=` partitions that fall outside
        `start_date`/`end_date` before any Parquet file in them is opened —
        the payoff for the partitioned layout on a dataset spanning years.
        """
        return self._scan_frame("ticks", name, start_date=start_date, end_date=end_date)

    def list_datasets(self, kind: str) -> list[str]:
        directory = self.root / kind
        if not directory.exists():
            return []
        names = {path.stem for path in directory.glob("*.parquet")}
        names |= {path.name for path in directory.iterdir() if path.is_dir()}
        return sorted(names)

    def write_order_book_events(self, name: str, events: list[OrderBookEvent]) -> Path:
        return self._write_partitioned("order_books", name, _frame_from_records(events))

    def read_order_book_events(
        self, name: str, *, start_date: date | None = None, end_date: date | None = None
    ) -> list[OrderBookEvent]:
        frame = self._scan_frame("order_books", name, start_date=start_date, end_date=end_date).collect()
        return [
            OrderBookEvent(
                symbol=row["symbol"],
                timestamp=row["timestamp"],
                side=Side(row["side"]),
                price=row["price"],
                quantity=row["quantity"],
                action=row.get("action") or "update",
                level=row.get("level") or 0,
            )
            for row in frame.iter_rows(named=True)
        ]

    def _validate_name(self, name: str) -> None:
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("dataset name must be a simple file stem")

    def _file_path(self, kind: str, name: str) -> Path:
        self._validate_name(name)
        directory = self.root / kind
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{name}.parquet"

    def _dataset_dir(self, kind: str, name: str) -> Path:
        self._validate_name(name)
        return self.root / kind / name

    def _write_partitioned(self, kind: str, name: str, frame: pl.DataFrame) -> Path:
        """Replace a ticks/order_books dataset with exactly these rows."""
        dataset_dir = self._dataset_dir(kind, name)
        single_file = self.root / kind / f"{name}.parquet"
        if single_file.exists():
            single_file.unlink()
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        if not frame.is_empty():
            self._write_partitions(dataset_dir, frame)
        return dataset_dir

    def _write_partitions(self, dataset_dir: Path, frame: pl.DataFrame) -> None:
        dated = frame.with_columns(pl.col("timestamp").dt.strftime("%Y-%m-%d").alias("_date"))
        for (date_value,), group in dated.group_by("_date", maintain_order=True):
            partition_dir = dataset_dir / f"date={date_value}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            group.drop("_date").write_parquet(partition_dir / f"part-{uuid4().hex[:12]}.parquet")

    def _append_rows(self, kind: str, name: str, rows) -> Path:
        """Append a batch as new partition files — never re-reads rows already on disk."""
        dataset_dir = self._dataset_dir(kind, name)
        if not rows:
            return dataset_dir
        incoming = _frame_from_records(rows)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        self._write_partitions(dataset_dir, incoming)
        return dataset_dir

    def _scan_frame(
        self, kind: str, name: str, *, start_date: date | None, end_date: date | None
    ) -> pl.LazyFrame:
        single_file = self.root / kind / f"{name}.parquet"
        parts: list[pl.LazyFrame] = []
        if single_file.exists():
            parts.append(pl.scan_parquet(single_file))
        dataset_dir = self._dataset_dir(kind, name)
        if dataset_dir.exists() and any(dataset_dir.glob("date=*")):
            partitioned = pl.scan_parquet(dataset_dir, hive_partitioning=True)
            if start_date is not None:
                partitioned = partitioned.filter(pl.col("date") >= start_date)
            if end_date is not None:
                partitioned = partitioned.filter(pl.col("date") <= end_date)
            parts.append(partitioned.drop("date"))
        if not parts:
            return pl.LazyFrame()
        combined = parts[0] if len(parts) == 1 else pl.concat(parts, how="vertical_relaxed")
        if single_file.exists() and (start_date is not None or end_date is not None):
            # The single file has no partitions to prune on, so its rows
            # still need the date filter applied post-hoc.
            combined = combined.filter(_date_range_expr(start_date, end_date))
        return combined


class TickRecorder:
    """Append ticks in memory and flush to a HistoricalDataStore."""

    def __init__(self, store: HistoricalDataStore, name: str) -> None:
        self.store = store
        self.name = name
        self.ticks: list[Tick] = []

    def record(self, tick: Tick) -> None:
        self.ticks.append(tick)

    def flush(self) -> Path:
        path = self.store._append_rows("ticks", self.name, self.ticks)
        self.ticks = []
        return path


class TickReplayer:
    """Replay locally stored ticks in timestamp order."""

    def __init__(self, store: HistoricalDataStore, name: str) -> None:
        self.store = store
        self.name = name

    def replay(self):
        yield from sorted(self.store.read_ticks(self.name), key=lambda tick: (tick.timestamp, tick.symbol))


class OrderBookRecorder:
    """Append order-book events in memory and flush to a HistoricalDataStore."""

    def __init__(self, store: HistoricalDataStore, name: str) -> None:
        self.store = store
        self.name = name
        self.events: list[OrderBookEvent] = []

    def record(self, event: OrderBookEvent) -> None:
        self.events.append(event)

    def flush(self) -> Path:
        path = self.store._append_rows("order_books", self.name, self.events)
        self.events = []
        return path


class OrderBookReplayer:
    """Replay order-book events and snapshots in timestamp order."""

    def __init__(self, store: HistoricalDataStore, name: str) -> None:
        self.store = store
        self.name = name

    def replay_events(self):
        yield from sorted(
            self.store.read_order_book_events(self.name),
            key=lambda event: (event.timestamp, event.symbol, event.level, event.side.value),
        )

    def replay_snapshots(self):
        bids: dict[str, dict[int, OrderBookLevel]] = {}
        asks: dict[str, dict[int, OrderBookLevel]] = {}
        for event in self.replay_events():
            symbol = event.symbol.upper()
            book = bids if event.side == Side.BUY else asks
            levels = book.setdefault(symbol, {})
            if event.action == "delete" or event.quantity <= 0:
                levels.pop(event.level, None)
            else:
                levels[event.level] = OrderBookLevel(event.price, event.quantity)
            bid_levels = bids.get(symbol, {})
            ask_levels = asks.get(symbol, {})
            yield OrderBookSnapshot(
                symbol=symbol,
                timestamp=event.timestamp,
                bids=tuple(bid_levels[index] for index in sorted(bid_levels)),
                asks=tuple(ask_levels[index] for index in sorted(ask_levels)),
            )


def order_book_snapshot_to_tick(snapshot: OrderBookSnapshot) -> Tick:
    bid = snapshot.best_bid
    ask = snapshot.best_ask
    mid = snapshot.mid_price
    if mid is None:
        quote = bid if bid is not None else ask
        if quote is None:
            raise ValueError("Cannot convert an empty order book to a tick")
        mid = quote.price
    return Tick(
        symbol=snapshot.symbol,
        timestamp=snapshot.timestamp,
        last_price=mid,
        bid_price=bid.price if bid is not None else None,
        ask_price=ask.price if ask is not None else None,
        bid_volume=bid.quantity if bid is not None else None,
        ask_volume=ask.quantity if ask is not None else None,
    )


def _date_range_expr(start_date: date | None, end_date: date | None) -> pl.Expr:
    expr = pl.col("timestamp").dt.date() >= start_date if start_date is not None else pl.lit(True)
    if end_date is not None:
        expr = expr & (pl.col("timestamp").dt.date() <= end_date)
    return expr


def _frame_from_records(rows) -> pl.DataFrame:
    """Bars are NamedTuples; ticks and order books are dataclasses."""
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame([row._asdict() if hasattr(row, "_asdict") else asdict(row) for row in rows])
