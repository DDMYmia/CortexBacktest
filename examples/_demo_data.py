"""Shared synthetic example inputs; weekdays here do not model exchange holidays."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl

from cortexbacktest import Bar


def business_dates(start: date, count: int) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def parsed_business_dates(start: str, count: int) -> list[date]:
    year, month, day = (int(part) for part in start.split("-"))
    return business_dates(date(year, month, day), count)


def business_date_strings(start: str, count: int) -> list[str]:
    return [day.isoformat() for day in parsed_business_dates(start, count)]


def flat_bars(prices: pl.DataFrame) -> list[Bar]:
    """Wide ISO-date closes as day-major flat OHLC bars, with synthetic volume."""
    symbols = [column for column in prices.columns if column != "date"]
    prices = prices.with_columns(pl.col("date").str.to_datetime())
    bars = []
    for row in prices.iter_rows(named=True):
        day = row["date"]
        timestamp = datetime(day.year, day.month, day.day, tzinfo=UTC)
        for symbol in symbols:
            price = float(row[symbol])
            bars.append(Bar(symbol=symbol, timestamp=timestamp, open=price, high=price, low=price, close=price, volume=10_000))
    return bars
