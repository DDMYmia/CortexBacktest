"""Shared event-record schemas and row encodings."""

from __future__ import annotations
from typing import Iterable
from .contracts import Fill, Order, OrderRecord, Position

ORDER_COLUMNS = [
    "timestamp",
    "order_id",
    "symbol",
    "side",
    "quantity",
    "order_type",
    "limit_price",
    "stop_price",
    "trail_amount",
    "trail_percent",
    "expires_at",
    "time_in_force",
    "tag",
    "oco_group",
    "parent_id",
    "status",
    "status_label",
    "reason",
]
OPEN_ORDER_COLUMNS = ORDER_COLUMNS[:-3]
FILL_COLUMNS = [
    "timestamp",
    "order_id",
    "symbol",
    "side",
    "quantity",
    "price",
    "commission",
    "slippage",
    "notional",
]
POSITION_COLUMNS = ["symbol", "quantity", "last_price", "market_value"]


def iter_order_record_rows(records: Iterable[OrderRecord]):
    for record in records:
        yield {
            **_order_row(record.order),
            "status": record.status.value,
            "status_label": record.status_label,
            "reason": record.reason,
        }


def iter_open_order_rows(orders: Iterable[Order]):
    for order in orders:
        yield _order_row(order)


def _order_row(order: Order):
    return {
        "timestamp": order.timestamp,
        "order_id": order.id,
        "symbol": order.symbol,
        "side": order.side.value,
        "quantity": order.quantity,
        "order_type": order.order_type.value,
        "limit_price": order.limit_price,
        "stop_price": order.stop_price,
        "trail_amount": order.trail_amount,
        "trail_percent": order.trail_percent,
        "expires_at": order.expires_at,
        "time_in_force": order.time_in_force.value,
        "tag": order.tag,
        "oco_group": order.oco_group,
        "parent_id": order.parent_id,
    }


def iter_fill_rows(fills: Iterable[Fill]):
    for fill in fills:
        yield {
            "timestamp": fill.timestamp,
            "order_id": fill.order_id,
            "symbol": fill.symbol,
            "side": fill.side.value,
            "quantity": fill.quantity,
            "price": fill.price,
            "commission": fill.commission,
            "slippage": fill.slippage,
            "notional": fill.quantity * fill.price,
        }


def iter_position_rows(positions: Iterable[Position]):
    for position in positions:
        yield {
            "symbol": position.symbol,
            "quantity": position.quantity,
            "last_price": position.last_price,
            "market_value": position.market_value,
        }


def _scalar_value(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value
