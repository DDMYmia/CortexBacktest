"""Target-weight rebalancing bridge for event-driven backtests."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, localcontext
import re

import polars as pl

from ..contracts import QUANTITY_TOLERANCE, Order, Position, Side
from ..data.backend import normalize_date_column, to_backend


@dataclass(frozen=True, slots=True)
class RebalanceConfig:
    """Controls target-weight to order conversion."""

    cash_buffer: float = 0.0
    min_trade_value: float = 0.0
    allow_fractional: bool = False
    tag: str = "rebalance"
    quantity_steps: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_document(cls, document):
        values = dict(document)
        if "quantity_steps" in values:
            values["quantity_steps"] = tuple(tuple(row) for row in values["quantity_steps"])
        return cls(**values)

    def to_document(self):
        values = asdict(self)
        if not self.quantity_steps:
            values.pop("quantity_steps")
        return values

    def __post_init__(self) -> None:
        if not 0 <= self.cash_buffer < 1:
            raise ValueError("cash_buffer must be in [0, 1)")
        if self.min_trade_value < 0:
            raise ValueError("min_trade_value must be non-negative")
        if self.quantity_steps:
            if (not self.allow_fractional or not isinstance(self.quantity_steps, tuple)
                    or any(not isinstance(row, tuple) or len(row) != 2
                           or not isinstance(row[0], str) or not row[0]
                           or not isinstance(row[1], str) or len(row[1]) > 128
                           or re.fullmatch(r"(?:0\.[0-9]*[1-9][0-9]*|[1-9][0-9]*(?:\.[0-9]+)?)", row[1]) is None
                           for row in self.quantity_steps)
                    or len(dict(self.quantity_steps)) != len(self.quantity_steps)):
                raise ValueError("Quantity steps require unique symbols, positive decimal strings and fractional sizing")


def orders_for_target_weights(
    *,
    timestamp: datetime,
    target_weights,
    prices: dict[str, float],
    positions: dict[str, Position],
    equity: float,
    config: RebalanceConfig | None = None,
) -> list[Order]:
    """Generate market orders needed to reach target weights."""
    cfg = config or RebalanceConfig()
    changes = quantity_changes_for_target_weights(
        target_weights=target_weights, prices=prices,
        quantities={symbol: position.quantity for symbol, position in positions.items()},
        equity=equity, config=cfg,
    )
    return [Order(symbol=symbol, timestamp=timestamp, side=Side.BUY if change > 0 else Side.SELL,
                  quantity=abs(change), tag=cfg.tag) for symbol, change in changes.items()]


def quantity_changes_for_target_weights(
    *, target_weights, prices: dict[str, float], quantities: dict[str, float], equity: float,
    config: RebalanceConfig | None = None,
) -> dict[str, float]:
    """Shared sizing for simulation and observed-portfolio planning; no broker objects.

    Return signed quantity changes after the existing cash buffer, minimum trade
    and fractional-lot rules. Callers own account scope, quote freshness and
    execution authorization. Missing prices retain the simulator's skip behavior;
    an external account caller must validate completeness before using this helper.
    """
    cfg = config or RebalanceConfig()
    if equity <= 0:
        return {}
    investable_equity = equity * (1.0 - cfg.cash_buffer)
    weights = _normalize_weights(target_weights)
    symbols = sorted(set(weights) | set(quantities))
    changes = {}
    steps = dict(cfg.quantity_steps)

    for symbol in symbols:
        price = prices.get(symbol)
        if price is None or price <= 0:
            continue
        target_weight = weights.get(symbol, 0.0)
        target_value = investable_equity * target_weight
        current_qty = quantities.get(symbol, 0.0)
        current_value = current_qty * price
        delta_value = target_value - current_value
        if abs(delta_value) < cfg.min_trade_value:
            continue
        quantity = abs(delta_value) / price
        if not cfg.allow_fractional:
            quantity = float(int(quantity))
        elif steps:
            if symbol not in steps:
                raise ValueError("Missing pinned quantity step")
            with localcontext() as context:
                context.prec = 160
                step = Decimal(steps[symbol])
                quantity = float((Decimal(str(quantity)) / step).to_integral_value(rounding=ROUND_DOWN) * step)
        if quantity <= 0:
            continue
        changes[symbol] = quantity if delta_value > 0 else -quantity
    return changes


class TargetWeightStrategy:
    """Strategy that converts a date x symbol target-weight matrix to orders."""

    def __init__(
        self,
        target_weights,
        *,
        config: RebalanceConfig | None = None,
        broker=None,
        signal_lag_sessions: int = 0,
        trade_at_open: bool = False,
        decision_times: bool = False,
    ) -> None:
        if signal_lag_sessions < 0:
            raise ValueError("signal_lag_sessions must be non-negative")
        if decision_times and (signal_lag_sessions or not trade_at_open):
            raise ValueError("Decision-time targets require open execution without a session lag")
        self.decision_times = decision_times
        self.target_weights = _target_weight_frame(target_weights)
        self._target_weight_rows = _target_weight_rows(self.target_weights, decision_times=decision_times)
        self._target_weight_dates = sorted(self._target_weight_rows)
        self.config = config or RebalanceConfig()
        self.broker = broker
        self.signal_lag_sessions = signal_lag_sessions
        self.trade_at_open = trade_at_open or bool(signal_lag_sessions)
        self._latest_prices: dict[str, float] = {}
        self._last_rebalance_date = None
        # Trading sessions seen from the bar feed, in order. The signal lag is
        # counted in sessions, so it must come from the calendar the bars
        # describe, not from the rows of the target-weight matrix.
        self._sessions: list = []

    def initialize(self) -> None:
        """Strategy lifecycle hook."""

    def on_bar(self, bar) -> list[Order]:
        if self.trade_at_open:
            return []
        self._latest_prices[bar.symbol] = bar.return_close
        return self._rebalance_orders(bar.timestamp)

    def on_bars(self, bars) -> list[Order]:
        if self.trade_at_open:
            return []
        for bar in bars:
            self._latest_prices[bar.symbol] = bar.return_close
        if not bars:
            return []
        return self._rebalance_orders(bars[0].timestamp)

    def on_before_bars(self, bars) -> list[Order]:
        """Place lagged target weights at the current bar open.

        This hook deliberately reads only `open`. The standard pipeline lags
        close-derived targets by one session. Explicit ``trade_at_open=True``
        with zero lag is for targets already aligned to executable instants;
        callers must establish that their decisions preceded those instants.
        """
        if not self.trade_at_open or not bars:
            return []
        for bar in bars:
            self._latest_prices[bar.symbol] = bar.open
        if self.decision_times:
            index = bisect_left(self._target_weight_dates, bars[0].timestamp) - 1
            if index < 0:
                return []
            return self._rebalance_orders(bars[0].timestamp, target_date=self._target_weight_dates[index])
        session_date = _session_date(bars[0].timestamp)
        if not self._sessions or self._sessions[-1] != session_date:
            self._sessions.append(session_date)
        signal_date = self._lagged_signal_date(session_date)
        if signal_date is None:
            return []
        return self._rebalance_orders(bars[0].timestamp, target_date=signal_date)

    def _rebalance_orders(self, timestamp, *, target_date=None) -> list[Order]:
        rebalance_date = target_date if self.decision_times else _session_date(timestamp)
        target_date = rebalance_date if target_date is None else target_date
        if target_date not in self._target_weight_rows:
            return []
        if self._last_rebalance_date == rebalance_date:
            return []
        if self.broker is None:
            raise ValueError("TargetWeightStrategy requires a broker reference")

        row = self._target_weight_rows[target_date]
        required_symbols = {
            str(symbol).upper()
            for symbol, weight in row.items()
            if abs(float(weight)) > QUANTITY_TOLERANCE
        } | set(self.broker.state.positions)
        if not required_symbols.issubset(self._latest_prices):
            return []
        available_prices = {symbol: self._latest_prices[symbol] for symbol in required_symbols}
        if not available_prices:
            return []
        self._last_rebalance_date = rebalance_date
        return orders_for_target_weights(
            timestamp=timestamp,
            target_weights=row,
            prices=available_prices,
            positions=self.broker.state.positions,
            equity=self.broker.state.equity(available_prices),
            config=self.config,
        )

    def _lagged_signal_date(self, rebalance_date):
        # The lag is counted in trading sessions, so it must be applied against
        # the actual session calendar the bars describe (self._sessions), not
        # against the rows of the target-weight matrix — a monthly matrix has
        # far fewer rows than sessions, so indexing into it directly would lag
        # by months instead of sessions whenever rebalance_date isn't itself a
        # matrix row.
        try:
            index = self._sessions.index(rebalance_date)
        except ValueError:
            return None
        if index < self.signal_lag_sessions:
            return None
        target_session = self._sessions[index - self.signal_lag_sessions]
        # Then asof-match that session against the (possibly sparser)
        # target-weight matrix: the latest row at or before target_session.
        row_index = bisect_right(self._target_weight_dates, target_session) - 1
        if row_index < 0:
            return None
        return self._target_weight_dates[row_index]

    def finalize(self) -> None:
        """Strategy lifecycle hook."""


def _normalize_weights(target_weights) -> dict[str, float]:
    if isinstance(target_weights, pl.Series):
        values = dict(enumerate(target_weights.to_list()))
    else:
        values = dict(target_weights)
    result = {}
    for symbol, value in values.items():
        if value is None or (isinstance(value, float) and value != value):
            continue
        result[str(symbol).upper()] = float(value)
    return result


def _session_date(timestamp):
    ts = timestamp
    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts.replace(hour=0, minute=0, second=0, microsecond=0)


def _target_weight_frame(target_weights) -> pl.DataFrame:
    data = to_backend(target_weights, backend="polars")
    if "date" not in data.columns:
        if _is_pandas(target_weights):
            data = pl.from_pandas(target_weights.reset_index(names="date"))
        else:
            raise ValueError("target_weights requires a date column for Polars input")
    symbol_columns = sorted(str(column).upper() for column in data.columns if column != "date")
    rename_map = {
        column: str(column).upper()
        for column in data.columns
        if column != "date" and str(column).upper() != column
    }
    if rename_map:
        data = data.rename(rename_map)
    return (
        normalize_date_column(data)
        .select([pl.col("date"), *[pl.col(symbol).cast(pl.Float64, strict=False).alias(symbol) for symbol in symbol_columns]])
        .sort("date")
    )


def _target_weight_rows(target_weights: pl.DataFrame, *, decision_times: bool = False) -> dict:
    rows = {}
    for row in target_weights.iter_rows(named=True):
        date = row["date"] if decision_times else _session_date(row["date"])
        if decision_times and date in rows:
            raise ValueError("Duplicate target decision time")
        weights = {
            symbol: float(value)
            for symbol, value in row.items()
            if symbol != "date" and value is not None and not (isinstance(value, float) and value != value)
        }
        rows[date] = weights
    return rows




def _is_pandas(frame) -> bool:
    return type(frame).__module__.split(".")[0] == "pandas"
