"""Carry delisting proceeds forward so cross-sectional research pays for them.

A delisted symbol simply stops having prices. Every downstream calculation reads
that as *absent* rather than as *lost*, and absent costs nothing:

- `forward_returns_matrix` yields `null` on the last day the symbol traded, so
  the position's terminal loss never enters the sample;
- IC and quantile spreads are computed over the surviving names only;
- a long-short portfolio drops the holding at no cost, as though it had been
  sold at the last quoted price.

That is survivorship bias, and in a factor study it flatters exactly the factors
that load on distress. The event broker already handles this correctly
(`CorporateActionType.DELISTING` closes the position at its cash-out value), but
the research path never saw those actions.

The fix is upstream of every consumer rather than inside any of them: write the
cash-out value into the price matrix from the ex-date forward. A worthless
delisting becomes a price of 0, so the terminal return is -100%; a cash-out at
30 against a last trade of 100 becomes -70%; and the flat series afterwards
represents proceeds sitting in cash. Zero proceeds can yield NaN from later
0/0 returns; downstream statistics still need their own missing-value policy.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import polars as pl

from ..contracts import CorporateAction, CorporateActionType


def apply_delisting_prices(
    prices: pl.DataFrame,
    actions: Iterable[CorporateAction],
) -> pl.DataFrame:
    """Return ``prices`` with each delisted symbol's cash-out value carried forward.

    Rows on and after a symbol's ex-date take the delisting ``amount``. Symbols
    with no delisting, and delistings for symbols absent from the matrix, are
    left alone. Non-delisting corporate actions are ignored here: splits and
    dividends belong to the price basis, which is the data provider's job, while
    a delisting has no price to adjust because trading stopped.
    """
    delistings = {
        action.symbol: action
        for action in actions
        if action.action_type == CorporateActionType.DELISTING
    }
    if not delistings:
        return prices

    columns = {column for column in prices.columns if column != "date"}
    applicable = {symbol: action for symbol, action in delistings.items() if symbol in columns}
    if not applicable:
        return prices

    return prices.with_columns(
        [
            pl.when(pl.col("date") >= _as_datetime(applicable[symbol].ex_date))
            .then(pl.lit(applicable[symbol].amount, dtype=pl.Float64))
            .otherwise(pl.col(symbol))
            .alias(symbol)
            for symbol in sorted(applicable)
        ]
    )


def delisted_symbols(actions: Iterable[CorporateAction]) -> frozenset[str]:
    """Symbols that stop trading, for callers that need the universe to shrink."""
    return frozenset(
        action.symbol
        for action in actions
        if action.action_type == CorporateActionType.DELISTING
    )


def _as_datetime(value: date) -> datetime:
    """Match the matrix's datetime column so the comparison does not silently miss."""
    if isinstance(value, datetime):
        return value
    return datetime(value.year, value.month, value.day)
