"""Corporate action schedule applied at the open of each session.

Only meaningful on unadjusted prices. See :class:`~..contracts.CorporateAction`
for the price-basis rule the engine enforces.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Iterable

from ..contracts import CorporateAction, CorporateActionType

# Order within one ex-date, chosen so each step sees the share count its data is
# quoted against:
#   1. share-count changes first (split, stock dividend) — a dividend is quoted
#      per post-split share, so paying first would use the wrong count;
#   2. cash dividends next, on the settled count;
#   3. identity changes (symbol change, merger) after the holding is economically
#      correct, since they move it somewhere else;
#   4. delisting last — it removes the position, and anything owed on the same
#      date has to be paid before the shares stop existing.
_ACTION_ORDER = {
    CorporateActionType.SPLIT: 0,
    CorporateActionType.STOCK_DIVIDEND: 1,
    CorporateActionType.CASH_DIVIDEND: 2,
    CorporateActionType.SYMBOL_CHANGE: 3,
    CorporateActionType.MERGER: 4,
    CorporateActionType.DELISTING: 5,
}


class CorporateActionSchedule:
    """Corporate actions indexed by ex-date.

    Sessions without an action are the common case, so the lookup is a dict hit
    on the session date rather than a scan.
    """

    def __init__(self, actions: Iterable[CorporateAction] = ()) -> None:
        by_date: dict[date, list[CorporateAction]] = defaultdict(list)
        for action in actions:
            by_date[action.ex_date].append(action)
        self._by_date = {
            ex_date: tuple(sorted(items, key=lambda item: _ACTION_ORDER[item.action_type]))
            for ex_date, items in by_date.items()
        }
        self._count = sum(len(items) for items in self._by_date.values())

    def __len__(self) -> int:
        return self._count

    def __bool__(self) -> bool:
        return self._count > 0

    def __iter__(self):
        """Iterate in ex-date order, then in the order applied within a session."""
        for ex_date in sorted(self._by_date):
            yield from self._by_date[ex_date]

    def actions_on(self, session: date) -> tuple[CorporateAction, ...]:
        return self._by_date.get(session, ())

    @property
    def symbols(self) -> frozenset[str]:
        return frozenset(action.symbol for actions in self._by_date.values() for action in actions)
