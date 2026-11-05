"""A feed never interleaves naive clock labels with timezone-aware instants."""

from datetime import UTC, datetime

import pytest

from cortexbacktest.contracts import Bar
from cortexbacktest.event.feed import ColumnarBarFeed

NAIVE = Bar("A", datetime(2026, 9, 15, 9, 30), 10, 12, 9, 11)
AWARE = Bar("B", datetime(2026, 9, 15, 13, 30, tzinfo=UTC), 10, 12, 9, 11)


@pytest.mark.parametrize("assume_sorted", [False, True])
def test_mixed_awareness_is_rejected(assume_sorted):
    with pytest.raises(ValueError, match="mixes naive and timezone-aware"):
        ColumnarBarFeed.from_bars([NAIVE, AWARE], assume_sorted=assume_sorted)


@pytest.mark.parametrize("bar", [NAIVE, AWARE])
def test_uniform_feeds_still_load(bar):
    assert next(iter(ColumnarBarFeed.from_bars([bar]))).timestamp == bar.timestamp
