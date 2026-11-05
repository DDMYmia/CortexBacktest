"""Event-feed semantics against the workspace's data-only adapter vectors."""
import json
from datetime import UTC, datetime
from pathlib import Path
import pytest
from cortexbacktest.contracts import Bar
from cortexbacktest.event.feed import ColumnarBarFeed

VECTORS = json.loads((Path(__file__).parent / 'fixtures/bar-adapter-vectors.json').read_text())
SPECIAL = {'nan': float('nan'), 'infinity': float('inf'), 'negative-infinity': -float('inf')}


@pytest.mark.parametrize('case', VECTORS['prices'], ids=lambda c: c['id'])
def test_price_vector(case):
    value = SPECIAL[case['special']] if 'special' in case else case['value']
    def parse():
        # Positive enclosing range; zero is rejected by the event execution boundary.
        return ColumnarBarFeed.from_bars([Bar('TEST', datetime(2026,9,15,tzinfo=UTC),value,12,.01,11)])
    if case['expected']['cortex'] == 'reject':
        with pytest.raises((ValueError, TypeError)):
            parse()
    else:
        result = next(iter(parse())).bars[0]
        assert result.open == value
        assert result.volume is None


@pytest.mark.parametrize('case', VECTORS['times'], ids=lambda c: c['id'])
def test_time_vector(case):
    def parse():
        stamp = datetime.fromisoformat(case['value'])
        return next(iter(ColumnarBarFeed.from_bars([Bar('TEST',stamp,10,12,9,11)]))).timestamp
    expected = case['expected']['cortex']
    if expected == 'reject':
        with pytest.raises((ValueError, TypeError)):
            parse()
    elif expected == 'naive':
        assert parse().tzinfo is None  # Retained label, not a verified UTC instant.
    else:
        assert parse().timestamp() * 1000 == case['epochMs']


@pytest.mark.parametrize('case', VECTORS['extras'], ids=lambda c: c['id'])
def test_unknown_volume_and_adjustment(case):
    def parse():
        bar = Bar('TEST', datetime(2026,9,15,tzinfo=UTC),10,12,9,11, **{case['field']:case['value']})
        return next(iter(ColumnarBarFeed.from_bars([bar]))).bars[0]
    if case['expected']['cortex'] == 'reject':
        with pytest.raises(TypeError):
            parse()
    else:
        assert getattr(parse(),case['field']) == case['value']


def test_feed_normalization_does_not_deduplicate_same_symbol_bars():
    bar = Bar('TEST',datetime(2026,9,15,tzinfo=UTC),10,12,9,11)
    feed = ColumnarBarFeed.from_bars([bar] * VECTORS['duplicates']['count'])
    assert len(next(iter(feed)).bars) == VECTORS['duplicates']['expected']['cortex']
