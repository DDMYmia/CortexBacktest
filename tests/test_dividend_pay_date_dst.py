"""`_combine_at_date` must not freeze a pytz offset across a DST transition.

pytz timezones cache the UTC offset they were localized with; naively reusing
a pytz tzinfo object on a different date (as ``datetime.combine`` would) can
silently shift the result by an hour whenever a DST transition falls between
the two dates. This matters for dividend settlement: `_apply_corporate_action`
re-anchors the ex-date bar's wall-clock time onto the pay date, and that
result is compared against later bar timestamps in `Ledger.settle_cash`
(`pending.settles_at <= timestamp`) — an hour of drift is enough to move a
receivable's settlement across a session boundary.
"""

from datetime import date, datetime

import pytz

from cortexbacktest.event.broker import _combine_at_date


def test_pytz_offset_is_recomputed_for_the_target_date_across_dst():
    tz = pytz.timezone("America/New_York")
    # Localized in winter (EST, UTC-5).
    ex_date_ts = tz.localize(datetime(2024, 1, 3, 9, 30))
    assert ex_date_ts.utcoffset().total_seconds() == -5 * 3600

    # Pay date is after the US spring-forward (EDT, UTC-4).
    pay_date = date(2024, 4, 3)
    pay_at = _combine_at_date(pay_date, ex_date_ts)

    assert pay_at.date() == pay_date
    assert pay_at.hour == 9 and pay_at.minute == 30
    assert pay_at.utcoffset().total_seconds() == -4 * 3600


def test_naive_timestamp_is_unaffected():
    pay_at = _combine_at_date(date(2024, 4, 3), datetime(2024, 1, 3, 9, 30))
    assert pay_at == datetime(2024, 4, 3, 9, 30)


def test_zoneinfo_timestamp_recomputes_correctly_without_special_casing():
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("America/New_York")
    ex_date_ts = datetime(2024, 1, 3, 9, 30, tzinfo=tz)
    pay_at = _combine_at_date(date(2024, 4, 3), ex_date_ts)
    assert pay_at.utcoffset().total_seconds() == -4 * 3600
