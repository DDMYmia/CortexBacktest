"""Reading evidence files: a repeated key is an error, and every clock carries its offset."""

from datetime import UTC, datetime
import json


def loads(raw):
    return json.loads(raw, object_pairs_hook=_unique)


def clock(value):
    """An ISO timestamp from an evidence file, returned in UTC; naive or non-string values are rejected."""
    if not isinstance(value, str):
        raise ValueError("ISO clock string required")
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() is None:
        raise ValueError("aware clock required")
    return stamp.astimezone(UTC)


def _unique(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON key")
    return result
