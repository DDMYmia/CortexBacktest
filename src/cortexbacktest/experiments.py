"""Atomic JSON experiment checkpoints; no pickled objects or engine state."""
from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_id(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def callable_identity(value: Any, *, code_version: str | None = None) -> dict[str, Any]:
    if value is None:
        return {"callable": None}
    name = f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__qualname__', type(value).__qualname__)}"
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError):
        if code_version is None:
            raise ValueError(f"cannot fingerprint {name}; provide code_version for checkpointing") from None
        source = code_version
    closure = inspect.getclosurevars(value).nonlocals if inspect.isfunction(value) else {}
    try:
        closure_json = canonical_json(closure)
    except (TypeError, ValueError):
        if code_version is None:
            raise ValueError(f"non-JSON closure in {name}; provide code_version") from None
        closure_json = code_version
    return {"callable": name, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "closure": closure_json, "code_version": code_version}


class ExperimentStore:
    """Single-experiment SQLite journal, safe for concurrent worker writes.

    Completed cases survive interruption. Identity mismatches are rejected.
    Retried failures run on fresh engines; partial engine state is never resumed.
    """
    def __init__(self, path: str | Path, identity: dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        signature = content_id(identity)
        with closing(self._connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS experiment (id INTEGER PRIMARY KEY CHECK(id=1), signature TEXT NOT NULL, identity_json TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS cases (case_id TEXT PRIMARY KEY, params_json TEXT NOT NULL, status TEXT NOT NULL, row_json TEXT, error TEXT, attempts INTEGER NOT NULL, updated_at TEXT NOT NULL)")
            db.execute("INSERT OR IGNORE INTO experiment VALUES (1, ?, ?)", (signature, canonical_json(identity)))
            if db.execute("SELECT signature FROM experiment WHERE id=1").fetchone()[0] != signature:
                raise ValueError("checkpoint identity mismatch: data, code, seed, or configuration changed")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def get(self, case: dict[str, Any]) -> dict[str, Any] | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT row_json FROM cases WHERE case_id=? AND status='complete'", (content_id(case),)).fetchone()
            return None if row is None else json.loads(row[0])

    def record(self, case: dict[str, Any], *, row: dict[str, Any] | None = None, error: str | None = None):
        with closing(self._connect()) as db, db:
            db.execute("""INSERT INTO cases VALUES (?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(case_id) DO UPDATE SET status=excluded.status,
                row_json=excluded.row_json, error=excluded.error,
                attempts=cases.attempts+1, updated_at=excluded.updated_at""",
                (content_id(case), canonical_json(case), "complete" if error is None else "failed",
                 None if row is None else canonical_json(row), error, datetime.now(UTC).isoformat()))

    def run(self, case: dict[str, Any], function: Callable[[], dict[str, Any]], *, retries: int = 0):
        cached = self.get(case)
        if cached is not None:
            return cached
        for attempt in range(retries + 1):
            try:
                row = function()
                self.record(case, row=row)
                return row
            except Exception as exc:
                self.record(case, error=f"{type(exc).__name__}: {exc}")
                if attempt == retries:
                    raise
        raise AssertionError("unreachable")
