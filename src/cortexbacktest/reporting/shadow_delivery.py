"""Immutable diagnostic delivery; this module has no execution interface."""

from datetime import UTC, datetime
from hashlib import sha256
from importlib.resources import files
import json
import os
from pathlib import Path
import tempfile

from jsonschema import Draft202012Validator, FormatChecker


_SCHEMA = json.loads(files(__package__).joinpath("shadow-decision-v1.schema.json").read_text())
_VALIDATOR = Draft202012Validator(_SCHEMA, format_checker=FormatChecker())


def write_shadow_decision(document: dict, path: str | Path) -> dict:
    """Publish exclusively. Receipt-time policy and dedup belong to the consumer."""
    _VALIDATOR.validate(document)
    payload = {k: v for k, v in document.items() if k != "decisionSha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if sha256(canonical).hexdigest() != document["decisionSha256"]:
        raise ValueError("Shadow decision digest mismatch")
    symbols = [row["symbol"] for row in document["decisions"]]
    if symbols != sorted(set(symbols)):
        raise ValueError("Shadow symbols must be unique and sorted")
    for row in document["decisions"]:
        event = datetime.fromisoformat(row["eventTime"]).astimezone(UTC).isoformat()
        identity = [document["strategySha256"], document["accountRef"], row["symbol"], event]
        expected = sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
        if expected != row["signalId"]:
            raise ValueError("Shadow signal identity mismatch")
    content = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent) as staged:
        staged.write(content)
        staged.flush()
        os.fsync(staged.fileno())
        os.link(staged.name, path)
    return {"path": str(path.resolve()), "sha256": sha256(content).hexdigest(),
            "decisionSha256": document["decisionSha256"]}
