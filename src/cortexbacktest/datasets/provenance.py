"""Point-in-time provenance contracts for persisted research datasets."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """Identity of a vendor dataset as it was known at a specific instant.

    ``as_of`` is intentionally mandatory: a dataset without a knowledge time
    cannot establish that a historical backtest avoided later revisions.
    """

    vendor: str
    dataset: str
    as_of: datetime
    version: str
    revision: str | None = None
    license_id: str | None = None
    adjustment_mode: str | None = None
    calendar_version: str | None = None
    universe_version: str | None = None
    corporate_action_version: str | None = None

    def __post_init__(self) -> None:
        if not self.vendor.strip() or not self.dataset.strip() or not self.version.strip():
            raise ValueError("vendor, dataset, and version must be non-empty")
        timestamp = _parse_timestamp(self.as_of)
        if timestamp.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        object.__setattr__(self, "as_of", timestamp)

    def require_verified_versions(self) -> None:
        """Reject absent evidence labels before marking research inputs verified."""
        versions = (self.adjustment_mode, self.calendar_version,
                    self.universe_version, self.corporate_action_version)
        if any(not isinstance(value, str) or not value.strip() or value.strip().lower() == "unknown"
               for value in versions):
            raise ValueError(
                "Strict research requires verified adjustment/calendar/universe/corporate-action versions"
            )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["as_of"] = self.as_of.isoformat()
        return result

    @classmethod
    def from_dict(cls, value: dict) -> "DatasetProvenance":
        return cls(
            vendor=str(value["vendor"]),
            dataset=str(value["dataset"]),
            as_of=_parse_timestamp(value["as_of"]),
            version=str(value["version"]),
            revision=value.get("revision"),
            license_id=value.get("license_id"),
            adjustment_mode=value.get("adjustment_mode"),
            calendar_version=value.get("calendar_version"),
            universe_version=value.get("universe_version"),
            corporate_action_version=value.get("corporate_action_version"),
        )


def sha256_file(path: str | Path) -> str:
    """Hash a persisted dataset component without loading it into memory."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_file_hashes(directory: str | Path, files: dict[str, str]) -> None:
    """Raise when any persisted component differs from its manifest digest."""
    root = Path(directory)
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Dataset component escapes snapshot directory")
        if not path.exists():
            raise FileNotFoundError(f"Dataset component is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Dataset component hash mismatch: {name}")


def _parse_timestamp(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
