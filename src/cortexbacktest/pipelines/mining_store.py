"""Immutable local checkpoints for mining; the task runner remains the store owner.

Writing the manifest last is the completion marker. Interrupted directories are
retained for diagnosis and never treated as a completed selection.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from hashlib import sha256
import io
import json
from pathlib import Path

import polars as pl

from ..event.manifest import engine_code_digest
from ..research.mining import MiningFactorSpec, MiningResult


def save_mining_result(result: MiningResult, directory: str | Path) -> str:
    """Freeze selection before outer evaluation; return manifest byte digest.

    A new directory is required. The caller retains the returned digest outside
    this directory (for example in the task runner's job receipt).
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=False)
    trials = target / "trials.parquet"
    result.trials.write_parquet(trials)
    manifest = {
        "version": 1, "engine_code_sha256": engine_code_digest(),
        "trials_sha256": sha256(trials.read_bytes()).hexdigest(),
        "selected": [asdict(spec) for spec in result.selected],
        "inner_sha256": result.inner_sha256, "inner_end": result.inner_end.isoformat(),
        "inner_availability_sha256": result.inner_availability_sha256,
        "inner_revision_sha256": result.inner_revision_sha256,
        "inner_execution_sha256": result.inner_execution_sha256,
        "inner_end_type": "datetime" if isinstance(result.inner_end, datetime) else "date",
        "horizon": result.horizon, "cost_bps": result.cost_bps, "settings": result.settings,
    }
    content = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with (target / "manifest.json").open("xb") as output:
        output.write(content)
    return sha256(content).hexdigest()


def load_mining_result(directory: str | Path, *, manifest_sha256: str) -> MiningResult:
    """Restore a frozen selection using the independently retained receipt.

    Code drift requires a new study. This is completed-selection recovery, not
    resumption of an interrupted proposal search or proof of historical PIT.
    """
    target = Path(directory)
    content = (target / "manifest.json").read_bytes()
    if sha256(content).hexdigest() != manifest_sha256:
        raise ValueError("mining_manifest_digest_mismatch")
    manifest = json.loads(content)
    if manifest["version"] != 1:
        raise ValueError("unsupported_mining_checkpoint_version")
    if manifest["engine_code_sha256"] != engine_code_digest():
        raise ValueError("mining_engine_changed")
    trials = (target / "trials.parquet").read_bytes()
    if sha256(trials).hexdigest() != manifest["trials_sha256"]:
        raise ValueError("mining_trials_digest_mismatch")
    end_type = {"date": date, "datetime": datetime}[manifest["inner_end_type"]]
    return MiningResult(
        tuple(MiningFactorSpec(**spec) for spec in manifest["selected"]),
        pl.read_parquet(io.BytesIO(trials)), manifest["inner_sha256"],
        end_type.fromisoformat(manifest["inner_end"]), manifest["horizon"],
        manifest["cost_bps"], manifest["settings"], manifest.get("inner_availability_sha256"),
        manifest.get("inner_revision_sha256"),
        manifest.get("inner_execution_sha256"),
    )
