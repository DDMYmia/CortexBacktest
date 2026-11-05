"""Digest-bound SEC receipt files through exact-cutoff joins to purged OOS."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from ..adapters.observation_file import load_observation_file
from ..adapters.sec_factors import RATIOS, build_sec_ratio_snapshot, join_sec_ratio_panel
from .factor_walk_forward import FactorWalkForwardConfig, FactorWalkForwardResult, run_factor_walk_forward
from .factor_workflow import FactorWorkflowConfig


def run_sec_factor_walk_forward(
    decisions: pl.DataFrame,
    receipt_files: Mapping[datetime, Sequence[tuple[str | Path, str]]],
    *,
    duration: str,
    max_age_days: int,
    split: FactorWalkForwardConfig,
    label_definition: str,
    data_id: str,
    config: FactorWorkflowConfig | None = None,
) -> FactorWalkForwardResult:
    """Evaluate the four frozen SEC ratios from six concept files per cutoff.

    Each file is paired with its independently supplied SHA-256. Decisions must
    already carry canonical instrument IDs, aware as_of, dates and causal labels.
    No identity inference, receipt backdating or missing-value filling occurs.
    Coverage and receipt provenance accompany the existing OOS report; they do
    not certify historical universe, prices or the supplied return labels.
    """
    snapshots, provenance = [], []
    for cutoff, files in sorted(receipt_files.items()):
        documents = [load_observation_file(path, sha256=digest, as_of=cutoff)
                     for path, digest in files]
        snapshot = build_sec_ratio_snapshot(documents, as_of=cutoff,
                                           duration=duration, max_age_days=max_age_days)
        snapshots.append(snapshot)
        provenance.append({"as_of": cutoff.isoformat(), "coverage": snapshot.coverage,
                           "files": [{"path": str(path), "sha256": digest} for path, digest in files]})
    panel = join_sec_ratio_panel(decisions, snapshots)
    result = run_factor_walk_forward(panel, tuple(RATIOS), split=split,
                                    label_definition=label_definition, data_id=data_id, config=config)
    return replace(result, manifest={**result.manifest, "sec_receipts": provenance})
