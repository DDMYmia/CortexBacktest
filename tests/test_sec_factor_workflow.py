from copy import deepcopy
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json

import numpy as np
import polars as pl
import pytest

from cortexbacktest.pipelines.factor_walk_forward import FactorWalkForwardConfig
from cortexbacktest.pipelines.factor_workflow import FactorWorkflowConfig
from cortexbacktest.pipelines.sec_factor_workflow import run_sec_factor_walk_forward
from test_sec_factors import documents


@pytest.fixture
def receipts(tmp_path):
    rng = np.random.default_rng(7)
    values = rng.uniform(10, 200, size=(6, 8))
    files, decisions = {}, []
    for day in range(12):
        cutoff = datetime(2026, 8, 3, tzinfo=UTC) + timedelta(days=day)
        batch = documents()
        for concept, doc in enumerate(batch):
            rows = []
            for issuer in range(8):
                row = deepcopy(doc['records'][0])
                row.update(instrument_id=str(issuer + 1), symbol=f'S{issuer}', as_of=cutoff.isoformat(),
                           factor_value_id=str(1 + concept * 8 + issuer), value=float(values[concept, issuer]))
                row['metadata']['cik'] = str(123 + issuer)
                row['event_id'] = row['event_id'].replace('sec:123:', f'sec:{123 + issuer}:')
                rows.append(row)
            doc.update(asOf=cutoff.isoformat(), records=rows, recordCount=len(rows))
            doc['scope']['symbols'] = [f'S{i}' for i in range(8)]
            path = tmp_path / f'{day}-{concept}.json'
            payload = json.dumps(doc).encode()
            path.write_bytes(payload)
            files.setdefault(cutoff, []).append((path, sha256(payload).hexdigest()))
        for issuer in range(8):
            decisions.append({'date': cutoff.date(), 'as_of': cutoff, 'instrument_id': str(issuer + 1),
                              'forward_return': float(rng.normal(0, 0.01))})
    return pl.DataFrame(decisions), files


def run(frame, files):
    return run_sec_factor_walk_forward(
        frame, files, duration='quarterly', max_age_days=180,
        split=FactorWalkForwardConfig(4, 4, 1),
        label_definition='synthetic next-session return; integration evidence only',
        data_id='synthetic-sec-receipts',
        config=FactorWorkflowConfig(min_names=8, quantiles=2, resamples=20, block_size=2))


def test_receipts_reach_oos_with_provenance_and_purged_boundaries(receipts):
    frame, files = receipts
    result = run(frame, files)
    assert len(result.folds) == 2
    assert len(result.oos.evaluations) == 4
    assert result.oos.daily['date'].n_unique() == 8
    assert result.folds[0]['train_end'] == datetime(2026, 8, 5).date()
    assert result.folds[0]['test_start'] == datetime(2026, 8, 7).date()
    assert len(result.manifest['sec_receipts']) == 12
    evidence = result.manifest['sec_receipts'][0]
    assert evidence['coverage']['output_ratios'] == 32
    assert evidence['files'][0]['sha256'] == next(iter(files.values()))[0][1]
    changed = run(frame.with_columns(pl.when(pl.col('date') >= datetime(2026, 8, 11).date())
                                    .then(-pl.col('forward_return')).otherwise(pl.col('forward_return'))
                                    .alias('forward_return')), files)
    assert result.folds[0] == changed.folds[0]


def test_modified_receipt_is_rejected_before_evaluation(receipts):
    frame, files = receipts
    path, _ = next(iter(files.values()))[0]
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='digest mismatch'):
        run(frame, files)


def test_missing_oos_cutoff_is_not_filled_from_another_receipt(receipts):
    frame, files = receipts
    del files[datetime(2026, 8, 8, tzinfo=UTC)]
    with pytest.raises(ValueError, match='missing sessions'):
        run(frame, files)
