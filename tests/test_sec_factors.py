from copy import deepcopy
from datetime import UTC, datetime, timedelta
from itertools import permutations

import polars as pl
import pytest

from cortexbacktest.adapters.sec_factors import (
    CONCEPTS, POLICY, build_sec_ratio_snapshot, join_sec_ratio_panel,
)


CUTOFF = datetime(2026, 9, 22, tzinfo=UTC)


def documents():
    values = [100.0, 10.0, 20.0, 200.0, 50.0, 30.0]
    result = []
    for index, ((factor, tag), value) in enumerate(zip(CONCEPTS.items(), values, strict=True)):
        start = "2026-04-01" if index < 3 else None
        row = {
            "contractVersion": "factor-observations-v1", "factor_value_id": str(index + 1),
            "instrument_id": "1", "entity_receipt_id": None, "symbol": "ABC",
            "symbol_source": "fmp", "factor_name": factor, "source": "sec-edgar",
            "source_version": "filing-1", "market_mic": "XNAS", "as_of": CUTOFF.isoformat(),
            "event_time": "2026-06-30T23:59:00Z", "value": value,
            "event_id": f"sec:123:us-gaap:{tag}:USD:{start or 'instant'}:2026-06-30",
            "metadata": {"definition": POLICY, "taxonomy": "us-gaap", "unit": "USD",
                         "filed": "2026-07-31",
                         "availability_policy": "receipt-and-source-date-end-utc-minus-12",
                         "cik": "123", "tag": tag, "period_start": start},
        }
        for stamp in ("score_known_at", "recorded_at", "entity_known_at", "identity_known_at", "known_at"):
            row[stamp] = "2026-08-01T12:00:00Z"
        result.append({"contractVersion": "observation-file-v1", "recordVersion": "factor-observations-v1",
                       "recordCount": 1, "asOf": CUTOFF.isoformat(),
                       "scope": {"start": "2026-01-01", "end": "2026-09-22",
                                 "factorName": factor, "symbolSource": "fmp", "symbols": ["ABC"]},
                       "records": [row]})
    return result


def build(docs, **kwargs):
    return build_sec_ratio_snapshot(docs, as_of=CUTOFF, max_age_days=180,
                                    duration=kwargs.get("duration", "quarterly"))


def test_ratios_keep_period_identity_and_receipt_provenance():
    result = build(documents())
    values = dict(result.observations.select("factor", "value").iter_rows())
    assert values == {"sec.net_margin": 0.1, "sec.operating_margin": 0.2,
                      "sec.liabilities_to_assets": 0.25, "sec.cash_to_assets": 0.15}
    assert result.observations["numerator_id"].null_count() == 0
    assert result.coverage["input_records"] == 6


@pytest.mark.parametrize("change,reason", [
    ({"unit": "EUR"}, "policy_or_unit"),
    ({"definition": "old"}, "policy_or_unit"),
    ({"period_start": "2026-01-01"}, "period_kind"),
])
def test_units_policy_and_ytd_do_not_silently_mix(change, reason):
    docs = documents()
    docs[1]["records"][0]["metadata"].update(change)
    result = build(docs)
    assert "sec.net_margin" not in result.observations["factor"].to_list()
    assert result.coverage["excluded"][reason] == 1


def test_revisions_must_pair_with_same_filing():
    docs = documents()
    docs[1]["records"][0]["source_version"] = "amendment"
    assert "sec.net_margin" not in build(docs).observations["factor"].to_list()


def test_new_snapshot_cannot_be_reused_for_old_decision():
    snapshot = build(documents())
    decisions = pl.DataFrame({"date": [1, 2], "instrument_id": ["1", "1"],
                              "as_of": [CUTOFF - timedelta(days=1), CUTOFF],
                              "forward_return": [0.01, 0.02]})
    panel = join_sec_ratio_panel(decisions, [snapshot]).sort("date")
    assert panel["sec.net_margin"].to_list() == [None, 0.1]
    assert panel["symbol"].to_list() == ["1", "1"]


def test_empty_receipts_preserve_missing_values():
    docs = documents()
    for document in docs:
        document["records"], document["recordCount"] = [], 0
    snapshot = build(docs)
    decisions = pl.DataFrame({"date": [1], "instrument_id": ["1"], "as_of": [CUTOFF],
                              "forward_return": [0.01]})
    panel = join_sec_ratio_panel(decisions, [snapshot])
    assert panel["sec.net_margin"].to_list() == [None]


def test_future_receipt_fails_validation():
    docs = documents()
    for field in ("recorded_at", "known_at"):
        docs[1]["records"][0][field] = (CUTOFF + timedelta(days=1)).isoformat()
    with pytest.raises(ValueError, match="exceeds"):
        build(docs)


@pytest.mark.parametrize("filed", [None, "not-a-date", "2026-06-29", "2026-08-01"])
def test_declared_sec_policy_requires_filing_date_evidence(filed):
    docs = documents()
    docs[1]["records"][0]["metadata"]["filed"] = filed
    with pytest.raises(ValueError, match="SEC availability"):
        build(docs)


def test_general_receipt_validation_does_not_replace_sec_source_bound():
    docs = documents()
    # Every receipt is before the cutoff and after the period end, but the
    # source claim is a second earlier than the declared date-only policy.
    docs[1]["records"][0]["score_known_at"] = "2026-08-01T11:59:59Z"
    with pytest.raises(ValueError, match="SEC availability"):
        build(docs)


@pytest.mark.parametrize("field", ["filed", "availability_policy"])
def test_missing_sec_availability_evidence_is_not_inferred(field):
    docs = documents()
    del docs[1]["records"][0]["metadata"][field]
    with pytest.raises(ValueError, match="SEC availability"):
        build(docs)


def test_equivalent_offset_and_later_receipt_keep_valid_source_bound():
    docs = documents()
    row = docs[1]["records"][0]
    row["score_known_at"] = "2026-08-01T08:00:00-04:00"
    row["recorded_at"] = row["known_at"] = "2026-09-01T12:00:00Z"
    result = build(docs).observations.filter(pl.col("factor") == "sec.net_margin")
    assert result["known_at"][0] == datetime(2026, 9, 1, 12, tzinfo=UTC)


def test_incomplete_export_and_missing_concept_fail():
    docs = documents()
    docs[0]["recordCount"] = 5001
    with pytest.raises(ValueError, match="record count"):
        build(docs)
    with pytest.raises(ValueError, match="missing SEC concept"):
        build(documents()[:-1])


def test_duplicate_snapshot_cannot_multiply_decision_rows():
    snapshot = build(documents())
    decisions = pl.DataFrame({"date": [1], "instrument_id": ["1"], "as_of": [CUTOFF],
                              "forward_return": [0.01]})
    with pytest.raises(ValueError, match="duplicate SEC"):
        join_sec_ratio_panel(decisions, [snapshot, deepcopy(snapshot)])


def test_stale_or_nonpositive_denominators_do_not_produce_ratios():
    docs = documents()
    docs[3]["records"][0]["value"] = 0.0
    result = build(docs)
    assert result.coverage["excluded"]["nonpositive_denominator"] == 2
    stale = build_sec_ratio_snapshot(documents(), as_of=CUTOFF, max_age_days=1, duration="quarterly")
    assert stale.observations.is_empty()
    assert stale.coverage["excluded"]["stale"] == 6


def test_two_complete_filings_with_equal_availability_are_ambiguous():
    docs = documents()
    # Distinct issuer contexts mapped onto one instrument are not resolved by
    # lexicographic accession order.
    for document in docs:
        extra = deepcopy(document["records"][0])
        extra["metadata"]["cik"] = "456"
        extra["event_id"] = extra["event_id"].replace("sec:123:", "sec:456:")
        extra["source_version"] = "filing-2"
        document["records"].append(extra)
        document["recordCount"] = 2
    with pytest.raises(ValueError, match="ambiguous SEC"):
        build(docs)


@pytest.mark.parametrize("order", list(permutations(range(3))))
@pytest.mark.parametrize("ambiguous", [True, False])
def test_filing_selection_does_not_depend_on_record_order(order, ambiguous):
    docs = documents()
    for document in docs:
        rows = []
        for index in range(3):
            row = deepcopy(document["records"][0])
            cik = str(123 + index)
            row["metadata"]["cik"] = cik
            row["event_id"] = row["event_id"].replace("sec:123:", f"sec:{cik}:")
            row["source_version"] = f"filing-{index}"
            row["factor_value_id"] += str(index)
            day = 1 if ambiguous and index < 2 else index + 1
            row["recorded_at"] = row["known_at"] = f"2026-09-0{day}T12:00:00Z"
            rows.append(row)
        document["records"] = [rows[index] for index in order]
        document["recordCount"] = 3
    if ambiguous:
        with pytest.raises(ValueError, match="ambiguous SEC"):
            build(docs)
    else:
        result = build(docs).observations
        assert result.height == 4
        assert result["accession"].unique().to_list() == ["filing-2"]
        assert result["known_at"].unique().to_list() == [datetime(2026, 9, 3, 12, tzinfo=UTC)]


@pytest.mark.parametrize('age', [True, False, 0, -1, 1.5, 180.0, float('nan'),
                                 float('inf'), -float('inf'), None, '180'])
def test_age_policy_rejects_nonpositive_or_noninteger_configuration(age):
    with pytest.raises(ValueError, match='max_age_days'):
        build_sec_ratio_snapshot(documents(), as_of=CUTOFF, max_age_days=age, duration='quarterly')


@pytest.mark.parametrize('field', ['date', 'instrument_id', 'as_of'])
def test_missing_decision_identity_is_not_reported_as_missing_sec_coverage(field):
    snapshot = build(documents())
    decisions = pl.DataFrame({'date': [1], 'instrument_id': ['1'], 'as_of': [CUTOFF],
                              'forward_return': [None]}, schema_overrides={'forward_return': pl.Float64})
    decisions = decisions.with_columns(pl.lit(None).cast(decisions.schema[field]).alias(field))
    with pytest.raises(ValueError, match='decision identity'):
        join_sec_ratio_panel(decisions, [snapshot])


def test_missing_forward_label_does_not_erase_available_sec_ratios():
    snapshot = build(documents())
    decisions = pl.DataFrame({'date': [1], 'instrument_id': ['1'], 'as_of': [CUTOFF],
                              'forward_return': [None]}, schema_overrides={'forward_return': pl.Float64})
    panel = join_sec_ratio_panel(decisions, [snapshot])
    assert panel['forward_return'].null_count() == 1
    assert panel['sec.net_margin'].to_list() == [0.1]


@pytest.mark.parametrize('factor', ['sec.net_margin', 'sec.operating_margin',
                                   'sec.liabilities_to_assets', 'sec.cash_to_assets'])
def test_existing_sec_columns_cannot_bypass_exact_cutoff_receipts(factor):
    snapshot = build(documents())
    # The new receipt cannot supply this older decision. A pre-attached value
    # must not survive under the factor name while the valid null gets _right.
    decisions = pl.DataFrame({'date': [1], 'instrument_id': ['1'],
                              'as_of': [CUTOFF - timedelta(days=1)],
                              'forward_return': [0.01], factor: [999.0]})
    with pytest.raises(ValueError, match='already contains SEC ratio columns'):
        join_sec_ratio_panel(decisions, [snapshot])


def test_unrelated_factors_survive_sec_attachment():
    snapshot = build(documents())
    decisions = pl.DataFrame({'date': [1], 'instrument_id': ['1'], 'as_of': [CUTOFF],
                              'forward_return': [0.01], 'momentum': [0.42]})
    panel = join_sec_ratio_panel(decisions, [snapshot])
    assert panel['momentum'].to_list() == [0.42]
    assert panel['sec.net_margin'].to_list() == [0.1]
