"""SEC ratios from complete, cutoff-specific observation receipt snapshots."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from math import isfinite
from numbers import Integral
from typing import Sequence

import polars as pl

from .observation_file import validate_observation_file


POLICY = "sec-us-gaap-usd-date-bound-v1"
CONCEPTS = {
    "sec.revenue": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "sec.net_income": "NetIncomeLoss",
    "sec.operating_income": "OperatingIncomeLoss",
    "sec.assets": "Assets",
    "sec.liabilities": "Liabilities",
    "sec.cash": "CashAndCashEquivalentsAtCarryingValue",
}
RATIOS = {
    "sec.net_margin": ("sec.net_income", "sec.revenue"),
    "sec.operating_margin": ("sec.operating_income", "sec.revenue"),
    "sec.liabilities_to_assets": ("sec.liabilities", "sec.assets"),
    "sec.cash_to_assets": ("sec.cash", "sec.assets"),
}


@dataclass(frozen=True, slots=True)
class SecRatioSnapshot:
    observations: pl.DataFrame
    coverage: dict[str, object]


def build_sec_ratio_snapshot(
    documents: Sequence[dict], *, as_of: datetime, max_age_days: int,
    duration: str,
) -> SecRatioSnapshot:
    """Pair identical filing contexts; never join by ticker or period end alone.

    Supply one complete observation-file-v1 per required concept with identical
    scope and cutoff, using the existing digest-verified file loader. Duration
    is explicitly annual (330..380 days) or quarterly (70..110 days); no YTD
    subtraction or TTM reconstruction is inferred. Instant ratios need no span.
    Results may only join decisions on both instrument_id AND as_of.
    """
    if (isinstance(max_age_days, bool) or not isinstance(max_age_days, Integral)
            or max_age_days < 1 or duration not in {"annual", "quarterly"}):
        raise ValueError("positive integer max_age_days and annual/quarterly duration are required")
    groups, excluded, accepted = {}, Counter(), Counter()
    factors, shared_scope = set(), None
    count = 0
    for document in documents:
        validate_observation_file(document, as_of=as_of)
        if document["recordVersion"] != "factor-observations-v1":
            raise ValueError("SEC ratios require factor observation files")
        scope = document["scope"]
        factor = scope["factorName"]
        if factor not in CONCEPTS or factor in factors:
            raise ValueError("supply exactly one snapshot per required SEC concept")
        factors.add(factor)
        identity_scope = (scope["start"], scope["end"], scope["symbolSource"], tuple(sorted(scope["symbols"])))
        if shared_scope is not None and identity_scope != shared_scope:
            raise ValueError("all SEC concept snapshots must declare the same scope")
        shared_scope = identity_scope
        for row in document["records"]:
            count += 1
            meta = row["metadata"] or {}
            if (row["source"] != "sec-edgar" or meta.get("definition") != POLICY
                    or meta.get("taxonomy") != "us-gaap" or meta.get("unit") != "USD"
                    or meta.get("tag") != CONCEPTS[factor] or not meta.get("cik")):
                excluded["policy_or_unit"] += 1
                continue
            end = datetime.fromisoformat(row["event_time"]).astimezone(UTC).date()
            try:
                filed = date.fromisoformat(meta["filed"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("SEC availability requires a valid filed date") from exc
            source_bound = datetime.combine(filed, time(12), UTC) + timedelta(days=1)
            if (filed < end
                    or meta.get("availability_policy") != "receipt-and-source-date-end-utc-minus-12"
                    or datetime.fromisoformat(row["score_known_at"]) != source_bound):
                raise ValueError("SEC availability contradicts the declared date-only policy")
            start = date.fromisoformat(meta["period_start"]) if meta.get("period_start") else None
            instant = factor in {"sec.assets", "sec.liabilities", "sec.cash"}
            bounds = (330, 380) if duration == "annual" else (70, 110)
            if (instant and start is not None or not instant
                    and (start is None or not bounds[0] <= (end - start).days <= bounds[1])):
                excluded["period_kind"] += 1
                continue
            expected = f"sec:{meta['cik']}:us-gaap:{CONCEPTS[factor]}:USD:{start or 'instant'}:{end}"
            if row["event_id"] != expected:
                excluded["event_identity"] += 1
                continue
            if (as_of.astimezone(UTC).date() - end).days > max_age_days:
                excluded["stale"] += 1
                continue
            # A single accession and CIK prevents combining different issuers or
            # mixing a restated numerator with an earlier denominator.
            key = (row["instrument_id"], meta["cik"], start, end, row["source_version"])
            context = groups.setdefault(key, {})
            if factor in context:
                raise ValueError("ambiguous SEC concept for one instrument/filing context")
            context[factor] = row
            accepted[factor] += 1
    if factors != set(CONCEPTS):
        raise ValueError(f"missing SEC concept snapshots: {sorted(set(CONCEPTS) - factors)}")
    candidates = {}
    seen_ranks = set()
    for (instrument, cik, start, end, accession), context in groups.items():
        for name, (numerator, denominator) in RATIOS.items():
            if numerator not in context or denominator not in context:
                if numerator in context or denominator in context:
                    excluded["unpaired_context"] += 1
                continue
            a, b = context[numerator], context[denominator]
            known = max(datetime.fromisoformat(a["known_at"]), datetime.fromisoformat(b["known_at"]))
            key = (instrument, name)
            rank = (end, known, accession)
            rank_key = (instrument, name, end, known)
            if rank_key in seen_ranks:
                raise ValueError("ambiguous SEC filing contexts at the same availability time")
            seen_ranks.add(rank_key)
            if key not in candidates or rank > candidates[key][0]:
                candidates[key] = (rank, a, b, cik, start)
    records = []
    for (instrument, name), ((end, known, accession), a, b, cik, start) in sorted(candidates.items()):
        # Do not fall back to an older positive denominator when the latest
        # complete context has an unusable one.
        if b["value"] <= 0:
            excluded["nonpositive_denominator"] += 1
            continue
        value = a["value"] / b["value"]
        if not isfinite(value):
            excluded["nonfinite_ratio"] += 1
            continue
        records.append({"instrument_id": instrument, "as_of": as_of, "factor": name,
                        "value": value, "known_at": known, "period_end": end,
                        "period_start": start, "cik": cik, "accession": accession,
                        "numerator_id": a["factor_value_id"], "denominator_id": b["factor_value_id"]})
    schema = {"instrument_id": pl.String, "as_of": pl.Datetime("us", "UTC"),
              "factor": pl.String, "value": pl.Float64, "known_at": pl.Datetime("us", "UTC"),
              "period_end": pl.Date, "period_start": pl.Date, "cik": pl.String,
              "accession": pl.String, "numerator_id": pl.String, "denominator_id": pl.String}
    return SecRatioSnapshot(pl.DataFrame(records, schema=schema), {
        "as_of": as_of.isoformat(), "input_records": count, "output_ratios": len(records),
        "excluded": dict(sorted(excluded.items())), "duration": duration,
        "accepted_by_concept": {factor: accepted[factor] for factor in CONCEPTS},
        "max_age_days": max_age_days, "policy": POLICY,
        "scope": shared_scope, "missing_ratios_are_null": True,
    })


def join_sec_ratio_panel(decisions: pl.DataFrame, snapshots: Sequence[SecRatioSnapshot]) -> pl.DataFrame:
    """Attach only exact-cutoff receipts, preserving missing ratios as null.

    Decisions carry date, instrument_id, as_of and forward_return. The workflow
    symbol key is the canonical instrument ID, not a mutable provider ticker.
    """
    required = {"date", "instrument_id", "as_of", "forward_return"}
    if not required <= set(decisions.columns):
        raise ValueError(f"missing decision columns: {sorted(required - set(decisions.columns))}")
    if collisions := set(RATIOS).intersection(decisions.columns):
        raise ValueError(f"decision panel already contains SEC ratio columns: {sorted(collisions)}")
    if any(decisions[column].null_count() for column in ("date", "instrument_id", "as_of")):
        raise ValueError("decision identity requires non-null date, instrument_id and as_of")
    if decisions.select(pl.struct("date", "instrument_id").is_duplicated().any()).item():
        raise ValueError("duplicate decision identity")
    if not snapshots:
        raise ValueError("at least one validated SEC snapshot is required")
    observations = pl.concat([snapshot.observations for snapshot in snapshots])
    if observations.select(pl.struct("instrument_id", "as_of", "factor").is_duplicated().any()).item():
        raise ValueError("duplicate SEC snapshot identity")
    wide = observations.pivot(on="factor", index=["instrument_id", "as_of"], values="value")
    wide = wide.with_columns([pl.lit(None, dtype=pl.Float64).alias(name)
                              for name in RATIOS if name not in wide.columns])
    return decisions.join(wide, on=["instrument_id", "as_of"], how="left").with_columns(
        pl.col("instrument_id").alias("symbol"))
