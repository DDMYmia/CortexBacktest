"Re-use the account-report contract for cash-flow research, not a second ledger."

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker, ValidationError
import polars as pl

from ..reporting.cashflow_returns import CashflowPerformance, cashflow_performance
from .. import strict_json

VALIDATOR = Draft202012Validator(
    json.loads(Path(__file__).with_name("account-report-v1.schema.json").read_bytes()),
    format_checker=FormatChecker(),
)


def account_cashflow_performance(path: str | Path, *, sha256: str, fx_path: str | Path | None = None,
                                 fx_sha256: str | None = None) -> CashflowPerformance:
    if (fx_path is None) != (fx_sha256 is None):
        raise ValueError("account_fx_path_and_digest_required_together")
    frozen_fx = None
    if fx_path is not None:
        from .ecb_fx import load_ecb_fx
        frozen_fx = load_ecb_fx(fx_path, sha256=fx_sha256)
    with localcontext() as context:
        context.prec = 80
        return _account_cashflow_performance(path, sha256=sha256, frozen_fx=frozen_fx, fx_sha256=fx_sha256)


def _account_cashflow_performance(path: str | Path, *, sha256: str, frozen_fx=None, fx_sha256=None) -> CashflowPerformance:
    with Path(path).open('rb') as stream:
        raw = stream.read(8_000_001)
    if len(raw) > 8_000_000 or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("account report digest/size mismatch")

    doc = strict_json.loads(raw)
    try:
        VALIDATOR.validate(doc)
    except ValidationError as error:
        raise ValueError("invalid account-report-v1") from error
    report = doc["report"]
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != doc["sourceReportSha256"]:
        raise ValueError("account report source binding mismatch")
    block = report.get("daily_returns")
    base = block["base_currency"] if block else "unknown"
    reason = None
    if not block or block["basis"] != "broker_twr_with_ledger_flows":
        reason = "missing_daily_returns"
    elif not block["days"] or any(row["issues"] for row in block["days"]):
        reason = "incomplete_broker_return_evidence"
    elif (
        not report["aggregation"]["cash_transactions_covered"]
        or report["aggregation"]["excluded_cash_event_ids"]
        or set(report["missing_sections"]) & {"CashReport", "CashTransactions", "Transfers"}
    ):
        reason = "cashflow_or_transfer_coverage_unverified"
    schema = {
        "start": pl.Datetime("us", "UTC"),
        "end": pl.Datetime("us", "UTC"),
        "opening": pl.Float64,
        "closing": pl.Float64,
        "externalFlow": pl.Float64,
        "return": pl.Float64,
    }
    binding = {
        "inputSha256": sha256,
        "sourceReportSha256": doc["sourceReportSha256"],
        "ledgerSha256": report["ledger_sha256"],
        "currency": base,
        "scope": "broker_report_arithmetic_not_authenticated_account_performance",
    }
    evidence = {
        'cashTransactionsCovered': report['aggregation']['cash_transactions_covered'],
        'excludedCashEvents': len(report['aggregation']['excluded_cash_event_ids']),
        'missingSections': report['missing_sections'],
        'observedReturnDays': len(block.get('days', [])) if block else 0,
        'unresolvedReturnDays': sum(bool(day['issues']) for day in block.get('days', [])) if block else 0,
        'fxPolicy': 'original_broker_flow_rates_reference_comparison_does_not_replace_them',
    }
    binding['evidenceCoverage'] = evidence
    if reason:
        return CashflowPerformance(
            {
                **binding,
                "status": "insufficient_evidence",
                "reason": reason,
                "twr": None,
                "mwr": None,
                "modifiedDietz": None,
            },
            pl.DataFrame(schema=schema),
        )
    rows = block["days"]
    nav = []
    activity, convention_activity = [], []
    for index, row in enumerate(rows):
        prior, today = Decimal(row["prior_nav"]), Decimal(row["nav"])
        day = datetime.fromisoformat(row["date"]).replace(tzinfo=timezone.utc)
        sources = row.get("flow_sources", [])
        for source in sources:
            amount, rate = Decimal(source["amount"]), Decimal(source["rate"])
            if amount == 0:
                continue
            cash = {
                "flow_id": source["event_id"],
                "amount": float(amount * rate),
                "currency": base,
                "kind": "deposit" if amount > 0 else "withdrawal",
            }
            prior_day = datetime.fromisoformat(row["prior_date"]).replace(tzinfo=timezone.utc)
            activity.append({**cash, "timestamp": day})
            convention_activity.append({**cash, "timestamp": prior_day})
        if index == 0:
            nav.append(
                {
                    "timestamp": datetime.fromisoformat(row["prior_date"]).replace(
                        tzinfo=timezone.utc
                    ),
                    "value": float(prior),
                    "currency": base,
                }
            )
        nav.append(
            {
                "timestamp": day,
                "value": float(today),
                "currency": base,
            }
        )
    valuations = pl.DataFrame(nav)
    flow_schema = {
        "flow_id": pl.String,
        "amount": pl.Float64,
        "currency": pl.String,
        "kind": pl.String,
        "timestamp": pl.Datetime("us", "UTC"),
    }
    fx_comparison = None
    if frozen_fx is not None:
        from datetime import time
        comparisons = []
        for day in block['days']:
            rate_date = datetime.fromisoformat(day['date']).date()
            # Broker dates do not prove transaction instants; this comparison is explicitly end-of-day.
            available_at = datetime.combine(rate_date, time.max, timezone.utc)
            for source in day.get('flow_sources', []):
                try:
                    converted = frozen_fx.convert(source['amount'], source_currency=source['currency'],
                        target_currency=base, rate_date=rate_date, available_at=available_at)
                except ValueError as error:
                    if str(error) not in {'fx_currency_outside_frozen_scope', 'fx_conversion_date_outside_frozen_scope',
                                          'fx_conversion_clock_outside_frozen_scope'}:
                        raise
                    converted = {'status': 'unavailable', 'reason': str(error)}
                row = {'eventId': source['event_id'], 'date': day['date'], 'sourceCurrency': source['currency'],
                       'brokerRate': source['rate'], **converted}
                if converted['status'] == 'available':
                    broker_amount = Decimal(source['amount']) * Decimal(source['rate'])
                    row['brokerConvertedAmount'] = str(broker_amount)
                    row['referenceMinusBrokerAmount'] = str(Decimal(converted['amount']) - broker_amount)
                comparisons.append(row)
        fx_comparison = {'inputSha256': fx_sha256, 'basis': 'ECB_reference_comparison_only_broker_returns_unchanged',
                         'clockPolicy': 'received_by_end_of_broker_report_date_not_actual_transaction_time',
                         'rows': comparisons, 'available': sum(row['status'] == 'available' for row in comparisons),
                         'unavailable': sum(row['status'] != 'available' for row in comparisons)}
    cutoff = nav[-1]["timestamp"]
    conventional = cashflow_performance(
        valuations,
        pl.DataFrame(convention_activity, schema=flow_schema),
        currency=base,
        coverage_complete=True,
        as_of=cutoff,
        flow_timing="start",
        allow_zero_capital=True,
    )
    dated = cashflow_performance(
        valuations,
        pl.DataFrame(activity, schema=flow_schema),
        currency=base,
        coverage_complete=True,
        as_of=cutoff,
        flow_timing="end",
        allow_zero_capital=True,
    )
    status = (
        "available"
        if conventional.summary["twr"]["status"] == dated.summary["mwr"]["status"] == "available"
        else "partial"
    )
    return CashflowPerformance(
        {
            **dated.summary,
            **binding,
            "status": status,
            "twr": {
                **conventional.summary["twr"],
                "basis": "declared_broker_start_of_day_convention",
            },
            "twrClockPolicy": "symbolic_day_boundaries_not_actual_transaction_instants",
            "mwrClockPolicy": "broker_report_dates_day_resolution",
            "fxPolicy": "explicit_broker_report_conversion_not_verified_spot_or_received_pit",
            **({"fxReferenceComparison": fx_comparison} if fx_comparison is not None else {}),
        },
        conventional.periods,
    )
