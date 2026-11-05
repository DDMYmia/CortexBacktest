"""Read frozen received ECB references; explicit dates and clocks, no rate filling."""
import csv
import hashlib
import io
import math
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from .. import strict_json


def _load(path, sha):
    if path.is_symlink():
        raise ValueError('fx_artifact_symlink')
    with path.open('rb') as stream:
        raw = stream.read(32_000_001)
    if len(raw) > 32_000_000 or hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('fx_artifact_digest_or_size')
    return strict_json.loads(raw), len(raw)


def load_ecb_fx(path, *, sha256):
    path = Path(path)
    manifest, size = _load(path, sha256)
    if (manifest['contractVersion'] != 'frozen-ecb-fx-v1' or manifest['baseCurrency'] != 'EUR'
            or manifest['quoteConvention'] != 'quote_currency_per_one_EUR'
            or manifest['qualificationIssued'] is not False or manifest['executionAuthorized'] is not False
            or manifest['missingDatePolicy'] != 'unavailable_no_forward_fill'):
        raise ValueError('fx_contract_invalid')
    cutoff = strict_json.clock(manifest['asOf'])
    latest = {}
    seen = set()
    for name, digest in manifest['files'].items():
        if Path(name).name != name or not name.startswith('receipt-') or name in {'.', '..'}:
            raise ValueError('fx_receipt_path_invalid')
        receipt, length = _load(path.with_name(name), digest)
        size += length
        if size > 64_000_000 or len(seen) >= 1000:
            raise ValueError('fx_bundle_limit')
        identity = receipt['receiptId']
        if type(identity) is not int or identity <= 0 or identity in seen:
            raise ValueError('fx_duplicate_receipt')
        seen.add(identity)
        doc = receipt['document']
        received, recorded = strict_json.clock(receipt['receivedAt']), strict_json.clock(receipt['recordedAt'])
        raw = (b'\xef\xbb\xbf' if doc['rawHadBom'] else b'') + doc['rawUtf8'].encode()
        if (doc['contractVersion'] != 'received-ecb-fx-v1' or doc['baseCurrency'] != 'EUR'
                or doc['quoteConvention'] != manifest['quoteConvention']
                or not received <= recorded <= cutoff or received != strict_json.clock(doc['receivedAt'])
                or hashlib.sha256(raw).hexdigest() != doc['rawSha256'] or doc['rawSha256'] != receipt['rawSha256']
                or doc['sourcePublishedAt'] is not None or doc['qualificationIssued'] is not False
                or doc['executionAuthorized'] is not False):
            raise ValueError('fx_receipt_binding_invalid')
        rows, keys = [], set()
        for item in csv.DictReader(io.StringIO(doc['rawUtf8'])):
            currency, day = item['CURRENCY'], date.fromisoformat(item['TIME_PERIOD']).isoformat()
            if (item['CURRENCY_DENOM'] != 'EUR' or item['FREQ'] != 'D' or item['EXR_TYPE'] != 'SP00'
                    or item['EXR_SUFFIX'] != 'A' or item.get('UNIT_MULT', '0') not in ('', '0')
                    or not doc['requestedStart'] <= day <= received.date().isoformat()
                    or (currency, day) in keys):
                raise ValueError('fx_raw_direction_date_invalid')
            keys.add((currency, day))
            value = float(item['OBS_VALUE']) if item['OBS_VALUE'].strip() else None
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError('fx_raw_rate_invalid')
            row = {'currency': currency, 'date': day, 'rate': value, 'status': 'available' if value is not None else 'missing_observation'}
            rows.append(row)
            scope = manifest['scope']
            if currency not in scope['currencies'] or not scope['start'] <= day <= scope['end']:
                continue
            key = currency, day
            candidate = {**row, 'receiptId': identity, 'rawSha256': receipt['rawSha256'],
                         'receivedAt': receipt['receivedAt'], 'recordedAt': receipt['recordedAt']}
            if key not in latest or (recorded, identity) > (strict_json.clock(latest[key]['recordedAt']), latest[key]['receiptId']):
                latest[key] = candidate
        if rows != doc['rates']:
            raise ValueError('fx_raw_projection_mismatch')
    expected = [latest[key] for key in sorted(latest)]
    if expected != manifest['rows']:
        raise ValueError('fx_selected_revision_mismatch')
    return FrozenECB(manifest, latest)


class FrozenECB:
    def __init__(self, manifest, rates):
        self.manifest, self.rates = manifest, rates

    def convert(self, amount, *, source_currency, target_currency, rate_date: date, available_at: datetime):
        if available_at.utcoffset() is None or available_at > strict_json.clock(self.manifest['asOf']):
            raise ValueError('fx_conversion_clock_outside_frozen_scope')
        day = rate_date.isoformat()
        scope = self.manifest['scope']
        if not scope['start'] <= day <= scope['end']:
            raise ValueError('fx_conversion_date_outside_frozen_scope')
        value = Decimal(str(amount))
        if not value.is_finite():
            raise ValueError('fx_amount_invalid')
        if source_currency == target_currency:
            if source_currency not in scope['currencies']:
                raise ValueError('fx_currency_outside_frozen_scope')
            return {'status': 'available', 'amount': str(value), 'currency': target_currency,
                    'rateDate': day, 'receiptIds': [], 'basis': 'same_currency_no_conversion'}
        factors, sources = {}, []
        for currency in {source_currency, target_currency}:
            if currency not in scope['currencies']:
                raise ValueError('fx_currency_outside_frozen_scope')
            if currency == 'EUR':
                factors[currency] = Decimal(1)
                continue
            row = self.rates.get((currency, day))
            if row is None or row['status'] != 'available' or strict_json.clock(row['recordedAt']) > available_at:
                return {'status': 'unavailable', 'reason': 'missing_or_not_yet_received_fx', 'currency': currency, 'date': day}
            factors[currency] = Decimal(str(row['rate']))
            sources.append(row['receiptId'])
        converted = value / factors[source_currency] * factors[target_currency]
        return {'status': 'available', 'amount': str(converted), 'currency': target_currency,
                'rateDate': day, 'receiptIds': sorted(set(sources)), 'basis': 'ECB_reference_not_broker_execution_rate'}
