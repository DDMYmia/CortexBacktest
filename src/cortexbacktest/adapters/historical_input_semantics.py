"""Check field-level historical availability against frozen alias projections."""
import hashlib
import json
from datetime import date, datetime, time, UTC
from zoneinfo import ZoneInfo

from .. import strict_json


def validate_historical_inputs(document, bindings, *, dates, cutoff, reference_receipts=None):
    if document.get('version') != 1 or document.get('certified') is not False:
        raise ValueError('unqualified_historical_input_semantics_required')
    rows = document.get('rows')
    if not isinstance(rows, list) or len(rows) != len(bindings):
        raise ValueError('historical_input_evidence_grid_incomplete')
    seen = set()
    for row, binding in zip(rows, bindings, strict=True):
        day = binding['effectiveOn']
        key = (binding['symbol'], day)
        if key in seen or day not in dates:
            raise ValueError('historical_input_evidence_grid_invalid')
        seen.add(key)
        opening = datetime.combine(datetime.fromisoformat(day).date(), time(9, 30), ZoneInfo('America/New_York')).astimezone(UTC)
        known = binding.get('recordedAt')
        received = datetime.fromisoformat(known) if known is not None else None
        if received is not None and (received.utcoffset() is None or received > cutoff):
            raise ValueError('historical_identity_receipt_after_export')
        reason = ('identity_unresolved' if binding['status'] != 'resolved' else
                  'default_identity_window_not_historical_evidence' if binding.get('windowDefault') else
                  'identity_receipt_missing' if received is None else
                  'identity_received_after_historical_open' if received > opening else None)
        digest = hashlib.sha256(json.dumps(binding, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2).encode()).hexdigest()
        expected = {'symbol': binding['symbol'], 'instrumentId': binding.get('instrumentId'), 'effectiveOn': day,
                    'requiredAt': opening.isoformat(),
                    'identity': {'status': 'available_mapping' if reason is None else 'unavailable', 'reason': reason,
                                 'source': 'received_alias_resolution', 'recordedAt': known, 'validFrom': binding.get('validFrom'),
                                 'evidenceSha256': digest, 'scope': 'mapping_only_not_continuous_security_certification'},
                    'currency': {'status': 'unavailable', 'reason': 'historical_currency_effective_interval_missing'},
                    'shareClass': {'status': 'unavailable', 'reason': 'historical_share_class_effective_interval_missing'},
                    'adjustment': {'status': 'unavailable', 'reason': 'historical_adjustment_method_and_action_coverage_missing'}}
        if reference_receipts is not None and binding.get('instrumentId') is not None:
            expected['referenceClaims'] = {field: select_reference_claim(reference_receipts,
                int(binding['instrumentId']), field, day, min(opening, cutoff))
                for field in ('identity', 'currency', 'shareClass', 'adjustment')}
        if row != expected:
            raise ValueError('historical_input_evidence_projection_mismatch')


def received_reference_claims(document, *, instruments, cutoff):
    if document.get('contractVersion') != 'historical-reference-receipts-v1' or document.get('certified') is not False:
        raise ValueError('historical_reference_receipt_contract')
    receipts = document['receipts']
    if not isinstance(receipts, list) or len(receipts) > 1000:
        raise ValueError('historical_reference_receipt_limit')
    seen = set()
    for row in receipts:
        identity = row['receipt_id']
        if type(identity) is not int or identity <= 0 or identity in seen or row['instrument_id'] not in instruments:
            raise ValueError('historical_reference_receipt_identity')
        seen.add(identity)
        evidence = row['evidence']
        receipt_bytes = evidence['receipt_utf8'].encode()
        raw = evidence['raw_utf8'].encode()
        receipt = strict_json.loads(receipt_bytes)
        raw_document = strict_json.loads(raw)
        normalized = dict(raw_document)
        normalized['claims'] = []
        for claim in raw_document['claims']:
            normalized_claim = {'value': None, 'sourceKnownAt': None, 'operation': 'assert', **claim}
            if normalized_claim['sourceKnownAt'] is not None:
                normalized_claim['sourceKnownAt'] = datetime.fromisoformat(normalized_claim['sourceKnownAt']).isoformat().replace('+00:00', 'Z')
            normalized['claims'].append(normalized_claim)
        if (hashlib.sha256(receipt_bytes).hexdigest() != row['source_sha256']
                or receipt != evidence['receipt'] or receipt['contractVersion'] != 'research-source-receipt-v1'
                or receipt['dataset'] != 'historical_reference' or receipt['status'] != 'received'
                or receipt['rawSha256'] != hashlib.sha256(raw).hexdigest() or receipt['rawBytes'] != len(raw)
                or normalized != row['attributes']
                or evidence['semanticReview'] != 'normalized_claims_not_vendor_verified'
                or evidence['historical_identity_certified'] is not False):
            raise ValueError('historical_reference_raw_binding')
        received, recorded = (datetime.fromisoformat(row[key]) for key in ('received_at', 'recorded_at'))
        if (received.utcoffset() is None or recorded.utcoffset() is None or not received <= recorded <= cutoff
                or received != datetime.fromisoformat(receipt['receivedAt'])):
            raise ValueError('historical_reference_receipt_clocks')
        attributes = row['attributes']
        if (set(attributes) != {'contractVersion', 'instrumentId', 'claims'}
                or attributes['contractVersion'] != 'historical-reference-claims-v1'
                or type(attributes['instrumentId']) is not int or attributes['instrumentId'] != row['instrument_id']):
            raise ValueError('historical_reference_instrument_binding')
        claims = attributes['claims']
        if not isinstance(claims, list) or not 1 <= len(claims) <= 1000:
            raise ValueError('historical_reference_claim_limit')
        keys = set()
        for claim in claims:
            if (set(claim) != {'revisionKey', 'field', 'value', 'validFrom', 'validTo', 'sourceKnownAt', 'operation', 'sourceIdentifier'}
                    or any(not isinstance(claim[key], str) or not 1 <= len(claim[key]) <= limit
                           for key, limit in [('revisionKey', 128), ('sourceIdentifier', 256)])
                    or (claim['value'] is not None and (not isinstance(claim['value'], str) or not 1 <= len(claim['value']) <= 256))):
                raise ValueError('historical_reference_claim_fields')
            if claim['field'] == 'currency' and claim['value'] is not None and (
                    len(claim['value']) != 3 or not claim['value'].isascii() or not claim['value'].isalpha() or not claim['value'].isupper()):
                raise ValueError('historical_reference_currency_invalid')
            key = claim['field'], claim['revisionKey']
            if key in keys or claim['field'] not in {'identity', 'currency', 'shareClass', 'adjustment'}:
                raise ValueError('historical_reference_claim_identity')
            keys.add(key)
            if (date.fromisoformat(claim['validFrom']) >= date.fromisoformat(claim['validTo'])
                    or claim['operation'] not in {'assert', 'withdraw'}
                    or (claim['operation'] == 'assert' and not claim['value'])):
                raise ValueError('historical_reference_claim_interval')
            known = claim['sourceKnownAt']
            if known is not None:
                clock = datetime.fromisoformat(known)
                if clock.utcoffset() is None or clock > received:
                    raise ValueError('historical_reference_source_clock')
    return receipts


def select_reference_claim(receipts, instrument, field, day, cutoff):
    latest = {}
    for row in sorted(receipts, key=lambda row: (datetime.fromisoformat(row['recorded_at']), row['receipt_id'])):
        if row['instrument_id'] != instrument or datetime.fromisoformat(row['recorded_at']) > cutoff:
            continue
        for claim in row['attributes']['claims']:
            if claim['field'] == field:
                latest[(row['evidence']['receipt']['source'], claim['revisionKey'])] = (claim, row)
    candidates = [(claim, row) for claim, row in latest.values() if claim['validFrom'] <= day < claim['validTo']]
    if not candidates:
        return {'status': 'unavailable', 'reason': 'historical_reference_interval_missing'}
    if len(candidates) != 1:
        return {'status': 'unavailable', 'reason': 'historical_reference_overlapping_claims'}
    claim, row = candidates[0]
    return {'status': 'unavailable' if claim['operation'] == 'withdraw' else 'received_claim',
            'reason': 'historical_reference_withdrawn' if claim['operation'] == 'withdraw' else 'vendor_semantics_not_verified',
            'claim': claim, 'receiptId': row['receipt_id'], 'receiptSha256': row['source_sha256'],
            'receivedAt': row['received_at'], 'recordedAt': row['recorded_at']}
