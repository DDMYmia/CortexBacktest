"Independently reparse received SEC cover bytes at the Cortex data boundary."

from datetime import date, datetime
from html.parser import HTMLParser
import hashlib
import json
import re

from .. import strict_json


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _time(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("aware SEC identity clock required")
    return stamp


def _cik(value):
    text = str(value)
    if re.fullmatch(r"[0-9]{1,10}", text) is None:
        raise ValueError("explicit SEC issuer CIK required")
    return text.zfill(10)


class _Cover(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == "ix:nonnumeric":
            attrs = dict(attrs)
            self.stack.append(((attrs.get("name") or "").lower(), attrs.get("contextref"), []))

    def handle_data(self, data):
        for _, _, chunks in self.stack:
            chunks.append(data)

    def handle_endtag(self, tag):
        if tag == "ix:nonnumeric" and self.stack:
            name, context, chunks = self.stack.pop()
            self.rows.append((name, context, " ".join("".join(chunks).split())))


def _one(rows, name):
    values = {value for field, _, value in rows if field == "dei:" + name}
    if len(values) != 1:
        raise ValueError("unique SEC cover fact required: " + name)
    return values.pop()


def _period(value):
    try:
        return date.fromisoformat(value)
    except ValueError:
        parts = re.fullmatch(r"([A-Za-z]+) ([0-9]{1,2}), ([0-9]{4})", value)
        months: tuple[str, ...] = tuple("January February March April May June July August September October November December".split())
        if parts is None or parts[1] not in months:
            raise ValueError("explicit SEC report period day required") from None
        return date(int(parts[3]), months.index(parts[1]) + 1, int(parts[2]))


def verify_sec_identities(document, targets, *, as_of, venues):
    if (
        document["contractVersion"] != "received-sec-identities-v1"
        or _time(document["asOf"]) != as_of
        or any(
            document[k] is not False
            for k in ["historicalIdentityCertified", "qualificationIssued", "executionAuthorized"]
        )
    ):
        raise ValueError("received SEC snapshot contract/clock required")
    records = document["records"]
    ids = [r["instrumentId"] for r in records]
    if len(ids) != 10 or set(ids) != set(targets) or any(type(i) is not int for i in ids):
        raise ValueError("exact ten-security SEC identity coverage required")
    availability, dates = {}, []
    for row in records:
        target = targets[row["instrumentId"]]
        evidence = row["evidence"]
        receipts, raw, receipt_shas = [], [], []
        for name in ["submissions", "filing"]:
            original = evidence[name + "_receipt_utf8"].encode()
            receipt = strict_json.loads(original)
            body = evidence[name + "_raw_utf8"].encode()
            if (
                receipt["contractVersion"] != "research-source-receipt-v1"
                or receipt["rawFile"] != "raw.data"
                or receipt["source"] != "sec"
                or receipt["dataset"] != name
                or receipt["status"] != "received"
                or receipt["httpStatus"] != 200
                or receipt["rawBytes"] != len(body)
                or receipt["rawSha256"] != _sha(body)
            ):
                raise ValueError("original SEC receipt/body digest mismatch")
            receipts.append(receipt)
            raw.append(body)
            receipt_shas.append(_sha(original))
        pair = {"submissionsSha256": receipt_shas[0], "filingSha256": receipt_shas[1]}
        if row["sourceSha256"] != _sha(
            json.dumps(pair, sort_keys=True, separators=(",", ":")).encode()
        ):
            raise ValueError("SEC identity source pair binding mismatch")
        sr, fr = receipts
        received = max(_time(r["receivedAt"]) for r in receipts)
        recorded = _time(row["recordedAt"])
        if _time(row["receivedAt"]) != received or not received <= recorded <= as_of:
            raise ValueError("SEC identity receipt/database cutoff mismatch")
        data = strict_json.loads(raw[0])
        cik = _cik(target["cik"])
        if (
            _cik(data["cik"]) != cik
            or sr["request"]["url"] != f"https://data.sec.gov/submissions/CIK{cik}.json"
        ):
            raise ValueError("SEC submissions issuer/source mismatch")
        acc = fr["scope"]["accessionNumber"]
        recent = data["filings"]["recent"]
        matches = [i for i, value in enumerate(recent["accessionNumber"]) if value == acc]
        if len(matches) != 1 or re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", acc) is None:
            raise ValueError("unique SEC accession required")
        item = {
            k: recent[k][matches[0]]
            for k in ["form", "primaryDocument", "filingDate", "reportDate", "acceptanceDateTime"]
        }
        primary = item["primaryDocument"]
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{primary}"
        if (
            item["form"] not in {"10-K", "10-Q"}
            or re.fullmatch(r"[A-Za-z0-9_.-]+\.htm[l]?", primary) is None
            or fr["request"]["url"] != url
            or _cik(fr["scope"]["cik"]) != cik
            or fr["scope"]["primaryDocument"] != primary
        ):
            raise ValueError("SEC primary document scope mismatch")
        accepted = _time(item["acceptanceDateTime"])
        filed, period = (
            date.fromisoformat(item["filingDate"]),
            date.fromisoformat(item["reportDate"]),
        )
        if (
            accepted > min(_time(r["receivedAt"]) for r in receipts)
            or period > filed
            or filed > received.date()
            or len(raw[1]) > 16 * 1024 * 1024
        ):
            raise ValueError("SEC source/receipt chronology invalid")
        cover = _Cover()
        cover.feed(raw[1].decode())
        if (
            cover.stack
            or _cik(_one(cover.rows, "entitycentralindexkey")) != cik
            or _one(cover.rows, "documenttype") != item["form"]
            or _period(_one(cover.rows, "documentperiodenddate")) != period
        ):
            raise ValueError("SEC cover issuer/type/period mismatch")
        symbol = target["primary_symbol"]
        contexts = {
            ctx
            for name, ctx, value in cover.rows
            if name == "dei:tradingsymbol" and value == symbol and ctx
        }
        classes = set()
        for context in contexts:
            group = [r for r in cover.rows if r[1] == context]
            titles = {value for name, _, value in group if name == "dei:security12btitle"}
            if not any(re.search(r"\bcommon\s+stock\b", title, re.I) for title in titles):
                continue
            title, exchange = _one(group, "security12btitle"), _one(group, "securityexchangename")
            if exchange not in venues:
                raise ValueError("unknown SEC common stock venue")
            classes.add((title, venues[exchange]))
        if len(classes) != 1:
            raise ValueError("one exact common-stock context required")
        title, mic = classes.pop()
        if mic != venues.get(target["exchange_mic"], target["exchange_mic"]):
            raise ValueError("SEC common stock venue conflicts with target")
        expected = {
            "contract_version": "sec-filing-identity-v1",
            "symbol": symbol,
            "cik": cik,
            "issuer_name": _one(cover.rows, "entityregistrantname"),
            "security_title": title,
            "exchange_mic": mic,
            "accession_number": acc,
            "form": item["form"],
            "report_period_end": period.isoformat(),
            "filing_date": filed.isoformat(),
            "source_accepted_at": accepted.isoformat(),
            "source_published_at": None,
            "source_time_policy": "SEC_acceptance_is_not_a_dissemination_timestamp",
            "received_at": received.isoformat(),
            "valid_from": None,
            "valid_to": None,
            "historical_identity_certified": False,
            "qualification_issued": False,
            "execution_authorized": False,
        }
        if row["attributes"] != expected or evidence["historical_identity_certified"] is not False:
            raise ValueError("SEC identity projection does not reproduce original sources")
        availability[row["instrumentId"]] = recorded
        dates.append(filed.isoformat())
    return availability, {
        "status": "received_filing_snapshots_verified",
        "snapshots": len(records),
        "knownAt": max(availability.values()).isoformat(),
        "filingDateRange": [min(dates), max(dates)],
        "sourcePublicationTimeAvailable": False,
        "continuousHistoricalIdentityCertified": False,
        "qualificationIssued": False,
        "executionAuthorized": False,
    }
