"""Evidence sufficiency: recompute each truth issue's amount from ONLY the records in one of its
evidence sets (+ policy rules). If this succeeds for every set, the evidence is sufficient and available."""
from __future__ import annotations

from .invariants import Books
from .money import convert
from .timeutil import local_date, parse


class EvidenceError(AssertionError):
    pass


def fetch(pub: dict, rid: str) -> tuple[str, dict]:
    if rid.startswith("L-"):
        return "ledger", next(r for r in pub["ledger"] if r["row_id"] == rid)
    if rid.startswith("FX-"):
        _, y, m, d, src, tgt = rid.split("-")
        return "fx", next(r for r in pub["fx"] if r["date"] == f"{y}-{m}-{d}" and r["source_currency"] == src and r["target_currency"] == tgt)
    for t in ("invoices", "payments"):
        for r in pub[t]:
            if r["doc_id"] == rid:
                return t, r
    raise EvidenceError(f"unknown id {rid}")


def recompute(pub: dict, issue_type: str, affected: list[str], evidence: list[str]) -> int:
    """Amount implied by the evidence records alone. Raises EvidenceError if they don't prove it."""
    recs = [fetch(pub, e) for e in evidence]
    docs = {r["doc_id"]: (t, r) for t, r in recs if t in ("invoices", "payments")}
    rows = [r for t, r in recs if t == "ledger"]
    fxs = [r for t, r in recs if t == "fx"]
    b = Books(pub)
    if not set(affected) <= set(docs):
        raise EvidenceError("affected documents not all in evidence")
    if len(rows) != 1:
        raise EvidenceError("expected exactly one ledger row")
    row = rows[0]
    amt = row["debit_minor"] or row["credit_minor"]

    def rate_from_evidence(ccy: str, ts: str) -> str | None:
        if ccy == b.base:
            return None
        d = local_date(ts, b.tz).isoformat()
        ok = [f for f in fxs if f["source_currency"] == ccy and f["date"] <= d]
        if not ok:
            raise EvidenceError("missing fx row")
        return max(ok, key=lambda f: f["date"])["rate"]

    if issue_type == "duplicate_invoice":
        (ta, a), (tb_, bb) = (docs[i] for i in affected)
        if not (ta == tb_ == "invoices" and a["delivery_ref"] == bb["delivery_ref"] and a["entity_id"] == bb["entity_id"]):
            raise EvidenceError("not the same delivery")
        dup = max((a, bb), key=lambda d: parse(d["issued_at"]))
        if row["invoice_id"] != dup["invoice_id"]:
            raise EvidenceError("row does not belong to the duplicate")
        return amt
    if issue_type == "fx_conversion":
        t, d = docs[affected[0]]
        ts = d["issued_at"] if t == "invoices" else d["timestamp"]
        total = d["total_minor"] if t == "invoices" else d["amount_minor"]
        correct = convert(total, d["currency"], b.base, rate_from_evidence(d["currency"], ts))
        return abs(amt - correct)
    if issue_type == "period_cutoff":
        t, d = docs[affected[0]]
        event = (d["delivered_at"] if d.get("doc_type") == "invoice" else d["issued_at"]) if t == "invoices" else d["timestamp"]
        if b.bucket(event) == b.bucket(row["timestamp"]):
            raise EvidenceError("no cutoff break visible")
        return amt
    if issue_type == "payment_allocation":
        pay = next(r for t, r in docs.values() if t == "payments")
        inv = next(r for t, r in docs.values() if t == "invoices")
        if inv["invoice_id"] not in (pay["invoice_refs"] or []):
            raise EvidenceError("payment does not reference invoice")
        alloc = min(pay["amount_minor"] + pay["fee_minor"], inv["total_minor"])
        if alloc == inv["total_minor"]:
            due = convert(inv["total_minor"], inv["currency"], b.base, rate_from_evidence(inv["currency"], inv["issued_at"]))
        else:
            due = convert(alloc, inv["currency"], b.base, rate_from_evidence(inv["currency"], inv["issued_at"]))
        if row["account"] == "1100":
            return due - row["credit_minor"]
        if row["account"] == "2150" and inv["currency"] == b.base:
            return row["credit_minor"]
        raise EvidenceError("row is not an allocation row")
    raise EvidenceError(f"unknown type {issue_type}")
