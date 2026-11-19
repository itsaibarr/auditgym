"""Independent accounting-invariant checker over the *visible* files only.

It never imports the generator: it re-derives what the books must say from documents, fx.csv and
policy rules, using only the shared `money.convert`. Used to prove (a) the correct books satisfy
I1-I8 before injection and (b) after injection exactly the declared discrepancies appear.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from .money import convert
from .timeutil import local_date, parse

BANKS = {"1000", "1010", "1020", "1030", "1040", "1050", "1060", "1070"}


@dataclass(frozen=True)
class Violation:
    invariant: str
    subject: str  # doc_id, JE number or "GL"
    delta: int

    def __str__(self) -> str:
        return f"{self.invariant}:{self.subject}:{self.delta}"


class Books:
    def __init__(self, pub: dict):
        self.pub = pub
        m = pub["manifest"]
        self.base = m["base_currency"]
        self.tz = m["timezone"]
        self.p_start = local_date(m["period_start"], self.tz)
        self.p_end = local_date(m["period_end"], self.tz)
        self.fx = {(r["date"], r["source_currency"]): r["rate"] for r in pub["fx"]}
        self.fx_dates = sorted({r["date"] for r in pub["fx"]})
        self.invoices = [d for d in pub["invoices"] if d["doc_type"] == "invoice"]
        self.cns = [d for d in pub["invoices"] if d["doc_type"] == "credit_note"]
        self.by_invno = {d["invoice_id"]: d for d in pub["invoices"]}
        self.jes: dict[str, list[dict]] = defaultdict(list)
        for r in pub["ledger"]:
            self.jes[je_of(r["row_id"])].append(r)
        self.opening = {k for k, rows in self.jes.items() if any(r["account"] == "3000" for r in rows)}

    def rate(self, ts: str, ccy: str) -> str | None:
        if ccy == self.base:
            return None
        d = local_date(ts, self.tz).isoformat()
        cands = [x for x in self.fx_dates if x <= d and (x, ccy) in self.fx]
        return self.fx[(cands[-1], ccy)]

    def to_base(self, amount: int, ccy: str, ts: str) -> int:
        return convert(amount, ccy, self.base, self.rate(ts, ccy))

    def bucket(self, ts: str) -> int:
        d = local_date(ts, self.tz)
        return -1 if d < self.p_start else (0 if d < self.p_end else 1)


def je_of(row_id: str) -> str:
    return row_id.rsplit("-", 1)[0]


def check(pub: dict) -> list[Violation]:
    b = Books(pub)
    out: list[Violation] = []
    led = pub["ledger"]

    # I1: each JE balances; I2: trial balance nets to zero
    for k, rows in sorted(b.jes.items()):
        d = sum(r["debit_minor"] for r in rows) - sum(r["credit_minor"] for r in rows)
        if d:
            out.append(Violation("I1", k, d))
    tb = sum(r["debit_minor"] - r["credit_minor"] for r in led)
    if tb:
        out.append(Violation("I2", "TB", tb))

    # I4: invoice total = sum of line items; line amount = qty * unit price
    for d in pub["invoices"]:
        s = sum(li["amount_minor"] for li in d["line_items"])
        bad = any(li["amount_minor"] != li["qty"] * li["unit_price_minor"] for li in d["line_items"])
        if s != d["total_minor"] or bad:
            out.append(Violation("I4", d["doc_id"], d["total_minor"] - s))

    ar_rows = [r for r in led if r["account"] == "1100"]
    # I5 + I6 for invoices and credit notes (source row: AR debit / AR credit without payment)
    for d in pub["invoices"]:
        rows = [r for r in ar_rows if r["invoice_id"] == d["invoice_id"] and r["payment_id"] is None]
        is_inv = d["doc_type"] == "invoice"
        rows = [r for r in rows if (r["debit_minor"] if is_inv else r["credit_minor"]) > 0]
        if len(rows) != 1:
            out.append(Violation("I5", d["doc_id"], -1))
            continue
        r = rows[0]
        amt = r["debit_minor"] if is_inv else r["credit_minor"]
        exp = b.to_base(d["total_minor"], d["currency"], d["issued_at"])
        if amt != exp:
            out.append(Violation("I5", d["doc_id"], abs(amt - exp)))
        if je_of(r["row_id"]) in b.opening:
            continue
        event = d["delivered_at"] if is_inv else d["issued_at"]
        if b.bucket(r["timestamp"]) != b.bucket(event):
            out.append(Violation("I6", d["doc_id"], exp))

    # I5 + I6 for payments (source row: bank debit for receipts, bank credit for refunds)
    for p in pub["payments"]:
        rows = [r for r in led if r["account"] in BANKS and r["payment_id"] == p["payment_id"]]
        if len(rows) != 1:
            out.append(Violation("I5", p["doc_id"], -1))
            continue
        r = rows[0]
        amt = r["debit_minor"] if p["kind"] == "receipt" else r["credit_minor"]
        exp = b.to_base(p["amount_minor"], p["currency"], p["timestamp"])
        if amt != exp:
            out.append(Violation("I5", p["doc_id"], abs(amt - exp)))
        if b.bucket(r["timestamp"]) != b.bucket(p["timestamp"]):
            out.append(Violation("I6", p["doc_id"], exp))

    # I7 (+ I3): simulate the AR subledger from documents, compare per-(payment, invoice) AR credits
    carried, open_doc = {}, {}
    for d in b.invoices:
        carried[d["invoice_id"]] = b.to_base(d["total_minor"], d["currency"], d["issued_at"])
        open_doc[d["invoice_id"]] = d["total_minor"]
    events = [(parse(c["issued_at"]), 0, "cn", c) for c in b.cns] + [(parse(p["timestamp"]), 1, "pay", p) for p in pub["payments"] if p["kind"] == "receipt"]
    events.sort(key=lambda e: (e[0], e[1], e[3]["doc_id"]))
    credited = defaultdict(int)
    for r in ar_rows:
        if r["payment_id"] is not None:
            credited[(r["payment_id"], r["invoice_id"])] += r["credit_minor"] - r["debit_minor"]
    for _, _, kind, doc in events:
        if kind == "cn":
            ref = doc["ref_invoice_id"]
            carried[ref] -= b.to_base(doc["total_minor"], doc["currency"], doc["issued_at"])
            open_doc[ref] -= doc["total_minor"]
            continue
        if not doc["invoice_refs"]:
            continue
        remaining = doc["amount_minor"] + doc["fee_minor"]
        for ino in doc["invoice_refs"]:
            inv = b.by_invno[ino]
            alloc = min(remaining, open_doc[ino])
            if alloc <= 0:
                continue
            if alloc == open_doc[ino]:
                rel = carried[ino]
            else:
                rel = convert(alloc, inv["currency"], b.base, b.rate(inv["issued_at"], inv["currency"]))
            carried[ino] -= rel
            open_doc[ino] -= alloc
            remaining -= alloc
            got = credited.get((doc["payment_id"], ino), 0)
            if got != rel:
                out.append(Violation("I7", "|".join(sorted([doc["doc_id"], inv["doc_id"]])), rel - got))
    gl_ar = sum(r["debit_minor"] - r["credit_minor"] for r in ar_rows)
    sub = sum(carried.values())
    if gl_ar != sub:
        out.append(Violation("I3", "GL1100", gl_ar - sub))

    # I8: each delivery billed at most once; the later-issued invoice is the duplicate
    by_ref = defaultdict(list)
    for d in b.invoices:
        if d["delivery_ref"]:
            by_ref[d["delivery_ref"]].append(d)
    for ref, docs in sorted(by_ref.items()):
        docs.sort(key=lambda d: parse(d["issued_at"]))
        for dup in docs[1:]:
            out.append(Violation("I8", dup["doc_id"], b.to_base(dup["total_minor"], dup["currency"], dup["issued_at"])))
    return out


def expected_violations(truth: dict) -> set[tuple[str, str, int]]:
    """The exact set of strict violations (I5/I6/I7/I8) that the declared issues must produce."""
    out = set()
    for t in truth["issues"]:
        ids = t["affected_ids"]
        if t["type"] == "duplicate_invoice":
            continue  # subject resolved in tests (the later-issued of the pair)
        if t["type"] == "payment_allocation":
            out.add(("I7", "|".join(ids), t["expected_amount_minor"]))
        elif t["type"] == "fx_conversion":
            out.add(("I5", ids[0], t["expected_amount_minor"]))
        elif t["type"] == "period_cutoff":
            out.add(("I6", ids[0], t["expected_amount_minor"]))
    return out
