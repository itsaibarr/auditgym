"""Config 6 (PROTOCOL.md addendum 5): deterministic post-filter on the model's findings, fixed before any result on set D.

The filter only DROPS findings; it never adds, edits or repairs one. Every check reads only the public rows the scaffold
observed (the same rows the workbook shows) and the published rules printed in the prompt. It does not call the env's
invariant checker or ExpertPolicy, and it never sees the truth.

Generic checks (every type):
  G1 every affected_id and evidence_id exists among the observed doc ids, ledger row ids and fx ids
     (round 2/3: c4 cites fx ids for dates without a rate -> grader reason `unknown_evidence_id`);
  G2 every affected_id is also in evidence_ids (the rules say evidence must include the doc(s); round 2/3 c4 FPs cite
     another doc instead -> `no_issue` / `insufficient_evidence`);
  G3 an exact repeat of an earlier finding (same type and affected ids) is dropped.
duplicate_invoice (rule: two invoices bill the same delivery_ref; amount = the duplicate's AR debit):
  D1 exactly two affected docs, both invoices, with the same delivery_ref;
  D2 evidence has the later-issued doc's AR debit row (1100, no payment) and claimed amount = its debit (+-tolerance).
period_cutoff (rule: the source row's company-local date is in a different period than the event):
  C1 exactly one affected doc, and evidence has its source row (invoice: 1100 debit; credit note: 1100 credit;
     payment: bank row 1000-1070);
  C2 the source row's period differs from the event's period (invoice: delivered_at; credit note: issued_at; payment:
     timestamp), periods compared on company-local dates (before / inside / after the reporting period);
  C3 a foreign-currency doc cites the prescribed fx row (last rate on or before the company-local document date);
  C4 claimed amount = the doc amount at the policy rate (+-tolerance).
Other types get only G1-G3.
"""
from __future__ import annotations

import sys
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from auditgym.timeutil import local_date  # noqa: E402

BANKS = {f"10{i:02d}" for i in range(0, 71)}   # 1000-1070 bank accounts, as printed in the rules


def _period(day, start, end):
    return -1 if day < start else (0 if day < end else 1)


def postfilter(payload: dict, obs0: dict, rows: dict) -> tuple[dict, list[str]]:
    """Return (payload with failing findings removed, list of drop reasons). Malformed payloads pass unchanged so that
    the env still grades schema errors exactly as before."""
    fs = payload.get("findings") if isinstance(payload, dict) else None
    if not isinstance(fs, list) or not all(isinstance(f, dict) for f in fs):
        return payload, []
    m = obs0["manifest"]
    tz, base, tol, exps = m["timezone"], m["base_currency"], m["amount_tolerance_minor"], m["currency_exponents"]
    p0, p1 = local_date(m["period_start"], tz), local_date(m["period_end"], tz)
    docs = {d["doc_id"]: d for d in rows["invoices"] + rows["payments"]}
    led = {r["row_id"]: r for r in rows["ledger"]}
    fx = {r["fx_id"]: r for r in rows["fx"]}
    known = set(docs) | set(led) | set(fx)

    def fx_row_for(d):
        day = local_date(d.get("issued_at") or d["timestamp"], tz).isoformat()
        c = [r for r in rows["fx"] if r["source_currency"] == d["currency"] and r["target_currency"] == base and r["date"] <= day]
        return max(c, key=lambda r: r["date"]) if c else None

    def is_src(r, d):
        if "payment_id" in d:
            return r["account"] in BANKS and r["payment_id"] == d["payment_id"]
        amt = r["debit_minor"] if d["doc_type"] == "invoice" else r["credit_minor"]
        return r["account"] == "1100" and r["invoice_id"] == d["invoice_id"] and r["payment_id"] is None and amt > 0

    def check(f):
        aff, ev = f.get("affected_ids"), f.get("evidence_ids")
        if not (isinstance(aff, list) and isinstance(ev, list) and all(isinstance(x, str) for x in aff + ev)):
            return None   # leave malformed findings to the env
        if not set(aff + ev) <= known:
            return "G1_unknown_id"
        if not set(aff) <= set(ev):
            return "G2_affected_not_in_evidence"
        amount = f.get("claimed_amount_minor")
        if f.get("type") == "duplicate_invoice":
            if len(aff) != 2 or any(a not in docs or docs[a].get("doc_type") != "invoice" for a in aff):
                return "D1_not_two_invoices"
            a, b = docs[aff[0]], docs[aff[1]]
            if a["delivery_ref"] != b["delivery_ref"]:
                return "D1_different_delivery_ref"
            dup = max((a, b), key=lambda d: d["issued_at"])
            src = [led[e] for e in ev if e in led and is_src(led[e], dup)]
            if not src:
                return "D2_no_duplicate_ar_row"
            if not isinstance(amount, int) or abs(amount - src[0]["debit_minor"]) > tol:
                return "D2_amount_not_ar_debit"
        elif f.get("type") == "period_cutoff":
            if len(aff) != 1 or aff[0] not in docs:
                return "C1_not_one_doc"
            d = docs[aff[0]]
            src = [led[e] for e in ev if e in led and is_src(led[e], d)]
            if not src:
                return "C1_no_source_row"
            event = d["timestamp"] if "payment_id" in d else (d["delivered_at"] if d["doc_type"] == "invoice" else d["issued_at"])
            if _period(local_date(src[0]["timestamp"], tz), p0, p1) == _period(local_date(event, tz), p0, p1):
                return "C2_same_period"
            doc_minor = d["amount_minor"] if "payment_id" in d else d["total_minor"]
            if d["currency"] == base:
                want = doc_minor
            else:
                r = fx_row_for(d)
                if r is None or r["fx_id"] not in ev:
                    return "C3_no_prescribed_fx_row"
                q = Decimal(doc_minor) * Decimal(r["rate"]) * Decimal(10) ** (exps[base] - exps[d["currency"]])
                want = int(q.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
            if not isinstance(amount, int) or abs(amount - want) > tol:
                return "C4_amount"
        return None

    kept, drops, seen = [], [], set()
    for f in fs:
        why = check(f)
        key = (f.get("type"), repr(f.get("affected_ids")))
        if why is None and key in seen:
            why = "G3_repeat"
        if why:
            drops.append(why)
        else:
            kept.append(f)
            seen.add(key)
    return {**payload, "findings": kept}, drops
