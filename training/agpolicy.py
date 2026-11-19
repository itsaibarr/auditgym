"""Scan-then-decide LLM policy scaffold for AuditGym (agent side only).

The scaffold is identical for every arm (base and fine-tuned):
  1. reset the env, then read every page of fx, invoices, payments and ledger with list_records(limit=50);
  2. render the *observed* rows into one compact text workbook (formatting only: no joins, no checks);
  3. the LLM answers once with {"findings": [...]};
  4. the scaffold adds episode_id and submits the object as-is. A schema violation is graded by the env (-1).

It imports only the public env contract (AuditEnv) and, for the SFT teacher, the truth-free ExpertPolicy.
Truth is read only through env.result() after the episode is over (runner side).
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta

TABLE_ORDER = ("fx", "invoices", "payments", "ledger")
PAGE = 50

SYSTEM = (
    "You are a meticulous financial auditor. You receive the complete books of one company for one reporting "
    "period as a text workbook. Find the errors and answer with one JSON object and nothing else:\n"
    '{"findings": [{"type": "...", "affected_ids": ["..."], "claimed_amount_minor": 0, "currency": "...", '
    '"evidence_ids": ["..."], "explanation": "..."}]}\n'
    "Use an empty list if the books are clean: every wrong finding costs reward, so report only errors the "
    "workbook proves. Use only ids that appear in the workbook. Keep explanations under 80 characters."
)


def _ts(s):
    """Drop seconds from an ISO-8601 instant, keep the offset exactly as written ('Z' / '+05:00')."""
    if not s:
        return "-"
    return re.sub(r"T(\d\d:\d\d):\d\d", r"T\1", s)


def _rules(m: dict) -> str:
    """Short restatement of the published policy (identical for every episode up to base currency and timezone)."""
    b, tz = m["base_currency"], m["timezone"]
    exps = ", ".join(f"{k} {v}" for k, v in sorted(m["currency_exponents"].items()))
    return "\n".join([
        "RULES",
        f"- Amounts are integer minor units. Ledger amounts are in base {b}; documents in their own currency. Exponents: {exps}.",
        f"- FX rate = {b} per 1 unit of source currency, never inverted. Rate date = company-local date of issued_at "
        "(invoice, credit note) or of timestamp (payment, refund); if that date has no rate, use the last rate on or before it. "
        "base_minor = round_half_even(doc_minor * rate * 10**(exp(base) - exp(doc))), once per document total.",
        "- Settlement: AR is relieved at the invoice's carried rate; differences go to 7100 gain / 7200 loss. A partial "
        "allocation relieves round(alloc * invoice rate); the allocation that clears the invoice relieves the remaining carried amount.",
        f"- Period: compare company-local dates ({tz}) after converting each timestamp. An invoice's AR posting must fall in "
        "the same period as its delivered_at; a payment's bank posting in the same period as its timestamp; a credit note "
        "posts at its issued_at. The opening entry (account 3000, at period start) carries prior-period invoices and is not "
        "an invoice posting.",
        "- Allocation: a receipt allocates amount_minor + fee_minor to invoice_refs in listed order, up to each invoice's open "
        "amount (after credit notes); any excess goes to 2300 refunds payable at the payment rate; a receipt without refs "
        "goes to 2150 unapplied cash. Fees post to 6500.",
        "- Billing: each delivery_ref is billed at most once; when two invoices bill the same delivery_ref, the later-issued "
        "one is the duplicate.",
        "- Accounts: 1000-1070 bank, 1100 AR, 2150 unapplied cash, 2300 refunds payable, 3000 opening balance, 6500 bank "
        "charges, 7100 FX gain, 7200 FX loss.",
        f"FINDINGS (currency is always {b}; amount tolerance 1 minor unit)",
        "- duplicate_invoice: affected_ids = [original doc, duplicate doc]; amount = the duplicate's AR debit; evidence must "
        "include both docs + the duplicate's AR row.",
        "- payment_allocation: affected_ids = [payment doc, invoice doc]; amount = AR under-credited for that invoice at its "
        "carried rate; evidence must include payment doc + invoice doc + the payment's AR row for that invoice (or its 2150 "
        "row for a base-currency payment) + the invoice-date fx row if the invoice is in a foreign currency.",
        "- fx_conversion: affected_ids = [doc]; amount = |recorded - correct| base amount of its converted row (invoice: AR "
        "debit; credit note: AR credit; payment: bank row); evidence must include the doc + the prescribed fx row + that row.",
        "- period_cutoff: affected_ids = [doc]; amount = the correct base amount (doc amount at the policy rate) of the "
        "mis-dated source row (AR for an invoice, bank for a payment); evidence must include the doc + the mis-dated row + "
        "the prescribed fx row if the doc is in a foreign currency.",
        "- Evidence ids are ledger row_ids, doc_ids or fx ids.",
    ])


def render(obs0: dict, rows: dict) -> str:
    """Workbook text from the reset observation and the rows the scaffold observed. Formatting only, plus two
    lossless cuts: fx rows more than 7 days before the earliest document date, and revenue/returns legs (4000/4100,
    always the mirror of the AR leg) and expense-only journal entries are not printed."""
    m = obs0["manifest"]
    docs = rows["invoices"] + rows["payments"]
    first = min((d.get("issued_at") or d["timestamp"])[:10] for d in docs) if docs else "0000-00-00"
    lo = (date.fromisoformat(first) - timedelta(days=7)).isoformat() if docs else first
    out = [
        f"EPISODE {m['episode_id']}; base currency {m['base_currency']}; company timezone {m['timezone']}",
        f"Period: {_ts(m['period_start'])} inclusive to {_ts(m['period_end'])} exclusive; export until {_ts(m['export_end'])}.",
        _rules(m),
        "",
        f"FX RATES (rate = {m['base_currency']} per 1 unit of source; business days only; listed from {lo}; "
        "fx id = FX-<YYYY-MM-DD>-<SRC>-<TGT>)",
    ]
    by_key: dict[tuple, list] = {}
    for r in rows["fx"]:
        if r["date"] >= lo:
            by_key.setdefault((r["source_currency"], r["target_currency"], r["date"][:7]), []).append(r)
    for (src, tgt, ym), rs in sorted(by_key.items()):
        out.append(f"{src}->{tgt} {ym}: " + ", ".join(f"{r['date'][8:]}={r['rate']}" for r in rs))
    out += ["", "INVOICES AND CREDIT NOTES (doc_id invoice_id type entity currency total_minor | issued_at | delivered_at | "
                "delivery_ref | ref_invoice_id)"]
    for d in rows["invoices"]:
        out.append(f"{d['doc_id']} {d['invoice_id']} {d['doc_type']} {d['entity_id']} {d['currency']} {d['total_minor']} | "
                   f"{_ts(d['issued_at'])} | {_ts(d['delivered_at'])} | {d['delivery_ref'] or '-'} | {d['ref_invoice_id'] or '-'}")
    out += ["", "PAYMENTS (doc_id payment_id kind entity currency amount_minor fee_minor | timestamp | invoice_refs)"]
    for p in rows["payments"]:
        refs = ",".join(p["invoice_refs"]) if p["invoice_refs"] else "none"
        out.append(f"{p['doc_id']} {p['payment_id']} {p['kind']} {p['entity_id']} {p['currency']} {p['amount_minor']} "
                   f"{p['fee_minor']} | {_ts(p['timestamp'])} | {refs}")
    out += ["", f"LEDGER (posting timestamp: row_id account Dr|Cr amount_{m['base_currency']}_minor invoice_id payment_id; "
                "revenue/returns mirror legs and expense-only entries omitted)"]
    jes: dict[str, list] = {}
    for r in rows["ledger"]:
        jes.setdefault(r["row_id"].rsplit("-", 1)[0], []).append(r)
    for rs in jes.values():
        if not any(r["invoice_id"] or r["payment_id"] or r["account"] == "3000" for r in rs):
            continue
        parts = []
        for r in rs:
            if r["account"] in ("4000", "4100"):
                continue
            side = f"Dr {r['debit_minor']}" if r["debit_minor"] else f"Cr {r['credit_minor']}"
            parts.append(f"{r['row_id']} {r['account']} {side} {r['invoice_id'] or '-'} {r['payment_id'] or '-'}")
        out.append(f"{_ts(rs[0]['timestamp'])}: " + "; ".join(parts))
    return "\n".join(out)


def messages(workbook: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": workbook}]


def scan(env, obs0):
    """Read every page of every table. Returns (rows by table, steps used). Same actions as ExpertPolicy's scan."""
    rows = {t: [] for t in TABLE_ORDER}
    steps = 0
    for t in TABLE_ORDER:
        cursor = None
        while True:
            obs, _, term, trunc, _ = env.step({"name": "list_records", "arguments": {"table": t, "cursor": cursor, "limit": PAGE}})
            steps += 1
            if term or trunc or obs["error"]:
                raise RuntimeError(f"scan failed on {t}: {obs['error']}")
            res = obs["result"]
            rows[t].extend(res["rows"])
            cursor = res["next_cursor"]
            if cursor is None:
                break
    return rows, steps


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def to_submission(text: str, episode_id: str):
    """LLM text -> the object the scaffold submits. No repair beyond stripping a think block / code fence and taking
    the outermost {...}. Returns (payload, parse_ok). If the text is not a JSON object with a findings key the raw
    text is submitted, so the env grades it (malformed -> reward -1)."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    t = _FENCE.sub("", t).strip()
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = json.loads(t[i:j + 1])
        except ValueError:
            obj = None
        if isinstance(obj, dict) and set(obj) <= {"findings", "episode_id"} and "findings" in obj:
            return {"episode_id": episode_id, "findings": obj["findings"]}, True
    return text, False


def target_text(findings: list[dict]) -> str:
    """SFT completion for the expert's findings (same key order as the output contract)."""
    keys = ("type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation")
    return json.dumps({"findings": [{k: f[k] for k in keys} for f in findings]}, ensure_ascii=False, separators=(", ", ": "))
