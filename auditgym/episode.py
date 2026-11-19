"""generate(seed, config) -> Episode(public, truth, provenance). Pure and deterministic.

IDs are assigned *after* injection, in canonical order, so they carry no information about errors.
"""
from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import os
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

from . import SCHEMA_VERSION, __version__
from .canon import assert_no_floats, dumps, sha256
from .inject import mutate, plan
from .invariants import check as check_invariants
from .money import EXPONENTS, ISO4217_LIST_DATE, ROUNDING
from .rng import Rng
from .timeutil import iso, local_instant
from .world import CHART_OF_ACCOUNTS, World, build_world

DEFAULT_SALT = "auditgym-open-splits-v1"  # public salt for open train/dev; closed runs set AUDITGYM_SALT
TOLERANCE_MINOR = 1
TABLES = ("ledger", "invoices", "payments", "fx")
LEDGER_COLUMNS = ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency", "entity_id", "invoice_id", "payment_id"]
FX_COLUMNS = ["date", "source_currency", "target_currency", "rate"]

DEFAULT_CONFIG: dict = {
    "base_currencies": ["KZT", "USD", "EUR"],
    "foreign_currencies": ["USD", "EUR", "KZT", "JPY", "KWD", "CNY", "GBP"],
    "timezones": ["Asia/Almaty", "Asia/Tashkent", "Europe/Berlin", "Asia/Dubai", "UTC"],
    "year": 2026,
    "months": list(range(1, 13)),
    "cutoff_window_days": 5,
    "customers": [6, 16],
    "opening_invoices": [2, 6],
    "invoices": [20, 70],
    "on_account": [1, 3],
    "expenses": [60, 260],
    "errors": {t: {"on": True, "count": [0, 2]} for t in ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")},
    "interacting": {"X1": 0.0, "X2": 0.0},
    "max_errors": 12,
}


def make_config(overrides: dict | None = None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    for k, v in (overrides or {}).items():
        if k not in DEFAULT_CONFIG:
            raise ValueError(f"unknown config key {k!r} (known: {sorted(DEFAULT_CONFIG)})")
        if k == "errors":
            for t, spec in v.items():
                if t not in cfg["errors"] or not isinstance(spec, dict) or set(spec) - {"on", "count"}:
                    raise ValueError(f"bad errors entry {t!r}: types {sorted(cfg['errors'])}, keys on/count")
                cfg["errors"][t] = {**cfg["errors"][t], **spec}
        elif k == "interacting":
            if set(v) - {"X1", "X2"}:
                raise ValueError(f"interacting takes X1/X2 probabilities, got {sorted(v)}")
            cfg["interacting"].update(v)
        else:
            cfg[k] = v
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    def rng(name, v):
        if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in v) and 0 <= v[0] <= v[1]):
            raise ValueError(f"{name} must be [min, max] integers with 0 <= min <= max, got {v!r}")
    for k in ("customers", "opening_invoices", "invoices", "on_account", "expenses"):
        rng(k, cfg[k])
    for t, spec in cfg["errors"].items():
        rng(f"errors.{t}.count", spec.get("count", [1, 1]))
    if cfg["customers"][0] < 2:
        raise ValueError("customers min must be >= 2 (one base-currency and one foreign-currency customer)")
    if not cfg["base_currencies"] or not all(c in EXPONENTS for c in cfg["base_currencies"] + cfg["foreign_currencies"]):
        raise ValueError(f"currencies must be non-empty and among {sorted(EXPONENTS)}")
    for b in cfg["base_currencies"]:
        if not any(c != b for c in cfg["foreign_currencies"]):
            raise ValueError(f"foreign_currencies needs at least one currency other than base {b}")
    if not all(isinstance(p, (int, float)) and not isinstance(p, bool) and 0 <= p <= 1 for p in cfg["interacting"].values()):
        raise ValueError("interacting probabilities must be numbers in [0, 1]")
    w = cfg["cutoff_window_days"]
    if not (isinstance(w, int) and not isinstance(w, bool) and 4 <= w <= 60):
        raise ValueError(f"cutoff_window_days must be an integer in [4, 60] (late-cutoff postings land up to 3 days after "
                         f"period end, inside the export window), got {w!r}")
    m = cfg["months"]
    if not (isinstance(m, list) and m and all(isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= 12 for x in m)):
        raise ValueError(f"months must be a non-empty list of integers 1..12, got {m!r}")
    y = cfg["year"]
    if not (isinstance(y, int) and not isinstance(y, bool) and 1971 <= y <= 2200):
        raise ValueError(f"year must be an integer in [1971, 2200], got {y!r}")
    me = cfg["max_errors"]
    if not (isinstance(me, int) and not isinstance(me, bool) and me >= 0):
        raise ValueError(f"max_errors must be an integer >= 0, got {me!r}")
    tzs = cfg["timezones"]
    if not (isinstance(tzs, list) and tzs):
        raise ValueError("timezones must be a non-empty list")
    from zoneinfo import ZoneInfo
    for tz in tzs:
        try:
            ZoneInfo(tz)
        except Exception:
            raise ValueError(f"unknown timezone {tz!r}") from None


def current_salt() -> str:
    return os.environ.get("AUDITGYM_SALT", DEFAULT_SALT)


@dataclass
class Episode:
    public: dict  # manifest, policy, ledger, invoices, payments, fx
    truth: dict
    provenance: dict
    clean_public: dict  # the correct books (pre-injection), for invariant checks

    @property
    def episode_id(self) -> str:
        return self.public["manifest"]["episode_id"]


def generate(seed: int | str, config: dict | None = None, salt: str | None = None, config_version: str | None = None) -> Episode:
    cfg = make_config(config)
    salt = current_salt() if salt is None else salt
    cfg_hash = sha256(cfg)[:16]
    key = hmac.new(salt.encode(), f"{seed}|{cfg_hash}|{__version__}".encode(), hashlib.sha256).hexdigest()
    episode_id = "ep_" + hmac.new(salt.encode(), f"episode|{key}".encode(), hashlib.sha256).hexdigest()[:16]

    w = build_world(key, cfg)
    ctx, prepared = plan(w, cfg, key)
    correct_jes = [(je, je.ts, [l.copy() for l in je.lines]) for je in w.jes]
    correct_docs = (list(w.invoices), list(w.payments))
    issues = mutate(ctx, prepared)

    ids = _assign_ids(w, key)
    public = _export_public(w, ids, [(je, je.ts, je.lines) for je in w.jes], w.invoices, w.payments, episode_id)
    clean = _export_public(w, ids, correct_jes, correct_docs[0], correct_docs[1], episode_id)

    rows_of = {}
    for je in w.jes:
        for i, line in enumerate(je.lines):
            rows_of[id(line)] = ids.row(je, i)

    def ref(x):
        if x[0] == "doc":
            return ids.doc[id(x[1])]
        if x[0] == "line":
            return rows_of[id(x[2])]
        return fx_id(x[1].isoformat(), x[2], w.base)

    tissues, events = [], []
    for iss in issues:
        tissues.append({
            "type": iss.type,
            "affected_ids": sorted(ids.doc[id(d)] for d in iss.affected),
            "expected_amount_minor": iss.amount,
            "currency": w.base,
            "tolerance_minor": TOLERANCE_MINOR,
            "evidence_sets": sorted(sorted(ref(x) for x in s) for s in iss.evidence),
            "link_id": iss.link,
            "invariant": iss.invariant,
        })
        events.append({
            "injector": iss.injector,
            "params": iss.params,
            "before": [{"je": ids.je_no[id(je)], "timestamp": _ts(w, ts), "rows": [_row(w, ids, je, i, l) for i, l in enumerate(lines)]}
                       for je, lines, ts in iss.before],
            "after": [{"je": ids.je_no[id(je)], "timestamp": _ts(w, je.ts), "rows": [_row(w, ids, je, i, l) for i, l in enumerate(je.lines)]}
                      for je in iss.jes],
        })
    order = sorted(range(len(tissues)), key=lambda i: (tissues[i]["type"], tissues[i]["affected_ids"]))
    truth_issues, prov_events = [], []
    for n, i in enumerate(order, 1):
        iid = f"I-{n:02d}"
        truth_issues.append({"issue_id": iid, **tissues[i]})
        prov_events.append({"issue_id": iid, **events[i]})
    keys = [(t["type"], tuple(t["affected_ids"])) for t in truth_issues]
    assert len(keys) == len(set(keys)), "truth (type, affected_ids) keys must be unique"
    truth = {"episode_id": episode_id, "issues": truth_issues}
    provenance = {
        "episode_id": episode_id, "generator_version": __version__, "schema_version": SCHEMA_VERSION,
        "config": cfg, "config_hash": cfg_hash, "config_version": config_version, "public_fingerprint": sha256(public), "events": prov_events,
        "invariants_before_injection": {"checked": ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8"], "violations": [str(v) for v in check_invariants(clean)]},
        "skipped_injectors": ctx.skipped,
        "inserted_documents": sorted(ids.doc[id(d)] for d in (set(w.invoices) | set(w.payments)) - set(correct_docs[0]) - set(correct_docs[1])),
    }
    assert_no_floats(public)
    assert_no_floats(truth)
    return Episode(public, truth, provenance, clean)


# ---------------------------------------------------------------------------------------- ids
def fx_id(date: str, src: str, tgt: str) -> str:
    return f"FX-{date}-{src}-{tgt}"


class _Ids:
    def __init__(self):
        self.cust: dict[int, str] = {}
        self.doc: dict[int, str] = {}
        self.inv_no: dict[int, str] = {}
        self.pay_no: dict[int, str] = {}
        self.je_no: dict[int, int] = {}
        self.dn: dict[str, str] = {}

    def row(self, je, i: int) -> str:
        return f"L-{self.je_no[id(je)]:05d}-{i + 1}"


def _unique_numbers(r: Rng, n: int, lo: int, hi: int) -> list[int]:
    seen, out = set(), []
    while len(out) < n:
        x = r.randint(lo, hi)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _assign_ids(w: World, key: str) -> _Ids:
    r = Rng(key, "ids")
    ids = _Ids()
    for c, n in zip(w.customers, _unique_numbers(r, len(w.customers), 1000, 9999)):
        ids.cust[id(c)] = f"C-{n}"
    docs = list(w.invoices) + list(w.payments)
    for d, n in zip(docs, _unique_numbers(r, len(docs), 10000, 99999)):
        ids.doc[id(d)] = f"D-{n}"
    formats = r.sample(["INV-{n}", "INV{n}", "INV/{y}/{n}", "SF-{n}"], r.randint(2, 3))
    weights = [r.randint(1, 5) for _ in formats]
    year = w.period_start.year
    invoices = sorted((i for i in w.invoices if i.kind == "invoice"), key=lambda i: (i.issued_at, i.key))
    n = r.randint(1000, 6000)
    numbers: dict[int, int] = {}
    fmt_of: dict[int, str] = {}
    for inv in invoices:
        if inv.dup_of is not None and inv.same_number:
            numbers[id(inv)] = numbers[id(inv.dup_of)]
            fmt_of[id(inv)] = r.choice([f for f in formats if f != fmt_of[id(inv.dup_of)]])
        else:
            n += 1
            numbers[id(inv)] = n
            fmt_of[id(inv)] = r.weighted(formats, weights)
        ids.inv_no[id(inv)] = fmt_of[id(inv)].format(n=numbers[id(inv)], y=year)
    cn_n = r.randint(100, 900)
    for cn in sorted((i for i in w.invoices if i.kind == "credit_note"), key=lambda i: (i.issued_at, i.key)):
        cn_n += 1
        ids.inv_no[id(cn)] = f"CN-{cn_n}"
    p_n = r.randint(1000, 8000)
    for p in sorted(w.payments, key=lambda p: (p.ts, p.key)):
        p_n += 1
        ids.pay_no[id(p)] = f"PAY-{p_n}"
    je_n = r.randint(10000, 60000)
    for je in sorted(w.jes, key=lambda j: (j.ts, j.key)):
        je_n += 1
        ids.je_no[id(je)] = je_n
    refs = sorted({i.delivery_ref for i in w.invoices if i.delivery_ref}, key=lambda s: int(s[2:]))
    for ref, num in zip(refs, _unique_numbers(r, len(refs), 100000, 999999)):
        ids.dn[ref] = f"DN-{num}"
    return ids


# ---------------------------------------------------------------------------------------- export
def _ts(w: World, dt) -> str:
    return iso(dt.astimezone(local_instant(w.period_start, 0, 0, w.tz).tzinfo))


def _doc_ts(w: World, dt, utc: bool) -> str | None:
    if dt is None:
        return None
    return iso(dt.astimezone(timezone.utc)) if utc else _ts(w, dt)


def _row(w: World, ids: _Ids, je, i: int, l) -> dict:
    return {
        "row_id": ids.row(je, i),
        "account": l.account,
        "debit_minor": l.debit,
        "credit_minor": l.credit,
        "currency": w.base,
        "entity_id": ids.cust[id(l.cust)] if l.cust is not None else None,
        "invoice_id": ids.inv_no[id(l.inv)] if l.inv is not None else None,
        "payment_id": ids.pay_no[id(l.pay)] if l.pay is not None else None,
    }


def _export_public(w: World, ids: _Ids, jes, invoices, payments, episode_id: str) -> dict:
    ledger = []
    for je, ts, lines in sorted(jes, key=lambda x: (x[1], ids.je_no[id(x[0])])):
        for i, l in enumerate(lines):
            ledger.append({"timestamp": _ts(w, ts), **_row(w, ids, je, i, l)})
    inv_rows = []
    for inv in sorted(invoices, key=lambda i: (i.issued_at, ids.doc[id(i)])):
        inv_rows.append({
            "doc_id": ids.doc[id(inv)],
            "invoice_id": ids.inv_no[id(inv)],
            "doc_type": inv.kind,
            "entity_id": ids.cust[id(inv.cust)],
            "entity_name": inv.cust.name,
            "issued_at": _doc_ts(w, inv.issued_at, inv.utc_written),
            "due_at": _doc_ts(w, inv.due_at, inv.utc_written),
            "delivered_at": _doc_ts(w, inv.delivered_at, inv.utc_written),
            "currency": inv.ccy,
            "total_minor": inv.total,
            "line_items": [dict(l) for l in inv.lines],
            "delivery_ref": ids.dn[inv.delivery_ref] if inv.delivery_ref else None,
            "service_period": inv.service_period,
            "ref_invoice_id": ids.inv_no[id(inv.ref)] if inv.ref is not None else None,
        })
    pay_rows = []
    for p in sorted(payments, key=lambda p: (p.ts, ids.doc[id(p)])):
        pay_rows.append({
            "doc_id": ids.doc[id(p)],
            "payment_id": ids.pay_no[id(p)],
            "kind": p.kind,
            "entity_id": ids.cust[id(p.cust)],
            "timestamp": _doc_ts(w, p.ts, p.utc_written),
            "currency": p.ccy,
            "amount_minor": p.amount,
            "fee_minor": p.fee,
            "invoice_refs": [ids.inv_no[id(i)] for i in p.refs] if p.refs else None,
        })
    fx_rows = [{"date": d.isoformat(), "source_currency": c, "target_currency": w.base, "rate": rate}
               for (d, c), rate in sorted(w.fx.items(), key=lambda kv: (kv[0][0], kv[0][1]))]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "base_currency": w.base,
        "period_start": iso(local_instant(w.period_start, 0, 0, w.tz)),
        "period_end": iso(local_instant(w.period_end, 0, 0, w.tz)),
        "period_end_exclusive": True,
        "export_end": iso(local_instant(w.export_end, 0, 0, w.tz)),
        "timezone": w.tz,
        "counts": {"ledger": len(ledger), "invoices": len(inv_rows), "payments": len(pay_rows), "fx": len(fx_rows),
                   "documents": len(inv_rows) + len(pay_rows)},
        "rounding": {"mode": ROUNDING, "unit": "minor", "convert_once_per": "document_total"},
        "currency_exponents": {c: EXPONENTS[c] for c in sorted({w.base, *w.foreign})},
        "amount_tolerance_minor": TOLERANCE_MINOR,
    }
    return {"manifest": manifest, "policy": make_policy(w), "ledger": ledger, "invoices": inv_rows, "payments": pay_rows, "fx": fx_rows}


def make_policy(w: World) -> dict:
    return {
        "functional_currency": w.base,
        "ledger_amounts": "All ledger amounts are in the functional (base) currency; document currency lives on invoices/payments.",
        "fx": {
            "rate_direction": "rate = units of base currency per 1 unit of source currency; never invert",
            "rate_date": {"invoice": "company-local date of issued_at", "credit_note": "company-local date of issued_at",
                          "payment": "company-local date of timestamp", "refund": "company-local date of timestamp",
                          "note": "an invoice translates at its issue date even when delivery (revenue recognition) was on an earlier date "
                                  "(IAS 21.22 practical rate: an invoice is always issued in the same period as, and never before, its delivery)"},
            "missing_date_rule": "use the last rate on or before that date (fx.csv has business days only)",
            "conversion": "base_minor = round(doc_minor * rate * 10**(exp(base) - exp(doc))), once per document total",
            "rounding_mode": ROUNDING,
            "settlement": "AR is relieved at the invoice's carried rate; realized differences go to 7100 (gain) / 7200 (loss). "
                          "A partial allocation relieves round(alloc * invoice rate); the allocation that clears the invoice relieves the remaining carried amount.",
        },
        "period": {"timezone": w.tz, "start_inclusive": iso(local_instant(w.period_start, 0, 0, w.tz)),
                   "end_exclusive": iso(local_instant(w.period_end, 0, 0, w.tz)),
                   "export_includes_post_period_window_until": iso(local_instant(w.export_end, 0, 0, w.tz)),
                   "rule": "compare company-local dates after converting each ISO-8601 instant to the company timezone"},
        "recognition": {"revenue": "delivered_at (company-local date) must fall in the same period as the invoice posting",
                        "payment": "the payment posting must fall in the same period as the payment timestamp",
                        "credit_note": "posted at its issued_at"},
        "allocation": {"rule": "a receipt allocates (amount_minor + fee_minor) to invoice_refs in listed order, up to each invoice's open "
                               "document amount (after credit notes); any excess is credited to 2300 refunds payable at the payment rate",
                       "no_refs": "a receipt with invoice_refs = null is an on-account receipt credited to 2150",
                       "fee": "fee_minor stated on the payment document is posted to 6500 bank charges"},
        "billing": {"rule": "each delivery_ref is billed at most once (net of credit notes)",
                    "duplicate": "when two invoices bill the same delivery_ref, the later-issued one (by issued_at instant) is the duplicate",
                    "service_period": "informational only; revenue is recognised at delivered_at"},
        "opening_balance": "the opening entry (account 3000, at period start) carries prior-period open invoices at their issue-date "
                           "conversion; it is not an invoice posting for the recognition rule. No payment, credit note or refund of an opening invoice is dated before the period start.",
        "revaluation": "no period-end revaluation of open foreign-currency items",
        "chart_of_accounts": CHART_OF_ACCOUNTS,
        "amount_tolerance_minor": TOLERANCE_MINOR,
        "findings": {
            "types": ["duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff"],
            "currency": "always the base currency",
            "amounts": {
                "duplicate_invoice": "the duplicate's AR debit in base currency",
                "payment_allocation": "AR under-credited for the referenced invoice, at the invoice's carried rate",
                "fx_conversion": "absolute difference between the recorded and the correct base amount of the converted row",
                "period_cutoff": "the correct base amount of the mis-dated posting (the source row: AR for an invoice, bank for a payment): "
                                 "the document amount converted at the policy rate. It equals the ledger row when only the date is wrong; if the "
                                 "same posting is also mis-converted, that difference belongs to the linked fx_conversion finding, not to this amount",
            },
            "affected_ids": {
                "duplicate_invoice": "doc_ids of both the original and the duplicate invoice",
                "payment_allocation": "doc_ids of the payment and the under-allocated invoice",
                "fx_conversion": "doc_id of the document whose conversion is wrong",
                "period_cutoff": "doc_id of the document posted in the wrong period",
            },
            "evidence_ids": "row_id (ledger), doc_id (invoices/payments) or fx id FX-<date>-<SRC>-<TGT>; policy is context, not evidence",
            "converted_row": "invoice: its AR debit (or revenue credit); credit note: its AR credit (or returns debit); payment: its bank row",
            "minimal_evidence": {
                "duplicate_invoice": "both invoice docs + the duplicate's AR (or revenue) row",
                "payment_allocation": "payment doc + invoice doc + the payment's AR row for that invoice (or its 2150 row for a base-currency payment) "
                                      "+ the invoice-date fx row if the invoice is in a foreign currency",
                "fx_conversion": "the document + the fx row the policy prescribes + the converted ledger row",
                "period_cutoff": "the document + the mis-dated ledger row (AR/revenue for an invoice, bank for a payment) "
                                 "+ the fx row the policy prescribes if the document is in a foreign currency",
            },
        },
    }


# ---------------------------------------------------------------------------------------- files
def write_episode(ep: Episode, out: Path, with_grader: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pub = ep.public
    (out / "manifest.json").write_text(json.dumps(pub["manifest"], indent=2, ensure_ascii=False) + "\n")
    (out / "policy.json").write_text(json.dumps(pub["policy"], indent=2, ensure_ascii=False) + "\n")
    (out / "ledger.csv").write_text(_csv(pub["ledger"], LEDGER_COLUMNS))
    (out / "fx.csv").write_text(_csv(pub["fx"], FX_COLUMNS))
    for t in ("invoices", "payments"):
        (out / f"{t}.jsonl").write_text("".join(dumps(r) + "\n" for r in pub[t]))
    if with_grader:
        g = out / "_grader"
        g.mkdir(exist_ok=True)
        (g / "truth.json").write_text(json.dumps(ep.truth, indent=2, ensure_ascii=False) + "\n")
        (g / "provenance.json").write_text(json.dumps(ep.provenance, indent=2, ensure_ascii=False) + "\n")


def _csv(rows: list[dict], cols: list[str]) -> str:
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(cols)
    for r in rows:
        wr.writerow(["null" if r[c] is None else r[c] for c in cols])
    return buf.getvalue()


def _int_or_null(v: str):
    return None if v == "null" else v


def load_public(d: Path) -> dict:
    pub = {"manifest": json.loads((d / "manifest.json").read_text()), "policy": json.loads((d / "policy.json").read_text())}
    with open(d / "ledger.csv", newline="") as f:
        pub["ledger"] = [{k: (int(v) if k in ("debit_minor", "credit_minor") else _int_or_null(v)) for k, v in row.items()} for row in csv.DictReader(f)]
    with open(d / "fx.csv", newline="") as f:
        pub["fx"] = list(csv.DictReader(f))
    for t in ("invoices", "payments"):
        pub[t] = [json.loads(l) for l in (d / f"{t}.jsonl").read_text().splitlines() if l]
    return pub


def load_grader(d: Path) -> tuple[dict, dict]:
    g = d / "_grader"
    return json.loads((g / "truth.json").read_text()), json.loads((g / "provenance.json").read_text())
