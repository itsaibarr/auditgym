"""Data & ground truth: accounting equalities before injection, exactly the declared discrepancy after,
independent evidence sufficiency, no contradictory data, realistic distractors, format rules."""
import re
from collections import Counter

import pytest

from auditgym.canon import assert_no_floats
from auditgym.invariants import check
from auditgym.money import EXPONENTS
from auditgym.recheck import recompute
from auditgym.timeutil import parse, local_date
from conftest import all_episodes

EPS = all_episodes()
IDS = [ep.episode_id for ep in EPS]


@pytest.mark.parametrize("ep", EPS, ids=IDS)
def test_invariants_hold_before_injection(ep):
    assert [str(v) for v in check(ep.clean_public)] == []


def _expected(ep):
    inv = {d["doc_id"]: d for d in ep.public["invoices"]}
    exp = set()
    for t in ep.truth["issues"]:
        a = t["affected_ids"]
        if t["type"] == "duplicate_invoice":
            dup = max((inv[i] for i in a), key=lambda d: parse(d["issued_at"]))
            exp.add(("I8", dup["doc_id"], t["expected_amount_minor"]))
        elif t["type"] == "payment_allocation":
            exp.add(("I7", "|".join(sorted(a)), t["expected_amount_minor"]))
        elif t["type"] == "fx_conversion":
            exp.add(("I5", a[0], t["expected_amount_minor"]))
        else:
            exp.add(("I6", a[0], t["expected_amount_minor"]))
    return exp


@pytest.mark.parametrize("ep", EPS, ids=IDS)
def test_exactly_declared_discrepancies_after_injection(ep):
    got = {(v.invariant, v.subject, v.delta) for v in check(ep.public) if v.invariant != "I3"}
    assert got == _expected(ep)


@pytest.mark.parametrize("ep", EPS, ids=IDS)
def test_every_evidence_set_independently_reproduces_amount(ep):
    for t in ep.truth["issues"]:
        assert t["evidence_sets"], t
        for es in t["evidence_sets"]:
            assert len(es) <= 10
            assert recompute(ep.public, t["type"], t["affected_ids"], es) == t["expected_amount_minor"], (t["issue_id"], es)


@pytest.mark.parametrize("ep", EPS, ids=IDS)
def test_truth_well_formed(ep):
    keys = [(t["type"], tuple(t["affected_ids"])) for t in ep.truth["issues"]]
    assert len(keys) == len(set(keys))
    assert len(ep.truth["issues"]) <= 12
    base = ep.public["manifest"]["base_currency"]
    known = {r["row_id"] for r in ep.public["ledger"]} | {d["doc_id"] for d in ep.public["invoices"] + ep.public["payments"]} | \
            {f"FX-{r['date']}-{r['source_currency']}-{r['target_currency']}" for r in ep.public["fx"]}
    for t in ep.truth["issues"]:
        assert t["currency"] == base and t["expected_amount_minor"] >= 100 * t["tolerance_minor"]
        assert all(i in known for s in t["evidence_sets"] for i in s) and all(i in known for i in t["affected_ids"])


@pytest.mark.parametrize("ep", EPS, ids=IDS)
def test_formats(ep):
    pub = ep.public
    assert_no_floats(pub)
    m = pub["manifest"]
    for r in pub["ledger"]:
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$", r["timestamp"])
        assert r["currency"] == m["base_currency"] and (r["debit_minor"] == 0) != (r["credit_minor"] == 0)
        assert r["debit_minor"] >= 0 and r["credit_minor"] >= 0
    for d in pub["invoices"] + pub["payments"]:
        for k in ("issued_at", "due_at", "delivered_at", "timestamp"):
            if d.get(k) is not None:
                assert parse(d[k]).utcoffset() is not None
        assert d["currency"] in EXPONENTS
    for r in pub["fx"]:
        assert re.match(r"^\d+(\.\d+)?$", r["rate"])
    keys = [(r["date"], r["source_currency"], r["target_currency"]) for r in pub["fx"]]
    assert len(keys) == len(set(keys)), "contradictory fx rows"
    for p in pub["payments"]:
        assert p["invoice_refs"] is None or p["invoice_refs"]  # explicit null, never []
    assert m["counts"]["ledger"] == len(pub["ledger"])


@pytest.mark.parametrize("ep", EPS, ids=IDS)
def test_no_leak_by_shape_or_order(ep):
    """Injected records share the same key order / format; ids carry no error markers."""
    shapes = Counter(tuple(r.keys()) for r in ep.public["invoices"])
    assert len(shapes) == 1
    blob = str({k: ep.public[k] for k in ("manifest", "ledger", "invoices", "payments", "fx")})
    assert not re.search(r"\b(dup|duplicate|error|inject|wrong|fake|bad)\b", blob, re.I)


def test_clean_episodes_have_distractors():
    clean = [ep for ep in EPS if not ep.truth["issues"]]
    assert len(clean) >= 5
    for ep in clean:
        pays = ep.public["payments"]
        refs = Counter(r for p in pays if p["invoice_refs"] for r in p["invoice_refs"])
        totals = Counter((d["entity_id"], d["total_minor"]) for d in ep.public["invoices"] if d["doc_type"] == "invoice")
        kinds = {
            "split_or_bulk": any(c > 1 for c in refs.values()) or any(p["invoice_refs"] and len(p["invoice_refs"]) > 1 for p in pays),
            "refund": any(p["kind"] == "refund" for p in pays) or any(d["doc_type"] == "credit_note" for d in ep.public["invoices"]),
            "recurring_equal_amounts": any(c > 1 for c in totals.values()),
            "on_account": any(p["invoice_refs"] is None for p in pays),
            "foreign_currency": any(d["currency"] != ep.public["manifest"]["base_currency"] for d in ep.public["invoices"]),
        }
        assert sum(kinds.values()) >= 4, kinds


def test_all_four_types_and_both_interacting_scenarios_occur():
    types = Counter(t["type"] for ep in EPS for t in ep.truth["issues"])
    links = Counter(t["link_id"] for ep in EPS for t in ep.truth["issues"] if t["link_id"])
    assert set(types) == {"duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff"}
    assert links["X1"] >= 3 and links["X2"] >= 2


def test_error_types_independently_configurable():
    from auditgym.episode import generate
    off = {t: {"on": False} for t in ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")}
    for t in off:
        ep = generate(55, {"errors": {**off, t: {"on": True, "count": [2, 2]}}})
        assert {i["type"] for i in ep.truth["issues"]} == {t}


def test_period_boundary_uses_company_timezone():
    """The D10 twin: an instant that is the next day in UTC but inside the period locally (or vice versa)."""
    hits = 0
    for ep in EPS:
        tz = ep.public["manifest"]["timezone"]
        for d in ep.public["invoices"]:
            if d["delivered_at"] and d["delivered_at"].endswith("+00:00") and tz != "UTC":
                if local_date(d["delivered_at"], tz).isoformat() != d["delivered_at"][:10]:
                    hits += 1
    assert hits > 0
