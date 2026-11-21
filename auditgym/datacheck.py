"""Independent per-episode data checks (the same ones tests/test_data.py runs), usable on any episode:
accounting invariants I1-I8 hold on the correct books, the injected books break exactly the declared issues,
and every evidence set reproduces its amount from its own records."""
from __future__ import annotations

from .invariants import check
from .recheck import EvidenceError, recompute
from .timeutil import parse


def kind(ep) -> str:
    """clean | interacting X1/X2 | errors, from the recorded injections (the case asks for 5 clean + 5 interacting)."""
    inj = {e["injector"].split(".")[0] for e in ep.provenance.get("events", [])}
    if not ep.truth["issues"]:
        return "clean"
    x = sorted(i for i in inj if i in ("X1", "X2"))
    return f"interacting {'+'.join(x)}" if x else "other"


def kind_counts(eps) -> str:
    ks = [kind(ep) for ep in eps]
    n_x = sum(k.startswith("interacting") for k in ks)
    return f"{ks.count('clean')} clean, {n_x} interacting, {len(ks) - ks.count('clean') - n_x} other"


def expected_violations(ep) -> set:
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


def verify_episode(ep) -> dict:
    pre = not check(ep.clean_public) if ep.clean_public else None
    got = {(v.invariant, v.subject, v.delta) for v in check(ep.public) if v.invariant != "I3"}
    declared = got == expected_violations(ep)
    evidence = True
    for t in ep.truth["issues"]:
        for es in t["evidence_sets"]:
            try:
                evidence &= recompute(ep.public, t["type"], t["affected_ids"], es) == t["expected_amount_minor"]
            except EvidenceError:
                evidence = False
    return {"invariants_before": pre, "declared_after": declared, "evidence": evidence,
            "ok": pre is not False and declared and evidence, "issues": len(ep.truth["issues"])}
