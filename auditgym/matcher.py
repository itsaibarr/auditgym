"""Shared matcher + terminal reward (case §05). Published rules:

- Findings are validated strictly (schemas.parse_findings). Any structural violation -> reward exactly -1.
- Eligibility of finding f for truth issue i: same type; sorted(affected_ids) equal (exact strings, no
  normalisation); |claimed - expected| <= tolerance_minor; same currency; every evidence id exists in the
  episode; evidence_ids is a superset of at least one allowed evidence set.
- Assignment is one-to-one and deterministic: truth (type, sorted affected_ids) keys are unique, so each
  finding is eligible for at most one issue. Issues are visited by issue_id; each takes the eligible
  finding with the lowest submission index. Everything else is FP (with a reason); unmatched issues are FN.
- P = TP/(TP+FP), R = TP/(TP+FN) (0 if denominator 0); F1 = 2TP/(2TP+FP+FN); empty truth & empty
  findings -> F1 = 1; empty truth & any finding -> F1 = 0.
- reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1); inspection_steps = all steps consumed before
  submit (including errors and calculator calls). The explanation text is never read.
- reward_experimental (reported separately, never mixed in): like reward but a finding with the right
  type + affected_ids and a wrong amount/currency/evidence earns 0.5 TP instead of FP.
"""
from __future__ import annotations

from fractions import Fraction

STEP_COST = Fraction(2, 1000)


def _clip(x: Fraction) -> float:
    return round(float(max(Fraction(-1), min(Fraction(1), x))), 6)


def _f1(tp, fp, fn, n_truth: int, n_find: int) -> Fraction:
    if n_truth == 0:
        return Fraction(1) if n_find == 0 else Fraction(0)
    den = 2 * tp + fp + fn
    return Fraction(2 * tp, 1) / den if den else Fraction(0)


def grade(findings: list[dict] | None, truth: dict, inspection_steps: int, known_ids: set[str], malformed: str | None = None) -> dict:
    """Grade a validated findings list. `findings=None` with `malformed` set means invalid structure."""
    issues = sorted(truth["issues"], key=lambda i: i["issue_id"])
    if malformed is not None:
        return {"valid": False, "error": malformed, "tp": 0, "fp": 0, "fn": len(issues), "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "reward": -1.0, "reward_experimental": -1.0, "inspection_steps": inspection_steps, "matches": [], "per_type": {}}
    findings = findings or []
    by_key = {(i["type"], tuple(sorted(i["affected_ids"]))): i for i in issues}
    eligible: dict[str, list[int]] = {i["issue_id"]: [] for i in issues}
    reasons: dict[int, str] = {}
    for n, f in enumerate(findings):
        key = (f["type"], tuple(sorted(f["affected_ids"])))
        iss = by_key.get(key)
        if iss is None:
            reasons[n] = "no_issue"
            continue
        if f["currency"] != iss["currency"]:
            reasons[n] = "currency"
        elif abs(f["claimed_amount_minor"] - iss["expected_amount_minor"]) > iss["tolerance_minor"]:
            reasons[n] = "amount"
        elif any(e not in known_ids for e in f["evidence_ids"]):
            reasons[n] = "unknown_evidence_id"
        elif not any(set(s) <= set(f["evidence_ids"]) for s in iss["evidence_sets"]):
            reasons[n] = "insufficient_evidence"
        else:
            eligible[iss["issue_id"]].append(n)
            reasons[n] = "eligible"
    matched: dict[int, str] = {}
    for iss in issues:
        for n in eligible[iss["issue_id"]]:
            if n not in matched:
                matched[n] = iss["issue_id"]
                break
    hit = set(matched.values())
    trace = []
    for n, f in enumerate(findings):
        if n in matched:
            trace.append({"finding": n, "result": "TP", "issue_id": matched[n]})
        else:
            trace.append({"finding": n, "result": "FP", "reason": "duplicate" if reasons[n] == "eligible" else reasons[n]})
    for iss in issues:
        if iss["issue_id"] not in hit:
            trace.append({"issue_id": iss["issue_id"], "result": "FN", "type": iss["type"]})
    tp, fp, fn = len(matched), len(findings) - len(matched), len(issues) - len(hit)
    f1 = _f1(tp, fp, fn, len(issues), len(findings))
    reward = _clip(2 * f1 - 1 - STEP_COST * inspection_steps)
    # experimental: half credit for right type + affected ids
    near_issues = {by_key[(f["type"], tuple(sorted(f["affected_ids"])))]["issue_id"] for n, f in enumerate(findings)
                   if n not in matched and reasons[n] in ("currency", "amount", "unknown_evidence_id", "insufficient_evidence")}
    near = len(near_issues - hit)  # at most one half-credit per unmatched issue; copies earn nothing
    xtp = Fraction(tp) + Fraction(near, 2)
    xf1 = _f1(xtp, Fraction(fp) - Fraction(near, 2), Fraction(fn) - Fraction(near, 2), len(issues), len(findings)) if issues else f1
    xf1 = min(xf1, Fraction(1))
    per_type: dict[str, dict] = {}
    for iss in issues:
        d = per_type.setdefault(iss["type"], {"tp": 0, "fp": 0, "fn": 0})
        d["tp" if iss["issue_id"] in hit else "fn"] += 1
    for n, f in enumerate(findings):
        if n not in matched:
            per_type.setdefault(f["type"], {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
    return {
        "valid": True, "error": None, "tp": tp, "fp": fp, "fn": fn,
        "precision": round(tp / (tp + fp), 6) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 6) if tp + fn else 0.0,
        "f1": round(float(f1), 6), "reward": reward,
        "reward_experimental": _clip(2 * xf1 - 1 - STEP_COST * inspection_steps),
        "inspection_steps": inspection_steps, "matches": trace, "per_type": per_type,
    }
