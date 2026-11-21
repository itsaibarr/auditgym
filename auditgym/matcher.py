"""Shared matcher + terminal reward (case §05). Published rules:

- Findings are validated strictly (schemas.parse_findings). Any structural violation -> reward exactly -1.
- Eligibility of finding f for truth issue i: same type; set(affected_ids) equal to the issue's set (exact strings, no
  normalisation; order and repeats inside the list do not matter); |claimed - expected| <= tolerance_minor (0 when an
  issue does not state one); same currency (exact string); every evidence id exists in the episode; set(evidence_ids)
  includes at least one allowed evidence set (extra existing evidence is fine).
- Assignment is one-to-one: a maximum-cardinality bipartite matching between findings (each copy is its own vertex)
  and issues. Among maximum matchings the deterministic choice is the lexicographically smallest sequence of
  (issue_id, finding index) pairs, issues ordered by issue_id (Unicode code points). Unmatched findings are FP (with a
  reason; an eligible but unmatched copy is "duplicate"), unmatched issues are FN. The order of findings can change
  which identical copy is matched, never TP/FP/FN or the reward.
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
    return float(max(Fraction(-1), min(Fraction(1), x)))  # exact rational, one float conversion (no rounding)


def _f1(tp, fp, fn, n_truth: int, n_find: int) -> Fraction:
    if n_truth == 0:
        return Fraction(1) if n_find == 0 else Fraction(0)
    den = 2 * tp + fp + fn
    return Fraction(2 * tp, 1) / den if den else Fraction(0)


def _max_matching(issue_ids: list[str], eligible: dict[str, list[int]]) -> dict[int, str]:
    """Maximum bipartite matching (augmenting paths), then the lexicographically smallest (issue_id, finding) pairs:
    issues in order take the lowest finding index that still allows a maximum matching of the rest."""
    def size(fixed: dict[str, int]) -> int:
        owner: dict[int, str] = {n: i for i, n in fixed.items()}
        free = [i for i in issue_ids if i not in fixed]

        def augment(i: str, seen: set) -> bool:
            for n in eligible[i]:
                if n in seen or n in fixed.values() and owner.get(n) in fixed:
                    continue
                seen.add(n)
                if n not in owner or augment(owner[n], seen):
                    owner[n] = i
                    return True
            return False
        return len(fixed) + sum(augment(i, set()) for i in free)
    best = size({})
    chosen: dict[str, int] = {}
    for i in issue_ids:
        for n in sorted(eligible[i]):
            if n not in chosen.values() and size({**chosen, i: n}) == best:
                chosen[i] = n
                break
        else:
            continue
    return {n: i for i, n in chosen.items()}


def grade(findings: list[dict] | None, truth: dict, inspection_steps: int, known_ids: set[str], malformed: str | None = None) -> dict:
    """Grade a validated findings list. `findings=None` with `malformed` set means invalid structure."""
    issues = sorted(truth["issues"], key=lambda i: i["issue_id"])
    if malformed is not None:
        return {"valid": False, "error": malformed, "tp": 0, "fp": 0, "fn": len(issues), "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "reward": -1.0, "reward_experimental": -1.0, "inspection_steps": inspection_steps, "matches": [], "per_type": {}}
    findings = findings or []
    eligible: dict[str, list[int]] = {i["issue_id"]: [] for i in issues}
    reasons: dict[int, str] = {}
    for n, f in enumerate(findings):
        aff, ev = set(f["affected_ids"]), set(f["evidence_ids"])
        cands = [i for i in issues if i["type"] == f["type"] and set(i["affected_ids"]) == aff]
        if not cands:
            reasons[n] = "no_issue"
            continue
        for iss in cands:  # the reason reported is the first failing check against the first candidate issue
            if f["currency"] != iss["currency"]:
                r = "currency"
            elif abs(f["claimed_amount_minor"] - iss["expected_amount_minor"]) > iss.get("tolerance_minor", 0):
                r = "amount"
            elif any(e not in known_ids for e in ev):
                r = "unknown_evidence_id"
            elif not any(set(s) <= ev for s in iss["evidence_sets"]):
                r = "insufficient_evidence"
            else:
                eligible[iss["issue_id"]].append(n)
                r = "eligible"
            if reasons.get(n) != "eligible":
                reasons[n] = r
    matched = _max_matching([i["issue_id"] for i in issues], eligible)
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
    near_issues = {i["issue_id"] for n, f in enumerate(findings)
                   if n not in matched and reasons[n] in ("currency", "amount", "unknown_evidence_id", "insufficient_evidence")
                   for i in issues if i["type"] == f["type"] and set(i["affected_ids"]) == set(f["affected_ids"])}
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
        "precision": float(Fraction(tp, tp + fp)) if tp + fp else 0.0,
        "recall": float(Fraction(tp, tp + fn)) if tp + fn else 0.0,
        "f1": float(f1), "reward": reward,
        "reward_experimental": _clip(2 * xf1 - 1 - STEP_COST * inspection_steps),
        "inspection_steps": inspection_steps, "matches": trace, "per_type": per_type,
    }
