"""Benign transformations for paired checks: a seeded bijection over opaque IDs + row-order shuffle.

Renamed: ledger row_id (JE number), doc_id, entity_id, payment_id, delivery_ref. Content-derived ids
(fx ids) and business invoice numbers are kept. Truth is remapped mechanically with the same bijection.
"""
from __future__ import annotations

import copy
import re

from .rng import Rng

_ROW = re.compile(r"^L-(\d{5})-(\d+)$")


def _numbers(r: Rng, n: int, lo: int, hi: int, avoid: set) -> list[int]:
    out, seen = [], set(avoid)
    while len(out) < n:
        x = r.randint(lo, hi)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_mapping(pub: dict, seed: int | str) -> dict[str, str]:
    r = Rng(str(seed), "perturb")
    m: dict[str, str] = {}
    jes = sorted({_ROW.match(x["row_id"]).group(1) for x in pub["ledger"]})
    for old, new in zip(jes, _numbers(r, len(jes), 10000, 99999, {int(j) for j in jes})):
        m["JE" + old] = f"{new:05d}"
    for row in pub["ledger"]:
        j, line = _ROW.match(row["row_id"]).groups()
        m[row["row_id"]] = f"L-{m['JE' + j]}-{line}"
    for prefix, values, lo, hi in (
        ("D-", [d["doc_id"] for d in pub["invoices"] + pub["payments"]], 10000, 99999),
        ("C-", sorted({d["entity_id"] for d in pub["invoices"] + pub["payments"]}), 1000, 9999),
        ("PAY-", [p["payment_id"] for p in pub["payments"]], 10000, 99999),
        ("DN-", sorted({d["delivery_ref"] for d in pub["invoices"] if d["delivery_ref"]}), 100000, 999999),
    ):
        olds = {int(v[len(prefix):]) for v in values}
        for v, n in zip(values, _numbers(r, len(values), lo, hi, olds)):
            m[v] = f"{prefix}{n}"
    return {k: v for k, v in m.items() if not k.startswith("JE")}


def _apply(x, m):
    if isinstance(x, str):
        return m.get(x, x)
    if isinstance(x, list):
        return [_apply(v, m) for v in x]
    if isinstance(x, dict):
        return {k: _apply(v, m) for k, v in x.items()}
    return x


def perturb(pub: dict, truth: dict, seed: int | str = 1, rename: bool = True, shuffle: bool = True) -> tuple[dict, dict, dict]:
    m = build_mapping(pub, seed) if rename else {}
    new_pub = {k: (_apply(v, m) if k in ("ledger", "invoices", "payments") else copy.deepcopy(v)) for k, v in pub.items()}
    new_truth = _apply(truth, m)
    for t in new_truth["issues"]:
        t["affected_ids"] = sorted(t["affected_ids"])
        t["evidence_sets"] = sorted(sorted(s) for s in t["evidence_sets"])
    if shuffle:
        r = Rng(str(seed), "shuffle")
        for t in ("ledger", "invoices", "payments", "fx"):
            r.shuffle(new_pub[t])
    return new_pub, new_truth, m


def map_findings(findings: list[dict], m: dict) -> list[dict]:
    return [{**f, "affected_ids": [m.get(i, i) for i in f["affected_ids"]], "evidence_ids": [m.get(i, i) for i in f["evidence_ids"]]} for f in findings]
