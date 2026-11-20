"""Episode family for the training experiment: small books so one workbook fits a T4 context.

Seeds are disjoint by construction: SFT data uses TRAIN_SEEDS, evaluation uses HELDOUT_SEEDS; neither range
overlaps configs/train.json (101-110), configs/dev.json (201-210) or configs/closed_example.json (9000+).
"""
from __future__ import annotations

OFF = {"on": False}
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CNY", "KWD", "KZT"]
BASE = {"customers": [2, 3], "opening_invoices": [1, 1], "invoices": [2, 5], "on_account": [0, 1], "expenses": [0, 2]}
ERR = ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")


def _errors(on: dict) -> dict:
    return {t: ({"on": True, "count": on[t]} if t in on else OFF) for t in ERR}


PROFILES = {
    "clean": {"errors": _errors({})},
    "duplicate_invoice": {"errors": _errors({"duplicate_invoice": [1, 2]})},
    "payment_allocation": {"errors": _errors({"payment_allocation": [1, 2]})},
    "fx_conversion": {"errors": _errors({"fx_conversion": [1, 2]})},
    "period_cutoff": {"errors": _errors({"period_cutoff": [1, 2]})},
    "mixed": {"errors": _errors({t: [0, 1] for t in ERR})},
    "X1": {"errors": _errors({t: [0, 1] for t in ERR}), "interacting": {"X1": 1.0}},
    "X2": {"errors": _errors({t: [0, 1] for t in ERR}), "interacting": {"X2": 1.0}},
}
# held-out mix: 12 clean, 6 per single type, 12 mixed, 6 X1, 6 X2 = 60
HELDOUT_MIX = {"clean": 12, "duplicate_invoice": 6, "payment_allocation": 6, "fx_conversion": 6, "period_cutoff": 6,
               "mixed": 12, "X1": 6, "X2": 6}
TRAIN_SEED0, HELDOUT_SEED0 = 100_000, 500_000


def config(seed: int, profile: str) -> dict:
    """Deterministic config for (seed, profile): one base and one foreign currency keep the fx table short."""
    base = ["KZT", "USD", "EUR"][seed % 3]
    foreign = [c for c in CURRENCIES if c != base][(seed // 3) % (len(CURRENCIES) - 1)]
    return {**BASE, **PROFILES[profile], "base_currencies": [base], "foreign_currencies": [foreign]}


def profile_cycle():
    """Profile of the k-th candidate seed in a stream (round-robin over the held-out mix proportions)."""
    order = [p for p, n in HELDOUT_MIX.items() for _ in range(n // 6)]
    i = 0
    while True:
        yield order[i % len(order)]
        i += 1
