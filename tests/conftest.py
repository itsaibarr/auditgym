import functools
import os

import pytest

from auditgym.episode import generate
from auditgym.splits import split_episodes

N_FRESH = int(os.environ.get("AUDITGYM_FRESH_SEEDS", "40"))
OFF = {t: {"on": False} for t in ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")}


@functools.lru_cache(maxsize=None)
def open_episodes():
    return [ep for _, ep in split_episodes("train") + split_episodes("dev")]


@functools.lru_cache(maxsize=None)
def fresh_episodes():
    """New seeds and parameter combinations (not in train/dev), incl. clean and interacting books."""
    eps = []
    for s in range(N_FRESH):
        if s % 5 == 0:
            cfg = {"errors": OFF}
        elif s % 5 == 1:
            cfg = {"interacting": {"X1": 1.0, "X2": 1.0}}
        else:
            cfg = {"interacting": {"X1": 0.3, "X2": 0.3}}
        eps.append(generate(10_000 + s, cfg))
    return eps


def all_episodes():
    return open_episodes() + fresh_episodes()


@pytest.fixture(scope="session")
def episodes():
    return all_episodes()


@pytest.fixture(scope="session")
def dirty():
    return next(ep for ep in all_episodes() if len(ep.truth["issues"]) >= 3)


@pytest.fixture(scope="session")
def clean():
    return next(ep for ep in all_episodes() if not ep.truth["issues"])
