"""Reproducibility & generalisation: determinism, replay, paired permutation + ID-rename checks."""
import json
import os
import subprocess
import sys

from auditgym.canon import dumps, sha256
from auditgym.episode import Episode, generate
from auditgym.perturb import map_findings, perturb
from auditgym.runner import run_episode
from conftest import all_episodes
from helpers import oracle_findings, submit_after


def test_same_seed_same_bytes():
    a, b = generate(77, {"interacting": {"X1": 1.0}}), generate(77, {"interacting": {"X1": 1.0}})
    assert dumps(a.public) == dumps(b.public) and dumps(a.truth) == dumps(b.truth)
    assert sha256(generate(78).public) != sha256(a.public)


def test_independent_of_pythonhashseed():
    code = "from auditgym.episode import generate; from auditgym.canon import sha256; e=generate(77); print(sha256([e.public, e.truth]))"
    outs = set()
    for h in ("0", "1", "random"):
        env = {**os.environ, "PYTHONHASHSEED": h}
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.strip())
    assert len(outs) == 1


def test_salt_changes_world():
    a, b = generate(5, salt="a"), generate(5, salt="b")
    assert sha256(a.public) != sha256(b.public)


def test_same_actions_same_transitions_and_rewards():
    ep = generate(91, {"interacting": {"X2": 1.0}})
    a = run_episode("expert", ep, "train", isolate=False).trajectory
    b = run_episode("expert", ep, "train", isolate=False).trajectory
    assert dumps(a) == dumps(b)


def test_paired_permutation_and_rename_preserve_oracle_and_mapped_answers():
    eps = all_episodes()[:20]
    for k, ep in enumerate(eps):
        pub2, truth2, m = perturb(ep.public, ep.truth, seed=k + 1)
        ep2 = Episode(pub2, truth2, ep.provenance, {})
        assert sorted(dumps(r) for r in pub2["ledger"]) != sorted(dumps(r) for r in ep.public["ledger"]) or not m
        o1 = run_episode("oracle", ep, "eval").result
        o2 = run_episode("oracle", ep2, "eval").result
        assert o1["reward"] == o2["reward"] == 1.0
        f = oracle_findings(ep)
        noisy = f + [{**x, "claimed_amount_minor": 0} for x in f[:1]]
        g1, _ = submit_after(ep, noisy, steps=7)
        g2, _ = submit_after(ep2, map_findings(noisy, m), steps=7)
        assert (g1["tp"], g1["fp"], g1["fn"], g1["reward"]) == (g2["tp"], g2["fp"], g2["fn"], g2["reward"])
        e1 = run_episode("expert", ep, "eval", isolate=False).result
        e2 = run_episode("expert", ep2, "eval", isolate=False).result
        assert (e1["tp"], e1["fp"], e1["fn"], e1["reward"]) == (e2["tp"], e2["fp"], e2["fn"], e2["reward"])
