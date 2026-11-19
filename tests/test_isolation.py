"""Isolation & leakage: truth-swap invariance, leakscan, child-process sandbox."""
import json

from auditgym.canon import dumps
from auditgym.episode import Episode
from auditgym.leakscan import scan_records
from auditgym.runner import ChildPolicy, run_episode
from conftest import all_episodes


def test_truth_swap_invariance(dirty):
    """Eval-mode observations and info are byte-identical whatever the hidden truth is."""
    fake = Episode(dirty.public, {"episode_id": dirty.episode_id, "issues": []}, dirty.provenance, {})
    a = run_episode("expert", dirty, "eval", isolate=False).trajectory
    b = run_episode("expert", fake, "eval", isolate=False).trajectory
    strip = lambda tr: [{k: v for k, v in s.items() if k in ("observation", "info", "reward", "terminated", "truncated")} for s in tr]
    assert dumps(strip(a)) == dumps(strip(b))


def test_leakscan_clean_on_all_baselines():
    for ep in all_episodes()[:12]:
        for spec in ("oracle", "expert", "reconcile", "flag_all", "random"):
            run = run_episode(spec, ep, "eval", isolate=spec in ("random",))
            assert scan_records(run.trajectory, "eval") == []


def test_leakscan_detects_planted_leak(dirty):
    rec = {"episode_id": "x", "step": 1, "observation": {"result": {"note": "I-01 expected_amount"}}, "info": {"steps_remaining": 1, "error_code": None}, "reward": 0.0}
    assert scan_records([rec], "eval")
    rec2 = {**rec, "observation": {"result": None}, "reward": 0.5}
    assert scan_records([rec2], "eval")


def test_child_policy_has_scrubbed_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDITGYM_SALT", "super-secret")
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, os, sys\n"
        "class P:\n"
        "    def reset(self, o, i):\n"
        "        return {'name': 'calculator', 'arguments': {'expression': '1'}, 'env': sorted(os.environ), 'cwd': os.listdir('.')}\n"
    )
    child = ChildPolicy("probe:P", policy_path=str(tmp_path))
    import time
    action, status = child.ask({"type": "reset", "observation": {}, "info": {}}, time.monotonic() + 20)
    child.close()
    assert status == "ok"
    assert "AUDITGYM_SALT" not in action["env"] and action["cwd"] == []


def test_policy_timeout_and_crash_truncate(tmp_path, dirty):
    (tmp_path / "bad.py").write_text(
        "import time\n"
        "class Slow:\n"
        "    def reset(self, o, i): time.sleep(5); return {}\n"
        "class Crash:\n"
        "    def reset(self, o, i): raise SystemExit(3)\n"
        "class Garbage:\n"
        "    def __init__(self): self.n = 0\n"
        "    def reset(self, o, i): print('not json', flush=True); return None\n"
        "    def step(self, *a): return {'name': 'submit', 'arguments': {'findings': {'episode_id': 'x', 'findings': []}}}\n"
    )
    from auditgym import runner
    r = runner.run_episode("bad:Slow", dirty, "eval", policy_path=str(tmp_path), step_timeout=1.0).result
    assert r["termination"] == "timeout" and r["tp"] == 0 and r["reward"] == -1.0
    r = runner.run_episode("bad:Crash", dirty, "eval", policy_path=str(tmp_path)).result
    assert r["termination"] == "policy_crash" and r["reward"] == -1.0


def test_oracle_not_reachable_from_child(tmp_path, dirty):
    """'oracle' is only the runner-internal privileged policy; children cannot load truth by name."""
    from auditgym.policies import BUILTIN
    assert "oracle" not in BUILTIN
