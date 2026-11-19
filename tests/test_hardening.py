"""Regression tests for council round-1 findings (attacks on isolation, robustness and data tells)."""
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

import pytest

from auditgym.episode import generate
from auditgym.runner import ChildPolicy, run_episode, run_policies
from auditgym.timeutil import local_date, parse
from conftest import all_episodes
from helpers import oracle_findings, play_actions, submit_after

EVIL = '''
import json, os, sys, time, subprocess
class Fork:
    def reset(self, o, i):
        try:
            pid = os.fork()
            if pid == 0:
                os._exit(0)
            forked = True
        except OSError:
            forked = False
        try:
            subprocess.run(["ps", "-E"], capture_output=True, timeout=5); ps = True
        except Exception:
            ps = False
        try:
            env = open(f"/proc/{os.getppid()}/environ", "rb").read(); parent_env = b"AUDITGYM_SALT" in env
        except Exception:
            parent_env = False
        return {"name": "calculator", "arguments": {"expression": "1"}, "forked": forked, "ps": ps, "parent_env": parent_env}
class DeepNest:
    def reset(self, o, i):
        sys.stdout.write("[" * 100000 + "]" * 100000 + "\\n"); sys.stdout.flush(); return {"name": "calculator", "arguments": {"expression": "1"}}
    def step(self, *a):
        return {"name": "submit", "arguments": {"findings": {"episode_id": "x", "findings": []}}}
class Surrogate:
    def reset(self, o, i):
        sys.stdout.write('{"name": "get_document", "arguments": {"doc_id": "\\\\ud800"}}\\n'); sys.stdout.flush()
        return {"name": "calculator", "arguments": {"expression": "1"}}
    def step(self, *a):
        return {"name": "submit", "arguments": {"findings": {"episode_id": "x", "findings": []}}}
class DupKeys:
    def reset(self, o, i):
        sys.stdout.write('{"name": "calculator", "name": "submit", "arguments": {}}\\n'); sys.stdout.flush()
        return {"name": "calculator", "arguments": {"expression": "1"}}
    def step(self, *a):
        return {"name": "submit", "arguments": {"findings": {"episode_id": "x", "findings": []}}}
class Flood:
    def reset(self, o, i):
        for _ in range(200):
            sys.stdout.write(json.dumps({"name": "list_records", "arguments": {"table": "ledger"}}) + "\\n")
        sys.stdout.flush()
        time.sleep(200)
class Learner:
    def reset(self, o, i):
        self.ep = o["manifest"]["episode_id"]
        return {"name": "submit", "arguments": {"findings": {"episode_id": self.ep, "findings": []}}}
    def end(self, o, r, term, trunc, info):
        open(os.environ["HOME"] + "/../never", "w")
class Reporter:
    """Reports the terminal transition through a calculator action on the next episode (cannot write files)."""
    def reset(self, o, i):
        self.ep = o["manifest"]["episode_id"]
        return {"name": "submit", "arguments": {"findings": {"episode_id": self.ep, "findings": []}}}
class Forger:
    """Round-2 PoC: plants symlinks where the runner will write (no bytes written, no fork)."""
    def __init__(self, root=""):
        self.root = root
    def reset(self, o, i):
        for name, target in (("evil:Forger", "oracle"), ("dangling", "nowhere")):
            try:
                os.symlink(os.path.join(self.root, target), os.path.join(self.root, name))
            except OSError:
                pass
        return {"name": "submit", "arguments": {"findings": {"episode_id": o["manifest"]["episode_id"], "findings": []}}}
class ReadRuns:
    def __init__(self, root=""):
        self.root = root
    def reset(self, o, i):
        seen = os.path.exists(os.path.join(self.root, "oracle", "trajectories.jsonl"))
        return {"name": "calculator", "arguments": {"expression": "1" if not seen else "2"}}
    def step(self, o, *a):
        return {"name": "submit", "arguments": {"findings": {"episode_id": "x", "findings": []}}}
'''


@pytest.fixture(scope="module")
def evil(tmp_path_factory):
    d = tmp_path_factory.mktemp("evil")
    (d / "evil.py").write_text(EVIL)
    return str(d)


def _ask_reset(evil, cls, env=None):
    child = ChildPolicy(f"evil:{cls}", policy_path=evil)
    try:
        return child.ask({"type": "reset", "observation": {}, "info": {}}, time.monotonic() + 20)
    finally:
        child.close()


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="RLIMIT_NPROC is not enforced for root; run as a normal user")
def test_child_cannot_fork_or_read_parent_environment(evil, monkeypatch):
    monkeypatch.setenv("AUDITGYM_SALT", "closed-secret")
    action, status = _ask_reset(evil, "Fork")
    assert status == "ok"
    assert action["forked"] is False and action["ps"] is False
    if sys.platform.startswith("linux"):
        from auditgym.runner import harden_runner
        harden_runner()
        assert action["parent_env"] is False


@pytest.mark.parametrize("cls", ["DeepNest", "Surrogate", "DupKeys"])
def test_hostile_output_lines_are_invalid_actions_not_crashes(evil, dirty, cls, tmp_path):
    out = run_policies([f"evil:{cls}"], [("dev", dirty)], tmp_path, mode="eval", policy_path=evil, dir_name=lambda s: "r")
    r = out[f"evil:{cls}"][0]
    assert r["termination"] in ("submitted", "schema_error") and r["steps"] == 3
    traj = [json.loads(l) for l in (tmp_path / "r" / "trajectories.jsonl").read_text().splitlines()]
    assert traj[1]["observation"]["error"]["code"] in ("invalid_action", "not_found")


def test_flooding_child_cannot_hang_the_runner(evil, dirty):
    t = time.monotonic()
    r = run_episode("evil:Flood", dirty, "eval", policy_path=evil, episode_timeout=6, step_timeout=6).result
    assert time.monotonic() - t < 20
    assert r["reward"] == -1.0 or r["termination"] in ("timeout", "step_limit", "time_limit")


def test_train_mode_terminal_transition_is_delivered(dirty):
    seen = {}

    class P:
        def reset(self, o, i):
            return {"name": "submit", "arguments": {"findings": {"episode_id": o["manifest"]["episode_id"], "findings": oracle_findings(dirty)}}}

        def end(self, o, r, term, trunc, info):
            seen.update(reward=r, term=term, grade=info.get("grade"))

    from auditgym import runner
    orig = runner.make_policy
    runner.make_policy = lambda *a, **k: runner.InProcess(P())
    try:
        run_episode("x", dirty, "train")
    finally:
        runner.make_policy = orig
    assert seen["reward"] == 1.0 and seen["term"] is True and seen["grade"]["f1"] == 1.0


def test_outputs_written_only_after_all_policies(evil, dirty, tmp_path):
    out = run_policies(["evil:ReadRuns", "oracle"], [("dev", dirty)], tmp_path, mode="eval", policy_path=evil,
                       args=[f"root={json.dumps(str(tmp_path))}"])
    traj = [json.loads(l) for l in (tmp_path / "evil:ReadRuns" / "trajectories.jsonl").read_text().splitlines()]
    assert traj[1]["observation"]["result"]["value"] == "1"  # oracle outputs did not exist while the policy ran


def test_open_splits_ignore_closed_salt(monkeypatch):
    from auditgym.splits import split_episodes
    a = [ep.episode_id for _, ep in split_episodes("dev")]
    monkeypatch.setenv("AUDITGYM_SALT", "closed-secret")
    b = [ep.episode_id for _, ep in split_episodes("dev")]
    c = generate(9001, {}).episode_id
    monkeypatch.delenv("AUDITGYM_SALT")
    assert a == b and generate(9001, {}).episode_id != c


def test_closed_split_file_is_honoured_by_cli(capsys):
    from auditgym.cli import build_parser, _episodes
    a = build_parser().parse_args(["run", "--policy", "oracle", "--config", "configs/closed_example.json", "--public-salt"])
    eps = _episodes(a)
    assert len(eps) == 20 and sum(1 for _, e in eps if not e.truth["issues"]) == 5
    assert sum(1 for _, e in eps if any(i["link_id"] for i in e.truth["issues"])) >= 5


def test_string_and_negative_seeds():
    from auditgym.splits import parse_seeds
    assert parse_seeds("1-3,jury-seed-1,-5") == [1, 2, 3, "jury-seed-1", -5]
    assert generate("jury-seed-1").episode_id != generate("jury-seed-2").episode_id


def test_issue_cap_counts_linked_issues():
    full = {"interacting": {"X1": 1.0, "X2": 1.0}, "errors": {t: {"on": True, "count": [4, 4]} for t in
            ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")}}
    for s in range(15):
        assert len(generate(s, full).truth["issues"]) <= 12


def test_experimental_reward_not_gamed_by_copies(dirty):
    f = oracle_findings(dirty)
    near = [{**f[0], "claimed_amount_minor": f[0]["claimed_amount_minor"] + 50 + k} for k in range(8)]
    one, _ = submit_after(dirty, near[:1])
    many, _ = submit_after(dirty, near)
    assert many["reward_experimental"] <= one["reward_experimental"]
    padded, _ = submit_after(dirty, f + near * 3)
    assert padded["reward_experimental"] < 1.0


@pytest.mark.parametrize("cursor", ["p7", "ledger:7:00000000", "invoices:50:{tag}", "ledger:50:{tag}\n"])
def test_forged_or_foreign_cursors_rejected(dirty, cursor):
    env, out = play_actions(dirty, [{"name": "list_records", "arguments": {"table": "ledger"}}])
    tag = out[0][0]["result"]["next_cursor"].split(":")[2]
    o = env.step({"name": "list_records", "arguments": {"table": "ledger", "cursor": cursor.format(tag=tag)}})[0]
    assert o["error"]["code"] == "invalid_cursor"


def test_currency_with_newline_is_malformed(dirty):
    f = oracle_findings(dirty)[0]
    res, _ = submit_after(dirty, [{**f, "currency": f["currency"] + "\n"}])
    assert res["valid"] is False and res["reward"] == -1.0


def test_gym_reset_without_seed_varies():
    gym = pytest.importorskip("gymnasium")
    import auditgym.contrib.gymnasium_env as g
    e = g.AuditGymEnv(time_limit_s=None)
    e.reset(seed=7)
    ids = {json.loads(e.reset()[0])["manifest"]["episode_id"] for _ in range(4)}
    assert len(ids) == 4


# ---------------------------------------------------------------------------------------- data tells
EPS = all_episodes()


def test_no_ledger_row_before_period_start():
    for ep in EPS:
        m = ep.public["manifest"]
        start = local_date(m["period_start"], m["timezone"])
        assert all(local_date(r["timestamp"], m["timezone"]) >= start for r in ep.public["ledger"]), ep.episode_id


def test_late_evening_receipts_are_not_a_tell():
    late, x1 = 0, 0
    for ep in EPS:
        tz = ep.public["manifest"]["timezone"]
        x1_docs = {i for t in ep.truth["issues"] if t["link_id"] == "X1" for i in t["affected_ids"]}
        for p in ep.public["payments"]:
            h = parse(p["timestamp"]).astimezone(__import__("zoneinfo").ZoneInfo(tz)).hour
            if h >= 20:
                late += 1
                x1 += p["doc_id"] in x1_docs
    assert late >= 20 and x1 / late < 0.3


def test_due_dates_and_issue_lags_have_no_injection_shape():
    for ep in EPS:
        dups = {i for t in ep.truth["issues"] if t["type"] == "duplicate_invoice" for i in t["affected_ids"]}
        for d in ep.public["invoices"]:
            if d["doc_type"] != "invoice":
                continue
            zi = __import__("zoneinfo").ZoneInfo(ep.public["manifest"]["timezone"])
            wall = lambda ts: parse(ts).astimezone(zi).replace(tzinfo=None)
            delta = wall(d["due_at"]) - wall(d["issued_at"])  # company wall clock (DST-safe)
            assert delta.seconds == 0 and delta.days in (14, 30, 45), (ep.episode_id, d["doc_id"])
    lagged_legit = sum(1 for ep in EPS for d in ep.public["invoices"] if d["delivered_at"] and d["doc_type"] == "invoice"
                       and (parse(d["issued_at"]) - parse(d["delivered_at"])).total_seconds() > 5 * 3600
                       and d["doc_id"] not in {i for t in ep.truth["issues"] if t["type"] == "duplicate_invoice" for i in t["affected_ids"]})
    assert lagged_legit >= 20


def test_no_real_brand_names():
    blob = json.dumps([ep.public["invoices"] for ep in EPS])
    for brand in ("Kaspi", "Halyk", "Air Astana", "Altyn", "Tulpar", "Kazakhtelecom"):
        assert brand not in blob


# ---------------------------------------------------------------------------------------- council round 2
def test_planted_symlinks_cannot_forge_or_crash_outputs(evil, dirty, tmp_path):
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere")
    out = run_policies(["evil:Forger", "oracle"], [("dev", dirty)], tmp_path, mode="eval", policy_path=evil,
                       args=[f"root={json.dumps(str(tmp_path))}"])
    d = tmp_path / "evil:Forger"
    assert d.is_dir() and not d.is_symlink()
    assert not any(f.is_symlink() for f in d.iterdir())
    on_disk = json.loads((d / "summary.json").read_text())["all"]["mean_return"]
    assert on_disk == out["evil:Forger"][0]["reward"] < 0
    assert json.loads((tmp_path / "oracle" / "summary.json").read_text())["all"]["mean_return"] == 1.0


def test_closed_eval_without_salt_is_refused(monkeypatch, tmp_path):
    from auditgym.cli import main
    monkeypatch.delenv("AUDITGYM_SALT", raising=False)
    assert main(["run", "--policy", "oracle", "--split", "configs/closed_example.json", "--out", str(tmp_path)]) == 2
    assert main(["run", "--policy", "oracle", "--seeds", "7", "--out", str(tmp_path)]) == 2


def test_duplicate_keys_rejected_in_string_findings_and_gym():
    from auditgym.schemas import parse_findings
    f, err = parse_findings('{"episode_id": "bogus", "episode_id": "E", "findings": []}', "E")
    assert f is None and "duplicate" in err
    gym = pytest.importorskip("gymnasium")
    import auditgym.contrib.gymnasium_env  # noqa: F401  (registers the env)
    env = gym.make("AuditGym-v1")
    env.reset(seed=3)
    obs, r, term, trunc, info = env.step('{"name": "calculator", "name": "submit", "arguments": {"expression": "1"}}')
    assert not term and json.loads(obs)["error"]["code"] == "invalid_action"


def test_leakscan_ignores_the_policys_own_echo():
    from auditgym.env import AuditEnv
    from auditgym.leakscan import scan_records
    env = AuditEnv("eval")
    env.reset(seed=1)
    act = {"name": "list_records", "arguments": {"table": "truth_grader"}}
    obs, r, term, trunc, info = env.step(act)
    assert "truth_grader" in json.dumps(obs)
    assert scan_records([{"episode_id": "e", "step": 1, "action": act, "observation": obs, "reward": r, "info": info}], "eval") == []


def test_x1_cutoff_amount_does_not_double_count_fx():
    from auditgym.invariants import BANKS
    n = 0
    for ep in EPS:
        x1 = {i["type"]: i for i in ep.truth["issues"] if i["link_id"] == "X1"}
        if not x1:
            continue
        pay = x1["period_cutoff"]["affected_ids"][0]
        pid = next(p["payment_id"] for p in ep.public["payments"] if p["doc_id"] == pay)
        row = next(r for r in ep.public["ledger"] if r["account"] in BANKS and r["payment_id"] == pid)
        assert abs(row["debit_minor"] - x1["period_cutoff"]["expected_amount_minor"]) == x1["fx_conversion"]["expected_amount_minor"]
        assert all(any(e.startswith("FX-") for e in s) for s in x1["period_cutoff"]["evidence_sets"])
        n += 1
    assert n >= 3


def test_duplicate_issue_timing_is_not_a_tell():
    flagged = dup_flagged = dups = 0
    for ep in EPS:
        zi = __import__("zoneinfo").ZoneInfo(ep.public["manifest"]["timezone"])
        later = {max((d for d in ep.public["invoices"] if d["doc_id"] in i["affected_ids"]), key=lambda d: parse(d["issued_at"]))["doc_id"]
                 for i in ep.truth["issues"] if i["type"] == "duplicate_invoice"}
        for d in ep.public["invoices"]:
            if d["doc_type"] != "invoice" or not d["delivered_at"]:
                continue
            lag = (parse(d["issued_at"]) - parse(d["delivered_at"])).total_seconds() / 3600
            hour = parse(d["issued_at"]).astimezone(zi).hour
            rule = (lag > 4 and lag != int(lag)) or 4 < lag < 24 or 2 <= hour < 8   # round-2 auditor's classifier
            flagged += rule
            dup_flagged += rule and d["doc_id"] in later
            dups += d["doc_id"] in later
    assert dups >= 10 and dup_flagged / max(flagged, 1) < 0.5


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc environ exposure is Linux-specific")
def test_runner_refuses_when_salt_is_exposed_by_another_process(dirty, tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setenv("AUDITGYM_SALT", "closed-secret")
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])  # stands in for tini / sh -c
    try:
        time.sleep(0.2)
        with pytest.raises(RuntimeError, match="AUDITGYM_SALT"):
            run_policies(["reconcile"], [("dev", dirty)], tmp_path, mode="eval")
    finally:
        holder.kill()
        holder.wait()
