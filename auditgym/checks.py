"""`auditgym check`: the single offline verification command.

1. pytest suite (contract, data/truth, reward, reproducibility, isolation, units) — one PASS/FAIL line per test group
2. the four baseline policies (+ expert) on train+dev in eval mode, isolated child processes
3. asserted orderings (oracle = 1 everywhere; oracle > expert > reconcile > flag_all/random on books with errors;
   flag_all = -1 on clean books; silence-like behaviour on clean beats accusations)
4. leakscan over every baseline trajectory
5. 20 paired checks (permutation + ID rename) and replay determinism
Writes runs/check/{summary.json,summary.md} and per-policy run dirs. Exit 0 = all pass, 1 = failure, 2 = error.
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from .canon import dumps
from .episode import Episode
from .leakscan import scan_run
from .perturb import perturb
from .runner import run_episode, run_policies, summarize
from .splits import ROOT, split_episodes

POLICIES = ["oracle", "expert", "reconcile", "flag_all", "random"]
STAGE_TESTS = "Running the test suite"
TEST_TOPIC = {  # test file -> what it checks, in the juror's words
    "test_contract": "the agent contract: actions, pages, errors", "test_data": "the data and the answer key",
    "test_reward": "that the reward cannot be gamed", "test_repro": "reproducibility", "test_isolation": "agent isolation (slow: real sandboxed processes)",
    "test_hardening": "bad inputs and the launchers (slow: each test starts a fresh process)", "test_preflight_leak": "salt-leak guards",
    "test_units": "money and currency arithmetic", "test_gymnasium": "the Gymnasium wrapper",
}
_COLLECTED = re.compile(r"collected (\d+) items?")
_RESULT = re.compile(r"^tests/(test_\w+)\.py::(\w+).* (PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\b")
# Slow files (they start subprocesses) first: progress that speeds up feels shorter than progress that stalls at the end.
SLOW_FIRST = ["test_hardening", "test_isolation", "test_preflight_leak", "test_gymnasium", "test_repro"]


def _test_files() -> list[str]:
    files = sorted((ROOT / "tests").glob("test_*.py"), key=lambda f: (f.stem not in SLOW_FIRST,
                   SLOW_FIRST.index(f.stem) if f.stem in SLOW_FIRST else 0, f.name))
    return [str(f) for f in files]


class Report:
    def __init__(self, as_json: bool):
        self.items: list[dict] = []
        self.as_json = as_json

    def add(self, cid: str, ok: bool, detail: str = "", seconds: float = 0.0) -> None:
        self.items.append({"id": cid, "ok": bool(ok), "seconds": round(seconds, 2), "detail": detail})
        if not self.as_json:
            print(f"{'PASS' if ok else 'FAIL'} {cid} {seconds:.1f}s {detail}", flush=True)

    @property
    def ok(self) -> bool:
        return all(i["ok"] for i in self.items)


def _pytest(rep: Report, quick: bool, say) -> None:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        xml = Path(td) / "junit.xml"
        drop = {"AUDITGYM_SALT", "AUDITGYM_SHELL", "AUDITGYM_LAUNCHER", "AUDITGYM_REPORT_HOST"}
        env = {**{k: v for k, v in os.environ.items() if k not in drop}, "AUDITGYM_FRESH_SEEDS": "15" if quick else "40", "PYTHONHASHSEED": "0"}
        # -vv: pyproject's addopts has -q, so this nets one line per test, which we count as the suite runs
        proc = subprocess.Popen([sys.executable, "-m", "pytest", "-vv", "-p", "no:cacheprovider", *_test_files(), f"--junitxml={xml}"],
                                cwd=ROOT, env={**env, "PYTHONUNBUFFERED": "1", "COLUMNS": "400"}, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        try:  # measured seconds per test: the bar moves with time spent, not with the (very uneven) test count
            per_test = {k: v for k, v in json.loads((ROOT / "tests" / "durations.json").read_text()).items() if k[0] != "_"}
        except (OSError, ValueError):
            per_test = {}
        lines, total, done, failed, spent = [], 0, 0, 0, 0.0
        weight = sum(per_test.values()) or 1.0
        longest: dict[str, float] = {}  # per test file: its slowest test's expected seconds
        for k, v in per_test.items():
            longest[k.split("::")[0]] = max(longest.get(k.split("::")[0], 0.0), v)
        q: queue.Queue = queue.Queue()  # read pytest on a thread, so the line can move while a slow test runs

        def pump() -> None:
            for ln in proc.stdout:
                q.put(ln)
            q.put(None)
        threading.Thread(target=pump, daemon=True).start()
        cur, t_last, shown_max = Path(_test_files()[0]).stem, time.monotonic(), [0.0]
        say(STAGE_TESTS, "collecting the tests (about 5 s)", 0, weight, "0%")

        def show(running: float = 0.0, note: str = "") -> None:
            # finished tests count with their measured seconds; the running one moves the bar at most up to its expected time.
            # The bar is time, so the count goes in the detail: early on, few slow tests are most of the time.
            shown = max(min(spent + running, weight * 0.99), shown_max[0])  # never moves back
            shown_max[0] = shown
            pct = int(100 * shown / weight)
            slow = " (the slow ones run first)" if total and pct > 100 * done / total + 15 else ""
            say(STAGE_TESTS, f"now testing {TEST_TOPIC.get(cur, cur)}{note} · {done}/{total} tests done{slow}"
                + (f" · {failed} failed" if failed else ""), shown, weight, f"{pct}%")
        while True:
            try:
                line = q.get(timeout=1.0)
            except queue.Empty:
                if total:
                    el, nxt = time.monotonic() - t_last, longest.get(cur, 0.5)
                    show(min(el, nxt), " (slower than usual on this machine)" if el > 2 * nxt + 3 else "")
                continue
            if line is None:
                break
            lines.append(line)
            if not total and (m := _COLLECTED.search(line)):
                total = int(m.group(1))
                show()
            elif m := _RESULT.match(line):
                done += 1
                failed += m.group(3) in ("FAILED", "ERROR")
                spent += per_test.get(f"{m.group(1)}::{m.group(2)}", 0.02)
                cur, t_last = m.group(1), time.monotonic()
                show()
        proc.wait()
        if total:
            say(STAGE_TESTS, "", weight, weight, f"100% · {done}/{total} tests")
        if not xml.exists():
            rep.add("tests.collect", False, "".join(lines)[-1000:], time.time() - t0)
            return
        groups: dict[str, list] = defaultdict(lambda: [0, 0, []])
        skipped: dict[str, int] = defaultdict(int)
        for case in ET.parse(xml).getroot().iter("testcase"):
            name = case.get("name").split("[")[0]
            key = f"{case.get('classname').split('.')[-1]}::{name}"
            failed = case.find("failure") is not None or case.find("error") is not None
            g = groups[key]
            g[0] += 1
            if case.find("skipped") is not None:
                skipped[key] += 1
            if failed:
                g[1] += 1
                g[2].append(case.get("name"))
        for key, (n, nf, names) in sorted(groups.items()):
            rep.add(f"tests.{key}", nf == 0, f"{n - nf - skipped[key]}/{n} cases" + (f" ({skipped[key]} skipped on this platform)" if skipped[key] else "")
                    + (f"; failed: {names[:3]}" if nf else ""))
    rep.add("tests.exit_code", proc.returncode == 0, f"pytest exit {proc.returncode}", time.time() - t0)


def run_check(quick: bool = False, as_json: bool = False, out: Path | None = None, progress=None) -> int:
    """progress(stage, detail="", done=None, total=None, count=None) is called as the check moves on (see tour.working)."""
    say = progress or (lambda stage, detail="", done=None, total=None, count=None: None)
    out = out or ROOT / "runs" / "check"
    os.environ.pop("AUDITGYM_SALT", None)  # the official check uses open data and its own salts, never a juror's
    from .runner import harden_runner
    harden_runner()  # non-dumpable: tests that scan /proc for the salt must not find it in this process
    rep = Report(as_json)
    t_all = time.time()
    try:
        say(STAGE_TESTS)
        _pytest(rep, quick, say)
        t0 = time.time()
        episodes = split_episodes("train") + split_episodes("dev")
        rep.add("data.open_splits", len(episodes) == 20, f"{len(episodes)} episodes (10 train + 10 dev)", time.time() - t0)

        summaries = {}
        stage = "Running 5 baseline policies on the 20 open episodes, each in an isolated process"
        say(stage)
        t0 = time.time()
        all_res = run_policies(POLICIES, episodes, out, mode="eval", meta={"sources": "train,dev"},
                               on_episode=lambda spec, n, tot: say(stage, f"now: {spec}", n, tot))
        for spec in POLICIES:
            res = all_res[spec]
            summaries[spec] = summarize(res)["all"]
            summaries[spec]["_results"] = res
            rep.add(f"baseline.{spec}", True, f"return={summaries[spec]['mean_return']:+.3f} F1={summaries[spec]['mean_f1']:.3f} "
                                              f"steps={summaries[spec]['mean_steps']}", time.time() - t0)

        def mean(spec, clean):
            rs = [r["reward"] for r in summaries[spec]["_results"] if r["clean"] == clean]
            return sum(rs) / len(rs)

        o = summaries["oracle"]["_results"]
        rep.add("reward.oracle_max_everywhere", all(r["reward"] == 1.0 and r["f1"] == 1.0 for r in o), "oracle reward == 1.0 on all 20")
        dirty_order = [mean(s, False) for s in ("oracle", "expert", "reconcile")]
        rep.add("reward.ordering_dirty", dirty_order[0] > dirty_order[1] > dirty_order[2] > max(mean("flag_all", False), mean("random", False)),
                "oracle > expert > reconcile > max(flag_all, random): " + ", ".join(f"{s}={mean(s, False):+.3f}" for s in POLICIES))
        rep.add("reward.flag_all_clean_is_minus_one", all(r["reward"] == -1.0 for r in summaries["flag_all"]["_results"] if r["clean"]),
                "flag_all on clean books = -1")
        rep.add("reward.expert_beats_naive_on_clean", mean("expert", True) > mean("reconcile", True),
                f"expert clean={mean('expert', True):+.3f} > reconcile clean={mean('reconcile', True):+.3f} (distractors bite naive rules)")
        rep.add("reward.flag_all_negative", summaries["flag_all"]["mean_return"] < 0 and summaries["flag_all"]["mean_return"] < summaries["reconcile"]["mean_return"],
                f"flag_all mean={summaries['flag_all']['mean_return']:+.3f}")
        rep.add("contract.no_crash_or_schema_error_in_baselines",
                all(r["termination"] in ("submitted", "step_limit") for s in POLICIES for r in summaries[s]["_results"]),
                "every baseline episode ends by submit or the 100-step limit")

        say("Scanning everything the policies saw for leaked answers")
        t0 = time.time()
        leaks = [p for spec in POLICIES for p in scan_run(out / spec)]
        rep.add("isolation.leakscan", not leaks, f"{len(leaks)} problems" + (f": {leaks[:3]}" if leaks else ""), time.time() - t0)
        from .runner import verify_run
        bad_runs = [p for spec in POLICIES for p in verify_run(out / spec)]
        rep.add("isolation.run_integrity", not bad_runs, "every run dir matches the hashes in its run.json" + (f": {bad_runs[:2]}" if bad_runs else ""))

        stage = "Grading shuffled and renamed copies of each episode (paired checks)"
        say(stage)
        t0 = time.time()
        bad = []
        for k, (_, ep) in enumerate(episodes):
            pub2, truth2, m = perturb(ep.public, ep.truth, seed=1000 + k)
            ep2 = Episode(pub2, truth2, ep.provenance, {})
            a, b = run_episode("oracle", ep2, "eval").result, run_episode("expert", ep, "eval", isolate=False).result
            b2 = run_episode("expert", ep2, "eval", isolate=False).result
            if a["reward"] != 1.0 or (b["tp"], b["fp"], b["fn"], b["reward"]) != (b2["tp"], b2["fp"], b2["fn"], b2["reward"]):
                bad.append(ep.episode_id)
            say(stage, "", k + 1, len(episodes))  # after the item: a finished stage reads 20/20, never 19/20
        rep.add("repro.paired_permutation_rename", not bad, f"{20 - len(bad)}/20 paired checks preserve oracle & expert outcomes", time.time() - t0)

        say("Replaying the recorded runs on regenerated episodes")
        t0 = time.time()
        regen = split_episodes("train") + split_episodes("dev")
        same = all(dumps([a.public, a.truth]) == dumps([b.public, b.truth]) for (_, a), (_, b) in zip(episodes, regen))
        rep.add("repro.regenerate_identical", same, "train+dev regenerate byte-identically", time.time() - t0)

        t0 = time.time()
        from .env import AuditEnv
        mism = 0
        recs = [json.loads(l) for l in (out / "reconcile" / "trajectories.jsonl").read_text().splitlines()]
        by_ep = defaultdict(list)
        for r in recs:
            by_ep[r["episode_id"]].append(r)
        eps = {ep.episode_id: ep for _, ep in episodes}
        for eid, steps in by_ep.items():
            env = AuditEnv("eval")
            obs, _ = env.reset(options={"episode": eps[eid]})
            ok = dumps(obs) == dumps(steps[0]["observation"])
            for r in steps[1:]:
                got = env.step(r["action"])
                ok &= dumps(list(got[:4])) == dumps([r["observation"], r["reward"], r["terminated"], r["truncated"]])
            mism += not ok
        rep.add("repro.replay_trajectories", mism == 0, f"{len(by_ep) - mism}/{len(by_ep)} episodes replay byte-identically", time.time() - t0)

        rows = []
        for spec in POLICIES:
            s = {k: v for k, v in summaries[spec].items() if k != "_results"}
            res = summaries[spec]["_results"]
            dirty = [r["reward"] for r in res if not r["clean"]]
            rows.append({"policy": spec, **s, "failed": sum(s.get("termination", {}).get(t, 0) for t in ("timeout", "policy_crash")),
                         "dirty_return": round(sum(dirty) / len(dirty), 4) if dirty else None,
                         "invalid": sum(1 for r in res if not r.get("valid", True))})
        summary = {"ok": rep.ok, "seconds": round(time.time() - t_all, 1), "checks": rep.items, "baselines": rows}
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        md = [f"# auditgym check: {'PASS' if rep.ok else 'FAIL'}", "", "| check | result | detail |", "|---|---|---|"]
        md += [f"| {i['id']} | {'PASS' if i['ok'] else 'FAIL'} | {i['detail']} |" for i in rep.items]
        md += ["", "| policy | P | R | F1 | return | steps | clean-FP | s/ep |", "|---|---|---|---|---|---|---|---|"]
        md += [f"| {r['policy']} | {r['micro_precision']} | {r['micro_recall']} | {r['mean_f1']} | {r['mean_return']:+.3f} | {r['mean_steps']} | {r['clean_fp_rate']} | {r['mean_wall_s']} |" for r in rows]
        (out / "summary.md").write_text("\n".join(md) + "\n")
        if as_json:
            print(json.dumps(summary, indent=2))
        else:
            from .cli import print_table
            print()
            print_table(rows, "baselines on train+dev (eval mode, isolated)")
            n_ok = sum(i["ok"] for i in rep.items)
            print(f"\n{'PASS' if rep.ok else 'FAIL'}: {n_ok}/{len(rep.items)} checks in {time.time() - t_all:.0f}s -> {out}/summary.md")
        return 0 if rep.ok else 1
    except Exception as exc:  # internal error
        import traceback
        traceback.print_exc()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
