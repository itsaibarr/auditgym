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
import subprocess
import sys
import tempfile
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


def _pytest(rep: Report, quick: bool) -> None:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        xml = Path(td) / "junit.xml"
        env = {**{k: v for k, v in os.environ.items() if k != "AUDITGYM_SALT"}, "AUDITGYM_FRESH_SEEDS": "15" if quick else "40", "PYTHONHASHSEED": "0"}
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests"), f"--junitxml={xml}"],
                              cwd=ROOT, env=env, capture_output=True, text=True)
        if not xml.exists():
            rep.add("tests.collect", False, proc.stdout[-500:] + proc.stderr[-500:], time.time() - t0)
            return
        groups: dict[str, list] = defaultdict(lambda: [0, 0, []])
        for case in ET.parse(xml).getroot().iter("testcase"):
            name = case.get("name").split("[")[0]
            key = f"{case.get('classname').split('.')[-1]}::{name}"
            failed = case.find("failure") is not None or case.find("error") is not None
            g = groups[key]
            g[0] += 1
            if failed:
                g[1] += 1
                g[2].append(case.get("name"))
        for key, (n, nf, names) in sorted(groups.items()):
            rep.add(f"tests.{key}", nf == 0, f"{n - nf}/{n} cases" + (f"; failed: {names[:3]}" if nf else ""))
    rep.add("tests.exit_code", proc.returncode == 0, f"pytest exit {proc.returncode}", time.time() - t0)


def run_check(quick: bool = False, as_json: bool = False, out: Path | None = None) -> int:
    out = out or ROOT / "runs" / "check"
    rep = Report(as_json)
    t_all = time.time()
    try:
        _pytest(rep, quick)
        t0 = time.time()
        episodes = split_episodes("train") + split_episodes("dev")
        rep.add("data.open_splits", len(episodes) == 20, f"{len(episodes)} episodes (10 train + 10 dev)", time.time() - t0)

        summaries = {}
        t0 = time.time()
        all_res = run_policies(POLICIES, episodes, out, mode="eval", meta={"sources": "train,dev"})
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

        t0 = time.time()
        leaks = [p for spec in POLICIES for p in scan_run(out / spec)]
        rep.add("isolation.leakscan", not leaks, f"{len(leaks)} problems" + (f": {leaks[:3]}" if leaks else ""), time.time() - t0)
        from .runner import verify_run
        bad_runs = [p for spec in POLICIES for p in verify_run(out / spec)]
        rep.add("isolation.run_integrity", not bad_runs, "every run dir matches the hashes in its run.json" + (f": {bad_runs[:2]}" if bad_runs else ""))

        t0 = time.time()
        bad = []
        for k, (_, ep) in enumerate(episodes):
            pub2, truth2, m = perturb(ep.public, ep.truth, seed=1000 + k)
            ep2 = Episode(pub2, truth2, ep.provenance, {})
            a, b = run_episode("oracle", ep2, "eval").result, run_episode("expert", ep, "eval", isolate=False).result
            b2 = run_episode("expert", ep2, "eval", isolate=False).result
            if a["reward"] != 1.0 or (b["tp"], b["fp"], b["fn"], b["reward"]) != (b2["tp"], b2["fp"], b2["fn"], b2["reward"]):
                bad.append(ep.episode_id)
        rep.add("repro.paired_permutation_rename", not bad, f"{20 - len(bad)}/20 paired checks preserve oracle & expert outcomes", time.time() - t0)

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
            rows.append({"policy": spec, **s})
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
