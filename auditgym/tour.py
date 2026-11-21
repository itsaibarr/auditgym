"""`auditgym tour`: the jury's whole flow in about a minute, ending with a scorecard per case criterion.

1. the official offline check (`auditgym check`)
2. a closed evaluation shaped like the organizers': fresh secret seeds and salt (never written anywhere),
   20 hidden-like episodes (5 clean, 5 interacting), the four organizer policy types + our truth-free expert
3. 20 paired checks (row permutation + ID rename) on those closed episodes
4. byte-identical replay with re-grading, run-dir integrity and leakscan of the closed runs
Writes <out>/report.md and <out>/report.json and prints copy-paste next steps for plugging in your own agent.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import re
import secrets
import shlex
import sys
import tempfile
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .canon import dumps
from .episode import Episode, generate
from .splits import ROOT, load_split_config

CRITERIA = [  # the case's five scoring criteria and their maximum points
    ("contract", "Contract and isolation", 25),
    ("data", "Data and answer key", 25),
    ("reward", "Reward robustness", 25),
    ("repro", "Reproducibility and generalization", 15),
    ("eng", "Engineering and explanation", 10),
]
POLICIES = ["random", "reconcile", "flag_all", "oracle", "expert"]
POLICY_LABEL = {p: p for p in POLICIES}
ROLE = {  # what each row is for, so the table reads without a footnote
    "random": "floor: random actions", "reconcile": "simple rule-based baseline", "flag_all": "floor: accuses everything",
    "oracle": "install check: reads the answer key, must be +1.000", "expert": "our solver: sees only what an agent sees",
}
FOOTNOTE = ("Reward: the case's score for one episode, from -1 to +1, higher is better · Precision and recall: pooled over all "
            "findings · F1: mean per episode (an empty answer on clean books scores F1 = 1)")


PAST = {"Running": "Ran", "Generating": "Generated", "Checking": "Checked", "Grading": "Graded", "Scanning": "Scanned",
        "Replaying": "Replayed", "Regenerating": "Regenerated"}


def _past(stage: str) -> str:
    verb, _, rest = stage.partition(" ")
    return f"{PAST[verb]} {rest}" if verb in PAST else stage


def _left(frac: float, elapsed: float, total: float, count: str | None) -> str:
    """"~N s left". The test suite's bar is weighted by measured seconds (its count shows a percent): early on it states
    the usual length. Any other count extrapolates only after at least 2 items finished; before that, no estimate."""
    timed = bool(count) and "%" in count
    if timed and (frac <= 0.15 or elapsed < 4):  # too early to extrapolate: machines differ 2-3x
        return "usually 30-90 s in total"
    if not timed and frac * total < 2:
        return ""
    return f"~{elapsed * (1 - frac) / frac:.0f} s left" if frac > 0 else ""


@contextlib.contextmanager
def working(con: Console, first: str | None = None):
    """Show what is running, as a checklist (like Docker BuildKit): each stage gets its own line with a spinner, a moving
    bar, a count when the total is known, the elapsed time, an estimate of the time left and what it is doing right now;
    a finished stage turns into a ✓ line in the past tense that keeps its time. Not a terminal (Docker without -t, CI
    logs): one plain line per stage. Yields say(stage, detail="", done=None, total=None, count=None); a new stage text
    starts a new line; count overrides the "done/total" text (e.g. when the bar is weighted by time)."""
    if not con.is_terminal:  # logs (jury harnesses capture to files): a line per stage, then a status line every 5 s
        import threading
        st: dict = {"stage": None, "t0": 0.0, "last": 0.0, "count": "", "frac": None, "total": 0.0, "weighted": None, "detail": "",
                    "moved": 0.0}
        stop = threading.Event()

        def say(stage: str, detail: str = "", done: float | None = None, total: float | None = None, count: str | None = None) -> None:
            now = time.monotonic()
            if stage != st["stage"]:
                done_line()
                st.update(stage=stage, t0=now, last=now, count="", frac=None, detail="")
                first = count or (f"{int(done or 0)}/{int(total)}" if total else "")
                con.print(f"  ... {stage}{' · ' + first if first else ''}", style="dim", markup=False, soft_wrap=True)
            st["detail"] = detail
            if total:
                if st["frac"] is None or min((done or 0) / total, 1.0) != st["frac"]:
                    st["moved"] = now
                st.update(frac=min((done or 0) / total, 1.0), total=total, weighted=count,
                          count=count or f"{int(done or 0)}/{int(total)}")

        def done_line() -> None:  # close the previous stage with its final count, so no stale "48/80 · ~3 s left" stays last
            if st["stage"] and st["frac"] is not None:
                con.print(f"      done: {st['count']} · {time.monotonic() - st['t0']:.0f} s", style="dim", markup=False, soft_wrap=True)

        def beat() -> None:  # runs even while a long step reports nothing (test collection, one slow episode)
            while not stop.wait(1.0):
                now = time.monotonic()
                if st["stage"] and now - st["last"] >= 3:
                    st["last"] = now
                    el = now - st["t0"]
                    left = (f"current run {now - st['moved']:.0f} s of at most 60 s" if "%" not in (st["weighted"] or "") and now - st["moved"] > 6
                            else _left(st["frac"], el, st["total"], st["weighted"]) if st["frac"] is not None else "")
                    con.print(f"      {st['count'] + ' · ' if st['count'] else 'still working · '}{el:.0f} s{' · ' + left if left else ''}"
                              f"{' · ' + st['detail'] if st['detail'] else ''}", style="dim", markup=False, soft_wrap=True)
        if first:
            say(first)
        th = threading.Thread(target=beat, daemon=True)
        th.start()
        try:
            yield say
        finally:
            stop.set()
            th.join()
            done_line()
        return
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    cols = (SpinnerColumn(finished_text="[green]✓[/]"), TextColumn("{task.description}"), BarColumn(bar_width=20),
            TextColumn("{task.fields[count]}"), TimeElapsedColumn(), TextColumn("[dim]{task.fields[left]}"),
            TextColumn("[dim]{task.fields[detail]}"))
    with Progress(*cols, console=con) as prog:
        cur: dict = {"id": None, "stage": None, "t0": 0.0, "count": "", "done": None, "moved": 0.0}

        def close() -> None:
            if cur["id"] is not None:
                t = next(t for t in prog.tasks if t.id == cur["id"])
                done_count = f"{int(t.total)}/{int(t.total)}" if t.total and re.fullmatch(r"\d+/\d+", cur["count"]) else cur["count"]
                prog.update(cur["id"], description=_past(cur["stage"]), total=t.total or 1, completed=t.total or 1, detail="",
                            left="", count=done_count)

        def say(stage: str, detail: str = "", done: float | None = None, total: float | None = None, count: str | None = None) -> None:
            if stage != cur["stage"]:
                close()
                cur.update(stage=stage, t0=time.monotonic(), count="", done=None, moved=time.monotonic(),
                           id=prog.add_task(stage, total=None, count="", detail="", left=""))
            kw: dict = {"detail": detail}
            if total:
                frac, el = min((done or 0) / total, 1.0), time.monotonic() - cur["t0"]
                if done != cur["done"]:
                    cur.update(done=done, moved=time.monotonic())
                stalled = "%" not in (count or "") and time.monotonic() - cur["moved"] > 6
                kw.update(total=total, completed=min(done or 0, total * 0.999),  # ✓ only on close
                          left=f"current run {time.monotonic() - cur['moved']:.0f} s of at most 60 s" if stalled else _left(frac, el, total, count))
                cur["count"] = count or f"{int(done or 0)}/{int(total)}"
                kw["count"] = cur["count"]
            prog.update(cur["id"], **kw)
        if first:
            say(first)
        try:
            yield say
        finally:
            close()


def _criterion(check_id: str) -> str:
    if check_id.startswith(("tests.test_data", "data.", "tests.test_units::test_convert")):
        return "data"
    if check_id.startswith(("tests.test_reward", "reward.", "baseline.")):
        return "reward"
    if check_id.startswith(("tests.test_repro", "repro.")):
        return "repro"
    if check_id.startswith(("tests.test_contract", "tests.test_isolation", "tests.test_hardening", "tests.test_gymnasium", "tests.test_preflight_leak",
                            "isolation.", "contract.", "tests.test_units", "tests.exit_code", "tests.collect")):
        return "contract"
    return "eng"


def _out_dir(out: str | None) -> Path:
    for cand in ([Path(out)] if out else [Path.cwd() / "auditgym-report", Path(tempfile.gettempdir()) / "auditgym-report"]):
        try:
            cand.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=cand):
                pass
            return cand.resolve()
        except OSError:
            continue
    who = f"uid {os.getuid()}" if hasattr(os, "getuid") else "this user"
    raise OSError(f"no writable output directory for {who} (tried {out or './auditgym-report'}). In Docker on Linux: "
                  "mkdir -p auditgym-report and add --user \"$(id -u):$(id -g)\" -v \"$PWD/auditgym-report:/report\" ... tour --out /report")


def _shown(p: Path) -> str:
    """Path as the user sees it: the Windows launcher mounts the host report folder at /report."""
    host = os.environ.get("AUDITGYM_REPORT_HOST")
    return str(p).replace("/report", host, 1) if host and str(p).startswith("/report") else str(p)


def _clean_txt(n: int, tour_shape: bool) -> str:
    """The tour's split has 5 clean episodes; an 'other' episode can also draw no errors at random."""
    extra = n - 5
    return f"{n} clean books: the 5 clean ones plus {extra} that drew no errors at random" if tour_shape and extra > 0 else f"{n} clean books"


def _rel(p: Path) -> str:
    """A path for a copy-paste command: relative to the current folder when it is inside it, shell-quoted."""
    try:
        return shlex.quote(str(p.relative_to(Path.cwd())))
    except ValueError:
        return shlex.quote(_shown(p))


def _step(con: Console, title: str, question: str, what: str) -> None:
    con.rule(f"[bold]{title}[/]  [dim]({question})[/]")
    con.print(what, style="dim", markup=False, soft_wrap=False)


def _closed_episodes(split: dict | None = None) -> tuple[list[tuple[str, Episode]], int, dict, str]:
    """Without a split: the organizers' shape (configs/closed_example.json: 5 clean, 5 interacting, 10 other) with NEW
    random seeds and a random salt that exists only in this process. With a split (the juror's, from stdin): exactly its
    seeds and configs, salted with AUDITGYM_SALT if set, else a random salt. Returns (episodes, seeds redrawn)."""
    from .splits import split_episodes
    if split is not None:
        salt = os.environ.get("AUDITGYM_SALT") or secrets.token_hex(32)
        old = os.environ.get("AUDITGYM_SALT")
        os.environ["AUDITGYM_SALT"] = salt  # split_episodes salts non-open splits from the environment
        try:
            eps = split_episodes("-", split)
        finally:
            if old is None:
                os.environ.pop("AUDITGYM_SALT", None)
        return sorted(eps, key=lambda x: x[1].episode_id), 0, split, salt
    salt = secrets.token_hex(32)
    split = load_split_config(str(ROOT / "configs" / "closed_example.json"))
    base = split.get("base_config", {})
    eps, redrawn, used = [], 0, []
    for e in split["episodes"]:
        while True:
            seed = f"tour-{secrets.token_hex(8)}"
            try:
                ep = generate(seed, {**base, **e.get("config", {})}, salt=salt, config_version=split.get("config_version"))
                break
            except ValueError:  # a config/seed the fail-closed generator refuses: draw another seed
                redrawn += 1
        eps.append(("tour", ep))
        used.append({**{k: v for k, v in e.items() if k != "seed"}, "seed": seed})
    used_split = {"split": "tour", "config_version": split.get("config_version"), "base_config": base, "episodes": used}
    return sorted(eps, key=lambda x: x[1].episode_id), redrawn, used_split, salt


def _replay_ok(run_dir: Path, eps: dict) -> tuple[int, int]:
    from .env import AuditEnv
    recs = [json.loads(l) for l in (run_dir / "trajectories.jsonl").read_text().splitlines() if l]
    results = {r["episode_id"]: r for r in (json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l)}
    by_ep: dict[str, list] = {}
    for r in recs:
        by_ep.setdefault(r["episode_id"], []).append(r)
    ok_n = 0
    for eid, steps in by_ep.items():
        env = AuditEnv("eval")
        obs, _ = env.reset(options={"episode": eps[eid]})
        ok = dumps(obs) == dumps(steps[0]["observation"])
        for r in steps[1:]:
            got = env.abort(r["info"].get("error_code") or "timeout") if r["action"] is None else env.step(r["action"])
            ok &= dumps(list(got[:4])) == dumps([r["observation"], r["reward"], r["terminated"], r["truncated"]])
        g = env.result()
        ok &= all(g[k] == results[eid][k] for k in ("tp", "fp", "fn", "reward"))
        ok_n += ok
    return ok_n, len(by_ep)


def run_tour(out: str | None = None, quick: bool = False, split: dict | None = None) -> int:
    from .checks import run_check
    from .datacheck import verify_episode
    from .leakscan import scan_run
    from .perturb import perturb
    from .runner import code_hash, run_episode, run_policies, summarize, verify_run

    # bound to the real stdout now: steps below capture sys.stdout, the live progress must not go there. Not a terminal
    # (Docker without -t, a pipe): 160 columns instead of Rich's 80, so the scorecard is not squeezed.
    con = Console(file=sys.stdout, highlight=False, width=None if sys.stdout.isatty() or os.environ.get("COLUMNS") else 160)
    # live progress goes to stderr (bound now, before steps capture it); tables, scorecard and next steps stay on stdout
    pcon = Console(file=sys.stderr, highlight=False, width=None if sys.stderr.isatty() or os.environ.get("COLUMNS") else 160)
    t_all = time.time()
    outdir = _out_dir(out)
    report: dict = {"auditgym_version": __version__, "code_hash": code_hash(), "python": platform.python_version(),
                    "platform": platform.platform(), "steps": {}}
    con.print(Panel.fit(
        f"[bold]AuditGym {__version__} · jury tour[/]\n"
        "Runs the whole verification in about a minute (40-90 s), then prints a pass/fail scorecard for each case criterion.\n"
        "Fully offline: no network, GPU, API keys or accounts needed. Ctrl-C stops it at any time.\n"
        "4 steps: hidden-episode evaluation → paired checks → replay and leak scan → the full official check → scorecard\n"
        f"[dim]Python {platform.python_version()} · {platform.system()} {platform.machine()} · build {report['code_hash']} (the same id is in every run.json)[/]\n"
        "Report folder: " + _shown(outdir),
        border_style="cyan"))
    in_docker = os.environ.get("AUDITGYM_LAUNCHER", "").startswith("docker run")
    if in_docker and out is None:
        con.print("[yellow]The report is written inside the container and disappears with it. To keep it: mkdir -p auditgym-report && "
                  "docker run --rm --network none --user \"$(id -u):$(id -g)\" -v \"$PWD/auditgym-report:/report\" auditgym:1.0.0 tour --out /report[/]",
                  soft_wrap=True)

    # 1. closed evaluation like the organizers' ---------------------------------------------------------------------
    _step(con, "Step 1/4 · Hidden-episode evaluation", "how the organizers will test agents",
          ("Uses the episodes from your split (read from stdin). " if split is not None else
           "20 new secret episodes (5 clean, 5 with interacting errors). ")
          + "Data checked independently, then 5 policies run in isolated processes. Under 10 s.")
    t0 = time.time()
    if split is not None and os.environ.get("AUDITGYM_SALT"):
        from .runner import salt_exposures
        if salt_exposures():  # npx / run.sh wrappers hold it: a policy could read it, so the tour does not use it
            os.environ.pop("AUDITGYM_SALT", None)
            con.print("Note: AUDITGYM_SALT is visible to the npx / run.sh wrapper process, so a policy could read it. The tour "
                      "uses a new random salt instead. For a strict closed evaluation with your own salt, use the Docker recipe "
                      "in the README.",
                      style="yellow", markup=False, soft_wrap=True)
    with working(pcon) as say, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        say("Generating the hidden episodes" if split is None else "Generating the episodes of your split")
        eps, redrawn, used_split, salt = _closed_episodes(split)
        closed = outdir / "closed"
        closed.mkdir(parents=True, exist_ok=True)
        if split is None:  # only the tour's own single-use seeds/salt are saved (for `replay`); never a juror's
            (closed / "split.json").write_text(json.dumps(used_split, indent=1, ensure_ascii=False) + "\n")
            (closed / "SALT").write_text(salt + "\n")
        old = os.environ.get("AUDITGYM_SALT")
        os.environ["AUDITGYM_SALT"] = salt
        try:
            stage = f"Running 5 policies on the {len(eps)} episodes, each in an isolated process"
            say(stage)
            res = run_policies(POLICIES, eps, closed, mode="eval",
                               meta={"sources": "-", "tour": "juror split" if split is not None else "random secret seeds"},
                               on_episode=lambda spec, n, tot: say(stage, f"now: {spec}", n, tot))
        finally:
            if old is None:
                os.environ.pop("AUDITGYM_SALT", None)
            else:
                os.environ["AUDITGYM_SALT"] = old
        say("Checking each episode's data with code that does not use the generator")
        checks = []
        for _, ep in eps:
            checks.append(verify_episode(ep))
            say("Checking each episode's data with code that does not use the generator", "", len(checks), len(eps))
    n_eps = len(eps)
    data_ok_n = sum(c["ok"] for c in checks)
    con.print(f"{'[green]✓[/]' if data_ok_n == n_eps else '[red]✗[/]'} Data, checked on each of the {n_eps} episodes by code that "
              f"does not use the generator ({sum(c['issues'] for c in checks)} errors injected, "
              f"{_clean_txt(sum(1 for c in checks if not c['issues']), split is None)}):")
    con.print(f"    the books balance before errors are injected (8 accounting equalities, e.g. every entry balances): {sum(c['invariants_before'] is not False for c in checks)}/{n_eps}\n"
              f"    the problems found in the books are exactly the declared errors: {sum(c['declared_after'] for c in checks)}/{n_eps}\n"
              f"    the evidence in the answer key reproduces the amounts: {sum(c['evidence'] for c in checks)}/{n_eps}")
    rows = {}
    tb = Table(show_header=True, header_style="bold")
    for h, j in (("Policy", "left"), ("Role", "left"), ("Precision", "right"), ("Recall", "right"), ("F1", "right"), ("Reward", "right"),
                 ("Reward, clean books", "right"), ("Steps", "right"), ("Sec/episode", "right")):
        tb.add_column(h, justify=j, no_wrap=h == "Policy", style="dim" if h == "Role" else None)
    for spec in POLICIES:
        s = summarize(res[spec])["all"]
        clean = [r["reward"] for r in res[spec] if r["clean"]]
        rows[spec] = {**{k: s[k] for k in ("micro_precision", "micro_recall", "mean_f1", "mean_return", "mean_steps", "mean_wall_s")},
                      "clean_return": round(sum(clean) / len(clean), 4) if clean else None}
        r = rows[spec]
        tb.add_row(POLICY_LABEL[spec], ROLE[spec], f"{r['micro_precision']:.3f}", f"{r['micro_recall']:.3f}", f"{r['mean_f1']:.3f}",
                   f"{r['mean_return']:+.3f}", "—" if r["clean_return"] is None else f"{r['clean_return']:+.3f}",
                   f"{r['mean_steps']:.1f}", f"{r['mean_wall_s']:.3f}")
    con.print(tb)
    term_bad = {sp: sum(r["termination"] not in ("submitted", "step_limit") for r in res[sp]) for sp in POLICIES}
    oracle_one = sum(r["reward"] == 1.0 for r in res["oracle"])
    flag_clean = all(r["reward"] == -1.0 for r in res["flag_all"] if r["clean"])
    dirty = {sp: [r["reward"] for r in res[sp] if not r["clean"]] for sp in POLICIES}  # the ordering check/README assert: books with errors
    dm = {sp: sum(v) / len(v) if v else 0.0 for sp, v in dirty.items()}
    order_txt = (f"on books with errors expert {dm['expert']:+.3f} > reconcile {dm['reconcile']:+.3f} ≥ "
                 f"max(flag_all {dm['flag_all']:+.3f}, random {dm['random']:+.3f})" if dirty["expert"]
                 else "order on books with errors not checked: this split has none")
    order_ok = (not any(term_bad.values()) and oracle_one == len(res["oracle"]) and flag_clean
                and (not dirty["expert"] or dm["expert"] > dm["reconcile"] > max(dm["flag_all"], dm["random"]) - 1e-9))
    con.print(FOOTNOTE + " · " + ("seeds and salt are new on every run" + (f" ({redrawn} seeds redrawn by the generator)" if redrawn else "")
              if split is None else "seeds from your split"), style="dim", markup=False, soft_wrap=True)
    con.print(f"{'[green]✓[/]' if order_ok else '[red]✗[/]'} Rewards come out in the required order: oracle = 1.0 on {oracle_one}/{n_eps} · "
              f"flag_all = −1 on clean books: {'yes' if flag_clean else 'no'} · {order_txt} · "
              f"crashes/timeouts: {sum(term_bad.values())} · {time.time() - t0:.0f} s")
    report["steps"]["closed_eval"] = {"ok": order_ok and data_ok_n == n_eps, "policies": rows, "episodes": n_eps,
                                      "source": "juror split" if split is not None else "random", "seeds_redrawn": redrawn,
                                      "data_checks": {"ok": data_ok_n, "total": n_eps}, "bad_terminations": term_bad}

    # 2. paired checks ---------------------------------------------------------------------------------------------
    _step(con, f"Step 2/4 · {n_eps} paired checks", "does the grade depend on anything it should not?",
          "Copies each hidden episode with its rows shuffled and its document and line IDs renamed. The grade must not change: "
          "oracle stays at 1.0 and expert gets exactly the same result. Usually a few seconds.")
    t0 = time.time()
    bad = []
    with working(pcon) as say:
        for k, (_, ep) in enumerate(eps):
            pub2, truth2, _ = perturb(ep.public, ep.truth, seed=secrets.randbits(32))
            ep2 = Episode(pub2, truth2, ep.provenance, {})
            a = run_episode("oracle", ep2, "eval").result
            b, b2 = (run_episode("expert", e, "eval", isolate=False).result for e in (ep, ep2))
            if a["reward"] != 1.0 or (b["tp"], b["fp"], b["fn"], b["reward"]) != (b2["tp"], b2["fp"], b2["fn"], b2["reward"]):
                bad.append(ep.episode_id)
            say("Grading shuffled and renamed copies of each episode", "", k + 1, len(eps))  # after the item, so it ends at n/n
    paired_ok = not bad
    con.print(f"{'[green]✓[/]' if paired_ok else '[red]✗[/]'} {len(eps) - len(bad)}/{len(eps)} pairs graded the same: oracle still 1.0, "
              f"expert result unchanged · {time.time() - t0:.0f} s")
    report["steps"]["paired"] = {"ok": paired_ok, "passed": len(eps) - len(bad), "total": len(eps)}

    # 3. replay, integrity, leakscan ---------------------------------------------------------------------------------
    _step(con, "Step 3/4 · Replay, integrity, leaks", "can anyone reproduce and trust these results?",
          "Re-runs the recorded actions and grades them again, checks every run folder against the tamper-proof fingerprints "
          "(HMAC) saved in its run.json, and scans everything the policies saw for leaked answers. Usually a few seconds.")
    t0 = time.time()
    epmap = {ep.episode_id: ep for _, ep in eps}
    with working(pcon) as say:
        say("Replaying the recorded reconcile run with re-grading")
        rep_ok, rep_n = _replay_ok(outdir / "closed" / "reconcile", epmap)
    old = os.environ.get("AUDITGYM_SALT")
    os.environ["AUDITGYM_SALT"] = salt  # verify the keyed signatures of this salted run
    try:
        integ_by = {spec: verify_run(outdir / "closed" / spec) for spec in POLICIES}
    finally:
        if old is None:
            os.environ.pop("AUDITGYM_SALT", None)
        else:
            os.environ["AUDITGYM_SALT"] = old
    integ = [p for v in integ_by.values() for p in v]
    leaks = [p for spec in POLICIES for p in scan_run(outdir / "closed" / spec)]
    tail_ok = rep_ok == rep_n and not integ and not leaks
    con.print(f"{'[green]✓[/]' if rep_ok == rep_n else '[red]✗[/]'} Replay: {rep_ok}/{rep_n} episodes reproduce byte for byte and get the same grade")
    con.print(f"{'[green]✓[/]' if not integ else '[red]✗[/]'} Integrity: {sum(not v for v in integ_by.values())}/{len(POLICIES)} run folders match their tamper-proof fingerprints (HMAC) in run.json")
    con.print(f"{'[green]✓[/]' if not leaks else '[red]✗[/]'} Leak scan: {len(leaks)} leaked answers in what the policies saw · {time.time() - t0:.0f} s")
    report["steps"]["replay_integrity_leaks"] = {"ok": tail_ok, "replay": [rep_ok, rep_n], "integrity_problems": integ[:5], "leaks": leaks[:5]}

    # 4. official check (the longest step, so it runs last: results first) -------------------------------------------------------------------------------------------
    _step(con, "Step 4/4 · Official check", "the same as running `auditgym check` yourself",
          "Runs the full test suite, then the 5 baseline policies on the 20 open episodes, then checks that rewards come out in "
          "the required order, scans for leaked answers, and replays the runs. Usually 30-90 s, longer on a slow machine.")
    t0 = time.time()
    cap_out, cap_err = io.StringIO(), io.StringIO()
    juror_salt = os.environ.pop("AUDITGYM_SALT", None)  # the official check uses its own open data, never the juror's salt
    try:
        with working(pcon) as say, contextlib.redirect_stdout(cap_out), contextlib.redirect_stderr(cap_err):
            rc = run_check(quick=quick, out=outdir / "check", progress=say)
    finally:
        if juror_salt is not None:
            os.environ["AUDITGYM_SALT"] = juror_salt
    if not (outdir / "check" / "summary.json").exists():
        tail = "\n".join((cap_out.getvalue() + cap_err.getvalue()).strip().splitlines()[-8:])
        con.print(f"auditgym check could not run (exit code {rc}):\n{tail}", style="red", markup=False, soft_wrap=True)
        return 2
    summary = json.loads((outdir / "check" / "summary.json").read_text())
    by_c = {k: [0, 0] for k, _, _ in CRITERIA}
    for it in summary["checks"]:
        c = by_c[_criterion(it["id"])]
        c[0] += it["ok"]
        c[1] += 1
    tb = Table(show_header=True, header_style="bold")
    tb.add_column("Case criterion")
    tb.add_column("Checks passed", justify="right")
    for k, name, _ in CRITERIA:
        p, n = by_c[k]
        if n:  # engineering and explanation is judged from the docs, not by checks
            tb.add_row(name, f"[{'green' if p == n else 'red'}]{p}/{n}[/]")
    con.print(tb)
    n_ok, n_all = sum(i["ok"] for i in summary["checks"]), len(summary["checks"])
    con.print(f"{'[green]✓ Step 4 of 4 passed[/]' if rc == 0 else '[red]✗ Step 4 of 4 failed[/]'}: {n_ok}/{n_all} checks passed "
              f"in {time.time() - t0:.0f} s · one line per check: {_shown(outdir / 'check' / 'summary.md')}")
    failed = [f"{i['id']}: {i['detail']}"[:160] for i in summary["checks"] if not i["ok"]]
    for f in failed[:10]:
        con.print(f"  FAIL {f}", style="red", markup=False, soft_wrap=True)
    report["steps"]["check"] = {"ok": rc == 0, "passed": n_ok, "total": n_all, "by_criterion": by_c, "failed": failed}

    # scorecard -------------------------------------------------------------------------------------------------------
    all_ok = rc == 0 and order_ok and paired_ok and tail_ok and data_ok_n == n_eps
    evidence = {
        "contract": f"{by_c['contract'][0]}/{by_c['contract'][1]} checks passed; {sum(term_bad.values())} crashes; "
                    f"{len(leaks)} leaks; {sum(not v for v in integ_by.values())}/{len(POLICIES)} run folders intact",
        "data": f"{by_c['data'][0]}/{by_c['data'][1]} checks passed; {data_ok_n}/{n_eps} hidden episodes pass all three data checks",
        "reward": f"{by_c['reward'][0]}/{by_c['reward'][1]} checks passed; oracle 1.0 on {oracle_one}/{n_eps}; "
                  f"flag_all −1 on clean books: {'yes' if flag_clean else 'no'}; {order_txt}",
        "repro": f"paired checks {len(eps) - len(bad)}/{len(eps)}; replay {rep_ok}/{rep_n}",
        "eng": "judged from the docs: versions and hashes in run.json · trajectories in closed/*/trajectories.jsonl · "
               "failure analysis in docs/FAILURE_ANALYSIS.md · limitations in the README",
    }
    con.rule("[bold]Scorecard by case criterion[/]")
    tb = Table(show_header=True, header_style="bold")
    tb.add_column("Criterion")
    tb.add_column("Max points", justify="right")
    tb.add_column("Status", justify="center")
    tb.add_column("Evidence from this run")
    status = {}
    for k, name, pts in CRITERIA:
        ok_k = status[k] = {"contract": by_c[k][0] == by_c[k][1] and not leaks and not integ and not any(term_bad.values()),
                "data": by_c[k][0] == by_c[k][1] and data_ok_n == n_eps,
                "reward": by_c[k][0] == by_c[k][1] and order_ok,
                "repro": by_c[k][0] == by_c[k][1] and paired_ok and rep_ok == rep_n,
                "eng": None}[k]
        tb.add_row(name, str(pts), "[dim]not measured[/]" if ok_k is None else "[green]✓ PASS[/]" if ok_k else "[red]✗ FAIL[/]", evidence[k])
    con.print(tb)
    con.print("Max points: the jury's weight for each criterion (case section 10); the jury computes the points from its own "
              "checks. PASS: every measurement for that criterion passed in this run. Engineering and explanation is judged "
              "from the docs, so the tour does not score it.", style="dim", markup=False)
    verdict = (f"{'[bold green]✓ ALL PASSED[/]' if all_ok else '[bold red]✗ SOME CHECKS FAILED[/]'} · "
               f"{sum(v is True for v in status.values())} of {sum(v is not None for v in status.values())} measured criteria passed · "
               f"{time.time() - t_all:.0f} s · report: {_shown(outdir / 'report.md')}")
    report.update({"ok": all_ok, "seconds": round(time.time() - t_all, 1), "evidence": evidence, "status": status})
    _write_report(outdir, report, rows)

    steps = next_steps()
    ag0 = os.environ.get("AUDITGYM_LAUNCHER", "auditgym")
    if not ag0.startswith("docker run") and split is None:
        c = outdir / "closed"
        steps.insert(0, ("Re-check this tour's hidden run yourself: replay it and grade it again (its split and one-time salt are saved in the report)",
                         f"AUDITGYM_SALT=$(cat {_rel(c / 'SALT')}) {ag0} replay {_rel(c / 'reconcile')} --split - < {_rel(c / 'split.json')}"))
    con.rule("[bold]Next steps[/]  (copy any command as is)")
    top = steps[:1] if steps[0][0].startswith("Re-check") else []
    top += [st for st in steps if st[0].startswith("Test your own agent, written as a Python")]
    spl = [st for st in steps if st[0].startswith(("Make your own hidden split", "Run this whole tour"))]
    shell = not ag0.startswith("docker run") and os.environ.get("AUDITGYM_SHELL") != "powershell"
    if shell and len(spl) == 2:  # one line: a new hidden split, then the tour on it
        top.append(("Run this whole tour on your own hidden split (new random seeds in the organizers' format)",
                    f"{spl[0][1]} && {spl[1][1]}"))
    else:
        top += spl
    short = (lambda c: c.replace(ag0, "auditgym")) if shell and ag0 != "auditgym" else (lambda c: c)
    if shell and ag0 != "auditgym":
        con.print("[bold cyan]»[/] [bold]A shortcut for the commands below[/]")
        q = f'"{ag0}"' if not any(c in ag0 for c in '"$`\\') else shlex.quote(ag0)  # readable when the launcher has single quotes
        con.print(f"  alias auditgym={q}", soft_wrap=True, markup=False)
    for label, cmd in top:
        con.print(f"[bold cyan]»[/] [bold]{label}[/]")
        con.print("  " + short(cmd), soft_wrap=True, markup=False)
    con.print(f"More commands, each with when to use it: {short(ag0) if not ag0.startswith('docker run') else 'docker run --rm auditgym:1.0.0'} --help"
              " · the full list is also in report.md", style="dim", markup=False, soft_wrap=True)
    opener = "open" if sys.platform == "darwin" else "xdg-open" if sys.platform.startswith("linux") else ""
    if opener and not ag0.startswith("docker run"):
        con.print(f"[bold cyan]»[/] [bold]Read the full report[/] (scorecard, policy table, every check): "
                  f"{opener} {_rel(outdir / 'report.md')}", soft_wrap=True)
    con.print()
    con.print(verdict)  # the verdict is the last line on screen
    with open(outdir / "report.md", "a", encoding="utf-8") as f:
        f.write("\n## What you can do next\n\n" + "".join(f"{label}:\n\n```\n{cmd}\n```\n\n" for label, cmd in steps))
    return 0 if all_ok else 1


def _write_report(outdir: Path, report: dict, rows: dict) -> None:
    (outdir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    sym = {True: "✓ PASS", False: "✗ FAIL", None: "not measured"}
    md = [f"# AuditGym {report['auditgym_version']} jury tour: {'ALL PASSED' if report['ok'] else 'SOME CHECKS FAILED'}", "",
          f"Python {report['python']} · {report['platform']} · code hash {report['code_hash']} · {report['seconds']} s", "",
          "## Scorecard by case criterion", "", "| Criterion | Max points | Status | Evidence from this run |", "|---|---|---|---|"]
    md += [f"| {name} | {pts} | {sym[report['status'][k]]} | {report['evidence'][k]} |" for k, name, pts in CRITERIA]
    ce = report["steps"]["closed_eval"]
    md += ["", f"## Hidden-episode evaluation ({ce['episodes']} episodes, {'your split' if ce.get('source') == 'juror split' else 'new random seeds'})", "",
           "| Policy | Role | Precision | Recall | F1 | Reward | Reward, clean books | Steps | Sec/episode |", "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {POLICY_LABEL[s]} | {ROLE[s]} | {r['micro_precision']:.3f} | {r['micro_recall']:.3f} | {r['mean_f1']:.3f} | {r['mean_return']:+.3f} | "
           f"{'—' if r['clean_return'] is None else format(r['clean_return'], '+.3f')} | {r['mean_steps']:.1f} | {r['mean_wall_s']:.3f} |"
           for s, r in rows.items()]
    md += ["", FOOTNOTE]
    st = report["steps"]
    md += ["", "## Steps", "", f"- Official check: {st['check']['passed']}/{st['check']['total']} checks passed",
           f"- Paired checks: {st['paired']['passed']}/{st['paired']['total']}",
           f"- Replay: {st['replay_integrity_leaks']['replay'][0]}/{st['replay_integrity_leaks']['replay'][1]}; "
           f"integrity problems: {len(st['replay_integrity_leaks']['integrity_problems'])}; leaks: {len(st['replay_integrity_leaks']['leaks'])}",
           "", "Run folders: `check/`, `closed/<policy>/` (trajectories.jsonl, results.jsonl, summary.json, run.json)."]
    (outdir / "report.md").write_text("\n".join(md) + "\n")


def next_steps() -> list[tuple[str, str]]:
    ag = os.environ.get("AUDITGYM_LAUNCHER", "auditgym")  # how the user started us (npx / run.sh / docker / auditgym)
    out, ex, trust = "out", ROOT / "examples", " --trust-policies"
    if ag.startswith("docker run"):  # results must land on the host: mount a folder, run as the host user
        ag = 'mkdir -p out && docker run --rm -i --network none --user "$(id -u):$(id -g)" -v "$PWD/out:/out" auditgym:1.0.0'
        out, trust = "/out", ""
    steps = [
        ("Make your own hidden split: new random seeds in the organizers' format", f"{ag} split > my_split.json" if not ag.startswith("mkdir")
         else "docker run --rm auditgym:1.0.0 split > my_split.json"),
        ("Run this whole tour on your own split", f"{ag} tour --split - --out {out}/tour < my_split.json"),
        ("Score our adapters of the case's 4 reference policies on your split", f"{ag} run --policy random,reconcile,flag_all,oracle --split - --public-salt --out {out} < my_split.json"),
        ("Test your own agent, written as a Python class with reset/step: init copies an example into ./my-agent to edit"
         + (" (--trust-policies: it is your own code and runs with your user's rights)" if trust else ""),
         f"{ag} init my-agent && {ag} run --policy my_policy:LookAround --policy-path my-agent --split dev --out {out}{trust}"
         if trust else f"{ag} run --policy my_policy:LookAround --policy-path /app/examples --split dev --out {out}"),
        ("Test your own agent, written as any program that reads and writes JSON lines on stdin/stdout", f"{ag} run --policy \"cmd:python3 {shlex.quote(str(ex / 'stdio_policy.py'))}\" --split dev --out {out}{trust}"),
        ("Compare runs side by side, broken down by error type", f"{ag} compare {out}/random {out}/reconcile {out}/flag_all {out}/oracle --per-type"),
        ("Play an episode by hand, seeing exactly what an agent sees", f"{ag} play"),
        ("Export JSON Schemas for actions and findings (ready-made tool definitions for an LLM agent)", f"{ag} schema --out {out}/schema"),
    ]
    if os.environ.get("AUDITGYM_SHELL") == "powershell":  # Windows launcher: PowerShell syntax
        steps = [(label, _powershell(cmd)) for label, cmd in steps]
    return steps


def _powershell(cmd: str) -> str:
    cmd = cmd.replace("mkdir -p out && ", "").replace('--user "$(id -u):$(id -g)" ', "").replace('"$PWD/out:/out"', '"${PWD}/out:/out"')
    if " > my_split.json" in cmd:
        cmd = cmd.replace(" > my_split.json", " | Out-File -Encoding utf8 my_split.json")
    if " < " in cmd:
        base, src = cmd.rsplit(" < ", 1)
        cmd = f"Get-Content {src} | {base}"
    return cmd
