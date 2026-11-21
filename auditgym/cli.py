"""auditgym command line. Tables are drawn with Rich (also when piped); --format md|csv|json for machine-readable output."""
from __future__ import annotations

import argparse
import cmd
import shlex
import hashlib
import json
import sys
from pathlib import Path

from . import __version__
from .canon import loads_strict
from .episode import generate, load_public, write_episode
from .splits import ROOT, dir_episode, load_split_config, parse_seeds, split_episodes

COLUMNS = [("policy", "Policy"), ("episodes", "Episodes"), ("micro_precision", "Precision"), ("micro_recall", "Recall"), ("mean_f1", "F1"),
           ("mean_return", "Reward"), ("dirty_return", "Reward, books with errors"), ("mean_return_experimental", "Reward, partial credit"), ("mean_steps", "Steps"),
           ("clean_fp_rate", "Clean books flagged"), ("invalid", "Invalid answers"), ("mean_wall_s", "Sec/episode"), ("failed", "Crashed or timed out")]
LEGEND = ("Precision and recall: pooled over all findings · F1: mean per episode (an empty answer on clean books scores F1 = 1, "
          "so a policy can have F1 > 0 with precision and recall 0) · Reward: the case's terminal reward, -1 to +1, higher is "
          "better (2·F1 - 1 minus a small cost per step) · Reward, books with errors: the same, only on episodes that have errors (the case's reward ordering is stated there) · Reward, partial credit: an extra variant reported separately, gives half credit for the right "
          "error with a wrong amount · Clean books flagged: share of clean episodes where the policy reported any error · Invalid answers: submissions that broke "
          "the findings schema (reward -1, nothing matched)")


def _console(stderr: bool = False):
    from rich.console import Console
    import os
    tty = sys.stderr.isatty() if stderr else _tty()
    return Console(stderr=stderr, highlight=False, width=None if tty or os.environ.get("COLUMNS") else 160)  # piped/Docker: no 80-column squeeze


def rich_table(title: str, headers: list[str], rows: list[list[str]], legend: str = "", left: int = 1) -> None:
    from rich.table import Table
    t = Table(title=title or None, title_justify="left", title_style="bold")
    for i, h in enumerate(headers):
        t.add_column(h, justify="left" if i < left else "right", no_wrap=i == 0, overflow="fold",
                     min_width=max(len(w) for w in h.split()))  # IDs stay whole; headers wrap between words, never inside one
    for r in rows:
        t.add_row(*r)
    con = _console()
    con.print(t)
    if legend:
        con.print(legend, style="dim", markup=False)


def _tty() -> bool:
    return sys.stdout.isatty()


def print_table(rows: list[dict], title: str = "", fmt: str = "auto") -> None:
    if fmt == "json":
        print(json.dumps(rows, indent=2))
        return
    if fmt == "csv":
        print(",".join(h for _, h in COLUMNS))
        for r in rows:
            print(",".join(str(r.get(k, "")) for k, _ in COLUMNS))
        return
    if fmt == "auto":
        rich_table(title, [h for _, h in COLUMNS], [[_fmt(r.get(k), k) for k, _ in COLUMNS] for r in rows], LEGEND)
        return
    if title:
        print(f"### {title}\n")
    print("| " + " | ".join(h for _, h in COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in rows:
        print("| " + " | ".join(_fmt(r.get(k), k) for k, _ in COLUMNS) + " |")


def _fmt(v, key: str = "") -> str:
    if v is None:
        return "-"
    if key == "mean_steps":
        return f"{v:.1f}"
    if isinstance(v, float):
        return f"{v:.1f}" if abs(v) > 1.5 else (f"{v:+.3f}" if v < 0 else f"{v:.3f}")
    return str(v)


# ------------------------------------------------------------------------------------------ episodes
def _episodes(a) -> list:
    if getattr(a, "episodes", None):
        import os
        has_truth = any((Path(d) / "_grader").is_dir() or any(Path(d).glob("*/_grader")) for d in a.episodes)
        if has_truth and getattr(a, "fn", None) is cmd_run and a.mode == "eval":
            if os.environ.get("AUDITGYM_SALT"):
                raise ValueError("closed evaluation must not read episodes from disk: their _grader/ truth is readable by the "
                                 "policy. Use --split - < closed_split.json")
            print("warning: these episode dirs contain _grader/ truth that the policy can read; fine for open data, "
                  "not for a closed evaluation", file=sys.stderr)
        out = []
        for d in a.episodes:
            p = Path(d)
            dirs = sorted(x for x in p.iterdir() if (x / "manifest.json").exists()) if not (p / "manifest.json").exists() else [p]
            out += [(p.name, dir_episode(x)) for x in dirs]
        return out
    cfg = json.loads(Path(a.config).read_text()) if getattr(a, "config", None) else None
    if cfg is not None and "episodes" in cfg:
        if getattr(a, "seeds", None):
            raise ValueError("--config is a split file with its own episodes/seeds; use --split <file> (or --split - for a closed eval), without --seeds")
        _closed_guard(a, "config")
        return split_episodes(a.config)
    if getattr(a, "seeds", None):
        _closed_guard(a, "seeds")
        base = (cfg or {}).get("base_config", cfg or {})
        return [("seeds", generate(s, base)) for s in parse_seeds(a.seeds)]
    out = []
    for s in a.split.split(","):
        if s == "-":  # closed split on stdin: never in argv or on a path the policy could read
            _closed_guard(a, "stdin")
            text = sys.stdin.read().lstrip("\ufeff")
            if not text.strip():
                raise ValueError("no split on stdin: docker run -i ... --split - < closed_split.json")
            cfg_in = loads_strict(text)
            sys.stdin.close()
            out += sorted(split_episodes("-", cfg_in), key=lambda x: x[1].episode_id)  # order reveals nothing
        elif s in ("train", "dev"):
            out += split_episodes(s)
        else:
            _closed_guard(a, "file")
            out += split_episodes(s)
    return out


def _closed_guard(a, how: str) -> None:
    """Non-open episodes. For a closed *evaluation* (run, eval mode) the salt must be secret and strong, and the
    seeds must not be readable by the policy (argv and mounted files are readable via /proc/<runner>/cmdline)."""
    import os
    salt = os.environ.get("AUDITGYM_SALT")
    closed_eval = getattr(a, "fn", None) is cmd_run and a.mode == "eval"
    if closed_eval and not salt and not a.public_salt:
        raise ValueError("closed evaluation needs a secret salt: docker run -i -e AUDITGYM_SALT=<secret> ... run --split - < split.json "
                         "(without it a policy can regenerate the truth from the open generator). "
                         "Pass --public-salt to run these episodes with the public default salt anyway.")
    if closed_eval and salt:
        if len(salt) < 32:
            raise ValueError("AUDITGYM_SALT must be at least 32 characters (e.g. `openssl rand -hex 32`); a short salt can be "
                             "guessed offline from an episode_id")
        if how != "stdin":
            raise ValueError("closed evaluation reads the split from stdin so the policy cannot see seeds in argv or a mounted "
                             "file: docker run -i ... run --policy ... --split - < closed_split.json")
    if not salt:
        print("note: AUDITGYM_SALT is not set, so these episodes use the public default salt. That is fine for trying things "
              "out, but an agent that imports the open generator could rebuild their answer key. For a closed evaluation, "
              "set a secret salt (see the README).", file=sys.stderr)


def cmd_generate(a) -> int:
    print("  ... preparing the episodes (a large one can take a few seconds)", file=sys.stderr, flush=True)
    if a.seed is not None:
        cfg = json.loads(Path(a.config).read_text()) if a.config else {}
        a.seed = parse_seeds(a.seed)[0]
        ep = generate(a.seed, cfg.get("base_config", cfg))
        write_episode(ep, Path(a.out), with_grader=not a.no_grader)
        print(f"wrote {a.out} ({ep.episode_id})")
        return 0
    for split in a.split.split(","):
        cfg = load_split_config(split)
        out = Path(a.out) / cfg.get("split", split)
        for i, (_, ep) in enumerate(split_episodes(split)):
            write_episode(ep, out / f"ep_{i:02d}", with_grader=not a.no_grader)
        print(f"wrote {len(cfg['episodes'])} episodes to {out}")
    return 0


def cmd_run(a) -> int:
    print("  ... preparing the episodes (a large one can take a few seconds)", file=sys.stderr, flush=True)
    from .runner import harden_runner, run_policies, salt_exposures, summarize
    harden_runner()  # first: same-user children (incl. the pre-flight probe) must not read /proc/<runner>/environ|mem
    exposed = salt_exposures()  # [] unless Linux and AUDITGYM_SALT is set
    if exposed:
        raise RuntimeError(f"refusing to start: the secret salt (AUDITGYM_SALT) is visible to another process in this container "
                           f"(process id {', '.join(map(str, exposed[:5]))}, for example the init process added by docker --init, or a "
                           "launcher script), so a policy could read it too. Start the runner directly: "
                           "docker run --rm -e AUDITGYM_SALT=... auditgym:1.0.0 run ...")
    episodes = _episodes(a)
    _shape_note(a, [ep for _, ep in episodes])
    specs = a.policy.split(",")
    print(f"  ... starting {len(specs)} {'policy' if len(specs) == 1 else 'policies'} on {len(episodes)} episode{'' if len(episodes) == 1 else 's'} "
          "(first checking that each one starts)", file=sys.stderr, flush=True)
    if any(x.startswith("-") or "=" not in x for x in (a.policy_arg or [])):
        raise ValueError(f"--policy-arg takes key=value pairs, got {a.policy_arg!r}")
    from .policies import BUILTIN
    from .runner import install_writable
    if a.policy_arg and all(s in BUILTIN or s == "oracle" for s in specs):
        import inspect
        known = {p for s in specs if s in BUILTIN for p in inspect.signature(BUILTIN[s]).parameters}
        unknown = sorted({x.split("=", 1)[0] for x in a.policy_arg} - known)
        if unknown:
            raise ValueError(f"--policy-arg key(s) {unknown} are not accepted by {specs}; known: {sorted(known) or 'none'}")
    custom = [s for s in specs if s != "oracle" and s not in BUILTIN]
    if custom and not a.trust_policies and install_writable():
        raise ValueError("this Python environment is writable by your user (the auditgym package, an import root on "
                         "sys.path, or the user site-packages), and the grader runs in this process: an untrusted policy "
                         "could modify it persistently (e.g. through __pycache__ or a .pth file). Evaluate third-party "
                         "policies in the Docker image (root-owned, user site disabled), or pass --trust-policies for your own code.")
    def name(spec: str) -> str:
        n = "".join(c if c.isalnum() or c in "._-" else "_" for c in spec).lstrip(".")
        if not n:
            raise ValueError(f"invalid policy spec {spec!r}")
        return n if len(n) <= 60 else n[:51] + "_" + hashlib.sha256(spec.encode()).hexdigest()[:8]
    if len({name(s) for s in specs}) < len(specs):
        raise ValueError(f"policy specs must be distinct (run dirs: {[name(s) for s in specs]})")
    source = a.split if not (a.episodes or a.seeds or a.config) else {"episodes": a.episodes, "seeds": a.seeds, "config": a.config}
    from .runner import probe_policy
    for spec in specs:
        if a.policy_path and not Path(a.policy_path).exists() and (ROOT / a.policy_path).is_dir():
            a.policy_path = str(ROOT / a.policy_path)  # e.g. `--policy-path examples` via npx: the package's own examples
        why = probe_policy(spec, a.policy_arg, a.policy_path, a.allow_forks)
        if why:
            hint = (f" (put your module in {a.policy_path}; example: examples/my_policy.py)" if a.policy_path else
                    " (pass --policy-path DIR with your module; example: examples/my_policy.py)") if not spec.startswith("cmd:") else ""
            raise ValueError(f"policy {spec!r} does not start{hint}:\n{why}")
    from rich.console import Console
    from .tour import working
    stage = (f"Running {len(specs)} {'policy' if len(specs) == 1 else 'policies'} on {len(episodes)} episode{'' if len(episodes) == 1 else 's'}"
             + (", each in an isolated process" if not a.no_isolate else ""))
    with working(_console(stderr=True)) as say:  # progress on stderr: stdout stays the results
        say(stage, "each episode: up to 60 s and 100 steps", 0, len(specs) * len(episodes))
        results = run_policies(specs, episodes, Path(a.out), mode=a.mode, args=a.policy_arg, policy_path=a.policy_path,
                               isolate=not a.no_isolate, allow_forks=a.allow_forks, meta={"sources": source, "isolate": not a.no_isolate, "allow_forks": a.allow_forks,
                                                         "trust_policies": a.trust_policies}, dir_name=name,
                               on_episode=lambda spec, n, tot: say(stage, f"now: {spec} (each episode: up to 60 s and 100 steps)", n, tot))
    rows, dead = [], []
    for spec in specs:
        res = results[spec]
        failed = sum(1 for r in res if r["termination"] in ("timeout", "policy_crash"))
        dirty = [r["reward"] for r in res if not r["clean"]]
        rows.append({"policy": spec, **summarize(res)["all"], "failed": failed, "invalid": sum(1 for r in res if not r.get("valid", True)),
                     "dirty_return": round(sum(dirty) / len(dirty), 4) if dirty else None})
        print(f"{spec}: {len(res)} episodes -> {Path(a.out) / name(spec)}", file=sys.stderr)
        if failed:
            print(f"warning: {spec}: {failed}/{len(res)} episodes ended by crash or timeout (graded as empty answers; "
                  f"see the stderr lines above)", file=sys.stderr)
        if failed == len(res):
            dead.append(spec)
    mode = "evaluation: rewards stay hidden from the policy until the end" if a.mode == "eval" else "training: the policy sees its final reward"
    print_table(rows, f"Results ({mode}; {len(next(iter(results.values()), []))} episodes per policy)", a.format)
    if dead:
        print(f"FAIL: {', '.join(dead)} crashed or timed out on every episode: these rows are not a score", file=sys.stderr)
        return 1
    return 0


def _verified(runs) -> bool:
    from .runner import verify_run
    probs = [p for d in runs for p in verify_run(Path(d))]
    for p in probs:
        print(f"FAIL integrity: {p}", file=sys.stderr)
    return not probs


def cmd_compare(a) -> int:
    if not _verified(a.runs):
        return 1
    rows = []
    for d in a.runs:
        s = json.loads((Path(d) / "summary.json").read_text())
        meta = json.loads((Path(d) / "run.json").read_text())
        part = s.get(f"split:{a.split}") if a.split else s["all"]
        if part is None:
            raise ValueError(f"{d}: no split {a.split!r} in this run (has: {sorted(k[6:] for k in s if k.startswith('split:'))})")
        res = [json.loads(l) for l in (Path(d) / "results.jsonl").read_text().splitlines() if l]
        if a.split:
            res = [r for r in res if r.get("split") == a.split]
        dirty = [r["reward"] for r in res if not r.get("clean")]
        rows.append({"policy": meta["policy"], **part,
                     "failed": sum(part.get("termination", {}).get(t, 0) for t in ("timeout", "policy_crash")),
                     "dirty_return": round(sum(dirty) / len(dirty), 4) if dirty else None,
                     "invalid": sum(1 for r in res if not r.get("valid", True))})
    print_table(rows, "Comparison of runs", a.format)
    if a.per_type:
        per = []
        for d in a.runs:
            s = json.loads((Path(d) / "summary.json").read_text())["all"]
            per += [[Path(d).name, t, str(v["tp"]), str(v["fp"]), str(v["fn"])] for t, v in sorted(s["per_type"].items())] or \
                [[Path(d).name, "(no valid answer: nothing to match)", "-", "-", "-"]]
        if a.format == "auto":
            print()
            rich_table("By error type", ["Run", "Error type", "Found", "False alarms", "Missed"], per,
                       "Found: planted errors the policy reported · False alarms: reported errors that were not planted · "
                       "Missed: planted errors it did not report", left=2)
        else:
            print(json.dumps([dict(zip(("run", "type", "tp", "fp", "fn"), r)) for r in per], indent=2) if a.format == "json" else
                  "\n".join(["run,type,tp,fp,fn"] + [",".join(r) for r in per]))
    return 0


def cmd_grade(a) -> int:
    from .env import AuditEnv
    d = Path(a.episode)
    findings = Path(a.findings).read_text()  # raw text: the env parses it strictly (duplicate keys, NaN -> invalid)
    env = AuditEnv("train")
    env.reset(options={"episode_dir": d})
    for _ in range(a.steps):
        env.step({"name": "calculator", "arguments": {"expression": "0"}})
    env.step({"name": "submit", "arguments": {"findings": findings}})
    print(json.dumps(env.result(), indent=2))
    return 0


def cmd_replay(a) -> int:
    from .canon import dumps
    from .env import AuditEnv
    from .policies import OraclePolicy
    from .runner import WITHHELD
    run = Path(a.run)
    print(f"  ... replaying {run}: checking its files, regenerating its episodes, re-running every action", file=sys.stderr, flush=True)
    if not _verified([run]):
        return 1
    meta = json.loads((run / "run.json").read_text())
    if meta.get("auditgym_version") != __version__:
        print(f"FAIL version mismatch: run made with {meta.get('auditgym_version')}, this is {__version__}")
        return 1
    if not (a.episodes or a.seeds or a.config) and a.split == "dev" and isinstance(meta.get("sources"), str):
        if meta["sources"] == "-" and sys.stdin.isatty():
            print("this is a closed run: pass the same split on stdin (replay RUN --split - < split.json) and AUDITGYM_SALT")
            return 1
        a.split = meta["sources"]
    recs = [json.loads(l) for l in (run / "trajectories.jsonl").read_text().splitlines() if l]
    eps = {ep.episode_id: ep for _, ep in _episodes(a)}
    by_ep: dict[str, list] = {}
    for r in recs:
        by_ep.setdefault(r["episode_id"], []).append(r)
    results = {r["episode_id"]: r for r in (json.loads(l) for l in (run / "results.jsonl").read_text().splitlines() if l)}
    replayed = mismatches = 0
    for eid, steps in by_ep.items():
        if eid not in eps:
            print(f"SKIP {eid}: episode not in the given source")
            continue
        env = AuditEnv(meta["mode"])
        obs, info = env.reset(options={"episode": eps[eid]})
        ok = dumps(obs) == dumps(steps[0]["observation"])
        for r in steps[1:]:
            act = r["action"]
            if meta.get("policy") == "oracle" and isinstance(act, dict) and act.get("arguments") == WITHHELD:
                act = OraclePolicy(eps[eid].truth).reset(steps[0]["observation"], None)  # closed run: rebuild from regenerated truth
            got = env.abort(r["info"].get("error_code") or "timeout") if act is None else env.step(act)
            ok &= dumps(list(got[:4])) == dumps([r["observation"], r["reward"], r["terminated"], r["truncated"]])
        g, res = env.result(), results.get(eid, {})
        if any(g[k] != res.get(k) for k in ("valid", "tp", "fp", "fn", "f1", "reward")):  # re-graded here, not trusted
            print(f"DIFF {eid}: re-graded {[g[k] for k in ('tp', 'fp', 'fn', 'reward')]} != results.jsonl "
                  f"{[res.get(k) for k in ('tp', 'fp', 'fn', 'reward')]}")
            ok = False
        replayed += 1
        mismatches += not ok
        print(f"{'OK  ' if ok else 'DIFF'} {eid} ({len(steps) - 1} steps)")
    if not replayed:
        print("FAIL nothing replayed: pass the episode source (--split/--episodes/--seeds) used for the run")
        return 1
    from .runner import summarize
    if replayed == len(results) and summarize(list(results.values())) != json.loads((run / "summary.json").read_text()):
        print("DIFF summary.json does not match results.jsonl")
        mismatches += 1
    print(f"{replayed - mismatches}/{replayed} episodes replay byte-identically and re-grade to results.jsonl")
    return 1 if mismatches else 0


def cmd_leakscan(a) -> int:
    from .leakscan import scan_run
    bad = 0
    for d in a.runs:
        problems = scan_run(Path(d))
        bad += len(problems)
        print(f"{'PASS' if not problems else 'FAIL'} {d}: {len(problems)} problems")
        for p in problems[:20]:
            print("  " + p)
    print(f"{'PASS' if not bad else 'FAIL'}: {len(a.runs)} runs scanned, {bad} problems")
    return 1 if bad else 0


def cmd_perturb(a) -> int:
    from .episode import Episode, load_grader
    from .perturb import perturb
    d = Path(a.episode)
    truth, prov = load_grader(d)
    pub, tr, m = perturb(load_public(d), truth, a.seed, rename=not a.no_rename, shuffle=not a.no_shuffle)
    write_episode(Episode(pub, tr, prov, {}), Path(a.out))
    (Path(a.out) / "_grader" / "id_mapping.json").write_text(json.dumps(m, indent=1) + "\n")
    print(f"wrote {a.out}: {len(m)} ids renamed, rows shuffled={not a.no_shuffle}")
    return 0


def cmd_schema(a) -> int:
    from .schemas import json_schemas
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, sch in json_schemas().items():
        (out / name).write_text(json.dumps(sch, indent=2) + "\n")
    print(f"wrote {len(json_schemas())} schemas to {out}")
    return 0


KINDS = ("Kind: interacting X1 = a partial foreign-currency payment across the period end, posted late at the wrong rate "
         "(3 linked errors); X2 = an invoice with a credit note that is later billed twice; the credit note is posted at the invoice's rate (2 linked errors).")


def _shape_note(a, eps) -> None:
    """The case's closed check uses 20 hidden episodes, 5 clean and 5 interacting: say so if a split from stdin differs."""
    from .datacheck import kind
    if getattr(a, "split", None) != "-":
        return
    ks = [kind(e) for e in eps]
    n_clean, n_x = ks.count("clean"), sum(k.startswith("interacting") for k in ks)
    if (len(ks), n_clean >= 5, n_x >= 5) != (20, True, True):
        print(f"note: this split has {len(ks)} episodes ({n_clean} clean, {n_x} interacting); the case's closed check uses 20 "
              "(5 clean, 5 interacting). Fine for exploring.", file=sys.stderr)


def _yes(v) -> str:
    return "n/a" if v is None else "yes" if v else "NO"


def cmd_datacheck(a) -> int:
    """The case's data check (section 10): on each episode, the books balance before the errors are injected, the
    injected books break exactly the declared errors, and every evidence set reproduces its amount from its own records.
    Uses code that does not import the generator. Episodes from a split are regenerated (so the books before injection
    exist); episode folders on disk only have the injected books, so that column says n/a."""
    print("  ... preparing the episodes (a large one can take a few seconds)", file=sys.stderr, flush=True)
    from rich.console import Console
    from .datacheck import kind, kind_counts, verify_episode
    from .tour import working
    rows, bad, eps = [], 0, sorted((ep for _, ep in _episodes(a)), key=lambda e: getattr(e, 'split_pos', 0))  # split order
    _shape_note(a, eps)
    with working(_console(stderr=True)) as say:
        checks = []
        for ep in eps:
            say(f"Checking the data of {len(eps)} episodes", "", len(checks), len(eps))
            checks.append(verify_episode(ep))
        say(f"Checking the data of {len(eps)} episodes", "", len(checks), len(eps))
    for ep, c in zip(eps, checks):
        bad += not c["ok"]
        rows.append([str(getattr(ep, "split_pos", "")), ep.episode_id, kind(ep), str(c["issues"]), _yes(c["invariants_before"]), _yes(c["declared_after"]),
                     _yes(c["evidence"]), "✓ PASS" if c["ok"] else "✗ FAIL"])
    rich_table("Independent data check (case section 10, 'Data and answer key')",
               ["#", "Episode", "Kind", "Errors injected", "Books balance before injection (I1-I8)", "Problems found = declared errors",
                "Evidence reproduces the amounts", "Result"], rows,
               "I1 each journal entry balances · I2 trial balance nets to zero · I3 AR control account = subledger · I4 invoice "
               "lines add up to the total · I5 FX conversion follows the policy · I6 period cut-off · I7 payments are allocated "
               "in full · I8 each delivery is billed once. Code: auditgym/datacheck.py, auditgym/invariants.py, "
               "auditgym/recheck.py; none of it imports the generator. " + KINDS)
    print(f"{len(rows) - bad}/{len(rows)} episodes pass all data checks ({kind_counts(eps)})")
    return 1 if bad else 0


def cmd_paired(a) -> int:
    """The case's paired checks (section 06/10): each episode is copied with its rows shuffled and its document and line
    IDs renamed; the oracle's answer under the new IDs must still get reward 1.0 (this decides the result and exit code).
    For every other --policy, whether its answer stays the same is reported per policy: a property of that policy."""
    print("  ... preparing the episodes (a large one can take a few seconds)", file=sys.stderr, flush=True)
    from .episode import Episode
    from .perturb import perturb
    from .runner import harden_runner, run_episode, salt_exposures
    harden_runner()
    if salt_exposures():
        raise RuntimeError("AUDITGYM_SALT is readable by another process (a wrapper such as npx/run.sh or docker --init); "
                           "run paired checks on a closed split with: docker run -i --rm -e AUDITGYM_SALT=... auditgym:1.0.0 paired --split - < split.json")
    from .datacheck import kind, kind_counts
    from .policies import BUILTIN
    specs = [s for s in a.policy.split(",") if s != "oracle"]
    if a.policy_path and not Path(a.policy_path).exists() and (ROOT / a.policy_path).is_dir():
        a.policy_path = str(ROOT / a.policy_path)
    from rich.console import Console
    from .tour import working
    rows, bad, eps = [], 0, sorted((ep for _, ep in _episodes(a)), key=lambda e: getattr(e, 'split_pos', 0))  # split order
    _shape_note(a, eps)
    stage = f"Grading shuffled and renamed copies of {len(eps)} episodes (oracle + {', '.join(specs)})"
    ctx = working(_console(stderr=True))
    say = ctx.__enter__()
    units, done_u = len(eps) * (1 + 2 * len(specs)), [0]  # one unit = one policy run on one episode (original or copy)

    def step(what: str, k: int) -> None:
        say(stage, f"now: {what}, episode {k + 1}/{len(eps)} (each run is cut at 60 s)", done_u[0], units,
            f"{done_u[0]}/{units} policy runs")
    say(stage, "", 0, units, f"0/{units} policy runs")
    per = {sp: {"same": 0, "steps_only": 0, "changed": [], "crashed": 0} for sp in specs}
    for k, ep in enumerate(eps):
        pub2, truth2, mapping = perturb(ep.public, ep.truth, seed=a.seed + k)
        ep2 = Episode(pub2, truth2, ep.provenance, {})
        step("oracle on the shuffled copy", k)
        o2 = run_episode("oracle", ep2, "eval").result
        done_u[0] += 1
        changed = []
        for spec in specs:
            rr = []
            for e, which in ((ep, "original"), (ep2, "shuffled copy")):
                step(f"{spec} on the {which}", k)
                rr.append(run_episode(spec, e, "eval", a.policy_arg, a.policy_path, isolate=spec not in BUILTIN).result)
                done_u[0] += 1
            r1, r2 = rr
            per[spec]["crashed"] += any(r.get("termination") in ("timeout", "policy_crash") for r in (r1, r2))
            hit = lambda r: sorted(m["issue_id"] for m in r.get("matches", []) if m.get("result") == "TP")  # issue ids survive renaming
            if all(r1[f] == r2[f] for f in ("tp", "fp", "fn")) and hit(r1) == hit(r2):  # the same errors found, same counts
                per[spec]["same"] += 1
                per[spec]["steps_only"] += r1["reward"] != r2["reward"]
            else:
                changed.append(spec)
                per[spec]["changed"].append(f"{ep.episode_id} {r1['tp']}/{r1['fp']}/{r1['fn']} {hit(r1)} → {r2['tp']}/{r2['fp']}/{r2['fn']} {hit(r2)}")
        ok = o2["reward"] == 1.0  # the case's paired check: the oracle's answer under the new IDs still gets the maximum
        bad += not ok
        rows.append([str(getattr(ep, "split_pos", "")), ep.episode_id, kind(ep), str(len(mapping)), f"{o2['reward']:+.3f}",
                     ("yes" if not changed else "NO, changed") if len(specs) == 1 else
                     f"{len(specs) - len(changed)}/{len(specs)}" + (f" (changed: {', '.join(changed)})" if changed else ""),
                     "✓ PASS" if ok else "✗ FAIL"])
        say(stage, "", done_u[0], units, f"{done_u[0]}/{units} policy runs")
    ctx.__exit__(None, None, None)
    rich_table("Paired checks (case sections 06 and 10, 'Reproducibility and generalization')",
               ["#", "Episode", "Kind", "IDs renamed", "Oracle reward on the copy",
                f"{specs[0]}: same answer on the copy" if len(specs) == 1 else "Other policies with the same answer", "Result"], rows,
               "Each copy has its rows shuffled and its document and line IDs renamed (auditgym/perturb.py). The case's paired "
               "check, and this command's result and exit code: the oracle's answer, mapped to the new IDs, still gets +1.000. " + KINDS)
    print()
    for spec, v in per.items():  # a property of each policy, not of the environment
        if v["crashed"]:
            print(f"warning: {spec} crashed or timed out on {v['crashed']}/{len(eps)} episodes (original or copy); those answers "
                  "count as empty, so 'same answer' below says little about it. Run it with `run` to see its stderr.")
        if not v["changed"]:
            print(f"{spec}: same answer (errors found, false alarms, misses) on {v['same']}/{len(eps)} copies"
                  + (f"; on {v['steps_only']} of them only the reward differs, from a different step count" if v["steps_only"] else ""))
        else:
            print(f"{spec}: answer changed on {len(v['changed'])}/{len(eps)} copies (found/false alarms/missed and the errors found, original → copy: "
                  f"{'; '.join(v['changed'][:3])}{' ...' if len(v['changed']) > 3 else ''}). This policy depends on row order or IDs; "
                  "it is a property of the policy, not an environment failure.")
    print(f"{len(rows) - bad}/{len(rows)} paired checks pass ({kind_counts(eps)}; case section 10: oracle +1.000 on every copy)")
    return 1 if bad else 0


def cmd_init(a) -> int:
    """Copy the example agent into a folder you can edit, and print how to run it."""
    import os
    import shutil
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    dst = d / "my_policy.py"
    if dst.exists():
        raise ValueError(f"{dst} already exists; pick another folder: auditgym init <folder>")
    shutil.copyfile(ROOT / "examples" / "my_policy.py", dst)
    ag = os.environ.get("AUDITGYM_LAUNCHER", "auditgym")
    print(f"wrote {dst}: a toy agent (class LookAround) that reads a few invoice pages and accuses nobody; edit reset/step.")
    print("Run it on the 10 open dev episodes (--trust-policies: it is your own code and runs with your user's rights):")
    print(f"  auditgym run --policy my_policy:LookAround --policy-path {shlex.quote(str(d))} --split dev --out out --trust-policies")
    if ag != "auditgym":
        print(f"  (auditgym = your launch command: {ag})")
    print("Expect reward +1 on clean books and -1 where errors exist, since it never reports any: about -0.6 on dev, where "
          "2 of the 10 books are clean. Our expert scores about +0.95 there; oracle +1.0.")
    return 0


def cmd_check(a) -> int:
    from .checks import run_check
    if a.json:
        return run_check(quick=a.quick, as_json=True, out=Path(a.out))
    from .tour import working  # a live status line in a terminal, one line per stage otherwise
    con = _console()
    con.print("auditgym check: the test suite (about 570 tests), then the 5 baseline policies on the 20 open episodes, reward-order "
              "checks, a leak scan, 20 paired checks and a replay. Usually 30-90 s at 2 CPU; a results table and "
              "PASS or FAIL with the number of checks passed come at the end.", markup=False, soft_wrap=True)
    with working(con) as say:
        return run_check(quick=a.quick, out=Path(a.out), progress=say)


# ------------------------------------------------------------------------------------------ play
class PlayShell(cmd.Cmd):
    intro = ("AuditGym interactive episode. Commands: list <table> [cursor] [limit] | get <table> <id> | doc <doc_id> | "
             "calc <expr> | submit <findings.json | inline JSON> | raw <action JSON> | quit")
    prompt = "audit> "

    def __init__(self, env, obs):
        super().__init__()
        self.env, self.obs = env, obs

    def _do(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        print(json.dumps(obs, indent=1, ensure_ascii=False)[:20000])
        if term or trunc:
            print(f"\nepisode over: reward={r} terminated={term} truncated={trunc}")
            g = self.env.result()
            print(json.dumps({k: g[k] for k in ("valid", "error", "tp", "fp", "fn", "precision", "recall", "f1", "reward") if k in g}, indent=1, ensure_ascii=False))
            return True
        return False

    def do_list(self, arg):
        p = arg.split()
        if not p:
            print("usage: list <table> [cursor] [limit]")
            return False
        args = {"table": p[0]}
        if len(p) > 1 and p[1] != "-":
            args["cursor"] = p[1]
        if len(p) > 2:
            args["limit"] = int(p[2])
        return self._do({"name": "list_records", "arguments": args})

    def do_get(self, arg):
        t, _, i = arg.partition(" ")
        return self._do({"name": "get_record", "arguments": {"table": t, "id": i.strip()}})

    def do_doc(self, arg):
        return self._do({"name": "get_document", "arguments": {"doc_id": arg.strip()}})

    def do_calc(self, arg):
        return self._do({"name": "calculator", "arguments": {"expression": arg}})

    def do_submit(self, arg):
        arg = arg.strip()
        data = loads_strict(Path(arg).read_text()) if arg and Path(arg).exists() else loads_strict(arg or "null")
        return self._do({"name": "submit", "arguments": {"findings": data}})

    def do_raw(self, arg):
        return self._do(loads_strict(arg))

    def do_quit(self, arg):
        return True

    do_EOF = do_quit

    def emptyline(self):
        return False


def cmd_play(a) -> int:
    from .env import AuditEnv
    eps = _episodes(a)
    _, ep = eps[a.index]
    env = AuditEnv("train")
    obs, _ = env.reset(options={"episode": ep})
    print(json.dumps({k: obs[k] for k in ("task", "manifest", "tables", "steps_remaining")}, indent=1, ensure_ascii=False))
    PlayShell(env, obs).cmdloop()
    return 0


# ------------------------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="auditgym", description=START.format(v=__version__), formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"auditgym {__version__}")
    sub = p.add_subparsers(dest="cmd", title="all commands", metavar="<command>")

    def cmdp(name: str, h: str):
        return sub.add_parser(name, help=h, description=h[0].upper() + h[1:] + ".")

    t = cmdp("tour", "start here: the whole jury check in about a minute (40-90 s), ending with a pass/fail scorecard for each case criterion")
    t.add_argument("--out", help="report directory (default: ./auditgym-report)")
    t.add_argument("--quick", action="store_true", help="fewer fresh seeds in the test step")
    t.add_argument("--split", help="'-' = run steps 2-4 on YOUR split from stdin (AUDITGYM_SALT if set and not visible to a wrapper process such as npx/run.sh, else a random salt)")
    t.set_defaults(fn=cmd_tour)

    spl = cmdp("split", "write a new hidden split: random seeds in the organizers' format (use: auditgym split > my_split.json)")
    spl.set_defaults(fn=cmd_split)

    def source(sp, default="dev"):
        sp.add_argument("--split", default=default, help="split name(s) from configs/, comma separated, or '-' to read a split JSON "
                        "from stdin (default: %(default)s)")
        sp.add_argument("--episodes", nargs="+", help="episode directories (or a directory of episodes) instead of a split")
        sp.add_argument("--seeds", help="seed list/ranges, e.g. 1-20,40 (closed runs: set AUDITGYM_SALT)")
        sp.add_argument("--config", help="config JSON for --seeds (a split file with 'episodes' is used as a split)")

    c = cmdp("check", "the official test suite and baseline checks on their own, offline; exit code 0 means everything passed")
    c.add_argument("--quick", action="store_true", help="fewer fresh seeds (CI smoke)")
    c.add_argument("--json", action="store_true", help="print one JSON document")
    c.add_argument("--out", default=str(ROOT / "runs" / "check"))
    c.set_defaults(fn=cmd_check)

    g = cmdp("generate", "write episode folders to disk (the open splits or one seed) so you can look at the data")
    g.add_argument("--split", default="train,dev")
    g.add_argument("--seed", help="integer or string seed")
    g.add_argument("--config")
    g.add_argument("--out", default=str(ROOT / "data"))
    g.add_argument("--no-grader", action="store_true", help="do not write _grader/ (truth, provenance)")
    g.set_defaults(fn=cmd_generate)

    r = cmdp("run", "test agents (policies) on a set of episodes: grades every answer and saves trajectories, results and a summary")
    r.add_argument("--policy", required=True, help="the agent(s) to test, comma separated: a built-in (random, reconcile, flag_all, "
                   "oracle, expert), your Python class as module:Class, or any program as 'cmd:<command>'")
    source(r)
    r.add_argument("--mode", choices=["eval", "train"], default="eval")
    r.add_argument("--out", default=str(ROOT / "runs"))
    r.add_argument("--policy-arg", action="append", default=[], help="k=v passed to the policy constructor")
    r.add_argument("--policy-path", help="directory added to the policy child's sys.path")
    r.add_argument("--no-isolate", action="store_true", help="run builtin policies in-process (debug only)")
    r.add_argument("--trust-policies", action="store_true", help="allow custom policies on a user-writable install (your own code only)")
    r.add_argument("--public-salt", action="store_true", help="allow an eval run of non-open episodes without AUDITGYM_SALT (not a closed evaluation)")
    r.add_argument("--allow-forks", action="store_true", help="let the policy spawn processes/threads (weakens isolation; e.g. node)")
    r.add_argument("--format", choices=["auto", "md", "json", "csv"], default="auto", help="auto = a table; md, json or csv for files and scripts")
    r.set_defaults(fn=cmd_run)

    cp = cmdp("compare", "show finished runs side by side; --per-type adds found / false alarms / missed for each error type")
    cp.add_argument("runs", nargs="+")
    cp.add_argument("--split")
    cp.add_argument("--per-type", action="store_true", help="add a table of found / false alarms / missed per error type")
    cp.add_argument("--format", choices=["auto", "md", "json", "csv"], default="auto", help="auto = a table; md, json or csv for files and scripts")
    cp.set_defaults(fn=cmd_compare)

    gr = cmdp("grade", "grade one findings.json against one episode folder and print how each finding was matched")
    gr.add_argument("--episode", required=True)
    gr.add_argument("--findings", required=True)
    gr.add_argument("--steps", type=int, default=0, help="inspection steps to charge")
    gr.set_defaults(fn=cmd_grade)

    rp = cmdp("replay", "re-run a finished run step by step and confirm it reproduces byte for byte and gets the same grade")
    rp.add_argument("run")
    source(rp)
    rp.set_defaults(fn=cmd_replay)

    lk = cmdp("leakscan", "scan the trajectories of runs for answer-key data that leaked into what the agent saw")
    lk.add_argument("runs", nargs="+")
    lk.set_defaults(fn=cmd_leakscan)

    pt = cmdp("perturb", "write a copy of an episode with shuffled rows and renamed IDs; its grade must not change (paired checks)")
    pt.add_argument("--episode", required=True)
    pt.add_argument("--out", required=True)
    pt.add_argument("--seed", type=int, default=1)
    pt.add_argument("--no-rename", action="store_true")
    pt.add_argument("--no-shuffle", action="store_true")
    pt.set_defaults(fn=cmd_perturb)

    ini = cmdp("init", "copy the example agent into a folder you can edit (default ./my-agent) and print the command that runs it")
    ini.add_argument("dir", nargs="?", default="my-agent")
    ini.set_defaults(fn=cmd_init)

    dc = cmdp("datacheck", "check the data of episodes independently: books balance before injection, the injected books break "
                           "exactly the declared errors, evidence reproduces the amounts (case section 10)")
    source(dc)
    dc.set_defaults(fn=cmd_datacheck)

    pc = cmdp("paired", "paired checks (case sections 06 and 10): shuffle rows and rename IDs in each episode; the oracle must stay "
                        "at 1.0 (this sets the exit code); for other policies it reports whether their answer changes, a property "
                        "of the policy")
    source(pc)
    pc.add_argument("--seed", type=int, default=1, help="seed of the first shuffle/rename; episode k uses seed + k")
    pc.add_argument("--policy", default="expert", help="policies to report on (does their answer change on the copy?), comma separated: built-ins, "
                    "module:Class (with --policy-path) or 'cmd:<command>'; oracle is always checked (default: %(default)s)")
    pc.add_argument("--policy-path", help="directory with your policy module")
    pc.add_argument("--policy-arg", action="append", default=[], help="k=v passed to the policy constructor")
    pc.set_defaults(fn=cmd_paired)

    pl = cmdp("play", "play one episode by hand, seeing exactly what an agent sees (train mode)")
    source(pl)
    pl.add_argument("--index", type=int, default=0)
    pl.set_defaults(fn=cmd_play)

    sc = cmdp("schema", "export JSON Schemas for actions and findings (ready-made tool definitions for an LLM agent)")
    sc.add_argument("--out", default=str(ROOT / "schema"))
    sc.set_defaults(fn=cmd_schema)
    return p


START = """AuditGym {v}: an RL environment and grader for AI financial auditors.
It generates a month of bookkeeping for a made-up company, plants accounting errors in it,
and grades whether an agent finds them. Runs offline: no network, API keys, GPU or accounts.

START HERE
  auditgym tour
      The whole jury check in about a minute (40-90 s). Ends with a pass/fail scorecard for each case criterion.

CHECK THE GRADER ON SEEDS ONLY YOU KNOW
  auditgym split > my_split.json
      Writes a new hidden split: random seeds in the organizers' format
      (5 clean books, 5 with interacting errors, 10 others).
  auditgym tour --split - < my_split.json
      Runs the tour on the episodes from that file instead of its own.
  auditgym datacheck --split - < my_split.json
      Checks each episode's data independently: books balance before the errors, the errors are exactly the declared ones.
  auditgym paired --split - < my_split.json
      Shuffles rows and renames IDs in each episode; the oracle must stay at 1.0 and our expert must not change.
  (The organizers' full process from the case, step by step: README section "Проверка организаторов по разделу 06 кейса".)

TEST AN AGENT
  auditgym init my-agent
      Copies an example agent into ./my-agent for you to edit, and prints the command that runs it.
  auditgym run --policy reconcile,random --split dev --out out
      Runs agents on a set of episodes and grades every answer: precision, recall, F1, reward.
  auditgym compare out/reconcile out/random --per-type
      Shows finished runs side by side, with found / false alarms / missed per error type.
  auditgym replay out/reconcile
      Re-runs a finished run and confirms it reproduces exactly and gets the same grade.

LOOK INSIDE
  auditgym play      Play one episode by hand, seeing exactly what an agent sees.
  auditgym check     The official test suite on its own. Exit code 0 means everything passed.
  auditgym schema    Export JSON Schemas for actions and findings (tool definitions for an LLM agent).

TERMS
  policy   the agent being tested. Built in: random, reconcile, flag_all and oracle (the case's four
           reference policies; oracle reads the answer key) and expert (our solver, no answer key).
  episode  one made-up company's books for one month, clean or with planted errors.
  split    a list of episode seeds. train and dev are open; a hidden split is kept by you or the organizers.
  salt     a secret mixed into the seeds so an agent cannot rebuild the answer key from the open generator.

Options of one command: auditgym <command> --help
"""


def cmd_tour(a) -> int:
    from .tour import run_tour
    split = None
    if a.split:
        if a.split != "-":
            raise ValueError("tour --split takes '-' (the split JSON on stdin), e.g. auditgym tour --split - < my_split.json")
        text = sys.stdin.read().lstrip("\ufeff")
        if not text.strip():
            raise ValueError("no split on stdin: auditgym tour --split - < my_split.json  (make one: auditgym split > my_split.json)")
        split = loads_strict(text)
    return run_tour(a.out, a.quick, split)


def cmd_split(a) -> int:
    """A new closed-split file in the organizers' shape (5 clean, 5 interacting, 10 other) with fresh random seeds."""
    import secrets
    cfg = load_split_config(str(ROOT / "configs" / "closed_example.json"))
    cfg["split"] = "jury"
    cfg["note"] = "Generated by `auditgym split`: fresh random seeds in the shape of configs/closed_example.json. Edit freely."
    for e in cfg["episodes"]:
        e["seed"] = f"jury-{secrets.token_hex(8)}"
    print(json.dumps(cfg, indent=1, ensure_ascii=False))
    print(f"wrote a split with {len(cfg['episodes'])} episodes: 5 clean, 5 interacting, 10 other, with fresh random seeds "
          "(datacheck shows the mix each episode really got)", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if not getattr(a, "fn", None):
        print(START.format(v=__version__))
        return 0
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        print("\nstopped. Nothing is lost: run the same command again (downloads and setup are cached).", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, KeyError, json.JSONDecodeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        if isinstance(e, PermissionError):
            print("hint: in Docker the image runs as user `auditor`; on Linux add --user \"$(id -u):$(id -g)\" for mounted "
                  "output dirs, and write generate/schema output under /tmp or a mounted dir", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
