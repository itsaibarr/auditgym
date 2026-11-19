"""auditgym command line. Plain text/markdown when piped; Rich tables in a terminal."""
from __future__ import annotations

import argparse
import cmd
import hashlib
import json
import sys
from pathlib import Path

from . import __version__
from .canon import loads_strict
from .episode import generate, load_public, write_episode
from .splits import ROOT, dir_episode, load_split_config, parse_seeds, split_episodes

COLUMNS = [("policy", "policy"), ("episodes", "episodes"), ("micro_precision", "P"), ("micro_recall", "R"), ("mean_f1", "F1"),
           ("mean_return", "return"), ("mean_return_experimental", "return_exp"), ("mean_steps", "steps"),
           ("clean_fp_rate", "clean-FP"), ("mean_wall_s", "s/ep")]


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
    if fmt == "auto" and _tty():
        from rich.console import Console
        from rich.table import Table
        t = Table(title=title or None)
        for _, h in COLUMNS:
            t.add_column(h, justify="left" if h == "policy" else "right")
        for r in rows:
            t.add_row(*[_fmt(r.get(k)) for k, _ in COLUMNS])
        Console().print(t)
        return
    if title:
        print(f"### {title}\n")
    print("| " + " | ".join(h for _, h in COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in rows:
        print("| " + " | ".join(_fmt(r.get(k)) for k, _ in COLUMNS) + " |")


def _fmt(v) -> str:
    if v is None:
        return "-"
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
            text = sys.stdin.read()
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
        print("warning: AUDITGYM_SALT is not set; non-open episodes use the PUBLIC default salt, so a policy that imports the "
              "open generator could regenerate their truth.", file=sys.stderr)


def cmd_generate(a) -> int:
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
    from .runner import run_policies, summarize
    episodes = _episodes(a)
    specs = a.policy.split(",")
    if any(x.startswith("-") or "=" not in x for x in (a.policy_arg or [])):
        raise ValueError(f"--policy-arg takes key=value pairs, got {a.policy_arg!r}")
    from .policies import BUILTIN
    from .runner import install_writable
    custom = [s for s in specs if s != "oracle" and s not in BUILTIN]
    if custom and not a.no_isolate and not a.trust_policies and install_writable():
        raise ValueError("this auditgym installation is writable by your user, and the grader runs in this process: an "
                         "untrusted policy could modify it persistently (e.g. through __pycache__). Evaluate third-party "
                         "policies in the Docker image (package owned by root), or pass --trust-policies for your own code.")
    def name(spec: str) -> str:
        n = "".join(c if c.isalnum() or c in "._-" else "_" for c in spec).lstrip(".")
        if not n:
            raise ValueError(f"invalid policy spec {spec!r}")
        return n if len(n) <= 60 else n[:51] + "_" + hashlib.sha256(spec.encode()).hexdigest()[:8]
    if len({name(s) for s in specs}) < len(specs):
        raise ValueError(f"policy specs must be distinct (run dirs: {[name(s) for s in specs]})")
    source = a.split if not (a.episodes or a.seeds or a.config) else {"episodes": a.episodes, "seeds": a.seeds, "config": a.config}
    results = run_policies(specs, episodes, Path(a.out), mode=a.mode, args=a.policy_arg, policy_path=a.policy_path,
                           isolate=not a.no_isolate, allow_forks=a.allow_forks, meta={"sources": source}, dir_name=name)
    rows = []
    for spec in specs:
        res = results[spec]
        rows.append({"policy": spec, **summarize(res)["all"]})
        bad = [r for r in res if r["termination"] in ("timeout", "policy_crash")]
        print(f"{spec}: {len(res)} episodes -> {Path(a.out) / name(spec)}", file=sys.stderr)
        if bad:
            print(f"warning: {spec}: {len(bad)}/{len(res)} episodes ended by {sorted({r['termination'] for r in bad})} "
                  f"(policy failed to start, crashed or timed out; graded as empty answers). Only if the policy itself needs "
                  f"threads or subprocesses: --allow-forks lifts RLIMIT_NPROC=1 (trusted policies only; see README, Isolation)", file=sys.stderr)
    print_table(rows, f"AuditGym {a.mode} run", a.format)
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
        rows.append({"policy": meta["policy"], **part})
    print_table(rows, "comparison", a.format)
    if a.per_type:
        for d in a.runs:
            s = json.loads((Path(d) / "summary.json").read_text())["all"]
            print(f"\n{Path(d).name} per type: " + ", ".join(f"{t}: tp={v['tp']} fp={v['fp']} fn={v['fn']}" for t, v in sorted(s["per_type"].items())))
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
        replayed += 1
        mismatches += not ok
        print(f"{'OK  ' if ok else 'DIFF'} {eid} ({len(steps) - 1} steps)")
    if not replayed:
        print("FAIL nothing replayed: pass the episode source (--split/--episodes/--seeds) used for the run")
        return 1
    print(f"{replayed - mismatches}/{replayed} episodes replay byte-identically")
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


def cmd_check(a) -> int:
    from .checks import run_check
    return run_check(quick=a.quick, as_json=a.json, out=Path(a.out))


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
            print(json.dumps({k: g[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1", "reward")}, indent=1))
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
    p = argparse.ArgumentParser(prog="auditgym", description="Reproducible RL environment for AI financial auditors.")
    p.add_argument("--version", action="version", version=f"auditgym {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def source(sp, default="dev"):
        sp.add_argument("--split", default=default, help="split name(s) from configs/, comma separated (default: %(default)s)")
        sp.add_argument("--episodes", nargs="+", help="episode directories (or a directory of episodes) instead of a split")
        sp.add_argument("--seeds", help="seed list/ranges, e.g. 1-20,40 (closed runs: set AUDITGYM_SALT)")
        sp.add_argument("--config", help="config JSON for --seeds (a split file with 'episodes' is used as a split)")

    c = sub.add_parser("check", help="the single verification command (offline). Exit 0 = pass")
    c.add_argument("--quick", action="store_true", help="fewer fresh seeds (CI smoke)")
    c.add_argument("--json", action="store_true", help="print one JSON document")
    c.add_argument("--out", default=str(ROOT / "runs" / "check"))
    c.set_defaults(fn=cmd_check)

    g = sub.add_parser("generate", help="write episodes for split(s) or one seed")
    g.add_argument("--split", default="train,dev")
    g.add_argument("--seed", help="integer or string seed")
    g.add_argument("--config")
    g.add_argument("--out", default=str(ROOT / "data"))
    g.add_argument("--no-grader", action="store_true", help="do not write _grader/ (truth, provenance)")
    g.set_defaults(fn=cmd_generate)

    r = sub.add_parser("run", help="run policy(ies) on episodes; writes trajectories/results/summary")
    r.add_argument("--policy", required=True, help="random|reconcile|flag_all|expert|oracle|pkg.module:Class|'cmd:<command>' (comma separated)")
    source(r)
    r.add_argument("--mode", choices=["eval", "train"], default="eval")
    r.add_argument("--out", default=str(ROOT / "runs"))
    r.add_argument("--policy-arg", action="append", default=[], help="k=v passed to the policy constructor")
    r.add_argument("--policy-path", help="directory added to the policy child's sys.path")
    r.add_argument("--no-isolate", action="store_true", help="run builtin policies in-process (debug only)")
    r.add_argument("--trust-policies", action="store_true", help="allow custom policies on a user-writable install (your own code only)")
    r.add_argument("--public-salt", action="store_true", help="allow an eval run of non-open episodes without AUDITGYM_SALT (not a closed evaluation)")
    r.add_argument("--allow-forks", action="store_true", help="let the policy spawn processes/threads (weakens isolation; e.g. node)")
    r.add_argument("--format", choices=["auto", "md", "json", "csv"], default="auto")
    r.set_defaults(fn=cmd_run)

    cp = sub.add_parser("compare", help="side-by-side table of runs")
    cp.add_argument("runs", nargs="+")
    cp.add_argument("--split")
    cp.add_argument("--per-type", action="store_true")
    cp.add_argument("--format", choices=["auto", "md", "json", "csv"], default="auto")
    cp.set_defaults(fn=cmd_compare)

    gr = sub.add_parser("grade", help="grade a findings.json against an episode directory (prints the match trace)")
    gr.add_argument("--episode", required=True)
    gr.add_argument("--findings", required=True)
    gr.add_argument("--steps", type=int, default=0, help="inspection steps to charge")
    gr.set_defaults(fn=cmd_grade)

    rp = sub.add_parser("replay", help="re-execute a run's actions and diff observations/rewards")
    rp.add_argument("run")
    source(rp)
    rp.set_defaults(fn=cmd_replay)

    lk = sub.add_parser("leakscan", help="scan run trajectories for closed-truth leakage")
    lk.add_argument("runs", nargs="+")
    lk.set_defaults(fn=cmd_leakscan)

    pt = sub.add_parser("perturb", help="write a permuted + ID-renamed copy of an episode (paired checks)")
    pt.add_argument("--episode", required=True)
    pt.add_argument("--out", required=True)
    pt.add_argument("--seed", type=int, default=1)
    pt.add_argument("--no-rename", action="store_true")
    pt.add_argument("--no-shuffle", action="store_true")
    pt.set_defaults(fn=cmd_perturb)

    pl = sub.add_parser("play", help="play an episode by hand (train mode)")
    source(pl)
    pl.add_argument("--index", type=int, default=0)
    pl.set_defaults(fn=cmd_play)

    sc = sub.add_parser("schema", help="export JSON Schemas (findings, action, tools for LLMs)")
    sc.add_argument("--out", default=str(ROOT / "schema"))
    sc.set_defaults(fn=cmd_schema)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, KeyError, json.JSONDecodeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        if isinstance(e, PermissionError):
            print("hint: in Docker the image runs as user `auditor`; on Linux add --user \"$(id -u):$(id -g)\" for mounted "
                  "output dirs, and write generate/schema output under /tmp or a mounted dir", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
