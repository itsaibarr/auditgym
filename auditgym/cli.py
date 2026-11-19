"""auditgym command line. Plain text/markdown when piped; Rich tables in a terminal."""
from __future__ import annotations

import argparse
import cmd
import json
import sys
from pathlib import Path

from . import __version__
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
        out = []
        for d in a.episodes:
            p = Path(d)
            dirs = sorted(x for x in p.iterdir() if (x / "manifest.json").exists()) if not (p / "manifest.json").exists() else [p]
            out += [(p.name, dir_episode(x)) for x in dirs]
        return out
    if getattr(a, "seeds", None):
        cfg = json.loads(Path(a.config).read_text()) if a.config else {}
        cfg = cfg.get("base_config", cfg)
        return [("seeds", generate(s, cfg)) for s in parse_seeds(a.seeds)]
    out = []
    for s in a.split.split(","):
        out += split_episodes(s)
    return out


def cmd_generate(a) -> int:
    if a.seed is not None:
        cfg = json.loads(Path(a.config).read_text()) if a.config else {}
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
    from .runner import run_policy, summarize
    episodes = _episodes(a)
    rows = []
    for spec in a.policy.split(","):
        name = spec.replace(":", "_").replace("/", "_").replace(" ", "_")[:60]
        out = Path(a.out) / name
        res = run_policy(spec, episodes, out, mode=a.mode, args=a.policy_arg, policy_path=a.policy_path, isolate=not a.no_isolate,
                         meta={"sources": a.split if not (a.episodes or a.seeds) else (a.episodes or "seeds")})
        s = summarize(res)["all"]
        rows.append({"policy": spec, **s})
        print(f"{spec}: {len(res)} episodes -> {out}", file=sys.stderr)
    print_table(rows, f"AuditGym {a.mode} on {a.split if not (a.episodes or a.seeds) else 'custom episodes'}", a.format)
    return 0


def cmd_compare(a) -> int:
    rows = []
    for d in a.runs:
        s = json.loads((Path(d) / "summary.json").read_text())
        meta = json.loads((Path(d) / "run.json").read_text())
        rows.append({"policy": meta["policy"], **(s.get(f"split:{a.split}") if a.split else s["all"])})
    print_table(rows, "comparison", a.format)
    if a.per_type:
        for d in a.runs:
            s = json.loads((Path(d) / "summary.json").read_text())["all"]
            print(f"\n{Path(d).name} per type: " + ", ".join(f"{t}: tp={v['tp']} fp={v['fp']} fn={v['fn']}" for t, v in sorted(s["per_type"].items())))
    return 0


def cmd_grade(a) -> int:
    from .env import AuditEnv
    d = Path(a.episode)
    findings = json.loads(Path(a.findings).read_text())
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
    run = Path(a.run)
    meta = json.loads((run / "run.json").read_text())
    recs = [json.loads(l) for l in (run / "trajectories.jsonl").read_text().splitlines() if l]
    eps = {ep.episode_id: ep for _, ep in _episodes(a)}
    mismatches = 0
    by_ep: dict[str, list] = {}
    for r in recs:
        by_ep.setdefault(r["episode_id"], []).append(r)
    for eid, steps in by_ep.items():
        if eid not in eps:
            print(f"SKIP {eid}: episode not in the given source")
            continue
        env = AuditEnv(meta["mode"])
        obs, info = env.reset(options={"episode": eps[eid]})
        ok = dumps(obs) == dumps(steps[0]["observation"])
        for r in steps[1:]:
            if r["action"] is None:
                got = env.abort(r["info"].get("error_code") or "timeout")
            else:
                got = env.step(r["action"])
            ok &= dumps(list(got[:4])) == dumps([r["observation"], r["reward"], r["terminated"], r["truncated"]])
        mismatches += not ok
        print(f"{'OK ' if ok else 'DIFF'} {eid} ({len(steps) - 1} steps)")
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
        data = json.loads(Path(arg).read_text()) if arg and Path(arg).exists() else json.loads(arg or "null")
        return self._do({"name": "submit", "arguments": {"findings": data}})

    def do_raw(self, arg):
        return self._do(json.loads(arg))

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
        sp.add_argument("--config", help="config JSON for --seeds")

    c = sub.add_parser("check", help="the single verification command (offline). Exit 0 = pass")
    c.add_argument("--quick", action="store_true", help="fewer fresh seeds (CI smoke)")
    c.add_argument("--json", action="store_true", help="print one JSON document")
    c.add_argument("--out", default=str(ROOT / "runs" / "check"))
    c.set_defaults(fn=cmd_check)

    g = sub.add_parser("generate", help="write episodes for split(s) or one seed")
    g.add_argument("--split", default="train,dev")
    g.add_argument("--seed", type=int)
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
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
