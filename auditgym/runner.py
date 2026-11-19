"""Policy runner. The runner owns generation, truth and grading.

Isolation in eval mode: every non-oracle policy is a child process speaking JSON lines, started through
auditgym.policy_host with a scrubbed environment (no AUDITGYM_SALT), an empty temporary cwd, rlimits incl.
RLIMIT_NPROC=1 (no forking), per-step and per-episode deadlines on both reads and writes, and a process-group
kill. On Linux the runner makes itself non-dumpable (PR_SET_DUMPABLE=0) so a same-user child cannot read its
/proc/<ppid>/environ (the salt). The oracle is privileged and runs in-process, last, acting only through
env.step(). All run outputs (trajectories, results, match traces) are written only after every policy of the
invocation has exited.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import platform
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .canon import dumps, loads_strict
from .env import AuditEnv
from .episode import Episode
from .policies import BUILTIN, OraclePolicy

STEP_TIMEOUT_S = 30.0
EPISODE_TIMEOUT_S = 60.0
MAX_LINE = 512_000  # the largest schema-valid submission is ~225 KB
_PKG = Path(__file__).resolve().parent


def harden_runner() -> None:
    """Linux: make the runner non-dumpable so same-user children cannot read /proc/<runner>/environ|mem."""
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 4
        except Exception:
            pass


def salt_exposures() -> list[int]:
    """Linux: pids of *other* processes whose environment or command line we (and so a same-user policy) can read
    and that contain AUDITGYM_SALT or its value, e.g. `docker run --init` (tini as PID 1), an `sh -c 'AUDITGYM_SALT=..'`
    wrapper or `uv run` as the parent."""
    salt = os.environ.get("AUDITGYM_SALT")
    if not sys.platform.startswith("linux") or not salt:
        return []
    needles = (b"AUDITGYM_SALT=", salt.encode())
    hits = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        for f in ("environ", "cmdline"):
            try:
                data = (d / f).read_bytes()
            except OSError:
                continue
            if any(n in data for n in needles):
                hits.append(int(d.name))
                break
    return hits


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(_PKG.rglob("*.py")):
        h.update(p.relative_to(_PKG).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def parse_action_line(line: bytes):
    """Returns (action, status): ok | invalid. Never raises."""
    try:
        return loads_strict(line), "ok"
    except (ValueError, RecursionError, UnicodeDecodeError):
        return line[:200].decode("utf-8", "replace"), "invalid"


@dataclass
class EpisodeRun:
    result: dict
    trajectory: list[dict] = field(default_factory=list)


class ChildPolicy:
    """Talks to a policy process over JSON lines with deadline-bounded reads and writes."""

    def __init__(self, spec: str, args: list[str] | None = None, policy_path: str | None = None, allow_forks: bool = False):
        harden_runner()
        self.tmp = tempfile.TemporaryDirectory(prefix="auditgym-policy-")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED": "0", "HOME": self.tmp.name, "LANG": "C.UTF-8",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        host = [sys.executable, "-I", "-m", "auditgym.policy_host"] + (["--allow-forks"] if allow_forks else [])
        if spec.startswith("cmd:"):
            argv = host + ["--exec", "--"] + shlex.split(spec[4:])
        else:
            argv = host + [spec, *(args or [])]
            if policy_path:
                argv.append(f"--path={os.path.abspath(policy_path)}")
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     cwd=self.tmp.name, env=env, start_new_session=True)
        os.set_blocking(self.proc.stdin.fileno(), False)
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.proc.stdout, selectors.EVENT_READ)
        self.buf = b""

    def _write(self, data: bytes, deadline: float) -> bool:
        fd = self.proc.stdin.fileno()
        view = memoryview(data)
        with selectors.DefaultSelector() as ws:
            ws.register(fd, selectors.EVENT_WRITE)
            while view:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not ws.select(timeout=remaining):
                    return False
                try:
                    n = os.write(fd, view)
                except BlockingIOError:
                    continue
                except OSError as e:
                    if e.errno in (errno.EPIPE, errno.EBADF):
                        raise BrokenPipeError from e
                    raise
                view = view[n:]
        return True

    def ask(self, msg: dict, deadline: float):
        """Send a message and read one action line. Returns (action, status) with status ok|timeout|crash|invalid."""
        try:
            if not self._write((json.dumps(msg, separators=(",", ":"), sort_keys=True) + "\n").encode(), deadline):
                return None, "timeout"
        except (BrokenPipeError, OSError):
            return None, "crash"
        while b"\n" not in self.buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.sel.select(timeout=remaining):
                return None, "timeout"
            try:
                chunk = os.read(self.proc.stdout.fileno(), 65536)
            except OSError:
                return None, "crash"
            if not chunk:
                return None, "crash"
            self.buf += chunk
            if len(self.buf) > MAX_LINE and b"\n" not in self.buf:
                self.buf = b""
                return "<oversized line>", "invalid"
        line, _, self.buf = self.buf.partition(b"\n")
        return parse_action_line(line)

    def close(self, final: dict | None = None) -> None:
        msg = {"type": "close", **(final or {})}
        try:
            self._write((json.dumps(msg, separators=(",", ":")) + "\n").encode(), time.monotonic() + 2)
        except (BrokenPipeError, OSError):
            pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self.proc.wait()
        self.sel.close()
        for f in (self.proc.stdin, self.proc.stdout):
            try:
                f.close()
            except OSError:
                pass
        self.tmp.cleanup()


class InProcess:
    def __init__(self, policy):
        self.p = policy

    def ask(self, msg, deadline):
        if msg["type"] == "reset":
            return self.p.reset(msg["observation"], msg["info"]), "ok"
        return self.p.step(msg["observation"], msg["reward"], msg["terminated"], msg["truncated"], msg["info"]), "ok"

    def close(self, final=None):
        if final and hasattr(self.p, "end"):
            self.p.end(final["observation"], final["reward"], final["terminated"], final["truncated"], final["info"])


def make_policy(spec: str, ep: Episode, args=None, policy_path=None, isolate: bool = True, allow_forks: bool = False):
    if spec == "oracle":
        return InProcess(OraclePolicy(ep.truth))
    if not isolate and spec in BUILTIN:
        return InProcess(BUILTIN[spec]())
    return ChildPolicy(spec, args, policy_path, allow_forks)


def run_episode(spec: str, ep: Episode, mode: str = "eval", args=None, policy_path=None, isolate: bool = True,
                episode_timeout: float = EPISODE_TIMEOUT_S, step_timeout: float = STEP_TIMEOUT_S, allow_forks: bool = False) -> EpisodeRun:
    env = AuditEnv(mode, time_limit_s=episode_timeout)
    t0 = time.monotonic()
    pol = make_policy(spec, ep, args, policy_path, isolate, allow_forks)
    traj = []
    eid = ep.episode_id
    final = None
    termination = None

    def log(action, obs, r, term, trunc, info):
        traj.append({"episode_id": eid, "policy": spec, "step": env.steps_used, "action": action, "observation": obs, "reward": r,
                     "terminated": term, "truncated": trunc, "info": info})

    try:
        obs, info = env.reset(options={"episode": ep})
        log(None, obs, 0.0, False, False, info)
        msg = {"type": "reset", "observation": obs, "info": info}
        while True:
            deadline = min(time.monotonic() + step_timeout, t0 + episode_timeout)
            action, status = pol.ask(msg, deadline)
            if status in ("timeout", "crash"):
                termination = "timeout" if status == "timeout" else "policy_crash"
                obs, r, term, trunc, info = env.abort(termination)
                log(None, obs, r, term, trunc, info)
                break
            obs, r, term, trunc, info = env.step(action)
            log(action, obs, r, term, trunc, info)
            if term or trunc:
                termination = ("submitted" if env.result()["valid"] else "schema_error") if term else (info.get("error_code") or "max_steps")
                final = {"observation": obs, "reward": r, "terminated": term, "truncated": trunc, "info": info}
                break
            msg = {"type": "step", "observation": obs, "reward": r, "terminated": term, "truncated": trunc, "info": info}
    finally:
        pol.close(final)
    g = env.result()
    result = {
        "episode_id": eid, "policy": spec, "mode": mode, "termination": termination, "steps": env.steps_used,
        "wall_s": round(time.monotonic() - t0, 3), "n_truth": len(ep.truth["issues"]), "clean": not ep.truth["issues"],
        **{k: g[k] for k in ("valid", "tp", "fp", "fn", "precision", "recall", "f1", "reward", "reward_experimental", "inspection_steps", "per_type", "matches")},
    }
    return EpisodeRun(result, traj)


def run_policies(specs: list[str], episodes: list[tuple[str, Episode]], out_root: Path, mode: str = "eval", args=None,
                 policy_path=None, isolate: bool = True, allow_forks: bool = False, meta: dict | None = None,
                 dir_name=None) -> dict[str, list[dict]]:
    """Run several policies (oracle last); write every output only after all of them have finished."""
    order = sorted(specs, key=lambda s: s == "oracle")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if isolate and not allow_forks and hasattr(os, "geteuid") and os.geteuid() == 0:
        print("warning: running as root: RLIMIT_NPROC is not enforced for root, so policies can fork. "
              "Run as an unprivileged user (the Docker image does).", file=sys.stderr)
    exposed = salt_exposures() if isolate else []
    if exposed:
        raise RuntimeError(f"AUDITGYM_SALT is readable in the environment or command line of other process(es) {exposed[:5]} (e.g. docker --init, "
                           "an sh -c / uv run wrapper); a policy could read it. Start the runner directly: "
                           "docker run --rm -e AUDITGYM_SALT=... auditgym:1.0.0 run ...")
    done: dict[str, list[EpisodeRun]] = {}
    for spec in order:
        runs = []
        for split, ep in episodes:
            er = run_episode(spec, ep, mode, args, policy_path, isolate, allow_forks=allow_forks)
            er.result["split"] = split
            runs.append(er)
        done[spec] = runs
    # Every child is dead now. A child could still have planted symlinks/dirs under out_root (it may know the
    # path from our argv), so never write through existing paths: build each run dir fresh, then swap it in.
    if out_root.is_symlink() or not out_root.is_dir():
        raise RuntimeError(f"output root {out_root} was replaced during the run; refusing to write")
    out = {}
    for spec in specs:
        d = out_root / (dir_name(spec) if dir_name else spec)
        if d.parent != out_root or d.name in ("", ".", "..") or d.name.startswith("."):
            raise ValueError(f"invalid run dir name {d.name!r}")
        tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=out_root))
        tmp.chmod(0o755)
        _write_run(spec, done[spec], episodes, tmp, mode, args, meta, dir_label=d.name)
        if os.path.lexists(d):  # whatever is there (dir, file, symlink, fifo, unreadable dir): move aside, never follow
            aside = Path(tempfile.mktemp(prefix=".old-", dir=out_root))
            os.replace(d, aside)
            if aside.is_dir() and not aside.is_symlink():
                shutil.rmtree(aside, ignore_errors=True)
            else:
                aside.unlink(missing_ok=True)
        os.replace(tmp, d)
        out[spec] = [er.result for er in done[spec]]
    return out


def run_policy(spec: str, episodes, out: Path, mode: str = "eval", args=None, policy_path=None, isolate: bool = True,
               meta: dict | None = None, allow_forks: bool = False) -> list[dict]:
    return run_policies([spec], episodes, out.parent, mode, args, policy_path, isolate, allow_forks, meta, dir_name=lambda s: out.name)[spec]


RUN_FILES = ("trajectories.jsonl", "results.jsonl", "summary.json")


def _file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_run(d: Path) -> list[str]:
    """Integrity of a run dir written by run_policies: regular single-link files whose hashes, dir name and policy
    match run.json. A policy cannot write bytes, so it cannot forge a matching run.json."""
    d = Path(d)
    if d.is_symlink() or not d.is_dir():
        return [f"{d}: not a regular directory"]
    probs = []
    for n in ("run.json", *RUN_FILES):
        p = d / n
        if p.is_symlink() or not p.is_file():
            probs.append(f"{p}: missing or not a regular file")
        elif p.stat().st_nlink != 1:
            probs.append(f"{p}: has extra hard links")
    if probs:
        return probs
    meta = json.loads((d / "run.json").read_text())
    if meta.get("dir") != d.name:
        probs.append(f"{d}: run.json belongs to run dir {meta.get('dir')!r}")
    for n, h in (meta.get("files") or {}).items():
        if n not in RUN_FILES or _file_sha(d / n) != h:
            probs.append(f"{d / n}: hash does not match run.json")
    if set(meta.get("files") or {}) != set(RUN_FILES):
        probs.append(f"{d}: run.json does not list the run files")
    return probs


def _write_run(spec, runs, episodes, out: Path, mode, args, meta, dir_label: str | None = None) -> None:
    closed_oracle = spec == "oracle" and bool(os.environ.get("AUDITGYM_SALT"))
    with open(out / "trajectories.jsonl", "w", encoding="utf-8", errors="backslashreplace") as f:
        for er in runs:
            for t in er.trajectory:
                if closed_oracle:  # the oracle's submit IS the closed truth: never persist it
                    t = {**t, "action": None if t["action"] is None else {"name": t["action"]["name"], "arguments": "<withheld: closed run>"}}
                f.write(dumps(t) + "\n")
    results = [er.result for er in runs]
    with open(out / "results.jsonl", "w", encoding="utf-8", errors="backslashreplace") as f:
        for r in results:
            f.write(dumps(r) + "\n")
    (out / "summary.json").write_text(json.dumps(summarize(results), indent=2) + "\n")
    run_meta = {
        "auditgym_version": __version__, "code_hash": code_hash(), "policy": spec, "policy_args": args or [], "mode": mode,
        "python": platform.python_version(), "platform": platform.platform(),
        "episodes": [ep.episode_id for _, ep in episodes], "splits": sorted({s for s, _ in episodes}),
        "config_hashes": sorted({ep.provenance["config_hash"] for _, ep in episodes}),
        "config_versions": sorted({str(ep.provenance.get("config_version")) for _, ep in episodes}),
        "salt": "custom" if os.environ.get("AUDITGYM_SALT") else "default-open", **(meta or {}),
    }
    run_meta["dir"] = dir_label or out.name
    run_meta["files"] = {n: _file_sha(out / n) for n in RUN_FILES}
    (out / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n")


def summarize(results: list[dict]) -> dict:
    def agg(rs):
        if not rs:
            return None
        n = len(rs)
        tp, fp, fn = (sum(r[k] for r in rs) for k in ("tp", "fp", "fn"))
        clean = [r for r in rs if r["clean"]]
        per_type: dict[str, dict] = {}
        for r in rs:
            for t, d in r["per_type"].items():
                acc = per_type.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})
                for k in acc:
                    acc[k] += d[k]
        term: dict[str, int] = {}
        for r in rs:
            term[r["termination"]] = term.get(r["termination"], 0) + 1
        return {
            "episodes": n,
            "micro_precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "micro_recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
            "mean_f1": round(sum(r["f1"] for r in rs) / n, 4),
            "mean_return": round(sum(r["reward"] for r in rs) / n, 4),
            "mean_return_experimental": round(sum(r["reward_experimental"] for r in rs) / n, 4),
            "mean_steps": round(sum(r["steps"] for r in rs) / n, 2),
            "mean_wall_s": round(sum(r["wall_s"] for r in rs) / n, 3),
            "clean_episodes": len(clean),
            "clean_fp_rate": round(sum(1 for r in clean if r["fp"] > 0) / len(clean), 4) if clean else None,
            "tp": tp, "fp": fp, "fn": fn, "per_type": per_type, "termination": term,
        }
    splits = sorted({r.get("split", "-") for r in results})
    return {"all": agg(results), **{f"split:{s}": agg([r for r in results if r.get("split", "-") == s]) for s in splits}}
