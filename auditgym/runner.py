"""Policy runner. The runner owns generation, truth and grading. In eval mode every non-oracle policy runs
as a child process (JSON lines over stdin/stdout) with a scrubbed environment (no AUDITGYM_SALT), an empty
temporary cwd, rlimits, per-step and per-episode deadlines and a process-group kill. The oracle is
privileged and runs in-process, still acting only through env.step(). Results and match traces are
written only after all policies have exited.
"""
from __future__ import annotations

import json
import os
import platform
import selectors
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .canon import dumps
from .env import AuditEnv
from .episode import Episode
from .policies import BUILTIN, OraclePolicy

STEP_TIMEOUT_S = 30.0
EPISODE_TIMEOUT_S = 60.0
MAX_LINE = 2_000_000


@dataclass
class EpisodeRun:
    result: dict
    trajectory: list[dict] = field(default_factory=list)


class ChildPolicy:
    """Talks to a policy process over JSON lines."""

    def __init__(self, spec: str, args: list[str] | None = None, policy_path: str | None = None):
        self.tmp = tempfile.TemporaryDirectory(prefix="auditgym-policy-")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED": "0", "HOME": self.tmp.name, "LANG": "C.UTF-8"}
        if spec.startswith("cmd:"):
            argv = shlex.split(spec[4:])
        else:
            argv = [sys.executable, "-I", "-m", "auditgym.policy_host", spec, *(args or [])]
            if policy_path:
                argv.append(f"--path={os.path.abspath(policy_path)}")
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     cwd=self.tmp.name, env=env, start_new_session=True)
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.proc.stdout, selectors.EVENT_READ)
        self.buf = b""

    def ask(self, msg: dict, deadline: float):
        """Send a message and read one action line. Returns (action, status) with status ok|timeout|crash|invalid."""
        try:
            self.proc.stdin.write((json.dumps(msg, separators=(",", ":")) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return None, "crash"
        while b"\n" not in self.buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, "timeout"
            if not self.sel.select(timeout=remaining):
                return None, "timeout"
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                return None, "crash"
            self.buf += chunk
            if len(self.buf) > MAX_LINE:
                self.buf = b""
                return "<oversized line>", "invalid"
        line, _, self.buf = self.buf.partition(b"\n")
        try:
            return json.loads(line), "ok"
        except ValueError:
            return line.decode("utf-8", "replace")[:200], "invalid"

    def close(self) -> None:
        try:
            self.proc.stdin.write(b'{"type":"close"}\n')
            self.proc.stdin.flush()
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
        self.tmp.cleanup()


class InProcess:
    def __init__(self, policy):
        self.p = policy

    def ask(self, msg, deadline):
        if msg["type"] == "reset":
            return self.p.reset(msg["observation"], msg["info"]), "ok"
        return self.p.step(msg["observation"], msg["reward"], msg["terminated"], msg["truncated"], msg["info"]), "ok"

    def close(self):
        pass


def make_policy(spec: str, ep: Episode, mode: str, args: list[str] | None = None, policy_path: str | None = None, isolate: bool = True):
    if spec == "oracle":
        return InProcess(OraclePolicy(ep.truth))
    if not isolate and spec in BUILTIN:
        return InProcess(BUILTIN[spec]())
    return ChildPolicy(spec, args, policy_path)


def run_episode(spec: str, ep: Episode, mode: str = "eval", args=None, policy_path=None, isolate: bool = True,
                episode_timeout: float = EPISODE_TIMEOUT_S, step_timeout: float = STEP_TIMEOUT_S) -> EpisodeRun:
    env = AuditEnv(mode, time_limit_s=episode_timeout)
    t0 = time.monotonic()
    pol = make_policy(spec, ep, mode, args, policy_path, isolate)
    traj = []
    obs, info = env.reset(options={"episode": ep})
    eid = ep.episode_id
    traj.append({"episode_id": eid, "step": 0, "action": None, "observation": obs, "reward": 0.0, "terminated": False, "truncated": False, "info": info})
    msg = {"type": "reset", "observation": obs, "info": info}
    termination = None
    try:
        while True:
            deadline = min(time.monotonic() + step_timeout, t0 + episode_timeout)
            action, status = pol.ask(msg, deadline)
            if status in ("timeout", "crash"):
                termination = "timeout" if status == "timeout" else "policy_crash"
                obs, r, term, trunc, info = env.abort(termination)
                traj.append({"episode_id": eid, "step": env.steps_used, "action": None, "observation": obs, "reward": r,
                             "terminated": term, "truncated": trunc, "info": info})
                break
            obs, r, term, trunc, info = env.step(action)
            traj.append({"episode_id": eid, "step": env.steps_used, "action": action, "observation": obs, "reward": r,
                         "terminated": term, "truncated": trunc, "info": info})
            if term or trunc:
                if term:
                    termination = "submitted" if env.result()["valid"] else "schema_error"
                else:
                    termination = info.get("error_code") or "max_steps"
                break
            msg = {"type": "step", "observation": obs, "reward": r, "terminated": term, "truncated": trunc, "info": info}
    finally:
        pol.close()
    g = env.result()
    result = {
        "episode_id": eid, "policy": spec, "mode": mode, "termination": termination, "steps": env.steps_used,
        "wall_s": round(time.monotonic() - t0, 3), "n_truth": len(ep.truth["issues"]), "clean": not ep.truth["issues"],
        **{k: g[k] for k in ("valid", "tp", "fp", "fn", "precision", "recall", "f1", "reward", "reward_experimental", "inspection_steps", "per_type", "matches")},
    }
    return EpisodeRun(result, traj)


def run_policy(spec: str, episodes: list[tuple[str, Episode]], out: Path, mode: str = "eval", args=None, policy_path=None,
               isolate: bool = True, meta: dict | None = None) -> list[dict]:
    """Run one policy over episodes; write run.json, trajectories.jsonl, results.jsonl, summary.json/.md at the end."""
    runs = []
    for split, ep in episodes:
        er = run_episode(spec, ep, mode, args, policy_path, isolate)
        er.result["split"] = split
        runs.append(er)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trajectories.jsonl", "w") as f:
        for er in runs:
            for t in er.trajectory:
                f.write(dumps(t) + "\n")
    results = [er.result for er in runs]
    with open(out / "results.jsonl", "w") as f:
        for r in results:
            f.write(dumps(r) + "\n")
    summary = summarize(results)
    (out / "run.json").write_text(json.dumps({
        "auditgym_version": __version__, "policy": spec, "policy_args": args or [], "mode": mode,
        "python": platform.python_version(), "platform": platform.platform(), "episodes": [ep.episode_id for _, ep in episodes],
        "config_hashes": sorted({ep.provenance["config_hash"] for _, ep in episodes}), "salt": "custom" if os.environ.get("AUDITGYM_SALT") else "default-open",
        **(meta or {}),
    }, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return results


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
