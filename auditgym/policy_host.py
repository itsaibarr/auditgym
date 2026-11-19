"""Child-process host for policies.

    python -I -m auditgym.policy_host <module:Class|builtin> [k=v ...] [--path=DIR] [--allow-forks]
    python -I -m auditgym.policy_host --exec [--allow-forks] -- <program> [args...]

Resource limits are applied *before* the policy is imported (or the program is exec'd): CPU time, no file
writes, few file descriptors, address space, and RLIMIT_NPROC=1 so the policy cannot fork helper processes
(e.g. `ps`, or a daemon that survives the process-group kill). Speaks JSON lines: reads
{"type": "reset"|"step"|"close", ...} from stdin and writes one action per line. The "close" message carries the
final transition (terminal reward in train mode); policies may implement end(observation, reward, terminated,
truncated, info).
"""
from __future__ import annotations

import importlib
import json
import os
import sys

try:  # imported before any policy path is added, so a policy file cannot shadow it
    import resource
except ImportError:  # non-POSIX
    resource = None


def apply_limits(allow_forks: bool = False) -> None:
    if resource is None:
        return
    limits = [(resource.RLIMIT_CPU, 120), (resource.RLIMIT_FSIZE, 0), (resource.RLIMIT_NOFILE, 64)]
    if not allow_forks and hasattr(resource, "RLIMIT_NPROC"):
        limits.append((resource.RLIMIT_NPROC, 1))
    for lim, val in limits:
        try:
            resource.setrlimit(lim, (val, val))
        except (ValueError, OSError):
            pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    except (ValueError, OSError, AttributeError):
        pass


def load_policy(spec: str, kwargs: dict):
    from auditgym.policies import BUILTIN
    if spec in BUILTIN:  # --policy-arg is shared by every policy of a run: builtins take only the args they know
        import inspect
        params = inspect.signature(BUILTIN[spec]).parameters
        return BUILTIN[spec](**{k: v for k, v in kwargs.items() if k in params})
    mod, _, cls = spec.partition(":")
    if not cls:
        raise SystemExit(f"policy spec must be a builtin name or module:Class, got {spec!r}")
    m = importlib.import_module(mod)
    if not hasattr(m, cls):
        raise SystemExit(f"{mod!r} resolved to {getattr(m, '__file__', m)!r}, which has no {cls!r}: if your module shares a "
                         "name with a standard-library module, rename it")
    return getattr(m, cls)(**kwargs)


def _coerce(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v


def main(argv: list[str]) -> None:
    allow = "--allow-forks" in argv
    argv = [a for a in argv if a != "--allow-forks"]
    if argv and argv[0] == "--exec":
        prog = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
        apply_limits(allow)
        os.execvp(prog[0], prog)
    spec, rest = argv[0], argv[1:]
    paths = [rest.pop()[len("--path="):]] if rest and rest[-1].startswith("--path=") else []  # runner appends it last
    if any(a.startswith("--") or "=" not in a for a in rest):
        raise SystemExit(f"policy args must be key=value, got {rest!r}")
    kwargs = {k: _coerce(v) for k, v in (a.split("=", 1) for a in rest)}
    apply_limits(allow)                   # limits first, while sys.path holds only the stdlib and auditgym
    import auditgym.policies  # noqa: F401  (pre-imports the stdlib modules builtins use)
    sys.path.extend(paths)                # appended, never prepended: policy files cannot shadow stdlib/auditgym
    out = os.fdopen(os.dup(1), "w", encoding="utf-8")  # private protocol channel
    os.dup2(2, 1)                         # the policy's own prints (Python or C level) go to stderr, never into the protocol
    sys.stdout = sys.stderr
    policy = load_policy(spec, kwargs)
    for line in sys.stdin:
        msg = json.loads(line)
        if msg["type"] == "close":
            if "observation" in msg and hasattr(policy, "end"):
                policy.end(msg["observation"], msg["reward"], msg["terminated"], msg["truncated"], msg["info"])
            break
        if msg["type"] == "reset":
            action = policy.reset(msg["observation"], msg["info"])
        else:
            action = policy.step(msg["observation"], msg["reward"], msg["terminated"], msg["truncated"], msg["info"])
        out.write(json.dumps(action, separators=(",", ":")) + "\n")
        out.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
