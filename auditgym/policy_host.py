"""Child-process host for Python policies: `python -I -m auditgym.policy_host <spec> [k=v ...]`.

Speaks JSON lines: reads {"type": "reset"|"step"|"close", ...} from stdin, writes one action per line.
Resource limits are applied before the policy module is imported.
"""
from __future__ import annotations

import importlib
import json
import sys


def _limits() -> None:
    try:
        import resource
        for lim, val in ((resource.RLIMIT_CPU, 120), (resource.RLIMIT_FSIZE, 0), (resource.RLIMIT_NOFILE, 256)):
            try:
                resource.setrlimit(lim, (val, val))
            except (ValueError, OSError):
                pass
        try:
            resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
        except (ValueError, OSError):
            pass
    except ImportError:
        pass


def load_policy(spec: str, kwargs: dict):
    from auditgym.policies import BUILTIN
    if spec in BUILTIN:
        return BUILTIN[spec](**kwargs)
    mod, _, cls = spec.partition(":")
    if not cls:
        raise SystemExit(f"policy spec must be a builtin name or module:Class, got {spec!r}")
    return getattr(importlib.import_module(mod), cls)(**kwargs)


def _coerce(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v


def main(argv: list[str]) -> None:
    spec, rest = argv[0], argv[1:]
    paths = [a.split("=", 1)[1] for a in rest if a.startswith("--path=")]
    kwargs = {k: _coerce(v) for k, v in (a.split("=", 1) for a in rest if not a.startswith("--path="))}
    sys.path[:0] = paths
    _limits()
    policy = load_policy(spec, kwargs)
    out = sys.stdout
    for line in sys.stdin:
        msg = json.loads(line)
        if msg["type"] == "close":
            break
        if msg["type"] == "reset":
            action = policy.reset(msg["observation"], msg["info"])
        else:
            action = policy.step(msg["observation"], msg["reward"], msg["terminated"], msg["truncated"], msg["info"])
        out.write(json.dumps(action, separators=(",", ":")) + "\n")
        out.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
