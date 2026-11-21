"""Versioned split configs -> episodes. configs/<split>.json lists seeds and per-episode overrides."""
from __future__ import annotations

import json
from pathlib import Path

from .episode import DEFAULT_SALT, Episode, generate, load_grader, load_public, make_config

ROOT = Path(__file__).resolve().parent.parent


def load_split_config(split_or_path: str) -> dict:
    p = Path(split_or_path)
    if not p.suffix:
        p = ROOT / "configs" / f"{split_or_path}.json"
    return json.loads(p.read_text())


OPEN_SPLITS = ("train", "dev")


SPLIT_KEYS = {"episodes", "base_config", "split", "config_version", "note"}
ENTRY_KEYS = {"seed", "config", "profile", "note"}


def split_episodes(split_or_path: str, cfg: dict | None = None) -> list[tuple[str, Episode]]:
    """Open splits (configs/train.json, configs/dev.json) always use the published salt so they match data/;
    any other split (a jury's closed split file, or a split dict read from stdin) uses AUDITGYM_SALT when set."""
    if cfg is None:
        cfg = load_split_config(split_or_path)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("episodes"), list) or not cfg["episodes"] \
            or not all(isinstance(e, dict) for e in cfg["episodes"]):
        raise ValueError(f"split {split_or_path!r} must be an object with a non-empty 'episodes' list of objects with a 'seed'")
    shape = [f"unknown split key {k!r} (known: {sorted(SPLIT_KEYS)})" for k in cfg if k not in SPLIT_KEYS]
    shape += [f"split entry {pos}: no 'seed'" for pos, e in enumerate(cfg["episodes"]) if "seed" not in e]
    shape += [f"split entry {pos}: unknown key {k!r} (known: {sorted(ENTRY_KEYS)})"
              for pos, e in enumerate(cfg["episodes"]) for k in e if k not in ENTRY_KEYS]
    if shape:  # report the file's shape problems together; the configs are checked once these are fixed
        raise ValueError(f"the split has {len(shape)} problem{'' if len(shape) == 1 else 's'}:\n  " + "\n  ".join(shape))
    bad_seeds = [f"split entry {pos}: seed must be an integer or a string, got {e['seed']!r}" for pos, e in enumerate(cfg["episodes"])
                 if isinstance(e["seed"], bool) or not isinstance(e["seed"], (int, str))]
    for pos, e in enumerate(cfg["episodes"]):
        if not isinstance(e.get("config", {}), dict):
            raise ValueError("episode 'config' must be an object")
    if not isinstance(cfg.get("base_config", {}), dict):
        raise ValueError("'base_config' must be an object")
    keys = [json.dumps([e["seed"], e.get("config", {})], sort_keys=True) for e in cfg["episodes"]]
    if len(set(keys)) != len(keys):
        first: dict = {}
        dups = [f"entries {first[k]} and {i}" for i, k in enumerate(keys) if k in first or first.setdefault(k, i) is None]
        dup_error = f"split lists the same (seed, config) twice: {'; '.join(dups)}"
    else:
        dup_error = None
    name = cfg.get("split", Path(split_or_path).stem if split_or_path != "-" else "closed")
    salt = DEFAULT_SALT if split_or_path in OPEN_SPLITS else None
    out, errors = [], bad_seeds + ([dup_error] if dup_error else [])
    base_problems: set = set()
    try:  # base_config problems are named once, not under every entry
        make_config(cfg.get("base_config", {}))
    except ValueError as exc:
        base_problems = set(str(exc).split("; "))
        errors.append(f"base_config: {exc}")

    def own(exc: ValueError) -> str:  # an entry's problems that the base_config line does not already name
        return "; ".join(p for p in str(exc).split("; ") if p not in base_problems)
    bad_pos = {i for i, e in enumerate(cfg["episodes"]) if isinstance(e["seed"], bool) or not isinstance(e["seed"], (int, str))}
    for pos, e in enumerate(cfg["episodes"]):
        if pos in bad_pos:  # its seed is already reported; still list its config problems in the same pass
            try:
                make_config({**cfg.get("base_config", {}), **e.get("config", {})})
            except ValueError as exc:
                if own(exc):
                    errors.append(f"split entry {pos}: {own(exc)}")
            continue
        try:
            ep = generate(e["seed"], {**cfg.get("base_config", {}), **e.get("config", {})}, salt=salt,
                          config_version=cfg.get("config_version"))
        except ValueError as exc:  # collect every bad entry, so one run of the command lists them all
            if own(exc):
                errors.append(f"split entry {pos} (seed {e['seed']!r}): {own(exc)}")
            continue
        ep.split_pos = pos  # shown by datacheck/paired so rows map back to the split file
        out.append((name, ep))
    if errors:
        raise ValueError(f"the split has {len(errors)} problem{'' if len(errors) == 1 else 's'}:\n  " + "\n  ".join(errors))
    ids = [ep.episode_id for _, ep in out]
    if len(set(ids)) != len(ids):
        raise ValueError("two split entries generate the same episode (e.g. seed 1 and \"1\", or an explicit default config)")
    return out


def dir_episode(d: Path) -> Episode:
    truth, prov = load_grader(d)
    return Episode(load_public(d), truth, prov, {})


def parse_seeds(s: str) -> list:
    """'1-20,40,jury-seed-1' -> ints for integer ranges/values, strings otherwise."""
    out: list = []
    for part in s.split(","):
        part = part.strip()
        a, sep, b = part.partition("-")
        if sep and a.isdigit() and b.isdigit():
            out += list(range(int(a), int(b) + 1))
        elif part.lstrip("-").isdigit():
            out.append(int(part))
        elif part:
            out.append(part)
    return out
