"""Versioned split configs -> episodes. configs/<split>.json lists seeds and per-episode overrides."""
from __future__ import annotations

import json
from pathlib import Path

from .episode import DEFAULT_SALT, Episode, generate, load_grader, load_public

ROOT = Path(__file__).resolve().parent.parent


def load_split_config(split_or_path: str) -> dict:
    p = Path(split_or_path)
    if not p.suffix:
        p = ROOT / "configs" / f"{split_or_path}.json"
    return json.loads(p.read_text())


OPEN_SPLITS = ("train", "dev")


def split_episodes(split_or_path: str, cfg: dict | None = None) -> list[tuple[str, Episode]]:
    """Open splits (configs/train.json, configs/dev.json) always use the published salt so they match data/;
    any other split (a jury's closed split file, or a split dict read from stdin) uses AUDITGYM_SALT when set."""
    if cfg is None:
        cfg = load_split_config(split_or_path)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("episodes"), list) or not cfg["episodes"] \
            or not all(isinstance(e, dict) and "seed" in e for e in cfg["episodes"]):
        raise ValueError(f"split {split_or_path!r} must be an object with a non-empty 'episodes' list of objects with a 'seed'")
    for e in cfg["episodes"]:
        if isinstance(e["seed"], bool) or not isinstance(e["seed"], (int, str)):
            raise ValueError(f"seed must be an integer or a string, got {e['seed']!r}")
        if not isinstance(e.get("config", {}), dict):
            raise ValueError("episode 'config' must be an object")
    if not isinstance(cfg.get("base_config", {}), dict):
        raise ValueError("'base_config' must be an object")
    keys = [json.dumps([e["seed"], e.get("config", {})], sort_keys=True) for e in cfg["episodes"]]
    if len(set(keys)) != len(keys):
        raise ValueError("split lists the same (seed, config) twice")
    name = cfg.get("split", Path(split_or_path).stem if split_or_path != "-" else "closed")
    salt = DEFAULT_SALT if split_or_path in OPEN_SPLITS else None
    out = []
    for pos, e in enumerate(cfg["episodes"]):
        try:
            ep = generate(e["seed"], {**cfg.get("base_config", {}), **e.get("config", {})}, salt=salt,
                          config_version=cfg.get("config_version"))
        except ValueError as exc:
            raise ValueError(f"split entry {pos} (seed {e['seed']!r}): {exc}") from None
        ep.split_pos = pos  # shown by datacheck/paired so rows map back to the split file
        out.append((name, ep))
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
