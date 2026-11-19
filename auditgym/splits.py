"""Versioned split configs -> episodes. configs/<split>.json lists seeds and per-episode overrides."""
from __future__ import annotations

import json
from pathlib import Path

from .episode import Episode, generate, load_grader, load_public

ROOT = Path(__file__).resolve().parent.parent


def load_split_config(split_or_path: str) -> dict:
    p = Path(split_or_path)
    if not p.suffix:
        p = ROOT / "configs" / f"{split_or_path}.json"
    return json.loads(p.read_text())


def split_episodes(split_or_path: str) -> list[tuple[str, Episode]]:
    cfg = load_split_config(split_or_path)
    name = cfg.get("split", Path(split_or_path).stem)
    return [(name, generate(e["seed"], {**cfg.get("base_config", {}), **e.get("config", {})})) for e in cfg["episodes"]]


def dir_episode(d: Path) -> Episode:
    truth, prov = load_grader(d)
    return Episode(load_public(d), truth, prov, {})


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out
