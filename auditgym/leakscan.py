"""Scan trajectories for closed-truth leakage: grader tokens in observations/info, eval-mode reward or
diagnostics, tracebacks and filesystem paths."""
from __future__ import annotations

import json
import re
from pathlib import Path

PATTERNS = [
    (re.compile(r"\bI-\d{2}\b"), "truth issue id"),
    (re.compile(r"truth|provenance|_grader|expected_amount|evidence_sets|injector", re.I), "grader vocabulary"),
    (re.compile(r"\b(E[1-4]|X[12])\.[a-z_]+"), "injector name"),
    (re.compile(r"Traceback|File \""), "traceback"),
    (re.compile(r"(/Users/|/home/|/app/|/tmp/|[A-Z]:\\\\)"), "filesystem path"),
    (re.compile(r"auditgym-open-splits|AUDITGYM_SALT", re.I), "salt"),
]
EVAL_INFO_KEYS = {"steps_remaining", "error_code"}


def scan_records(records: list[dict], mode: str) -> list[str]:
    problems = []
    for rec in records:
        where = f"{rec.get('episode_id')}#{rec.get('step')}"
        blob = json.dumps({"observation": rec.get("observation"), "info": rec.get("info")}, ensure_ascii=False)
        for pat, what in PATTERNS:
            mm = pat.search(blob)
            if mm:
                problems.append(f"{where}: {what}: {mm.group(0)!r}")
        if mode == "eval":
            if rec.get("reward") not in (0, 0.0):
                problems.append(f"{where}: non-zero reward in eval mode")
            if set(rec.get("info") or {}) - EVAL_INFO_KEYS:
                problems.append(f"{where}: extra info keys in eval mode: {sorted(set(rec['info']) - EVAL_INFO_KEYS)}")
        elif rec.get("step", 0) and not (rec.get("terminated") or rec.get("truncated")) and rec.get("reward") not in (0, 0.0):
            problems.append(f"{where}: non-zero intermediate reward")
    return problems


def scan_run(run_dir: Path) -> list[str]:
    meta = json.loads((run_dir / "run.json").read_text())
    recs = [json.loads(l) for l in (run_dir / "trajectories.jsonl").read_text().splitlines() if l]
    return scan_records(recs, meta.get("mode", "eval"))
