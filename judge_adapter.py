"""Genesis FinTech jury adapter for AuditGym (interface: https://genesis-exe.com/fintech-checks.html, 20.11.2026).

Inside the container (WORKDIR /app, Python 3.11 with the locked dependencies on PATH):

    docker run --rm -i --network none --entrypoint python auditgym:1.0.0 -c "import judge_adapter; ..."

    generate_episode(seed, config, output_dir)  -> writes manifest.json, policy.json, ledger.csv, invoices.jsonl,
        payments.jsonl, fx.csv and the closed truth.json + provenance.json (top level of output_dir, section 03 names).
        `config` is AuditGym's documented generator config (README "Конфигурация генератора"), e.g.
        {"errors": {"fx_conversion": {"on": true, "count": [1, 1]}}, "interacting": {"X1": 1.0}}; {} = defaults.
        The secret salt is AUDITGYM_SALT when set (closed evaluation), else the public default.
    make_env() -> object with
        reset(seed, config) -> (observation, info)
            config={"mode": "closed", "episode_dir": dir}: loads exactly that episode (ids unchanged), reward is 0
            on every step including the last; config={"mode": "train", "team_config": {...}} or no episode_dir:
            generates from seed with team_config, the terminal reward is returned.
        step({"name": <action>, "args": {...}}) -> (observation, reward, terminated, truncated, info)
    score_episode(episode_dir, findings, inspection_steps) -> {tp, fp, fn, precision, recall, f1, return}
        the closed grader call; an invalid submit gives return -1 and None for the rest, the reason goes to stderr.

Tables: ledger, invoices, payments, fx (fx records have the derived key fx_id = FX-<date>-<SRC>-<TGT>). Record ids:
ledger.row_id, invoices/payments.doc_id. Error types: duplicate_invoice, payment_allocation, fx_conversion,
period_cutoff. Amount tolerance: manifest.amount_tolerance_minor (per issue: truth tolerance_minor; 0 if absent).
Observations: first {"task", "manifest", "policy", "remaining_steps"}; then {"result", "error", "remaining_steps"},
error = null or {"code", "message"} with a public code. Results: list_records -> {"records", "next_cursor"};
get_record / get_document -> the record / document object; calculator -> a decimal string; submit ->
{"accepted": true} or a structure error (the episode ends either way). info never holds closed data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from auditgym.env import AuditEnv
from auditgym.episode import Episode, fx_id, generate, load_public, write_episode
from auditgym.matcher import grade
from auditgym.schemas import parse_findings

TASK = ("Audit one company for one reporting period. Tables: ledger (key row_id), invoices and payments (key doc_id), "
        "fx (key fx_id). Actions: list_records{table, cursor, limit<=50}, get_record{table, id}, get_document{doc_id}, "
        "calculator{expression}, submit{findings}. Error types: duplicate_invoice, payment_allocation, fx_conversion, "
        "period_cutoff; the books may be clean. Amounts are integer minor units; read the policy for conversion, "
        "period and allocation rules.")
PRIVATE = ("truth.json", "provenance.json")


def _grader_files(d: Path) -> tuple[dict, dict]:
    """truth/provenance: top level (adapter episodes) or _grader/ (AuditGym's own episode folders)."""
    for base in (d, d / "_grader"):
        if (base / "truth.json").exists():
            prov = base / "provenance.json"
            return json.loads((base / "truth.json").read_text()), json.loads(prov.read_text()) if prov.exists() else {}
    raise FileNotFoundError(f"no truth.json in {d} or {d / '_grader'}")


def generate_episode(seed: int, config: dict, output_dir: str) -> None:
    ep = generate(seed, config or None)
    out = Path(output_dir)
    write_episode(ep, out, with_grader=False)
    (out / "truth.json").write_text(json.dumps(ep.truth, indent=2, ensure_ascii=False) + "\n")
    (out / "provenance.json").write_text(json.dumps(ep.provenance, indent=2, ensure_ascii=False) + "\n")


def _known_ids(pub: dict) -> set[str]:
    ids = {r["row_id"] for r in pub["ledger"]} | {r["doc_id"] for t in ("invoices", "payments") for r in pub[t]}
    return ids | {fx_id(r["date"], r["source_currency"], r["target_currency"]) for r in pub["fx"]}


def score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict:
    d = Path(episode_dir)
    pub = load_public(d)
    truth, _ = _grader_files(d)
    parsed, err = parse_findings(findings, pub["manifest"]["episode_id"])
    if err is not None:
        print(f"judge_adapter: invalid submit: {err}", file=sys.stderr)
        return {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None, "return": -1.0}
    g = grade(parsed, truth, inspection_steps, _known_ids(pub))
    return {"tp": g["tp"], "fp": g["fp"], "fn": g["fn"], "precision": g["precision"], "recall": g["recall"],
            "f1": g["f1"], "return": g["reward"]}


class JudgeEnv:
    """AuditEnv behind the jury's envelope. The grader state stays in this (jury-side) object; the policy only ever
    receives what reset/step return."""

    def __init__(self):
        self._env: AuditEnv | None = None
        self._closed = False

    def reset(self, seed=None, config=None):
        config = dict(config or {})
        self._closed = config.get("mode") == "closed"
        self._env = AuditEnv(mode="eval" if self._closed else "train")
        if config.get("episode_dir"):
            d = Path(config["episode_dir"])
            truth, prov = _grader_files(d)
            obs, _ = self._env.reset(options={"episode": Episode(public=load_public(d), truth=truth, provenance=prov, clean_public={})})
        elif self._closed:
            raise ValueError("closed mode needs config['episode_dir']")
        else:
            obs, _ = self._env.reset(seed=seed, config=config.get("team_config") or None)
        first = {"task": TASK, "manifest": obs["manifest"], "policy": obs["policy"], "remaining_steps": obs["steps_remaining"]}
        return first, {"remaining_steps": obs["steps_remaining"]}

    def step(self, action):
        if self._env is None:
            raise RuntimeError("call reset() first")
        if isinstance(action, dict) and set(action) <= {"name", "args"} and "name" in action:
            inner = {"name": action["name"], "arguments": action.get("args", {})}
        else:
            inner = action  # the env answers with a bounded invalid_action error and spends the step
        name = inner.get("name") if isinstance(inner, dict) else None
        obs, reward, term, trunc, info = self._env.step(inner)
        result, error = obs["result"], obs["error"]
        if error is None and result is not None:
            if name == "list_records":
                result = {"records": result["rows"], "next_cursor": result["next_cursor"]}
            elif name == "get_record":
                result = result["record"]
            elif name == "get_document":
                result = result["document"]
            elif name == "calculator":
                result = result["value"]
            elif name == "submit":
                g = self._env.result()
                if g is not None and not g["valid"]:  # bounded structure error, nothing about correctness
                    result, error = None, {"code": "invalid_findings", "message": "findings do not match the schema or limits"}
                else:
                    result = {"accepted": True}
        out = {"result": result, "error": error, "remaining_steps": obs["steps_remaining"]}
        info = {"remaining_steps": obs["steps_remaining"], "error_code": error["code"] if error else None}
        return out, (0.0 if self._closed else float(reward)), bool(term), bool(trunc), info


def make_env() -> JudgeEnv:
    return JudgeEnv()
