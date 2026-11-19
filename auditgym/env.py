"""AuditEnv: Gymnasium-style JSON contract (case §04). Plain class; see contrib/gymnasium_env.py for a gym wrapper.

reset(seed, config, options) -> (observation, info)
step(action) -> (observation, reward, terminated, truncated, info)

Action envelope: {"name": <action>, "arguments": {...}}. Observations contain only what was requested,
the result or a bounded error, and steps_remaining. Intermediate reward is always 0. In "train" mode the
terminal reward is returned; in "eval" mode every reward is 0 and the grade is kept for the runner
(env.result()), released only after the episode.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import __version__
from .calc import CalcError, evaluate
from .canon import dumps
from .episode import Episode, fx_id, generate, load_grader, load_public
from .matcher import grade
from .schemas import MAX_EVIDENCE, MAX_EXPLANATION, MAX_FINDINGS, TABLES, parse_findings

MAX_STEPS = 100
PAGE = 50
TIME_LIMIT_S = 60.0
PRIMARY_KEYS = {"ledger": "row_id", "invoices": "doc_id", "payments": "doc_id", "fx": "fx_id"}
ARGS = {
    "list_records": ({"table"}, {"cursor", "limit"}),
    "get_record": ({"table", "id"}, set()),
    "get_document": ({"doc_id"}, set()),
    "calculator": ({"expression"}, set()),
    "submit": ({"findings"}, set()),
}
_CURSOR = re.compile(r"(ledger|invoices|payments|fx):(0|[1-9][0-9]{0,6}):([0-9a-f]{8})")
TASK = ("You are auditing one company for one reporting period. Inspect the ledger, invoices, payments and FX "
        "rates through the actions, then submit findings with evidence. Error types: duplicate_invoice, "
        "payment_allocation, fx_conversion, period_cutoff. The books may also be clean. Amounts are integer "
        "minor units of the base currency. Read the policy for conversion, period and allocation rules.")


def _copy(x):
    return json.loads(json.dumps(x))


def _echo(x) -> str:
    """Echo only the policy's own (scalar) input, bounded; never serialise arbitrary structures."""
    if isinstance(x, str):
        return x[:64]
    if isinstance(x, (int, float, bool)) or x is None:
        return str(x)[:64]
    return f"<{type(x).__name__}>"


class AuditEnv:
    metadata = {"name": "AuditGym-v1", "version": __version__}

    def __init__(self, mode: str = "train", max_steps: int = MAX_STEPS, time_limit_s: float | None = TIME_LIMIT_S, clock=time.monotonic):
        """time_limit_s: wall-clock episode limit (case: 60 s). None disables it (offline RL with slow policies)."""
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        self.mode = mode
        self.max_steps = max_steps
        self.time_limit_s = time_limit_s
        self._clock = clock
        self._pub: dict | None = None
        self.__truth: dict | None = None
        self.__result: dict | None = None

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int | str | None = None, config: dict | None = None, options: dict | None = None):
        options = options or {}
        if "episode" in options:
            ep: Episode = options["episode"]
            pub, truth = ep.public, ep.truth
        elif "episode_dir" in options:
            d = Path(options["episode_dir"])
            pub, (truth, _) = load_public(d), load_grader(d)
        else:
            if seed is None:
                raise ValueError("reset needs a seed (or options.episode / options.episode_dir)")
            ep = generate(seed, config)
            pub, truth = ep.public, ep.truth
        self._load(pub, truth)
        m = pub["manifest"]
        obs = {
            "task": TASK,
            "manifest": _copy(m),
            "policy": _copy(pub["policy"]),
            "tables": {t: {"primary_key": PRIMARY_KEYS[t], "count": len(self._rows[t]),
                           "columns": sorted(self._rows[t][0]) if self._rows[t] else []} for t in TABLES},
            "available_actions": {name: {"required": sorted(req), "optional": sorted(opt)} for name, (req, opt) in ARGS.items()},
            "action_format": {"name": "<action>", "arguments": {"...": "..."}},
            "finding_types": ["duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff"],
            "limits": {"max_steps": self.max_steps, "page_size": PAGE, "max_findings": MAX_FINDINGS,
                       "max_evidence_ids": MAX_EVIDENCE, "max_explanation_chars": MAX_EXPLANATION, "time_limit_s": self.time_limit_s},
            "steps_remaining": self.max_steps,
        }
        return obs, {"steps_remaining": self.max_steps, "error_code": None}

    def _load(self, pub: dict, truth: dict) -> None:
        self._pub = pub
        self.__truth = truth
        self.__result = None
        self._episode_id = pub["manifest"]["episode_id"]
        fx_rows = [{"fx_id": fx_id(r["date"], r["source_currency"], r["target_currency"]), **r} for r in pub["fx"]]
        # canonical key order, so observations are byte-identical whether the episode was generated or loaded from disk
        self._rows = {t: json.loads(dumps(rows)) for t, rows in  # recursive sorted keys
                      (("ledger", pub["ledger"]), ("invoices", pub["invoices"]), ("payments", pub["payments"]), ("fx", fx_rows))}
        self._index = {t: {r[PRIMARY_KEYS[t]]: r for r in rows} for t, rows in self._rows.items()}
        self._known = {k for idx in self._index.values() for k in idx}
        self._steps = 0
        self._done = False
        self._terminated = self._truncated = False
        self._t0 = self._clock()

    def result(self) -> dict | None:
        """Runner-side: the grade, available once the episode is over."""
        return _copy(self.__result) if self._done else None

    @property
    def steps_used(self) -> int:
        return self._steps

    # ------------------------------------------------------------------ step
    def step(self, action):
        if self._pub is None:
            raise RuntimeError("call reset() first")
        if self._done:
            return self._obs(None, "episode_over", "the episode is over; call reset()"), 0.0, self._terminated, self._truncated, self._info("episode_over")
        self._steps += 1
        if self.time_limit_s is not None and self._clock() - self._t0 > self.time_limit_s:
            return self._finish_truncated(self._obs(None, "time_limit", "episode time limit exceeded"), "time_limit")
        name, args, err = self._parse(action)
        if err:
            obs = self._obs(None, *err)
        elif name == "submit":
            return self._submit(args["findings"])
        else:
            try:
                obs = self._obs(getattr(self, "_a_" + name)(**args), None, None)
            except _ActionError as e:
                obs = self._obs(None, e.code, e.message)
        if self._steps >= self.max_steps:
            return self._finish_truncated(obs, "step_limit")
        return obs, 0.0, False, False, self._info(obs["error"]["code"] if obs["error"] else None)

    def _parse(self, action):
        if not isinstance(action, dict) or set(action) - {"name", "arguments"} or "name" not in action:
            return None, None, ("invalid_action", "action must be an object {name, arguments}")
        name, args = action["name"], action.get("arguments", {})
        if not isinstance(name, str) or name not in ARGS:
            return None, None, ("unknown_action", f"unknown action '{_echo(name)}'")
        if not isinstance(args, dict):
            return None, None, ("invalid_arguments", "arguments must be an object")
        req, opt = ARGS[name]
        if not req <= set(args) or set(args) - req - opt:
            return None, None, ("invalid_arguments", f"{name} takes required {sorted(req)} optional {sorted(opt)}")
        return name, args, None

    def _obs(self, result, code, message):
        return {"result": result, "error": None if code is None else {"code": code, "message": message},
                "steps_remaining": self.max_steps - self._steps}

    def _info(self, code):
        return {"steps_remaining": self.max_steps - self._steps, "error_code": code}

    # ------------------------------------------------------------------ actions
    def _table(self, table):
        if not isinstance(table, str) or table not in self._rows:
            raise _ActionError("unknown_table", f"unknown table '{_echo(table)}'")
        return self._rows[table]

    def _a_list_records(self, table, cursor=None, limit=None):
        rows = self._table(table)
        if limit is None:
            limit = PAGE
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= PAGE:
            raise _ActionError("invalid_arguments", "limit must be an integer 1..50")
        if cursor is None:
            start = 0
        else:
            m = _CURSOR.fullmatch(cursor) if isinstance(cursor, str) else None
            if not m or m.group(1) != table or int(m.group(2)) > len(rows) or m.group(3) != self._cursor_tag(table, int(m.group(2))):
                raise _ActionError("invalid_cursor", f"invalid cursor '{_echo(cursor)}'")
            start = int(m.group(2))
        end = min(start + limit, len(rows))
        nxt = f"{table}:{end}:{self._cursor_tag(table, end)}" if end < len(rows) else None
        return {"table": table, "rows": _copy(rows[start:end]), "next_cursor": nxt, "total": len(rows)}

    def _cursor_tag(self, table: str, offset: int) -> str:
        """Cursors are issued by the env only (stable for the episode, bound to the table)."""
        import hashlib
        return hashlib.sha256(f"{self._episode_id}|{table}|{offset}|cursor".encode()).hexdigest()[:8]

    def _a_get_record(self, table, id):
        self._table(table)
        if not isinstance(id, str) or id not in self._index[table]:
            raise _ActionError("not_found", f"no record '{_echo(id)}' in table '{table}'")
        return {"table": table, "record": _copy(self._index[table][id])}

    def _a_get_document(self, doc_id):
        if isinstance(doc_id, str):
            for t in ("invoices", "payments"):
                if doc_id in self._index[t]:
                    d = self._index[t][doc_id]
                    return {"doc_id": doc_id, "kind": d.get("doc_type") or d.get("kind"), "document": _copy(d)}
        raise _ActionError("not_found", f"no document '{_echo(doc_id)}'")

    def _a_calculator(self, expression):
        try:
            return {"expression": _echo(expression), "value": evaluate(expression)}
        except CalcError as e:
            raise _ActionError("calc_error", str(e)) from None

    # ------------------------------------------------------------------ termination
    def _grade(self, findings, inspection_steps, malformed=None):
        self.__result = grade(findings, self.__truth, inspection_steps, self._known, malformed)
        self._done = True
        return self.__result

    def _reward(self, g):
        return g["reward"] if self.mode == "train" else 0.0

    def _terminal_info(self, g, code):
        info = self._info(code)
        if self.mode == "train":
            info["grade"] = {k: g[k] for k in ("valid", "tp", "fp", "fn", "precision", "recall", "f1", "reward", "reward_experimental")}
        return info

    def _submit(self, raw):
        findings, err = parse_findings(raw, self._episode_id)
        g = self._grade(findings, self._steps - 1, malformed=err)
        self._terminated = True
        obs = self._obs({"submitted": True}, None, None)
        return obs, self._reward(g), True, False, self._terminal_info(g, None)

    def abort(self, reason: str):
        """Runner-side: the policy timed out or crashed -> truncated, graded as an empty answer."""
        if self._done:
            return self._obs(None, "episode_over", "the episode is over"), 0.0, self._terminated, self._truncated, self._info("episode_over")
        return self._finish_truncated(self._obs(None, reason, "policy " + reason.replace("_", " ")), reason)

    def _finish_truncated(self, obs, code):
        g = self._grade([], self._steps)
        self._truncated = True
        obs = dict(obs)
        if obs["error"] is None:
            obs["error"] = {"code": code, "message": "budget exhausted; episode truncated with an empty answer"}
        return obs, self._reward(g), False, True, self._terminal_info(g, code)


class _ActionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message
