"""Baseline policies. Interface (also spoken over the JSON-lines child protocol):

    policy.reset(observation, info) -> action
    policy.step(observation, reward, terminated, truncated, info) -> action

Built-ins: random, flag_all, reconcile (naive truth-free rules), expert (truth-free invariant auditor),
oracle (grader-only: needs the truth, runs inside the runner, submits at step 1).
"""
from __future__ import annotations

import math

from ..invariants import Books, check
from ..rng import Rng

TABLE_ORDER = ("fx", "invoices", "payments", "ledger")


class Policy:
    def reset(self, observation: dict, info: dict) -> dict:
        raise NotImplementedError

    def step(self, observation: dict, reward: float, terminated: bool, truncated: bool, info: dict) -> dict:
        raise NotImplementedError


def submit(episode_id: str, findings: list[dict]) -> dict:
    return {"name": "submit", "arguments": {"findings": {"episode_id": episode_id, "findings": findings[:30]}}}


def finding(type_: str, affected: list[str], amount: int, ccy: str, evidence: list[str], why: str) -> dict:
    return {"type": type_, "affected_ids": sorted(affected), "claimed_amount_minor": int(amount), "currency": ccy,
            "evidence_ids": evidence[:10], "explanation": why[:1000]}


class RandomPolicy(Policy):
    """Fixed-seed random explorer: random valid-looking actions, rare random submissions."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def reset(self, observation, info):
        self.r = Rng(str(self.seed), observation["manifest"]["episode_id"])
        self.ep = observation["manifest"]["episode_id"]
        self.base = observation["manifest"]["base_currency"]
        self.seen: list[str] = []
        return self._act()

    def step(self, observation, reward, terminated, truncated, info):
        res = observation.get("result") or {}
        for row in res.get("rows", []):
            self.seen.append(row.get("row_id") or row.get("doc_id") or row.get("fx_id"))
        return self._act()

    def _act(self):
        x = self.r.random()
        if x < 0.01 and self.seen:
            k = self.r.randint(1, 3)
            return submit(self.ep, [finding(self.r.choice(["duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff"]),
                                            [self.r.choice(self.seen)], self.r.randint(1, 10**6), self.base,
                                            [self.r.choice(self.seen)], "random") for _ in range(k)])
        if x < 0.55:
            return {"name": "list_records", "arguments": {"table": self.r.choice(list(TABLE_ORDER)), "cursor": f"p{50 * self.r.randint(0, 3)}"}}
        if x < 0.8 and self.seen:
            return {"name": "get_document", "arguments": {"doc_id": self.r.choice(self.seen)}}
        if x < 0.9:
            return {"name": "get_record", "arguments": {"table": "ledger", "id": self.r.choice(self.seen) if self.seen else "L-1"}}
        return {"name": "calculator", "arguments": {"expression": f"{self.r.randint(1, 999)}*{self.r.randint(1, 99)}"}}


class ScanPolicy(Policy):
    """Reads every page of every table (stable cursors), then decides findings from the full public view."""

    tables = TABLE_ORDER

    def reset(self, observation, info):
        self.ep = observation["manifest"]["episode_id"]
        self.pub = {"manifest": observation["manifest"], "policy": observation["policy"], "ledger": [], "invoices": [], "payments": [], "fx": []}
        self.queue = list(self.tables)
        self.cursor = None
        return self._next()

    def step(self, observation, reward, terminated, truncated, info):
        res = observation.get("result") or {}
        if "rows" in res:
            self.pub[res["table"]].extend(res["rows"])
            self.cursor = res["next_cursor"]
            if self.cursor is None:
                self.queue.pop(0)
        return self._next()

    def _next(self):
        if self.queue:
            return {"name": "list_records", "arguments": {"table": self.queue[0], "cursor": self.cursor, "limit": 50}}
        return submit(self.ep, self.decide(self.pub))

    def decide(self, pub: dict) -> list[dict]:
        raise NotImplementedError


class FlagAllPolicy(ScanPolicy):
    """Attack: accuse as many documents of as many error types as the 30-finding limit allows."""

    tables = ("invoices", "payments")

    def decide(self, pub):
        base = pub["manifest"]["base_currency"]
        out = []
        docs = pub["invoices"] + pub["payments"]
        types = ["duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff"]
        for i, d in enumerate(docs):
            t = types[i % 4]
            amt = d.get("total_minor", d.get("amount_minor", 0))
            out.append(finding(t, [d["doc_id"]], amt, base, [d["doc_id"]], "flag everything"))
        return out[:30]


class ReconcilePolicy(ScanPolicy):
    """A deliberately naive, truth-free reconciliation: string dates, no tz, no carried-rate logic.
    It falls for the distractors (recurring amounts, overpayments, tz boundaries) by design."""

    def decide(self, pub):
        m = pub["manifest"]
        base = m["base_currency"]
        p0, p1 = m["period_start"][:10], m["period_end"][:10]
        fx = {(r["date"], r["source_currency"]): r for r in pub["fx"]}
        led = pub["ledger"]
        out = []
        invs = [d for d in pub["invoices"] if d["doc_type"] == "invoice"]
        # duplicates: same entity + same total within 7 days
        invs_sorted = sorted(invs, key=lambda d: d["issued_at"])
        for i, a in enumerate(invs_sorted):
            for b in invs_sorted[i + 1:]:
                if a["entity_id"] == b["entity_id"] and a["total_minor"] == b["total_minor"] and a["currency"] == b["currency"] \
                        and b["issued_at"][:10] <= _add_days(a["issued_at"][:10], 7):
                    row = next((r for r in led if r["invoice_id"] == b["invoice_id"] and r["account"] == "1100"), None)
                    if row:
                        out.append(finding("duplicate_invoice", [a["doc_id"], b["doc_id"]], row["debit_minor"], base,
                                           [a["doc_id"], b["doc_id"], row["row_id"]], "same customer and total"))
        # allocation (same currency only): AR credits for the payment vs amount + fee
        for p in pub["payments"]:
            if p["kind"] != "receipt" or not p["invoice_refs"] or p["currency"] != base:
                continue
            credited = sum(r["credit_minor"] for r in led if r["payment_id"] == p["payment_id"] and r["account"] == "1100")
            gap = p["amount_minor"] + p["fee_minor"] - credited
            if gap > 0 and len(p["invoice_refs"]) == 1:
                inv = next((d for d in invs if d["invoice_id"] == p["invoice_refs"][0]), None)
                un = next((r for r in led if r["payment_id"] == p["payment_id"] and r["account"] in ("2150", "2300")), None)
                if inv and un:
                    out.append(finding("payment_allocation", [p["doc_id"], inv["doc_id"]], gap, base,
                                       [p["doc_id"], inv["doc_id"], un["row_id"]], "payment not fully applied"))
        # fx: rate of the literal date in the timestamp string
        for d in invs:
            if d["currency"] == base:
                continue
            row = next((r for r in led if r["invoice_id"] == d["invoice_id"] and r["account"] == "1100" and r["payment_id"] is None and r["debit_minor"] > 0), None)
            rate = _rate_on(fx, d["issued_at"][:10], d["currency"])
            if row and rate:
                from ..money import convert
                exp = convert(d["total_minor"], d["currency"], base, rate["rate"])
                if abs(exp - row["debit_minor"]) > 1:
                    out.append(finding("fx_conversion", [d["doc_id"]], abs(exp - row["debit_minor"]), base,
                                       [d["doc_id"], _fx_id(rate), row["row_id"]], "rate mismatch"))
        # cutoff: delivered date string vs posting date string, naive period test
        for d in invs:
            row = next((r for r in led if r["invoice_id"] == d["invoice_id"] and r["account"] == "1100" and r["payment_id"] is None and r["debit_minor"] > 0), None)
            if not row or not d["delivered_at"]:
                continue
            ev_in = p0 <= d["delivered_at"][:10] < p1
            post_in = p0 <= row["timestamp"][:10] < p1
            if ev_in != post_in and d["delivered_at"][:10] >= p0:
                out.append(finding("period_cutoff", [d["doc_id"]], row["debit_minor"], base, [d["doc_id"], row["row_id"]], "period mismatch"))
        return out[:30]


class ExpertPolicy(ScanPolicy):
    """Truth-free auditor: rebuilds the books from observed rows and applies the published policy
    (independent invariant checker). Shows every episode is solvable from visible evidence alone."""

    def decide(self, pub):
        pub = {**pub, "fx": [{k: v for k, v in r.items() if k != "fx_id"} for r in pub["fx"]]}
        b = Books(pub)
        base = b.base
        led = pub["ledger"]
        docs = {d["doc_id"]: d for d in pub["invoices"] + pub["payments"]}
        out = []
        for v in check(pub):
            if v.invariant == "I8":
                dup = docs[v.subject]
                orig = min((d for d in b.invoices if d["delivery_ref"] == dup["delivery_ref"] and d["doc_id"] != dup["doc_id"]),
                           key=lambda d: d["issued_at"])
                row = _src_row(led, dup)
                out.append(finding("duplicate_invoice", [orig["doc_id"], dup["doc_id"]], v.delta, base,
                                   [orig["doc_id"], dup["doc_id"], row["row_id"]], "same delivery billed twice"))
            elif v.invariant == "I5" and v.delta > 0:
                d = docs[v.subject]
                row = _src_row(led, d)
                ts = d.get("issued_at") or d["timestamp"]
                ev = [d["doc_id"], row["row_id"]] + ([_fx_id_for(b, ts, d["currency"])] if d["currency"] != base else [])
                out.append(finding("fx_conversion", [d["doc_id"]], v.delta, base, ev, "converted amount differs from policy rate"))
            elif v.invariant == "I6":
                d = docs[v.subject]
                out.append(finding("period_cutoff", [d["doc_id"]], v.delta, base, [d["doc_id"], _src_row(led, d)["row_id"]],
                                   "posted in a different period than the event"))
            elif v.invariant == "I7":
                a, c = v.subject.split("|")
                pay = docs[a] if "payment_id" in docs[a] else docs[c]
                inv = docs[c] if pay is docs[a] else docs[a]
                ar = next((r for r in led if r["account"] == "1100" and r["payment_id"] == pay["payment_id"] and r["invoice_id"] == inv["invoice_id"]), None)
                un = next((r for r in led if r["account"] == "2150" and r["payment_id"] == pay["payment_id"]), None)
                ev = [pay["doc_id"], inv["doc_id"], (ar or un)["row_id"]]
                if inv["currency"] != base:
                    ev.append(_fx_id_for(b, inv["issued_at"], inv["currency"]))
                out.append(finding("payment_allocation", [pay["doc_id"], inv["doc_id"]], v.delta, base, ev, "AR under-credited"))
        return out[:30]


class OraclePolicy(Policy):
    """Grader-only oracle: given the truth, submits every issue with its first evidence set at step 1."""

    def __init__(self, truth: dict):
        self.truth = truth

    def reset(self, observation, info):
        ep = observation["manifest"]["episode_id"]
        return submit(ep, [finding(i["type"], i["affected_ids"], i["expected_amount_minor"], i["currency"], i["evidence_sets"][0], "oracle")
                           for i in self.truth["issues"]])

    def step(self, observation, reward, terminated, truncated, info):
        return {"name": "calculator", "arguments": {"expression": "0"}}


# ---------------------------------------------------------------------------------------- helpers
def _src_row(led, d):
    if "payment_id" in d:
        from ..invariants import BANKS
        return next(r for r in led if r["account"] in BANKS and r["payment_id"] == d["payment_id"])
    is_inv = d["doc_type"] == "invoice"
    return next(r for r in led if r["account"] == "1100" and r["invoice_id"] == d["invoice_id"] and r["payment_id"] is None
                and (r["debit_minor"] if is_inv else r["credit_minor"]) > 0)


def _fx_id(r):
    return f"FX-{r['date']}-{r['source_currency']}-{r['target_currency']}"


def _fx_id_for(b: Books, ts: str, ccy: str) -> str:
    from ..timeutil import local_date
    d = local_date(ts, b.tz).isoformat()
    day = max(x for x in b.fx_dates if x <= d and (x, ccy) in b.fx)
    return f"FX-{day}-{ccy}-{b.base}"


def _rate_on(fx, day: str, ccy: str):
    for _ in range(7):
        if (day, ccy) in fx:
            return fx[(day, ccy)]
        day = _add_days(day, -1)
    return None


def _add_days(day: str, n: int) -> str:
    from datetime import date, timedelta
    return (date.fromisoformat(day) + timedelta(days=n)).isoformat()


BUILTIN = {"random": RandomPolicy, "flag_all": FlagAllPolicy, "reconcile": ReconcilePolicy, "expert": ExpertPolicy}
__all__ = ["Policy", "BUILTIN", "OraclePolicy", "RandomPolicy", "FlagAllPolicy", "ReconcilePolicy", "ExpertPolicy", "math"]
