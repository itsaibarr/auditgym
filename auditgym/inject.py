"""Error injectors. Two phases: `prepare` adds legitimate scenario documents with their correct postings,
then (after the correct books are snapshotted) `mutate` changes the recorded books and records the causal
change. Truth is derived from these mutations, never written by hand."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .money import convert
from .rng import Rng
from .timeutil import local_instant
from .world import (AR, FX_GAIN, FX_LOSS, RETURNS, REVENUE, UNAPPLIED, JE, Invoice, Line, Payment, World,
                    bank_account)

MIN_DELTA = 100  # every injected amount >= 100 x tolerance (tolerance = 1 minor unit)

TYPES = ("duplicate_invoice", "payment_allocation", "fx_conversion", "period_cutoff")


@dataclass
class Issue:
    type: str
    affected: list  # documents (Invoice | Payment)
    amount: int
    evidence: list[list[tuple]]  # each set: [("doc", d) | ("line", je, line) | ("fx", date, ccy)]
    invariant: str
    injector: str
    params: dict
    link: str | None = None
    before: list = field(default_factory=list)  # [(je, [line copies], ts)] or [] for new JEs
    jes: list = field(default_factory=list)


class Abort(Exception):
    pass


# ---------------------------------------------------------------------------------------- helpers
class Ctx:
    def __init__(self, w: World, key: str):
        self.w = w
        self.r = Rng(key, "inject")
        self.busy: set[int] = set()

    def je_of(self, doc) -> JE:
        return next(j for j in self.w.jes if j.src is doc)

    def pays_of(self, inv: Invoice) -> list[Payment]:
        return [p for p in self.w.payments if p.refs and inv in p.refs]

    def cns_of(self, inv: Invoice) -> list[Invoice]:
        return [c for c in self.w.invoices if c.ref is inv]

    def in_p(self, dt: datetime) -> bool:
        return self.w.in_period(dt)

    def mark(self, *docs) -> None:
        for d in docs:
            self.busy.add(id(d))
            if isinstance(d, Invoice):
                for p in self.pays_of(d):
                    self.busy.add(id(p))
                for c in self.cns_of(d):
                    self.busy.add(id(c))
                if d.ref is not None:
                    self.busy.add(id(d.ref))
            if isinstance(d, Payment) and d.refs:
                for i in d.refs:
                    self.busy.add(id(i))

    def free(self, d) -> bool:
        return id(d) not in self.busy

    def last_day(self):
        return self.w.period_end - timedelta(days=1)

    def snapshot(self, je: JE):
        return (je, [l.copy() for l in je.lines], je.ts)

    def add_je(self, je: JE) -> None:
        self.w.jes.append(je)
        self.w.jes.sort(key=lambda j: j.ts)
        je.key = f"je{len(self.w.jes)}x"

    def unpaid_foreign(self, days_before_end: int) -> list[Invoice]:
        w = self.w
        limit = w.period_end - timedelta(days=days_before_end)
        return [i for i in w.invoices if i.kind == "invoice" and not i.opening and i.ccy != w.base and self.free(i)
                and self.in_p(i.issued_at) and self.in_p(i.delivered_at) and w.local(i.issued_at) < limit
                and not self.pays_of(i) and not self.cns_of(i) and i.dup_of is None]


def _fresh_foreign_invoice(c: "Ctx", days_before_end: int) -> Invoice:
    """Add a legitimate, correctly posted, unpaid foreign-currency invoice (prepare phase only)."""
    from .world import _lines
    w, r = c.w, c.r
    cust = r.choice([x for x in w.customers if x.ccy != w.base])
    span = (w.period_end - w.period_start).days - days_before_end - 1
    d = w.period_start + timedelta(days=r.randint(1, max(1, span)))
    t = local_instant(d, r.randint(9, 16), r.randint(0, 59), w.tz)
    lines = _lines(r, w, cust.ccy)
    issued = t + timedelta(minutes=r.randint(10, 90))
    inv = Invoice(f"inv{len(w.invoices)}", "invoice", cust, cust.ccy, issued,
                  issued + timedelta(days=r.choice([14, 30, 45])), t, lines, sum(l["amount_minor"] for l in lines), f"dn{len(w.invoices)}",
                  utc_written=r.chance(0.3))
    w.invoices.append(inv)
    b = w.to_base(inv.total, inv.ccy, inv.issued_at)
    c.add_je(JE("", inv.issued_at, [Line(AR, b, 0, inv.cust, inv, role="ar"), Line(REVENUE, 0, b, inv.cust, inv, role="revenue")], "invoice", inv))
    return inv


def _line(je: JE, role: str) -> Line:
    return next(l for l in je.lines if l.role == role)


def _rebalance_fx(je: JE) -> None:
    je.lines = [l for l in je.lines if l.role != "fx"]
    diff = sum(l.debit for l in je.lines) - sum(l.credit for l in je.lines)
    p = next((l.pay for l in je.lines if l.pay is not None), None)
    cust = next((l.cust for l in je.lines if l.cust is not None), None)
    if diff > 0:
        je.lines.append(Line(FX_GAIN, 0, diff, cust, None, p, role="fx"))
    elif diff < 0:
        je.lines.append(Line(FX_LOSS, -diff, 0, cust, None, p, role="fx"))


def _fx_ref(w: World, dt: datetime, ccy: str) -> tuple:
    return ("fx", w.rate_date(w.local(dt), ccy), ccy)


def _wrong_rate_date(c: Ctx, correct_dt: datetime, ccy: str):
    """A business date whose effective rate differs from the correct one."""
    w = c.w
    good = w.rate_date(w.local(correct_dt), ccy)
    idx = w.fx_dates.index(good)
    for _ in range(10):
        k = c.r.choice([-5, -4, -3, -2, -1, 1, 2, 3])
        j = idx + k
        if 0 <= j < len(w.fx_dates) and w.fx_dates[j] != good:
            return w.fx_dates[j]
    raise Abort("no alternative fx date")


# ---------------------------------------------------------------------------------------- E1
def _clone(c: Ctx, orig: Invoice, shift: timedelta) -> tuple[Invoice, JE]:
    w = c.w
    issued = orig.issued_at + shift
    dup = Invoice(f"inv{len(w.invoices)}", "invoice", orig.cust, orig.ccy, issued, issued + (orig.due_at - orig.issued_at),
                  orig.delivered_at, [dict(l) for l in orig.lines], orig.total, orig.delivery_ref,
                  service_period=orig.service_period, utc_written=c.r.chance(0.3), dup_of=orig, same_number=c.r.chance(0.5))
    w.invoices.append(dup)
    b = w.to_base(dup.total, dup.ccy, dup.issued_at)
    je = JE("", issued, [Line(AR, b, 0, dup.cust, dup, role="ar"), Line(REVENUE, 0, b, dup.cust, dup, role="revenue")], "invoice", dup)
    c.add_je(je)
    return dup, je


def _dup_issue(dup: Invoice, je: JE, orig: Invoice, injector: str, link=None) -> Issue:
    ar, rev = _line(je, "ar"), _line(je, "revenue")
    return Issue("duplicate_invoice", [orig, dup], ar.debit,
                 [[("doc", orig), ("doc", dup), ("line", je, ar)], [("doc", orig), ("doc", dup), ("line", je, rev)]],
                 "I8", injector, {"same_number": dup.same_number, "shift_minutes": int((dup.issued_at - orig.issued_at).total_seconds() // 60)},
                 link=link, jes=[je])


def e1_prepare(c: Ctx):
    w = c.w
    limit = w.period_end - timedelta(days=4)
    cands = [i for i in w.invoices if i.kind == "invoice" and not i.opening and c.free(i) and c.in_p(i.issued_at)
             and c.in_p(i.delivered_at) and w.local(i.issued_at) < limit and not c.cns_of(i) and i.dup_of is None
             and "tz_boundary" not in i.tags]
    if not cands:
        raise Abort("no E1 target")
    orig = c.r.choice(cands)
    c.mark(orig)
    return orig


def e1_mutate(c: Ctx, orig: Invoice) -> list[Issue]:
    shift = timedelta(hours=c.r.randint(2, 70))
    dup, je = _clone(c, orig, shift)
    c.mark(dup)
    return [_dup_issue(dup, je, orig, "E1.clone_invoice")]


# ---------------------------------------------------------------------------------------- E2
def e2_prepare(c: Ctx):
    w = c.w
    cands = [p for p in w.payments if p.kind == "receipt" and p.ccy == w.base and p.refs and len(p.refs) == 1 and not p.fee
             and not p.tags & {"overpay", "on_account"} and c.free(p) and c.in_p(p.ts) and c.free(p.refs[0])]
    cands = [p for p in cands if _line(c.je_of(p), "ar").credit >= 10 * MIN_DELTA
             and c.pays_of(p.refs[0]) == [p] and not c.cns_of(p.refs[0])]
    if not cands:
        raise Abort("no E2 target")
    p = c.r.choice(cands)
    c.mark(p, p.refs[0])
    return p


def e2_mutate(c: Ctx, p: Payment) -> list[Issue]:
    je = c.je_of(p)
    before = c.snapshot(je)
    ar = _line(je, "ar")
    inv = p.refs[0]
    if c.r.chance(0.4):  # allocation missing entirely
        k = ar.credit
        je.lines.remove(ar)
        mode = "missing"
    else:  # partial allocation
        k = ar.credit * c.r.randint(2, 8) // 10
        ar.credit -= k
        mode = "partial"
    un = Line(UNAPPLIED, 0, k, p.cust, None, p, role="unapplied")
    je.lines.append(un)
    ev = [[("doc", p), ("doc", inv), ("line", je, un)]]
    if mode == "partial":
        ev.append([("doc", p), ("doc", inv), ("line", je, ar)])
    return [Issue("payment_allocation", [p, inv], k, ev, "I7", "E2.unallocated_" + mode, {"mode": mode}, before=[before], jes=[je])]


# ---------------------------------------------------------------------------------------- E3
def e3_prepare(c: Ctx):
    w = c.w
    invs = c.unpaid_foreign(1)
    pays = [p for p in w.payments if p.kind == "receipt" and p.ccy != w.base and p.refs and c.free(p) and c.in_p(p.ts)
            and not p.tags & {"overpay"} and all(c.free(i) for i in p.refs)]
    pool = [("inv", i) for i in invs] + [("pay", p) for p in pays]
    if not pool:
        raise Abort("no E3 target")
    kind, doc = c.r.choice(pool)
    c.mark(doc)
    return kind, doc


def e3_mutate(c: Ctx, target) -> list[Issue]:
    w = c.w
    kind, doc = target
    je = c.je_of(doc)
    before = c.snapshot(je)
    dt = doc.issued_at if kind == "inv" else doc.ts
    correct = w.to_base(doc.total if kind == "inv" else doc.amount, doc.ccy, dt)
    if kind == "pay" and c.r.chance(0.5) and w.rate_date(w.local(doc.refs[0].issued_at), doc.ccy) != w.rate_date(w.local(dt), doc.ccy):
        wrong_d = w.rate_date(w.local(doc.refs[0].issued_at), doc.ccy)
        how = "invoice_date_rate_used_for_payment"
    else:
        wrong_d = _wrong_rate_date(c, dt, doc.ccy)
        how = "wrong_rate_date"
    amount = doc.total if kind == "inv" else doc.amount
    wrong = convert(amount, doc.ccy, w.base, w.fx[(wrong_d, doc.ccy)])
    delta = abs(wrong - correct)
    if delta < MIN_DELTA:
        raise Abort("fx delta too small")
    if kind == "inv":
        ar, rev = _line(je, "ar"), _line(je, "revenue")
        ar.debit = rev.credit = wrong
        ev = [[("doc", doc), _fx_ref(w, dt, doc.ccy), ("line", je, ar)], [("doc", doc), _fx_ref(w, dt, doc.ccy), ("line", je, rev)]]
    else:
        bank = _line(je, "bank")
        bank.debit = wrong
        _rebalance_fx(je)
        ev = [[("doc", doc), _fx_ref(w, dt, doc.ccy), ("line", je, bank)]]
    return [Issue("fx_conversion", [doc], delta, ev, "I5", "E3." + how,
                  {"used_rate_date": wrong_d.isoformat(), "correct_rate_date": w.rate_date(w.local(dt), doc.ccy).isoformat()},
                  before=[before], jes=[je])]


# ---------------------------------------------------------------------------------------- E4
def e4_prepare(c: Ctx):
    w = c.w
    def ok(i):
        return (i.kind == "invoice" and not i.opening and c.free(i) and i.dup_of is None and not c.cns_of(i)
                and "tz_boundary" not in i.tags)
    early = [i for i in w.invoices if ok(i) and "boundary" in i.tags and w.local(i.delivered_at) >= w.period_end]
    late = [i for i in w.invoices if ok(i) and "boundary" in i.tags and c.in_p(i.delivered_at)]
    pool = [("premature", i) for i in early] + [("late", i) for i in late]
    if not pool:
        raise Abort("no E4 target")
    mode, inv = c.r.choice(pool)
    c.mark(inv)
    return mode, inv


def e4_mutate(c: Ctx, target) -> list[Issue]:
    w = c.w
    mode, inv = target
    je = c.je_of(inv)
    before = c.snapshot(je)
    if mode == "premature":
        je.ts = local_instant(c.last_day(), c.r.randint(15, 21), c.r.randint(0, 59), w.tz)
    else:
        je.ts = local_instant(w.period_end + timedelta(days=c.r.randint(1, 3)), c.r.randint(9, 12), c.r.randint(0, 59), w.tz)
    w.jes.sort(key=lambda j: j.ts)
    ar, rev = _line(je, "ar"), _line(je, "revenue")
    return [Issue("period_cutoff", [inv], ar.debit, [[("doc", inv), ("line", je, ar)], [("doc", inv), ("line", je, rev)]],
                  "I6", "E4." + mode, {"mode": mode}, before=[before], jes=[je])]


# ---------------------------------------------------------------------------------------- X1
def x1_prepare(c: Ctx):
    """Partial foreign-currency payment at the period boundary (legit doc; correct posting)."""
    w = c.w
    cands = c.unpaid_foreign(5)
    inv = c.r.choice(cands) if cands and c.r.chance(0.7) else _fresh_foreign_invoice(c, 5)
    ts = local_instant(c.last_day(), c.r.randint(9, 23), c.r.randint(0, 59), w.tz)
    amt = inv.total * c.r.randint(3, 6) // 10
    part = amt * c.r.randint(2, 5) // 10
    inv_rate = w.rate(w.local(inv.issued_at), inv.ccy)
    correct_bank = convert(amt, inv.ccy, w.base, w.rate(w.local(ts), inv.ccy))
    # the receipt is booked on a business day of the post-period window, at that day's rate
    days = [d for d in w.fx_dates if w.period_end <= d < w.export_end]
    new_day = next((d for d in days if abs(convert(amt, inv.ccy, w.base, w.fx[(d, inv.ccy)]) - correct_bank) >= MIN_DELTA), None)
    if new_day is None:
        raise Abort("x1 no booking day with a material rate change")
    wrong_bank = convert(amt, inv.ccy, w.base, w.fx[(new_day, inv.ccy)])
    alloc_delta = convert(amt, inv.ccy, w.base, inv_rate) - convert(part, inv.ccy, w.base, inv_rate)
    if abs(wrong_bank - correct_bank) < MIN_DELTA or alloc_delta < MIN_DELTA or part <= 0:
        raise Abort("x1 deltas too small")
    c.mark(inv)
    p = Payment(f"pay{len(w.payments)}", "receipt", inv.cust, inv.ccy, ts, amt, [inv], utc_written=c.r.chance(0.5))
    w.payments.append(p)
    c.mark(p)
    rel = convert(amt, inv.ccy, w.base, inv_rate)
    je = JE("", ts, [Line(bank_account(w, p.ccy), correct_bank, 0, p.cust, None, p, role="bank"), Line(AR, 0, rel, p.cust, inv, p, role="ar")], "receipt", p)
    _rebalance_fx(je)
    c.add_je(je)
    return p, new_day, part


def x1_mutate(c: Ctx, target) -> list[Issue]:
    w = c.w
    p, new_day, part = target
    inv = p.refs[0]
    je = c.je_of(p)
    before = c.snapshot(je)
    correct_bank = w.to_base(p.amount, p.ccy, p.ts)
    wrong_rate = w.fx[(new_day, p.ccy)]
    wrong_bank = convert(p.amount, p.ccy, w.base, wrong_rate)
    inv_rate = w.rate(w.local(inv.issued_at), inv.ccy)
    rel_ok = convert(p.amount, inv.ccy, w.base, inv_rate)
    rel_rec = convert(part, inv.ccy, w.base, inv_rate)
    fx_delta, alloc_delta = abs(wrong_bank - correct_bank), rel_ok - rel_rec
    je.ts = local_instant(new_day, c.r.randint(9, 12), c.r.randint(0, 59), w.tz)
    w.jes.sort(key=lambda j: j.ts)
    bank, ar = _line(je, "bank"), _line(je, "ar")
    bank.debit = wrong_bank
    ar.credit = rel_rec
    un = Line(UNAPPLIED, 0, convert(p.amount - part, p.ccy, w.base, wrong_rate), p.cust, None, p, role="unapplied")
    je.lines.insert(2, un)
    _rebalance_fx(je)
    common = dict(before=[before], jes=[je], link="X1")
    return [
        Issue("period_cutoff", [p], wrong_bank, [[("doc", p), ("line", je, bank)]], "I6", "X1.cutoff", {"posted_on": new_day.isoformat()}, **common),
        Issue("fx_conversion", [p], fx_delta, [[("doc", p), _fx_ref(w, p.ts, p.ccy), ("line", je, bank)]], "I5", "X1.fx",
              {"used_rate_date": new_day.isoformat()}, **common),
        Issue("payment_allocation", [p, inv], alloc_delta, [[("doc", p), ("doc", inv), _fx_ref(w, inv.issued_at, inv.ccy), ("line", je, ar)]],
              "I7", "X1.partial_allocation", {"allocated_doc_minor": part}, **common),
    ]


# ---------------------------------------------------------------------------------------- X2
def x2_prepare(c: Ctx):
    """Foreign invoice with a credit note (legit), later duplicated; the credit note is booked at the wrong rate."""
    w = c.w
    cands = c.unpaid_foreign(10)
    inv = c.r.choice(cands) if cands and c.r.chance(0.7) else _fresh_foreign_invoice(c, 10)
    t = inv.issued_at + timedelta(days=c.r.randint(3, 6), hours=c.r.randint(0, 4))
    if w.rate_date(w.local(t), inv.ccy) == w.rate_date(w.local(inv.issued_at), inv.ccy) or not c.in_p(t):
        raise Abort("x2 cn date")
    cn_amt = inv.total * c.r.randint(1, 3) // 10
    inv_rate = w.rate(w.local(inv.issued_at), inv.ccy)
    if abs(convert(cn_amt, inv.ccy, w.base, inv_rate) - w.to_base(cn_amt, inv.ccy, t)) < MIN_DELTA:
        raise Abort("x2 delta")
    cn = Invoice(f"inv{len(w.invoices)}", "credit_note", inv.cust, inv.ccy, t, t, None,
                 [{"sku": "ADJ", "description": "Price adjustment", "qty": 1, "unit_price_minor": cn_amt, "amount_minor": cn_amt}],
                 cn_amt, None, ref=inv, utc_written=c.r.chance(0.3))
    w.invoices.append(cn)
    b = w.to_base(cn_amt, cn.ccy, t)
    c.add_je(JE("", t, [Line(RETURNS, b, 0, cn.cust, cn, role="returns"), Line(AR, 0, b, cn.cust, cn, role="ar")], "credit_note", cn))
    c.mark(inv, cn)
    return inv, cn


def x2_mutate(c: Ctx, target) -> list[Issue]:
    w = c.w
    inv, cn = target
    je = c.je_of(cn)
    before = c.snapshot(je)
    correct = w.to_base(cn.total, cn.ccy, cn.issued_at)
    wrong = convert(cn.total, cn.ccy, w.base, w.rate(w.local(inv.issued_at), inv.ccy))
    if abs(wrong - correct) < MIN_DELTA:
        raise Abort("x2 delta")
    ar, ret = _line(je, "ar"), _line(je, "returns")
    ar.credit = ret.debit = wrong
    dup, dje = _clone(c, inv, timedelta(hours=c.r.randint(20, 50)))
    c.mark(dup)
    return [
        _dup_issue(dup, dje, inv, "X2.duplicate", link="X2"),
        Issue("fx_conversion", [cn], abs(wrong - correct),
              [[("doc", cn), _fx_ref(w, cn.issued_at, cn.ccy), ("line", je, ar)], [("doc", cn), _fx_ref(w, cn.issued_at, cn.ccy), ("line", je, ret)]],
              "I5", "X2.credit_note_at_invoice_rate", {"used_rate_date": w.rate_date(w.local(inv.issued_at), inv.ccy).isoformat()},
              link="X2", before=[before], jes=[je]),
    ]


INJECTORS = {
    "duplicate_invoice": (e1_prepare, e1_mutate),
    "payment_allocation": (e2_prepare, e2_mutate),
    "fx_conversion": (e3_prepare, e3_mutate),
    "period_cutoff": (e4_prepare, e4_mutate),
    "X1": (x1_prepare, x1_mutate),
    "X2": (x2_prepare, x2_mutate),
}


def plan(w: World, cfg: dict, key: str) -> tuple[Ctx, list]:
    """Decide which injectors run (independently configurable per type) and run their prepare phase."""
    c = Ctx(w, key)
    rp = Rng(key, "plan")
    jobs = []
    for name in ("X1", "X2"):
        if rp.chance(cfg.get("interacting", {}).get(name, 0.0)):
            jobs.append(name)
    for t in TYPES:
        spec = cfg.get("errors", {}).get(t, {"on": False})
        if spec.get("on"):
            jobs += [t] * rp.randint(*spec.get("count", [1, 1]))
    size = {"X1": 3, "X2": 2}
    capped, total = [], 0
    for j in jobs:  # the cap counts issues, not injectors (X1 yields 3 linked issues, X2 yields 2)
        if total + size.get(j, 1) <= cfg.get("max_errors", 12):
            capped.append(j)
            total += size.get(j, 1)
    jobs = capped
    prepared = []
    for name in jobs:
        for _attempt in range(6):
            try:
                prepared.append((name, INJECTORS[name][0](c)))
                break
            except Abort:
                continue
    return c, prepared


def mutate(c: Ctx, prepared: list) -> list[Issue]:
    issues: list[Issue] = []
    for name, target in prepared:
        try:
            issues += INJECTORS[name][1](c, target)
        except Abort:
            continue
    return issues
