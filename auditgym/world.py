"""Hidden company world + posting engine (correct books).

Truth-first: a seeded set of business events is posted through fixed journal-entry templates
(see .claude/skills/audit-domain-expert/references/journal-entry-templates.md). The result is the
*correct* books. Error injectors (inject.py) later mutate a copy of these books.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from .money import EXPONENTS, convert
from .rng import Rng
from .timeutil import daterange, is_business_day, local_instant

# ---- chart of accounts -------------------------------------------------------------------
BANK_BASE = "1000"
BANK_FOREIGN = {"USD": "1010", "EUR": "1020", "JPY": "1030", "KWD": "1040", "CNY": "1050", "GBP": "1060", "KZT": "1070"}
AR = "1100"
ACCRUED_LIAB = "2100"
UNAPPLIED = "2150"
REFUNDS_PAYABLE = "2300"
EQUITY = "3000"
REVENUE = "4000"
RETURNS = "4100"
EXPENSES = "6000"
BANK_CHARGES = "6500"
FX_GAIN = "7100"
FX_LOSS = "7200"

CHART_OF_ACCOUNTS = {
    "1000": "Bank (base currency)", "1010": "Bank USD (valued in base)", "1020": "Bank EUR (valued in base)",
    "1030": "Bank JPY (valued in base)", "1040": "Bank KWD (valued in base)", "1050": "Bank CNY (valued in base)",
    "1060": "Bank GBP (valued in base)", "1070": "Bank KZT (valued in base)",
    "1100": "Accounts receivable (control)", "2100": "Accrued liabilities", "2150": "Unapplied cash / customer advances",
    "2300": "Customer refunds payable", "3000": "Equity / opening balance", "4000": "Revenue",
    "4100": "Sales returns and allowances", "6000": "Operating expenses", "6500": "Bank charges",
    "7100": "Realized FX gain", "7200": "Realized FX loss",
}

LEVEL = {"USD": 1.0, "KZT": 478.0, "EUR": 0.93, "JPY": 152.0, "KWD": 0.307, "CNY": 7.15, "GBP": 0.79}

_ADJ = ["Zhasyl", "Sary", "Kok", "Zhana", "Ak", "Biryuza", "Severo", "Nord", "Step", "Tau", "Samal", "Orion",
        "Vega", "Ornek", "Arman", "Dala", "Kokzhal", "Zhuldyz", "Kumis", "Sapa", "Shyn", "Ural", "Irtysh", "Baikonur"]
_NOUN = ["Trade", "Logistic", "Agro", "Stroi", "Tech", "Service", "Pack", "Metal", "Textile", "Pharm", "Energy",
         "Media", "Soft", "Market", "Foods", "Line", "Group", "Systems", "Supply", "Print"]
_SUFFIX = ["LLP", "JSC", "LLC", "GmbH", "Ltd", "TOO"]
_SKUS = [("SKU-CABLE", "Copper cable, 100 m"), ("SKU-PALLET", "Wooden pallet"), ("SKU-PUMP", "Water pump"),
         ("SKU-LIC", "Software licence, monthly"), ("SKU-AUDIT", "Consulting hours"), ("SKU-SEED", "Seed batch"),
         ("SKU-STEEL", "Steel sheet, 2 mm"), ("SKU-PAINT", "Industrial paint, 20 l"), ("SKU-TRUCK", "Freight service"),
         ("SKU-FILTER", "Air filter"), ("SKU-PIPE", "PVC pipe, 6 m"), ("SKU-SUPPORT", "Support plan, monthly")]


@dataclass(eq=False)
class Customer:
    key: str
    name: str
    ccy: str
    subscription: bool = False


@dataclass(eq=False)
class Invoice:
    key: str
    kind: str  # "invoice" | "credit_note"
    cust: Customer
    ccy: str
    issued_at: datetime
    due_at: datetime
    delivered_at: datetime | None
    lines: list[dict]
    total: int
    delivery_ref: str | None
    ref: "Invoice | None" = None
    opening: bool = False
    service_period: str | None = None
    utc_written: bool = False
    dup_of: "Invoice | None" = None
    same_number: bool = False
    tags: set = field(default_factory=set)


@dataclass(eq=False)
class Payment:
    key: str
    kind: str  # "receipt" | "refund"
    cust: Customer
    ccy: str
    ts: datetime
    amount: int
    refs: list[Invoice] | None
    fee: int = 0
    utc_written: bool = False
    tags: set = field(default_factory=set)


@dataclass(eq=False)
class Line:
    account: str
    debit: int
    credit: int
    cust: Customer | None = None
    inv: Invoice | None = None
    pay: Payment | None = None
    role: str = ""  # internal: ar | revenue | bank | fx | unapplied | refund | fee | expense | equity | returns

    def copy(self) -> "Line":
        return Line(self.account, self.debit, self.credit, self.cust, self.inv, self.pay, self.role)


@dataclass(eq=False)
class JE:
    key: str
    ts: datetime
    lines: list[Line]
    src_kind: str  # invoice | credit_note | receipt | refund | opening | expense | accrual | reversal
    src: object | None = None


@dataclass
class World:
    key: str
    base: str
    tz: str
    period_start: date
    period_end: date  # exclusive
    export_end: date  # exclusive, includes post-period cutoff window
    foreign: list[str]
    fx: dict[tuple[date, str], str]  # (date, src) -> rate (base per 1 src)
    fx_dates: list[date]
    customers: list[Customer]
    invoices: list[Invoice]
    payments: list[Payment]
    jes: list[JE]
    scenario: dict = field(default_factory=dict)

    # --- fx helpers ------------------------------------------------------------------------
    def rate_date(self, d: date, src: str) -> date:
        while (d, src) not in self.fx:
            d -= timedelta(days=1)
        return d

    def rate(self, d: date, src: str) -> str | None:
        if src == self.base:
            return None
        return self.fx[(self.rate_date(d, src), src)]

    def local(self, dt: datetime) -> date:
        return dt.astimezone(ZoneInfo(self.tz)).date()

    def to_base(self, amount: int, ccy: str, dt: datetime) -> int:
        return convert(amount, ccy, self.base, self.rate(self.local(dt), ccy))

    def in_period(self, dt: datetime) -> bool:
        return self.period_start <= self.local(dt) < self.period_end


def _sig6(x: float) -> str:
    d = Decimal(repr(x))
    q = Decimal(1).scaleb(d.adjusted() - 5)
    return format(d.quantize(q).normalize(), "f")


def _nice_price(rng: Rng, usd: float, ccy: str) -> int:
    major = max(1, int(usd * LEVEL[ccy]))
    digits = len(str(major))
    q = 10 ** max(0, digits - 3)
    major = major // q * q
    e = EXPONENTS[ccy]
    minor = major * 10**e
    if e and rng.chance(0.35):
        minor += rng.randint(1, 10**e - 1)
    return minor


# ======================================================================================
def build_world(key: str, cfg: dict) -> World:
    """Generate the hidden world and post the correct books."""
    r = Rng(key, "settings")
    base = r.choice(cfg["base_currencies"])
    tz = r.choice(cfg["timezones"])
    year = cfg.get("year", 2026)
    month = r.choice(cfg["months"])
    period_start = date(year, month, 1)
    period_end = date(year + (month == 12), month % 12 + 1, 1)
    window = cfg.get("cutoff_window_days", 5)
    export_end = period_end + timedelta(days=window)
    pool = [c for c in cfg["foreign_currencies"] if c != base]
    foreign = r.sample(pool, r.randint(1, min(3, len(pool))))

    # ---- fx: seeded random walk, business days only (weekend gaps) ----------------------
    rf = Rng(key, "fx")
    fx: dict[tuple[date, str], str] = {}
    fx_start = period_start - timedelta(days=50)
    levels = {c: LEVEL[base] / LEVEL[c] * (1 + (rf.random() - 0.5) * 0.06) for c in foreign}
    fx_dates = []
    for d in daterange(fx_start, export_end + timedelta(days=1)):
        if not is_business_day(d):
            continue
        fx_dates.append(d)
        for c in foreign:
            levels[c] *= 1 + (rf.random() - 0.5) * 0.012
            fx[(d, c)] = _sig6(levels[c])

    w = World(key, base, tz, period_start, period_end, export_end, foreign, fx, fx_dates, [], [], [], [])

    # ---- customers -------------------------------------------------------------------------
    rc = Rng(key, "customers")
    n_cust = rc.randint(*cfg["customers"])
    names = set()
    for i in range(n_cust):
        while True:
            name = f"{rc.choice(_ADJ)} {rc.choice(_NOUN)} {rc.choice(_SUFFIX)}"
            if name not in names:
                names.add(name)
                break
        ccy = base if rc.chance(0.5) else rc.choice(foreign)
        w.customers.append(Customer(f"c{i}", name, ccy, subscription=rc.chance(0.2)))
    if not any(c.ccy != base for c in w.customers):
        w.customers[0].ccy = foreign[0]
    if not any(c.ccy == base for c in w.customers):
        w.customers[-1].ccy = base

    _make_invoices(w, cfg)
    _make_payments(w, cfg)
    _make_expenses(w, cfg)
    post_books(w)
    return w


def _rand_time(r: Rng, w: World, d: date, late: bool = False) -> datetime:
    if late:
        return local_instant(d, 23, r.randint(5, 55), w.tz)
    return local_instant(d, r.randint(8, 19), r.randint(0, 59), w.tz)


def lagged_issue(r: Rng, w: World, delivered: datetime, min_minutes: int = 130) -> datetime:
    """Invoice billed hours to days after delivery (random minutes), at a local business-ish hour 08:00-23:59.
    Shared by legit late billing and injected duplicates so issue timing does not tell them apart."""
    t = delivered + timedelta(minutes=r.randint(min_minutes, max(min_minutes, 78 * 60)))
    loc = t.astimezone(ZoneInfo(w.tz))
    if loc.hour < 8:
        t = local_instant(loc.date(), 8, r.randint(0, 59), w.tz)
    return t


def _lines(r: Rng, w: World, ccy: str, n: int | None = None, usd_scale: float = 1.0) -> list[dict]:
    out = []
    for sku, desc in r.sample(_SKUS, n or r.randint(1, 4)):
        qty = r.randint(1, 12)
        unit = _nice_price(r, (20 + r.random() * 900) * usd_scale, ccy)
        out.append({"sku": sku, "description": desc, "qty": qty, "unit_price_minor": unit, "amount_minor": unit * qty})
    return out


def _new_invoice(w: World, r: Rng, cust: Customer, delivered: datetime, issued: datetime, lines: list[dict] | None = None,
                 opening: bool = False) -> Invoice:
    lines = lines or _lines(r, w, cust.ccy)
    inv = Invoice(
        key=f"inv{len(w.invoices)}", kind="invoice", cust=cust, ccy=cust.ccy, issued_at=issued,
        due_at=issued + timedelta(days=r.choice([14, 30, 45])), delivered_at=delivered, lines=lines,
        total=sum(l["amount_minor"] for l in lines), delivery_ref=f"dn{len(w.invoices)}", opening=opening,
        utc_written=r.chance(0.3),
    )
    w.invoices.append(inv)
    return inv


def _period_days(w: World) -> list[date]:
    return list(daterange(w.period_start, w.period_end))


def _make_invoices(w: World, cfg: dict) -> None:
    r = Rng(w.key, "invoices")
    days = _period_days(w)
    # opening AR: invoices issued last month, still open at period start (D18)
    for _ in range(r.randint(*cfg["opening_invoices"])):
        d = w.period_start - timedelta(days=r.randint(5, 28))
        t = _rand_time(r, w, d)
        _new_invoice(w, r, r.choice(w.customers), t, t + timedelta(minutes=r.randint(10, 300)), opening=True)
    # period sales
    n = r.randint(*cfg["invoices"])
    for _ in range(n):
        d = r.choice(days)
        late = r.chance(0.06)  # D10: deliveries just before local midnight (some written in UTC)
        t = _rand_time(r, w, d, late=late)
        issued = t + timedelta(minutes=r.randint(1, 4) if late else r.randint(10, 240))
        if not late and r.chance(0.15):  # billed 1-3 days after delivery, still within the same period
            lagged = lagged_issue(r, w, t)
            if w.local(lagged) < w.period_end:
                issued = lagged
        inv = _new_invoice(w, r, r.choice(w.customers), t, issued)
        if late:
            inv.utc_written = True
            inv.tags.add("tz_boundary")
    # boundary deliveries: last days of period and first days of the cutoff window
    for d in [w.period_end - timedelta(days=k) for k in (1, 2)] + [w.period_end + timedelta(days=k) for k in (0, 1, 2)]:
        for _ in range(r.randint(1, 2)):
            t = _rand_time(r, w, d)
            inv = _new_invoice(w, r, r.choice(w.customers), t, t + timedelta(minutes=r.randint(10, 120)))
            inv.tags.add("boundary")
    # D10 clean twin just after midnight local in P+1, written in UTC (looks like P in UTC)
    t = local_instant(w.period_end, 0, r.randint(5, 50), w.tz)
    inv = _new_invoice(w, r, r.choice(w.customers), t, t + timedelta(minutes=r.randint(2, 30)))
    inv.utc_written = True
    inv.tags.add("tz_boundary")
    # D1: subscriptions: identical weekly invoices, distinct deliveries / service periods
    for cust in [c for c in w.customers if c.subscription]:
        lines = _lines(r, w, cust.ccy, n=1, usd_scale=0.5)
        for wk, d in enumerate(days[r.randint(0, 6)::7]):
            t = local_instant(d, 9, 0, w.tz)
            inv = _new_invoice(w, r, cust, t, t + timedelta(minutes=5), lines=[dict(l) for l in lines])
            inv.service_period = f"{d.isoformat()}/P7D"
            inv.tags.add("recurring")
    # D14: same amount, different customers, same day
    if len(w.customers) >= 2:
        a, b = r.sample(w.customers, 2)
        if a.ccy == b.ccy:
            d = r.choice(days)
            lines = _lines(r, w, a.ccy, n=1)
            for cust in (a, b):
                t = _rand_time(r, w, d)
                _new_invoice(w, r, cust, t, t + timedelta(minutes=20), lines=[dict(l) for l in lines]).tags.add("same_amount")


def _pay_ts(r: Rng, w: World, after: datetime, lo: int = 1, hi: int = 25) -> datetime:
    """Receipt time: never before the period opens (opening AR is carried at full value), sometimes late in the
    evening, sometimes on the last day of the period (legitimate twins of the X1 boundary payment)."""
    d = max(w.local(after), w.period_start) + timedelta(days=r.randint(lo, hi))
    x = r.random()
    last = w.period_end - timedelta(days=1)
    if x < 0.06 and w.local(after) < last:
        d = last
    if x < 0.15:
        return local_instant(d, r.randint(20, 23), r.randint(0, 59), w.tz)
    return _rand_time(r, w, d)


def _make_payments(w: World, cfg: dict) -> None:
    r = Rng(w.key, "payments")
    export_end_dt = local_instant(w.export_end, 0, 0, w.tz)
    invoices = [i for i in w.invoices if i.kind == "invoice"]
    used: set[int] = set()
    by_cust: dict[str, list[Invoice]] = {}
    for inv in invoices:
        by_cust.setdefault(inv.cust.key, []).append(inv)

    def add(cust, ccy, ts, amount, refs, kind="receipt", fee=0, tag=None):
        if ts >= export_end_dt:
            return None
        p = Payment(f"pay{len(w.payments)}", kind, cust, ccy, ts, amount, refs, fee, utc_written=r.chance(0.3))
        if tag:
            p.tags.add(tag)
        w.payments.append(p)
        return p

    patterns = ["full", "split", "partial", "unpaid", "bulk", "overpay", "fee", "credit_note"]
    weights = [44, 10, 8, 16, 7, 5, 5, 5]
    for inv in invoices:
        if id(inv) in used or "boundary" in inv.tags and not inv.opening and w.local(inv.issued_at) >= w.period_end:
            continue
        pat = r.weighted(patterns, weights)
        if pat == "fee" and inv.ccy != w.base:
            pat = "full"
        used.add(id(inv))
        if pat == "full":
            add(inv.cust, inv.ccy, _pay_ts(r, w, inv.issued_at), inv.total, [inv])
        elif pat == "split":  # D2
            k = r.randint(2, 3)
            parts = _split(r, inv.total, k)
            t = inv.issued_at
            for part in parts:
                t = _pay_ts(r, w, t, 1, 8)
                add(inv.cust, inv.ccy, t, part, [inv], tag="split")
        elif pat == "partial":  # D4 genuine open balance
            add(inv.cust, inv.ccy, _pay_ts(r, w, inv.issued_at), inv.total * r.randint(3, 7) // 10, [inv], tag="partial")
        elif pat == "bulk":  # D3
            others = [o for o in by_cust[inv.cust.key] if id(o) not in used and o.ccy == inv.ccy][:2]
            group = [inv] + others
            for o in others:
                used.add(id(o))
            last = max(group, key=lambda x: x.issued_at)
            add(inv.cust, inv.ccy, _pay_ts(r, w, last.issued_at), sum(g.total for g in group), group, tag="bulk")
        elif pat == "overpay":  # D7
            excess = max(inv.total // 20, 10 ** EXPONENTS[inv.ccy])
            p = add(inv.cust, inv.ccy, _pay_ts(r, w, inv.issued_at, 1, 12), inv.total + excess, [inv], tag="overpay")
            if p:
                add(inv.cust, inv.ccy, _pay_ts(r, w, p.ts, 1, 6), excess, [inv], kind="refund", tag="refund")
        elif pat == "fee":  # D12 (base currency only; fee stated on the payment document)
            fee = max(inv.total // 200, 100)
            add(inv.cust, inv.ccy, _pay_ts(r, w, inv.issued_at), inv.total - fee, [inv], fee=fee, tag="fee")
        elif pat == "credit_note":  # D5
            cn_amount = inv.total * r.randint(1, 3) // 10
            t = max(inv.issued_at, local_instant(w.period_start, 9, 0, w.tz)) + timedelta(days=r.randint(1, 6), hours=r.randint(0, 5), minutes=r.randint(0, 59))
            if t >= export_end_dt:
                continue
            cn = Invoice(f"inv{len(w.invoices)}", "credit_note", inv.cust, inv.ccy, t, t, None,
                         [{"sku": "ADJ", "description": "Price adjustment", "qty": 1, "unit_price_minor": cn_amount,
                           "amount_minor": cn_amount}], cn_amount, None, ref=inv, utc_written=r.chance(0.3))
            w.invoices.append(cn)
            add(inv.cust, inv.ccy, _pay_ts(r, w, t, 1, 10), inv.total - cn_amount, [inv], tag="net_of_cn")
    # D17: on-account receipts without invoice references
    for _ in range(r.randint(*cfg["on_account"])):
        cust = r.choice(w.customers)
        d = r.choice(_period_days(w))
        add(cust, cust.ccy, _rand_time(r, w, d), _nice_price(r, 100 + r.random() * 2000, cust.ccy), None, tag="on_account")


def _split(r: Rng, total: int, k: int) -> list[int]:
    cuts = sorted(r.randint(total // 5, total * 4 // 5) for _ in range(k - 1))
    parts, prev = [], 0
    for c in cuts + [total]:
        parts.append(c - prev)
        prev = c
    return [p for p in parts if p > 0] if all(p > 0 for p in parts) else [total]


def _make_expenses(w: World, cfg: dict) -> None:
    r = Rng(w.key, "expenses")
    days = _period_days(w)
    rent = _nice_price(r, 1500, w.base)
    for d in days[::7]:  # recurring equal amounts (rent): legitimate
        w.jes.append(JE("", local_instant(d, 10, 0, w.tz), [Line(EXPENSES, rent, 0, role="expense"), Line(BANK_BASE, 0, rent, role="bank")], "expense"))
    for _ in range(r.randint(*cfg["expenses"])):
        d = r.choice(days)
        amt = _nice_price(r, 5 + r.random() * 600, w.base)
        lines = [Line(EXPENSES, amt, 0, role="expense"), Line(BANK_BASE, 0, amt, role="bank")]
        if r.chance(0.15):
            fee = max(amt // 100, 1)
            lines = [Line(EXPENSES, amt, 0, role="expense"), Line(BANK_CHARGES, fee, 0, role="fee"), Line(BANK_BASE, 0, amt + fee, role="bank")]
        w.jes.append(JE("", _rand_time(r, w, d), lines, "expense"))
    # D11 accrued expense at period end + automatic reversal on day 1 of next period
    amt = _nice_price(r, 700, w.base)
    last = w.period_end - timedelta(days=1)
    w.jes.append(JE("", local_instant(last, 18, 0, w.tz), [Line(EXPENSES, amt, 0, role="expense"), Line(ACCRUED_LIAB, 0, amt, role="accrual")], "accrual"))
    w.jes.append(JE("", local_instant(w.period_end, 9, 0, w.tz), [Line(ACCRUED_LIAB, amt, 0, role="accrual"), Line(EXPENSES, 0, amt, role="expense")], "reversal"))


# ======================================================================================
def bank_account(w: World, ccy: str) -> str:
    return BANK_BASE if ccy == w.base else BANK_FOREIGN[ccy]


def post_books(w: World) -> None:
    """Post every document event through the JE templates, in time order, tracking AR state."""
    r = Rng(w.key, "opening")
    opening_cash = _nice_price(r, 40000, w.base)
    open_lines = [Line(BANK_BASE, opening_cash, 0, role="bank")]
    total = opening_cash
    for inv in [i for i in w.invoices if i.opening]:
        b = w.to_base(inv.total, inv.ccy, inv.issued_at)
        open_lines.append(Line(AR, b, 0, inv.cust, inv, role="ar"))
        total += b
    open_lines.append(Line(EQUITY, 0, total, role="equity"))
    jes = [JE("", local_instant(w.period_start, 0, 0, w.tz), open_lines, "opening")]

    events: list[tuple[datetime, int, object]] = []
    for i, inv in enumerate(w.invoices):
        if not inv.opening:
            events.append((inv.issued_at, i, inv))
    for i, p in enumerate(w.payments):
        events.append((p.ts, 10_000 + i, p))
    events.sort(key=lambda e: (e[0], e[1]))

    state = ArState(w)
    for inv in [i for i in w.invoices if i.opening]:
        state.open_invoice(inv)
    for ts, _, ev in events:
        if isinstance(ev, Invoice):
            jes.append(state.post_invoice(ev) if ev.kind == "invoice" else state.post_credit_note(ev))
        elif ev.kind == "receipt":
            jes.append(state.post_receipt(ev))
        else:
            jes.append(state.post_refund(ev))
    w.jes = jes + w.jes
    w.jes.sort(key=lambda j: j.ts)
    for i, je in enumerate(w.jes):
        je.key = f"je{i}"


class ArState:
    """AR subledger in base currency (carried amounts) and in document currency (open amounts)."""

    def __init__(self, w: World):
        self.w = w
        self.carried: dict[int, int] = {}
        self.open_doc: dict[int, int] = {}
        self.excess: dict[int, tuple[int, int]] = {}  # invoice -> (doc excess, base carried) awaiting refund

    def open_invoice(self, inv: Invoice) -> int:
        b = self.w.to_base(inv.total, inv.ccy, inv.issued_at)
        self.carried[id(inv)] = b
        self.open_doc[id(inv)] = inv.total
        return b

    def post_invoice(self, inv: Invoice) -> JE:  # T1
        b = self.open_invoice(inv)
        return JE("", inv.issued_at, [Line(AR, b, 0, inv.cust, inv, role="ar"), Line(REVENUE, 0, b, inv.cust, inv, role="revenue")], "invoice", inv)

    def post_credit_note(self, cn: Invoice) -> JE:  # T7, at the credit note's own date rate
        b = self.w.to_base(cn.total, cn.ccy, cn.issued_at)
        self.carried[id(cn.ref)] -= b
        self.open_doc[id(cn.ref)] -= cn.total
        return JE("", cn.issued_at, [Line(RETURNS, b, 0, cn.cust, cn, role="returns"), Line(AR, 0, b, cn.cust, cn, role="ar")], "credit_note", cn)

    def relief(self, inv: Invoice, alloc_doc: int) -> int:
        """Base amount of AR relieved for an allocation: carried rate; final allocation clears the remainder."""
        if alloc_doc == self.open_doc[id(inv)]:
            return self.carried[id(inv)]
        rate = self.w.rate(self.w.local(inv.issued_at), inv.ccy)
        return convert(alloc_doc, inv.ccy, self.w.base, rate)

    def post_receipt(self, p: Payment) -> JE:  # T2/T3/T4/T5/T6/T11
        w = self.w
        bank = w.to_base(p.amount, p.ccy, p.ts)
        lines = [Line(bank_account(w, p.ccy), bank, 0, p.cust, None, p, role="bank")]
        if p.fee:
            lines.append(Line(BANK_CHARGES, p.fee, 0, p.cust, None, p, role="fee"))
        if not p.refs:
            lines.append(Line(UNAPPLIED, 0, bank, p.cust, None, p, role="unapplied"))
            return JE("", p.ts, lines, "receipt", p)
        remaining = p.amount + p.fee
        credits = 0
        for inv in p.refs:
            alloc = min(remaining, self.open_doc[id(inv)])
            if alloc <= 0:
                continue
            rel = self.relief(inv, alloc)
            self.carried[id(inv)] -= rel
            self.open_doc[id(inv)] -= alloc
            remaining -= alloc
            credits += rel
            lines.append(Line(AR, 0, rel, p.cust, inv, p, role="ar"))
        if remaining > 0:  # overpayment -> refunds payable at the payment rate
            ex = w.to_base(remaining, p.ccy, p.ts)
            self.excess[id(p.refs[0])] = (remaining, ex)
            credits += ex
            lines.append(Line(REFUNDS_PAYABLE, 0, ex, p.cust, p.refs[0], p, role="refund"))
        diff = bank + p.fee - credits
        if diff > 0:
            lines.append(Line(FX_GAIN, 0, diff, p.cust, None, p, role="fx"))
        elif diff < 0:
            lines.append(Line(FX_LOSS, -diff, 0, p.cust, None, p, role="fx"))
        return JE("", p.ts, lines, "receipt", p)

    def post_refund(self, p: Payment) -> JE:  # T6 second leg
        w = self.w
        inv = p.refs[0]
        _, carried = self.excess.pop(id(inv))
        bank = w.to_base(p.amount, p.ccy, p.ts)
        lines = [Line(REFUNDS_PAYABLE, carried, 0, p.cust, inv, p, role="refund"), Line(bank_account(w, p.ccy), 0, bank, p.cust, None, p, role="bank")]
        diff = carried - bank
        if diff > 0:
            lines.append(Line(FX_GAIN, 0, diff, p.cust, None, p, role="fx"))
        elif diff < 0:
            lines.append(Line(FX_LOSS, -diff, 0, p.cust, None, p, role="fx"))
        return JE("", p.ts, lines, "refund", p)
