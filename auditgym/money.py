"""Money and FX: integer minor units, ISO 4217 exponents, one shared conversion."""
from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal, localcontext

# ISO 4217 List One (SIX, published 2026-09-17) minor-unit exponents, pinned subset.
EXPONENTS: dict[str, int] = {"KZT": 2, "USD": 2, "EUR": 2, "CNY": 2, "GBP": 2, "JPY": 0, "KWD": 3}
ISO4217_LIST_DATE = "2026-09-17"
ROUNDING = "ROUND_HALF_EVEN"


def exponent(ccy: str) -> int:
    return EXPONENTS[ccy]


def convert(amount_minor: int, src: str, base: str, rate: str | None) -> int:
    """base_minor = round_half_even(amount_minor * rate * 10**(exp(base) - exp(src))).

    `rate` is a decimal string: units of `base` per 1 unit of `src`. Same currency needs no rate.
    """
    if src == base:
        return amount_minor
    if rate is None:
        raise ValueError("rate required for cross-currency conversion")
    with localcontext() as ctx:
        ctx.prec = 50
        value = Decimal(amount_minor) * Decimal(rate) * (Decimal(10) ** (exponent(base) - exponent(src)))
        return int(value.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def fmt(amount_minor: int, ccy: str) -> str:
    e = exponent(ccy)
    sign = "-" if amount_minor < 0 else ""
    a = abs(amount_minor)
    if e == 0:
        return f"{sign}{a} {ccy}"
    return f"{sign}{a // 10**e}.{a % 10**e:0{e}d} {ccy}"
