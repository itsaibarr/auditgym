from decimal import Decimal

import pytest
from hypothesis import given, settings, strategies as st

from auditgym.calc import CalcError, evaluate
from auditgym.money import convert


@pytest.mark.parametrize("expr,val", [("1+2*3", "7"), ("(1+2)*3", "9"), ("10/4", "2.5"), ("-5+2", "-3"), ("478.25*1000", "478250"), ("0.1+0.2", "0.3")])
def test_calc_ok(expr, val):
    assert evaluate(expr) == val


@pytest.mark.parametrize("expr", ["", "x", "1e3", "2**3", "abs(-1)", "1_000", "[1]", "1;2", "lambda: 1", "().__class__", "9" * 300, "1 if 1 else 2", "True+1"])
def test_calc_rejects(expr):
    with pytest.raises(CalcError):
        evaluate(expr)


def test_convert_half_even_and_exponents():
    assert convert(100000, "USD", "KZT", "478.25") == 47825000
    assert convert(5, "USD", "KZT", "0.5") == 2  # 2.5 -> 2 (half even)
    assert convert(15, "USD", "KZT", "0.5") == 8  # 7.5 -> 8
    assert convert(1000, "JPY", "KZT", "3.2155") == 321550  # JPY exponent 0
    assert convert(1000, "KWD", "USD", "3.25") == 325  # KWD exponent 3


@settings(max_examples=200, derandomize=True, database=None)
@given(st.integers(min_value=0, max_value=10**12), st.decimals(min_value=Decimal("0.0001"), max_value=Decimal("1000"), places=4))
def test_convert_is_correctly_rounded(amount, rate):
    got = convert(amount, "USD", "EUR", str(rate))
    exact = Decimal(amount) * rate
    assert abs(Decimal(got) - exact) <= Decimal("0.5")
