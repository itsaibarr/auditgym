"""Calculator action: arithmetic over decimals only. An ast whitelist walker, never eval()."""
from __future__ import annotations

import ast
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext

MAX_LEN = 256
_MAX_ABS = Decimal(10) ** 40


class CalcError(ValueError):
    pass


def evaluate(expression: str) -> str:
    if not isinstance(expression, str) or not expression.strip() or len(expression) > MAX_LEN:
        raise CalcError("expression must be a non-empty string of at most 256 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise CalcError("syntax error") from exc
    with localcontext() as ctx:
        ctx.prec = 40
        ctx.traps[DivisionByZero] = True
        ctx.traps[InvalidOperation] = True
        try:
            value = _eval(tree.body, expression, 0)
        except (DivisionByZero, InvalidOperation, ZeroDivisionError) as exc:
            raise CalcError("arithmetic error") from exc
        value = value.normalize() if value == value.to_integral_value() else value
    text = format(value, "f")
    return text


def _eval(node: ast.AST, src: str, depth: int) -> Decimal:
    if depth > 50:
        raise CalcError("expression too deep")
    if isinstance(node, ast.Constant) and type(node.value) in (int, float) and not isinstance(node.value, bool):
        literal = ast.get_source_segment(src, node)
        if literal is None or not all(c.isdigit() or c in "._" for c in literal) or "_" in literal:
            raise CalcError("only plain decimal literals are allowed")
        return Decimal(literal)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval(node.operand, src, depth + 1)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        a = _eval(node.left, src, depth + 1)
        b = _eval(node.right, src, depth + 1)
        if isinstance(node.op, ast.Add):
            r = a + b
        elif isinstance(node.op, ast.Sub):
            r = a - b
        elif isinstance(node.op, ast.Mult):
            r = a * b
        else:
            if b == 0:
                raise CalcError("division by zero")
            r = a / b
        if abs(r) > _MAX_ABS:
            raise CalcError("result out of range")
        return r
    raise CalcError("only + - * / ( ) and decimal numbers are allowed")
