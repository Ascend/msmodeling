from __future__ import annotations

import re
from typing import Any

class SafeExprEval:
    """Restricted evaluator for dimension expressions."""

    _ALLOWED_FUNCS = {
        "max": max,
        "min": min,
        "abs": abs,
        "align": lambda x, alg: ((x + alg - 1) // alg) * alg,
    }
    _FORBIDDEN_PATTERN = re.compile(r"(?:\b|[\._ ])__|\.")

    def __init__(self, variables: dict[str, int]):
        self.vars = variables

    def eval(self, expr: str) -> Any:
        if self._FORBIDDEN_PATTERN.search(expr):
            raise ValueError(f"Expression contains forbidden patterns (dots or double underscores): {expr}")
        safe_globals = {"__builtins__": {}}
        safe_globals.update(self._ALLOWED_FUNCS)
        try:
            return eval(expr, safe_globals, self.vars)  # noqa: S307
        except Exception as exc:
            raise ValueError(f"Failed to eval expression '{expr}' with vars {self.vars}: {exc}") from exc

def _split_dims(s: str) -> list[str]:
    result: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in s:
        if ch == "," and depth == 0:
            result.append("".join(buf).strip())
            buf = []
        else:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            buf.append(ch)
    if buf:
        result.append("".join(buf).strip())
    return [r for r in result if r]

def _eval_dim(expr: str, evaluator: SafeExprEval) -> int:
    expr = expr.strip()
    try:
        return int(expr)
    except ValueError:
        return int(evaluator.eval(expr))

def _parse_shape_expr(shape_str: str, evaluator: SafeExprEval) -> tuple[int, ...]:
    s = shape_str.strip()
    if s == "()":
        return ()
    inner = s.lstrip("(").rstrip(")")
    if not inner.strip():
        return ()
    dims = _split_dims(inner)
    return tuple(_eval_dim(d, evaluator) for d in dims)
