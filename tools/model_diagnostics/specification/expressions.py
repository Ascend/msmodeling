# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Safe ShapeExpr / DTypeExpr evaluation without Python eval."""

from __future__ import annotations

import ast
import operator
from typing import Mapping

from tools.model_diagnostics.domain.models import DType, TensorShape
from tools.model_diagnostics.specification.errors import SpecificationLoadError

_BIN_OPS: dict[type[ast.operator], object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}

_UNARY_OPS: dict[type[ast.unaryop], object] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCTIONS = frozenset({"max", "min", "ceil", "abs"})
_UNKNOWN_TOKENS = frozenset({"?", "unknown"})


def _ceil_int(value: float | int) -> int:
    import math

    return int(math.ceil(value))


_FUNCTION_IMPL = {
    "max": max,
    "min": min,
    "ceil": _ceil_int,
    "abs": abs,
}


def _eval_node(node: ast.AST, env: Mapping[str, object]) -> object:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)
    if isinstance(node, ast.Constant):
        if node.value is None:
            return None
        if isinstance(node.value, (int, float, str)) and not isinstance(node.value, bool):
            return node.value
        raise SpecificationLoadError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise SpecificationLoadError(f"unknown expression variable: {node.id}")
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand, env))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)
        if left is None or right is None:
            return None
        if type(node.op) in (ast.Div, ast.FloorDiv, ast.Mod):
            if not isinstance(right, (int, float)) or right == 0:
                raise SpecificationLoadError("division by zero or non-numeric divisor")
        if type(node.op) is ast.Div:
            if isinstance(left, int) and isinstance(right, int):
                if left % right != 0:
                    raise SpecificationLoadError(f"non-integral division in shape expression: {left}/{right}")
                return left // right
        try:
            return _BIN_OPS[type(node.op)](left, right)
        except ZeroDivisionError as error:
            raise SpecificationLoadError("division by zero or non-numeric divisor") from error
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
            raise SpecificationLoadError("only max/min/ceil/abs calls are allowed")
        if node.keywords:
            raise SpecificationLoadError("keyword arguments are not allowed in expressions")
        args = [_eval_node(arg, env) for arg in node.args]
        if any(arg is None for arg in args):
            return None
        return _FUNCTION_IMPL[node.func.id](*args)
    if isinstance(node, ast.List):
        return [_eval_node(element, env) for element in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(element, env) for element in node.elts)
    raise SpecificationLoadError(f"unsupported expression syntax: {type(node).__name__}")


def _parse_expression(expression: str) -> ast.AST:
    try:
        return ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SpecificationLoadError(f"invalid expression syntax: {expression!r}") from exc


def evaluate_shape(expression: str, env: Mapping[str, object]) -> TensorShape | None:
    """Evaluate a shape expression.

    Whole-shape unknown tokens (``?`` / ``unknown``) and any dimension that
    resolves to ``None`` yield ``None``, which Theory records as missing shape
    evidence for later ``INCOMPLETE`` comparison.
    """

    text = expression.strip()
    if text in _UNKNOWN_TOKENS:
        return None
    value = _eval_node(_parse_expression(expression), env)
    if value is None:
        return None
    if isinstance(value, list):
        value = tuple(value)
    if not isinstance(value, tuple) or not value:
        raise SpecificationLoadError(f"shape expression must evaluate to a non-empty tuple: {expression!r}")
    dims: list[int] = []
    for dimension in value:
        if dimension is None:
            return None
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise SpecificationLoadError(f"shape dimensions must be integers: {expression!r}")
        if dimension < 0:
            raise SpecificationLoadError(f"shape dimensions must be non-negative: {expression!r}")
        dims.append(dimension)
    return tuple(dims)


def evaluate_dtype(expression: str, env: Mapping[str, object]) -> DType | None:
    text = expression.strip()
    if text in _UNKNOWN_TOKENS:
        return None
    value = _eval_node(_parse_expression(expression), env)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SpecificationLoadError(f"dtype expression must evaluate to a non-empty string: {expression!r}")
    return value
