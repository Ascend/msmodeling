# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Same-config comparison between generated and captured operation sequences.

The comparison implements the engine's D1 acceptance caliber: for two
flattened operation lists of equal length, verify per op that ``func`` matches
and that every nested tensor in ``args`` / ``kwargs`` / ``out`` matches on the
four-tuple **(shape, dtype, stride, requires_grad)**.  Non-tensor leaves are
compared exactly.

Region marker ops carry namespace-specific IDs (real captures hold raw
allocator addresses, template expansions hold the 1000000+ remap range), so
both sides are normalized to canonical ids before comparing: begin ids map to
their first-appearance order and end/copy markers inherit the canonical id of
the begin they reference.  Two sequences with identical reference structure
therefore end up with identical canonical ids, while genuine reference errors
(dangling copies, mispaired ends) still surface as diffs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from template_generate import _region_arg_index, _region_kind, _replace_sequence_item
from tensor_cast.performance_model.op_invoke_info import OpInvokeInfo


@dataclass
class OpDiff:
    """One concrete mismatch between expected and generated values."""

    op_index: int
    func: str
    path: str
    kind: str
    expected: Any
    actual: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "op_index": self.op_index,
            "func": self.func,
            "path": self.path,
            "kind": self.kind,
            "expected": _short(self.expected),
            "actual": _short(self.actual),
        }


@dataclass
class CompareReport:
    """Aggregated result of one same-config comparison."""

    expected_ops: int = 0
    generated_ops: int = 0
    length_match: bool = False
    func_mismatches: int = 0
    shape_mismatches: int = 0
    dtype_mismatches: int = 0
    stride_mismatches: int = 0
    requires_grad_mismatches: int = 0
    scalar_mismatches: int = 0
    structure_mismatches: int = 0
    alias_mismatches: int = 0
    diffs: list[OpDiff] = field(default_factory=list)

    @property
    def total_mismatches(self) -> int:
        return (
            self.func_mismatches
            + self.shape_mismatches
            + self.dtype_mismatches
            + self.stride_mismatches
            + self.requires_grad_mismatches
            + self.scalar_mismatches
            + self.structure_mismatches
            + self.alias_mismatches
        )

    @property
    def passed(self) -> bool:
        return self.length_match and self.total_mismatches == 0

    def as_dict(self, max_diffs: int = 20) -> dict[str, Any]:
        return {
            "expected_ops": self.expected_ops,
            "generated_ops": self.generated_ops,
            "length_match": self.length_match,
            "passed": self.passed,
            "mismatches": {
                "func": self.func_mismatches,
                "shape": self.shape_mismatches,
                "dtype": self.dtype_mismatches,
                "stride": self.stride_mismatches,
                "requires_grad": self.requires_grad_mismatches,
                "scalar": self.scalar_mismatches,
                "structure": self.structure_mismatches,
                "alias": self.alias_mismatches,
                "total": self.total_mismatches,
            },
            "first_diffs": [diff.as_dict() for diff in self.diffs[:max_diffs]],
        }


def _short(value: Any, limit: int = 6) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        shown = [_short(item, limit) for item in value[:limit]]
        return {"__type__": type(value).__name__, "items": shown, "len": len(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _tensor_quad(tensor: torch.Tensor) -> tuple[tuple[int, ...], str, tuple[int, ...], bool]:
    return (
        tuple(int(dimension) for dimension in tensor.shape),
        str(tensor.dtype),
        tuple(int(stride) for stride in tensor.stride()),
        bool(tensor.requires_grad),
    )


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    shape, dtype, stride, requires_grad = _tensor_quad(tensor)
    return {"shape": list(shape), "dtype": dtype, "stride": list(stride), "requires_grad": requires_grad}


def _to_opinvoke(op: Any) -> OpInvokeInfo:
    """Accept CapturedOp / OpInvokeInfo / mapping uniformly."""

    if isinstance(op, OpInvokeInfo):
        return op
    func = op.func if hasattr(op, "func") else op["func"]
    args = op.args if hasattr(op, "args") else op.get("args", ())
    kwargs = op.kwargs if hasattr(op, "kwargs") else op.get("kwargs", {})
    out = op.out if hasattr(op, "out") else op.get("out")
    if not isinstance(func, str):
        return OpInvokeInfo(func, args, kwargs, out)
    from template_generate import _restore_op

    return OpInvokeInfo(_restore_op(func), args, kwargs, out)


def _normalise_region_ids(ops: Sequence[OpInvokeInfo]) -> list[OpInvokeInfo]:
    """Return a copy whose region-marker IDs are canonically re-keyed.

    Region ids are namespace-specific (real captures carry raw allocator
    addresses, template expansions carry the 1000000+ remap range).  Ids are
    therefore compared by ROLE, not value: each side's begin ids are mapped to
    their first-appearance order (0, 1, 2, ...) and end/copy markers inherit
    the canonical id of the begin they reference.  Unpaired or dangling
    markers keep a distinctive canonical slot so genuine reference errors
    still surface as diffs (the old shape-based reassignment masked those by
    clobbering both sides to the same last-closed region).
    """

    from template_generate import _region_arg_index, _region_kind

    cloned: list[OpInvokeInfo] = []
    for op in ops:
        # Only the top-level region argument is replaced below. Tensor and
        # nested payloads stay read-only, so copying them is unnecessary.
        cloned.append(OpInvokeInfo(op.func, op.args, op.kwargs, op.out))

    canonical: dict[Any, int] = {}
    dangling = -1

    def canonical_id(raw: Any) -> int:
        nonlocal dangling
        if raw in canonical:
            return canonical[raw]
        canonical[raw] = dangling
        dangling -= 1
        return canonical[raw]

    for op in cloned:
        kind = _region_kind(op.func)
        if kind is None or not isinstance(op.args, (list, tuple)) or not op.args:
            continue
        arg_index = _region_arg_index(op.args)
        if arg_index is None:
            continue
        op.args = _replace_sequence_item(op.args, arg_index, canonical_id(op.args[arg_index]))
    return cloned


def _func_label(op: OpInvokeInfo) -> str:
    return str(op.func)


def _compare_tree(
    expected: Any,
    actual: Any,
    path: str,
    report: CompareReport,
    op_index: int,
    func: str,
    keep_diffs: bool,
    max_diffs: int,
) -> None:
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        expected_quad = _tensor_quad(expected)
        actual_quad = _tensor_quad(actual)
        for position, name in enumerate(("shape", "dtype", "stride", "requires_grad")):
            if expected_quad[position] != actual_quad[position]:
                attr = {
                    "shape": "shape_mismatches",
                    "dtype": "dtype_mismatches",
                    "stride": "stride_mismatches",
                    "requires_grad": "requires_grad_mismatches",
                }[name]
                setattr(report, attr, getattr(report, attr) + 1)
                if keep_diffs and len(report.diffs) < max_diffs:
                    report.diffs.append(
                        OpDiff(op_index, func, path, name, _tensor_summary(expected), _tensor_summary(actual))
                    )
        return
    if isinstance(expected, torch.Tensor) or isinstance(actual, torch.Tensor):
        report.structure_mismatches += 1
        if keep_diffs and len(report.diffs) < max_diffs:
            report.diffs.append(OpDiff(op_index, func, path, "tensorness", _short(expected), _short(actual)))
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            report.structure_mismatches += 1
            if keep_diffs and len(report.diffs) < max_diffs:
                report.diffs.append(
                    OpDiff(op_index, func, path, "dict_keys", sorted(map(str, expected)), sorted(map(str, actual)))
                )
            return
        for key in expected:
            _compare_tree(expected[key], actual[key], f"{path}.{key}", report, op_index, func, keep_diffs, max_diffs)
        return
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            report.structure_mismatches += 1
            if keep_diffs and len(report.diffs) < max_diffs:
                report.diffs.append(OpDiff(op_index, func, path, "seq_length", len(expected), len(actual)))
            return
        if type(expected) is not type(actual) and not (isinstance(expected, list) and isinstance(actual, list)):
            # list/tuple mixing is tolerated only when the capture side is
            # ambiguous; flag it so callers can tighten the caliber later.
            report.structure_mismatches += 1
            if keep_diffs and len(report.diffs) < max_diffs:
                report.diffs.append(
                    OpDiff(op_index, func, path, "seq_type", type(expected).__name__, type(actual).__name__)
                )
            return
        for index, (item_expected, item_actual) in enumerate(zip(expected, actual)):
            _compare_tree(item_expected, item_actual, f"{path}[{index}]", report, op_index, func, keep_diffs, max_diffs)
        return
    if type(expected) is not type(actual):
        report.structure_mismatches += 1
        if keep_diffs and len(report.diffs) < max_diffs:
            report.diffs.append(
                OpDiff(op_index, func, path, "leaf_type", type(expected).__name__, type(actual).__name__)
            )
        return
    if expected != actual:
        if isinstance(expected, (bool, int, float, str)):
            report.scalar_mismatches += 1
            if keep_diffs and len(report.diffs) < max_diffs:
                report.diffs.append(OpDiff(op_index, func, path, "scalar", expected, actual))
        else:
            report.structure_mismatches += 1
            if keep_diffs and len(report.diffs) < max_diffs:
                report.diffs.append(OpDiff(op_index, func, path, "value", _short(expected), _short(actual)))


def tensor_alias_paths(ops: Iterable[Any]) -> dict[tuple[int, str], tuple[int, str]]:
    """Map every tensor occurrence to its first occurrence in the graph.

    Object identities differ between captures; occurrence paths do not.  This
    detects both broken producer/consumer links and accidentally merged tensors.
    Storage sharing between distinct tensor objects is outside this contract.
    """
    first: dict[int, tuple[int, str]] = {}
    paths: dict[tuple[int, str], tuple[int, str]] = {}

    def visit(value: Any, op_index: int, path: str) -> None:
        if isinstance(value, torch.Tensor):
            occurrence = (op_index, path)
            paths[occurrence] = first.setdefault(id(value), occurrence)
        elif isinstance(value, Mapping):
            for key in sorted(value, key=str):
                visit(value[key], op_index, f"{path}[{key!r}]")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, op_index, f"{path}[{index}]")

    for index, op in enumerate(ops):
        for name in ("args", "kwargs", "out"):
            value = op.get(name) if isinstance(op, Mapping) else getattr(op, name)
            visit(value, index, name)
    return paths


def compare_op_sequences(
    expected_ops: Iterable[Any],
    generated_ops: Iterable[Any],
    *,
    max_diffs: int = 50,
) -> CompareReport:
    """Compare two flattened op sequences and return a :class:`CompareReport`.

    Region-marker IDs are normalized on both sides before comparison, so the
    identity-based IDs from a live capture never cause false mismatches.
    """

    expected_list = [_to_opinvoke(op) for op in expected_ops]
    generated_list = [_to_opinvoke(op) for op in generated_ops]

    report = CompareReport(expected_ops=len(expected_list), generated_ops=len(generated_list))
    report.length_match = len(expected_list) == len(generated_list)
    if not report.length_match:
        return report

    expected_aliases = tensor_alias_paths(expected_list)
    generated_aliases = tensor_alias_paths(generated_list)
    for occurrence in sorted(expected_aliases.keys() & generated_aliases.keys()):
        if expected_aliases[occurrence] != generated_aliases[occurrence]:
            report.alias_mismatches += 1
            if len(report.diffs) < max_diffs:
                index, path = occurrence
                report.diffs.append(
                    OpDiff(
                        index,
                        _func_label(expected_list[index]),
                        path,
                        "alias",
                        expected_aliases[occurrence],
                        generated_aliases[occurrence],
                    )
                )

    expected_norm = _normalise_region_ids(expected_list)
    generated_norm = _normalise_region_ids(generated_list)

    for index, (expected_op, generated_op) in enumerate(zip(expected_norm, generated_norm)):
        keep_diffs = len(report.diffs) < max_diffs
        expected_func = _func_label(expected_op)
        generated_func = _func_label(generated_op)
        if expected_func != generated_func:
            report.func_mismatches += 1
            if keep_diffs and len(report.diffs) < max_diffs:
                report.diffs.append(OpDiff(index, expected_func, "func", "func", expected_func, generated_func))
            continue
        _compare_tree(expected_op.args, generated_op.args, "args", report, index, expected_func, keep_diffs, max_diffs)
        _compare_tree(
            expected_op.kwargs, generated_op.kwargs, "kwargs", report, index, expected_func, keep_diffs, max_diffs
        )
        _compare_tree(expected_op.out, generated_op.out, "out", report, index, expected_func, keep_diffs, max_diffs)
    return report


def region_marker_counts(ops: Iterable[Any]) -> dict[str, int]:
    """Count region markers and total ops under the fixed caliber."""

    total = 0
    begins = ends = copies = 0
    for op in ops:
        total += 1
        kind = _region_kind(getattr(op, "func", op.get("func") if isinstance(op, dict) else ""))
        if kind == "begin":
            begins += 1
        elif kind == "end":
            ends += 1
        elif kind == "copy":
            copies += 1
    return {
        "total_ops": total,
        "region_begin": begins,
        "region_end": ends,
        "region_copy": copies,
        "non_marker_ops": total - begins - ends - copies,
    }


def first_region_id_args(ops: Iterable[Any]) -> list[int]:
    """Debug helper: list region-marker id positions (arg index) per op."""

    positions: list[int] = []
    for op in ops:
        func = getattr(op, "func", op.get("func") if isinstance(op, dict) else "")
        if _region_kind(func) is None:
            continue
        args = getattr(op, "args", op.get("args") if isinstance(op, dict) else ())
        positions.append(_region_arg_index(args) if _region_arg_index(args) is not None else -1)
    return positions


def compare_performance_ops(expected_ops, generated_ops, *, max_diffs=20):
    """Keep strict diagnostics, but do not reject simulation graphs for stride."""
    report = compare_op_sequences(expected_ops, generated_ops, max_diffs=max_diffs)
    result = report.as_dict(max_diffs=max_diffs)
    result['strict_passed'] = report.passed
    result['acceptance'] = 'performance_v1_ignore_stride'
    result['passed'] = report.length_match and report.total_mismatches == report.stride_mismatches
    return result


__all__ = [
    "CompareReport",
    "OpDiff",
    "compare_op_sequences",
    "compare_performance_ops",
    "first_region_id_args",
    "region_marker_counts",
    "tensor_alias_paths",
]
