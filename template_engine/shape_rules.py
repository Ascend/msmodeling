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

"""Semantic shape derivation for template-engine ops (exact, zero statistics).

The template builder fits per-dimension formulas across capture anchors and
degrades every dimension it cannot explain to ``EXACT_LOOKUP``.  Many of those
degradations are not statistical at all: the output shape follows
deterministically from the input shapes plus the op's own arguments.  This
module derives those shapes exactly-by-construction so the stat-fit caller can
consult it first and fall through only when no rule applies.

Public entry point::

    derive_shape_semantically(op, input_shapes=None, config=None) -> dict | None

``op`` is a captured op record (``template_engine.capture_ops.CapturedOp`` or
anything exposing ``func`` / ``args`` / ``kwargs``; ``out`` is never read for
derivation - rules are pure functions of inputs + arguments).  ``config`` is
the parsed cache-key config mapping; current rules do not need it but the
signature is fixed so callers can pass it uniformly.

The return value is ``None`` when no rule applies (the caller should fall
through to statistical fitting) or a dict of output dims::

    {"out": <out tree>}

where ``<out tree>`` mirrors the structure of the op's output:

* a single tensor output      -> ``{"shape": [dim, ...]}``
* a list/tuple of tensor outs -> a list/tuple of the same per-tensor dicts
* a dict of tensor outs       -> a dict mapping the same keys to the dicts

Every dim is an ``int`` or a symbolic token (see `Mask classes`).  A dim is
returned symbolically only when it is propagated verbatim from an input dim;
any rule that would have to do arithmetic on a symbolic value declines (returns
``None``) instead of guessing.

Mask classes
------------

Region-id scalars use the capture-time placeholder convention (ids allocated
from 1_000_000 upward, see ``template_generate._reassign_region_ids``), and
runtime-opaque integers (allocator/pointer-like, above ``2**40``) are masked to
the ``"<rt>"`` token by :func:`mask_scalar` (this module owns the caliber).
The graph comparator checks non-tensor leaves exactly. Opaque values in
shapes they are treated as symbolic: propagated verbatim when the shape
structure carries them through, and never used as numbers.

Alignment of ``input_shapes``
-----------------------------

``input_shapes`` optionally carries the caller's view of the input shapes.  It
must align 1:1 with `tensor_input_shapes(op)` - the depth-first left-to-right
order of tensor leaves in ``op.args`` - entry-wise as either a shape iterable
or a ``(shape, stride)`` pair.  When the lengths disagree, the module recomputes
shapes from the captured args so a rule can never run against misaligned data.

Architecture (reading guide)
----------------------------
Owns the semantic shape RULES (exact-by-construction) and the runtime-int
masking caliber.  Input: one captured op (+ optional aligned input shapes);
output: ``{"out": <out tree>}`` of int/symbolic dims, or ``None`` ("no rule
applies - fall through to fitting").

Entry points:

- ``derive_shape_semantically`` (L815) — the rule dispatcher (the only entry
  the builder / stat-fit caller needs)
- ``_rule_*`` functions         — one rule per op family (cat, view, ...)
- ``mask_scalar`` (L102) / ``is_symbolic`` (L110) — the ``"<rt>"`` masking
  caliber (canonical owner; the repeat-folder comparator lets integer leaves
  drift instead)
- ``validate_shape_rules`` (L886) — the self-check harness

Read this if you want to understand how an output shape was derived without
statistics, and where the ``"<rt>"`` / region-id symbolic tokens come from.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard repo dependency
    torch = None  # type: ignore[assignment]


#: Integers above this bound are allocator/pointer-like runtime values and are
#: masked to the ``"<rt>"`` token (shared with capture normalization).
RUNTIME_INT_MASK = 2**40

#: Capture-time region-id placeholder allocation base
#: (``template_generate._reassign_region_ids``).
REGION_ID_PLACEHOLDER_BASE = 1_000_000

#: Symbolic token for runtime-opaque integers.
RT_SYMBOL = "<rt>"

_INT64_MAX = 9223372036854775807


def mask_scalar(value: Any) -> Any:
    """Mask runtime-opaque integers to ``"<rt>"``; pass everything else."""

    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > RUNTIME_INT_MASK:
        return RT_SYMBOL
    return value


def is_symbolic(value: Any) -> bool:
    """Whether a dim value must be treated as symbolic rather than numeric.

    Symbolic means: non-integer tokens (``"<rt>"``, ``"<region-id>"``, ...) or
    integers in the region-id placeholder range (>= 1_000_000) or above the
    runtime mask (``2**40``).
    """

    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= REGION_ID_PLACEHOLDER_BASE or abs(value) > RUNTIME_INT_MASK
    return True


def tensor_input_shapes(op: Any) -> list[tuple[int, ...]]:
    """Shapes of the tensor leaves of ``op.args`` in depth-first order."""

    return [tuple(int(dim) for dim in tensor.shape) for tensor in _iter_tensor_leaves(op.args)]


def _iter_tensor_leaves(value: Any) -> Iterable[Any]:
    if _looks_like_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _iter_tensor_leaves(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensor_leaves(item)


def _looks_like_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _rule_name(func: Any) -> str:
    return str(func)


def _arg(op: Any, index: int, name: str, default: Any = None) -> Any:
    """Schema-style argument fetch: positional first, then kwargs."""

    args = op.args if isinstance(op.args, (list, tuple)) else ()
    if index < len(args):
        return args[index]
    kwargs = op.kwargs if isinstance(op.kwargs, Mapping) else {}
    return kwargs.get(name, default)


class _Spec:
    """Lightweight tensor stand-in: exactly what the rules need."""

    __slots__ = ("shape", "stride", "dtype", "itemsize")

    def __init__(
        self,
        shape: Sequence[Any],
        stride: Sequence[int] | None = None,
        dtype: Any = None,
        itemsize: int | None = None,
    ):
        self.shape = list(shape)
        self.stride = list(stride) if stride is not None else None
        self.dtype = dtype
        self.itemsize = itemsize


def _resolve_specs(op: Any, input_shapes: Any) -> list[_Spec]:
    """Build per-leaf specs, preferring caller-provided aligned shapes."""

    leaves = list(_iter_tensor_leaves(op.args))
    specs: list[_Spec] = []
    aligned: list[Any] | None = None
    if isinstance(input_shapes, (list, tuple)) and len(input_shapes) == len(leaves):
        aligned = list(input_shapes)
    for index, tensor in enumerate(leaves):
        entry = aligned[index] if aligned is not None else tensor
        shape, stride, dtype, itemsize = _spec_entry(entry, tensor)
        specs.append(_Spec(shape, stride, dtype, itemsize))
    return specs


def _spec_entry(entry: Any, tensor: Any) -> tuple[list[Any], list[int] | None, Any, int | None]:
    if _looks_like_tensor(entry):
        return (
            _dim_list(entry.shape),
            [int(stride) for stride in entry.stride()],
            entry.dtype,
            _tensor_itemsize(entry),
        )
    if (
        isinstance(entry, tuple)
        and len(entry) == 2
        and isinstance(entry[0], (list, tuple))
        and isinstance(entry[1], (list, tuple))
    ):
        shape, stride = entry
        return (_dim_list(shape), [int(stride) for stride in stride], getattr(tensor, "dtype", None), None)
    if isinstance(entry, (list, tuple)):
        return (_dim_list(entry), None, getattr(tensor, "dtype", None), None)
    raise TypeError(f"unsupported input_shapes entry: {entry!r}")


def _tensor_itemsize(tensor: Any) -> int | None:
    """Byte size of one stored element, when the real tensor exposes it."""

    element_size = getattr(tensor, "element_size", None)
    if not callable(element_size):
        return None
    try:
        return int(element_size())
    except Exception:  # noqa: BLE001 - meta/symbolic tensors may refuse
        return None


def _dim_list(shape: Any) -> list[Any]:
    return [mask_scalar(int(dim)) if isinstance(dim, int) and not isinstance(dim, bool) else dim for dim in shape]


def _tensor_shape(spec: _Spec) -> list[Any]:
    return spec.shape


def _num_dims(spec: _Spec) -> int:
    return len(spec.shape)


def _normalize_dim(dim: Any, ndim: int, extra: int = 0) -> int | None:
    """Normalize a possibly-negative dimension index; None when invalid."""

    if isinstance(dim, bool) or not isinstance(dim, int):
        return None
    limit = ndim + extra
    if dim < -limit or dim >= limit:
        return None
    return dim + limit if dim < 0 else dim


# --------------------------------------------------------------------------------------
# Rule: cat (aten.cat, tensor_cast.cat)
# --------------------------------------------------------------------------------------

CAT_FUNCS = frozenset({"aten.cat.default", "tensor_cast.cat.default"})


def _rule_cat(op: Any, specs: list[_Spec]) -> dict[str, Any] | None:
    tensors = _arg(op, 0, "tensors")
    if not isinstance(tensors, (list, tuple)) or not tensors:
        return None
    dim = _arg(op, 1, "dim", 0)
    inputs = specs[: len(tensors)]
    if len(inputs) != len(tensors):
        return None
    first = inputs[0]
    ndim = _num_dims(first)
    if any(_num_dims(item) != ndim for item in inputs[1:]):
        return None
    dim = _normalize_dim(dim, ndim)
    if dim is None:
        return None
    cat_size = 0
    for item in inputs:
        size = item.shape[dim]
        if is_symbolic(size):
            return None
        cat_size += int(size)
    # Non-concat dims must agree across inputs (they do in every valid cat).
    for position in range(ndim):
        if position == dim:
            continue
        reference = first.shape[position]
        for item in inputs[1:]:
            if _norm_dim(item.shape[position]) != _norm_dim(reference):
                return None
    out_shape = list(first.shape)
    out_shape[dim] = cat_size
    return {"out": {"shape": out_shape}}


# --------------------------------------------------------------------------------------
# Rule: split_with_sizes
# --------------------------------------------------------------------------------------

SPLIT_FUNCS = frozenset({"aten.split_with_sizes.default", "aten.split.sizes", "aten.split_with_sizes"})


def _rule_split(op: Any, specs: list[_Spec]) -> dict[str, Any] | None:
    if not specs:
        return None
    sizes = _arg(op, 1, "split_sizes")
    if not isinstance(sizes, (list, tuple)):
        return None
    dim = _arg(op, 2, "dim", 0)
    ndim = _num_dims(specs[0])
    dim = _normalize_dim(dim, ndim)
    if dim is None:
        return None
    out: list[dict[str, Any]] = []
    for size in sizes:
        if is_symbolic(size):
            return None
        piece = list(specs[0].shape)
        piece[dim] = int(size)
        out.append({"shape": piece})
    return {"out": out}


# --------------------------------------------------------------------------------------
# Rule: all_to_all (tensor_cast.all_to_all)
# --------------------------------------------------------------------------------------

ALL_TO_ALL_FUNCS = frozenset({"tensor_cast.all_to_all.default"})


def _rule_all_to_all(op: Any, specs: list[_Spec]) -> dict[str, Any] | None:
    if not specs:
        return None
    output_sizes = _arg(op, 1, "output_split_sizes")
    if not isinstance(output_sizes, (list, tuple)):
        return None
    total = 0
    for size in output_sizes:
        if is_symbolic(size):
            return None
        total += int(size)
    shape = list(specs[0].shape)
    if not shape:
        return None
    shape[0] = total
    return {"out": {"shape": shape}}


# --------------------------------------------------------------------------------------
# Rules: view / reshape (-1 inference), expand / broadcast_to, squeeze
# --------------------------------------------------------------------------------------

VIEW_FUNCS = frozenset({"aten.view.default", "aten.reshape.default", "aten.reshape"})


def _rule_view(op: Any, specs: list[_Spec]) -> dict[str, Any] | None:
    if not specs:
        return None
    size = _arg(op, 1, "size")
    if not isinstance(size, (list, tuple)):
        return None
    target = [_norm_dim(item) for item in size]
    if any(item is None for item in target):
        return None
    minus_positions = [index for index, item in enumerate(target) if item == -1]
    if len(minus_positions) > 1:
        return None
    if minus_positions:
        # Infer the -1 from the input numel: only possible when every input
        # dim and every other target dim is numeric.
        known = 1
        for item in target:
            if item == -1:
                continue
            if is_symbolic(item):
                return None
            known *= int(item)
        if known == 0:
            return None
        numel = 1
        for dim in specs[0].shape:
            if is_symbolic(dim):
                return None
            numel *= int(dim)
        if numel % known != 0:
            return None
        target[minus_positions[0]] = numel // known
    return {"out": {"shape": target}}


EXPAND_FUNCS = frozenset(
    {
        "aten.expand.default",
        "aten.expand",
        "aten.broadcast_to.default",
        "aten.broadcast_to",
    }
)


def _rule_expand(op: Any, specs: list[_Spec]) -> dict[str, Any] | None:
    if not specs:
        return None
    size = _arg(op, 1, "size")
    if not isinstance(size, (list, tuple)):
        return None
    in_shape = specs[0].shape
    if len(size) < len(in_shape):
        return None
    offset = len(size) - len(in_shape)
    out: list[Any] = []
    for index, item in enumerate(size):
        item = _norm_dim(item)
        if item is None:
            return None
        if item == -1:
            if index < offset:
                return None
            out.append(in_shape[index - offset])
            continue
        if index >= offset:
            source = in_shape[index - offset]
            if is_symbolic(source):
                # torch requires source == 1 or source == item; unverifiable.
                return None
            source = int(source)
            if source == item or source == 1:
                out.append(item)
                continue
            return None
        # A brand-new leading dim: broadcast_to semantics require it to be
        # produced from nothing (expand of a prepended dim = plain new dim).
        out.append(item)
    return {"out": {"shape": out}}


SQUEEZE_FUNCS = frozenset({"aten.squeeze.default", "aten.squeeze.dim", "aten.squeeze.int", "aten.squeeze"})


def _rule_squeeze(op: Any, specs: list[_Spec]) -> dict[str, Any] | None:
    if not specs:
        return None
    in_shape = specs[0].shape
    dim = _arg(op, 1, "dim", None)
    if dim is None:
        # Squeeze every size-1 dim: only decidable when all dims are numeric.
        if any(is_symbolic(item) for item in in_shape):
            return None
        return {"out": {"shape": [item for item in in_shape if int(item) != 1]}}
    dim = _normalize_dim(dim, len(in_shape))
    if dim is None:
        return None
    size = in_shape[dim]
    if is_symbolic(size):
        return None
    out = list(in_shape)
    if size == 1:
        del out[dim]
    return {"out": {"shape": out}}


# --------------------------------------------------------------------------------------
# Rules: permute / transpose, unsqueeze, select, slice
# --------------------------------------------------------------------------------------

PERMUTE_FUNCS = frozenset({"aten.permute.default", "aten.permute"})


def _rule_permute(op: Any, specs: list[_Spec]) -> dict[str, Any] | None:
    if not specs:
        return None
    perm = _arg(op, 1, "dims")
    if not isinstance(perm, (list, tuple)):
        return None
    in_shape = specs[0].shape
    if len(perm) != len(in_shape):
        return None
    normalized: list[int] = []
    seen: set[int] = set()
    for item in perm:
        item = _normalize_dim(item, len(in_shape))
        if item is None or item in seen:
            return None
        seen.add(item)
        normalized.append(item)
    return {"out": {"shape": [in_shape[index] for index in normalized]}}


TRANSPOSE_FUNCS = frozenset({"aten.transpose.dim", "aten.transpose.int", "aten.transpose"})


def _rule_transpose(op: Any, specs: list[_Spec]) -> dict[str, Any] | None:
    if not specs:
        return None
    dim0 = _normalize_dim(_arg(op, 1, "dim0"), _num_dims(specs[0]))
    dim1 = _normalize_dim(_arg(op, 2, "dim1"), _num_dims(specs[0]))
    if dim0 is None or dim1 is None:
        return None
    out = list(specs[0].shape)
    out[dim0], out[dim1] = out[dim1], out[dim0]
    return {"out": {"shape": out}}


UNSQUEEZE_FUNCS = frozenset({"aten.unsqueeze.default", "aten.unsqueeze"})


def _rule_unsqueeze(op: Any, specs: list[_Spec]) -> dict[str, Any] | None:
    if not specs:
        return None
    dim = _normalize_dim(_arg(op, 1, "dim"), _num_dims(specs[0]) + 1)
    if dim is None:
        return None
    out = list(specs[0].shape)
    out.insert(dim, 1)
    return {"out": {"shape": out}}


SELECT_FUNCS = frozenset({"aten.select.int", "aten.select.Tensor", "aten.select.default", "aten.select"})


def _rule_select(op: Any, specs: list[_Spec]) -> dict[str, Any] | None:
    if not specs:
        return None
    dim = _normalize_dim(_arg(op, 1, "dim"), _num_dims(specs[0]))
    if dim is None:
        return None
    out = list(specs[0].shape)
    del out[dim]
    return {"out": {"shape": out}}


SLICE_FUNCS = frozenset({"aten.slice.Tensor", "aten.slice.int", "aten.slice.t"})


def _slice_length(size: int, start: Any, end: Any, step: Any) -> int | None:
    """PyTorch ``slice`` dim length for one numeric dimension."""

    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        return None
    start = 0 if start is None else int(start)
    end = size if end is None else int(end)
    if start < 0:
        start += size
    if end < 0:
        end += size
    start = max(0, min(start, size))
    end = max(0, min(end, size))
    if end <= start:
        return 0
    return (end - start + step - 1) // step


def _slice_bound(value: Any) -> Any:
    """Normalise slice bounds: the INT64_MAX sentinel means unbounded (None)."""

    if isinstance(value, int) and not isinstance(value, bool) and value == _INT64_MAX:
        return None
    return value


def _rule_slice(op: Any, specs: list[_Spec]) -> dict[str, Any] | None:
    if not specs:
        return None
    dim = _normalize_dim(_arg(op, 1, "dim", 0), _num_dims(specs[0]))
    if dim is None:
        return None
    start = _arg(op, 2, "start")
    end = _slice_bound(_arg(op, 3, "end"))
    step = _arg(op, 4, "step", 1)
    size = specs[0].shape[dim]
    if is_symbolic(size) or is_symbolic(step):
        return None
    # None bounds are legitimate ("to the ends"); everything else must be a
    # plain integer - symbolic tokens cannot be arithmetically clamped.
    if start is not None and start != _INT64_MAX and is_symbolic(start):
        return None
    if end is not None and is_symbolic(end):
        return None
    length = _slice_length(int(size), start, end, step)
    if length is None:
        return None
    out = list(specs[0].shape)
    out[dim] = length
    return {"out": {"shape": out}}


# --------------------------------------------------------------------------------------
# Rule: grouped_matmul family (per-group M accumulation, one concat output)
# --------------------------------------------------------------------------------------

#: Plain GMM variants: out = (sum_i x_i.shape[0], w0.shape[1]).
GMM_PLAIN_FUNCS = frozenset(
    {
        "tensor_cast.grouped_matmul.default",
        "tensor_cast.grouped_matmul_quant.default",
        "tensor_cast.grouped_matmul_quant_int4.default",
        "tensor_cast.grouped_matmul_fp8.default",
        "tensor_cast.grouped_matmul_mxfp4.default",
    }
)

#: Unpacked SwiGLU variants: out = (sum_i x_i.shape[0], w0.shape[1] // 2).
GMM_SWIGLU_FUNCS = frozenset(
    {
        "tensor_cast.grouped_matmul_swiglu.default",
        "tensor_cast.grouped_matmul_quant_swiglu.default",
        "tensor_cast.grouped_matmul_fp8_swiglu.default",
    }
)

#: Packed 4-bit SwiGLU variants (gate/up width decoded from the packed weight).
GMM_SWIGLU_PACKED_FUNCS = frozenset(
    {
        "tensor_cast.grouped_matmul_quant_int4_swiglu.default",
        "tensor_cast.grouped_matmul_mxfp4_swiglu.default",
    }
)

#: GMM + SwiGLU + post-quant: out = ((M, N // 2), (M, ceil(N / 2 / group_size))).
GMM_SWIGLU_QUANT_FUNCS = frozenset({"tensor_cast.grouped_matmul_mxfp4_swiglu_quant.default"})


def _gmm_group_tensors(op: Any, specs: list[_Spec]) -> tuple[list[_Spec], list[_Spec]] | None:
    """Extract the per-group x and w specs of a grouped_matmul-style op."""

    x_list = _arg(op, 0, "x")
    w_list = _arg(op, 1, "w")
    if not isinstance(x_list, (list, tuple)) or not isinstance(w_list, (list, tuple)):
        return None
    if not x_list or not w_list:
        return None
    # The first 2*len(x_list) leaves are x_0..x_n, w_0..w_m in DFS order.
    x_specs = specs[: len(x_list)]
    w_specs = specs[len(x_list) : len(x_list) + len(w_list)]
    if len(x_specs) != len(x_list) or len(w_specs) != len(w_list):
        return None
    return x_specs, w_specs


def _gmm_total_m(x_specs: Sequence[_Spec]) -> int | None:
    total = 0
    for spec in x_specs:
        if _num_dims(spec) < 1:
            return None
        rows = spec.shape[0]
        if is_symbolic(rows):
            return None
        total += int(rows)
    return total


def _gmm_head_dim(w_specs: Sequence[_Spec], position: int = 1) -> int | None:
    spec = w_specs[0]
    if _num_dims(spec) < 2:
        return None
    value = spec.shape[position]
    if is_symbolic(value):
        return None
    return int(value)


def _rule_grouped_matmul(op: Any, specs: list[_Spec]) -> dict[str, Any] | None:
    groups = _gmm_group_tensors(op, specs)
    if groups is None:
        return None
    x_specs, w_specs = groups
    total_m = _gmm_total_m(x_specs)
    columns = _gmm_head_dim(w_specs)
    if total_m is None or columns is None:
        return None
    return {"out": {"shape": [total_m, columns]}}


def _rule_grouped_matmul_swiglu(op: Any, specs: list[_Spec]) -> dict[str, Any] | None:
    groups = _gmm_group_tensors(op, specs)
    if groups is None:
        return None
    x_specs, w_specs = groups
    total_m = _gmm_total_m(x_specs)
    columns = _gmm_head_dim(w_specs)
    if total_m is None or columns is None or columns % 2 != 0:
        return None
    return {"out": {"shape": [total_m, columns // 2]}}


def _pack_factor(spec: _Spec) -> int | None:
    """``(element_size * 8) // 4`` for a 4-bit-packed weight dtype."""

    bits: int | None = None
    if spec.itemsize is not None:
        bits = spec.itemsize * 8
    else:
        name = str(getattr(spec.dtype, "name", None) or spec.dtype)
        # Packed 4-bit storage keeps two nibbles per stored byte-sized element.
        if "int4" in name or "fp4" in name or "mxfp4" in name:
            bits = 8
        elif "int8" in name or "fp8" in name or "float8" in name:
            bits = 8
        elif "int16" in name or "float16" in name or "bfloat16" in name:
            bits = 16
        elif "int32" in name or "float32" in name:
            bits = 32
    if bits is None or bits <= 0 or bits % 4 != 0:
        return None
    return bits // 4


def _rule_grouped_matmul_swiglu_packed(op: Any, specs: list[_Spec]) -> dict[str, Any] | None:
    groups = _gmm_group_tensors(op, specs)
    if groups is None:
        return None
    x_specs, w_specs = groups
    total_m = _gmm_total_m(x_specs)
    weight = w_specs[0]
    if total_m is None or _num_dims(weight) != 2 or _num_dims(x_specs[0]) < 1:
        return None
    input_features = x_specs[0].shape[-1]
    if is_symbolic(input_features) or int(input_features) <= 0:
        return None
    pack_factor = _pack_factor(weight)
    if pack_factor is None:
        return None
    elements = 1
    for dim in weight.shape:
        if is_symbolic(dim):
            return None
        elements *= int(dim)
    logical_elements = elements * pack_factor
    if logical_elements % int(input_features) != 0:
        return None
    gate_up = logical_elements // int(input_features)
    if gate_up % 2 != 0:
        return None
    return {"out": {"shape": [total_m, gate_up // 2]}}


def _rule_grouped_matmul_swiglu_quant(op: Any, specs: list[_Spec]) -> dict[str, Any] | None:
    groups = _gmm_group_tensors(op, specs)
    if groups is None:
        return None
    x_specs, w_specs = groups
    total_m = _gmm_total_m(x_specs)
    columns = _gmm_head_dim(w_specs)
    group_size = _arg(op, 6, "group_size")
    if total_m is None or columns is None or columns % 2 != 0:
        return None
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
        return None
    swiglu_width = columns // 2
    scale_width = (swiglu_width + group_size - 1) // group_size
    return {
        "out": (
            {"shape": [total_m, swiglu_width]},
            {"shape": [total_m, scale_width]},
        )
    }


# --------------------------------------------------------------------------------------
# Rule: empty_like / clone-style shape preservation
# --------------------------------------------------------------------------------------

EMPTY_LIKE_FUNCS = frozenset({"aten.empty_like.default", "aten.empty_like"})


def _rule_empty_like(op: Any, specs: list[_Spec]) -> dict[str, Any] | None:
    if not specs:
        return None
    return {"out": {"shape": list(specs[0].shape)}}


# --------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------

#: func name -> (rule label, rule callable)
RULES: dict[str, tuple[str, Any]] = {}
for _func in CAT_FUNCS:
    RULES[_func] = ("cat", _rule_cat)
for _func in SPLIT_FUNCS:
    RULES[_func] = ("split", _rule_split)
for _func in ALL_TO_ALL_FUNCS:
    RULES[_func] = ("all_to_all", _rule_all_to_all)
for _func in VIEW_FUNCS:
    RULES[_func] = ("view", _rule_view)
for _func in EXPAND_FUNCS:
    RULES[_func] = ("expand", _rule_expand)
for _func in SQUEEZE_FUNCS:
    RULES[_func] = ("squeeze", _rule_squeeze)
for _func in PERMUTE_FUNCS:
    RULES[_func] = ("permute", _rule_permute)
for _func in TRANSPOSE_FUNCS:
    RULES[_func] = ("transpose", _rule_transpose)
for _func in UNSQUEEZE_FUNCS:
    RULES[_func] = ("unsqueeze", _rule_unsqueeze)
for _func in SELECT_FUNCS:
    RULES[_func] = ("select", _rule_select)
for _func in SLICE_FUNCS:
    RULES[_func] = ("slice", _rule_slice)
for _func in GMM_PLAIN_FUNCS:
    RULES[_func] = ("grouped_matmul", _rule_grouped_matmul)
for _func in GMM_SWIGLU_FUNCS:
    RULES[_func] = ("grouped_matmul_swiglu", _rule_grouped_matmul_swiglu)
for _func in GMM_SWIGLU_PACKED_FUNCS:
    RULES[_func] = ("grouped_matmul_swiglu_packed", _rule_grouped_matmul_swiglu_packed)
for _func in GMM_SWIGLU_QUANT_FUNCS:
    RULES[_func] = ("grouped_matmul_swiglu_quant", _rule_grouped_matmul_swiglu_quant)
for _func in EMPTY_LIKE_FUNCS:
    RULES[_func] = ("empty_like", _rule_empty_like)

RULES = dict(sorted(RULES.items()))

#: Rule labels in stable display order.
RULE_LABELS: tuple[str, ...] = tuple(dict.fromkeys(label for label, _ in RULES.values()))


def _norm_dim(value: Any) -> Any:
    """Normalize a dim for comparison: mask runtime-opaque ints."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return RT_SYMBOL if abs(value) > RUNTIME_INT_MASK else int(value)
    return value


def derive_shape_semantically(
    op: Any,
    input_shapes: Any = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Derive the output shape(s) of ``op`` exactly from inputs + arguments.

    Returns ``{"out": <out tree>}`` (see module docstring) or ``None`` when no
    rule applies.  Never reads ``op.out``; safe to call on ops without
    ground-truth outputs.
    """

    func = _rule_name(getattr(op, "func", None))
    rule = RULES.get(func)
    if rule is None:
        return None
    del config  # rules are shape-structural; config kept for signature parity
    specs = _resolve_specs(op, input_shapes)
    try:
        return rule[1](op, specs)
    except (IndexError, TypeError, ValueError, AttributeError):
        # A malformed instance must never break the caller's build.
        return None


# --------------------------------------------------------------------------------------
# Corpus-wide validation harness (read-only over the capture cache)
# --------------------------------------------------------------------------------------


def _captured_out_tree(out: Any) -> Any:
    """Mirror the derived tree structure for a captured output."""

    if _looks_like_tensor(out):
        return {"shape": _dim_list(out.shape)}
    if isinstance(out, (list, tuple)):
        items = [_captured_out_tree(item) for item in out]
        return items if isinstance(out, list) else tuple(items)
    if isinstance(out, Mapping):
        return {key: _captured_out_tree(out[key]) for key in out}
    return out


def _tree_dims_equal(derived: Any, actual: Any) -> bool:
    """Bit-exact dim comparison between a derived tree and a captured tree."""

    if isinstance(derived, Mapping) and isinstance(actual, Mapping):
        if set(derived) != set(actual):
            return False
        return all(_tree_dims_equal(derived[key], actual[key]) for key in derived)
    if isinstance(derived, (list, tuple)) and isinstance(actual, (list, tuple)):
        if isinstance(derived, list) != isinstance(actual, list):
            return False
        return len(derived) == len(actual) and all(
            _tree_dims_equal(item, other) for item, other in zip(derived, actual)
        )
    return _norm_dim(derived) == _norm_dim(actual)


def _count_dims(tree: Any) -> int:
    """Number of output dims represented by a derived out tree."""

    if isinstance(tree, Mapping):
        if "shape" in tree:
            return len(tree["shape"])
        return sum(_count_dims(item) for item in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(_count_dims(item) for item in tree)
    return 0


def validate_shape_rules(
    cache_dir: str | None = "compile_ops_cache_v4_serving",
    *,
    keys: Sequence[str] | None = None,
    capture_limit: int | None = None,
    max_reported_mismatches: int = 40,
) -> dict[str, Any]:
    """Validate every rule against every matching op instance in the corpus.

    Read-only: loads captures through the builder's cache loader and compares
    each rule's derived output dims against the captured output bit-exactly
    (every dim, with the runtime-int mask applied on both sides).
    """

    from .template_builder import discover_cache_keys, load_captures_from_cache

    if keys is None:
        keys = discover_cache_keys(cache_dir)
    if capture_limit is not None:
        keys = keys[:capture_limit]
    captures = load_captures_from_cache(keys, cache_dir)

    counters: dict[str, dict[str, int]] = {
        label: {"instances": 0, "applied": 0, "declined": 0, "exact": 0, "mismatch": 0, "dims": 0}
        for label in RULE_LABELS
    }
    mismatches: list[dict[str, Any]] = []
    total_ops = 0
    for capture in captures:
        for op_index, op in enumerate(capture.ops):
            total_ops += 1
            func = _rule_name(getattr(op, "func", None))
            entry = RULES.get(func)
            if entry is None:
                continue
            label = entry[0]
            bucket = counters[label]
            bucket["instances"] += 1
            derived = derive_shape_semantically(op, None, capture.config)
            if derived is None:
                bucket["declined"] += 1
                continue
            bucket["applied"] += 1
            bucket["dims"] += _count_dims(derived.get("out"))
            actual_tree = _captured_out_tree(op.out)
            if _tree_dims_equal(derived.get("out"), actual_tree):
                bucket["exact"] += 1
            else:
                bucket["mismatch"] += 1
                if len(mismatches) < max_reported_mismatches:
                    mismatches.append(
                        {
                            "rule": label,
                            "func": func,
                            "capture": capture.key,
                            "op_index": op_index,
                            "derived": _compact(derived.get("out")),
                            "actual": _compact(actual_tree),
                        }
                    )

    covered = sum(bucket["applied"] for bucket in counters.values())
    exact = sum(bucket["exact"] for bucket in counters.values())
    return {
        "captures": len(captures),
        "ops": total_ops,
        "rules": counters,
        "covered_instances": covered,
        "exact_instances": exact,
        "covered_dims": sum(bucket["dims"] for bucket in counters.values()),
        "declined_instances": covered and sum(bucket["declined"] for bucket in counters.values()),
        "mismatch_instances": sum(bucket["mismatch"] for bucket in counters.values()),
        "mismatches": mismatches,
    }


def _compact(tree: Any, limit: int = 6) -> Any:
    """Shorten huge trees for the mismatch report."""

    if isinstance(tree, Mapping) and "shape" in tree:
        shape = tree["shape"]
        if isinstance(shape, list) and len(shape) > limit:
            return {"shape": shape[:limit] + ["..."]}
        return tree
    if isinstance(tree, (list, tuple)):
        if len(tree) > 3:
            head = [_compact(item, limit) for item in list(tree)[:3]]
            return (head + ["..."]) if isinstance(tree, list) else tuple(head + ["..."])
        return [_compact(item, limit) for item in tree]
    return tree


def _print_report(report: Mapping[str, Any]) -> None:
    logger = logging.getLogger(__name__)
    logger.info("captures: %s   ops scanned: %s", report["captures"], report["ops"])
    logger.info(
        "%-32s %10s %9s %9s %9s %9s %10s",
        "rule",
        "instances",
        "applied",
        "declined",
        "exact",
        "mismatch",
        "dims",
    )
    for label, bucket in report["rules"].items():
        logger.info(
            "%-32s %10d %9d %9d %9d %9d %10d",
            label,
            bucket["instances"],
            bucket["applied"],
            bucket["declined"],
            bucket["exact"],
            bucket["mismatch"],
            bucket.get("dims", 0),
        )
    logger.info(
        "coverage: %s op instances covered (%s exact, %s mismatch, %s output dims)",
        report["covered_instances"],
        report["exact_instances"],
        report["mismatch_instances"],
        report.get("covered_dims", 0),
    )
    for item in report["mismatches"]:
        logger.warning("MISMATCH %s", item)


if __name__ == "__main__":
    _print_report(validate_shape_rules())


__all__ = [
    "RULES",
    "RULE_LABELS",
    "RT_SYMBOL",
    "RUNTIME_INT_MASK",
    "REGION_ID_PLACEHOLDER_BASE",
    "derive_shape_semantically",
    "is_symbolic",
    "mask_scalar",
    "tensor_input_shapes",
    "validate_shape_rules",
]
