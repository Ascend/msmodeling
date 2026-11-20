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

from typing import Optional

import torch


_executed = set()


def run_once(key, fn, *args, **kargs):
    if (key, fn) not in _executed:
        _executed.add((key, fn))
        fn(*args, **kargs)


def is_view_op(op):
    return op.is_view or op == torch.ops.aten._unsafe_view.default


def is_noop_self_copy_op(op, args=()) -> bool:
    # Similar to view ops, we keep this op visible in runtime/trace, but
    # model it as zero-cost when it is a semantic no-op copy_(x, x).
    if op != torch.ops.aten.copy_.default:
        return False
    # Defensive fallback: if call shape is unexpected, do not classify it as no-op.
    if len(args) < 2:
        return False
    dst, src = args[0], args[1]
    return isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor) and dst is src


def accounted_elements(tensor: torch.Tensor) -> int:
    """Elements a tensor occupies, ignoring the size ``numel()`` over-states.

    A view whose strides overlap or are zero - ``expand``/``broadcast_to`` of a
    small table, which is how models hand a per-request mask or rope index to
    the graph - reports ``numel()`` elements but spans only the elements of its
    base storage and allocates nothing. Charging ``numel()`` bills e.g. a
    1 KB table of 128 rope values as the 134 MB it is broadcast to.

    Strided slices span *more* than ``numel()`` and keep ``numel()``, so
    non-overlapping tensors are unaffected.
    """
    numel = int(tensor.numel())
    if numel == 0 or tensor.layout != torch.strided:
        return numel
    span = 1
    for size, step in zip(tensor.shape, tensor.stride()):
        span += (int(size) - 1) * abs(int(step))
    # ``span - 1`` is the highest offset the view touches. An injective layout
    # touches ``numel`` distinct offsets, so it can never span fewer elements
    # than ``numel``; only a zero/overlapping stride can collapse the count.
    return min(numel, span)


def bytes_of_tensor(tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> float:
    """
    Calculates the size of a tensor in bytes.
    This is a centralized function to ensure consistent byte calculation,
    especially for dtypes that might be packed.

    Both accounts that consume it want the same number:

    * **Allocation** (``MemoryTracker``): the bytes this tensor adds to the
      peak, which for a view is its share of the storage, not ``numel``.
    * **Traffic** (``op_invoke_info.get_memory_access_properties`` and the
      per-op ``memory_read_bytes``/``memory_write_bytes`` overrides): the bytes
      the op moves between the kernel and HBM.  For every non-overlapping
      layout that is exactly ``numel`` (a strided slice or transpose spans at
      least its logical elements), and for a stride-0 broadcast it is the base
      table the kernel actually fetches - reading the virtual extent would
      over-count a per-request mask or rope table by its broadcast factor.
    """
    dtype = tensor.dtype if dtype is None else dtype
    return bytes_of_elements(accounted_elements(tensor), dtype)


_bytes_of_dtype = {
    torch.int1: 1 / 8,
    torch.uint1: 1 / 8,
    torch.int2: 1 / 4,
    torch.uint2: 1 / 4,
    torch.int3: 3 / 8,
    torch.uint3: 3 / 8,
    torch.int4: 1 / 2,
    torch.uint4: 1 / 2,
    torch.int5: 5 / 8,
    torch.uint5: 5 / 8,
    torch.int6: 6 / 8,
    torch.uint6: 6 / 8,
    torch.int7: 7 / 8,
    torch.uint7: 7 / 8,
}


def bytes_of_elements(nelements: int, dtype: torch.dtype) -> float:
    """
    Calculates the size of a number of elements in bytes.
    This is a centralized function to ensure consistent byte calculation,
    especially for dtypes that might be packed.
    """
    return _bytes_of_dtype.get(dtype, dtype.itemsize) * nelements


def is_hif8_op(func) -> bool:
    """Recognize dedicated HIF8 operator targets by name."""
    name = str(func).removeprefix("torch.ops.")
    return name.startswith("tensor_cast.") and "hif8" in name
