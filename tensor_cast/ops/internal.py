from typing import List

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("_internal_mark_region_begin")
def _(
    x: torch.Tensor,
    id: int,
) -> torch.Tensor:
    """Mark the beginning of a region of execution.

    Must not return an input alias: under ``torch.compile``, identity aliases can be
    DCE'd so ``mark_region_begin`` disappears while ``mark_region_end`` remains,
    which breaks ``Runtime.repeat_op_invoke_infos`` pairing.
    """
    return x.clone()


@register_tensor_cast_op("_internal_mark_region_end")
def _(
    x: torch.Tensor,
    id: int,
) -> torch.Tensor:
    """Mark the end of a region of execution.

    Non-aliasing return keeps the end marker in compiled graphs (same rationale as
    ``_internal_mark_region_begin``).
    """
    return x.clone()


@register_tensor_cast_op("_internal_copy_region")
def _(
    x: torch.Tensor,
    id: int,
) -> torch.Tensor:
    """Copy a region of execution marked previously."""
    return x


@register_tensor_cast_op("_internal_wait_and_bind")
def _(
    x: torch.Tensor,
    stream_id: int,
    deps: List[torch.Tensor],
) -> torch.Tensor:
    """Bind real ops through the paired record to ``stream_id`` after waiting on ``deps``.

    This is a control-flow anchor used by multistream lowering. It does not modify
    the data carried by ``x``. Instead, the runtime interprets it as:

    1. real ops through the paired ``_internal_record`` execute on ``stream_id``;
    2. the first real op waits until all dependency tokens in ``deps`` are ready.

    Example:
        y = _internal_wait_and_bind(x, 1, [token0])
        z = real_op(y)
        token1 = _internal_record(z, 1)

    Here ``real_op`` runs on stream 1 only after ``token0`` is ready.
    """
    # torch.library custom ops are not allowed to return an input alias.
    return x.clone()


@register_tensor_cast_op("_internal_order_barrier")
def _(
    x: torch.Tensor,
    dep: torch.Tensor,
) -> torch.Tensor:
    """Force ``dep`` to be sequenced before consumers of the result under ``torch.compile``.

    Values of ``x`` are unchanged (clone only to satisfy custom-op non-aliasing rules).
    Used when two subgraphs are logically ordered in Python but share no tensor edge,
    so AOT/Inductor would otherwise sink the earlier work (e.g. draft embedding) after
    a later independent path (e.g. context KV ``reshape_and_cache``).
    """
    del dep  # FX keeps the arg as a real dependency; value is unused.
    return x.clone()


@register_tensor_cast_op("_internal_record")
def _(
    x: torch.Tensor,
    stream_id: int,
) -> torch.Tensor:
    """Create a control token marking completion of the paired stream region.

    This op is paired with ``_internal_wait_and_bind`` during multistream lowering.
    The returned scalar tensor is a runtime control token, not a model activation.

    Example:
        y = real_op(x)
        token = _internal_record(y, 0)
        z = _internal_wait_and_bind(other, 1, [token])

    The wait op can use ``token`` to express that a later op must not start until
    ``real_op`` on stream 0 has completed.
    """
    return torch.empty((), dtype=torch.int64, device=x.device)
