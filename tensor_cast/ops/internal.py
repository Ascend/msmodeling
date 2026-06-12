from typing import List

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("_internal_mark_region_begin")
def _(
    x: torch.Tensor,
    id: int,
) -> torch.Tensor:
    """Mark the beginning of a region of execution."""
    return x


@register_tensor_cast_op("_internal_mark_region_end")
def _(
    x: torch.Tensor,
    id: int,
) -> torch.Tensor:
    """Mark the end of a region of execution."""
    return x


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
    """Bind the next real op on ``x`` to ``stream_id`` after waiting on ``deps``.

    This is a control-flow anchor used by multistream lowering. It does not modify
    the data carried by ``x``. Instead, the runtime interprets it as:

    1. the next real op consuming ``x`` should execute on ``stream_id``;
    2. that real op must wait until all dependency tokens in ``deps`` are ready.

    Example:
        y = _internal_wait_and_bind(x, 1, [token0])
        z = real_op(y)
        token1 = _internal_record(z, 1)

    Here ``real_op`` runs on stream 1 only after ``token0`` is ready.
    """
    # torch.library custom ops are not allowed to return an input alias.
    return x.clone()


@register_tensor_cast_op("_internal_record")
def _(
    x: torch.Tensor,
    stream_id: int,
) -> torch.Tensor:
    """Create a control token marking completion of the preceding real op.

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
