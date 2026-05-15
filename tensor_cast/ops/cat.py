from typing import Sequence

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("cat")
def _(tensors: Sequence[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """
    Shape-only replacement for aten.cat that preserves dtype on meta tensors.
    """
    if not tensors:
        raise ValueError("tensor_cast.cat expects a non-empty tensor list")
    ref = tensors[0]
    for t in tensors[1:]:
        if t.dtype != ref.dtype:
            raise ValueError("tensor_cast.cat expects all input tensors to have the same dtype")
    out_shape = list(ref.shape)
    if dim < 0:
        dim = dim + len(out_shape)
    out_dim = 0
    for t in tensors:
        out_dim += t.shape[dim]
    out_shape[dim] = out_dim
    return torch.empty(
        out_shape,
        dtype=ref.dtype,
        device=ref.device,
    )
