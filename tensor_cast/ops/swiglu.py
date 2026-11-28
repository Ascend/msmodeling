from typing import Tuple

import torch

from ..utils import register_tensor_cast_op


def _symmetric_quant_scale_shape(x_shape: torch.Size, dims: list[int]) -> torch.Size:
    if not dims:
        return torch.Size([])
    scale_shape = list(x_shape)
    for dim in dims:
        scale_shape[dim] = 1
    return torch.Size(scale_shape)


@register_tensor_cast_op("swiglu")
def _(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.shape != up.shape:
        raise RuntimeError(f"Shape mismatch in swiglu: gate {gate.shape} vs up {up.shape}")

    output_shape = list(gate.shape)
    return torch.empty(output_shape, dtype=gate.dtype, device="meta")


@register_tensor_cast_op("m3_swiglu")
def _(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    if gate.shape != up.shape:
        raise RuntimeError(f"Shape mismatch in m3_swiglu: gate {gate.shape} vs up {up.shape}")

    output_shape = list(gate.shape)
    return torch.empty(output_shape, dtype=gate.dtype, device="meta")


@register_tensor_cast_op("m3_swiglu_quant")
def _(
    gate: torch.Tensor,
    up: torch.Tensor,
    alpha: float,
    limit: float,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused M3 SwiGLU + post activation quant (maps to silu_and_mul_post_quant on NPU)."""
    if gate.shape != up.shape:
        raise RuntimeError(f"Shape mismatch in m3_swiglu_quant: gate {gate.shape} vs up {up.shape}")

    output_shape = list(gate.shape)
    scale_shape = _symmetric_quant_scale_shape(gate.shape, [-1])
    return (
        torch.empty(output_shape, dtype=torch.int8, device=gate.device),
        torch.empty(scale_shape, dtype=torch.float32, device=gate.device),
    )
