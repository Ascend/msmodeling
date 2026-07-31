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


def _check_gate_up_shape_match(gate: torch.Tensor, up: torch.Tensor, op_name: str) -> None:
    if gate.shape != up.shape:
        raise RuntimeError(f"Shape mismatch in {op_name}: gate {gate.shape} vs up {up.shape}")


@register_tensor_cast_op("swiglu")
def _(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    _check_gate_up_shape_match(gate, up, "swiglu")

    output_shape = list(gate.shape)
    return torch.empty(output_shape, dtype=gate.dtype, device="meta")


@register_tensor_cast_op("m3_swiglu")
def _(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """Fused M3 SwiGLU-OAI meta op.

    alpha and limit are accepted to keep the TensorCast op signature aligned
    with the upstream MiniMax-M3 activation.  This meta op only infers output
    shape and dtype, so those scalar values do not affect the returned tensor.
    """
    _check_gate_up_shape_match(gate, up, "m3_swiglu")

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
    """Fused M3 SwiGLU-OAI + post activation quant meta op.

    alpha, limit, and group_size are accepted for interface compatibility with
    the upstream activation and quantized down-projection path.  Shape inference
    uses per-sample activation quantization over the last dimension, so
    group_size does not change the scale tensor shape here.
    """
    _check_gate_up_shape_match(gate, up, "m3_swiglu_quant")

    scale_shape = _symmetric_quant_scale_shape(gate.shape, [-1])
    return (
        torch.empty(list(gate.shape), dtype=torch.int8, device=gate.device),
        torch.empty(scale_shape, dtype=torch.float32, device=gate.device),
    )
