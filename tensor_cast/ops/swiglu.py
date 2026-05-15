import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("swiglu")
def _(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.shape != up.shape:
        raise RuntimeError(f"Shape mismatch in swiglu: gate {gate.shape} vs up {up.shape}")

    output_shape = list(gate.shape)
    return torch.empty(output_shape, dtype=gate.dtype, device="meta")
