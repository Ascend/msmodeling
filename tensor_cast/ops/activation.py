import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("gelu")
def _(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """Return GELU output metadata."""
    del approximate
    return torch.empty_like(x).contiguous()


@register_tensor_cast_op("silu")
def _(x: torch.Tensor) -> torch.Tensor:
    """Return SiLU output metadata."""
    return torch.empty_like(x).contiguous()


@register_tensor_cast_op("gated_residual_add")
def _(residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Return gated residual-add output metadata."""
    output_shape = torch.broadcast_shapes(residual.shape, update.shape, gate.shape)
    output_dtype = torch.promote_types(torch.promote_types(residual.dtype, update.dtype), gate.dtype)
    return torch.empty(output_shape, dtype=output_dtype, device=residual.device)
