from typing import List, Optional, Tuple

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("grouped_matmul")
def _(
    x: List[torch.Tensor],
    w: List[torch.Tensor],
    bias: List[Optional[torch.Tensor]],
) -> torch.Tensor:
    """
    Perform grouped quantized matrix multiplication. The arguments follow
    the same convention as `static_quant_linear` but are grouped as lists.
    The output is a concatenation of the individual matmul results, not a list
    of tensors.
    """
    M = sum(xi.shape[0] for xi in x)
    N = w[0].shape[1]
    return torch.empty((M, N), dtype=x[0].dtype, device="meta")


@register_tensor_cast_op("grouped_matmul_quant")
@register_tensor_cast_op("grouped_matmul_quant_int4")
def _(
    x: List[torch.Tensor],
    w: List[torch.Tensor],
    w_scale: List[torch.Tensor],
    w_offset: List[Optional[torch.Tensor]],
    x_scale: List[torch.Tensor],
    x_offset: List[Optional[torch.Tensor]],
    bias: List[Optional[torch.Tensor]],
    out_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """Similar to `grouped_matmul` but with quantization parameters."""
    if out_dtype is None:
        out_dtype = x[0].dtype
    M = sum(xi.shape[0] for xi in x)
    N = w[0].shape[1]
    return torch.empty((M, N), dtype=out_dtype, device="meta")


@register_tensor_cast_op("grouped_matmul_fp8")
@register_tensor_cast_op("grouped_matmul_mxfp4")
def _(
    x: List[torch.Tensor],
    w: List[torch.Tensor],
    w_scale: List[torch.Tensor],
    x_scale: List[torch.Tensor],
    bias: List[Optional[torch.Tensor]],
    out_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """Similar to `grouped_matmul` but for FP8 quantization."""
    if out_dtype is None:
        out_dtype = x[0].dtype
    M = sum(xi.shape[0] for xi in x)
    N = w[0].shape[1]
    return torch.empty((M, N), dtype=out_dtype, device="meta")


def _grouped_matmul_swiglu_out(
    x: List[torch.Tensor],
    w: List[torch.Tensor],
    dtype: torch.dtype,
    *,
    packed_4bit_weight: bool,
) -> torch.Tensor:
    M = sum(xi.shape[0] for xi in x)
    if not w:
        return torch.empty((M, 0), dtype=dtype, device="meta")

    weight = w[0]
    if packed_4bit_weight:
        if not x or x[0].shape[-1] <= 0:
            raise ValueError("packed 4-bit GMM+SwiGLU requires a positive input feature dimension")
        input_features = x[0].shape[-1]
        pack_factor = (weight.element_size() * 8) // 4
        logical_weight_elements = weight.numel() * pack_factor
        if logical_weight_elements % input_features != 0:
            raise ValueError("packed 4-bit weight elements must be divisible by the input feature dimension")
        gate_up_features = logical_weight_elements // input_features
    else:
        gate_up_features = weight.shape[1]

    if gate_up_features % 2 != 0:
        raise ValueError("GMM+SwiGLU gate/up feature dimension must be even")
    return torch.empty((M, gate_up_features // 2), dtype=dtype, device="meta")


@register_tensor_cast_op("grouped_matmul_swiglu")
def _(
    x: List[torch.Tensor],
    w: List[torch.Tensor],
    bias: List[Optional[torch.Tensor]],
) -> torch.Tensor:
    dtype = x[0].dtype if x else torch.float32
    return _grouped_matmul_swiglu_out(x, w, dtype, packed_4bit_weight=False)


@register_tensor_cast_op("grouped_matmul_quant_swiglu")
def _(
    x: List[torch.Tensor],
    w: List[torch.Tensor],
    w_scale: List[torch.Tensor],
    w_offset: List[Optional[torch.Tensor]],
    x_scale: List[torch.Tensor],
    x_offset: List[Optional[torch.Tensor]],
    bias: List[Optional[torch.Tensor]],
    out_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    if out_dtype is None:
        out_dtype = x[0].dtype if x else torch.float32
    return _grouped_matmul_swiglu_out(x, w, out_dtype, packed_4bit_weight=False)


@register_tensor_cast_op("grouped_matmul_quant_int4_swiglu")
def _(
    x: List[torch.Tensor],
    w: List[torch.Tensor],
    w_scale: List[torch.Tensor],
    w_offset: List[Optional[torch.Tensor]],
    x_scale: List[torch.Tensor],
    x_offset: List[Optional[torch.Tensor]],
    bias: List[Optional[torch.Tensor]],
    out_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    if out_dtype is None:
        out_dtype = x[0].dtype if x else torch.float32
    return _grouped_matmul_swiglu_out(x, w, out_dtype, packed_4bit_weight=True)


@register_tensor_cast_op("grouped_matmul_fp8_swiglu")
def _(
    x: List[torch.Tensor],
    w: List[torch.Tensor],
    w_scale: List[torch.Tensor],
    x_scale: List[torch.Tensor],
    bias: List[Optional[torch.Tensor]],
    out_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    if out_dtype is None:
        out_dtype = x[0].dtype if x else torch.float32
    return _grouped_matmul_swiglu_out(x, w, out_dtype, packed_4bit_weight=False)


@register_tensor_cast_op("grouped_matmul_mxfp4_swiglu")
def _(
    x: List[torch.Tensor],
    w: List[torch.Tensor],
    w_scale: List[torch.Tensor],
    x_scale: List[torch.Tensor],
    bias: List[Optional[torch.Tensor]],
    out_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    if out_dtype is None:
        out_dtype = x[0].dtype if x else torch.float32
    return _grouped_matmul_swiglu_out(x, w, out_dtype, packed_4bit_weight=True)


@register_tensor_cast_op("grouped_matmul_mxfp4_swiglu_quant")
def _(
    x: List[torch.Tensor],
    w: List[torch.Tensor],
    w_scale: List[torch.Tensor],
    x_scale: List[torch.Tensor],
    bias: List[Optional[torch.Tensor]],
    out_dtype: Optional[torch.dtype],
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Native MXFP4 GMM1 fusion: GMM + SwiGLU + post-activation MX quant.

    The payload/scale pair is directly consumable by the following MXFP4
    grouped matmul.  ``out_dtype`` describes the high-precision epilogue
    computation before quantization; it is retained for performance modeling.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    M = sum(xi.shape[0] for xi in x)
    N = w[0].shape[1] if w else 0
    swiglu_width = N // 2
    scale_width = (swiglu_width + group_size - 1) // group_size
    return (
        torch.empty((M, swiglu_width), dtype=torch.int4, device="meta"),
        torch.empty((M, scale_width), dtype=torch.float8_e8m0fnu, device="meta"),
    )
