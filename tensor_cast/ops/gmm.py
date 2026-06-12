from typing import List, Optional

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


@register_tensor_cast_op("grouped_matmul_swiglu")
def _(
    x: List[torch.Tensor],
    w: List[torch.Tensor],
    bias: List[Optional[torch.Tensor]],
) -> torch.Tensor:
    M = sum(xi.shape[0] for xi in x)
    N = w[0].shape[1] if w else 0
    gmm_out_shape = (M, N)
    dtype = x[0].dtype if x else torch.float32

    swiglu_out_shape = gmm_out_shape
    return torch.empty(swiglu_out_shape, dtype=dtype, device="meta")


@register_tensor_cast_op("grouped_matmul_quant_swiglu")
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

    M = sum(xi.shape[0] for xi in x)
    N = w[0].shape[1] if w else 0
    gmm_out_shape = (M, N)

    swiglu_out_shape = gmm_out_shape
    return torch.empty(swiglu_out_shape, dtype=out_dtype, device="meta")


@register_tensor_cast_op("grouped_matmul_fp8_swiglu")
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

    M = sum(xi.shape[0] for xi in x)
    N = w[0].shape[1] if w else 0
    gmm_out_shape = (M, N)

    swiglu_out_shape = gmm_out_shape
    return torch.empty(swiglu_out_shape, dtype=out_dtype, device="meta")
