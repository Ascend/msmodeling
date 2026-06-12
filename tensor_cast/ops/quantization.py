from typing import List, Optional, Tuple

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("quantize")
def _(
    x: torch.Tensor,
    scale: torch.Tensor,
    offset: Optional[torch.Tensor],
    out_dtype: torch.dtype = torch.int8,
) -> torch.Tensor:
    """`out = clamp(round(x / scale) + offset)`"""
    return torch.empty_like(x, dtype=out_dtype)


@register_tensor_cast_op("dynamic_quantize_asymmetric")
def _(
    x: torch.Tensor,
    dims: List[int],
    scale_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dynamically quantize the input tensor `x` to `out_dtype` with symmetric quantization.
    The quantization scale is computed based on the absolute max value of `x` along the specified dimensions.

    Args:
        x: The input tensor to be quantized.
        dims: The dimensions along which to compute the quantization scale.
        scale_dtype: The data type for the quantization scale (default: torch.float32).
        out_dtype: The target data type for quantization (default: torch.int8).

    Returns:
        A tuple containing:
        - The quantized tensor.
        - The quantization scale tensor. When `dims` is empty, the scale is a scalar tensor.
          Otherwise, the scale tensor has the same shape as `x` with the specified `dims` reduced to size 1.
        - The quantization offset tensor, the same shape as the scale tensor but with dtype torch.int32.
    """
    if len(dims) == 0:
        scale_shape = torch.Size([])
    else:
        scale_shape = list(x.shape)
        for dim in dims:
            scale_shape[dim] = 1
        scale_shape = torch.Size(scale_shape)
    return (
        torch.empty_like(x, dtype=out_dtype),
        torch.empty(scale_shape, dtype=scale_dtype, device="meta"),
        torch.empty(scale_shape, dtype=torch.int32, device="meta"),
    )


@register_tensor_cast_op("dynamic_quantize_symmetric")
def _(
    x: torch.Tensor,
    dims: List[int],
    scale_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Same as `dynamic_quantize_asymmetric` but for symmetric quantization (no offset).
    """
    if len(dims) == 0:
        scale_shape = torch.Size([])
    else:
        scale_shape = list(x.shape)
        for dim in dims:
            scale_shape[dim] = 1
        scale_shape = torch.Size(scale_shape)
    return (
        torch.empty_like(x, dtype=out_dtype),
        torch.empty(scale_shape, dtype=scale_dtype, device="meta"),
    )


@register_tensor_cast_op("dynamic_quantize_mxfp4")
def _(
    x: torch.Tensor,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamically quantize the input tensor `x` to MXFP4. The quantization is applied
    per channel group along the last dimension, where each channel group contains
    `group_size` channels. The quantization is symmetric.

    Args:
        x: The input tensor to be quantized.
        group_size: The channel group size for MXFP4 quantization.

    Returns:
        A tuple containing:
        - The quantized tensor.
        - The quantization scale tensor of shape (K_group,).
    """
    K = x.shape[-1]
    K_group = (K + group_size - 1) // group_size
    return (
        torch.empty_like(x, dtype=torch.int4),
        torch.empty((K_group,), dtype=torch.float8_e8m0fnu, device="meta"),
    )
