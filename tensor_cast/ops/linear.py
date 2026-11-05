# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from typing import Optional

import torch

from ..quantize_utils import validate_hif8_matmul
from ..utils import register_tensor_cast_op


@register_tensor_cast_op("batch_matmul_quant")
def _(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    w_offset: Optional[torch.Tensor],
    x_scale: Optional[torch.Tensor],
    x_offset: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """Quantized batched matrix multiplication.

    Semantically equivalent to a batch of ``static_quant_linear`` calls with
    one weight matrix per batch item.  Unlike grouped matmul, the batch axis is
    preserved in the output and does not represent routed expert groups.

    Args:
        x: ``(B, M, K)`` quantized activations.
        w: ``(B, K, N)`` quantized weights.
        w_scale: Weight dequantization scale.
        w_offset: Optional weight dequantization offset.
        x_scale: Optional activation dequantization scale.
        x_offset: Optional activation dequantization offset.
        bias: Optional output bias.
        out_dtype: Floating-point output dtype. This must be specified explicitly.
    """
    if x.ndim != 3 or w.ndim != 3:
        raise ValueError("batch_matmul_quant expects rank-3 activation and weight tensors")
    if x.shape[0] != w.shape[0] or x.shape[2] != w.shape[1]:
        raise ValueError("batch_matmul_quant requires compatible batch and contraction dimensions")
    if out_dtype is None:
        raise ValueError("batch_matmul_quant requires an explicit floating-point out_dtype")
    if not out_dtype.is_floating_point:
        raise ValueError("batch_matmul_quant requires a floating-point out_dtype")
    return torch.empty((x.shape[0], x.shape[1], w.shape[2]), dtype=out_dtype, device="meta")


@register_tensor_cast_op("static_quant_linear")
@register_tensor_cast_op("static_quant_linear_int4")
def _(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    w_offset: Optional[torch.Tensor],
    x_scale: Optional[torch.Tensor],
    x_offset: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """
    Semantically equivalent to
    `out = (dequant(x) @ dequant(w) + bias).to_dtype(out_dtype)` if x_scale is not None
    `out = (x @ dequant(w) + bias).to_dtype(out_dtype)` otherwise

    Args:
        x: (M, K)
        x_scale: scalar, (M, 1)
        x_offset: scalar, (M, 1)
        w: (K, N), for int4 (K/2, N)
        w_scale: scalar or (N,)
        w_offset: scalar or (N,)
        bias: Optional[(N,)] bias tensor
    """
    if out_dtype is None:
        out_dtype = x.dtype
    return torch.empty((x.shape[0], w.shape[1]), dtype=out_dtype, device="meta")


@register_tensor_cast_op("fp8_linear")
def _(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """
    FP8 linear operation. Both x and w are in FP8 format.
    Semantically equivalent to:
    `out = (dequant_fp8(x) @ dequant_fp8(w) + bias).to_dtype(out_dtype)`

    Args:
        x: (M, K) in FP8 format
        w: (K, N) in FP8 format
        x_scale: scalar or (M, 1) for activation scale
        w_scale: scalar or (N,) for weight scale
        bias: Optional[(N,)] bias tensor
        out_dtype: output data type
    """
    if out_dtype is None:
        out_dtype = torch.float16  # Default output for FP8
    return torch.empty((x.shape[0], w.shape[1]), dtype=out_dtype, device="meta")


@register_tensor_cast_op("mxfp4_linear")
def _(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """
    MXFP4 linear operation. Both x and w are in MXFP4 format with logical
    dimension as (M, K) and (K, N). The MXFP4 format is logically represented
    with torch.int4.

    Semantically equivalent to:
    `out = (dequant_mxfp4(x) @ dequant_mxfp4(w) + bias).to_dtype(out_dtype)`

    Args:
        x: (M, K) in MXFP4 format
        w: (K, N) in MXFP4 format
        x_scale: (M, Kg) for activation scale in torch.float8_e8m0fnu
        w_scale: (N, Kg) for weight scale in torch.float8_e8m0fnu
        bias: Optional[(N,)] bias tensor
        out_dtype: output data type
    """
    if out_dtype is None:
        out_dtype = torch.float16  # Default output for MXFP4
    return torch.empty((x.shape[0], w.shape[1]), dtype=out_dtype, device="meta")


@register_tensor_cast_op("hif8_linear")
def hif8_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    validate_hif8_matmul(x, w, x_scale, w_scale, bias, out_dtype)
    return torch.empty((x.shape[0], w.shape[1]), dtype=out_dtype, device=x.device)
