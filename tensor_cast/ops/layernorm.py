from typing import List, Tuple

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("rms_norm")
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x).contiguous()


@register_tensor_cast_op("rms_norm_quant")
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    quant_scale: torch.Tensor,
    quant_offset: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype = torch.int8,
) -> torch.Tensor:
    return torch.empty_like(x, dtype=out_dtype).contiguous()


@register_tensor_cast_op("add_rms_norm")
def _(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x).contiguous()


@register_tensor_cast_op("add_rms_norm2")
def _(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x).contiguous(), torch.empty_like(x).contiguous()


@register_tensor_cast_op("add_rms_norm_quant")
def _(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    quant_scale: torch.Tensor,
    quant_offset: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype = torch.int8,
) -> torch.Tensor:
    return torch.empty_like(x, dtype=out_dtype).contiguous()


@register_tensor_cast_op("add_rms_norm_quant2")
def _(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    quant_scale: torch.Tensor,
    quant_offset: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x, dtype=out_dtype).contiguous(), torch.empty_like(x).contiguous()


@register_tensor_cast_op("rms_norm_dynamic_quant_symmetric")
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    dims: List[int],
    scale_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.tensor_cast.dynamic_quantize_symmetric(x, dims, scale_dtype=scale_dtype, out_dtype=out_dtype)


@register_tensor_cast_op("rms_norm_dynamic_quant_asymmetric")
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    dims: List[int],
    scale_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.tensor_cast.dynamic_quantize_asymmetric(x, dims, scale_dtype=scale_dtype, out_dtype=out_dtype)


@register_tensor_cast_op("add_rms_norm_dynamic_quant_symmetric")
def _(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    dims: List[int],
    scale_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.tensor_cast.dynamic_quantize_symmetric(x, dims, scale_dtype=scale_dtype, out_dtype=out_dtype)


@register_tensor_cast_op("add_rms_norm_dynamic_quant_asymmetric")
def _(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    dims: List[int],
    scale_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.tensor_cast.dynamic_quantize_asymmetric(x, dims, scale_dtype=scale_dtype, out_dtype=out_dtype)


@register_tensor_cast_op("add_rms_norm_dynamic_quant2_symmetric")
def _(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    dims: List[int],
    scale_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x2 = torch.empty_like(x).contiguous()
    x1, scale = torch.ops.tensor_cast.dynamic_quantize_symmetric(x, dims, scale_dtype=scale_dtype, out_dtype=out_dtype)
    return x1, scale, x2


@register_tensor_cast_op("add_rms_norm_dynamic_quant2_asymmetric")
def _(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    dims: List[int],
    scale_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x2 = torch.empty_like(x).contiguous()
    x1, scale, offset = torch.ops.tensor_cast.dynamic_quantize_asymmetric(
        x, dims, scale_dtype=scale_dtype, out_dtype=out_dtype
    )
    return x1, scale, offset, x2


@register_tensor_cast_op("rms_norm_dynamic_quant_mxfp4")
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.tensor_cast.dynamic_quantize_mxfp4(x, group_size=group_size)


@register_tensor_cast_op("add_rms_norm_dynamic_quant_mxfp4")
def _(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.tensor_cast.dynamic_quantize_mxfp4(x, group_size=group_size)


@register_tensor_cast_op("add_rms_norm_dynamic_quant2_mxfp4")
def _(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x2 = torch.empty_like(x).contiguous()
    x1, scale = torch.ops.tensor_cast.dynamic_quantize_mxfp4(x, group_size=group_size)
    return x1, scale, x2
