from typing import List, Optional

import torch

from ..utils import exact_division, register_tensor_cast_op


@register_tensor_cast_op("all_to_all")
def _(
    x: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    rank: int,
    rank_group: List[int],
) -> torch.Tensor:
    output_num = sum(output_split_sizes)
    return torch.empty((output_num, *x.shape[1:]), dtype=x.dtype, device=x.device)


@register_tensor_cast_op("all_reduce")
def _(x: torch.Tensor, rank: int, rank_group: List[int]) -> torch.Tensor:
    return torch.empty_like(x)


@register_tensor_cast_op("reduce_scatter")
def _(x: torch.Tensor, dim: int, rank: int, rank_group: List[int]) -> torch.Tensor:
    world_size = len(rank_group)
    new_shape = list(x.shape)
    new_shape[dim] = exact_division(new_shape[dim], world_size)
    return torch.empty(new_shape, dtype=x.dtype, device=x.device)


@register_tensor_cast_op("all_gather")
def _(x: torch.Tensor, dim: int, rank: int, rank_group: List[int]) -> torch.Tensor:
    world_size = len(rank_group)
    new_shape = list(x.shape)
    new_shape[dim] = new_shape[dim] * world_size
    return torch.empty(new_shape, dtype=x.dtype, device=x.device)


@register_tensor_cast_op("matmul_all_reduce")
def _(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    bias: Optional[torch.Tensor],
    rank: int,
    rank_group: List[int],
) -> torch.Tensor:
    matmul_out = torch.matmul(mat1, mat2)
    return torch.empty_like(matmul_out)


@register_tensor_cast_op("static_quant_linear_all_reduce")
def _(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    w_offset: Optional[torch.Tensor],
    x_scale: Optional[torch.Tensor],
    x_offset: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    rank: int,
    rank_group: List[int],
) -> torch.Tensor:
    linear_out = torch.ops.tensor_cast.static_quant_linear.default(
        x,
        w,
        w_scale,
        w_offset,
        x_scale,
        x_offset,
        bias,
        out_dtype if out_dtype is not None else x.dtype,
    )
    return torch.empty_like(linear_out)


@register_tensor_cast_op("static_quant_linear_int4_all_reduce")
def _(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    w_offset: Optional[torch.Tensor],
    x_scale: Optional[torch.Tensor],
    x_offset: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    rank: int,
    rank_group: List[int],
) -> torch.Tensor:
    linear_out = torch.ops.tensor_cast.static_quant_linear_int4.default(
        x,
        w,
        w_scale,
        w_offset,
        x_scale,
        x_offset,
        bias,
        out_dtype if out_dtype is not None else x.dtype,
    )
    return torch.empty_like(linear_out)


@register_tensor_cast_op("fp8_linear_all_reduce")
def _(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    rank: int,
    rank_group: List[int],
) -> torch.Tensor:
    linear_out = torch.ops.tensor_cast.fp8_linear.default(
        x,
        w,
        x_scale,
        w_scale,
        bias,
        out_dtype,
    )
    return torch.empty_like(linear_out)


@register_tensor_cast_op("mxfp4_linear_all_reduce")
def _(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    rank: int,
    rank_group: List[int],
) -> torch.Tensor:
    linear_out = torch.ops.tensor_cast.mxfp4_linear.default(
        x,
        w,
        x_scale,
        w_scale,
        bias,
        out_dtype,
    )
    return torch.empty_like(linear_out)
