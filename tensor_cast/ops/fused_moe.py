from typing import List, Optional, Tuple

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("init_routing_v2")
def _(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Repeat the input tokens top-k times, and rearrange them according to the order of the experts
    selected by the tokens.

    Args:
        x: (bsz, seq_len, hidden_size), the tokens
        topk_indices: (bsz, seq_len, top_k), the top-k experts selected by each token

    Returns:
        permuted_x: (bsz * seq_len * top_k, hidden_size)
    """
    num_tokens = topk_indices.numel()
    return torch.empty((num_tokens, x.shape[-1]), dtype=x.dtype, device=x.device)


@register_tensor_cast_op("unpermute_tokens")
def _(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Rearrange the input tokens (initially sorted by their selected experts) by token indices.

    Args:
        x: (bsz * seq_len * top_k, hidden_size), the tokens
        topk_indices: (bsz, seq_len, top_k), the top-k experts selected by each token

    Returns:
        unpermuted_x: (bsz, seq_len, top_k, hidden_size)
    """
    return torch.empty_like(x).view(*topk_indices.shape, x.shape[-1])


@register_tensor_cast_op("dispatch_ffn_combine")
def _(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    gmm1_w: List[torch.Tensor],
    gmm1_bias: List[Optional[torch.Tensor]],
    gmm2_w: List[torch.Tensor],
    gmm2_bias: List[Optional[torch.Tensor]],
    rank: int,
    rank_group: List[int],
) -> torch.Tensor:
    """Fused MoE FFN: routing + gate_up_proj(SwiGLU) + down_proj + all_to_all.
    BF16 variant. Args carry weights/bias from region (not activations).
    M dimension derived from expert_indices.numel().
    """
    hidden_size = x.shape[-1]
    return torch.empty((*expert_indices.shape, hidden_size), dtype=x.dtype, device=x.device)


@register_tensor_cast_op("dispatch_ffn_combine_quant")
@register_tensor_cast_op("dispatch_ffn_combine_quant_int4")
def _(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    gmm1_w: List[torch.Tensor],
    gmm1_w_scale: List[torch.Tensor],
    gmm1_w_offset: List[Optional[torch.Tensor]],
    gmm1_x_scale: List[torch.Tensor],
    gmm1_x_offset: List[Optional[torch.Tensor]],
    gmm1_bias: List[Optional[torch.Tensor]],
    gmm1_out_dtype: Optional[torch.dtype],
    gmm2_w: List[torch.Tensor],
    gmm2_w_scale: List[torch.Tensor],
    gmm2_w_offset: List[Optional[torch.Tensor]],
    gmm2_x_scale: List[torch.Tensor],
    gmm2_x_offset: List[Optional[torch.Tensor]],
    gmm2_bias: List[Optional[torch.Tensor]],
    gmm2_out_dtype: Optional[torch.dtype],
    rank: int,
    rank_group: List[int],
) -> torch.Tensor:
    """Fused MoE FFN: W8A8/W4A8 quant variant.
    GMM weight/scale args mirror grouped_matmul_quant[_swiglu] sans x.
    """
    hidden_size = x.shape[-1]
    return torch.empty((*expert_indices.shape, hidden_size), dtype=x.dtype, device=x.device)


@register_tensor_cast_op("dispatch_ffn_combine_fp8")
@register_tensor_cast_op("dispatch_ffn_combine_mxfp4")
def _(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    gmm1_w: List[torch.Tensor],
    gmm1_w_scale: List[torch.Tensor],
    gmm1_x_scale: List[torch.Tensor],
    gmm1_bias: List[Optional[torch.Tensor]],
    gmm1_out_dtype: Optional[torch.dtype],
    gmm2_w: List[torch.Tensor],
    gmm2_w_scale: List[torch.Tensor],
    gmm2_x_scale: List[torch.Tensor],
    gmm2_bias: List[Optional[torch.Tensor]],
    gmm2_out_dtype: Optional[torch.dtype],
    rank: int,
    rank_group: List[int],
) -> torch.Tensor:
    """Fused MoE FFN: FP8/MXFP4 quant variant.
    GMM weight/scale args mirror grouped_matmul_fp8[_swiglu] sans x.
    """
    hidden_size = x.shape[-1]
    return torch.empty((*expert_indices.shape, hidden_size), dtype=x.dtype, device=x.device)


@register_tensor_cast_op("moe_gating_top_k_softmax")
def _(x: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused operation for Mixture of Experts (MoE) gating, combining softmax and top-k selection.

    This function is designed to handle both the softmax operation over the gating logits
    and the top-k selection of experts in one step. It returns the shape of the expected output
    tensors (experts_weights, and experts_indices) without performing any computation.

    Args:
        x (torch.Tensor): A tensor of containing the raw unnormalized logits for each experts.
                          These logits will be used to compute the softmax probabilities and
                          select the top-k experts.
        top_k (int): The number of top experts to select based on their softmax probabilities.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - topk_weights (torch.Tensor): Corresponding normalized weights (e.g., after softmax),
              with shape `(*x.shape[:-1], top_k)`, dtype and device as input `x`.
            - topk_indices (torch.Tensor): Indices of the selected experts,
              with shape `(*x.shape[:-1], top_k)` and device as input `x`, dtype int64.
    """
    out_shape = (*x.shape[:-1], top_k)
    return (
        torch.empty(out_shape, dtype=x.dtype, device=x.device),
        torch.empty(out_shape, dtype=torch.int64, device=x.device),
    )
