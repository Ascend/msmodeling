from typing import Optional, Tuple

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("concat_and_cache_mla", mutates_args=("kv_cache",))
def _(
    kv_c_normed: torch.Tensor,
    k_rot: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """
    concat `kv_c_normed` and `k_rot` with into `kv_cache` according to `slot_mapping`.

    Args:
        kv_c_normed: (num_tokens, kv_lora_rank)
        k_rot: (num_tokens, qk_rope_head_dim)
        kv_cache: (total_num_blocks, block_size, kv_lora_rank + qk_rope_head_dim)
        slot_mapping: see `AttentionMetadataBase`
    """


@register_tensor_cast_op("mlapo")
def _(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_a_proj_weight: Optional[torch.Tensor],
    q_a_layernorm_weight: Optional[torch.Tensor],
    q_b_proj_weight: Optional[torch.Tensor],
    kv_a_proj_weight: Optional[torch.Tensor],
    kv_a_layernorm_weight: torch.Tensor,
    num_heads: int,
    qk_head_dim: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    q_lora_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused MLA preprocessing op that models RMS norm, matmuls, and RoPE rotation.

    Args:
        hidden_states: (num_tokens, hidden_size) activations entering MLA.
        cos/sin: rotary embedding caches shaped (1, seq_len, qk_rope_head_dim).
        q_a_proj_weight / q_b_proj_weight: LoRA weights with shapes
            (hidden_size, q_lora_rank) and (q_lora_rank, num_heads * qk_head_dim).
        q_a_layernorm_weight: RMSNorm scale for the LoRA branch (q_lora_rank,).
        kv_a_proj_weight: (hidden_size, kv_lora_rank + qk_rope_head_dim) matrix
            producing compressed key/value streams; kv_a_layernorm_weight matches
            its last dimension.
        num_heads/qk_* dims/kv_lora_rank/q_lora_rank: structural scalars that
            describe the MLA layout.

    Returns:
        q_states: (num_tokens, num_heads, qk_head_dim)
        kv_c_normed: (num_tokens, kv_lora_rank)
        k_rot: (num_tokens, qk_rope_head_dim)
    """

    num_tokens = hidden_states.size(0)
    device = hidden_states.device
    dtype = hidden_states.dtype
    return (
        torch.empty((num_tokens, num_heads, qk_head_dim), dtype=dtype, device=device),
        torch.empty((num_tokens, kv_lora_rank), dtype=dtype, device=device),
        torch.empty((num_tokens, qk_rope_head_dim), dtype=dtype, device=device),
    )


@register_tensor_cast_op("mlapo_quant")
def _(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_a_proj_weight: Optional[torch.Tensor],
    q_a_layernorm_weight: Optional[torch.Tensor],
    q_b_proj_weight: Optional[torch.Tensor],
    kv_a_proj_weight: Optional[torch.Tensor],
    kv_a_layernorm_weight: torch.Tensor,
    num_heads: int,
    qk_head_dim: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    q_lora_rank: int,
    q_a_proj_scale: torch.Tensor,
    q_a_proj_offset: Optional[torch.Tensor],
    q_b_proj_scale: torch.Tensor,
    q_b_proj_offset: Optional[torch.Tensor],
    kv_a_proj_scale: torch.Tensor,
    kv_a_proj_offset: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantized variant of the fused MLA preprocessing op.

    Args mirror `mlapo`, but q_a/q_b/kv_a *_scale/*_offset tensors encode the
    quantization scheme (per-tensor/per-group) applied to their respective
    linear layers.

    Returns:
        q_states: (num_tokens, num_heads, qk_head_dim)
        kv_c_normed: (num_tokens, kv_lora_rank)
        k_rot: (num_tokens, qk_rope_head_dim)
    """

    num_tokens = hidden_states.size(0)
    device = hidden_states.device
    dtype = hidden_states.dtype
    return (
        torch.empty((num_tokens, num_heads, qk_head_dim), dtype=dtype, device=device),
        torch.empty((num_tokens, kv_lora_rank), dtype=dtype, device=device),
        torch.empty((num_tokens, qk_rope_head_dim), dtype=dtype, device=device),
    )


@register_tensor_cast_op("multihead_latent_attention")
def _(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: Optional[torch.Tensor],
    W_UK_T: Optional[torch.Tensor],
    W_UV: Optional[torch.Tensor],
    kv_b_proj: Optional[torch.Tensor],
    v_head_dim: int,
    index_topk: Optional[int] = None,
    cached_topk_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This op computes multi-head latent attention (MLA). It is supposed to use different
    algorithms for prefill and decode shapes while the input sequences could fuse prefill
    and decode sequences and should be handled separately with different algorithms.

    We judge the prefill or decode phase according to the query length per `query_start_loc`.
    If the query length is

    For prefill (non-strict math/code):
        k_nope, v = (kv_c_normed @ kv_b_proj).view(-1, num_heads, qk_nope_head_dim + v_head_dim).split(dim=-1)
        softmax(q @ (k_nope, k_rot)) @ v

        kv_c_normed: (num_tokens, kv_lora_rank)
            The normalized and compressed key-value states.
        k_rot: (num_tokens, qk_rope_head_dim)
            The slice of key after applying rotation embedding.

    For decode (non-strict math/code):
        softmax(q @ W_UK_T @ k_cache) @ v_cache @ W_UV

    Args:
        q: (num_tokens, num_heads, qk_nope_head_dim+qk_rope_head_dim)
            The query states after compression and decompression.
        kv_cache: (total_num_blocks, block_size, kv_lora_rank + qk_rope_head_dim)
            The cached key-value states with current KV states already updated.
        block_table/query_start_loc/seq_lens: see `AttentionMetadataBase`
        W_UK_T, W_UV: (num_heads, qk_nope_head_dim, kv_lora_rank), (num_heads, kv_lora_rank, v_head_dim)
            used in the decode phase, None if only prefill sequences are provided.
        kv_b_proj: (kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))
            used in the prefill phase, None if only decode sequences are provided.
        index_topk: Number of top-K tokens for sparse attention
        cached_topk_indices: Preselected token positions for DSA sparse attention;
            prevents dsa_index from being optimized away by torch.compile.
    Returns:
        (num_tokens, num_heads, v_head_dim)
    """
    return torch.empty(q.shape[0], q.shape[1], v_head_dim, dtype=q.dtype, device="meta")


@register_tensor_cast_op("multihead_latent_attention_quant")
def _(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: Optional[torch.Tensor],
    W_UK_T: Optional[torch.Tensor],
    W_UV: Optional[torch.Tensor],
    kv_b_proj: Optional[torch.Tensor],
    v_head_dim: int,
    query_scale: torch.Tensor,
    query_offset: Optional[torch.Tensor],
    kv_scale: torch.Tensor,
    kv_offset: Optional[torch.Tensor],
    kv_projected_scale: torch.Tensor,
    kv_projected_offset: Optional[torch.Tensor],
    qk_scale: torch.Tensor,
    qk_offset: Optional[torch.Tensor],
    v_scale: torch.Tensor,
    v_offset: Optional[torch.Tensor],
    attention_prob_scale: torch.Tensor,
    attention_prob_offset: Optional[torch.Tensor],
    kv_b_proj_scale: torch.Tensor,
    kv_b_proj_offset: Optional[torch.Tensor],
    out_scale: Optional[torch.Tensor],
    out_offset: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    index_topk: Optional[int] = None,
    cached_topk_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Similar to `multihead_latent_attention` but with quantization support.
    For prefill (non-strict math/code):
        quant_kv_proj = quant(kv_c_normed @ kv_b_proj, kv_projected_scale, kv_projected_offset)
        k_nope, v = quant_kv_proj.view(-1, num_heads, qk_nope_head_dim + v_head_dim).split(dim=-1)
        out_fp = quant(softmax(q @ (k_nope, k_rot)), attention_prob_scale, attention_prob_offset) @ v
        out = quant(out_fp, out_scale, out_offset) # optional

    For decode (non-strict math/code):
        quant_qk = quant(q @ W_UK_T, qk_scale, qk_offset)
        quant_scores = quant(softmax(quant_qk @ k_cache), attention_prob_scale, attention_prob_offset)
        out_fp = quant(quant_scores @ v_cache, v_scale, v_offset) @ W_UV
        out = quant(out_fp, out_scale, out_offset) # optional

    Returns:
        (num_tokens, num_heads, v_head_dim)
    """
    if out_dtype is None:
        out_dtype = q.dtype
    return torch.empty(
        q.shape[0], q.shape[1], v_head_dim, dtype=out_dtype, device="meta"
    )


@register_tensor_cast_op("dsa_index")
def _(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """
    Perform index score using FP8 precision.
    Args:
        q (torch.Tensor): The Q tensor, must be contiguous.
        q_s (torch.Tensor): The scaling factor for Q (float), must be contiguous.
        k (torch.Tensor): The K tensor, must be contiguous.
        k_s (torch.Tensor): The scaling factor for K (e8m0 here), must be contiguous.
        fp8 q @ fp8 k -> fp32 logits
        relu(fp32 logits) * q_s (weights) -> fp32 logits
        fp32 logits -> fp32 logits_sum
        fp32 logits_sum * k_s (e8m0) -> fp32 index_score
    """
    # out shape: (batch, num_heads, seq_len)
    return torch.empty(
        q.shape[0], q_s.shape[2], q_s.shape[1], dtype=q.dtype, device="meta"
    )


@register_tensor_cast_op("dsa_index_cache", mutates_args=("k_cache",))
def _(
    k: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_tables: Optional[torch.Tensor] = None,
) -> None:
    """
    In-place write of token Key vectors into KV cache.

    1. Continuous Mode (block_tables is None):
       - Logic: k_cache[slot_mapping[i]] = k[i]
       - The 'slot_mapping' tensor directly provides the physical row index in k_cache
         for each token in the input 'k'.

    2. Paged Attention Mode (block_tables is provided):
       - Logic: Derive (block_idx, offset) from slot_mapping.
       - Lookup physical block ID: block_id = block_tables[batch_idx, block_idx_logical]
       - Write: k_cache[block_id, offset] = k[i]
       - This allows non-contiguous memory allocation, crucial for long-context inference.

    Args:
        k: Input Key tensor. Shape varies by backend: [num_tokens, head_dim] or [batch, num_tokens, head_dim].
        k_cache: Global KV cache buffer.
                 Shape: [max_seq_len, head_dim] (continuous) or [num_blocks, block_size, head_dim] (paged).
        slot_mapping: Physical indices mapping each token to its location in k_cache. Shape: [num_tokens].
        block_tables: Page table for paged attention. Shape: [batch_size, max_blocks_per_seq]. Optional.
    """
