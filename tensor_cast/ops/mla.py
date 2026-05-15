from typing import Optional, Tuple

import torch
from torch._subclasses.fake_tensor import is_fake

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("kv_rmsnorm_rope_cache", mutates_args=("kv_cache",))
def _(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused KV RmsNorm + RoPE + Cache write for MLA attention.

    Equivalent to vllm-ascend's torch_npu.npu_kv_rmsnorm_rope_cache().
    This is a meta implementation for TensorCast performance modeling.

    Algorithm:
        1. Split kv into kv_c (compressed) and k_pe (rope part)
        2. Apply RmsNorm to kv_c: kv_c_normed = kv_c * gamma / sqrt(mean(kv_c^2) + epsilon)
        3. Apply RoPE to k_pe using cos/sin
        4. Write both kv_c_normed and rotated k_pe to kv_cache at slot_mapping positions

    Args:
        kv: Input tensor of shape (num_tokens, kv_lora_rank + qk_rope_head_dim)
            Must be contiguous and have correct dtype (BF16/FP16)
        gamma: RmsNorm weight of shape (kv_lora_rank,)
        cos, sin: Rotary embeddings of shape (1, seq_len, qk_rope_head_dim)
        kv_cache: Cache tensor of shape (total_blocks, block_size, kv_lora_rank + qk_rope_head_dim)
        slot_mapping: Cache slot indices of shape (num_tokens,)
            Must be in range [0, total_blocks * block_size)
        kv_lora_rank: Dimension of compressed KV (must be > 0)
        qk_rope_head_dim: Dimension of RoPE part (must be > 0)
        epsilon: RmsNorm epsilon (default: 1e-6)

    Returns:
        k_pe: RoPE-rotated key of shape (num_tokens, qk_rope_head_dim)
        kv_c_normed: Normalized compressed KV of shape (num_tokens, kv_lora_rank)

    Note:
        This is a meta operation for performance modeling.
        The actual implementation in vllm-ascend uses torch_npu.npu_kv_rmsnorm_rope_cache.
    """
    num_tokens = kv.size(0)
    device = kv.device
    dtype = kv.dtype
    return (
        torch.empty((num_tokens, qk_rope_head_dim), dtype=dtype, device=device),
        torch.empty((num_tokens, kv_lora_rank), dtype=dtype, device=device),
    )


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused MLA preprocessing op that models RMS norm, matmuls, and RoPE rotation.

    Args:
        hidden_states: (num_tokens, hidden_size) activations entering MLA.
        cos/sin: rotary embedding caches shaped (1, seq_len, qk_rope_head_dim).
        q_a_proj_weight / q_b_proj_weight: LoRA weights with shapes
            (q_lora_rank, hidden_size) and (num_heads * qk_head_dim, q_lora_rank).
        q_a_layernorm_weight: RMSNorm scale for the LoRA branch (q_lora_rank,).
        kv_a_proj_weight: (kv_lora_rank + qk_rope_head_dim, hidden_size) matrix
            producing compressed key/value streams; kv_a_layernorm_weight matches
            its last dimension.
        num_heads/qk_* dims/kv_lora_rank/q_lora_rank: structural scalars that
            describe the MLA layout.

    Returns:
        q_states: (num_tokens, num_heads, qk_head_dim)
        kv_c_normed: (num_tokens, kv_lora_rank)
        k_rot: (num_tokens, qk_rope_head_dim)
        qa_normed: (num_tokens, q_lora_rank) when q_lora_rank is set;
            otherwise an empty last-dimension tensor that the caller converts back to None.
    """

    num_tokens = hidden_states.size(0)
    device = hidden_states.device
    dtype = hidden_states.dtype
    qa_normed_dim = q_lora_rank or 0
    return (
        torch.empty((num_tokens, num_heads, qk_head_dim), dtype=dtype, device=device),
        torch.empty((num_tokens, kv_lora_rank), dtype=dtype, device=device),
        torch.empty((num_tokens, qk_rope_head_dim), dtype=dtype, device=device),
        torch.empty((num_tokens, qa_normed_dim), dtype=dtype, device=device),
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantized variant of the fused MLA preprocessing op.

    Args mirror `mlapo`, but q_a/q_b/kv_a *_scale/*_offset tensors encode the
    quantization scheme (per-tensor/per-group) applied to their respective
    linear layers.

    Returns:
        q_states: (num_tokens, num_heads, qk_head_dim)
        kv_c_normed: (num_tokens, kv_lora_rank)
        k_rot: (num_tokens, qk_rope_head_dim)
        qa_normed: (num_tokens, q_lora_rank) when q_lora_rank is set;
            otherwise an empty last-dimension tensor that the caller converts back to None.
    """

    num_tokens = hidden_states.size(0)
    device = hidden_states.device
    dtype = hidden_states.dtype
    qa_normed_dim = q_lora_rank or 0
    return (
        torch.empty((num_tokens, num_heads, qk_head_dim), dtype=dtype, device=device),
        torch.empty((num_tokens, kv_lora_rank), dtype=dtype, device=device),
        torch.empty((num_tokens, qk_rope_head_dim), dtype=dtype, device=device),
        torch.empty((num_tokens, qa_normed_dim), dtype=dtype, device=device),
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
    topk_limit: Optional[int] = None,
    topk_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This op computes multi-head latent attention (MLA). It is supposed to use different
    algorithms for prefill and decode shapes while the input sequences could fuse prefill
    and decode sequences and should be handled separately with different algorithms.

    We judge the prefill or decode phase according to the query length per `query_start_loc`.
    If the query length is

    For prefill (non-strict math/code):
        k_nope, v = (kv_c_normed @ kv_b_proj).view(-1, num_heads, qk_nope_head_dim + v_head_dim).split(dim=-1)
        softmax(q @ (k_nope, k_rot) + sparse_mask(topk_indices)) @ v

    For decode (non-strict math/code):
        softmax(q @ W_UK_T @ k_cache + sparse_mask(topk_indices)) @ v_cache @ W_UV

    `sparse_mask(topk_indices)` is omitted when `topk_indices` is None.

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
        topk_limit: Number of top-K tokens for sparse attention
        topk_indices: Preselected token positions for sparse attention.
    Returns:
        (num_tokens, num_heads, v_head_dim)
    """
    if topk_indices is not None:
        # Keep the sparse top-k tensor live in the semantic graph via its shape.
        _ = topk_indices.shape[-1]
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
    topk_limit: Optional[int],
    topk_indices: Optional[torch.Tensor],
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
) -> torch.Tensor:
    """
    Similar to `multihead_latent_attention` but with quantization support.

    For prefill (non-strict math/code):
        quant_kv_proj = quant(kv_c_normed @ kv_b_proj, kv_projected_scale, kv_projected_offset)
        k_nope, v = quant_kv_proj.view(-1, num_heads, qk_nope_head_dim + v_head_dim).split(dim=-1)
        out_fp = quant(
            softmax(q @ (k_nope, k_rot) + sparse_mask(topk_indices)),
            attention_prob_scale,
            attention_prob_offset,
        ) @ v
        out = quant(out_fp, out_scale, out_offset) # optional

    For decode (non-strict math/code):
        quant_qk = quant(q @ W_UK_T, qk_scale, qk_offset)
        quant_scores = quant(
            softmax(quant_qk @ k_cache + sparse_mask(topk_indices)),
            attention_prob_scale,
            attention_prob_offset,
        )
        out_fp = quant(quant_scores @ v_cache, v_scale, v_offset) @ W_UV
        out = quant(out_fp, out_scale, out_offset) # optional

    `sparse_mask(topk_indices)` is omitted when `topk_indices` is None.

    Args:
        topk_limit: Number of top-K tokens for sparse attention
        topk_indices: Preselected token positions for sparse attention.

    Returns:
        (num_tokens, num_heads, v_head_dim)
    """
    if topk_indices is not None:
        # Keep the sparse top-k tensor live in the semantic graph via its shape.
        _ = topk_indices.shape[-1]
    if out_dtype is None:
        out_dtype = q.dtype
    return torch.empty(q.shape[0], q.shape[1], v_head_dim, dtype=out_dtype, device="meta")


@register_tensor_cast_op("dsa_indexer", mutates_args=("indexer_cache",))
def _(
    hidden_states: torch.Tensor,
    qa_normed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    indexer_cache: torch.Tensor,
    slot_mapping: Optional[torch.Tensor],
    block_tables: Optional[torch.Tensor],
    seq_lens: Optional[torch.Tensor],
    wq_b_weight: torch.Tensor,
    wk_weight: torch.Tensor,
    weights_proj_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    qk_rope_head_dim: int,
    topk_limit: int,
) -> torch.Tensor:
    """
    Fused DSA indexer semantic block.

    For the DeepSeek-V3.2-style fp8 path (non-strict math/code):
        q = rope(wq_b(qa_normed))
        k = rope(k_norm(wk(hidden_states)))

        q = rotate_activation(q)
        k = rotate_activation(k)

        q_fp8, q_scale = act_quant(q)
        k_fp8, k_scale = act_quant(k)
        k_cache, k_scale_cache = append(indexer_cache, k_fp8, k_scale)

        weights = weights_proj(hidden_states) * num_heads**-0.5
        weights = weights.unsqueeze(-1) * q_scale * head_dim**-0.5

        index_score = fp8_index(q_fp8, weights, k_cache, k_scale_cache)
        topk_indices = topk(index_score, k=min(topk_limit, active_seq_len), dim=-1).indices

    Compared with the fp8 path, the bf16 / GLM5-style path removes:
        - rotate_activation on q and k
        - act_quant on q and k
        - scale-cache writes alongside the key cache
        - fp8-specific relu / q-scale / k-scale score shaping

    and instead uses direct cache scoring plus head reduction:
        weights = weights_proj(hidden_states) * num_heads**-0.5
        head_scores = (q @ k_cache.transpose(-1, -2)) * head_dim**-0.5
        index_score = reduce_sum(head_scores * weights.unsqueeze(-1), dim=-2)
        topk_indices = topk(index_score, k=min(topk_limit, active_seq_len), dim=-1).indices

    Returns:
        topk_indices: (batch, seq_len, min(topk_limit, active_seq_len))
    """
    batch, seq_len, _ = hidden_states.shape
    # torch.compile traces this op with FakeTensors; avoid extracting a
    # data-dependent Python int from seq_lens in that path.
    if is_fake(hidden_states):
        topk = topk_limit
    else:
        active_seq_len = int(seq_lens.max().item()) if seq_lens is not None else seq_len
        topk = min(topk_limit, active_seq_len)
    return torch.empty(batch, seq_len, topk, dtype=torch.long, device=hidden_states.device)
