from typing import Optional, Tuple

import torch
from torch._subclasses.fake_tensor import is_fake

from ..utils import register_tensor_cast_op


def local_mla_phase_token_counts(local_tokens: int, num_prefill_tokens: int, num_decode_tokens: int) -> tuple[int, int]:
    """Split rank-local tokens across Prefill and Decode without independent clamps.

    Forward traces global ``num_prefill_tokens`` / ``num_decode_tokens``.
    SequenceParallelPass then rewrites activations to T/TP but leaves those
    integers unchanged. Independent ``min(local, declared_phase)`` is valid for
    a single populated phase, but mixed batches can produce
    ``prefill_local + decode_local > local_tokens``. Scale both phases so they
    jointly occupy the local activation.
    """
    local_tokens = int(local_tokens)
    prefill = max(int(num_prefill_tokens), 0)
    decode = max(int(num_decode_tokens), 0)
    if local_tokens <= 0:
        return 0, 0
    if prefill <= 0:
        return 0, min(local_tokens, decode)
    if decode <= 0:
        return min(local_tokens, prefill), 0
    declared = prefill + decode
    prefill_local = (prefill * local_tokens) // declared
    return prefill_local, local_tokens - prefill_local


def _mla_phase_token_count(
    activation: torch.Tensor,
    declared_tokens: int,
    sibling_tokens: int,
    *,
    is_prefill: bool,
) -> int:
    prefill, decode = local_mla_phase_token_counts(
        int(activation.shape[0]),
        declared_tokens if is_prefill else sibling_tokens,
        sibling_tokens if is_prefill else declared_tokens,
    )
    return prefill if is_prefill else decode


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
    q_lora_rank: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused MLA preprocessing op that models RMS norm, matmuls, and RoPE rotation.

    Args:
        hidden_states: (num_tokens, hidden_size) activations entering MLA.
        cos/sin: rotary embedding caches shaped (1, seq_len, qk_rope_head_dim).
        q_a_proj_weight / q_b_proj_weight: LoRA weights with shapes
            (q_lora_rank, hidden_size) and (num_heads * qk_head_dim, q_lora_rank).
            When q_lora_rank is None, q_a_proj_weight is the direct q_proj weight
            and q_b_proj_weight/q_a_layernorm_weight are None.
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
    q_lora_rank: Optional[int],
    q_a_proj_scale: torch.Tensor,
    q_a_proj_offset: Optional[torch.Tensor],
    q_b_proj_scale: Optional[torch.Tensor],
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
    projected_kv: torch.Tensor,
    absorbed_q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: Optional[torch.Tensor],
    v_head_dim: int,
    kv_lora_rank: int,
    topk_limit: Optional[int] = None,
    topk_indices: Optional[torch.Tensor] = None,
    *,
    is_decode_values: Optional[list[bool]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This op represents the core multi-head latent attention kernel: score
    calculation, softmax, and value aggregation.  Phase-specific Q/KV/V linear
    projections are outside this op's performance-modeling boundary.

    We judge the prefill or decode phase according to the query length per `query_start_loc`.
    If the query length is

    For prefill (non-strict math/code):
        softmax(q @ k + sparse_mask(topk_indices)) @ v

    For decode (non-strict math/code):
        softmax(q_absorbed @ k_cache + sparse_mask(topk_indices)) @ v_cache

    `sparse_mask(topk_indices)` is omitted when `topk_indices` is None.

    Args:
        q: (num_tokens, num_heads, qk_nope_head_dim+qk_rope_head_dim)
            The query states after compression and decompression.
        projected_kv: Prefill KV projection output. The token dimension is zero
            when the batch has no Prefill requests.
        absorbed_q: Decode Q-absorption output. The token dimension is zero
            when the batch has no Decode requests.
        kv_cache: (total_num_blocks, block_size, kv_lora_rank + qk_rope_head_dim)
            The cached key-value states with current KV states already updated.
        block_table/query_start_loc/seq_lens: see `AttentionMetadataBase`
        topk_limit: Number of top-K tokens for sparse attention.
        topk_indices: Preselected token positions for sparse attention.
            For dense MLA (DeepSeek-V3) these are typically None; the DSA sparse
            path uses the dedicated mla_sparse_attention[_quant] ops. They are
            retained here so both ops share one analytic helper — when provided,
            the helper clamps the attended context length to topk (see
            _multihead_latent_attention_properties_helper).
    Returns:
        Prefill values with width ``v_head_dim`` and Decode latent values with
        width ``kv_lora_rank``.
    """
    del is_decode_values
    return (
        torch.empty(projected_kv.shape[0], projected_kv.shape[1], v_head_dim, dtype=q.dtype, device="meta"),
        torch.empty(absorbed_q.shape[0], absorbed_q.shape[1], kv_lora_rank, dtype=q.dtype, device="meta"),
    )


@register_tensor_cast_op("mla_kv_projection")
def _(
    kv_c_normed: torch.Tensor,
    kv_b_proj: torch.Tensor,
    num_prefill_tokens: int,
    num_heads: int,
    num_decode_tokens: int,
) -> torch.Tensor:
    """Prefill compressed-KV projection with its physical output width."""
    output_dim = kv_b_proj.shape[1] // num_heads
    num_tokens = _mla_phase_token_count(kv_c_normed, num_prefill_tokens, num_decode_tokens, is_prefill=True)
    return torch.empty(num_tokens, num_heads, output_dim, dtype=kv_c_normed.dtype, device="meta")


@register_tensor_cast_op("mla_kv_projection_quant")
def _(
    kv_c_normed: torch.Tensor,
    kv_b_proj: torch.Tensor,
    num_prefill_tokens: int,
    num_heads: int,
    num_decode_tokens: int,
    projected_scale: torch.Tensor,
    projected_offset: Optional[torch.Tensor],
    weight_scale: torch.Tensor,
    weight_offset: Optional[torch.Tensor],
) -> torch.Tensor:
    """Quantized variant of :func:`mla_kv_projection`."""
    output_dim = kv_b_proj.shape[1] // num_heads
    num_tokens = _mla_phase_token_count(kv_c_normed, num_prefill_tokens, num_decode_tokens, is_prefill=True)
    return torch.empty(num_tokens, num_heads, output_dim, dtype=kv_c_normed.dtype, device="meta")


@register_tensor_cast_op("mla_q_absorb_projection")
def _(
    q: torch.Tensor,
    W_UK_T: torch.Tensor,
    num_decode_tokens: int,
    qk_rope_head_dim: int,
    num_prefill_tokens: int,
) -> torch.Tensor:
    """Decode Q absorption with the physical latent-plus-RoPE width."""
    return torch.empty(
        _mla_phase_token_count(q, num_decode_tokens, num_prefill_tokens, is_prefill=False),
        W_UK_T.shape[0],
        W_UK_T.shape[-1] + qk_rope_head_dim,
        dtype=q.dtype,
        device="meta",
    )


@register_tensor_cast_op("mla_q_absorb_projection_quant")
def _(
    q: torch.Tensor,
    W_UK_T: torch.Tensor,
    num_decode_tokens: int,
    qk_rope_head_dim: int,
    num_prefill_tokens: int,
    qk_scale: torch.Tensor,
    qk_offset: Optional[torch.Tensor],
    weight_scale: torch.Tensor,
    weight_offset: Optional[torch.Tensor],
) -> torch.Tensor:
    """Quantized variant of :func:`mla_q_absorb_projection`."""
    return torch.empty(
        _mla_phase_token_count(q, num_decode_tokens, num_prefill_tokens, is_prefill=False),
        W_UK_T.shape[0],
        W_UK_T.shape[-1] + qk_rope_head_dim,
        dtype=q.dtype,
        device="meta",
    )


@register_tensor_cast_op("mla_v_up_projection")
def _(attention_output: torch.Tensor, W_UV: torch.Tensor) -> torch.Tensor:
    """Decode latent-value up-projection to the physical value-head width."""
    return torch.empty(
        attention_output.shape[0], W_UV.shape[0], W_UV.shape[-1], dtype=attention_output.dtype, device="meta"
    )


@register_tensor_cast_op("mla_v_up_projection_quant")
def _(attention_output: torch.Tensor, W_UV: torch.Tensor) -> torch.Tensor:
    """Quantized variant of :func:`mla_v_up_projection`."""
    return torch.empty(
        attention_output.shape[0], W_UV.shape[0], W_UV.shape[-1], dtype=attention_output.dtype, device="meta"
    )


@register_tensor_cast_op("mla_merge_phase_outputs")
def _(
    prefill_output: torch.Tensor,
    decode_output: torch.Tensor,
    total_tokens: int,
    is_decode_values: Optional[list[bool]],
    query_lens_values: Optional[list[int]],
) -> torch.Tensor:
    """Restore packed request order after phase-specific MLA kernels."""
    del is_decode_values, query_lens_values
    packed_tokens = int(prefill_output.shape[0]) + int(decode_output.shape[0])
    num_tokens = packed_tokens if packed_tokens > 0 else int(total_tokens)
    num_heads = prefill_output.shape[1] if prefill_output.shape[0] else decode_output.shape[1]
    value_dim = prefill_output.shape[-1] if prefill_output.shape[0] else decode_output.shape[-1]
    return torch.empty(num_tokens, num_heads, value_dim, dtype=prefill_output.dtype, device="meta")


@register_tensor_cast_op("multihead_latent_attention_quant")
def _(
    q: torch.Tensor,
    projected_kv: torch.Tensor,
    absorbed_q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: Optional[torch.Tensor],
    v_head_dim: int,
    kv_lora_rank: int,
    topk_limit: Optional[int],
    topk_indices: Optional[torch.Tensor],
    query_scale: torch.Tensor,
    query_offset: Optional[torch.Tensor],
    kv_scale: torch.Tensor,
    kv_offset: Optional[torch.Tensor],
    v_scale: torch.Tensor,
    v_offset: Optional[torch.Tensor],
    attention_prob_scale: torch.Tensor,
    attention_prob_offset: Optional[torch.Tensor],
    out_scale: Optional[torch.Tensor],
    out_offset: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    *,
    is_decode_values: Optional[list[bool]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantized core-attention variant of `multihead_latent_attention`.

    For prefill (non-strict math/code), projected KV is already provided:
        out_fp = quant(
            softmax(q @ (k_nope, k_rot) + sparse_mask(topk_indices)),
            attention_prob_scale,
            attention_prob_offset,
        ) @ v
        out = quant(out_fp, out_scale, out_offset) # optional

    For decode (non-strict math/code), absorbed Q is already provided:
        quant_scores = quant(
            softmax(absorbed_q @ k_cache + sparse_mask(topk_indices)),
            attention_prob_scale,
            attention_prob_offset,
        )
        latent_out = quant(quant_scores @ v_cache, v_scale, v_offset)

    `sparse_mask(topk_indices)` is omitted when `topk_indices` is None.

    Args:
        topk_limit: Number of top-K tokens for sparse attention
        topk_indices: Preselected token positions for sparse attention.

    Returns:
        (num_tokens, num_heads, v_head_dim)
    """
    del is_decode_values
    if out_dtype is None:
        out_dtype = q.dtype
    return (
        torch.empty(projected_kv.shape[0], projected_kv.shape[1], v_head_dim, dtype=out_dtype, device="meta"),
        torch.empty(absorbed_q.shape[0], absorbed_q.shape[1], kv_lora_rank, dtype=out_dtype, device="meta"),
    )


@register_tensor_cast_op("mla_sparse_attention")
def _(
    q: torch.Tensor,
    projected_kv: torch.Tensor,
    absorbed_q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: Optional[torch.Tensor],
    v_head_dim: int,
    kv_lora_rank: int,
    topk_limit: Optional[int] = None,
    topk_indices: Optional[torch.Tensor] = None,
    *,
    is_decode_values: Optional[list[bool]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sparse MLA attention (DSA path) for DeepSeek-V3.2 and GLM-5.1.

    Semantically identical to multihead_latent_attention; registered as a
    separate TC op so the profiling database maps it to SparseFlashAttention
    (SFA) instead of FusedInferAttentionScore (FIA).

    ``is_decode_values`` carries the explicit per-request phase from attention
    metadata so profiling does not infer decode from chunked-prefill shapes.

    vllm-ascend dispatch condition: hf_config has index_topk → AscendSFABackend
    → npu_sparse_flash_attention.
    """
    return (
        torch.empty(projected_kv.shape[0], projected_kv.shape[1], v_head_dim, dtype=q.dtype, device="meta"),
        torch.empty(absorbed_q.shape[0], absorbed_q.shape[1], kv_lora_rank, dtype=q.dtype, device="meta"),
    )


@register_tensor_cast_op("mla_sparse_attention_quant")
def _(
    q: torch.Tensor,
    projected_kv: torch.Tensor,
    absorbed_q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: Optional[torch.Tensor],
    v_head_dim: int,
    kv_lora_rank: int,
    topk_limit: Optional[int],
    topk_indices: Optional[torch.Tensor],
    query_scale: torch.Tensor,
    query_offset: Optional[torch.Tensor],
    kv_scale: torch.Tensor,
    kv_offset: Optional[torch.Tensor],
    v_scale: torch.Tensor,
    v_offset: Optional[torch.Tensor],
    attention_prob_scale: torch.Tensor,
    attention_prob_offset: Optional[torch.Tensor],
    out_scale: Optional[torch.Tensor],
    out_offset: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    *,
    is_decode_values: Optional[list[bool]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantized sparse MLA attention (DSA path). SFA variant of
    multihead_latent_attention_quant. Used by DeepSeek-V3.2 and GLM-5.1
    when quant config is enabled. ``is_decode_values`` has the same phase
    semantics as the BF16 sparse MLA op.
    """
    if out_dtype is None:
        out_dtype = q.dtype
    return (
        torch.empty(projected_kv.shape[0], projected_kv.shape[1], v_head_dim, dtype=out_dtype, device="meta"),
        torch.empty(absorbed_q.shape[0], absorbed_q.shape[1], kv_lora_rank, dtype=out_dtype, device="meta"),
    )


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
    query_lens: Optional[torch.Tensor] = None,
    *,
    is_decode_values: Optional[list[bool]] = None,
) -> torch.Tensor:
    """
    Fused DSA indexer semantic block.

    ``query_lens`` carries the per-request query boundaries used by the
    sequence-parallel rank projection and analytic fallback. ``is_decode_values``
    carries the explicit per-request phase for both performance-model paths.

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
