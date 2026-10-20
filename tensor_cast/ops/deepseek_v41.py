"""Semantic operators for DeepSeek V4.1 CSA2, Engram, mHC and Vision."""

from typing import Optional, Tuple

import torch

from ..utils import register_tensor_cast_op


@register_tensor_cast_op("deepseek_v41_vision_rope")
def deepseek_v41_vision_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    grid_height: int,
    grid_width: int,
    rope_theta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shape contract for DeepSeek-ViT bidirectional 2D RoPE."""
    del grid_height, grid_width, rope_theta
    return torch.empty_like(query), torch.empty_like(key)


@register_tensor_cast_op("deepseek_v41_vision_downsample")
def deepseek_v41_vision_downsample(
    hidden_states: torch.Tensor,
    grid_height: int,
    grid_width: int,
    downsample_ratio: int,
) -> torch.Tensor:
    """Model the aligner's padded r x r pixel-unshuffle rearrangement."""
    if hidden_states.ndim != 3:
        raise ValueError("DeepSeek V4.1 vision downsample expects [images, patches, hidden]")
    if int(hidden_states.shape[1]) != int(grid_height) * int(grid_width):
        raise ValueError("Vision patch count must match grid_height * grid_width")
    ratio = int(downsample_ratio)
    out_height = (int(grid_height) + ratio - 1) // ratio
    out_width = (int(grid_width) + ratio - 1) // ratio
    return torch.empty(
        hidden_states.shape[0],
        out_height * out_width,
        hidden_states.shape[-1] * ratio * ratio,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )


@register_tensor_cast_op("deepseek_v41_image_merge")
def deepseek_v41_image_merge(
    text_embeddings: torch.Tensor,
    image_features: torch.Tensor,
    image_token_types: torch.Tensor,
    image_start: torch.Tensor,
    image_end: torch.Tensor,
    image_newline: torch.Tensor,
) -> torch.Tensor:
    """Overwrite the V4.1 image span while preserving the text embedding shape."""
    del image_features, image_start, image_end, image_newline
    if image_token_types.shape != text_embeddings.shape[:-1]:
        raise ValueError("DeepSeek V4.1 image token types must align with input embeddings")
    return torch.empty_like(text_embeddings)


@register_tensor_cast_op("moe_gating_top_k_vl")
def moe_gating_top_k_vl(
    scores: torch.Tensor,
    top_k: int,
    normalize_weights: bool,
    route_scale: float,
    text_bias: torch.Tensor,
    image_bias: torch.Tensor,
    image_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """V4.1 routing tail selecting text/image correction bias per token."""
    del normalize_weights, route_scale
    if text_bias.shape != image_bias.shape or text_bias.shape[-1] != scores.shape[-1]:
        raise ValueError("DeepSeek V4.1 text/image MoE biases must match the expert axis")
    if image_mask.numel() != scores.numel() // scores.shape[-1]:
        raise ValueError("DeepSeek V4.1 image mask must contain one value per routed token")
    out_shape = (*scores.shape[:-1], int(top_k))
    return (
        torch.empty(out_shape, dtype=scores.dtype, device=scores.device),
        torch.empty(out_shape, dtype=torch.int64, device=scores.device),
    )


@register_tensor_cast_op("engram_hash_lookup")
def engram_hash_lookup(
    compressed_token_ids: torch.Tensor,
    history: Optional[torch.Tensor],
    num_hash_cols: int,
    max_ngram_size: int,
    table_count: int = 1,
    request_count: int = 1,
    token_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Emit deterministic-shaped n-gram hash IDs for Engram table access."""
    del token_mask
    if compressed_token_ids.ndim != 2:
        raise ValueError("Engram hash input_ids must be [batch, sequence]")
    history_len = max(int(max_ngram_size) - 1, 0)
    hashes = torch.empty(
        (*compressed_token_ids.shape, int(table_count), int(num_hash_cols)),
        dtype=torch.int64,
        device=compressed_token_ids.device,
    )
    next_history = torch.empty(int(request_count), history_len, dtype=torch.int64, device=compressed_token_ids.device)
    return hashes, next_history


@register_tensor_cast_op("engram_inject")
def engram_inject(
    hidden_states: torch.Tensor,
    table_values: torch.Tensor,
    wkv_weight: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    hc_mult: int,
    token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Model Engram wkv projection, normalized gate and residual injection."""
    del wkv_weight, q_weight, k_weight, token_mask
    if hidden_states.ndim not in (3, 4) or table_values.ndim != 4:
        raise ValueError("Engram hidden must be [B,S,Hc,D] or compiled [B,S*Hc,D]")
    if hidden_states.ndim == 3:
        return torch.empty(
            hidden_states.shape[0],
            table_values.shape[1],
            int(hc_mult),
            hidden_states.shape[-1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    return torch.empty_like(hidden_states)


@register_tensor_cast_op("engram_table_lookup")
def engram_table_lookup(
    hash_ids: torch.Tensor,
    table_rows: int,
    head_dim: int,
    row_bytes: int,
    shard_world_size: int,
) -> torch.Tensor:
    """Fetch and dequantize row-sharded FP8 Engram embeddings."""
    del table_rows, row_bytes, shard_world_size
    return torch.empty((*hash_ids.shape, int(head_dim)), dtype=torch.bfloat16, device=hash_ids.device)


@register_tensor_cast_op("fp4_kv_quantize")
def _(
    x: torch.Tensor,
    group_size: int,
    scale_dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Represent a KV/index stream as FP4 payload plus per-group scales."""

    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError(f"FP4 group_size must be positive, got {group_size}")
    groups = (int(x.shape[-1]) + group_size - 1) // group_size
    scale_shape = (*x.shape[:-1], groups)
    if x.is_meta:
        return torch.empty_like(x, dtype=torch.int4), torch.empty(scale_shape, dtype=scale_dtype, device=x.device)
    values = x.float()
    padded = torch.nn.functional.pad(values, (0, groups * group_size - values.shape[-1]))
    grouped = padded.reshape(*values.shape[:-1], groups, group_size)
    scales = grouped.abs().amax(dim=-1).clamp_min(1e-8)
    try:
        quantized = torch.round(grouped / scales.unsqueeze(-1)).clamp(-8, 7).reshape(padded.shape)
        payload = quantized[..., : values.shape[-1]].to(torch.int4)
    except NotImplementedError:
        # CPU PyTorch has no arithmetic implementation for int4; preserve the
        # payload dtype/shape contract for eager graph probes.
        payload = torch.empty_like(x, dtype=torch.int4)
    return payload, scales.to(scale_dtype)


@register_tensor_cast_op("fp4_kv_dequantize")
def _(
    payload: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode an FP4 payload for sparse attention semantic consumption."""

    if payload.is_meta or scales.is_meta:
        return torch.empty_like(payload, dtype=out_dtype)
    expanded = scales.float().repeat_interleave(int(group_size), dim=-1)[..., : payload.shape[-1]]
    try:
        return (payload.float() * expanded).to(out_dtype)
    except NotImplementedError:
        return torch.empty_like(payload, dtype=out_dtype)


@register_tensor_cast_op("csa2_compressor")
@register_tensor_cast_op("csa2_compressor_v2")
def _(
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    compress_ratio: int,
    head_dim: int,
    rope_head_dim: int,
    rotate: bool,
    seq_lens: Optional[torch.Tensor] = None,
    query_lens: Optional[torch.Tensor] = None,
    pending_kv: Optional[torch.Tensor] = None,
    pending_scores: Optional[torch.Tensor] = None,
    start_pos: int = 0,
    compressed_slot_mapping: Optional[torch.Tensor] = None,
    wkv_weight: Optional[torch.Tensor] = None,
    wgate_weight: Optional[torch.Tensor] = None,
    norm_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Propagate r=1/r=2 compressed-KV and decode-tail shapes.

    The pending tensors are explicit graph inputs so chunked prefill/decode can
    carry an incomplete r=2 group without process-global mutable state.
    """

    del rope_head_dim, rotate
    if hidden_states.ndim == 3:
        batch, seq_len, _ = hidden_states.shape
        logical_hidden_states = hidden_states
    elif hidden_states.ndim == 2:
        batch = 1
        for lengths in (query_lens, seq_lens):
            if isinstance(lengths, torch.Tensor) and lengths.numel() > 0:
                batch = int(lengths.numel())
                break
        total_tokens, hidden = hidden_states.shape
        seq_len = (int(total_tokens) + batch - 1) // batch
        if not hidden_states.is_meta and int(total_tokens) % batch == 0:
            logical_hidden_states = hidden_states.reshape(batch, int(total_tokens) // batch, hidden)
            seq_len = int(total_tokens) // batch
        else:
            logical_hidden_states = hidden_states
    else:
        raise ValueError(f"CSA2 compressor expects [B,S,D] or token-local [T,D], got {tuple(hidden_states.shape)}")
    ratio = int(compress_ratio)
    if ratio not in (1, 2):
        raise ValueError(f"CSA2 compressor supports ratios 1 and 2, got {ratio}")
    start_pos = max(int(start_pos), 0)
    rows = (start_pos + seq_len) // ratio - start_pos // ratio
    tail_rows = (start_pos + seq_len) % ratio

    # TensorCast normally traces this op with Meta tensors.  Keep that path
    # allocation-only, while making the eager fallback numerically meaningful
    # for unit tests and CPU graph probes.
    if hidden_states.is_meta:
        compressed = torch.empty(batch, rows, int(head_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        tail_kv = torch.empty(batch, tail_rows, int(head_dim), dtype=torch.float32, device=hidden_states.device)
        tail_scores = torch.empty(batch, tail_rows, int(head_dim), dtype=torch.float32, device=hidden_states.device)
        return compressed, torch.empty_like(kv_cache), tail_kv, tail_scores

    if logical_hidden_states.ndim != 3:
        raise ValueError("Eager token-local CSA2 input must divide evenly across requests")
    if wkv_weight is None:
        projected = torch.nn.functional.pad(
            logical_hidden_states, (0, max(head_dim - logical_hidden_states.shape[-1], 0))
        )[..., :head_dim].float()
    else:
        projected = torch.nn.functional.linear(logical_hidden_states.float(), wkv_weight.float())
    scores = (
        torch.nn.functional.linear(logical_hidden_states.float(), wgate_weight.float())
        if ratio == 2 and wgate_weight is not None
        else projected
    )

    if ratio == 1:
        pooled = projected
        tail_kv = projected[:, :0]
        tail_scores = scores[:, :0]
    else:
        use_pending = start_pos % ratio != 0
        old_kv = (
            pending_kv.float() if use_pending and pending_kv is not None and pending_kv.numel() else projected[:, :0]
        )
        old_scores = (
            pending_scores.float()
            if use_pending and pending_scores is not None and pending_scores.numel()
            else scores[:, :0]
        )
        if old_scores.ndim == 2 and old_scores.numel():
            old_scores = old_scores.unsqueeze(-1).expand(-1, -1, head_dim)
        all_kv = torch.cat((old_kv, projected), dim=1)
        all_scores = torch.cat((old_scores, scores), dim=1)
        # The pending prefix completes the group at the beginning of a decode
        # call; prefill starts with an empty prefix and is grouped normally.
        complete = (all_kv.shape[1] // ratio) * ratio
        grouped_kv = all_kv[:, :complete].reshape(batch, -1, ratio, head_dim)
        grouped_scores = all_scores[:, :complete].reshape(batch, -1, ratio, head_dim)
        pooled = (grouped_kv * grouped_scores.softmax(dim=2)).sum(dim=2)
        tail_kv = all_kv[:, complete:]
        tail_scores = all_scores[:, complete:]
    # RMSNorm and compressed-RoPE are represented as shape-preserving semantic
    # stages here; the production kernel applies the configured theta/FP4.
    pooled = pooled / torch.sqrt(pooled.square().mean(dim=-1, keepdim=True) + 1e-20)
    pooled = pooled.to(hidden_states.dtype)
    if pooled.shape[1] != rows:
        pooled = pooled[:, -rows:] if rows else pooled[:, :0]
    # Cache writes follow the two branches that consume this unrotated latent.
    return pooled, kv_cache.clone(), tail_kv, tail_scores


@register_tensor_cast_op("csa2_cache_update")
def _(
    cache: torch.Tensor,
    update: torch.Tensor,
    slot_mapping: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Publish a source-owned cache plane with a dependency on its new rows."""
    if cache.is_meta or update.is_meta:
        return torch.empty_like(cache)
    result = cache.clone()
    if update.numel() == 0:
        return result
    flat = result.reshape(-1, result.shape[-1])
    values = update.reshape(-1, update.shape[-1]).to(result.dtype)
    if slot_mapping is None or slot_mapping.numel() == 0:
        flat[: values.shape[0]].copy_(values)
    else:
        slots = slot_mapping.reshape(-1).long()[: values.shape[0]]
        valid = (slots >= 0) & (slots < flat.shape[0])
        source = values[: slots.shape[0]]
        if result.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
            # CPU indexing has no FP8 kernel; byte views preserve the exact payload.
            flat.view(torch.uint8).index_copy_(0, slots[valid], source.view(torch.uint8)[valid])
        else:
            flat.index_copy_(0, slots[valid], source[valid])
    return result


@register_tensor_cast_op("csa2_indexer")
def _(
    q_states: torch.Tensor,
    index_k: torch.Tensor,
    index_topk: int,
    candidate_mask: Optional[torch.Tensor] = None,
    compressed_valid_len: Optional[int] = None,
    index_k_scale: Optional[torch.Tensor] = None,
    index_k_group_size: int = 32,
    query_group_size: int = 32,
    compress_ratio: int = 1,
    rope_head_dim: int = 0,
    compress_rope_theta: float = 160000.0,
    start_pos: int = 0,
    owns_index_k: bool = False,
    index_source_dim: int = 0,
) -> torch.Tensor:
    """Propagate CSA2 indexer top-k shape with optional candidate masking."""

    del query_group_size, rope_head_dim, compress_rope_theta
    del owns_index_k, index_source_dim

    batch, seq_len = q_states.shape[:2]
    batched_keys = index_k.ndim == 3 and int(index_k.shape[0]) == batch
    if batched_keys:
        active = int(index_k.shape[1])
    elif index_k.ndim >= 3:
        active = 1
        for dim in index_k.shape[:-1]:
            active *= int(dim)
    else:
        active = int(index_k.shape[0]) if index_k.ndim else 0
    if candidate_mask is not None and candidate_mask.ndim >= 2:
        active = min(active, int(candidate_mask.shape[-1]))
    if compressed_valid_len is not None:
        active = min(active, max(int(compressed_valid_len), 0))
    topk = min(int(index_topk), max(active, 1))
    if q_states.is_meta or index_k.is_meta:
        return torch.empty(batch, seq_len, topk, dtype=torch.int32, device=q_states.device)
    keys = index_k.float() if batched_keys else index_k.reshape(-1, index_k.shape[-1]).float()[:active]
    if index_k_scale is not None:
        scales = index_k_scale.float()
        scales = scales.repeat_interleave(int(index_k_group_size), dim=-1)
        keys = keys * scales[..., : keys.shape[-1]]
    q = q_states.float().mean(dim=-2)
    width = min(q.shape[-1], keys.shape[-1])
    if batched_keys:
        score = torch.matmul(q[..., :width], keys[..., :width].transpose(-1, -2))
    else:
        score = torch.matmul(q[..., :width], keys[:, :width].transpose(0, 1))
    if candidate_mask is not None:
        mask = candidate_mask.reshape(batch, -1)
        mask = mask[:, : score.shape[-1]]
        if mask.shape[-1] < score.shape[-1]:
            mask = torch.nn.functional.pad(mask, (0, score.shape[-1] - mask.shape[-1]), value=False)
        score = score.masked_fill(~mask[:, None, :], -torch.inf)
    reachable = (int(start_pos) + torch.arange(1, seq_len + 1, device=q_states.device)) // max(int(compress_ratio), 1)
    reachable = reachable.clamp(max=active)
    score = score.masked_fill(torch.arange(score.shape[-1], device=score.device) >= reachable[:, None], -torch.inf)
    values, indices = score.topk(topk, dim=-1, largest=True, sorted=False)
    order = indices.argsort(dim=-1)
    indices = indices.gather(-1, order)
    valid = torch.isfinite(values).gather(-1, order)
    return torch.where(valid, indices, -1).to(torch.int32)


@register_tensor_cast_op("csa2_compressed_rope")
def _(
    latent: torch.Tensor,
    rope_head_dim: int,
    compress_ratio: int,
    start_pos: int,
    rope_theta: float,
) -> torch.Tensor:
    """Apply the compressed-position RoPE domain before FP4 cache storage."""
    del rope_head_dim, compress_ratio, start_pos, rope_theta
    return torch.empty_like(latent)


@register_tensor_cast_op("csa2_index_select")
def _(
    query_payload: torch.Tensor,
    query_scale: Optional[torch.Tensor],
    weights: torch.Tensor,
    index_k: torch.Tensor,
    index_k_scale: Optional[torch.Tensor],
    index_topk: int,
    compress_ratio: int,
    start_pos: int,
    candidate_ids: Optional[torch.Tensor],
    candidate_block_size: int,
    candidate_topk_blocks: int,
    is_candidate_source: bool,
    uses_candidates: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select compressed positions and optionally publish score-derived blocks."""
    batch, seq_len, heads, width = query_payload.shape
    if weights.shape != (batch, seq_len, heads):
        raise ValueError("CSA2 index weights must have one value per query head")
    batched_keys = index_k.ndim == 3 and index_k.shape[0] == batch
    active = index_k.shape[1] if batched_keys else index_k.numel() // width
    topk = min(int(index_topk), max(int(active), 1))
    candidate_width = int(candidate_topk_blocks) if is_candidate_source else 0
    selected = torch.empty(batch, seq_len, topk, dtype=torch.int32, device=query_payload.device)
    candidates = torch.empty(batch, seq_len, candidate_width, dtype=torch.int32, device=query_payload.device)
    if query_payload.is_meta or index_k.is_meta:
        return selected, candidates
    if not active:
        return selected.fill_(-1), candidates.fill_(-1)

    key = index_k.float() if batched_keys else index_k.reshape(-1, width).float()[:active]
    if index_k_scale is not None:
        scale = (
            index_k_scale.float()
            if batched_keys
            else index_k_scale.reshape(-1, index_k_scale.shape[-1]).float()[:active]
        )
        key = key * scale.repeat_interleave(max(width // scale.shape[-1], 1), dim=-1)[..., :width]
    if query_scale is None:
        query = query_payload.float()
    else:
        groups = query_scale.shape[-1]
        query = (
            query_payload.float() * query_scale.float().repeat_interleave(max(width // groups, 1), dim=-1)[..., :width]
        )
    scores = torch.einsum("bshd,bkd->bshk" if batched_keys else "bshd,kd->bshk", query, key).relu_()
    scores = (scores * weights.float().unsqueeze(-1)).sum(dim=2)
    reachable = (int(start_pos) + torch.arange(1, seq_len + 1, device=query_payload.device)) // int(compress_ratio)
    scores = scores.masked_fill(
        torch.arange(active, device=query_payload.device) >= reachable.view(1, -1, 1), -torch.inf
    )
    if uses_candidates:
        if candidate_ids is None:
            raise ValueError("CSA2 reindex requires candidate block IDs from its source")
        blocks = torch.arange(active, device=query_payload.device) // int(candidate_block_size)
        allowed = (blocks.view(1, 1, 1, -1) == candidate_ids.long().unsqueeze(-1)).any(dim=2)
        scores = scores.masked_fill(~allowed, -torch.inf)
    if is_candidate_source:
        block_size = int(candidate_block_size)
        padded = torch.nn.functional.pad(scores, (0, -active % block_size), value=-torch.inf)
        block_scores = padded.reshape(batch, seq_len, -1, block_size).amax(dim=-1)
        newest = ((reachable - 1).clamp_min(0) // block_size).clamp_max(block_scores.shape[-1] - 1)
        block_scores.scatter_(-1, newest.view(1, -1, 1).expand(batch, -1, 1), torch.inf)
        block_topk = min(candidate_width, block_scores.shape[-1])
        candidate_values, candidate_indices = block_scores.topk(block_topk, dim=-1)
        candidates.fill_(-1)
        candidates[..., :block_topk] = torch.where(candidate_values > -torch.inf, candidate_indices, -1).to(torch.int32)
    values, indices = scores.topk(topk, dim=-1, sorted=False)
    indices = indices.sort(dim=-1).values
    valid = torch.isfinite(scores.gather(-1, indices))
    return torch.where(valid, indices, -1).to(torch.int32), candidates


@register_tensor_cast_op("csa2_query_quant")
def _(query: torch.Tensor, group_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the indexer query to the fixed MXFP4 payload/scale format."""
    if int(group_size) != 32:
        raise ValueError("CSA2 query quantization uses fixed MXFP4 group size 32")
    return torch.ops.tensor_cast.dynamic_quantize_mxfp4(query, 32)


@register_tensor_cast_op("csa2_prepare_indices")
def _(indices: torch.Tensor, positions: torch.Tensor, window_width: int) -> torch.Tensor:
    """Convert compressed positions to the shared window/compressed index domain."""
    valid = indices >= 0
    shifted = torch.where(valid, indices + int(window_width), torch.full_like(indices, -1))
    return shifted.to(torch.int32)


@register_tensor_cast_op("csa2_select_candidate_blocks")
def _(compressed_kv: torch.Tensor, block_size: int, topk_blocks: int) -> torch.Tensor:
    """Return one candidate mask for the L20 candidate producer."""

    rows = int(compressed_kv.shape[1]) if compressed_kv.ndim >= 2 else 0
    blocks = max((rows + int(block_size) - 1) // int(block_size), 1)
    width = min(int(topk_blocks), blocks)
    if compressed_kv.is_meta:
        return torch.empty(compressed_kv.shape[0], rows, dtype=torch.bool, device=compressed_kv.device)
    scores = compressed_kv.float().pow(2).mean(dim=-1)
    padded = torch.nn.functional.pad(scores, (0, blocks * int(block_size) - rows), value=-torch.inf)
    block_scores = padded.reshape(compressed_kv.shape[0], blocks, int(block_size)).amax(dim=-1)
    latest = torch.zeros_like(block_scores, dtype=torch.bool)
    latest[:, max(0, (rows - 1) // int(block_size))] = True
    chosen = block_scores.topk(width, dim=-1).indices
    mask = torch.zeros_like(block_scores, dtype=torch.bool).scatter(1, chosen, True) | latest
    return mask.repeat_interleave(int(block_size), dim=-1)[:, :rows]


@register_tensor_cast_op("sparse_index_gather_attention")
def _(
    q: torch.Tensor,
    window_kv: torch.Tensor,
    compressed_kv: Optional[torch.Tensor],
    topk_indices: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    head_dim: int,
    window_valid_len: Optional[torch.Tensor] = None,
    compressed_valid_len: Optional[torch.Tensor] = None,
    candidate_mask: Optional[torch.Tensor] = None,
    kv_dependency: Optional[torch.Tensor] = None,
    compressed_kv_scale: Optional[torch.Tensor] = None,
    compressed_group_size: int = 16,
    layer_id: int = -1,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """CSA2 gather attention with physically separate cache streams.

    ``topk_indices`` contains logical positions: non-negative values address
    the window stream, while compressed positions are resolved against the
    separately-owned ``compressed_kv`` stream by the eventual kernel.  Invalid
    positions are encoded as ``-1`` and remain in the index bandwidth count.
    """

    batch, seq_len, num_heads, _ = q.shape
    if q.is_meta or window_kv.is_meta or (compressed_kv is not None and compressed_kv.is_meta):
        return torch.empty(batch, seq_len, num_heads, int(head_dim), dtype=out_dtype, device=q.device)
    del kv_dependency

    def flatten_cache(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 4:
            return value.reshape(value.shape[0], -1, value.shape[-1])
        if value.ndim == 3 and value.shape[0] != batch:
            return value.reshape(1, -1, value.shape[-1]).expand(batch, -1, -1)
        return value

    window = flatten_cache(window_kv).float()
    compressed = flatten_cache(compressed_kv).float() if compressed_kv is not None else None
    if compressed is not None and compressed_kv_scale is not None:
        scales = flatten_cache(compressed_kv_scale).float()
        expanded_scales = scales.repeat_interleave(int(compressed_group_size), dim=-1)
        compressed = compressed * expanded_scales[..., : compressed.shape[-1]]
    qf = q.float()
    result = torch.zeros(batch, seq_len, num_heads, int(head_dim), dtype=torch.float32, device=q.device)
    physical_window_width = min(int(window.shape[1]), int(topk_indices.shape[-1]))

    def length_for(value: Optional[torch.Tensor], batch_id: int, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, torch.Tensor) and value.numel():
            try:
                return int(value.reshape(-1)[min(batch_id, value.numel() - 1)].item())
            except (RuntimeError, ValueError):
                return default
        return int(value)

    for b in range(batch):
        for t in range(seq_len):
            idx = topk_indices[b, t].long()
            valid = idx >= 0
            window_limit = min(length_for(window_valid_len, b, window.shape[1]), window.shape[1])
            window_width = physical_window_width
            compressed_limit = min(
                length_for(compressed_valid_len, b, compressed.shape[1] if compressed is not None else 0),
                compressed.shape[1] if compressed is not None else 0,
            )
            k_parts, v_parts = [], []
            if window_width and window.shape[1]:
                wi = idx[:window_width].clamp(0, max(window.shape[1] - 1, 0))
                k_parts.append(window[b, wi])
                v_parts.append(valid[:window_width] & (idx[:window_width] < window_limit))
            if compressed is not None and compressed.shape[1] and idx.numel() > window_width:
                ci = (idx[window_width:] - window_width).clamp(0, max(compressed.shape[1] - 1, 0))
                k_parts.append(compressed[b, ci])
                compressed_valid = valid[window_width:] & ((idx[window_width:] - window_width) < compressed_limit)
                if candidate_mask is not None:
                    cm = candidate_mask.reshape(batch, -1)[b]
                    if cm.numel():
                        ci_unclamped = (idx[window_width:] - window_width).clamp(0, cm.shape[0] - 1)
                        compressed_valid = compressed_valid & cm[ci_unclamped]
                    else:
                        compressed_valid = torch.zeros_like(compressed_valid)
                v_parts.append(compressed_valid)
            if not k_parts:
                continue
            keys = torch.cat(k_parts, dim=0)
            valid_mask = torch.cat(v_parts, dim=0)
            keys = keys.masked_fill(~valid_mask[:, None], 0)
            scores = torch.einsum("hd,kd->hk", qf[b, t, :, :head_dim], keys) * float(softmax_scale)
            scores = scores.masked_fill(~valid_mask[None, :], -torch.inf)
            if attn_sink is not None and attn_sink.numel():
                sink = attn_sink.reshape(-1)[:num_heads].to(scores.dtype).reshape(num_heads, 1)
                probs = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
            else:
                probs = torch.softmax(scores, dim=-1)
            result[b, t] = torch.einsum("hk,kd->hd", probs, keys)
    return result.to(out_dtype)
