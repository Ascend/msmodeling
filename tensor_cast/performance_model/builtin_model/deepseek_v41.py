"""Analytic properties for DeepSeek V4.1 semantic operators."""

import torch
from torch._subclasses.fake_tensor import is_fake

from .. import _accumulate_compute_ops
from ..op_invoke_info import OpInvokeInfo
from ..utils import bytes_of_elements


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.deepseek_v41_vision_rope.default)
def _vision_rope(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    query, key = op.args[:2]
    props = op.get_memory_access_properties()
    _accumulate_compute_ops(props, query.dtype, gp_ops=(query.numel() + key.numel()) * 6)
    return props


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.deepseek_v41_vision_downsample.default)
def _vision_downsample(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    # Pixel-unshuffle is a padded layout transform: no arithmetic, but it reads
    # the ViT output and writes the expanded aligner input once.
    return op.get_memory_access_properties()


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.deepseek_v41_image_merge.default)
def _image_merge(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    return op.get_memory_access_properties()


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.moe_gating_top_k_vl.default)
def _moe_gating_top_k_vl(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    scores, top_k, normalize_weights = op.args[:3]
    num_experts = int(scores.shape[-1])
    num_tokens = scores.numel() // max(num_experts, 1)
    props = op.get_memory_access_properties()
    bias_select = num_tokens * num_experts * 2  # select + bias add
    topk = num_tokens * num_experts
    gather = num_tokens * int(top_k)
    normalize = num_tokens * int(top_k) * 2 if bool(normalize_weights) else 0
    scale = num_tokens * int(top_k)
    _accumulate_compute_ops(
        props,
        scores.dtype,
        gp_ops=bias_select + topk + gather + normalize + scale,
    )
    return props


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.engram_hash_lookup.default)
def _engram_hash_lookup(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    compressed_ids, history, num_hash_cols, max_ngram_size, table_count = op.args[:5]
    props = op.get_memory_access_properties(exclude_output_ids={0, 1})
    tokens = int(compressed_ids.numel())
    props.memory_read_bytes += tokens * compressed_ids.element_size()
    if history is not None:
        props.memory_read_bytes += history.numel() * history.element_size()
    props.memory_read_bytes += tokens * int(table_count) * int(num_hash_cols) * 8
    props.compute_ops.setdefault(torch.float32, OpInvokeInfo.ComputeOps()).gp_ops += (
        tokens * int(table_count) * int(num_hash_cols) * max(int(max_ngram_size), 1) * 4
    )
    return props


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.engram_inject.default)
def _engram_inject(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    hidden, table_values, wkv_weight, q_weight, k_weight, hc_mult = op.args[:6]
    token_mask = op.args[6] if len(op.args) > 6 else None
    # Keep hidden-state read and residual-output write in the generic traffic
    # accounting. Only table/weight inputs are charged explicitly below.
    props = op.get_memory_access_properties(exclude_input_ids={1, 2, 3, 4})
    if hidden.ndim == 4:
        batch, seq, _, dim = (int(x) for x in hidden.shape)
    else:
        batch, _, dim = (int(x) for x in hidden.shape)
        seq = int(table_values.shape[1])
    hash_cols = int(table_values.shape[-2])
    head_dim = int(table_values.shape[-1])
    props.memory_read_bytes += table_values.numel() * table_values.element_size()
    props.memory_read_bytes += (
        wkv_weight.numel() * wkv_weight.element_size()
        + q_weight.numel() * q_weight.element_size()
        + k_weight.numel() * k_weight.element_size()
    )
    mask_gp = batch * seq * int(hc_mult) if token_mask is not None else 0
    props.compute_ops.setdefault(torch.float32, OpInvokeInfo.ComputeOps()).gp_ops += (
        batch * seq * int(hc_mult) * int(dim) * 12 + mask_gp
    )
    props.compute_ops.setdefault(torch.bfloat16, OpInvokeInfo.ComputeOps()).mma_ops += (
        batch * seq * hash_cols * int(head_dim) * int(dim) * (int(hc_mult) + 1)
    )
    return props


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.engram_table_lookup.default)
def _engram_table_lookup(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    hash_ids, table_rows, head_dim, row_bytes, shard_world_size = op.args
    props = op.get_memory_access_properties(exclude_output_ids={0})
    logical_rows = int(hash_ids.numel())
    local_rows = (logical_rows + max(int(shard_world_size), 1) - 1) // max(int(shard_world_size), 1)
    props.host_to_device_bytes += local_rows * int(row_bytes)
    props.host_to_device_serial = True
    props.memory_write_bytes += logical_rows * int(head_dim) * 2
    props.compute_ops.setdefault(torch.float32, OpInvokeInfo.ComputeOps()).gp_ops += local_rows * int(head_dim) * 2
    return props


def _logical_rows(tensor: torch.Tensor, batch: int | None = None) -> int:
    """Return token rows for batched or paged ``[..., width]`` caches."""
    if tensor.ndim < 2:
        return 0
    if batch is not None and tensor.ndim == 3 and int(tensor.shape[0]) == int(batch):
        return int(tensor.shape[1])
    rows = 1
    for dim in tensor.shape[:-1]:
        rows *= int(dim)
    return rows


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.fp4_kv_quantize.default)
def _fp4_kv_quantize(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    x = op.args[0]
    payload, scales = op.out
    props = op.get_memory_access_properties()
    _accumulate_compute_ops(props, torch.float32, gp_ops=2 * x.numel() + scales.numel())
    return props


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.fp4_kv_dequantize.default)
def _fp4_kv_dequantize(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    payload, scales = op.args[:2]
    group_size = max(int(op.args[2]), 1)
    props = op.get_memory_access_properties(exclude_output_ids={0})
    groups = (int(payload.shape[-1]) + group_size - 1) // group_size
    _accumulate_compute_ops(props, torch.float32, gp_ops=payload.numel() + groups * payload.numel())
    return props


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.csa2_compressor.default)
def _csa2_compressor(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    hidden_states = op.args[0]
    ratio = int(op.args[2])
    head_dim = int(op.args[3])
    if hidden_states.ndim == 3:
        batch, seq_len, hidden = hidden_states.shape
    elif hidden_states.ndim == 2:
        total_tokens, hidden = hidden_states.shape
        batch = 1
        for arg_idx in (7, 6):
            lengths = op.args[arg_idx] if len(op.args) > arg_idx else None
            if isinstance(lengths, torch.Tensor) and lengths.numel() > 0:
                batch = int(lengths.numel())
                break
        seq_len = (int(total_tokens) + batch - 1) // batch
    else:
        raise ValueError(f"Unsupported CSA2 compressor hidden shape: {tuple(hidden_states.shape)}")
    start_pos = int(op.args[10]) if len(op.args) > 10 else 0
    slot_mapping = op.args[11] if len(op.args) > 11 else None
    rows = (start_pos + seq_len) // ratio - start_pos // ratio
    props = op.get_memory_access_properties(exclude_output_ids={0, 1, 2}, exclude_input_ids={1})
    # r=2 has wkv+wgate; r=1 has only the plain wkv projection.
    # Count multiply and add separately, as for the standalone GEMM handlers.
    projection_ops = 2 * (2 if ratio == 2 else 1) * batch * seq_len * hidden * head_dim
    pooling_ops = batch * rows * ratio * head_dim * 3 if ratio == 2 else 0
    _accumulate_compute_ops(props, torch.float32, mma_ops=projection_ops, gp_ops=pooling_ops)
    props.memory_write_bytes += batch * rows * head_dim * hidden_states.element_size()
    # The source cache write is indexed, so charge one slot metadata read per
    # produced row when an explicit mapping is present.
    if isinstance(slot_mapping, torch.Tensor):
        props.memory_read_bytes += batch * rows * 4
    return props


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.csa2_compressor_v2.default)
def _csa2_compressor_v2(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    return _csa2_compressor(op)


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.csa2_cache_update.default)
def _csa2_cache_update(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    cache, update = op.args[:2]
    props = op.get_memory_access_properties(exclude_input_ids={0}, exclude_output_ids={0})
    props.memory_write_bytes += bytes_of_elements(update.numel(), cache.dtype)
    return props


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.csa2_index_select.default)
def _csa2_index_select(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    query_payload, _, _, key = op.args[:4]
    batch, seq_len, heads, width = query_payload.shape
    capacity = _logical_rows(key, int(batch))
    ratio, start_pos = int(op.args[6]), int(op.args[7])
    candidates, block_size, candidate_topk = op.args[8:11]
    produces_candidates, uses_candidates = bool(op.args[11]), bool(op.args[12])
    block_size = max(int(block_size), 1)
    scored_rows = 0
    producer_rows = 0
    for token in range(int(seq_len)):
        reachable = min(capacity, max((start_pos + token + 1) // max(ratio, 1), 0))
        producer_rows += reachable
        if uses_candidates:
            candidate_width = int(candidates.shape[-1]) if isinstance(candidates, torch.Tensor) else int(candidate_topk)
            reachable = min(reachable, candidate_width * block_size)
        scored_rows += reachable
    props = op.get_memory_access_properties(exclude_input_ids={3, 4})
    _accumulate_compute_ops(props, query_payload.dtype, mma_ops=2 * batch * heads * scored_rows * width)
    _accumulate_compute_ops(props, torch.float32, gp_ops=batch * heads * scored_rows * 3)
    props.memory_read_bytes += bytes_of_elements(batch * scored_rows * width, key.dtype)
    key_scale = op.args[4]
    if isinstance(key_scale, torch.Tensor):
        props.memory_read_bytes += bytes_of_elements(batch * scored_rows * key_scale.shape[-1], key_scale.dtype)
    if produces_candidates:
        _accumulate_compute_ops(props, torch.float32, gp_ops=batch * producer_rows)
    return props


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.csa2_query_quant.default)
def _csa2_query_quant(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    query = op.args[0]
    props = op.get_memory_access_properties()
    group_size = int(op.args[1]) if len(op.args) > 1 else 32
    groups = (query.shape[-1] + group_size - 1) // group_size
    _accumulate_compute_ops(props, torch.float32, gp_ops=3 * query.numel())
    props.memory_write_bytes += bytes_of_elements(query.numel(), torch.int4)
    props.memory_write_bytes += bytes_of_elements(
        query.numel() // max(query.shape[-1], 1) * groups,
        torch.float8_e8m0fnu,
    )
    return props


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.csa2_prepare_indices.default)
def _csa2_prepare_indices(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    indices = op.args[0]
    props = op.get_memory_access_properties()
    _accumulate_compute_ops(props, torch.int32, gp_ops=indices.numel() * 2)
    return props


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.csa2_indexer.default)
def _csa2_indexer(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    q_states, index_k, index_topk = op.args[:3]
    candidate_mask = op.args[3] if len(op.args) > 3 else None
    compressed_valid_len = op.args[4] if len(op.args) > 4 else None
    index_k_scale = op.args[5] if len(op.args) > 5 else None
    index_k_group_size = max(int(op.args[6]), 1) if len(op.args) > 6 else 32
    query_group_size = max(int(op.args[7]), 1) if len(op.args) > 7 else 32
    compress_ratio = max(int(op.args[8]), 1) if len(op.args) > 8 else 1
    rope_head_dim = max(int(op.args[9]), 0) if len(op.args) > 9 else 0
    start_pos = max(int(op.args[11]), 0) if len(op.args) > 11 else 0
    owns_index_k = bool(op.args[12]) if len(op.args) > 12 else False
    index_source_dim = max(int(op.args[13]), 0) if len(op.args) > 13 else 0
    batch, seq_len = q_states.shape[:2]
    active = _logical_rows(index_k, batch)
    if candidate_mask is not None and isinstance(candidate_mask, torch.Tensor) and candidate_mask.ndim >= 2:
        active = min(active, int(candidate_mask.shape[-1]))
    if compressed_valid_len is not None:
        active = min(active, max(int(compressed_valid_len), 0))
    heads = q_states.shape[-2]
    head_dim = q_states.shape[-1]
    props = op.get_memory_access_properties(exclude_output_ids={0})
    query_elements = int(q_states.numel())
    query_groups = batch * seq_len * int(heads) * ((int(head_dim) + query_group_size - 1) // query_group_size)
    query_quant_gp = 3 * query_elements + query_groups
    props.memory_readwrite_bytes += 2 * query_elements * q_states.element_size() + 2 * query_groups
    _accumulate_compute_ops(props, torch.float32, gp_ops=query_quant_gp)

    if owns_index_k:
        produced_rows = max((start_pos + seq_len) // compress_ratio - start_pos // compress_ratio, 0)
        index_elements = batch * produced_rows * int(head_dim)
        index_groups = batch * produced_rows * ((int(head_dim) + index_k_group_size - 1) // index_k_group_size)
        projection_mma = 2 * batch * produced_rows * index_source_dim * int(head_dim)
        index_rope_gp = batch * produced_rows * min(rope_head_dim, int(head_dim)) * 6
        index_quant_gp = 3 * index_elements + index_groups
        _accumulate_compute_ops(props, q_states.dtype, mma_ops=projection_mma)
        _accumulate_compute_ops(props, torch.float32, gp_ops=index_rope_gp + index_quant_gp)
        props.memory_write_bytes += bytes_of_elements(index_elements, torch.int4) + index_groups
    score_mma = 2 * batch * seq_len * heads * active * head_dim
    _accumulate_compute_ops(props, q_states.dtype, mma_ops=score_mma, gp_ops=batch * seq_len * active)
    props.memory_read_bytes += batch * seq_len * min(int(index_topk), max(active, 1)) * 4
    if index_k_scale is not None:
        scale_groups = (int(index_k.shape[-1]) + index_k_group_size - 1) // index_k_group_size
        props.memory_read_bytes += batch * active * scale_groups * index_k_scale.element_size()
        _accumulate_compute_ops(props, torch.float32, gp_ops=batch * seq_len * active * head_dim)
    return props


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.csa2_compressed_rope.default)
def _csa2_compressed_rope(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    latent = op.args[0]
    rope_head_dim = min(max(int(op.args[1]), 0), int(latent.shape[-1]))
    rows = latent.numel() // max(int(latent.shape[-1]), 1)
    props = op.get_memory_access_properties()
    _accumulate_compute_ops(props, latent.dtype, gp_ops=rows * rope_head_dim * 6)
    return props


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.csa2_select_candidate_blocks.default)
def _csa2_candidates(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    compressed_kv, block_size, topk_blocks = op.args
    rows = compressed_kv.shape[1]
    blocks = min(int(topk_blocks), max((rows + int(block_size) - 1) // int(block_size), 1))
    props = op.get_memory_access_properties(exclude_output_ids={0})
    props.memory_read_bytes += bytes_of_elements(
        int(compressed_kv.shape[0]) * int(rows) * int(compressed_kv.shape[-1]),
        compressed_kv.dtype,
    )
    props.memory_write_bytes += compressed_kv.shape[0] * blocks
    _accumulate_compute_ops(props, torch.float32, gp_ops=compressed_kv.shape[0] * rows)
    return props


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.sparse_index_gather_attention.default)
def _csa2_gather_attention(op: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    q, window_kv, compressed_kv, topk_indices = op.args[:4]
    head_dim = int(op.args[6])
    query_tokens = int(q.shape[0] * q.shape[1])
    topk = int(topk_indices.shape[-1])
    window_valid_len = op.args[7] if len(op.args) > 7 else None
    compressed_valid_len = op.args[8] if len(op.args) > 8 else None
    candidate_mask = op.args[9] if len(op.args) > 9 else None
    compressed_kv_scale = op.args[11] if len(op.args) > 11 else None
    layer_id = int(op.args[13]) if len(op.args) > 13 else -1
    heads = int(q.shape[2])
    block = 64

    def active_len(value, default):
        if isinstance(value, (int, float)):
            return max(int(value), 0)
        if isinstance(value, torch.Tensor) and not is_fake(value) and value.numel():
            try:
                return max(int(value.max().item()), 0)
            except Exception:
                return int(default)
        return int(default)

    window_capacity = _logical_rows(window_kv, int(q.shape[0]))
    compressed_capacity = _logical_rows(compressed_kv, int(q.shape[0])) if compressed_kv is not None else 0
    active_window = min(active_len(window_valid_len, window_capacity), window_capacity)
    if compressed_kv is None and layer_id >= 2:
        # V4.1 reuses one of four physical owner caches.  The semantic op
        # may not carry that cache handle through the HF attention boundary,
        # but consumers still read the selected compressed positions.
        active_compressed = max(topk - active_window, 0)
    else:
        active_compressed = (
            0
            if compressed_kv is None
            else min(active_len(compressed_valid_len, compressed_capacity), compressed_capacity)
        )
    if candidate_mask is not None and isinstance(candidate_mask, torch.Tensor):
        active_compressed = min(active_compressed, int(candidate_mask.shape[-1]))
    active_slots = min(topk, active_window + active_compressed)
    padded_topk = ((max(active_slots, 1) + block - 1) // block) * block
    props = op.get_memory_access_properties(exclude_input_ids={1, 2, 3, 9, 10})
    mma_ops = query_tokens * padded_topk * heads * head_dim * 4
    gp_ops = query_tokens * padded_topk * heads * 8
    _accumulate_compute_ops(props, q.dtype, mma_ops=mma_ops, gp_ops=gp_ops)
    props.memory_read_bytes += query_tokens * min(topk, active_window) * head_dim * window_kv.element_size()
    compressed_read_slots = min(max(topk - active_window, 0), active_compressed)
    if compressed_kv is not None:
        props.memory_read_bytes += bytes_of_elements(
            query_tokens * compressed_read_slots * head_dim,
            compressed_kv.dtype,
        )
    elif layer_id >= 2:
        props.memory_read_bytes += bytes_of_elements(
            query_tokens * compressed_read_slots * head_dim,
            torch.int4,
        )
    if compressed_kv_scale is not None or layer_id >= 2:
        scale_groups = (head_dim + 16 - 1) // 16
        scale_element_size = compressed_kv_scale.element_size() if compressed_kv_scale is not None else 1
        props.memory_read_bytes += query_tokens * compressed_read_slots * scale_groups * scale_element_size
        _accumulate_compute_ops(props, torch.float32, gp_ops=query_tokens * active_compressed * head_dim)
    props.memory_read_bytes += query_tokens * topk * 4
    return props
