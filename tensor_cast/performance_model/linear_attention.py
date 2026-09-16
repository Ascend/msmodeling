"""Shared analytic cost formulas for gated-delta linear attention.

The semantic operators stay close to their model wrappers, while this module
owns the algorithmic KDA costs shared by Qwen, Kimi, Bailing, and GLM.  Model
specific estimators supply their gate semantics and persistent-state layout.
"""

import torch

from .cost_utils import (
    accumulate_compute_ops,
    byte_count,
    elementwise_sigmoid_ops,
    elementwise_silu_ops,
    elementwise_softplus_ops,
    l2norm_ops,
    rmsnorm_ops,
)
from .op_invoke_info import OpInvokeInfo


_SCRATCH_ROUND_TRIP_TRAFFIC_FACTOR = 2
_CHUNK_ACTIVATION_TOKEN_SCRATCH_ROUND_TRIPS = 2
_CHUNK_FP32_VECTOR_SCRATCH_ROUND_TRIPS = 2
_CHUNK_FP32_MATRIX_SCRATCH_BUFFERS = 4
_CHUNK_FP32_SCALAR_VECTOR_WIDTH = 3
CHUNK_EXTRA_STATIC_KERNELS = 8


def chunk_gated_delta_rule_ops(
    batch_size: int,
    seq_len: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    chunk_size: int = 64,
) -> tuple[int, int, int]:
    """Return MMA, activation-dtype GP, and FP32 GP costs for chunk KDA."""
    padded_seq_len = ((seq_len + chunk_size - 1) // chunk_size) * chunk_size
    num_chunks = padded_seq_len // chunk_size
    batch_heads = batch_size * num_v_heads
    valid_positions = batch_heads * seq_len
    total_positions = batch_heads * padded_seq_len
    total_chunk_pairs = batch_heads * num_chunks * chunk_size * chunk_size

    intra_chunk_mma_ops = total_chunk_pairs * (head_k_dim * 4 + head_v_dim * 2)
    inter_chunk_mma_ops = (
        total_chunk_pairs * (head_k_dim + head_v_dim) * 2 + total_positions * head_k_dim * head_v_dim * 6
    )
    qk_l2norm_gp_ops = l2norm_ops(valid_positions, head_k_dim) * 2
    prefix_correction_gp_ops = batch_heads * num_chunks * (chunk_size - 1) * chunk_size * (2 * chunk_size - 1) // 3

    # After the explicit FP32 cast, KDA recurrence, exponentials, cumsums,
    # masking, and gated updates execute in FP32.
    chunk_rule_fp32_gp_ops = (
        total_positions * head_k_dim
        + total_positions * (head_k_dim + head_v_dim)
        + total_positions * 3
        + total_chunk_pairs * 6
        + prefix_correction_gp_ops
        + total_positions * head_k_dim
        + total_positions * head_v_dim * 2
        + batch_heads * num_chunks * (2 * head_k_dim * head_v_dim + 1)
    )
    return intra_chunk_mma_ops + inter_chunk_mma_ops, qk_l2norm_gp_ops, chunk_rule_fp32_gp_ops


def recurrent_gated_delta_rule_ops(
    batch_size: int,
    seq_len: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[int, int, int]:
    """Return MMA, activation-dtype GP, and FP32 GP costs for recurrent KDA."""
    num_tokens = batch_size * seq_len
    total_positions = num_tokens * num_v_heads
    recurrent_mma_ops = num_tokens * num_v_heads * head_k_dim * head_v_dim * 4
    qk_l2norm_gp_ops = l2norm_ops(total_positions, head_k_dim) * 2
    recurrent_fp32_gp_ops = (
        total_positions * head_k_dim
        + total_positions * (head_v_dim * 2 + 2)
        + total_positions * head_k_dim * head_v_dim * 2
    )
    return recurrent_mma_ops, qk_l2norm_gp_ops, recurrent_fp32_gp_ops


def gated_delta_rule_ops(
    batch_size: int,
    seq_len: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    use_recurrent: bool,
    chunk_size: int = 64,
) -> tuple[int, int, int]:
    """Select the KDA algorithm while preserving its compute-dtype buckets."""
    if use_recurrent:
        return recurrent_gated_delta_rule_ops(batch_size, seq_len, num_v_heads, head_k_dim, head_v_dim)
    return chunk_gated_delta_rule_ops(batch_size, seq_len, num_v_heads, head_k_dim, head_v_dim, chunk_size)


def chunk_gated_delta_rule_scratch_bytes(
    batch_size: int,
    seq_len: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    chunk_size: int,
    activation_dtype: torch.dtype,
) -> int:
    """Return HBM round-trip traffic for the shared chunk-KDA workspace."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    padded_seq_len = ((seq_len + chunk_size - 1) // chunk_size) * chunk_size
    num_chunks = padded_seq_len // chunk_size
    batch_heads = batch_size * num_v_heads
    padded_positions = batch_heads * padded_seq_len
    activation_token_scratch = byte_count(padded_positions * (head_k_dim + head_v_dim), activation_dtype)
    fp32_vector_scratch = byte_count(
        padded_positions * (2 * head_k_dim + head_v_dim + _CHUNK_FP32_SCALAR_VECTOR_WIDTH), torch.float32
    )
    fp32_matrix_scratch = byte_count(
        batch_heads * num_chunks * chunk_size * chunk_size * _CHUNK_FP32_MATRIX_SCRATCH_BUFFERS, torch.float32
    )
    return (
        _CHUNK_ACTIVATION_TOKEN_SCRATCH_ROUND_TRIPS * activation_token_scratch
        + _CHUNK_FP32_VECTOR_SCRATCH_ROUND_TRIPS * fp32_vector_scratch
        + _SCRATCH_ROUND_TRIP_TRAFFIC_FACTOR * fp32_matrix_scratch
    )


def _state_bytes(
    batch_size: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    state_dtype: torch.dtype = torch.float32,
) -> int:
    return byte_count(batch_size * num_v_heads * head_k_dim * head_v_dim, state_dtype)


def _add_chunk_scratch_memory(
    properties: OpInvokeInfo.PerformanceProperties,
    batch_size: int,
    seq_len: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    chunk_size: int,
    activation_dtype: torch.dtype,
) -> None:
    properties.memory_readwrite_bytes += chunk_gated_delta_rule_scratch_bytes(
        batch_size,
        seq_len,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        chunk_size,
        activation_dtype,
    )
    properties.extra_static_cost_count += CHUNK_EXTRA_STATIC_KERNELS


def _add_state_memory(
    properties: OpInvokeInfo.PerformanceProperties,
    batch_size: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    state_read_passes: int,
    state_write_passes: int,
) -> None:
    if state_read_passes < 0 or state_write_passes < 0:
        raise ValueError(
            "Linear attention state pass counts must be non-negative, "
            f"got read={state_read_passes}, write={state_write_passes}."
        )
    state_bytes = _state_bytes(batch_size, num_v_heads, head_k_dim, head_v_dim)
    properties.memory_read_bytes += state_read_passes * state_bytes
    properties.memory_write_bytes += state_write_passes * state_bytes


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.linear_attn_apply_padding_mask.default)
def _padding_mask_properties(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    return op_invoke_info.get_memory_access_properties()


def _causal_conv_properties(op_invoke_info: OpInvokeInfo, include_state: bool) -> OpInvokeInfo.PerformanceProperties:
    mixed_qkv = op_invoke_info.args[0]
    conv_kernel_size = op_invoke_info.args[1]
    batch_size = mixed_qkv.size(0)
    conv_dim = mixed_qkv.size(1)
    seq_len = mixed_qkv.size(2)
    properties = op_invoke_info.get_memory_access_properties()

    conv_gp_ops = batch_size * seq_len * conv_dim * conv_kernel_size * 2 + elementwise_silu_ops(
        batch_size * seq_len * conv_dim
    )
    accumulate_compute_ops(properties, mixed_qkv.dtype, gp_ops=conv_gp_ops)
    properties.memory_read_bytes += byte_count(conv_dim * conv_kernel_size, mixed_qkv.dtype)
    if include_state:
        properties.memory_readwrite_bytes += byte_count(batch_size * conv_dim * conv_kernel_size, mixed_qkv.dtype)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.linear_attn_causal_conv.default)
def _causal_conv_prefill_properties(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    return _causal_conv_properties(op_invoke_info, include_state=False)


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.linear_attn_causal_conv_update.default)
def _causal_conv_decode_properties(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    return _causal_conv_properties(op_invoke_info, include_state=True)


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.linear_attn_fused_gdn_gating.default)
def _gdn_gating_properties(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    query = op_invoke_info.args[0]
    b = op_invoke_info.args[2]
    a_log = op_invoke_info.args[4]
    dt_bias = op_invoke_info.args[5]
    num_v_heads = op_invoke_info.args[6]

    batch_size = query.size(0)
    seq_len = query.size(1)
    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={4, 5})
    properties.memory_read_bytes += byte_count(num_v_heads, a_log.dtype)
    properties.memory_read_bytes += byte_count(num_v_heads, dt_bias.dtype)

    num_gate_elements = batch_size * seq_len * num_v_heads
    beta_gp_ops = elementwise_sigmoid_ops(num_gate_elements)
    g_gp_ops = num_v_heads + num_gate_elements * (1 + elementwise_softplus_ops(1) + 1 + 1)
    accumulate_compute_ops(properties, b.dtype, gp_ops=beta_gp_ops)
    accumulate_compute_ops(properties, torch.float32, gp_ops=g_gp_ops)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.linear_attn_chunk_gated_delta_rule.default)
def _chunk_gated_delta_rule_properties(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    query = op_invoke_info.args[0]
    value = op_invoke_info.args[2]
    chunk_size = op_invoke_info.args[5]
    state_read_passes = op_invoke_info.args[6]
    state_write_passes = op_invoke_info.args[7]

    batch_size = query.size(0)
    seq_len = query.size(1)
    num_v_heads = query.size(2)
    head_k_dim = query.size(3)
    head_v_dim = value.size(3)

    properties = op_invoke_info.get_memory_access_properties()
    _add_state_memory(
        properties,
        batch_size,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        state_read_passes,
        state_write_passes,
    )
    _add_chunk_scratch_memory(
        properties,
        batch_size,
        seq_len,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        chunk_size,
        query.dtype,
    )
    attn_mma_ops, hidden_gp_ops, fp32_gp_ops = chunk_gated_delta_rule_ops(
        batch_size,
        seq_len,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        chunk_size,
    )
    accumulate_compute_ops(properties, query.dtype, gp_ops=hidden_gp_ops)
    accumulate_compute_ops(properties, torch.float32, mma_ops=attn_mma_ops, gp_ops=fp32_gp_ops)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.linear_attn_recurrent_gated_delta_rule.default)
def _recurrent_gated_delta_rule_properties(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    query = op_invoke_info.args[0]
    value = op_invoke_info.args[2]
    state_read_passes = op_invoke_info.args[5]
    state_write_passes = op_invoke_info.args[6]

    batch_size = query.size(0)
    seq_len = query.size(1)
    num_v_heads = query.size(2)
    head_k_dim = query.size(3)
    head_v_dim = value.size(3)

    properties = op_invoke_info.get_memory_access_properties()
    _add_state_memory(
        properties,
        batch_size,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        state_read_passes,
        state_write_passes,
    )
    recurrent_mma_ops, hidden_gp_ops, fp32_gp_ops = recurrent_gated_delta_rule_ops(
        batch_size,
        seq_len,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )
    accumulate_compute_ops(properties, query.dtype, gp_ops=hidden_gp_ops)
    accumulate_compute_ops(properties, torch.float32, mma_ops=recurrent_mma_ops, gp_ops=fp32_gp_ops)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.linear_attn_gated_rmsnorm.default)
def _gated_rmsnorm_properties(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    core_attn_out = op_invoke_info.args[0]
    batch_size = core_attn_out.size(0)
    seq_len = core_attn_out.size(1)
    num_v_heads = core_attn_out.size(2)
    head_v_dim = core_attn_out.size(3)
    num_rows = batch_size * seq_len * num_v_heads
    num_elements = num_rows * head_v_dim

    properties = op_invoke_info.get_memory_access_properties()
    gated_rmsnorm_gp_ops = (
        rmsnorm_ops(num_rows, head_v_dim) + num_elements + elementwise_silu_ops(num_elements) + num_elements
    )
    accumulate_compute_ops(properties, torch.float32, gp_ops=gated_rmsnorm_gp_ops)
    return properties
