import logging
import math
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch

from .. import ops  # noqa: F401
from ..device import DeviceProfile
from ..utils import is_fp8_dtype, performance_dtype
from .bound_analyzer import StatsKey
from .base import PerformanceModel
from .op_estimator_registry import register_op_estimator
from .op_invoke_info import OpInvokeInfo
from .utils import bytes_of_elements, bytes_of_tensor, is_noop_self_copy_op, is_view_op

logger = logging.getLogger(__name__)
# Deduplication: Each (dtype, category) combination is warned only once to avoid hundreds of duplicate logs
_warned_unsupported_dtypes = set()


def _get_device_ops_for_dtype(
    perf_ops: dict[torch.dtype, float],
    dtype: torch.dtype,
) -> Optional[float]:
    return perf_ops.get(performance_dtype(dtype))


def _load_custom_op():
    try:
        custom_op_dir = Path(__file__).resolve().parent / "custom_op"

        if not custom_op_dir.exists():
            logger.warning("custom operator folder %s not found", custom_op_dir)
            return False

        for py_file in custom_op_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            module_name = py_file.stem
            import importlib.util

            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
        return True

    except Exception:
        logger.warning("Failed to load custom op modules ", exc_info=True)
        return False


@OpInvokeInfo.register_op_properties(torch.ops.aten.bmm.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 2
    mat1 = op_invoke_info.args[0]
    mat2 = op_invoke_info.args[1]
    assert isinstance(mat1, torch.Tensor)
    assert isinstance(mat2, torch.Tensor)
    assert mat1.ndim == 3
    assert mat2.ndim == 3
    b = mat1.size(0)
    m = mat1.size(1)
    k = mat1.size(2)
    n = mat2.size(2)
    assert mat2.size(0) == b
    assert mat2.size(1) == k

    mma_ops = b * m * n * k * 2
    if mma_ops == 0:
        return OpInvokeInfo.PerformanceProperties()

    properties = op_invoke_info.get_memory_access_properties()
    properties.compute_ops[mat1.dtype] = OpInvokeInfo.ComputeOps()
    properties.compute_ops[mat1.dtype].mma_ops = mma_ops
    return properties


def _mm_properties_helper(op_invoke_info: OpInvokeInfo, mat1, mat2, bias) -> OpInvokeInfo.PerformanceProperties:
    # Get the logical dimensions of the operation.
    # mat1 is (M, K).
    m = mat1.size(0)
    k = mat1.size(1)
    n = mat2.size(1)

    # Matrix Multiplication: mat1 @ mat2
    # Cost is M * N * K fused multiply-adds (FMAs), which are 2 FLOPs each.
    matmul_ops = m * n * k * 2

    # Bias Addition: ... + bias
    # M * N additions.
    bias_ops = 0
    if bias is not None:
        bias_ops = m * n

    if matmul_ops == 0:
        return OpInvokeInfo.PerformanceProperties()

    properties = op_invoke_info.get_memory_access_properties()
    properties.compute_ops[mat1.dtype] = OpInvokeInfo.ComputeOps()
    properties.compute_ops[mat1.dtype].mma_ops = matmul_ops
    if bias is not None:
        compute_ops = properties.compute_ops.setdefault(bias.dtype, OpInvokeInfo.ComputeOps())
        compute_ops.gp_ops = bias_ops
        properties.compute_ops[bias.dtype] = compute_ops

    return properties


@OpInvokeInfo.register_op_properties(torch.ops.aten.mm.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 2
    return _mm_properties_helper(op_invoke_info, op_invoke_info.args[0], op_invoke_info.args[1], None)


def _static_quant_linear_properties_helper(
    op_invoke_info: OpInvokeInfo, x, w, w_offset, bias, is_int4: bool
) -> OpInvokeInfo.PerformanceProperties:
    # Get the logical dimensions of the operation.
    # x is (M, K).
    m = x.size(0)
    k = x.size(1)

    if is_int4:
        # The new Grouped MatMul + SwiGLU fusion pass uses
        # optimized/tilled weight layouts that break
        # the old hardcoded 'K/2' assumption. We must infer dimensions dynamically.

        # 1. Dynamic packing: Adapt to any storage dtype (uint8=2x, int32=8x) instead of hardcoding '2'.
        pack_factor = (w.element_size() * 8) // 4

        # 2. Conservation law: Total logical values = Physical elements × Packing factor.
        # This remains true regardless of how dimensions are shuffled or tiled.
        logical_total_elements = w.numel() * pack_factor

        if logical_total_elements % k != 0:
            raise AssertionError(
                f"Shape mismatch: Cannot infer logical N. "
                f"Input K={k}, Weight shape={w.shape}, Dtype={w.dtype}. "
                f"Logical elements ({logical_total_elements}) is not divisible by K."
            )

        n = logical_total_elements // k
    else:
        n = w.size(1)

    # Dequantization of weights: dequant(w) if w is int4
    # Here, we suppose HW supports int8 @ int8 but not int8 @ int4 directly.
    # The operation is semantically `(w - w_offset) * w_scale`.
    dequant_ops = 0
    if is_int4:
        if w_offset is not None:
            # K * N subtractions (offset) + K * N multiplications (scale)
            dequant_ops = k * n * 2
        else:
            # K * N multiplications (scale only)
            dequant_ops = k * n

    # Matrix Multiplication: dequant(x) @ dequant(w)
    # Cost is M * N * K fused multiply-adds (FMAs), which are 2 FLOPs each.
    matmul_ops = m * n * k * 2

    # Bias Addition: ... + bias
    # M * N additions.
    bias_ops = 0
    if bias is not None:
        bias_ops = m * n

    if matmul_ops == 0:
        return OpInvokeInfo.PerformanceProperties()

    properties = op_invoke_info.get_memory_access_properties()
    properties.compute_ops[x.dtype] = OpInvokeInfo.ComputeOps()
    properties.compute_ops[x.dtype].mma_ops = matmul_ops
    if is_int4:
        # TODO(jgong5): use fp32 flops for int4->int8, should use something more accurate
        compute_ops = properties.compute_ops.setdefault(torch.float32, OpInvokeInfo.ComputeOps())
        compute_ops.gp_ops = dequant_ops
    if bias is not None:
        compute_ops = properties.compute_ops.setdefault(bias.dtype, OpInvokeInfo.ComputeOps())
        compute_ops.gp_ops += bias_ops
        properties.compute_ops[bias.dtype] = compute_ops

    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.static_quant_linear_int4.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 3
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    w_offset = op_invoke_info.args[3] if len(op_invoke_info.args) > 3 else None
    bias = op_invoke_info.args[6] if len(op_invoke_info.args) > 6 else None
    return _static_quant_linear_properties_helper(op_invoke_info, x, w, w_offset, bias, is_int4=True)


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.static_quant_linear.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 3
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    w_offset = op_invoke_info.args[3] if len(op_invoke_info.args) > 3 else None
    bias = op_invoke_info.args[6] if len(op_invoke_info.args) > 6 else None
    return _static_quant_linear_properties_helper(op_invoke_info, x, w, w_offset, bias, is_int4=False)


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.fp8_linear.default)
@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.mxfp4_linear.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 3
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    bias = op_invoke_info.args[4] if len(op_invoke_info.args) > 4 else None
    return _static_quant_linear_properties_helper(op_invoke_info, x, w, None, bias, is_int4=False)


@OpInvokeInfo.register_op_properties(torch.ops.aten.embedding.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 2
    weight = op_invoke_info.args[0]
    indices = op_invoke_info.args[1]
    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={0})
    properties.memory_read_bytes += bytes_of_tensor(indices, weight.dtype) * weight.shape[-1]
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.aten.index_select.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 3
    input = op_invoke_info.args[0]
    dim = op_invoke_info.args[1]
    index = op_invoke_info.args[2]
    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={0})
    properties.memory_read_bytes += bytes_of_tensor(input) * index.numel() / input.shape[dim]
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.reshape_and_cache.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 4
    key = op_invoke_info.args[0]
    value = op_invoke_info.args[1]
    kv_cache = op_invoke_info.args[2]

    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={2})
    properties.memory_write_bytes += bytes_of_tensor(key, kv_cache.dtype) + bytes_of_tensor(value, kv_cache.dtype)
    return properties


def _attention_properties_helper(
    op_invoke_info: OpInvokeInfo,
    query,
    key,
    seq_lens,
    query_lens,
    softmax_dtype,
) -> OpInvokeInfo.PerformanceProperties:
    block_table = op_invoke_info.args[4]
    if query.ndim == 4:
        # The core computation involves multiplying query tokens for a sequence with all
        # key tokens of that same sequence. Under uniform sequence lengths across the batch,
        # this product is the same for every sequence, and the total across the batch is:
        # batch_size * query_len_per_seq * key_len_per_seq.
        # This gives a measure of the total QK^T and Score*V interactions.
        assert block_table is None, "4D query implies no KV cache; block_table must be None"
        batch_size, query_len_per_seq, num_q_heads, head_size = query.size()
        assert key.ndim == 4, "key size must be 4"
        _, key_len_per_seq, _, _ = key.size()
        context_len_product_sum = batch_size * query_len_per_seq * key_len_per_seq
    else:
        hidden_size = query.size(-1)
        head_size = key.size(-1)
        assert hidden_size % head_size == 0
        num_q_heads = hidden_size // head_size

        context_len_product_sum = torch.sum(query_lens.to(seq_lens.dtype) * seq_lens).item()

    # 1. First Batched Matrix Multiplication (BMM): Q @ K^T
    # For each query head, this is a sum of (num_tokens_per_seq * seq_len) dot products,
    # where each dot product has `head_size` multiply-adds.
    # Total FMA ops = sum(num_tokens_i * seq_len_i) * num_q_heads * head_size
    # Total FLOPs = FMA_ops * 2
    bmm1_ops = context_len_product_sum * num_q_heads * head_size * 2

    # 2. Softmax
    # This operates on the score matrix. The number of elements is sum(num_tokens_i * seq_len_i) * num_q_heads.
    # Each softmax element (exp, sum, div) is often approximated as ~4 FLOPs.
    softmax_ops = context_len_product_sum * num_q_heads * 4

    # 3. Second Batched Matrix Multiplication (BMM): Scores @ V
    # This has the same computational cost as the first BMM.
    # Total FMA ops = sum(num_tokens_i * seq_len_i) * num_q_heads * head_size
    # Total FLOPs = FMA_ops * 2
    bmm2_ops = context_len_product_sum * num_q_heads * head_size * 2

    if block_table is None:
        properties = op_invoke_info.get_memory_access_properties()
    else:
        properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={1, 2})
        properties.memory_read_bytes += torch.sum(
            seq_lens * 2 * bytes_of_elements(key.size(-1) * key.size(-2), key.dtype)
        ).item()

    compute_ops = properties.compute_ops.setdefault(query.dtype, OpInvokeInfo.ComputeOps())
    compute_ops.mma_ops = bmm1_ops + bmm2_ops
    compute_ops = properties.compute_ops.setdefault(softmax_dtype, OpInvokeInfo.ComputeOps())
    compute_ops.gp_ops = softmax_ops

    return properties


def _default_query_lens_and_request_total_seq_lens(
    query,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = query.size(-2)
    batch_size = query.size(0) if query.ndim == 3 else 1
    request_total_seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long)
    query_lens = torch.full((batch_size,), seq_len, dtype=torch.long)
    return query_lens, request_total_seq_lens


def _normalize_query_lens_and_request_total_seq_lens(
    query: torch.Tensor,
    query_lens: Optional[torch.Tensor],
    request_total_seq_lens: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if query_lens is None or request_total_seq_lens is None:
        return _default_query_lens_and_request_total_seq_lens(query)
    return query_lens, request_total_seq_lens


def _elementwise_sigmoid_ops(numel: int) -> int:
    return numel * 4


def _elementwise_softplus_ops(numel: int) -> int:
    return numel * 4


def _elementwise_silu_ops(numel: int) -> int:
    return numel * 6


def _rmsnorm_ops(num_rows: int, row_width: int) -> int:
    # Approximate RMSNorm by mean(square) + rsqrt + normalization.
    return num_rows * row_width * 5


def _l2norm_ops(num_rows: int, row_width: int) -> int:
    # Approximate L2 norm by square + reduction + rsqrt + scaling.
    return num_rows * row_width * 4


def _accumulate_compute_ops(
    properties: OpInvokeInfo.PerformanceProperties,
    dtype: torch.dtype,
    mma_ops: int = 0,
    gp_ops: int = 0,
) -> None:
    if mma_ops == 0 and gp_ops == 0:
        return
    delta = OpInvokeInfo.PerformanceProperties(
        compute_ops={
            dtype: OpInvokeInfo.ComputeOps(mma_ops=mma_ops, gp_ops=gp_ops),
        }
    )
    properties.combine(delta, compute_only=True)


def _bytes(num_elements: int, dtype: torch.dtype) -> int:
    return int(bytes_of_elements(num_elements, dtype))


def _linear_attention_state_bytes(
    batch_size: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    state_dtype: torch.dtype = torch.float32,
) -> int:
    return _bytes(batch_size * num_v_heads * head_k_dim * head_v_dim, state_dtype)


_LA_SCRATCH_ROUND_TRIP_TRAFFIC_FACTOR = 2
_LA_CHUNK_ACTIVATION_TOKEN_SCRATCH_ROUND_TRIPS = 2
_LA_CHUNK_FP32_VECTOR_SCRATCH_ROUND_TRIPS = 2
_LA_CHUNK_FP32_MATRIX_SCRATCH_BUFFERS = 4
_LA_CHUNK_FP32_SCALAR_VECTOR_WIDTH = 3
_LA_CHUNK_EXTRA_STATIC_KERNELS = 8


def _add_linear_attention_chunk_scratch_memory(
    properties: OpInvokeInfo.PerformanceProperties,
    batch_size: int,
    seq_len: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    chunk_size: int,
    activation_dtype: torch.dtype,
) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    padded_seq_len = ((seq_len + chunk_size - 1) // chunk_size) * chunk_size
    num_chunks = padded_seq_len // chunk_size
    batch_heads = batch_size * num_v_heads
    padded_positions = batch_heads * padded_seq_len

    # Fused LA keeps one semantic op, but the chunk rule still needs scratch for
    # k_beta/v_beta, fp32 normalized/cast vectors, decay vectors, and triangular
    # chunk matrices. Count this as HBM round-trip traffic, not as persistent IO.
    activation_token_scratch = _bytes(padded_positions * (head_k_dim + head_v_dim), activation_dtype)
    fp32_vector_scratch = _bytes(
        padded_positions * (2 * head_k_dim + head_v_dim + _LA_CHUNK_FP32_SCALAR_VECTOR_WIDTH),
        torch.float32,
    )
    fp32_matrix_scratch = _bytes(
        batch_heads * num_chunks * chunk_size * chunk_size * _LA_CHUNK_FP32_MATRIX_SCRATCH_BUFFERS,
        torch.float32,
    )
    properties.memory_readwrite_bytes += (
        _LA_CHUNK_ACTIVATION_TOKEN_SCRATCH_ROUND_TRIPS * activation_token_scratch
        + _LA_CHUNK_FP32_VECTOR_SCRATCH_ROUND_TRIPS * fp32_vector_scratch
        + _LA_SCRATCH_ROUND_TRIP_TRAFFIC_FACTOR * fp32_matrix_scratch
    )
    properties.extra_static_cost_count += _LA_CHUNK_EXTRA_STATIC_KERNELS


def _add_linear_attention_state_memory(
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
    state_bytes = _linear_attention_state_bytes(batch_size, num_v_heads, head_k_dim, head_v_dim)
    properties.memory_read_bytes += state_read_passes * state_bytes
    properties.memory_write_bytes += state_write_passes * state_bytes


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.linear_attn_apply_padding_mask.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    return op_invoke_info.get_memory_access_properties()


def _la_causal_conv_properties(op_invoke_info: OpInvokeInfo, include_state: bool) -> OpInvokeInfo.PerformanceProperties:
    mixed_qkv = op_invoke_info.args[0]
    conv_kernel_size = op_invoke_info.args[1]
    batch_size = mixed_qkv.size(0)
    conv_dim = mixed_qkv.size(1)
    seq_len = mixed_qkv.size(2)
    properties = op_invoke_info.get_memory_access_properties()

    conv_gp_ops = batch_size * seq_len * conv_dim * conv_kernel_size * 2 + _elementwise_silu_ops(
        batch_size * seq_len * conv_dim
    )
    _accumulate_compute_ops(properties, mixed_qkv.dtype, gp_ops=conv_gp_ops)
    properties.memory_read_bytes += _bytes(conv_dim * conv_kernel_size, mixed_qkv.dtype)
    if include_state:
        properties.memory_readwrite_bytes += _bytes(batch_size * conv_dim * conv_kernel_size, mixed_qkv.dtype)
    return properties


def _linear_attention_common_ops(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
) -> Tuple[int, int, int, int]:
    num_tokens = batch_size * seq_len
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = key_dim * 2 + value_dim

    # in_proj_qkv + in_proj_z + in_proj_b + in_proj_a + out_proj
    projection_mma_ops = num_tokens * (
        hidden_size * conv_dim * 2
        + hidden_size * value_dim * 2
        + hidden_size * num_v_heads * 2
        + hidden_size * num_v_heads * 2
        + value_dim * hidden_size * 2
    )

    conv_gp_ops = num_tokens * conv_dim * conv_kernel_size * 2 + _elementwise_silu_ops(num_tokens * conv_dim)
    beta_gp_ops = _elementwise_sigmoid_ops(num_tokens * num_v_heads)

    # g = -exp(A_log.float()) * softplus(a.float() + dt_bias)
    g_gp_ops = num_v_heads + num_tokens * num_v_heads * (1 + _elementwise_softplus_ops(1) + 1 + 1)

    gated_rmsnorm_gp_ops = (
        _rmsnorm_ops(num_tokens, value_dim)
        + num_tokens * value_dim
        + _elementwise_silu_ops(num_tokens * value_dim)
        + num_tokens * value_dim
    )

    return (
        projection_mma_ops,
        conv_gp_ops,
        beta_gp_ops,
        g_gp_ops + gated_rmsnorm_gp_ops,
    )


def _linear_attention_chunk_gated_delta_ops(
    batch_size: int,
    seq_len: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    chunk_size: int = 64,
) -> Tuple[int, int, int]:
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

    qk_l2norm_gp_ops = _l2norm_ops(valid_positions, head_k_dim) * 2
    prefix_correction_gp_ops = batch_heads * num_chunks * (chunk_size - 1) * chunk_size * (2 * chunk_size - 1) // 3

    # After the explicit float32 cast in torch_chunk_gated_delta_rule, the rest of
    # the recurrence, exponentials, cumsums, masking, and gated updates run in fp32.
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

    return (
        intra_chunk_mma_ops + inter_chunk_mma_ops,
        qk_l2norm_gp_ops,
        chunk_rule_fp32_gp_ops,
    )


def _linear_attention_recurrent_gated_delta_ops(
    batch_size: int,
    seq_len: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> Tuple[int, int, int]:
    num_tokens = batch_size * seq_len
    total_positions = num_tokens * num_v_heads

    recurrent_mma_ops = num_tokens * num_v_heads * head_k_dim * head_v_dim * 4
    qk_l2norm_gp_ops = _l2norm_ops(total_positions, head_k_dim) * 2
    recurrent_fp32_gp_ops = (
        total_positions * head_k_dim
        + total_positions * (head_v_dim * 2 + 2)
        + total_positions * head_k_dim * head_v_dim * 2
    )

    return recurrent_mma_ops, qk_l2norm_gp_ops, recurrent_fp32_gp_ops


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.linear_attention.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 8
    hidden_states = op_invoke_info.args[0]
    cache_position = op_invoke_info.args[2]
    num_k_heads = op_invoke_info.args[3]
    num_v_heads = op_invoke_info.args[4]
    head_k_dim = op_invoke_info.args[5]
    head_v_dim = op_invoke_info.args[6]
    conv_kernel_size = op_invoke_info.args[7]

    has_previous_state = False
    if cache_position is not None and cache_position.numel() > 0:
        # Check if it's a meta tensor (no actual data)
        is_meta = hasattr(cache_position, "is_meta") and cache_position.is_meta
        if not is_meta:
            try:
                has_previous_state = cache_position[0].item() > 0
            except RuntimeError:
                # If we can't get the value, default to prefill mode
                has_previous_state = False

    batch_size = hidden_states.size(0)
    seq_len = hidden_states.size(1)
    hidden_size = hidden_states.size(2)

    properties = op_invoke_info.get_memory_access_properties()
    (
        projection_mma_ops,
        conv_gp_ops,
        beta_gp_ops,
        fp32_common_gp_ops,
    ) = _linear_attention_common_ops(
        batch_size,
        seq_len,
        hidden_size,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel_size,
    )

    # Determine path:
    # 1. seq_len == 1 and has_previous_state=True → decode (recurrent)
    # 2. seq_len == 1 and has_previous_state=False → prefill (chunk)
    # 3. seq_len > 1 → prefill (chunk)
    if seq_len == 1 and has_previous_state:
        # Single token with previous context → decode
        (
            attn_mma_ops,
            hidden_gp_ops,
            fp32_gp_ops,
        ) = _linear_attention_recurrent_gated_delta_ops(batch_size, seq_len, num_v_heads, head_k_dim, head_v_dim)
    else:
        (
            attn_mma_ops,
            hidden_gp_ops,
            fp32_gp_ops,
        ) = _linear_attention_chunk_gated_delta_ops(batch_size, seq_len, num_v_heads, head_k_dim, head_v_dim)

    _accumulate_compute_ops(
        properties,
        hidden_states.dtype,
        mma_ops=projection_mma_ops,
        gp_ops=conv_gp_ops + beta_gp_ops + hidden_gp_ops,
    )
    _accumulate_compute_ops(
        properties,
        torch.float32,
        mma_ops=attn_mma_ops,
        gp_ops=fp32_common_gp_ops + fp32_gp_ops,
    )
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.linear_attn_causal_conv.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    return _la_causal_conv_properties(op_invoke_info, include_state=False)


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.linear_attn_causal_conv_update.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    return _la_causal_conv_properties(op_invoke_info, include_state=True)


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.linear_attn_fused_gdn_gating.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    query = op_invoke_info.args[0]
    b = op_invoke_info.args[2]
    a_log = op_invoke_info.args[4]
    dt_bias = op_invoke_info.args[5]
    num_v_heads = op_invoke_info.args[6]

    batch_size = query.size(0)
    seq_len = query.size(1)
    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={4, 5})
    properties.memory_read_bytes += _bytes(num_v_heads, a_log.dtype)
    properties.memory_read_bytes += _bytes(num_v_heads, dt_bias.dtype)

    num_gate_elements = batch_size * seq_len * num_v_heads
    beta_gp_ops = _elementwise_sigmoid_ops(num_gate_elements)
    g_gp_ops = num_v_heads + num_gate_elements * (1 + _elementwise_softplus_ops(1) + 1 + 1)
    _accumulate_compute_ops(properties, b.dtype, gp_ops=beta_gp_ops)
    _accumulate_compute_ops(properties, torch.float32, gp_ops=g_gp_ops)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.linear_attn_chunk_gated_delta_rule.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
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
    _add_linear_attention_state_memory(
        properties,
        batch_size,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        state_read_passes,
        state_write_passes,
    )
    _add_linear_attention_chunk_scratch_memory(
        properties,
        batch_size,
        seq_len,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        chunk_size,
        query.dtype,
    )
    attn_mma_ops, hidden_gp_ops, fp32_gp_ops = _linear_attention_chunk_gated_delta_ops(
        batch_size,
        seq_len,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        chunk_size,
    )
    _accumulate_compute_ops(properties, query.dtype, gp_ops=hidden_gp_ops)
    _accumulate_compute_ops(properties, torch.float32, mma_ops=attn_mma_ops, gp_ops=fp32_gp_ops)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.linear_attn_recurrent_gated_delta_rule.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
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
    _add_linear_attention_state_memory(
        properties,
        batch_size,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        state_read_passes,
        state_write_passes,
    )
    recurrent_mma_ops, hidden_gp_ops, fp32_gp_ops = _linear_attention_recurrent_gated_delta_ops(
        batch_size,
        seq_len,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )
    _accumulate_compute_ops(properties, query.dtype, gp_ops=hidden_gp_ops)
    _accumulate_compute_ops(properties, torch.float32, mma_ops=recurrent_mma_ops, gp_ops=fp32_gp_ops)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.linear_attn_gated_rmsnorm.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    core_attn_out = op_invoke_info.args[0]
    batch_size = core_attn_out.size(0)
    seq_len = core_attn_out.size(1)
    num_v_heads = core_attn_out.size(2)
    head_v_dim = core_attn_out.size(3)
    num_rows = batch_size * seq_len * num_v_heads
    num_elements = num_rows * head_v_dim

    properties = op_invoke_info.get_memory_access_properties()
    gated_rmsnorm_gp_ops = (
        _rmsnorm_ops(num_rows, head_v_dim) + num_elements + _elementwise_silu_ops(num_elements) + num_elements
    )
    _accumulate_compute_ops(properties, torch.float32, gp_ops=gated_rmsnorm_gp_ops)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.attention.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 8
    query = op_invoke_info.args[0]
    key = op_invoke_info.args[1]
    request_total_seq_lens = op_invoke_info.args[6]
    query_lens = op_invoke_info.args[7]
    query_lens, request_total_seq_lens = _normalize_query_lens_and_request_total_seq_lens(
        query, query_lens, request_total_seq_lens
    )
    return _attention_properties_helper(op_invoke_info, query, key, request_total_seq_lens, query_lens, query.dtype)


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.attention_quant.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 15
    query = op_invoke_info.args[0]
    key = op_invoke_info.args[1]
    request_total_seq_lens = op_invoke_info.args[6]
    query_lens = op_invoke_info.args[7]
    is_query_scaled = op_invoke_info.args[8] is not None and not torch.isclose(
        op_invoke_info.args[8], torch.tensor(1.0)
    )
    out_dtype = op_invoke_info.args[14]
    query_lens, request_total_seq_lens = _normalize_query_lens_and_request_total_seq_lens(
        query, query_lens, request_total_seq_lens
    )
    if out_dtype is None or out_dtype == query.dtype:
        # use half as default softmax dtype
        softmax_dtype = torch.half
    else:
        softmax_dtype = out_dtype
    properties = _attention_properties_helper(
        op_invoke_info, query, key, request_total_seq_lens, query_lens, softmax_dtype
    )

    # According to
    #   `out = dequant(quant(softmax(dequant(Q @ K^T)), attention_prob_scale/offset) @ V)`
    # Calculate additional quantization and dequantization ops

    # 0. Calculate dimensions for quantization ops
    hidden_size = query.size(-1)
    head_size = key.size(-1)
    num_q_heads = hidden_size // head_size
    num_tokens_per_seq = query_lens
    context_len_product_sum = torch.sum(
        num_tokens_per_seq.to(request_total_seq_lens.dtype) * request_total_seq_lens
    ).item()

    # FP8: only 1 op per element (scale multiplication, no offset applied).
    # Assume FP8 is not natively supported
    qdq_op_factor = 1 if is_fp8_dtype(key.dtype) else 2

    # 1. Dequantization of Q @ K^T (score matrix):
    #    scale multiplication + optional offset subtraction
    # Number of elements: context_len_product_sum * num_q_heads
    # Assuming 2 ops per element (scale + offset) for worst case
    dequant_qkt_ops = context_len_product_sum * num_q_heads * qdq_op_factor

    # 2. Quantization of softmax output (attention probabilities):
    #    scale multiplication + optional offset addition
    # Same number of elements as above
    quant_softmax_ops = context_len_product_sum * num_q_heads * qdq_op_factor

    # 3. Dequantization of final output:
    #    scale multiplication + optional offset subtraction
    # Number of elements: total_tokens * num_q_heads * head_size
    if out_dtype is None or out_dtype == query.dtype:
        dequant_output_ops = 0
    else:
        total_tokens = torch.sum(num_tokens_per_seq).item()
        dequant_output_ops = total_tokens * num_q_heads * head_size * qdq_op_factor

    if is_query_scaled:
        dequant_qkt_ops += context_len_product_sum * num_q_heads

    # Add quantization/dequantization ops to gp_ops
    total_quant_dequant_ops = dequant_qkt_ops + quant_softmax_ops + dequant_output_ops
    _accumulate_compute_ops(
        properties,
        softmax_dtype,
        gp_ops=total_quant_dequant_ops,
    )

    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.concat_and_cache_mla.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 4
    kv_c_normed = op_invoke_info.args[0]
    k_rot = op_invoke_info.args[1]
    kv_cache = op_invoke_info.args[2]

    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={2})
    properties.memory_write_bytes += bytes_of_tensor(kv_c_normed, dtype=kv_cache.dtype) + bytes_of_tensor(
        k_rot, dtype=kv_cache.dtype
    )
    return properties


_PREDICTIVE_DECODING_THRESHOLD = 6


def _mlapo_properties_helper(
    op_invoke_info: OpInvokeInfo,
    hidden_states,
    kv_a_proj_weight,
    num_heads,
    qk_head_dim,
    qk_rope_head_dim,
    kv_lora_rank,
    q_lora_rank,
) -> OpInvokeInfo.PerformanceProperties:
    num_tokens = hidden_states.size(0)
    hidden_size = hidden_states.size(1)

    total_mma_ops = 0
    total_gp_ops = 0

    # Fused MLA preprocessing op that models RMS norm, matmuls, and RoPE

    # Op1: q_a_proj
    # Shapes: (num_tokens, hidden_size) @ (hidden_size, q_lora_rank)
    op1_ops = num_tokens * hidden_size * q_lora_rank * 2

    # Op2: q_a_layernorm
    # Each RMS norm element (mean, variance, scale) is approximated as ~5 FLOPs.
    op2_ops = num_tokens * q_lora_rank * 5

    # Op3: q_b_proj
    # Shapes: (num_tokens, q_lora_rank) @ (q_lora_rank, num_heads * qk_head_dim)
    op3_ops = num_tokens * q_lora_rank * num_heads * qk_head_dim * 2

    # Op4: q_RoPE
    # Each RoPE element (multiply by cos, rotate + multiply by sin, add) is approximated as ~3 FLOPs.
    op4_ops = num_tokens * num_heads * qk_rope_head_dim * 3

    # Op5: kv_a_proj_with_mqa
    # Shapes: (num_tokens, hidden_size) @ (hidden_size, kv_lora_rank + qk_rope_head_dim)
    op5_ops = num_tokens * hidden_size * (kv_lora_rank + qk_rope_head_dim) * 2

    # Op6: kv_a_layernorm
    op6_ops = num_tokens * q_lora_rank * 5

    # Op7: k_RoPE
    op7_ops = num_tokens * qk_rope_head_dim * 3

    total_mma_ops += op1_ops + op3_ops + op5_ops
    total_gp_ops += op2_ops + op4_ops + op6_ops + op7_ops

    properties = op_invoke_info.get_memory_access_properties()
    compute_ops = properties.compute_ops.setdefault(kv_a_proj_weight.dtype, OpInvokeInfo.ComputeOps())
    compute_ops.mma_ops += total_mma_ops
    compute_ops = properties.compute_ops.setdefault(hidden_states.dtype, OpInvokeInfo.ComputeOps())
    compute_ops.gp_ops += total_gp_ops
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.mlapo.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    hidden_states = op_invoke_info.args[0]
    kv_a_proj_weight = op_invoke_info.args[6]
    num_heads = op_invoke_info.args[8]
    qk_head_dim = op_invoke_info.args[9]
    qk_rope_head_dim = op_invoke_info.args[11]
    kv_lora_rank = op_invoke_info.args[12]
    q_lora_rank = op_invoke_info.args[13]

    return _mlapo_properties_helper(
        op_invoke_info,
        hidden_states,
        kv_a_proj_weight,
        num_heads,
        qk_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        q_lora_rank,
    )


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.mlapo_quant.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    hidden_states = op_invoke_info.args[0]
    kv_a_proj_weight = op_invoke_info.args[6]
    num_heads = op_invoke_info.args[8]
    qk_head_dim = op_invoke_info.args[9]
    qk_rope_head_dim = op_invoke_info.args[11]
    kv_lora_rank = op_invoke_info.args[12]
    q_lora_rank = op_invoke_info.args[13]
    q_a_proj_offset = op_invoke_info.args[15]
    q_b_proj_offset = op_invoke_info.args[17]
    kv_a_proj_offset = op_invoke_info.args[19]
    num_tokens = hidden_states.size(0)
    hidden_size = hidden_states.size(1)
    properties = _mlapo_properties_helper(
        op_invoke_info,
        hidden_states,
        kv_a_proj_weight,
        num_heads,
        qk_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        q_lora_rank,
    )
    qdq_op_factor1 = 2 if q_a_proj_offset else 1
    qdq_op_factor2 = 2 if q_b_proj_offset else 1
    qdq_op_factor3 = 2 if kv_a_proj_offset else 1
    if is_fp8_dtype(kv_a_proj_weight.dtype):
        # QDQ for q_a_proj
        quant1_ops = num_tokens * hidden_size
        dequant1_ops = hidden_size * q_lora_rank
        # QDQ for q_b_proj
        quant2_ops = num_tokens * q_lora_rank
        dequant2_ops = q_lora_rank * num_heads * qk_head_dim
        # QDQ for kv_a_proj
        quant3_ops = num_tokens * hidden_size
        dequant3_ops = hidden_size * (kv_lora_rank + qk_rope_head_dim)
    else:
        # QDQ for q_a_proj
        quant1_ops = num_tokens * hidden_size * qdq_op_factor1
        dequant1_ops = hidden_size * q_lora_rank * qdq_op_factor1
        # QDQ for q_b_proj
        quant2_ops = num_tokens * q_lora_rank * qdq_op_factor2
        dequant2_ops = q_lora_rank * num_heads * qk_head_dim * qdq_op_factor2
        # QDQ for kv_a_proj
        quant3_ops = num_tokens * hidden_size * qdq_op_factor3
        dequant3_ops = hidden_size * (kv_lora_rank + qk_rope_head_dim) * qdq_op_factor3
    total_quant_dequant_ops = quant1_ops + dequant1_ops + quant2_ops + dequant2_ops + quant3_ops + dequant3_ops
    _accumulate_compute_ops(
        properties,
        hidden_states.dtype,
        gp_ops=total_quant_dequant_ops,
    )

    return properties


# Sparse MLA reads a small, index-selected KV working set from paged cache.  The
# roofline model keeps this hardware-agnostic by converting physical KV bytes into
# effective memory bytes with a data-movement efficiency, rather than modeling a
# concrete cache level or DMA pipeline.  Formula: effective_bytes = physical_bytes / eta,
# where eta is a sparse/paged data-movement efficiency in (0, 1].
_SPARSE_MLA_S1_BASE_SIZE = 64
# Reference sparse page count for decode efficiency interpolation.  More selected
# pages slightly improve effective bandwidth because the fixed random-access overhead
# is amortized over more useful KV bytes.
_SPARSE_MLA_REFERENCE_SPARSE_PAGE_COUNT = 16
# Decode sparse-KV reads are highly scattered: the calibrated efficiency is low and
# changes logarithmically with selected page count, with floor/ceil clamps to avoid
# overfitting very small or very large top-k shapes.  Formula:
# eta_decode = clamp(base + scale * log2(page_count / reference_page_count), floor, ceil).
_SPARSE_MLA_PAGED_KV_READ_EFFICIENCY_DECODE_BASE = 0.025
_SPARSE_MLA_PAGED_KV_READ_EFFICIENCY_DECODE_LOG2_PAGE_SCALE = 0.006
_SPARSE_MLA_PAGED_KV_READ_EFFICIENCY_DECODE_FLOOR = 0.015
_SPARSE_MLA_PAGED_KV_READ_EFFICIENCY_DECODE_CEIL = 0.080
# Prefill sparse-KV access is much more sequential across query tokens, so use a
# fixed calibrated efficiency for the current fixed-block-size paged-cache layout.
# Formula: eta_prefill = prefill_eta.
_SPARSE_MLA_PAGED_KV_READ_EFFICIENCY_PREFILL = 0.92
_SPARSE_MLA_SPARSE_INDEX_DTYPE_BYTES = 4


def _sparse_mla_decode_mask(query_lens: torch.Tensor) -> torch.Tensor:
    """Return which requests use decode-like sparse MLA accounting. The predictive decode threshold is set one token above the largest observed decode/MTP target bundle, so query_len < threshold covers both single-token decode and small MTP decode without checking active context length or block span."""
    return query_lens < _PREDICTIVE_DECODING_THRESHOLD


def _sparse_mla_query_tile_count(query_lens: torch.Tensor) -> torch.Tensor:
    """Compute ceil(query_len / S1_BASE_SIZE) for decode KV staging. SparseFlashAttention stages one selected sparse-KV working set per S1 tile, so using tile count rather than token count models reuse across adjacent decode/MTP query tokens."""
    return torch.div(query_lens + _SPARSE_MLA_S1_BASE_SIZE - 1, _SPARSE_MLA_S1_BASE_SIZE, rounding_mode="floor")


def _estimate_sparse_mla_decode_page_efficiency(sparse_page_count: int) -> float:
    """Estimate decode sparse-KV efficiency as clamp(base + scale * log2(page_count / reference), floor, ceil).
    The logarithmic term models random-access overhead being amortized as more useful sparse pages are selected,
    while clamps keep extrapolated very-small and very-large top-k shapes bounded.
    """
    page_count = max(sparse_page_count, 1)
    log2_page_scale = math.log2(page_count / _SPARSE_MLA_REFERENCE_SPARSE_PAGE_COUNT)
    eta = (
        _SPARSE_MLA_PAGED_KV_READ_EFFICIENCY_DECODE_BASE
        + _SPARSE_MLA_PAGED_KV_READ_EFFICIENCY_DECODE_LOG2_PAGE_SCALE * log2_page_scale
    )
    return max(
        _SPARSE_MLA_PAGED_KV_READ_EFFICIENCY_DECODE_FLOOR,
        min(eta, _SPARSE_MLA_PAGED_KV_READ_EFFICIENCY_DECODE_CEIL),
    )


def _estimate_sparse_mla_prefill_page_efficiency() -> float:
    """Return the calibrated sparse-MLA prefill KV-read efficiency. Prefill has many adjacent query tokens and the current paged-cache block size is fixed, so block-size scaling would reduce to a constant and is intentionally omitted."""
    return _SPARSE_MLA_PAGED_KV_READ_EFFICIENCY_PREFILL


def _estimate_sparse_mla_paged_kv_read_breakdown(
    kv_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    request_total_seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    sparse_topk: Optional[int],
    topk_indices: Optional[torch.Tensor] = None,
):
    """Break down sparse MLA cache traffic for roofline memory bytes. The helper first caps each request's attention length with sparse_topk, then converts selected KV entries into physical bytes, applies decode/prefill efficiencies separately, and returns effective bytes = physical_bytes / eta so random sparse gathers reduce usable bandwidth without hard-coding a cache level."""
    block_size = kv_cache.size(1)
    cache_entry_size = bytes_of_elements(kv_cache.size(-1), kv_cache.dtype)
    # Dense MLA has no sparse index indirection, so physical and effective KV read
    # bytes are identical.  Return the same compact schema as sparse mode: effective
    # bytes for the calibrated path, physical bytes for the quant/raw path, and the
    # optional index/table streams consumed by the caller.
    if sparse_topk is None:
        selected_cache_entries = int(torch.sum(request_total_seq_lens).item())
        kv_read_bytes = selected_cache_entries * cache_entry_size
        block_table_read_bytes = 0
        if block_table is not None:
            dense_block_table_entries = (request_total_seq_lens + block_size - 1) // block_size
            block_table_read_bytes = bytes_of_elements(
                int(torch.sum(dense_block_table_entries).item()), block_table.dtype
            )
        return {
            "kv_effective_read_bytes": kv_read_bytes,
            "block_table_read_bytes": block_table_read_bytes,
            "sparse_index_read_bytes": 0,
            "kv_physical_read_bytes": kv_read_bytes,
            "physical_block_table_read_bytes": block_table_read_bytes,
            "physical_sparse_index_read_bytes": 0,
        }

    # Sparse top-k caps the logical attention length.  This is the useful KV work
    # before accounting for paged-cache indirection and sparse gather inefficiency.
    # Formula: attn_len_i = min(request_total_seq_len_i, sparse_topk).
    attn_lens = torch.minimum(
        request_total_seq_lens,
        torch.tensor(sparse_topk, device=request_total_seq_lens.device),
    )
    sparse_page_counts = torch.div(attn_lens + block_size - 1, block_size, rounding_mode="floor")

    # raw_* is the unoptimized selected-KV traffic: every query token reads its own
    # sparse KV set.  It is kept for the quant MLA path, whose calibration still uses
    # the older selected-KV accounting.  Formula: raw_entries = sum_i(query_len_i * attn_len_i).
    raw_sparse_cache_entries = query_lens.to(attn_lens.dtype) * attn_lens
    raw_selected_cache_entries = int(torch.sum(raw_sparse_cache_entries).item())
    raw_selected_kv_physical_read_bytes = raw_selected_cache_entries * cache_entry_size

    is_decode = _sparse_mla_decode_mask(query_lens)
    # Decode kernels stage sparse KV per S1 tile, so multiple adjacent query tokens can
    # reuse the same selected pages.  Prefill keeps per-token accounting because its
    # query dimension is already large enough to expose mostly sequential access.
    # Formula: selected_entries = sum_i(query_units_i * attn_len_i), where decode
    # query_units_i = ceil(query_len_i / S1_BASE_SIZE), prefill query_units_i = query_len_i.
    cache_read_query_units = query_lens.clone()
    if torch.any(is_decode).item():
        cache_read_query_units[is_decode] = _sparse_mla_query_tile_count(query_lens[is_decode])
    sparse_cache_entries = cache_read_query_units.to(attn_lens.dtype) * attn_lens
    selected_cache_entries = int(torch.sum(sparse_cache_entries).item())

    is_prefill = ~is_decode
    decode_entries = int(torch.sum(sparse_cache_entries[is_decode]).item()) if torch.any(is_decode).item() else 0
    prefill_entries = int(torch.sum(sparse_cache_entries[is_prefill]).item()) if torch.any(is_prefill).item() else 0
    decode_physical_read_bytes = decode_entries * cache_entry_size
    prefill_physical_read_bytes = prefill_entries * cache_entry_size

    # Apply efficiency separately for decode and prefill so mixed batches do not get
    # a single blended coefficient before the physical byte split is known.
    # Formula: physical_bytes = selected_entries * cache_entry_size.
    decode_efficiency = 1.0
    if decode_entries > 0:
        decode_sparse_page_count = int(torch.max(sparse_page_counts[is_decode]).item())
        decode_efficiency = _estimate_sparse_mla_decode_page_efficiency(decode_sparse_page_count)
    prefill_efficiency = 1.0
    if prefill_entries > 0:
        prefill_efficiency = _estimate_sparse_mla_prefill_page_efficiency()

    # Effective bytes are the physical bytes divided by data-movement efficiency; the
    # analytic roofline then treats them as ordinary memory traffic.  Formula:
    # effective_bytes = decode_physical_bytes / eta_decode + prefill_physical_bytes / eta_prefill.
    selected_kv_effective_read_bytes = (
        decode_physical_read_bytes / decode_efficiency + prefill_physical_read_bytes / prefill_efficiency
    )
    # If topk_indices is not passed as an input tensor, account for the implicit sparse
    # index stream consumed by the kernel.  Explicit topk_indices are already covered by
    # get_memory_access_properties().
    sparse_index_read_bytes = 0
    physical_sparse_index_read_bytes = 0
    if topk_indices is None:
        sparse_index_read_bytes = selected_cache_entries * _SPARSE_MLA_SPARSE_INDEX_DTYPE_BYTES
        physical_sparse_index_read_bytes = raw_selected_cache_entries * _SPARSE_MLA_SPARSE_INDEX_DTYPE_BYTES

    block_table_read_bytes = 0
    raw_block_table_read_bytes = 0
    if block_table is not None:
        block_table_entries = int(
            torch.sum(cache_read_query_units.to(sparse_page_counts.dtype) * sparse_page_counts).item()
        )
        raw_block_table_entries = int(torch.sum(query_lens.to(sparse_page_counts.dtype) * sparse_page_counts).item())
        block_table_read_bytes = bytes_of_elements(block_table_entries, block_table.dtype)
        raw_block_table_read_bytes = bytes_of_elements(raw_block_table_entries, block_table.dtype)

    return {
        "kv_effective_read_bytes": selected_kv_effective_read_bytes,
        "block_table_read_bytes": block_table_read_bytes,
        "sparse_index_read_bytes": sparse_index_read_bytes,
        "kv_physical_read_bytes": raw_selected_kv_physical_read_bytes,
        "physical_block_table_read_bytes": raw_block_table_read_bytes,
        "physical_sparse_index_read_bytes": physical_sparse_index_read_bytes,
    }


def _multihead_latent_attention_properties_helper(
    op_invoke_info: OpInvokeInfo,
    softmax_dtype: torch.dtype,
    enable_sparse_mla_paged_kv_efficiency: bool = True,
) -> OpInvokeInfo.PerformanceProperties:
    # 1. Argument and Dimension Extraction
    assert len(op_invoke_info.args) >= 10
    (
        q,
        kv_cache,
        _block_table,
        query_start_loc,
        request_total_seq_lens,
        query_lens,
        W_UK_T,
        W_UV,
        kv_b_proj,
        v_head_dim,
        *rest,
    ) = op_invoke_info.args

    topk_limit = rest[0] if len(rest) > 0 else None
    topk_indices = rest[1] if len(rest) > 1 else None

    # Extract dimensions from input tensors
    num_heads = q.size(1)
    q_head_dim = q.size(2)
    kv_lora_rank = W_UK_T.size(-1)
    qk_rope_head_dim = kv_cache.size(-1) - kv_lora_rank
    qk_nope_head_dim = q_head_dim - qk_rope_head_dim
    sparse_topk = topk_indices.shape[-1] if topk_indices is not None else topk_limit

    # 2. Separate Prefill and Decode Sequences.
    # q_len == 5 with a long active context is the MTP target decode shape in SFA traces.
    num_tokens_per_seq = query_lens
    is_decode = _sparse_mla_decode_mask(num_tokens_per_seq)
    is_prefill = ~is_decode

    total_fma_ops = 0
    total_gp_ops = 0
    exclude_input_ids = {1, 2, 6, 7, 8}  # kv_cache, block_table, W_UK_T, W_UV, kv_b_proj

    # 3. Calculate FLOPs for the Prefill Phase
    num_prefill_tokens = torch.sum(num_tokens_per_seq[is_prefill]).item()
    if num_prefill_tokens > 0:
        assert kv_b_proj is not None
        exclude_input_ids = exclude_input_ids - {8}  # kv_b_proj
        prefill_request_total_seq_lens = request_total_seq_lens[is_prefill]
        prefill_num_tokens_per_seq = num_tokens_per_seq[is_prefill]

        # Op 1: Project compressed KV: `kv_c_normed @ kv_b_proj`
        # Shapes: (num_prefill_tokens, kv_lora_rank) @ (kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))
        kv_proj_out_dim = num_heads * (qk_nope_head_dim + v_head_dim)
        prefill_op1_ops = num_prefill_tokens * kv_proj_out_dim * kv_lora_rank * 2

        # For attention ops, we need the sum of (query_len * key_len) over the batch
        prefill_attn_len = (
            torch.clamp(prefill_request_total_seq_lens, max=sparse_topk)
            if sparse_topk is not None
            else prefill_request_total_seq_lens
        )
        prefill_context_sum = torch.sum(prefill_num_tokens_per_seq.to(prefill_attn_len.dtype) * prefill_attn_len).item()

        # Op 2: Score calculation: `q @ K`
        prefill_op2_ops = prefill_context_sum * num_heads * q_head_dim * 2

        # Op 3: Softmax
        prefill_op3_ops = prefill_context_sum * num_heads * 4

        # Op 4: Score aggregation: `Scores @ V`
        prefill_op4_ops = prefill_context_sum * num_heads * v_head_dim * 2

        total_fma_ops += prefill_op1_ops + prefill_op2_ops + prefill_op4_ops
        total_gp_ops += prefill_op3_ops

    # 4. Calculate FLOPs for the Decode Phase
    num_decode_tokens = torch.sum(num_tokens_per_seq[is_decode]).item()
    if num_decode_tokens > 0:
        assert W_UK_T is not None and W_UV is not None
        exclude_input_ids = exclude_input_ids - {6, 7}  # W_UK_T, W_UV
        decode_request_total_seq_lens = request_total_seq_lens[is_decode]
        decode_num_tokens_per_seq = num_tokens_per_seq[is_decode]

        # The total number of key/value tokens to attend to across all decode sequences
        decode_attn_len = (
            torch.clamp(decode_request_total_seq_lens, max=sparse_topk)
            if sparse_topk is not None
            else decode_request_total_seq_lens
        )
        decode_context_sum = torch.sum(decode_num_tokens_per_seq.to(decode_attn_len.dtype) * decode_attn_len).item()

        # The decode formula is: softmax(q_nope @ W_UK_T @ k_cache) @ v_cache @ W_UV
        # Op 1: `q_nope @ W_UK_T`
        # Shapes: (num_decode_tokens, num_heads, qk_nope_head_dim) @ (num_heads, qk_nope_head_dim, kv_lora_rank)
        decode_op1_ops = num_decode_tokens * num_heads * qk_nope_head_dim * kv_lora_rank * 2

        # Op 2: `(result_op1, q_rope) @ kv_cache`
        decode_op2_ops = decode_context_sum * num_heads * (kv_lora_rank + qk_rope_head_dim) * 2

        # Op 3: Softmax
        decode_op3_ops = decode_context_sum * num_heads * 4

        # Op 4: `Scores @ v_cache`
        decode_op4_ops = decode_context_sum * num_heads * kv_lora_rank * 2

        # Op 5: `(result_op4) @ W_UV`
        # Shapes: (num_decode_tokens, num_heads, kv_lora_rank) @ (num_heads, kv_lora_rank, v_head_dim)
        decode_op5_ops = num_decode_tokens * num_heads * kv_lora_rank * v_head_dim * 2

        total_fma_ops += decode_op1_ops + decode_op2_ops + decode_op4_ops + decode_op5_ops
        total_gp_ops += decode_op3_ops

    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids=exclude_input_ids)  # exclude kv_cache

    # Estimate paged sparse-KV reads with a local page efficiency. Decode shapes reuse
    # the staged sparse KV set across an S1 tile, while prefill reads scale with query tokens.
    sparse_mla_paged_kv = _estimate_sparse_mla_paged_kv_read_breakdown(
        kv_cache,
        _block_table,
        request_total_seq_lens,
        query_lens,
        sparse_topk,
        topk_indices,
    )
    if enable_sparse_mla_paged_kv_efficiency:
        properties.memory_read_bytes += (
            sparse_mla_paged_kv["kv_effective_read_bytes"]
            + sparse_mla_paged_kv["block_table_read_bytes"]
            + sparse_mla_paged_kv["sparse_index_read_bytes"]
        )
    else:
        properties.memory_read_bytes += (
            sparse_mla_paged_kv["kv_physical_read_bytes"]
            + sparse_mla_paged_kv["physical_block_table_read_bytes"]
            + sparse_mla_paged_kv["physical_sparse_index_read_bytes"]
        )

    compute_ops = properties.compute_ops.setdefault(q.dtype, OpInvokeInfo.ComputeOps())
    compute_ops.mma_ops = total_fma_ops
    compute_ops = properties.compute_ops.setdefault(softmax_dtype, OpInvokeInfo.ComputeOps())
    compute_ops.gp_ops = total_gp_ops

    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.multihead_latent_attention.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    q = op_invoke_info.args[0]
    return _multihead_latent_attention_properties_helper(op_invoke_info, q.dtype)


def _calculate_mla_quant_ops(
    op_invoke_info: OpInvokeInfo,
    num_heads: int,
    q_head_dim: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    query_start_loc: torch.Tensor,
    request_total_seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    out_dtype: torch.dtype,
    q_dtype: torch.dtype,
) -> int:
    """
    Calculate quantization/dequantization ops for MLA quantization.
    Check `torch.ops.tensor_cast.multihead_latent_attention_quant` docstring for details.
    """
    # Separate prefill and decode sequences.
    num_tokens_per_seq = query_lens
    is_decode = _sparse_mla_decode_mask(num_tokens_per_seq)
    is_prefill = ~is_decode

    total_quant_dequant_ops = 0

    # Calculate quant/dequant ops for prefill phase
    num_prefill_tokens = torch.sum(num_tokens_per_seq[is_prefill]).item()
    if num_prefill_tokens > 0:
        prefill_request_total_seq_lens = request_total_seq_lens[is_prefill]
        prefill_num_tokens_per_seq = num_tokens_per_seq[is_prefill]
        prefill_context_sum = torch.sum(
            prefill_num_tokens_per_seq.to(prefill_request_total_seq_lens.dtype) * prefill_request_total_seq_lens
        ).item()

        # 1. Quantization of kv_c_normed @ kv_b_proj output
        # Number of elements: num_prefill_tokens * num_heads * (qk_nope_head_dim + v_head_dim)
        # Each quantization: scale multiplication + optional offset addition (2 ops worst case)
        kv_proj_out_dim = num_heads * (qk_nope_head_dim + v_head_dim)
        quant_kv_proj_ops = num_prefill_tokens * kv_proj_out_dim * 2

        # 2. Quantization of attention probabilities (softmax output)
        # Number of elements: prefill_context_sum * num_heads
        quant_attention_prob_ops = prefill_context_sum * num_heads * 2

        total_quant_dequant_ops += quant_kv_proj_ops + quant_attention_prob_ops

    # Calculate quant/dequant ops for decode phase
    num_decode_tokens = torch.sum(num_tokens_per_seq[is_decode]).item()
    if num_decode_tokens > 0:
        decode_request_total_seq_lens = request_total_seq_lens[is_decode]
        decode_num_tokens_per_seq = num_tokens_per_seq[is_decode]
        decode_context_sum = torch.sum(
            decode_num_tokens_per_seq.to(decode_request_total_seq_lens.dtype) * decode_request_total_seq_lens
        ).item()

        # 1. Quantization of q @ W_UK_T output
        # Number of elements: num_decode_tokens * num_heads * kv_lora_rank
        quant_qk_ops = num_decode_tokens * num_heads * kv_lora_rank * 2

        # 2. Quantization of attention probabilities (softmax output)
        # Number of elements: decode_context_sum * num_heads
        quant_attention_prob_ops = decode_context_sum * num_heads * 2

        # 3. Quantization of (Scores @ v_cache) output before @ W_UV
        # Number of elements: num_decode_tokens * num_heads * kv_lora_rank
        quant_v_ops = num_decode_tokens * num_heads * kv_lora_rank * 2

        total_quant_dequant_ops += quant_qk_ops + quant_attention_prob_ops + quant_v_ops

    # Optional final output quantization (both prefill and decode)
    # This is only applied if out_dtype is same as q_dtype
    if out_dtype is None or out_dtype == q_dtype:
        total_tokens = torch.sum(num_tokens_per_seq).item()
        # Number of elements: total_tokens * num_heads * v_head_dim
        quant_output_ops = total_tokens * num_heads * v_head_dim * 2
        total_quant_dequant_ops += quant_output_ops

    return total_quant_dequant_ops


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.multihead_latent_attention_quant.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    q = op_invoke_info.args[0]
    kv_cache = op_invoke_info.args[1]
    query_start_loc = op_invoke_info.args[3]
    request_total_seq_lens = op_invoke_info.args[4]
    query_lens = op_invoke_info.args[5]
    W_UK_T = op_invoke_info.args[6]
    v_head_dim = op_invoke_info.args[9]
    out_dtype = op_invoke_info.kwargs.get("out_dtype")
    if out_dtype is None and len(op_invoke_info.args) > 28:
        out_dtype = op_invoke_info.args[28]

    if out_dtype is None or out_dtype == q.dtype:
        # use half as default softmax dtype
        softmax_dtype = torch.half
    else:
        softmax_dtype = out_dtype

    # Get base properties from helper. Quant MLA keeps selected KV traffic as
    # physical bytes without sparse-page efficiency expansion until quant SFA
    # profiles are available for calibrating an effective-byte model.
    properties = _multihead_latent_attention_properties_helper(
        op_invoke_info,
        softmax_dtype,
        enable_sparse_mla_paged_kv_efficiency=False,
    )

    # Extract dimensions (reuse logic instead of duplicating)
    num_heads = q.size(1)
    q_head_dim = q.size(2)
    kv_lora_rank = W_UK_T.size(-1)
    qk_rope_head_dim = kv_cache.size(-1) - kv_lora_rank
    qk_nope_head_dim = q_head_dim - qk_rope_head_dim

    # Calculate additional quant/dequant ops
    total_quant_dequant_ops = _calculate_mla_quant_ops(
        op_invoke_info,
        num_heads,
        q_head_dim,
        kv_lora_rank,
        qk_nope_head_dim,
        v_head_dim,
        query_start_loc,
        request_total_seq_lens,
        query_lens,
        out_dtype,
        q.dtype,
    )

    # Add all quantization/dequantization ops to gp_ops
    _accumulate_compute_ops(
        properties,
        softmax_dtype,
        gp_ops=total_quant_dequant_ops,
    )

    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.grouped_matmul.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 3
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    bias = op_invoke_info.args[2]
    assert len(x) == len(w) == len(bias)
    properties = op_invoke_info.get_memory_access_properties()
    for xi, wi, biasi in zip(x, w, bias):
        properties_i = _mm_properties_helper(op_invoke_info, xi, wi, biasi)
        properties.combine(properties_i, compute_only=True)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.grouped_matmul_quant.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 8
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    w_offset = op_invoke_info.args[3]
    bias = op_invoke_info.args[6]
    assert len(x) == len(w) == len(w_offset) == len(bias)

    properties = op_invoke_info.get_memory_access_properties()
    for xi, wi, w_offseti, biasi in zip(x, w, w_offset, bias):
        properties_i = _static_quant_linear_properties_helper(op_invoke_info, xi, wi, w_offseti, biasi, is_int4=False)
        properties.combine(properties_i, compute_only=True)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.grouped_matmul_quant_int4.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 8
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    w_offset = op_invoke_info.args[3]
    bias = op_invoke_info.args[6]
    assert len(x) == len(w) == len(w_offset) == len(bias)

    properties = op_invoke_info.get_memory_access_properties()
    for xi, wi, w_offseti, biasi in zip(x, w, w_offset, bias):
        properties_i = _static_quant_linear_properties_helper(op_invoke_info, xi, wi, w_offseti, biasi, is_int4=True)
        properties.combine(properties_i, compute_only=True)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.grouped_matmul_fp8.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 6
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    bias = op_invoke_info.args[4]
    assert len(x) == len(w) == len(bias)
    properties = op_invoke_info.get_memory_access_properties()
    for xi, wi, biasi in zip(x, w, bias):
        properties_i = _static_quant_linear_properties_helper(op_invoke_info, xi, wi, None, biasi, is_int4=False)
        properties.combine(properties_i, compute_only=True)
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.grouped_matmul_mxfp4.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 6
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    bias = op_invoke_info.args[4]
    assert len(x) == len(w) == len(bias)
    properties = op_invoke_info.get_memory_access_properties()
    for xi, wi, biasi in zip(x, w, bias):
        properties_i = _static_quant_linear_properties_helper(op_invoke_info, xi, wi, None, biasi, is_int4=True)
        properties.combine(properties_i, compute_only=True)
    return properties


def _swiglu_fusion_properties_helper(
    op_invoke_info: OpInvokeInfo,
    x: List[torch.Tensor],
    w: List[torch.Tensor],
    bias: List[Optional[torch.Tensor]],
    w_offset: Optional[List[Optional[torch.Tensor]]],
    mm_helper: Callable,
    is_int4_weight: bool,
) -> OpInvokeInfo.PerformanceProperties:
    """
    Common performance modeling logic for all grouped_matmul_*_swiglu variants.

    Args:
        w_offset: If provided, uses quantized helper signature (info, x, w, offset, bias).
                  If None, uses standard helper signature (info, x, w, bias).
    """
    if not x:
        dtype = torch.float32
        properties = op_invoke_info.get_memory_access_properties()
        properties.compute_ops[dtype] = OpInvokeInfo.ComputeOps()
        return properties

    dtype = x[0].dtype if x else torch.float32
    total_swiglu_ops = 0
    properties = op_invoke_info.get_memory_access_properties()

    count = len(x)

    for i in range(count):
        xi = x[i]
        wi = w[i]
        biasi = bias[i] if (bias and i < len(bias)) else None
        w_offseti = w_offset[i] if (w_offset and i < len(w_offset)) else None

        # 1. Calculate MatMul Costs
        if mm_helper.__name__ == "_static_quant_linear_properties_helper":
            props_i = mm_helper(op_invoke_info, xi, wi, w_offseti, biasi, is_int4_weight)
        else:
            props_i = mm_helper(op_invoke_info, xi, wi, biasi)

        properties.combine(props_i, compute_only=True)

        # 2. Calculate SwiGLU Activation Costs (Internal Logic)
        M = xi.shape[0]
        k = xi.size(1)

        if k > 0 and wi.numel() > 0:
            n_total = 0
            if is_int4_weight:
                # Quantized (Int4/MXFP4): Infer logical N from packed storage
                pack_factor = (wi.element_size() * 8) // 4
                logical_total = wi.numel() * pack_factor
                if logical_total % k == 0:
                    n_total = logical_total // k
            else:
                # Non-quantized: Use physical shape directly
                if wi.dim() == 2:
                    n_total = wi.shape[1]
                else:
                    n_total = wi.shape[-1]

                # Safety fallback for shape mismatches
                if wi.dim() == 2 and wi.shape[0] != k and wi.numel() % k == 0:
                    n_total = wi.numel() // k

            if n_total > 0:
                n_gate = n_total // 2
                # SiLU (~6 FLOPs) + Gate Mul (1 FLOP) = 7 FLOPs
                total_swiglu_ops += M * n_gate * 7

    # 3. Accumulate SwiGLU ops into gp_ops
    _accumulate_compute_ops(properties, dtype, gp_ops=total_swiglu_ops)

    return properties


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.grouped_matmul_swiglu.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    # Args: (x, w, bias)
    return _swiglu_fusion_properties_helper(
        op_invoke_info,
        x=op_invoke_info.args[0],
        w=op_invoke_info.args[1],
        bias=op_invoke_info.args[2],
        w_offset=None,
        mm_helper=_mm_properties_helper,
        is_int4_weight=False,
    )


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.grouped_matmul_quant_swiglu.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    # Args: (x, w, w_scale, w_offset, x_scale, x_offset, bias, ...) -> offset=3, bias=6
    return _swiglu_fusion_properties_helper(
        op_invoke_info,
        x=op_invoke_info.args[0],
        w=op_invoke_info.args[1],
        bias=op_invoke_info.args[6],
        w_offset=op_invoke_info.args[3],
        mm_helper=_static_quant_linear_properties_helper,
        is_int4_weight=False,
    )


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.grouped_matmul_quant_int4_swiglu.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    # Args: offset=3, bias=6
    return _swiglu_fusion_properties_helper(
        op_invoke_info,
        x=op_invoke_info.args[0],
        w=op_invoke_info.args[1],
        bias=op_invoke_info.args[6],
        w_offset=op_invoke_info.args[3],
        mm_helper=_static_quant_linear_properties_helper,
        is_int4_weight=True,
    )


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.grouped_matmul_fp8_swiglu.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    # Args: (x, w, w_scale, x_scale, bias, ...) -> bias=4, no offset
    return _swiglu_fusion_properties_helper(
        op_invoke_info,
        x=op_invoke_info.args[0],
        w=op_invoke_info.args[1],
        bias=op_invoke_info.args[4],
        w_offset=None,
        mm_helper=_static_quant_linear_properties_helper,
        is_int4_weight=False,
    )


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.grouped_matmul_mxfp4_swiglu.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    # Args: bias=4, no offset
    return _swiglu_fusion_properties_helper(
        op_invoke_info,
        x=op_invoke_info.args[0],
        w=op_invoke_info.args[1],
        bias=op_invoke_info.args[4],
        w_offset=None,
        mm_helper=_static_quant_linear_properties_helper,
        is_int4_weight=True,
    )


@OpInvokeInfo.register_op_properties(torch.ops.aten.addmm.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 3 or len(op_invoke_info.args) == 5
    (input, mat1, mat2) = op_invoke_info.args[:3]

    # mat1:[M,K], mat2:[K,N]
    M, K = mat1.shape
    N = mat2.shape[-1]

    # mat_output = mat1 @ mat2 ; mat_output: [M,N]
    bmm1 = 2 * M * N * K

    if bmm1 == 0:
        return OpInvokeInfo.PerformanceProperties()

    properties = op_invoke_info.get_memory_access_properties()
    compute_ops = properties.compute_ops.setdefault(input.dtype, OpInvokeInfo.ComputeOps())
    compute_ops.mma_ops = bmm1
    return properties


@OpInvokeInfo.register_op_properties(torch.ops.aten.convolution.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    # op_invoke_info.args length: torch.nn.functional.conv2d is 7, nn.Conv2d is 9
    assert len(op_invoke_info.args) == 7 or len(op_invoke_info.args) == 9
    # Conv2D input:(B, C_in, H, W), weight:(C_out, C_in/groups, K_h, K_w)
    # Conv3D input:(B, C_in, D, H, W), weight:(C_out, C_in/groups, K_d, K_h, K_w)
    (
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
    ) = op_invoke_info.args[:6]
    if len(op_invoke_info.args) == 9:
        groups = op_invoke_info.args[8]
    else:
        groups = op_invoke_info.args[6]

    input_shape = input.shape
    weight_shape = weight.shape
    B = input_shape[0]
    C_in = input_shape[1]
    C_out = weight_shape[0]
    if input.dim() == 3:
        # Conv1D
        _, _, L_in = input_shape
        _, _, K_l = weight_shape
        (s_l,) = stride
        (p_l,) = padding
        (d_l,) = dilation

        L_out = math.floor((L_in + 2 * p_l - d_l * (K_l - 1) - 1) / s_l + 1)

        flops_per_output = 2 * (C_in / groups) * K_l
        total_flops = B * C_out * L_out * flops_per_output
        if bias is not None:
            total_flops += B * C_out * L_out

    elif input.dim() == 4:
        # Conv2D
        _, _, H_in, W_in = input_shape
        _, _, K_h, K_w = weight_shape
        s_h, s_w = stride
        p_h, p_w = padding
        d_h, d_w = dilation

        H_out = math.floor((H_in + 2 * p_h - d_h * (K_h - 1) - 1) / s_h + 1)
        W_out = math.floor((W_in + 2 * p_w - d_w * (K_w - 1) - 1) / s_w + 1)

        flops_per_output = 2 * (C_in / groups) * K_h * K_w
        total_flops = B * C_out * H_out * W_out * flops_per_output

        if bias is not None:
            total_flops += B * C_out * H_out * W_out

    elif input.dim() == 5:
        # Conv3D
        _, _, D_in, H_in, W_in = input_shape
        _, _, K_d, K_h, K_w = weight_shape
        s_d, s_h, s_w = stride
        p_d, p_h, p_w = padding
        d_d, d_h, d_w = dilation

        D_out = math.floor((D_in + 2 * p_d - d_d * (K_d - 1) - 1) / s_d + 1)
        H_out = math.floor((H_in + 2 * p_h - d_h * (K_h - 1) - 1) / s_h + 1)
        W_out = math.floor((W_in + 2 * p_w - d_w * (K_w - 1) - 1) / s_w + 1)

        flops_per_output = 2 * (C_in / groups) * K_d * K_h * K_w
        total_flops = B * C_out * D_out * H_out * W_out * flops_per_output

        if bias is not None:
            total_flops += B * C_out * D_out * H_out * W_out

    else:
        raise ValueError(f"Unsupported convolution dimension: {input.dim()}")

    if total_flops == 0:
        return OpInvokeInfo.PerformanceProperties()

    properties = op_invoke_info.get_memory_access_properties()
    compute_ops = properties.compute_ops.setdefault(input.dtype, OpInvokeInfo.ComputeOps())
    compute_ops.mma_ops = total_flops
    return properties


def _estimate_static_cost(
    perf_properties: OpInvokeInfo.PerformanceProperties,
    device_profile: DeviceProfile,
) -> float:
    for dtype, compute_ops in perf_properties.compute_ops.items():
        if _get_device_ops_for_dtype(device_profile.mma_ops, dtype) is None:
            continue
        if compute_ops.mma_ops > 0:
            return device_profile.static_cost.mma_op_cost_s
    return device_profile.static_cost.gp_op_cost_s


def _estimate_default_without_static_cost(
    op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile
) -> PerformanceModel.Result:
    if is_view_op(op_invoke_info.func) or is_noop_self_copy_op(op_invoke_info.func, op_invoke_info.args):
        return PerformanceModel.Result(0.0)
    perf_properties = op_invoke_info.get_perf_properties()
    # By default, we do not consider instruction-level parallelism when counting computation time
    mma_ops_time_s = 0
    gp_ops_time_s = 0
    for dtype, compute_ops in perf_properties.compute_ops.items():
        if compute_ops.mma_ops > 0:
            device_mma_ops = _get_device_ops_for_dtype(device_profile.mma_ops, dtype)
            if device_mma_ops is not None:
                device_mma_ops *= device_profile.compute_efficiency
                mma_ops_time_s += compute_ops.mma_ops / device_mma_ops
            else:
                logger.warning(
                    "Ignoring mma compute ops of %s for %s since it is not supported on %s",
                    dtype,
                    op_invoke_info.func,
                    device_profile.name,
                )
        if compute_ops.gp_ops > 0:
            device_gp_ops = _get_device_ops_for_dtype(device_profile.gp_ops, dtype)
            if device_gp_ops is not None:
                device_gp_ops *= device_profile.compute_efficiency
                gp_ops_time_s += compute_ops.gp_ops / device_gp_ops
            else:
                logger.warning(
                    "Ignoring gp compute ops of %s for %s since it is not supported on %s",
                    dtype,
                    op_invoke_info.func,
                    device_profile.name,
                )
    compute_time_s = mma_ops_time_s + gp_ops_time_s
    memory_bandwidth = device_profile.memory_bandwidth_bytes_ps * device_profile.memory_efficiency
    memory_read_time_s = perf_properties.memory_read_bytes / memory_bandwidth
    memory_write_time_s = perf_properties.memory_write_bytes / memory_bandwidth
    memory_readwrite_time_s = perf_properties.memory_readwrite_bytes / memory_bandwidth
    memory_access_time_s = memory_read_time_s + memory_write_time_s + memory_readwrite_time_s
    time_s = max(compute_time_s, memory_access_time_s)
    result = PerformanceModel.Result(
        execution_time_s=time_s,
        statistics={
            "memory_read_time_s": memory_read_time_s,
            "memory_write_time_s": memory_write_time_s,
            "memory_readwrite_time_s": memory_readwrite_time_s,
            StatsKey.MEMORY_ACCESS: memory_access_time_s,
            StatsKey.COMPUTE: compute_time_s,
            StatsKey.MMA_OPS: mma_ops_time_s,
            StatsKey.GP_OPS: gp_ops_time_s,
            "is_compute_bound": compute_time_s > memory_access_time_s,
        },
    )
    return result


def _estimate_default(op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile) -> PerformanceModel.Result:
    result = _estimate_default_without_static_cost(op_invoke_info, device_profile)
    if result.execution_time_s == 0:
        return result
    perf_properties = op_invoke_info.get_perf_properties()
    static_cost_time_s = _estimate_static_cost(perf_properties, device_profile) * (
        1 + perf_properties.extra_static_cost_count
    )
    result.execution_time_s += static_cost_time_s
    result.statistics["static_cost_time_s"] = static_cost_time_s
    return result


register_op_estimator(None, None)(_estimate_default)


@register_op_estimator(torch.ops.tensor_cast._internal_wait_and_bind.default, None)
@register_op_estimator(torch.ops.tensor_cast._internal_record.default, None)
def _estimate_internal_multistream_anchor(
    op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile
) -> PerformanceModel.Result:
    return PerformanceModel.Result(0.0)


@register_op_estimator(torch.ops.tensor_cast.all_reduce.default, None)
@register_op_estimator(torch.ops.tensor_cast.all_gather.default, None)
@register_op_estimator(torch.ops.tensor_cast.reduce_scatter.default, None)
@register_op_estimator(torch.ops.tensor_cast.all_to_all.default, None)
def _estimate_collective_comm(op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile) -> PerformanceModel.Result:
    from .comm_analytic import CommAnalyticModel

    result = _estimate_default_without_static_cost(op_invoke_info, device_profile)
    comm_model = CommAnalyticModel(device_profile)
    comm_result = comm_model.process_op(op_invoke_info)
    result.combine(comm_result)
    result.execution_time_s += device_profile.static_cost.comm_op_cost_s
    return result


def _tag_statistics(stats: dict[str, object], prefix: str) -> dict[str, object]:
    tagged: dict[str, object] = {}
    for key, value in stats.items():
        key_name = key.value if hasattr(key, "value") else key
        tagged[f"{prefix}.{key_name}"] = value
    return tagged


def _combine_linear_all_reduce_results(
    linear_result: PerformanceModel.Result,
    comm_result: PerformanceModel.Result,
    overlap_label: str,
    stats_prefix: str,
    time_key: str,
) -> PerformanceModel.Result:
    result = PerformanceModel.Result(linear_result.execution_time_s, dict(linear_result.statistics))
    result.combine(PerformanceModel.Result(comm_result.execution_time_s, dict(comm_result.statistics)))
    result.statistics = {
        "overlap_model": overlap_label,
        time_key: linear_result.execution_time_s,
        "all_reduce_time_s": comm_result.execution_time_s,
    }
    result.statistics.update(_tag_statistics(linear_result.statistics, stats_prefix))
    result.statistics.update(_tag_statistics(comm_result.statistics, "all_reduce"))
    return result


@register_op_estimator(torch.ops.tensor_cast.matmul_all_reduce.default, None)
def _estimate_matmul_all_reduce(op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile) -> PerformanceModel.Result:
    mat1 = op_invoke_info.args[0]
    mat2 = op_invoke_info.args[1]
    rank = op_invoke_info.args[3]
    rank_group = op_invoke_info.args[4]

    mm_info = OpInvokeInfo(
        torch.ops.aten.mm.default,
        (mat1, mat2),
        None,
        op_invoke_info.out,
    )
    mm_result = _estimate_default(mm_info, device_profile)

    comm_info = OpInvokeInfo(
        torch.ops.tensor_cast.all_reduce.default,
        (op_invoke_info.out, rank, rank_group),
        None,
        op_invoke_info.out,
    )
    comm_result = _estimate_collective_comm(comm_info, device_profile)

    result = PerformanceModel.Result(mm_result.execution_time_s, dict(mm_result.statistics))
    result.combine(PerformanceModel.Result(comm_result.execution_time_s, dict(comm_result.statistics)))
    result.statistics = {
        "overlap_model": "max(matmul, all_reduce)",
        "matmul_time_s": mm_result.execution_time_s,
        "all_reduce_time_s": comm_result.execution_time_s,
    }
    result.statistics.update(_tag_statistics(mm_result.statistics, "matmul"))
    result.statistics.update(_tag_statistics(comm_result.statistics, "all_reduce"))
    return result


@register_op_estimator(torch.ops.tensor_cast.static_quant_linear_all_reduce.default, None)
def _estimate_static_quant_linear_all_reduce(
    op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile
) -> PerformanceModel.Result:
    (
        x,
        w,
        w_scale,
        w_offset,
        x_scale,
        x_offset,
        bias,
        out_dtype,
        rank,
        rank_group,
    ) = op_invoke_info.args

    linear_info = OpInvokeInfo(
        torch.ops.tensor_cast.static_quant_linear.default,
        (x, w, w_scale, w_offset, x_scale, x_offset, bias, out_dtype),
        None,
        op_invoke_info.out,
    )
    linear_result = _estimate_default(linear_info, device_profile)

    comm_info = OpInvokeInfo(
        torch.ops.tensor_cast.all_reduce.default,
        (op_invoke_info.out, rank, rank_group),
        None,
        op_invoke_info.out,
    )
    comm_result = _estimate_collective_comm(comm_info, device_profile)

    return _combine_linear_all_reduce_results(
        linear_result,
        comm_result,
        "max(static_quant_linear, all_reduce)",
        "static_quant_linear",
        "static_quant_linear_time_s",
    )


@register_op_estimator(torch.ops.tensor_cast.static_quant_linear_int4_all_reduce.default, None)
def _estimate_static_quant_linear_int4_all_reduce(
    op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile
) -> PerformanceModel.Result:
    (
        x,
        w,
        w_scale,
        w_offset,
        x_scale,
        x_offset,
        bias,
        out_dtype,
        rank,
        rank_group,
    ) = op_invoke_info.args

    linear_info = OpInvokeInfo(
        torch.ops.tensor_cast.static_quant_linear_int4.default,
        (x, w, w_scale, w_offset, x_scale, x_offset, bias, out_dtype),
        None,
        op_invoke_info.out,
    )
    linear_result = _estimate_default(linear_info, device_profile)

    comm_info = OpInvokeInfo(
        torch.ops.tensor_cast.all_reduce.default,
        (op_invoke_info.out, rank, rank_group),
        None,
        op_invoke_info.out,
    )
    comm_result = _estimate_collective_comm(comm_info, device_profile)

    return _combine_linear_all_reduce_results(
        linear_result,
        comm_result,
        "max(static_quant_linear_int4, all_reduce)",
        "static_quant_linear_int4",
        "static_quant_linear_int4_time_s",
    )


@register_op_estimator(torch.ops.tensor_cast.fp8_linear_all_reduce.default, None)
def _estimate_fp8_linear_all_reduce(
    op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile
) -> PerformanceModel.Result:
    (
        x,
        w,
        x_scale,
        w_scale,
        bias,
        out_dtype,
        rank,
        rank_group,
    ) = op_invoke_info.args

    linear_info = OpInvokeInfo(
        torch.ops.tensor_cast.fp8_linear.default,
        (x, w, x_scale, w_scale, bias, out_dtype),
        None,
        op_invoke_info.out,
    )
    linear_result = _estimate_default(linear_info, device_profile)

    comm_info = OpInvokeInfo(
        torch.ops.tensor_cast.all_reduce.default,
        (op_invoke_info.out, rank, rank_group),
        None,
        op_invoke_info.out,
    )
    comm_result = _estimate_collective_comm(comm_info, device_profile)

    return _combine_linear_all_reduce_results(
        linear_result,
        comm_result,
        "max(fp8_linear, all_reduce)",
        "fp8_linear",
        "fp8_linear_time_s",
    )


@register_op_estimator(torch.ops.tensor_cast.mxfp4_linear_all_reduce.default, None)
def _estimate_mxfp4_linear_all_reduce(
    op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile
) -> PerformanceModel.Result:
    (
        x,
        w,
        x_scale,
        w_scale,
        bias,
        out_dtype,
        rank,
        rank_group,
    ) = op_invoke_info.args

    linear_info = OpInvokeInfo(
        torch.ops.tensor_cast.mxfp4_linear.default,
        (x, w, x_scale, w_scale, bias, out_dtype),
        None,
        op_invoke_info.out,
    )
    linear_result = _estimate_default(linear_info, device_profile)

    comm_info = OpInvokeInfo(
        torch.ops.tensor_cast.all_reduce.default,
        (op_invoke_info.out, rank, rank_group),
        None,
        op_invoke_info.out,
    )
    comm_result = _estimate_collective_comm(comm_info, device_profile)

    return _combine_linear_all_reduce_results(
        linear_result,
        comm_result,
        "max(mxfp4_linear, all_reduce)",
        "mxfp4_linear",
        "mxfp4_linear_time_s",
    )


# ---------------------------------------------------------------------------
# DFC (DispatchFFNCombine) analytic roofline estimator
# ---------------------------------------------------------------------------


_INT4_GMM_TARGETS = frozenset(
    {
        torch.ops.tensor_cast.grouped_matmul_quant_int4.default,
        torch.ops.tensor_cast.grouped_matmul_quant_int4_swiglu.default,
    }
)


def _compute_time_from_properties(
    properties: OpInvokeInfo.PerformanceProperties,
    device_profile: DeviceProfile,
) -> float:
    """Extract FLOPs from op properties and compute time (no memory).

    Logic mirrors _estimate_default_without_static_cost but returns only
    the compute component, ignoring memory access time.  Used by the DFC
    estimator to avoid double-counting intermediate activation HBM access.
    """
    compute_time_s = 0.0
    for dtype, compute_ops in properties.compute_ops.items():
        device_mma_ops = _get_device_ops_for_dtype(device_profile.mma_ops, dtype)
        if compute_ops.mma_ops > 0 and device_mma_ops is not None:
            device_mma_ops *= device_profile.compute_efficiency
            compute_time_s += compute_ops.mma_ops / device_mma_ops
        device_gp_ops = _get_device_ops_for_dtype(device_profile.gp_ops, dtype)
        if compute_ops.gp_ops > 0 and device_gp_ops is not None:
            device_gp_ops *= device_profile.compute_efficiency
            compute_time_s += compute_ops.gp_ops / device_gp_ops
    return compute_time_s


def _logical_weight_k(w: torch.Tensor, is_int4: bool) -> int:
    """Derive logical K (input dimension) from a weight tensor.

    For INT4 packed weights the physical shape encodes 2 values per byte,
    so shape[0] is K/pack_factor.  Uses the same pack_factor formula as
    _static_quant_linear_properties_helper (L120-139).

    For all other dtypes (BF16, INT8, FP8, MXFP4) shape[0] is logical K.
    """
    if is_int4 and w.dim() == 2:
        pack_factor = (w.element_size() * 8) // 4
        logical_total = w.numel() * pack_factor
        return logical_total // w.shape[1]
    return w.shape[0]


def _estimate_dfc_common(
    op_invoke_info: OpInvokeInfo,
    device_profile: DeviceProfile,
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    gmm1_swiglu_target,
    gmm1_w_args: tuple,
    gmm2_target,
    gmm2_w_args: tuple,
    rank: int,
    rank_group,
) -> PerformanceModel.Result:
    """Core DFC estimator: T_dfc = max(T_compute, T_memory) + T_comm.

    T_compute = T_gmm1 + T_gmm2  (sum, no pipeline overlap)
    T_memory  = real HBM only (x + weights + output, NOT intermediates)
    T_comm    = 2 * T_all_to_all  (serial, no overlap with compute)
    """
    # --- Build dummy activation for grouped_matmul calls ---
    # M_total = total dispatched tokens across all experts = bs * seq * top_k
    M_total = expert_indices.numel()
    hidden_size = x.shape[-1]

    # Use the weight dtype for dummy activation so that MMA throughput is
    # computed for the correct dtype (e.g. INT8 for W8A8, not float16).
    # This is critical: grouped_matmul properties key MMA ops off x.dtype.
    first_w = gmm1_w_args[0]
    if isinstance(first_w, (list, tuple)):
        first_w = first_w[0] if first_w else None
    raw_weight_dtype = first_w.dtype if first_w is not None else x.dtype
    # INT4 packed weights use uint8 storage (2 values per byte), but MMA
    # runs on INT8.  Map uint8 → int8 so DeviceProfile throughput lookup works.
    weight_dtype = torch.int8 if raw_weight_dtype == torch.uint8 else raw_weight_dtype

    # Determine number of experts from the weight list length.
    # gmm1_w_args[0] is the weight list (List[Tensor]), one per expert.
    first_w_list = gmm1_w_args[0]
    num_experts = len(first_w_list) if isinstance(first_w_list, (list, tuple)) else 1

    # Build dummy activation list matching the expert count.
    # Distribute M_total evenly across experts so FLOPs and memory are both correct.
    # FLOPs = sum(M_i * N * K * 2) = M_total * N * K * 2 when all experts share (N,K).
    tokens_per_expert = max(1, M_total // num_experts) if num_experts > 0 else M_total
    dummy_gmm1_x_list = [
        torch.empty((tokens_per_expert, hidden_size), dtype=weight_dtype, device="meta") for _ in range(num_experts)
    ]

    # --- GMM1 (gate_up_proj + SwiGLU) ---
    gmm1_full_args = _build_grouped_gmm_args_for_estimator(gmm1_swiglu_target, dummy_gmm1_x_list, gmm1_w_args)
    gmm1_out = gmm1_swiglu_target(*gmm1_full_args)
    gmm1_info = OpInvokeInfo(gmm1_swiglu_target, gmm1_full_args, None, gmm1_out)
    gmm1_props = gmm1_info.get_perf_properties()
    gmm1_compute_s = _compute_time_from_properties(gmm1_props, device_profile)

    # --- GMM2 (down_proj) ---
    gmm2_first_w_list = gmm2_w_args[0]
    gmm2_first_w = gmm2_first_w_list[0] if isinstance(gmm2_first_w_list, (list, tuple)) else gmm2_first_w_list
    gmm2_weight_dtype = gmm2_first_w.dtype if gmm2_first_w is not None else weight_dtype
    # Derive logical K from weight shape. For INT4 packed weights the physical
    # shape[0] is K/pack_factor; use the same is_int4 + pack_factor logic as
    # _static_quant_linear_properties_helper (L120-139) for consistency.
    is_int4 = gmm2_target in _INT4_GMM_TARGETS
    gmm2_K = _logical_weight_k(gmm2_first_w, is_int4)
    gmm2_num_experts = len(gmm2_first_w_list) if isinstance(gmm2_first_w_list, (list, tuple)) else 1
    dummy_gmm2_x_list = [
        torch.empty((tokens_per_expert, gmm2_K), dtype=gmm2_weight_dtype, device="meta")
        for _ in range(gmm2_num_experts)
    ]

    gmm2_full_args = _build_grouped_gmm_args_for_estimator(gmm2_target, dummy_gmm2_x_list, gmm2_w_args)
    gmm2_out = gmm2_target(*gmm2_full_args)
    gmm2_info = OpInvokeInfo(gmm2_target, gmm2_full_args, None, gmm2_out)
    gmm2_props = gmm2_info.get_perf_properties()
    gmm2_compute_s = _compute_time_from_properties(gmm2_props, device_profile)

    total_compute_s = gmm1_compute_s + gmm2_compute_s

    # --- HBM memory: x + expert_indices + all weights + output (no intermediates) ---
    memory_bytes = 0.0
    # Input activation + routing indices
    memory_bytes += bytes_of_tensor(x)
    memory_bytes += bytes_of_tensor(expert_indices)
    # Output
    memory_bytes += bytes_of_tensor(op_invoke_info.out)
    # GMM1 weight args (already weights-only, iterate all list-of-tensor args)
    for a in gmm1_w_args:
        if isinstance(a, (list, tuple)):
            for t in a:
                if isinstance(t, torch.Tensor):
                    memory_bytes += bytes_of_tensor(t)
        elif isinstance(a, torch.Tensor):
            memory_bytes += bytes_of_tensor(a)
    # GMM2 weight args
    for a in gmm2_w_args:
        if isinstance(a, (list, tuple)):
            for t in a:
                if isinstance(t, torch.Tensor):
                    memory_bytes += bytes_of_tensor(t)
        elif isinstance(a, torch.Tensor):
            memory_bytes += bytes_of_tensor(a)

    memory_bandwidth = device_profile.memory_bandwidth_bytes_ps * device_profile.memory_efficiency
    memory_time_s = memory_bytes / memory_bandwidth

    # --- Communication: 2 x all_to_all (dispatch + combine) ---
    # Comm runs on the ROUTED tensor (after init_routing_v2), not pre-routing x.
    # Routed tensor shape = (M_total, hidden_size) where M_total = bs*seq*top_k.
    comm_time_s = 0.0
    ep_size = len(rank_group) if isinstance(rank_group, (list, tuple)) else 1
    if ep_size > 1 and M_total > 0:
        tokens_per_rank = max(1, M_total // ep_size)
        split_sizes = [tokens_per_rank] * ep_size
        # DFC kernel dispatches quantized tokens: INT8 for W8A8, BF16 for BF16.
        # weight_dtype matches the dispatch dtype for all current variants.
        # TODO: if a future variant breaks this assumption, add explicit
        # comm_dtype parameter to _estimate_dfc_common.
        routed_x = torch.empty((M_total, hidden_size), dtype=weight_dtype, device="meta")

        comm_info = OpInvokeInfo(
            torch.ops.tensor_cast.all_to_all.default,
            (routed_x, split_sizes, split_sizes, rank, rank_group),
            None,
            routed_x,
        )
        one_a2a_result = _estimate_collective_comm(comm_info, device_profile)
        comm_time_s = 2 * one_a2a_result.execution_time_s

    # --- Combine: max(compute, memory) + comm ---
    # Spec formula: T_dfc = max(T_compute, T_memory) + T_comm
    # No extra static_cost — DFC is a single fused kernel launch.
    roofline_time_s = max(total_compute_s, memory_time_s)
    total_time_s = roofline_time_s + comm_time_s

    result = PerformanceModel.Result(
        execution_time_s=total_time_s,
        statistics={
            "overlap_model": "max(gmm1+gmm2, memory) + 2*all_to_all",
            "gmm1_compute_s": gmm1_compute_s,
            "gmm2_compute_s": gmm2_compute_s,
            StatsKey.COMPUTE: total_compute_s,
            StatsKey.MEMORY_ACCESS: memory_time_s,
            "memory_bytes": memory_bytes,
            "comm_time_s": comm_time_s,
            "is_compute_bound": total_compute_s > memory_time_s,
        },
    )
    return result


def _build_grouped_gmm_args_for_estimator(gmm_target, dummy_x_list: list[torch.Tensor], gmm_w_args: tuple) -> tuple:
    """Materialize grouped_matmul args for estimator-only dummy invocation."""
    if gmm_target in {
        torch.ops.tensor_cast.grouped_matmul_quant_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_quant_int4_swiglu.default,
        torch.ops.tensor_cast.grouped_matmul_quant.default,
        torch.ops.tensor_cast.grouped_matmul_quant_int4.default,
    }:
        if len(gmm_w_args) != 5:
            raise ValueError(
                f"Unexpected DFC grouped quant GMM weight arg count for {gmm_target}: expected 5, got {len(gmm_w_args)}"
            )
        gmm_w, gmm_ws, gmm_wo, gmm_bias, gmm_dt = gmm_w_args
        x_scale = [torch.empty((), dtype=torch.float32, device="meta")] * len(dummy_x_list)
        x_offset = [None] * len(dummy_x_list)
        return (
            dummy_x_list,
            gmm_w,
            gmm_ws,
            gmm_wo,
            x_scale,
            x_offset,
            gmm_bias,
            gmm_dt,
        )
    return (dummy_x_list, *gmm_w_args)


@register_op_estimator(torch.ops.tensor_cast.dispatch_ffn_combine.default, None)
def _estimate_dfc_bf16(op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile) -> PerformanceModel.Result:
    (x, expert_indices, gmm1_w, gmm1_bias, gmm2_w, gmm2_bias, rank, rank_group) = op_invoke_info.args
    return _estimate_dfc_common(
        op_invoke_info,
        device_profile,
        x,
        expert_indices,
        gmm1_swiglu_target=torch.ops.tensor_cast.grouped_matmul_swiglu.default,
        gmm1_w_args=(gmm1_w, gmm1_bias),
        gmm2_target=torch.ops.tensor_cast.grouped_matmul.default,
        gmm2_w_args=(gmm2_w, gmm2_bias),
        rank=rank,
        rank_group=rank_group,
    )


@register_op_estimator(torch.ops.tensor_cast.dispatch_ffn_combine_quant.default, None)
def _estimate_dfc_quant(op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile) -> PerformanceModel.Result:
    (
        x,
        ei,
        gmm1_w,
        gmm1_ws,
        gmm1_wo,
        gmm1_bias,
        gmm1_dt,
        gmm2_w,
        gmm2_ws,
        gmm2_wo,
        gmm2_bias,
        gmm2_dt,
        rank,
        rg,
    ) = op_invoke_info.args
    return _estimate_dfc_common(
        op_invoke_info,
        device_profile,
        x,
        ei,
        gmm1_swiglu_target=torch.ops.tensor_cast.grouped_matmul_quant_swiglu.default,
        gmm1_w_args=(gmm1_w, gmm1_ws, gmm1_wo, gmm1_bias, gmm1_dt),
        gmm2_target=torch.ops.tensor_cast.grouped_matmul_quant.default,
        gmm2_w_args=(gmm2_w, gmm2_ws, gmm2_wo, gmm2_bias, gmm2_dt),
        rank=rank,
        rank_group=rg,
    )


@register_op_estimator(torch.ops.tensor_cast.dispatch_ffn_combine_quant_int4.default, None)
def _estimate_dfc_quant_int4(op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile) -> PerformanceModel.Result:
    (
        x,
        ei,
        gmm1_w,
        gmm1_ws,
        gmm1_wo,
        gmm1_bias,
        gmm1_dt,
        gmm2_w,
        gmm2_ws,
        gmm2_wo,
        gmm2_bias,
        gmm2_dt,
        rank,
        rg,
    ) = op_invoke_info.args
    return _estimate_dfc_common(
        op_invoke_info,
        device_profile,
        x,
        ei,
        gmm1_swiglu_target=torch.ops.tensor_cast.grouped_matmul_quant_int4_swiglu.default,
        gmm1_w_args=(gmm1_w, gmm1_ws, gmm1_wo, gmm1_bias, gmm1_dt),
        gmm2_target=torch.ops.tensor_cast.grouped_matmul_quant_int4.default,
        gmm2_w_args=(gmm2_w, gmm2_ws, gmm2_wo, gmm2_bias, gmm2_dt),
        rank=rank,
        rank_group=rg,
    )


@register_op_estimator(torch.ops.tensor_cast.dispatch_ffn_combine_fp8.default, None)
def _estimate_dfc_fp8(op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile) -> PerformanceModel.Result:
    (
        x,
        ei,
        gmm1_w,
        gmm1_ws,
        gmm1_xs,
        gmm1_bias,
        gmm1_dt,
        gmm2_w,
        gmm2_ws,
        gmm2_xs,
        gmm2_bias,
        gmm2_dt,
        rank,
        rg,
    ) = op_invoke_info.args
    return _estimate_dfc_common(
        op_invoke_info,
        device_profile,
        x,
        ei,
        gmm1_swiglu_target=torch.ops.tensor_cast.grouped_matmul_fp8_swiglu.default,
        gmm1_w_args=(gmm1_w, gmm1_ws, gmm1_xs, gmm1_bias, gmm1_dt),
        gmm2_target=torch.ops.tensor_cast.grouped_matmul_fp8.default,
        gmm2_w_args=(gmm2_w, gmm2_ws, gmm2_xs, gmm2_bias, gmm2_dt),
        rank=rank,
        rank_group=rg,
    )


@register_op_estimator(torch.ops.tensor_cast.dispatch_ffn_combine_mxfp4.default, None)
def _estimate_dfc_mxfp4(op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile) -> PerformanceModel.Result:
    (
        x,
        ei,
        gmm1_w,
        gmm1_ws,
        gmm1_xs,
        gmm1_bias,
        gmm1_dt,
        gmm2_w,
        gmm2_ws,
        gmm2_xs,
        gmm2_bias,
        gmm2_dt,
        rank,
        rg,
    ) = op_invoke_info.args
    return _estimate_dfc_common(
        op_invoke_info,
        device_profile,
        x,
        ei,
        gmm1_swiglu_target=torch.ops.tensor_cast.grouped_matmul_mxfp4_swiglu.default,
        gmm1_w_args=(gmm1_w, gmm1_ws, gmm1_xs, gmm1_bias, gmm1_dt),
        gmm2_target=torch.ops.tensor_cast.grouped_matmul_mxfp4.default,
        gmm2_w_args=(gmm2_w, gmm2_ws, gmm2_xs, gmm2_bias, gmm2_dt),
        rank=rank,
        rank_group=rg,
    )


# dsa_indexer scores query tokens against a paged historical indexer cache.  Model
# sparse/random cache access as physical historical-cache bytes expanded by an
# efficiency factor; do not model hardware-specific L1/L2/MTE behavior here.
# Formula: historical_effective_read_bytes = historical_physical_read_bytes / eta.
_DSA_INDEXER_PAGED_CACHE_READ_EFFICIENCY_DECODE_BASE = 0.30
_DSA_INDEXER_PAGED_CACHE_READ_EFFICIENCY_DECODE_LOG2_BLOCK_SCALE = 0.23
_DSA_INDEXER_PAGED_CACHE_READ_EFFICIENCY_DECODE_REFERENCE_BLOCK_COUNT = 128
_DSA_INDEXER_PAGED_CACHE_READ_EFFICIENCY_DECODE_FLOOR = 0.15
# Prefill walks many adjacent queries over the same historical cache span, so use
# a fixed calibrated efficiency for the current fixed-block-size paged-cache layout.
# Formula: eta_prefill = prefill_eta.
_DSA_INDEXER_PAGED_CACHE_READ_EFFICIENCY_PREFILL = 0.90
# Fraction of prefill query tokens that effectively cause independent historical
# cache reads after adjacent-query reuse is amortized.  This is a generic data reuse
# factor, not a cache-level-specific optimization.  Formula:
# prefill_query_units = ceil(query_len * prefill_query_scale).
_DSA_INDEXER_PREFILL_CACHE_READ_QUERY_SCALE = 0.25


def _sum_request_cache_blocks(request_total_seq_lens: torch.Tensor, block_size: int) -> int:
    """Estimate total active cache blocks as sum(ceil(seq_len_i / block_size)). dsa_indexer receives active request lengths from the op invocation, so the roofline model can use those lengths directly instead of falling back to shape-only block-table estimates."""
    cache_blocks = torch.div(request_total_seq_lens + block_size - 1, block_size, rounding_mode="floor")
    return int(cache_blocks.sum().item())


def _is_dsa_indexer_decode_mode(query_len: int) -> bool:
    """Classify dsa_indexer decode mode with the same threshold rule as sparse MLA. Reusing _sparse_mla_decode_mask keeps single-token decode and small MTP decode classification consistent across the two sparse-attention-related operators."""
    return bool(_sparse_mla_decode_mask(torch.tensor(query_len)).item())


def _estimate_dsa_indexer_decode_page_efficiency(logical_cache_block_count: int) -> float:
    """Estimate decode historical-cache read efficiency from active cache blocks.
    eta = max(floor, min(base + scale * log2(block_count / reference_block_count), 1));
    more blocks amortize random page traversal, but efficiency is bounded to
    avoid hardware-specific cache assumptions.
    """
    block_count = max(logical_cache_block_count, 1)
    log2_block_scale = math.log2(block_count / _DSA_INDEXER_PAGED_CACHE_READ_EFFICIENCY_DECODE_REFERENCE_BLOCK_COUNT)
    eta = (
        _DSA_INDEXER_PAGED_CACHE_READ_EFFICIENCY_DECODE_BASE
        + _DSA_INDEXER_PAGED_CACHE_READ_EFFICIENCY_DECODE_LOG2_BLOCK_SCALE * log2_block_scale
    )
    return max(_DSA_INDEXER_PAGED_CACHE_READ_EFFICIENCY_DECODE_FLOOR, min(eta, 1.0))


def _estimate_dsa_indexer_paged_cache_read_breakdown(
    hidden_states: torch.Tensor,
    indexer_cache: torch.Tensor,
    head_dim: int,
    request_total_seq_lens: torch.Tensor,
    fp8_mode: bool = False,
):
    """Break down dsa_indexer cache bytes into penalized historical reads and physical append writes. It computes context_work = query_units * sum(seq_lens), historical_physical_bytes from cache element width plus fp8 scale bytes, then historical_effective_bytes = historical_physical_bytes / eta because only historical random reads lose bandwidth; appends remain linear writes."""
    batch, query_len, _ = hidden_states.shape
    index_head_dim = indexer_cache.size(-1)
    block_size = indexer_cache.size(1)
    dtype_size = indexer_cache.element_size()
    is_decode = _is_dsa_indexer_decode_mode(query_len)
    logical_cache_block_count = _sum_request_cache_blocks(request_total_seq_lens, block_size)
    # Decode touches a small number of query tokens, so efficiency is dominated by
    # random page traversal and improves slowly as more cache blocks amortize overhead.
    # Formula: eta_decode = max(floor, min(base + scale * log2(block_count / reference_block_count), 1)).
    if is_decode:
        eta = _estimate_dsa_indexer_decode_page_efficiency(logical_cache_block_count)
    else:
        # Prefill sees adjacent-query reuse and longer contiguous cache walks.  The
        # current paged-cache block size is fixed, so use the calibrated constant directly.
        eta = _DSA_INDEXER_PAGED_CACHE_READ_EFFICIENCY_PREFILL
    summed_seq_lens = int(request_total_seq_lens.sum().item())
    # Decode has little query-side reuse, so every query token probes the historical
    # cache.  Prefill amortizes repeated historical-cache reads across neighboring
    # query tokens with the generic query scale above.  Formula:
    # context_work = query_units * sum(request_total_seq_lens).
    if is_decode:
        query_units = query_len
    else:
        query_units = math.ceil(query_len * _DSA_INDEXER_PREFILL_CACHE_READ_QUERY_SCALE)
    context_work = query_units * summed_seq_lens
    # Historical reads are the only traffic penalized by sparse/random access.  The
    # append path writes the current query tokens into cache linearly and remains
    # physical byte traffic.  Formula:
    # historical_physical_read_bytes = context_work * (index_head_dim * dtype_size + fp8_scale_bytes_per_token).
    historical_cache_read_bytes = context_work * index_head_dim * dtype_size
    scale_bytes_per_token = ((head_dim + 127) // 128) * 4 if fp8_mode else 0
    historical_scale_read_bytes = context_work * scale_bytes_per_token
    historical_physical_read_bytes = historical_cache_read_bytes + historical_scale_read_bytes
    historical_effective_read_bytes = historical_physical_read_bytes / eta if historical_physical_read_bytes != 0 else 0
    append_cache_write_bytes = batch * query_len * index_head_dim * dtype_size
    append_scale_write_bytes = batch * query_len * scale_bytes_per_token

    return {
        "historical_effective_read_bytes": historical_effective_read_bytes,
        "append_cache_write_bytes": append_cache_write_bytes,
        "append_scale_write_bytes": append_scale_write_bytes,
    }


def _estimate_dsa_indexer_breakdown(
    hidden_states: torch.Tensor,
    qa_normed: torch.Tensor,
    indexer_cache: torch.Tensor,
    num_heads: int,
    head_dim: int,
    qk_rope_head_dim: int,
    topk_limit: int,
    request_total_seq_lens: torch.Tensor,
    fp8_mode: bool = False,
):
    """Estimate dsa_indexer MMA/GP work and cache traffic for analytic roofline. Projection and score math use active sequence lengths for compute, while cache traffic is delegated to the paged-cache breakdown so sparse historical reads are expanded by efficiency and append writes stay physical."""
    batch, seq_len, hidden_size = hidden_states.shape
    q_lora_rank = qa_normed.shape[-1]

    # DeepSeek-V3.2 splits the Indexer into a few distinct buckets:
    #   1) q projection from the low-rank query stream into [num_heads, head_dim]
    #   2) k projection from hidden states into head_dim
    #   3) head-routing projection from hidden states into num_heads
    #   4) RoPE on the query/key rotary slices
    #   5) fp8-only rotate_activation + blockwise quantization
    #   6) fp8_index-style scoring over the active cache
    #   7) top-k selection over the active sequence axis
    #
    # The counts below are intentionally split so the roofline model can place the
    # heavy matrix multiplies in MMA buckets and the elementwise / reduction work in
    # GP buckets.  All matrix multiply terms use FMA -> 2 FLOPs.

    # 1) q projection: (B * S, q_lora_rank) @ (q_lora_rank, H * D)
    q_proj_mma = 2 * batch * seq_len * q_lora_rank * num_heads * head_dim

    # 2) k projection: (B * S, hidden_size) @ (hidden_size, D)
    k_proj_mma = 2 * batch * seq_len * hidden_size * head_dim

    # 3) head routing projection: (B * S, hidden_size) @ (hidden_size, H)
    # This is the learned head-weight projection used to mix per-head scores.
    weights_proj_mma = 2 * batch * seq_len * hidden_size * num_heads

    # 4) RoPE work on the q/k rotary slices.  The q slice is per-head, while k is
    #    shared across heads, so the two terms have different widths.
    rope_gp = batch * seq_len * (num_heads * qk_rope_head_dim + qk_rope_head_dim) * 3

    # 5) fp8-only activation rotation and blockwise quantization.  DeepSeek-V3.2's
    #    reference kernel applies a Hadamard-style rotate_activation before quantizing
    #    q and k to FP8.  We model those as GP because they are elementwise transforms.
    rotate_activation_gp = 0
    act_quant_gp = 0
    if fp8_mode:
        # rotate_activation is applied after q/k are fully reassembled, so it runs
        # over the full query head tensor plus the full key tensor.
        rotate_activation_gp = batch * seq_len * (num_heads * head_dim + head_dim)
        # act_quant performs blockwise fp8 quantization over the same full tensors.
        act_quant_gp = batch * seq_len * (num_heads * head_dim + head_dim)

    # 6) score work.  The reference implementation computes a per-head score tensor
    #    against the active cache, then combines those head scores with the learned
    #    routing weights into one index score.
    active_cache_len = int(request_total_seq_lens.max().item())
    qk_index_mma = 2 * batch * seq_len * num_heads * active_cache_len * head_dim

    paged_cache_read_breakdown = _estimate_dsa_indexer_paged_cache_read_breakdown(
        hidden_states,
        indexer_cache,
        head_dim,
        request_total_seq_lens=request_total_seq_lens,
        fp8_mode=fp8_mode,
    )
    # BF16/GLM5-style head mixing keeps the learned weight multiply in the base path.
    # The fp8-only terms are layered on top when fp8_mode is enabled.
    head_weight_mul_gp = batch * seq_len * num_heads * active_cache_len
    head_reduce_gp = batch * seq_len * num_heads * active_cache_len
    head_relu_gp = 0
    head_q_scale_mul_gp = 0
    head_k_scale_mul_gp = 0
    if fp8_mode:
        # DeepSeek-V3.2 adds fp8-only score shaping around the head mix.
        head_relu_gp = batch * seq_len * num_heads * active_cache_len
        head_q_scale_mul_gp = batch * seq_len * num_heads * active_cache_len
        head_k_scale_mul_gp = batch * seq_len * active_cache_len

    # 7) top-k selection over the active axis.
    topk_gp = batch * seq_len * active_cache_len

    return {
        "q_proj_mma": q_proj_mma,
        "k_proj_mma": k_proj_mma,
        "weights_proj_mma": weights_proj_mma,
        "rope_gp": rope_gp,
        "rotate_activation_gp": rotate_activation_gp,
        "act_quant_gp": act_quant_gp,
        "qk_index_mma": qk_index_mma,
        "head_relu_gp": head_relu_gp,
        "head_q_scale_mul_gp": head_q_scale_mul_gp,
        "head_weight_mul_gp": head_weight_mul_gp,
        "head_reduce_gp": head_reduce_gp,
        "head_k_scale_mul_gp": head_k_scale_mul_gp,
        "topk_gp": topk_gp,
        **paged_cache_read_breakdown,
    }


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.dsa_indexer.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    hidden_states = op_invoke_info.args[0]
    qa_normed = op_invoke_info.args[1]
    indexer_cache = op_invoke_info.args[4]
    seq_lens = op_invoke_info.args[7]
    num_heads = op_invoke_info.args[12]
    head_dim = op_invoke_info.args[13]
    qk_rope_head_dim = op_invoke_info.args[14]
    topk_limit = op_invoke_info.args[15]

    # The cache dtype is already chosen by the upstream attention quant config,
    # so the roofline model only needs to read it, not re-decide fp8 policy here.
    fp8_mode = is_fp8_dtype(indexer_cache.dtype)

    breakdown = _estimate_dsa_indexer_breakdown(
        hidden_states,
        qa_normed,
        indexer_cache,
        num_heads,
        head_dim,
        qk_rope_head_dim,
        topk_limit,
        request_total_seq_lens=seq_lens,
        fp8_mode=fp8_mode,
    )
    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={4})

    # Projection math is the dominant MMA part of the Indexer path.
    _accumulate_compute_ops(
        properties,
        hidden_states.dtype,
        mma_ops=(breakdown["q_proj_mma"] + breakdown["k_proj_mma"] + breakdown["weights_proj_mma"]),
        gp_ops=(
            breakdown["rope_gp"]
            + breakdown["rotate_activation_gp"]
            + breakdown["act_quant_gp"]
            + breakdown["head_weight_mul_gp"]
            + breakdown["head_reduce_gp"]
            + breakdown["topk_gp"]
        ),
    )

    # The core q/k scoring kernel is bucketed separately: in bf16 mode it stays in the
    # activation dtype bucket, while in fp8 mode it moves to the cache/score dtype.
    score_dtype = indexer_cache.dtype if fp8_mode else hidden_states.dtype
    _accumulate_compute_ops(
        properties,
        score_dtype,
        mma_ops=breakdown["qk_index_mma"],
        gp_ops=(breakdown["head_relu_gp"] + breakdown["head_q_scale_mul_gp"] + breakdown["head_k_scale_mul_gp"]),
    )

    properties.memory_read_bytes += breakdown["historical_effective_read_bytes"]
    properties.memory_write_bytes += breakdown["append_cache_write_bytes"] + breakdown["append_scale_write_bytes"]
    return properties


from . import builtin_model  # noqa: E402,F401  # Triggers built-in op registrations at import time.

_load_custom_op()
