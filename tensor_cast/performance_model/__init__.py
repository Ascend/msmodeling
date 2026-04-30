import logging
import math
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch

from .. import ops  # noqa: F401
from ..device import DeviceProfile
from .analytic import StatsKey
from .base import PerformanceModel
from .op_estimator_registry import register_op_estimator
from .op_invoke_info import OpInvokeInfo
from .utils import bytes_of_elements, bytes_of_tensor, is_noop_self_copy_op, is_view_op

logger = logging.getLogger(__name__)


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


def _mm_properties_helper(
    op_invoke_info: OpInvokeInfo, mat1, mat2, bias
) -> OpInvokeInfo.PerformanceProperties:
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
        compute_ops = properties.compute_ops.setdefault(
            bias.dtype, OpInvokeInfo.ComputeOps()
        )
        compute_ops.gp_ops = bias_ops
        properties.compute_ops[bias.dtype] = compute_ops

    return properties


@OpInvokeInfo.register_op_properties(torch.ops.aten.mm.default)
def _(op_invoke_info: OpInvokeInfo) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 2
    return _mm_properties_helper(
        op_invoke_info, op_invoke_info.args[0], op_invoke_info.args[1], None
    )


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
        compute_ops = properties.compute_ops.setdefault(
            torch.float32, OpInvokeInfo.ComputeOps()
        )
        compute_ops.gp_ops = dequant_ops
        properties.compute_ops[torch.float32] = OpInvokeInfo.ComputeOps()
    if bias is not None:
        compute_ops = properties.compute_ops.setdefault(
            bias.dtype, OpInvokeInfo.ComputeOps()
        )
        compute_ops.gp_ops = bias_ops
        properties.compute_ops[bias.dtype] = compute_ops

    return properties


@OpInvokeInfo.register_op_properties(
    torch.ops.tensor_cast.static_quant_linear_int4.default
)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 3
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    w_offset = op_invoke_info.args[3] if len(op_invoke_info.args) > 3 else None
    bias = op_invoke_info.args[6] if len(op_invoke_info.args) > 6 else None
    return _static_quant_linear_properties_helper(
        op_invoke_info, x, w, w_offset, bias, is_int4=True
    )


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.static_quant_linear.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 3
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    w_offset = op_invoke_info.args[3] if len(op_invoke_info.args) > 3 else None
    bias = op_invoke_info.args[6] if len(op_invoke_info.args) > 6 else None
    return _static_quant_linear_properties_helper(
        op_invoke_info, x, w, w_offset, bias, is_int4=False
    )


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.fp8_linear.default)
@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.mxfp4_linear.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 3
    x = op_invoke_info.args[0]
    w = op_invoke_info.args[1]
    bias = op_invoke_info.args[4] if len(op_invoke_info.args) > 4 else None
    return _static_quant_linear_properties_helper(
        op_invoke_info, x, w, None, bias, is_int4=False
    )


@OpInvokeInfo.register_op_properties(torch.ops.aten.embedding.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) >= 2
    weight = op_invoke_info.args[0]
    indices = op_invoke_info.args[1]
    properties = op_invoke_info.get_memory_access_properties(exclude_input_ids={0})
    properties.memory_read_bytes += (
        bytes_of_tensor(indices, weight.dtype) * weight.shape[-1]
    )
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
    properties.memory_read_bytes += (
        bytes_of_tensor(input) * index.numel() / input.shape[dim]
    )
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
    properties.memory_write_bytes += bytes_of_tensor(
        key, kv_cache.dtype
    ) + bytes_of_tensor(value, kv_cache.dtype)
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
        assert block_table is None, (
            "4D query implies no KV cache; block_table must be None"
        )
        batch_size, query_len_per_seq, num_q_heads, head_size = query.size()
        assert key.ndim == 4, "key size must be 4"
        _, key_len_per_seq, _, _ = key.size()
        context_len_product_sum = batch_size * query_len_per_seq * key_len_per_seq
    else:
        hidden_size = query.size(-1)
        head_size = key.size(-1)
        assert hidden_size % head_size == 0
        num_q_heads = hidden_size // head_size

        context_len_product_sum = torch.sum(
            query_lens.to(seq_lens.dtype) * seq_lens
        ).item()

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
        properties = op_invoke_info.get_memory_access_properties(
            exclude_input_ids={1, 2}
        )
        properties.memory_read_bytes += torch.sum(
            seq_lens * 2 * bytes_of_elements(key.size(-1) * key.size(-2), key.dtype)
        ).item()

    compute_ops = properties.compute_ops.setdefault(
        query.dtype, OpInvokeInfo.ComputeOps()
    )
    compute_ops.mma_ops = bmm1_ops + bmm2_ops
    compute_ops = properties.compute_ops.setdefault(
        softmax_dtype, OpInvokeInfo.ComputeOps()
    )
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

    conv_gp_ops = num_tokens * conv_dim * conv_kernel_size * 2 + _elementwise_silu_ops(
        num_tokens * conv_dim
    )
    beta_gp_ops = _elementwise_sigmoid_ops(num_tokens * num_v_heads)

    # g = -exp(A_log.float()) * softplus(a.float() + dt_bias)
    g_gp_ops = num_v_heads + num_tokens * num_v_heads * (
        1 + _elementwise_softplus_ops(1) + 1 + 1
    )

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
        total_chunk_pairs * (head_k_dim + head_v_dim) * 2
        + total_positions * head_k_dim * head_v_dim * 6
    )

    qk_l2norm_gp_ops = _l2norm_ops(valid_positions, head_k_dim) * 2
    prefix_correction_gp_ops = (
        batch_heads
        * num_chunks
        * (chunk_size - 1)
        * chunk_size
        * (2 * chunk_size - 1)
        // 3
    )

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
    num_k_heads = op_invoke_info.args[3]
    num_v_heads = op_invoke_info.args[4]
    head_k_dim = op_invoke_info.args[5]
    head_v_dim = op_invoke_info.args[6]
    conv_kernel_size = op_invoke_info.args[7]

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

    # TensorCast does not thread cache state through this fused op. Use seq_len=1
    # as the decode heuristic and otherwise model the chunk_gated_delta prefill path.
    if seq_len == 1:
        (
            attn_mma_ops,
            hidden_gp_ops,
            fp32_gp_ops,
        ) = _linear_attention_recurrent_gated_delta_ops(
            batch_size, seq_len, num_v_heads, head_k_dim, head_v_dim
        )
    else:
        (
            attn_mma_ops,
            hidden_gp_ops,
            fp32_gp_ops,
        ) = _linear_attention_chunk_gated_delta_ops(
            batch_size, seq_len, num_v_heads, head_k_dim, head_v_dim
        )

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


@OpInvokeInfo.register_op_properties(torch.ops.tensor_cast.attention.default)
def _(
    op_invoke_info: OpInvokeInfo,
) -> OpInvokeInfo.PerformanceProperties:
    assert len(op_invoke_info.args) == 8
    query = op_invoke_info.args[0]
    key = op_invoke_info.args[1]
    request_total_seq_lens = op_invoke_info.args[6]
    query_lens = op_invoke_info.args[7]
    query_lens, request_total_seq_lens = (
        _normalize_query_lens_and_request_total_seq_lens(
            query, query_lens, request_total_seq_lens
        )
    )
    return _attention_properties_helper(
        op_invoke_info, query, key, request_total_seq_lens, query_lens, query.dtype
    )


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
    query_lens, request_total_seq_lens = (
        _normalize_query_lens_and_request_total_seq_lens(
            query, query_lens, request_total_seq_lens
        )
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

    # FP8 (e4m3fn/e5m2): Only 1 op per element (scale multiplication ONLY, no offset applied)
    # Assume FP8 is not natively supported
    QDQ_OP_FACTOR_MAP = {torch.float8_e4m3fn: 1, torch.float8_e5m2: 1, torch.int8: 2}
    qdq_op_factor = QDQ_OP_FACTOR_MAP.get(key.dtype, 2)

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
    properties.memory_write_bytes += bytes_of_tensor(
        kv_c_normed, dtype=kv_cache.dtype
    ) + bytes_of_tensor(k_rot, dtype=kv_cache.dtype)
    return properties


_PREDICTIVE_DECODING_THRESHOLD = 5


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
    compute_ops = properties.compute_ops.setdefault(
        kv_a_proj_weight.dtype, OpInvokeInfo.ComputeOps()
    )
    compute_ops.mma_ops += total_mma_ops
    compute_ops = properties.compute_ops.setdefault(
        hidden_states.dtype, OpInvokeInfo.ComputeOps()
    )
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
    if kv_a_proj_weight.dtype == torch.float8_e5m2:
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
    total_quant_dequant_ops = (
        quant1_ops
        + dequant1_ops
        + quant2_ops
        + dequant2_ops
        + quant3_ops
        + dequant3_ops
    )
    _accumulate_compute_ops(
        properties,
        hidden_states.dtype,
        gp_ops=total_quant_dequant_ops,
    )

    return properties


def _multihead_latent_attention_properties_helper(
    op_invoke_info: OpInvokeInfo,
    softmax_dtype: torch.dtype,
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

    # 2. Separate Prefill and Decode Sequences
    # A sequence is in "decode" if it's processing only one query token.
    # Otherwise, it's in "prefill".
    num_tokens_per_seq = query_lens
    is_decode = num_tokens_per_seq < _PREDICTIVE_DECODING_THRESHOLD
    is_prefill = ~is_decode

    total_fma_ops = 0
    total_gp_ops = 0
    exclude_input_ids = {1, 6, 7, 8}  # kv_cache, W_UK_T, W_UV, kv_b_proj

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
        prefill_context_sum = torch.sum(
            prefill_num_tokens_per_seq.to(prefill_attn_len.dtype) * prefill_attn_len
        ).item()

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
        decode_context_sum = torch.sum(
            decode_num_tokens_per_seq.to(decode_attn_len.dtype) * decode_attn_len
        ).item()

        # The decode formula is: softmax(q_nope @ W_UK_T @ k_cache) @ v_cache @ W_UV
        # Op 1: `q_nope @ W_UK_T`
        # Shapes: (num_decode_tokens, num_heads, qk_nope_head_dim) @ (num_heads, qk_nope_head_dim, kv_lora_rank)
        decode_op1_ops = (
            num_decode_tokens * num_heads * qk_nope_head_dim * kv_lora_rank * 2
        )

        # Op 2: `(result_op1, q_rope) @ kv_cache`
        decode_op2_ops = (
            decode_context_sum * num_heads * (kv_lora_rank + qk_rope_head_dim) * 2
        )

        # Op 3: Softmax
        decode_op3_ops = decode_context_sum * num_heads * 4

        # Op 4: `Scores @ v_cache`
        decode_op4_ops = decode_context_sum * num_heads * kv_lora_rank * 2

        # Op 5: `(result_op4) @ W_UV`
        # Shapes: (num_decode_tokens, num_heads, kv_lora_rank) @ (num_heads, kv_lora_rank, v_head_dim)
        decode_op5_ops = num_decode_tokens * num_heads * kv_lora_rank * v_head_dim * 2

        total_fma_ops += (
            decode_op1_ops + decode_op2_ops + decode_op4_ops + decode_op5_ops
        )
        total_gp_ops += decode_op3_ops

    properties = op_invoke_info.get_memory_access_properties(
        exclude_input_ids=exclude_input_ids
    )  # exclude kv_cache

    # Estimate memory read from the KV Cache.
    # The size of a cached entry is (kv_lora_rank + qk_rope_head_dim).
    cache_entry_size = bytes_of_elements(kv_cache.size(-1), kv_cache.dtype)
    if sparse_topk is not None:
        decode_request_total_seq_lens = torch.minimum(
            request_total_seq_lens,
            torch.tensor(sparse_topk, device=request_total_seq_lens.device),
        )
        actual_request_total_seq_lens = torch.where(
            is_decode, decode_request_total_seq_lens, request_total_seq_lens
        )
        properties.memory_read_bytes += (
            torch.sum(actual_request_total_seq_lens).item() * cache_entry_size
        )
    else:
        properties.memory_read_bytes += torch.sum(
            request_total_seq_lens * cache_entry_size
        ).item()

    compute_ops = properties.compute_ops.setdefault(q.dtype, OpInvokeInfo.ComputeOps())
    compute_ops.mma_ops = total_fma_ops
    compute_ops = properties.compute_ops.setdefault(
        softmax_dtype, OpInvokeInfo.ComputeOps()
    )
    compute_ops.gp_ops = total_gp_ops

    return properties


@OpInvokeInfo.register_op_properties(
    torch.ops.tensor_cast.multihead_latent_attention.default
)
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
    # Separate prefill and decode sequences
    num_tokens_per_seq = query_lens
    is_decode = num_tokens_per_seq < _PREDICTIVE_DECODING_THRESHOLD
    is_prefill = ~is_decode

    total_quant_dequant_ops = 0

    # Calculate quant/dequant ops for prefill phase
    num_prefill_tokens = torch.sum(num_tokens_per_seq[is_prefill]).item()
    if num_prefill_tokens > 0:
        prefill_request_total_seq_lens = request_total_seq_lens[is_prefill]
        prefill_num_tokens_per_seq = num_tokens_per_seq[is_prefill]
        prefill_context_sum = torch.sum(
            prefill_num_tokens_per_seq.to(prefill_request_total_seq_lens.dtype)
            * prefill_request_total_seq_lens
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
            decode_num_tokens_per_seq.to(decode_request_total_seq_lens.dtype)
            * decode_request_total_seq_lens
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


@OpInvokeInfo.register_op_properties(
    torch.ops.tensor_cast.multihead_latent_attention_quant.default
)
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

    # Get base properties from helper
    properties = _multihead_latent_attention_properties_helper(
        op_invoke_info, softmax_dtype
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
        properties_i = _static_quant_linear_properties_helper(
            op_invoke_info, xi, wi, w_offseti, biasi, is_int4=False
        )
        properties.combine(properties_i, compute_only=True)
    return properties


@OpInvokeInfo.register_op_properties(
    torch.ops.tensor_cast.grouped_matmul_quant_int4.default
)
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
        properties_i = _static_quant_linear_properties_helper(
            op_invoke_info, xi, wi, w_offseti, biasi, is_int4=True
        )
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
        properties_i = _static_quant_linear_properties_helper(
            op_invoke_info, xi, wi, None, biasi, is_int4=False
        )
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
        properties_i = _static_quant_linear_properties_helper(
            op_invoke_info, xi, wi, None, biasi, is_int4=True
        )
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
            props_i = mm_helper(
                op_invoke_info, xi, wi, w_offseti, biasi, is_int4_weight
            )
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


@OpInvokeInfo.register_op_properties(
    torch.ops.tensor_cast.grouped_matmul_swiglu.default
)
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


@OpInvokeInfo.register_op_properties(
    torch.ops.tensor_cast.grouped_matmul_quant_swiglu.default
)
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


@OpInvokeInfo.register_op_properties(
    torch.ops.tensor_cast.grouped_matmul_quant_int4_swiglu.default
)
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


@OpInvokeInfo.register_op_properties(
    torch.ops.tensor_cast.grouped_matmul_fp8_swiglu.default
)
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


@OpInvokeInfo.register_op_properties(
    torch.ops.tensor_cast.grouped_matmul_mxfp4_swiglu.default
)
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
    compute_ops = properties.compute_ops.setdefault(
        input.dtype, OpInvokeInfo.ComputeOps()
    )
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
    compute_ops = properties.compute_ops.setdefault(
        input.dtype, OpInvokeInfo.ComputeOps()
    )
    compute_ops.mma_ops = total_flops
    return properties


def _estimate_static_cost(
    op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile
) -> float:
    perf_properties = op_invoke_info.get_perf_properties()
    for dtype in DeviceProfile.DTYPES:
        if dtype in perf_properties.compute_ops:
            if dtype not in device_profile.mma_ops:
                continue
            compute_ops = perf_properties.compute_ops[dtype]
            if compute_ops.mma_ops > 0:
                return device_profile.static_cost.mma_op_cost_s
    return device_profile.static_cost.gp_op_cost_s


def _estimate_default_without_static_cost(
    op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile
) -> PerformanceModel.Result:
    if is_view_op(op_invoke_info.func) or is_noop_self_copy_op(
        op_invoke_info.func, op_invoke_info.args
    ):
        return PerformanceModel.Result(0.0)
    perf_properties = op_invoke_info.get_perf_properties()
    # By default, we do not consider instruction-level parallelism when counting computation time
    mma_ops_time_s = 0
    gp_ops_time_s = 0
    for dtype in DeviceProfile.DTYPES:
        if dtype in perf_properties.compute_ops:
            compute_ops = perf_properties.compute_ops[dtype]
            if compute_ops.mma_ops > 0:
                if dtype in device_profile.mma_ops:
                    device_mma_ops = (
                        device_profile.mma_ops[dtype]
                        * device_profile.compute_efficiency
                    )
                    mma_ops_time_s += compute_ops.mma_ops / device_mma_ops
                else:
                    logger.warning(
                        "Ignoring mma compute ops of %s for %s since it is not supported on %s",
                        dtype,
                        op_invoke_info,
                        device_profile,
                    )
            if compute_ops.gp_ops > 0:
                if dtype in device_profile.gp_ops:
                    compute_ops = perf_properties.compute_ops[dtype]
                    device_gp_ops = (
                        device_profile.gp_ops[dtype] * device_profile.compute_efficiency
                    )
                    gp_ops_time_s += compute_ops.gp_ops / device_gp_ops
                else:
                    logger.warning(
                        "Ignoring gp compute ops of %s for %s since it is not supported on %s",
                        dtype,
                        op_invoke_info,
                        device_profile,
                    )
    compute_time_s = mma_ops_time_s + gp_ops_time_s
    memory_bandwidth = (
        device_profile.memory_bandwidth_bytes_ps * device_profile.memory_efficiency
    )
    memory_read_time_s = perf_properties.memory_read_bytes / memory_bandwidth
    memory_write_time_s = perf_properties.memory_write_bytes / memory_bandwidth
    memory_readwrite_time_s = perf_properties.memory_readwrite_bytes / memory_bandwidth
    memory_access_time_s = (
        memory_read_time_s + memory_write_time_s + memory_readwrite_time_s
    )
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


def _estimate_default(
    op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile
) -> PerformanceModel.Result:
    result = _estimate_default_without_static_cost(op_invoke_info, device_profile)
    if result.execution_time_s == 0:
        return result
    result.execution_time_s += _estimate_static_cost(op_invoke_info, device_profile)
    return result


register_op_estimator(None, None)(_estimate_default)


@register_op_estimator(torch.ops.tensor_cast.all_reduce.default, None)
@register_op_estimator(torch.ops.tensor_cast.all_gather.default, None)
@register_op_estimator(torch.ops.tensor_cast.reduce_scatter.default, None)
@register_op_estimator(torch.ops.tensor_cast.all_to_all.default, None)
def _estimate_collective_comm(
    op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile
) -> PerformanceModel.Result:
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
    result = PerformanceModel.Result(
        linear_result.execution_time_s, dict(linear_result.statistics)
    )
    result.combine(
        PerformanceModel.Result(
            comm_result.execution_time_s, dict(comm_result.statistics)
        )
    )
    result.statistics = {
        "overlap_model": overlap_label,
        time_key: linear_result.execution_time_s,
        "all_reduce_time_s": comm_result.execution_time_s,
    }
    result.statistics.update(_tag_statistics(linear_result.statistics, stats_prefix))
    result.statistics.update(_tag_statistics(comm_result.statistics, "all_reduce"))
    return result


@register_op_estimator(torch.ops.tensor_cast.matmul_all_reduce.default, None)
def _estimate_matmul_all_reduce(
    op_invoke_info: OpInvokeInfo, device_profile: DeviceProfile
) -> PerformanceModel.Result:
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

    result = PerformanceModel.Result(
        mm_result.execution_time_s, dict(mm_result.statistics)
    )
    result.combine(
        PerformanceModel.Result(
            comm_result.execution_time_s, dict(comm_result.statistics)
        )
    )
    result.statistics = {
        "overlap_model": "max(matmul, all_reduce)",
        "matmul_time_s": mm_result.execution_time_s,
        "all_reduce_time_s": comm_result.execution_time_s,
    }
    result.statistics.update(_tag_statistics(mm_result.statistics, "matmul"))
    result.statistics.update(_tag_statistics(comm_result.statistics, "all_reduce"))
    return result


@register_op_estimator(
    torch.ops.tensor_cast.static_quant_linear_all_reduce.default, None
)
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


@register_op_estimator(
    torch.ops.tensor_cast.static_quant_linear_int4_all_reduce.default, None
)
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


def _estimate_dsa_indexer_breakdown(
    hidden_states: torch.Tensor,
    qa_normed: torch.Tensor,
    indexer_cache: torch.Tensor,
    num_heads: int,
    head_dim: int,
    qk_rope_head_dim: int,
    topk_limit: int,
    request_total_seq_lens: Optional[torch.Tensor] = None,
    fp8_mode: bool = False,
):
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
    max_request_total_seq_len = (
        int(request_total_seq_lens.max().item())
        if request_total_seq_lens is not None
        else None
    )
    active_cache_len = max_request_total_seq_len or seq_len
    qk_index_mma = 2 * batch * seq_len * num_heads * active_cache_len * head_dim

    # Cache traffic uses the logical cache extent when runtime lengths are available;
    # otherwise it falls back to the preallocated cache buffer length.
    cache_len = max_request_total_seq_len or indexer_cache.size(1)
    cache_rw_bytes = (
        batch * cache_len * indexer_cache.size(-1) * indexer_cache.element_size()
    )
    scale_cache_rw_bytes = 0
    if fp8_mode:
        # The fp8 path writes both the fp8 key cache and the accompanying scale cache.
        # The scale cache is a small fp32 side buffer, approximated as one 32-bit scale
        # per 128-wide block, sized by the same logical cache length.
        scale_cache_rw_bytes = batch * cache_len * ((head_dim + 127) // 128) * 4

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
        "cache_rw_bytes": cache_rw_bytes,
        "scale_cache_rw_bytes": scale_cache_rw_bytes,
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
    fp8_mode = indexer_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

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
        mma_ops=(
            breakdown["q_proj_mma"]
            + breakdown["k_proj_mma"]
            + breakdown["weights_proj_mma"]
        ),
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
        gp_ops=(
            breakdown["head_relu_gp"]
            + breakdown["head_q_scale_mul_gp"]
            + breakdown["head_k_scale_mul_gp"]
        ),
    )

    # The cache itself is dense, and the reference kernel also maintains a separate
    # scale cache in fp8 mode.  The local wrapper only exposes one cache tensor, so we
    # count the scale-cache bytes as read/write traffic as well to keep the roofline
    # bound conservative.
    properties.memory_readwrite_bytes += (
        breakdown["cache_rw_bytes"] + breakdown["scale_cache_rw_bytes"]
    )
    return properties


_load_custom_op()
