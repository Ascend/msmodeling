"""ProfilingDataSource: CSV-backed data source with op_mapping + FRACTAL_NZ.

Maintenance Guide
=================
See docs/perf_database/tutorial/OP_PLUGIN_MAPPING_TUTORIAL.md §14-§15 for
the full SOP and checklists.  Below is a quick-reference map of the extension
points inside this file.

Adding a new op (most common — YAML-only):
    Edit op_mapping.yaml → add an entry under operator_mappings.
    No Python changes needed if the shape matches an existing rule in
    _inputs_match() (identity, batch_strip, transpose, padding, flatten).

Adding custom shape normalization (when TC ↔ CSV shapes differ):
    1. Add a kernel frozenset constant (search for _SWIGLU_KERNELS to find the area).
    2. Write a module-level normalizer function
       (search for _normalize_rope_inputs, _normalize_reshape_and_cache_inputs).
    3. Add a branch in _inputs_match() that dispatches to the new normalizer
       when kernel_type is in the new frozenset
       (search for "if kernel_type in _SWIGLU_KERNELS" to find existing branches).

Adding a composite decomposer (1 TC op → N NPU kernels, runtime-dependent):
    1. Write a decompose function returning List[SubKernelSpec]
       (search for _decompose_mla_common, _decompose_mlapo_common).
    2. Register it in COMPOSITE_DECOMPOSERS dict
       (search for "COMPOSITE_DECOMPOSERS" to find the dict definition).

Adding a new query_mode (when compute/attention/elementwise/moe don't fit):
    1. Implement _lookup_<mode>() method in ProfilingDataSource.
    2. Add a branch in lookup() dispatch chain
       (search for "query_mode" to find the dispatch logic).

Adding dtype support:
    Update DTYPE_MAP, _DTYPE_COMPAT, _DTYPE_RELAXED_KERNELS
    (search for each name to find its definition).

CANN version upgrade:
    See OP_PLUGIN_MAPPING_TUTORIAL.md §15 for the full checklist.
    Key code touchpoints: kernel frozensets, _DTYPE_COMPAT, decomposer
    functions, and the CSV column expectations in _load_csv / _latency_col.
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import yaml

from ... import config
from ...device import DeviceProfile
from .data_source import (
    DataSourcePerformanceModel,
    QueryResult,
    QuerySource,
    ShapeMatchInfo,
    SubKernelShapeInfo,
)


if TYPE_CHECKING:
    from ...device import CommGrid
    from ...model_config import ParallelConfig
    from ..op_invoke_info import OpInvokeInfo

logger = logging.getLogger(__name__)

_LATENCY_COLS = (
    "Average Duration(us)",
    "Profiling Average Duration(us)",
    "Duration(us)",
)
_PROFILING_DURATION_COL = "Profiling Average Duration(us)"
_PROFILING_AICORE_TIME_COL = "Profiling Average aicore_time(us)"
_PROFILING_AIV_TIME_COL = "Profiling Average aiv_time(us)"
_EFFECTIVE_LATENCY_COL = "__effective_latency_us"

# vLLM-Ascend 0.18 / CANN 8.5 ``mla_preprocess_0_mix_aic`` contract.
# The opaque decode kernel currently accepts at most 1024 query tokens and
# stores both quantized weights in 32-element FRACTAL_NZ blocks.
_MLA_PREPROCESS_MAX_DECODE_TOKENS = 1024
_MLA_PREPROCESS_FRACTAL_NZ_BLOCK = 32
_MLA_PREPROCESS_INPUT_COUNT = 22
_MLA_PREPROCESS_OUTPUT_COUNT = 5

# torch dtype -> Profiling dtype string
DTYPE_MAP = {
    torch.bfloat16: "DT_BF16",
    torch.float16: "DT_BF16",  # FP16 treated as BF16 on Ascend
    torch.int8: "INT8",
    torch.int32: "INT32",
    torch.int64: "INT64",
    torch.float32: "FLOAT",
    torch.bool: "BOOL",
}


def fractal_nz_to_nd(nz_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Restore FRACTAL_NZ tiled shape to ND shape.
    [..., H, W, block_h, block_w] -> [..., H*block_w, W*block_h]

    Examples:
    - BF16: [K/16, N/16, 16, 16] -> (K, N)
    - INT8: [N/32, K/16, 16, 32] -> (K, N) after H*block_w, W*block_h
    - Batched: [E, N/32, K/16, 16, 32] -> (E, K, N)
    """
    if len(nz_shape) < 4:
        # Some grouped-matmul rows are already restored to a 3D expert-weight
        # shape but still carry FRACTAL_NZ in the exported format column.
        return nz_shape
    *batch, H, W, block_h, block_w = nz_shape
    return (*batch, H * block_w, W * block_h)


def _normalize_func_name(func) -> str:
    """Convert torch op to string matching op_mapping.yaml keys.
    e.g. torch.ops.aten.mm.default -> 'aten.mm.default'
         torch.ops.tensor_cast.attention.default -> 'tensor_cast.attention.default'
    """
    s = str(func)
    return s.removeprefix("torch.ops.")


def _parse_shape_str(s: str) -> List[Tuple[int, ...]]:
    """Parse CSV shape string -> list of tuples.
    e.g. '"136,5120;320,48,16,16"' -> [(136,5120), (320,48,16,16)]
    """
    s = s.strip().strip('"')
    shapes = []
    for part in s.split(";"):
        part = part.strip()
        if part:
            if part == "()":
                shapes.append(())
                continue
            shapes.append(tuple(int(x) for x in part.split(",")))
    return shapes


def _parse_str_list(s: str) -> List[str]:
    """Parse 'A;B;C' -> ['A', 'B', 'C']"""
    s = s.strip().strip('"')
    return [x.strip() for x in s.split(";") if x.strip()]


def _tensor_int_values(value: Any) -> Optional[List[int]]:
    if not isinstance(value, torch.Tensor):
        return None
    try:
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    except (RuntimeError, TypeError, ValueError):
        return None


def _cumulative_even_split(total_tokens: int, batch_size: int) -> Optional[List[int]]:
    if total_tokens <= 0 or batch_size <= 0 or total_tokens < batch_size:
        return None
    base, extra = divmod(total_tokens, batch_size)
    result: List[int] = []
    total = 0
    for index in range(batch_size):
        total += base + (1 if index < extra else 0)
        result.append(total)
    return result


def _parse_runtime_int_list_cell(value: Any) -> Optional[List[int]]:
    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip().replace(";", ",")
    if not cleaned:
        return None
    try:
        return [int(item.strip()) for item in cleaned.split(",") if item.strip()]
    except ValueError:
        return None


def _optional_runtime_str(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _build_mla_preprocess_expected_shapes(
    *,
    num_tokens: int,
    hidden_size: int,
    local_num_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    block_size: int,
) -> Optional[Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]]:
    """Build the versioned CANN 8.5 input/output layout for opaque MLAPO."""

    nz_block = _MLA_PREPROCESS_FRACTAL_NZ_BLOCK
    if hidden_size % nz_block != 0 or q_lora_rank % nz_block != 0:
        return None

    fused_qkv_dim = q_lora_rank + kv_lora_rank + qk_rope_head_dim
    q_output_dim = local_num_heads * (qk_nope_head_dim + qk_rope_head_dim)
    cache_blocks = math.ceil(num_tokens / block_size)
    input_shapes = [
        (num_tokens, hidden_size),
        (1, hidden_size // nz_block, fused_qkv_dim, nz_block),
        (fused_qkv_dim,),
        (q_lora_rank,),
        (q_lora_rank,),
        (1, q_lora_rank // nz_block, q_output_dim, nz_block),
        (q_output_dim,),
        (kv_lora_rank,),
        (num_tokens, qk_rope_head_dim),
        (num_tokens, qk_rope_head_dim),
        (local_num_heads, qk_nope_head_dim, kv_lora_rank),
        (cache_blocks, block_size, 1, kv_lora_rank),
        (cache_blocks, block_size, 1, qk_rope_head_dim),
        (num_tokens,),
        (1,),
        (1,),
        (fused_qkv_dim,),
        (1,),
        (1,),
        (q_output_dim,),
        (1,),
        (1,),
    ]
    output_shapes = [
        (num_tokens, local_num_heads, kv_lora_rank),
        input_shapes[11],
        (num_tokens, local_num_heads, qk_rope_head_dim),
        input_shapes[12],
        (num_tokens, q_lora_rank),
    ]
    return input_shapes, output_shapes


def _uses_glm5_sampling_bmm_mul(
    tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
) -> bool:
    """Return whether CANN lowers the known GLM5 K=1 sampling BMM to Mul."""

    if len(tc_inputs) < 2:
        return False
    lhs_shape, _ = tc_inputs[0]
    rhs_shape, _ = tc_inputs[1]
    return lhs_shape == (1, 32, 1) and len(rhs_shape) == 3 and rhs_shape[:2] == (1, 1) and rhs_shape[2] > 0


def _rank_zero_sparse_runtime_vectors(
    *,
    seq_lens: torch.Tensor,
    global_query_tokens: int,
    local_query_tokens: int,
    query_lens: Optional[torch.Tensor] = None,
    require_query_lens_for_multiple_requests: bool = False,
) -> Optional[Tuple[List[int], List[int]]]:
    """Mirror vLLM-Ascend DSA context-parallel metadata for TP rank 0."""

    seq_values = _tensor_int_values(seq_lens)
    if seq_values is None or not seq_values:
        return None
    query_values = _tensor_int_values(query_lens) if isinstance(query_lens, torch.Tensor) else None
    query_values_are_valid = (
        query_values is not None
        and len(query_values) == len(seq_values)
        and all(value >= 0 for value in query_values)
        and sum(query_values) == global_query_tokens
    )
    if not query_values_are_valid:
        if require_query_lens_for_multiple_requests and len(seq_values) > 1:
            return None
        if sum(seq_values) == global_query_tokens:
            # This equality is the full-prefill case: the query covers every
            # token in each sequence. Decode includes historical KV context,
            # so its sequence sum is larger and takes the even-split fallback.
            query_values = list(seq_values)
        else:
            base, extra = divmod(global_query_tokens, len(seq_values))
            query_values = [base + (1 if index < extra else 0) for index in range(len(seq_values))]

    local_cumulative: List[int] = []
    local_kv_lengths: List[int] = []
    global_start = 0
    local_total = 0
    for query_length, seq_length in zip(query_values, seq_values):
        global_end = global_start + query_length
        local_end = min(global_end, local_query_tokens)
        local_count = max(0, local_end - global_start)
        local_total += local_count
        local_cumulative.append(local_total)
        # Rank 0 sees the sequence context that precedes this query plus its
        # local SP query slice.  For a first 4096-token chunk at TP16 this is
        # 256, while a later chunk keeps the preceding context and adds 256.
        local_kv_lengths.append(seq_length - (global_end - local_end) if local_count > 0 else 0)
        global_start = global_end

    if not local_cumulative or local_cumulative[-1] != local_query_tokens:
        return None
    return local_cumulative, local_kv_lengths


def _sparse_runtime_attention_params(
    *,
    work_tokens: int,
    work_heads: int,
    head_dim: int,
    seq_lens: torch.Tensor,
    avg_seq_len: Optional[int],
    topk: Optional[int],
    block_size: Optional[int],
    include_sparse_block_size: bool,
    query_lengths_values: Optional[List[int]] = None,
    kv_lengths_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    batch_size = int(seq_lens.numel())
    kv_lengths = kv_lengths_values if kv_lengths_values is not None else _tensor_int_values(seq_lens)
    query_lengths = (
        query_lengths_values if query_lengths_values is not None else _cumulative_even_split(work_tokens, batch_size)
    )
    runtime_avg_seq_len = (
        sum(kv_lengths) // len(kv_lengths) if kv_lengths is not None and kv_lengths_values is not None else avg_seq_len
    )
    valid_blocks = None
    if kv_lengths is not None and isinstance(block_size, int) and block_size > 0:
        valid_blocks = [math.ceil(length / block_size) for length in kv_lengths]
    params: Dict[str, Any] = {
        "q_shape_3d": (work_tokens, work_heads, head_dim),
        "avg_seq_len": runtime_avg_seq_len,
        "sparse_mode": 3,
        "num_kv_heads": 1,
        "num_heads": work_heads,
        "input_layout": "TND",
        "cache_layout": "PA_BSND",
        "kv_cache_mode": "paged",
        "topk": topk,
        "block_size": block_size,
        "batch_size": batch_size,
        "actual_seq_lengths_values": query_lengths,
        "actual_seq_lengths_kv_values": kv_lengths,
        "block_table_valid_blocks": valid_blocks,
    }
    required_fields = [
        "avg_seq_len",
        "sparse_mode",
        "num_kv_heads",
        "num_heads",
        "input_layout",
        "cache_layout",
        "kv_cache_mode",
        "topk",
        "block_size",
        "actual_seq_lengths_values",
        "actual_seq_lengths_kv_values",
        "block_table_valid_blocks",
    ]
    if include_sparse_block_size:
        params["sparse_block_size"] = 1
        params["sparse_indices_pattern"] = "uniform"
        if kv_lengths is not None and kv_lengths and isinstance(topk, int) and topk > 0:
            params["sparse_indices_valid_count"] = min(topk, max(kv_lengths))
        else:
            params["sparse_indices_valid_count"] = None
        required_fields.extend(("sparse_block_size", "sparse_indices_pattern"))
        if params["sparse_indices_valid_count"] is not None:
            required_fields.append("sparse_indices_valid_count")
    params["required_context_fields"] = tuple(required_fields)
    return params


def _parse_fia_q_shape(input_shapes_str: str) -> Optional[Tuple[int, ...]]:
    """Parse slot 0 (Q shape) from FIA CSV Input Shapes string."""
    if not input_shapes_str or not input_shapes_str.strip():
        return None
    parts = input_shapes_str.split(";")
    if not parts or not parts[0].strip():
        return None
    try:
        return tuple(int(x) for x in parts[0].strip().split(","))
    except ValueError:
        return None


def _normalize_fia_q_shape(q_shape: Tuple[int, ...], head_dim: int = 0) -> Optional[Tuple[int, ...]]:
    """Normalize FIA Q shape to 3D (T, N, D).

    3D (T, N, D) → identity; 4D (B, N, 1, D) → squeeze; 2D (T, H) → reshape.
    """
    ndim = len(q_shape)
    if ndim == 3:
        return q_shape
    if ndim == 4 and q_shape[2] == 1:
        return (q_shape[0], q_shape[1], q_shape[3])
    if ndim == 2 and head_dim > 0 and q_shape[1] % head_dim == 0:
        return (q_shape[0], q_shape[1] // head_dim, head_dim)
    return None


def _infer_sparse_mode(query_lens) -> int:
    """Infer FIA sparse_mode from query_lens.

    TC attention op does not pass sparse_mode explicitly.
    Both prefill and decode use sparse_mode=3 (causal) in vLLM profiling data:
    - Prefill: causal mask (right_down_causal)
    - Decode (paged_runtime): also sparse_mode=3 in profiling CSVs

    Note: MLA decode (mla_paged_runtime) uses sparse_mode=0, but MLA goes
    through the decomposer path which hardcodes sparse_mode directly,
    so this function is only called for non-MLA attention.
    """
    return 3  # right_down_causal — matches profiling CSV for both prefill and decode


# while TC's aten.mm receives (K,N) after F.linear transpose.
# FRACTAL_NZ weights restore to (K,N) directly — no transpose needed.
_MATMUL_KERNELS = frozenset(
    {
        "MatMulV2",
        "MatMulV3",
        "MatMulCommon",
        "MatMul",
        "QuantBatchMatmulV3",
        "BatchMatMulV2",
        "TransposeBatchMatMul",
        "GroupedMatmul",
        "GroupedMatmulSwigluQuant",
    }
)

_RELAXED_DTYPE_MATMUL_KERNELS = frozenset(
    {
        "MatMulV2",
    }
)

_RELAXED_DTYPE_PAD_KERNELS = frozenset(
    {
        "PadV3",
        "PadV3AiCore",
    }
)

# SwiGlu kernel types: TC dispatches 2 inputs (gate, up) as separate tensors,
# but profiling CSVs store 1 concatenated input along last dim.
_SWIGLU_KERNELS = frozenset({"SwiGlu"})

# RoPE kernel types: TC dispatches (B,H,S,D) layout with [Q, K, cos, sin],
# but profiling CSVs store (B,S,H,D) layout with [K, Q, cos, sin] and
# cos/sin have an extra head dim (1).
_ROPE_KERNELS = frozenset({"ApplyRotaryPosEmb", "_triton_rope", "split_qkv_rmsnorm_rope_kernel"})

# ReshapeAndCache kernel types: TC dispatches (key, value, kv_cache, slot_mapping)
# with key/value as 2D (N, D) and a single merged kv_cache (2, blocks, block_size, heads, D).
# Profiling CSVs store (key, value, cache_k, cache_v, slot_mapping) with key/value
# as 3D (N, 1, D) and separate cache_k/cache_v tensors.
_RESHAPE_AND_CACHE_KERNELS = frozenset({"ReshapeAndCacheNdKernel", "reshape_and_cache_200000000"})

# Dtype groups that are considered equivalent for matching purposes.
# NPU _triton_rope profiling records K as FLOAT (FP32) while TC dispatches
# BF16 — the kernel internally up-casts, but performance is the same.
_DTYPE_COMPAT = {"DT_BF16": "FLOAT_GROUP", "FLOAT": "FLOAT_GROUP"}

# Kernel types that allow relaxed dtype matching via _DTYPE_COMPAT.
# For non-quant matmul kernels, some model code paths upcast inputs/weights to
# FP32 in eager code for numerical stability, while Ascend profiling records
# the realized kernel as BF16. Allow FLOAT <-> DT_BF16 compatibility so shape
# matching can still reuse the measured kernel entry. Quant matmul kernels keep
# strict dtype matching because their dtype semantics differ from plain matmul.
_DTYPE_RELAXED_KERNELS = _ROPE_KERNELS | _RELAXED_DTYPE_MATMUL_KERNELS | _RELAXED_DTYPE_PAD_KERNELS | {"MoeGatingTopK"}

# Kernel types where TC may produce 3D (B, M, D) shapes that should
# match CSV's 2D (B*M, D) shapes by flattening the leading two dims.
# This happens when TC keeps an explicit batch dimension that profiling
# absorbs into the token/sequence dimension.
_FLATTEN_BATCH_KERNELS = frozenset(
    {
        "AscendQuantV2",
        "DynamicQuant",
        "RmsNorm",
        "AddRmsNormBias",
        "AddRmsNorm",
        "DispatchFFNCombine",
    }
)

# Kernel types where TC produces 3D (T, H, D) per-head shapes that should
# match CSV's 2D (T, H*D) shapes by merging the last two dims.
# This is specific to MLA quantize where NPU reshapes to hidden_dim before quantize.
_MERGE_LAST_DIMS_KERNELS = frozenset({"AscendQuantV2", "DynamicQuant"})

# Common NPU tile alignment sizes (Da Vinci Cube unit)
# BF16: 16x16, INT8: 16x32
# Minimum raw (unpadded) dim value for each block size to avoid false-positive
# matches on small dimensions like head counts (4, 8, 16, ...).
# Block size 8 is only valid for sequence/token dims (≥ 64 tokens).
_BLOCK_SIZES = (16, 32, 64)
_BLOCK_SIZE_MIN_DIM: Dict[int, int] = {8: 64}

# FIA avg_seq_len tolerance: avg_seq_len is a workload descriptor (not a shape
# dim). FIA latency is continuous in seq_len — a 16-token gap at context ≈ 4000
# means < 0.4% KV cache difference.  This tolerance does NOT match fundamentally
# different workloads (e.g. decode avg=1 vs prefill avg=4097).
_AVG_SEQ_LEN_TOLERANCE = 16

# Byte sizes for profiling dtype strings (for elementwise byte-ratio scaling)
_DTYPE_BYTE_SIZES = {
    "DT_BF16": 2,
    "DT_FLOAT16": 2,
    "FLOAT": 4,
    "DT_FLOAT": 4,
    "INT8": 1,
    "DT_INT8": 1,
    "INT16": 2,
    "INT32": 4,
    "DT_INT32": 4,
    "INT64": 8,
    "DT_INT64": 8,
}

# Query modes handled by dedicated _lookup_<mode>() methods.
# Tests import this to avoid duplicating the dispatch contract.
SUPPORTED_QUERY_MODES: frozenset[str] = frozenset({"attention_special", "elementwise", "moe_fused", "mtp_projection"})


def _dtype_byte_size(dtype_str: str) -> int:
    """Return byte size for a profiling dtype string. Returns 0 for unknown."""
    return _DTYPE_BYTE_SIZES.get(dtype_str, 0)


def _normalize_rope_inputs(
    tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
) -> List[Tuple[Tuple[int, ...], torch.dtype]]:
    """Normalize RoPE inputs from TC layout to profiling CSV layout.

    Full (4 inputs):
      TC:  [Q(B,Hq,S,D), K(B,Hk,S,D), cos(1,S,D), sin(1,S,D)]
      CSV: [K(B,S,Hk,D), Q(B,S,Hq,D), cos(B,S,1,D), sin(B,S,1,D)]

    Truncated (2 inputs, tc_input_count=2):
      TC:  [Q(B,Hq,S,D), K(B,Hk,S,D)]
      CSV: [K(B,S,Hk,D), Q(B,S,Hq,D)]

    Transformations:
    1. Swap Q and K (TC: [Q,K,...] → CSV: [K,Q,...])
    2. Transpose H,S dims in Q and K: (B,H,S,D) → (B,S,H,D)
    3. (Full only) Insert head dim=1 for cos/sin: (1,S,D) → (1,S,1,D)
    """
    q_shape, q_dtype = tc_inputs[0]
    k_shape, k_dtype = tc_inputs[1]

    # Transpose Q and K: (B,H,S,D) → (B,S,H,D)
    if len(q_shape) == 4:
        q_shape = (q_shape[0], q_shape[2], q_shape[1], q_shape[3])
    if len(k_shape) == 4:
        k_shape = (k_shape[0], k_shape[2], k_shape[1], k_shape[3])

    # Reorder: [Q, K] → [K, Q]
    result = [
        (k_shape, k_dtype),
        (q_shape, q_dtype),
    ]

    # Process cos/sin if present (full 4-input case)
    if len(tc_inputs) >= 4:
        cos_shape, cos_dtype = tc_inputs[2]
        sin_shape, sin_dtype = tc_inputs[3]
        if len(cos_shape) == 3:
            cos_shape = (cos_shape[0], cos_shape[1], 1, cos_shape[2])
        if len(sin_shape) == 3:
            sin_shape = (sin_shape[0], sin_shape[1], 1, sin_shape[2])
        result.append((cos_shape, cos_dtype))
        result.append((sin_shape, sin_dtype))

    return result


def _normalize_reshape_and_cache_inputs(
    tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
) -> Optional[List[Tuple[Tuple[int, ...], torch.dtype]]]:
    """Normalize reshape_and_cache inputs from TC layout to profiling CSV layout.

    TC dispatches 4 inputs:
      [key(N, D), value(N, D), kv_cache(2, blocks, block_size, heads, D), slot_mapping(N,)]

    Profiling CSV has 5 inputs:
      [key(N, 1, D), value(N, 1, D), cache_k(blocks, block_size, heads, D),
       cache_v(blocks, block_size, heads, D), slot_mapping(N,)]

    Transformations:
    1. key/value: insert dim=1 at position 1: (N, D) → (N, 1, D)
    2. kv_cache: split merged (2, blocks, block_size, heads, D) into
       cache_k and cache_v, each (blocks, block_size, heads, D)
    3. slot_mapping: keep as-is, but move to position 4 (after cache_v)

    Returns None if inputs don't match the expected TC layout.
    """
    if len(tc_inputs) != 4:
        return None

    key_shape, key_dtype = tc_inputs[0]
    value_shape, value_dtype = tc_inputs[1]
    kv_cache_shape, kv_cache_dtype = tc_inputs[2]
    slot_mapping_shape, slot_mapping_dtype = tc_inputs[3]

    # Validate: key/value should be 2D (N, D)
    if len(key_shape) != 2 or len(value_shape) != 2:
        return None

    # Validate: kv_cache should have leading dim=2 (merged k+v cache)
    if len(kv_cache_shape) < 2 or kv_cache_shape[0] != 2:
        return None

    # Transform key/value: (N, D) → (N, 1, D)
    key_csv = (key_shape[0], 1, key_shape[1])
    value_csv = (value_shape[0], 1, value_shape[1])

    # Split kv_cache: (2, blocks, block_size, heads, D) → (blocks, block_size, heads, D)
    cache_single_shape = kv_cache_shape[1:]

    return [
        (key_csv, key_dtype),
        (value_csv, value_dtype),
        (cache_single_shape, kv_cache_dtype),
        (cache_single_shape, kv_cache_dtype),
        (slot_mapping_shape, slot_mapping_dtype),
    ]


def _strip_batch_dim(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Strip leading batch dim=1 from TC shapes.
    TC keeps explicit batch: (1, seq, dim). Profiling flattens: (seq, dim).
    Only strip if leading dim is exactly 1.
    """
    if len(shape) > 1 and shape[0] == 1:
        return shape[1:]
    return shape


def _is_block_padded(tc_dim: int, csv_dim: int) -> bool:
    """Check if tc_dim is a block-padded version of csv_dim.
    TC pads sequence dims to NPU tile alignment; profiling stores unpadded shapes.

    _BLOCK_SIZES are checked unconditionally. _BLOCK_SIZE_MIN_DIM entries are
    only checked when csv_dim meets the minimum threshold (prevents false
    positives on small dims like head counts).
    """
    if tc_dim <= csv_dim:
        return False
    if any(tc_dim == ((csv_dim + bs - 1) // bs) * bs for bs in _BLOCK_SIZES):
        return True
    return any(
        csv_dim >= min_dim and tc_dim == ((csv_dim + bs - 1) // bs) * bs for bs, min_dim in _BLOCK_SIZE_MIN_DIM.items()
    )


def get_topology_tier(comm_grid: "CommGrid", group: List[int]) -> int:
    """Determine topology tier index for a communication group.

    Finds the outermost grid dimension where ranks differ, then returns the
    most specific (fastest) topology that covers that span.

    Mirrors CommAnalyticModel._get_topology_idx_for_group logic, but operates
    directly on CommGrid to avoid importing the model layer.

    Args:
        comm_grid: CommGrid with .grid (torch.Tensor) and .topologies (dict).
        group: list of rank IDs in the communication group.

    Returns:
        topology tier index (key into comm_grid.topologies).
    """

    def _rank_to_coord(rank: int) -> List[int]:
        coord = []
        temp = rank
        for dim_size in reversed(comm_grid.grid.shape):
            coord.insert(0, temp % dim_size)
            temp //= dim_size
        return coord

    coords = [_rank_to_coord(r) for r in group]

    diff_dim = -1
    for dim_idx in range(comm_grid.grid.dim()):
        first = coords[0][dim_idx]
        if any(c[dim_idx] != first for c in coords[1:]):
            diff_dim = dim_idx
            break

    if diff_dim == -1:
        # All ranks identical (shouldn't happen for group > 1); use fastest tier.
        return max(comm_grid.topologies.keys())

    for start_dim in sorted(comm_grid.topologies.keys(), reverse=True):
        if start_dim <= diff_dim:
            return start_dim

    raise ValueError(f"No topology found for group spanning grid dimension {diff_dim}")


# Query modes handled by dedicated _lookup_<mode>() methods.
# Tests import this to avoid duplicating the dispatch contract.


def _project_tp_sharded_linear_dimension(
    tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
    tp_size: Optional[int],
    *,
    shard_axis: int,
    min_dim: int = 1,
) -> Optional[List[Tuple[Tuple[int, ...], torch.dtype]]]:
    """Project a logical linear shape to the per-rank TP shape in profiling."""
    if tp_size is None or tp_size <= 1 or len(tc_inputs) < 2:
        return None
    activation_shape, activation_dtype = tc_inputs[0]
    weight_shape, weight_dtype = tc_inputs[1]
    if len(activation_shape) != 2 or len(weight_shape) != 2:
        return None
    input_dim = activation_shape[-1]
    if input_dim != weight_shape[0]:
        return None
    if shard_axis == 0:
        if input_dim < min_dim or input_dim % tp_size != 0:
            return None
        sharded_input_dim = input_dim // tp_size
        projected = [
            (activation_shape[:-1] + (sharded_input_dim,), activation_dtype),
            ((sharded_input_dim, weight_shape[1]), weight_dtype),
        ]
    elif shard_axis == 1:
        output_dim = weight_shape[1]
        if output_dim < min_dim or output_dim % tp_size != 0:
            return None
        projected = [
            (activation_shape, activation_dtype),
            ((weight_shape[0], output_dim // tp_size), weight_dtype),
        ]
    else:
        return None
    projected.extend(tc_inputs[2:])
    return projected


def _project_tp_sharded_linear_inputs(
    tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
    tp_size: Optional[int],
    min_input_dim: int = 1,
) -> Optional[List[Tuple[Tuple[int, ...], torch.dtype]]]:
    return _project_tp_sharded_linear_dimension(tc_inputs, tp_size, shard_axis=0, min_dim=min_input_dim)


def _project_tp_sharded_output_linear_inputs(
    tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
    tp_size: Optional[int],
    min_output_dim: int = 1,
) -> Optional[List[Tuple[Tuple[int, ...], torch.dtype]]]:
    return _project_tp_sharded_linear_dimension(tc_inputs, tp_size, shard_axis=1, min_dim=min_output_dim)


# ---- MLA / MLAPO composite decomposition ----


@dataclass
class SubKernelSpec:
    """Specification for a sub-kernel in composite decomposition."""

    kernel_type: str
    input_shapes: List[Tuple[int, ...]]
    dtype: str  # Profiling dtype string, e.g. "DT_BF16"
    input_dtypes: Optional[List[str]] = None
    query_mode: str = "compute"  # "compute" | "attention" | cache-write modes
    attention_params: Optional[Dict[str, Any]] = field(default=None)
    cache_params: Optional[Dict[str, Any]] = field(default=None)
    runtime_params: Optional[Dict[str, Any]] = field(default=None)
    tc_input_count: Optional[int] = None
    alternate_kernel_types: Optional[List[str]] = None
    # Marks a sub-kernel as the dominant attention kernel (FIA/SFA). On a CSV
    # miss the composite lookup returns None to force a complete analytic
    # fallback instead of a PARTIAL result that silently drops attention
    # latency. This is independent of query_mode: SFA matches in compute mode
    # (its CSV has no avg_seq_len column) but is still an attention kernel.
    is_attention: bool = False


def _is_decode_mla(args: tuple) -> bool:
    """Determine if MLA op is in decode mode.

    query_lens (args[5]) is None or all 1s → decode.
    """
    query_lens = args[5]
    if query_lens is None:
        return True
    if isinstance(query_lens, torch.Tensor):
        try:
            return query_lens.max().item() <= 1
        except Exception:
            return True
    return True


def _resolve_batch_phase(
    op_invoke_info: "OpInvokeInfo",
    expected_requests: Optional[int] = None,
) -> Optional[str]:
    """Resolve an explicit whole-batch prefill/decode phase when available.

    ``is_decode_values`` originates from ``RequestInfo.is_decode`` and remains
    reliable for chunked prefill, where query length is smaller than the total
    sequence length. The legacy ``phase`` kwarg takes precedence when both are
    present; contradictory homogeneous values are logged. Mixed batches return
    ``"mixed"`` and malformed metadata returns ``"invalid"`` so callers can
    fail closed instead of reintroducing shape-based phase misclassification.
    """

    phase = op_invoke_info.kwargs.get("phase")
    is_decode_values = op_invoke_info.kwargs.get("is_decode_values")
    if phase in ("prefill", "decode"):
        request_phase = None
        if (
            isinstance(is_decode_values, (list, tuple))
            and is_decode_values
            and (expected_requests is None or len(is_decode_values) == expected_requests)
            and all(isinstance(value, bool) for value in is_decode_values)
        ):
            if all(is_decode_values):
                request_phase = "decode"
            elif not any(is_decode_values):
                request_phase = "prefill"
        if request_phase is not None and request_phase != phase:
            logger.warning(
                "Legacy phase=%s contradicts is_decode_values=%s for %s; legacy phase takes precedence",
                phase,
                is_decode_values,
                op_invoke_info.func,
            )
        return phase

    if is_decode_values is None:
        return None
    if not isinstance(is_decode_values, (list, tuple)) or not is_decode_values:
        logger.warning(
            "Invalid is_decode_values=%r for %s; refusing shape inference",
            is_decode_values,
            op_invoke_info.func,
        )
        return "invalid"
    if expected_requests is not None and len(is_decode_values) != expected_requests:
        logger.warning(
            "is_decode_values length %d does not match %d requests for %s; refusing shape inference",
            len(is_decode_values),
            expected_requests,
            op_invoke_info.func,
        )
        return "invalid"
    if not all(isinstance(value, bool) for value in is_decode_values):
        logger.warning(
            "Non-boolean is_decode_values=%r for %s; refusing shape inference",
            is_decode_values,
            op_invoke_info.func,
        )
        return "invalid"
    if all(is_decode_values):
        return "decode"
    if not any(is_decode_values):
        return "prefill"
    return "mixed"


def _decompose_mla_common(
    op_invoke_info: "OpInvokeInfo",
    mapping: dict,
    first_kernel_type: str,
    alternate_kernel_types: Optional[List[str]] = None,
    attention_kernel_type: str = "FusedInferAttentionScore",
) -> Optional[List[SubKernelSpec]]:
    """Shared MLA decomposition for BF16 and quantized variants.

    Decode: first_kernel_type(q@W_UK_T) + attention_kernel_type + TransposeBatchMatMul(out@W_UV)
    Prefill: MatMulV2(kv_c@kv_b_proj) + attention_kernel_type

    Args:
        first_kernel_type: "BatchMatMulV2" for BF16, "QuantBatchMatmulV3" for quant.
        alternate_kernel_types: Optional fallback kernel types for the
            first decode matmul sub-kernel.
        attention_kernel_type: Kernel type for the attention sub-kernel.
            "FusedInferAttentionScore" for dense MLA (default);
            "SparseFlashAttention" for sparse MLA (DeepSeek-V3.2 / GLM-5.1).
    """
    args = op_invoke_info.args
    if len(args) < 10:
        return None
    decomposer_options = mapping.get("decomposer_options", {})
    dsa_cp_layout = decomposer_options.get("dsa_cp_layout", {})
    sp_heads_already_global = bool(dsa_cp_layout.get("attention_heads_already_global"))
    q = args[0]  # (num_tokens, num_heads, qk_head_dim)
    seq_lens = args[4]  # (batch_size,)
    dtype_str = DTYPE_MAP.get(q.dtype)
    if dtype_str is None:
        return None

    if not isinstance(seq_lens, torch.Tensor):
        return None
    batch_size = seq_lens.shape[0]
    num_heads = q.shape[1]
    kv_cache = args[1]  # (total_blocks, block_size, kv_lora_rank + qk_rope_head_dim)
    head_dim = kv_cache.shape[-1]
    num_tokens = q.shape[0]
    is_sparse_attention = attention_kernel_type == "SparseFlashAttention"

    try:
        avg_seq_len = int(seq_lens.float().mean().item())
    except (RuntimeError, ValueError):
        avg_seq_len = 0

    sparse_phase = _resolve_batch_phase(op_invoke_info, batch_size)
    if sparse_phase == "mixed":
        logger.debug(
            "MLA op %s has mixed prefill/decode requests; falling back to analytic model (is_decode_values=%s)",
            op_invoke_info.func,
            op_invoke_info.kwargs.get("is_decode_values"),
        )
        return None
    if sparse_phase == "invalid":
        return None
    if sparse_phase is None:
        sparse_phase = "prefill" if avg_seq_len and num_tokens >= avg_seq_len else "decode"
    has_sparse_absorption_weights = (
        is_sparse_attention
        and len(args) > 7
        and isinstance(args[6], torch.Tensor)
        and isinstance(args[7], torch.Tensor)
    )
    shape_is_decode = _is_decode_mla(args)
    if is_sparse_attention and sparse_phase == "prefill" and shape_is_decode:
        logger.warning(
            "Sparse MLA op %s has explicit prefill phase but decode-shaped query_lens; falling back to analytic model",
            op_invoke_info.func,
        )
        return None

    if shape_is_decode or has_sparse_absorption_weights:
        W_UK_T = args[6]  # (num_heads, qk_nope_head_dim, kv_lora_rank)
        W_UV = args[7]  # (num_heads, kv_lora_rank, v_head_dim)
        if W_UK_T is None or W_UV is None:
            return None

        qk_nope_head_dim = W_UK_T.shape[1]
        kv_lora_rank = W_UK_T.shape[2]
        v_head_dim_val = W_UV.shape[2]

        # Fix MISS #5: FIA decode Q only sees kv_lora_rank (512), not full head_dim (576).
        # The rope dim (64) is handled by InterleaveRope separately.
        fia_head_dim = kv_lora_rank  # 512, not head_dim=576
        fia_q_raw = (batch_size, num_heads, 1, fia_head_dim)
        fia_q_normalized = _normalize_fia_q_shape(fia_q_raw, fia_head_dim)

        work_tokens = num_tokens
        work_heads = num_heads
        runtime_vectors = None
        if is_sparse_attention:
            tp_size = mapping.get("_runtime_tp_size")
            if (
                sparse_phase == "prefill"
                and mapping.get("_runtime_sequence_parallel")
                and isinstance(tp_size, int)
                and tp_size > 1
            ):
                work_tokens = math.ceil(num_tokens / tp_size)
                if not sp_heads_already_global:
                    work_heads = num_heads * tp_size
                runtime_vectors = _rank_zero_sparse_runtime_vectors(
                    seq_lens=seq_lens,
                    global_query_tokens=num_tokens,
                    local_query_tokens=work_tokens,
                    query_lens=args[5] if isinstance(args[5], torch.Tensor) else None,
                )

        if is_sparse_attention:
            topk_limit = args[10] if len(args) > 10 and isinstance(args[10], int) else None
            topk_indices = args[11] if len(args) > 11 else None
            if topk_limit is None and isinstance(topk_indices, torch.Tensor) and topk_indices.ndim > 0:
                topk_limit = topk_indices.shape[-1]
            attn_spec = SubKernelSpec(
                kernel_type=attention_kernel_type,
                input_shapes=[],
                dtype=dtype_str,
                query_mode="attention",
                attention_params=_sparse_runtime_attention_params(
                    work_tokens=work_tokens,
                    work_heads=work_heads,
                    head_dim=fia_head_dim,
                    seq_lens=seq_lens,
                    avg_seq_len=avg_seq_len,
                    topk=topk_limit,
                    block_size=int(kv_cache.shape[-2]),
                    include_sparse_block_size=True,
                    query_lengths_values=runtime_vectors[0] if runtime_vectors is not None else None,
                    kv_lengths_values=runtime_vectors[1] if runtime_vectors is not None else None,
                ),
                is_attention=True,
            )
        else:
            attn_spec = SubKernelSpec(
                kernel_type=attention_kernel_type,
                input_shapes=[],
                dtype=dtype_str,
                query_mode="attention",
                attention_params={
                    "q_shape_3d": fia_q_normalized or (batch_size, num_heads, fia_head_dim),
                    "avg_seq_len": avg_seq_len,
                    "sparse_mode": 0,  # decode uses no_mask
                    "num_kv_heads": 1,  # MLA compressed attention: single KV head
                },
                is_attention=True,
            )

        # QuantBatchMatmulV3 CSV has extra inputs (bias columns) beyond
        # the 2 TC shapes; tc_input_count=2 tells shape matching to only
        # compare the first 2 CSV inputs. BF16 BatchMatMulV2/BatchMatMulNd
        # CSV inputs already match the 2 TC shapes, so no override is needed.
        first_tc_input_count = 2 if first_kernel_type == "QuantBatchMatmulV3" else None

        # Fix MISS #6: NPU BatchMatMulV2/BatchMatMulNd/TransposeBatchMatMul
        # use heads-first layout (H,T,D), not (T,H,D).

        specs = [
            SubKernelSpec(
                kernel_type=first_kernel_type,
                input_shapes=[
                    (work_heads, work_tokens, qk_nope_head_dim),
                    (work_heads, qk_nope_head_dim, kv_lora_rank),
                ],
                dtype=dtype_str,
                tc_input_count=first_tc_input_count,
                alternate_kernel_types=alternate_kernel_types,
            ),
            attn_spec,
            SubKernelSpec(
                kernel_type="TransposeBatchMatMul",
                input_shapes=[
                    (work_heads, work_tokens, kv_lora_rank),
                    (work_heads, kv_lora_rank, v_head_dim_val),
                ],
                dtype=dtype_str,
            ),
        ]
        tail_options = decomposer_options.get("prefill_tail_transpose", {})
        if is_sparse_attention and sparse_phase == "prefill" and tail_options:
            requires_sp = bool(tail_options.get("requires_sequence_parallel"))
            runtime_sp = bool(mapping.get("_runtime_sequence_parallel"))
            if not requires_sp or runtime_sp:
                tp_size = mapping.get("_runtime_tp_size")
                tail_kernel_type = tail_options.get("kernel_type")
                if not isinstance(tp_size, int) or tp_size <= 1 or not isinstance(tail_kernel_type, str):
                    return None
                tail_width = num_heads * v_head_dim_val
                if sp_heads_already_global and dsa_cp_layout.get("tail_width_partition") == "tp":
                    if tail_width % tp_size != 0:
                        return None
                    tail_width //= tp_size
                specs.append(
                    SubKernelSpec(
                        kernel_type=tail_kernel_type,
                        input_shapes=[(work_tokens, tp_size, tail_width), (3,)],
                        dtype=dtype_str,
                        input_dtypes=[dtype_str, "INT64"],
                        tc_input_count=2,
                    )
                )
        return specs
    else:
        # Prefill: MatMulV2(kv_c@kv_b_proj) + FusedInferAttentionScore
        # vllm-ascend v0.18.0: MLA prefill uses FIA (unified, RING kernel removed)
        kv_b_proj = args[8]  # (kv_lora_rank, num_heads*(qk_nope_head_dim+v_head_dim))
        if kv_b_proj is None:
            logger.debug("MLA prefill: kv_b_proj is None, fallback to analytic")
            return None

        kv_lora_rank = kv_b_proj.shape[0]
        # Fix MISS #4: FIA prefill uses TND layout: (num_tokens, num_heads, qk_nope_head_dim).
        # qk_head_dim = q.shape[2], qk_rope_head_dim = head_dim - kv_lora_rank,
        # qk_nope_head_dim = qk_head_dim - qk_rope_head_dim.
        qk_head_dim = q.shape[2]
        qk_rope_head_dim = head_dim - kv_lora_rank
        qk_nope_head_dim_pf = qk_head_dim - qk_rope_head_dim
        fia_q_shape_3d = (num_tokens, num_heads, qk_nope_head_dim_pf)

        if attention_kernel_type == "SparseFlashAttention":
            # GLM5's SFA call consumes the latent 512-wide ql_nope tensor,
            # not the 192-wide pre-absorption MLA query.  With sequence
            # parallel enabled, vllm-ascend shards tokens across TP ranks and
            # gathers the TP-local heads into the TND head axis:
            #   (T, N_local, D) -> (T / TP, N_local * TP, D)
            # This projection is guarded by both runtime SP state and an
            # explicit versioned mapping rule; it must never be inferred from
            # T=4096 alone.
            sfa_tokens = num_tokens
            sfa_heads = num_heads
            runtime_vectors = None
            tp_size = mapping.get("_runtime_tp_size")
            if mapping.get("_runtime_sequence_parallel") and isinstance(tp_size, int) and tp_size > 1:
                sfa_tokens = math.ceil(num_tokens / tp_size)
                if not sp_heads_already_global:
                    sfa_heads = num_heads * tp_size
                runtime_vectors = _rank_zero_sparse_runtime_vectors(
                    seq_lens=seq_lens,
                    global_query_tokens=num_tokens,
                    local_query_tokens=sfa_tokens,
                    query_lens=args[5] if isinstance(args[5], torch.Tensor) else None,
                )
            topk_limit = args[10] if len(args) > 10 and isinstance(args[10], int) else None
            topk_indices = args[11] if len(args) > 11 else None
            if topk_limit is None and isinstance(topk_indices, torch.Tensor) and topk_indices.ndim > 0:
                topk_limit = topk_indices.shape[-1]
            prefill_attn_spec = SubKernelSpec(
                kernel_type=attention_kernel_type,
                input_shapes=[],
                dtype=dtype_str,
                query_mode="attention",
                attention_params=_sparse_runtime_attention_params(
                    work_tokens=sfa_tokens,
                    work_heads=sfa_heads,
                    head_dim=kv_lora_rank,
                    seq_lens=seq_lens,
                    avg_seq_len=avg_seq_len,
                    topk=topk_limit,
                    block_size=int(kv_cache.shape[-2]),
                    include_sparse_block_size=True,
                    query_lengths_values=runtime_vectors[0] if runtime_vectors is not None else None,
                    kv_lengths_values=runtime_vectors[1] if runtime_vectors is not None else None,
                ),
                is_attention=True,
            )
        else:
            prefill_attn_spec = SubKernelSpec(
                kernel_type=attention_kernel_type,
                input_shapes=[],
                dtype=dtype_str,
                query_mode="attention",
                attention_params={
                    "q_shape_3d": fia_q_shape_3d,
                    "avg_seq_len": avg_seq_len,
                    "sparse_mode": 3,  # causal mask for prefill
                    # MLA prefill: K/V are decompressed via kv_b_proj to
                    # (T, num_heads, qk_nope_head_dim), so num_kv_heads =
                    # num_heads (= q.shape[1], already TP-divided).
                    # This differs from MLA decode where KV stays compressed
                    # as a single latent vector (num_kv_heads=1) and FIA v2
                    # handles the absorption internally.
                    # Ref: vllm-ascend mla_v1.py
                    #   _forward_prefill(): num_key_value_heads=self.num_heads
                    #   _forward_decode():  num_key_value_heads=self.num_kv_heads (=1)
                    "num_kv_heads": num_heads,
                },
                is_attention=True,
            )

        return [
            SubKernelSpec(
                kernel_type="MatMulV2",
                input_shapes=[
                    (num_tokens, kv_lora_rank),
                    tuple(kv_b_proj.shape),
                ],
                dtype=dtype_str,
                tc_input_count=2,
            ),
            prefill_attn_spec,
        ]


def _decompose_mla(op_invoke_info: "OpInvokeInfo", mapping: dict) -> Optional[List[SubKernelSpec]]:
    """Decompose multihead_latent_attention (BF16)."""
    return _decompose_mla_common(
        op_invoke_info,
        mapping,
        "BatchMatMulV2",
        alternate_kernel_types=["BatchMatMulNd"],
    )


def _decompose_mla_quant(op_invoke_info: "OpInvokeInfo", mapping: dict) -> Optional[List[SubKernelSpec]]:
    """Decompose multihead_latent_attention_quant."""
    return _decompose_mla_common(op_invoke_info, mapping, "QuantBatchMatmulV3")


def _decompose_mla_sparse(op_invoke_info: "OpInvokeInfo", mapping: dict) -> Optional[List[SubKernelSpec]]:
    """Decompose mla_sparse_attention (BF16). Maps attention kernel to SparseFlashAttention.

    SFA requires enriched runtime context. A CSV lacking effective sequence
    length, phase, sparse mode, KV heads, layout, or top-k must miss rather than
    selecting the first row with a matching Q shape.
    """
    return _decompose_mla_common(
        op_invoke_info,
        mapping,
        "BatchMatMulV2",
        alternate_kernel_types=["BatchMatMulNd"],
        attention_kernel_type="SparseFlashAttention",
    )


def _decompose_mla_sparse_quant(op_invoke_info: "OpInvokeInfo", mapping: dict) -> Optional[List[SubKernelSpec]]:
    """Decompose mla_sparse_attention_quant. Maps attention kernel to SparseFlashAttention.

    Note: alternate_kernel_types is intentionally omitted — QuantBatchMatmulV3 has no
    ND variant fallback (unlike BF16 BatchMatMulV2 which falls back to BatchMatMulNd).

    When SparseFlashAttention.csv is absent, falls back to analytic model (same as BF16
    sparse variant). See _decompose_mla_sparse for details, including the
    tc_input_count=1 rationale for the SFA attention sub-kernel.
    """
    return _decompose_mla_common(
        op_invoke_info,
        mapping,
        "QuantBatchMatmulV3",
        attention_kernel_type="SparseFlashAttention",
    )


def _decompose_mlapo_common(
    op_invoke_info: "OpInvokeInfo",
    mapping: dict,
    matmul_kernel_type: str,
    min_args: int = 14,
) -> Optional[List[SubKernelSpec]]:
    """Shared MLAPO decomposition for BF16 and quantized variants.

    TC mlapo fuses: q_a_proj + q_a_norm + q_b_proj + kv_a_proj + kv_a_norm + rope.
    NPU fuses q_a_proj + kv_a_proj into a single fused_qkv_a_proj matmul
    (output dim = q_lora_rank + kv_lora_rank + rope_dim = 2112 for DSv3),
    then runs q_b_proj separately.  Decompose to match profiling data:
      1. fused_qkv_a_proj: matmul(hidden, [q_lora_rank+kv_proj_dim, hidden_size])
      2. q_b_proj: matmul(q_compressed, q_b_proj_weight)
      3. KvRmsNormRopeCache (norm + rope post-projection)

    Args:
        matmul_kernel_type: "MatMulV2" for BF16, "QuantBatchMatmulV3" for quant.
        min_args: Minimum args count (14 for BF16, 20 for quant).

    Args layout (tensor_cast/ops/mla.py):
        args[0]: hidden_states (num_tokens, hidden_size)
        args[3]: q_a_proj_weight (q_lora_rank, hidden_size) — Optional
        args[5]: q_b_proj_weight — Optional; may be sliced by SinkSplitPass
        args[6]: kv_a_proj_weight (kv_lora_rank+rope_dim, hidden_size) — Optional
        args[8]: num_heads (int) — used to compute full q_b_proj shape
        args[9]: qk_head_dim (int) — used to compute full q_b_proj shape
    """
    args = op_invoke_info.args
    if len(args) < min_args:
        return None

    hidden_states = args[0]
    q_a_proj = args[3]
    q_b_proj = args[5]
    kv_a_proj = args[6]

    if hidden_states is None or q_a_proj is None or q_b_proj is None or kv_a_proj is None:
        return None

    dtype_str = DTYPE_MAP.get(hidden_states.dtype)
    if dtype_str is None:
        return None

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    q_lora_rank = q_a_proj.shape[0]
    kv_proj_dim = kv_a_proj.shape[0]
    fused_proj_dim = q_lora_rank + kv_proj_dim

    # Some versioned stacks expose the sequence-parallel physical kernels for
    # MLAPO only when SP is enabled; their profiling shapes use TP-local tokens
    # while the TensorCast fused op retains the logical/global token axis.  This
    # projection is mapping opt-in and runtime guarded so it cannot silently
    # rewrite ordinary MLAPO or Decode shapes.
    decomposer_options = mapping.get("decomposer_options", {})
    visible_regime = decomposer_options.get("visible_kernel_regime", {})
    runtime_sequence_parallel = bool(mapping.get("_runtime_sequence_parallel"))
    if visible_regime.get("requires_sequence_parallel") and not runtime_sequence_parallel:
        opaque_kernel = decomposer_options.get("opaque_kernel")
        if matmul_kernel_type != "QuantBatchMatmulV3" or not isinstance(opaque_kernel, dict):
            return None
        num_heads = args[8]
        qk_nope_head_dim = args[10]
        qk_rope_head_dim = args[11]
        kv_lora_rank = args[12]
        q_lora_rank_arg = args[13]
        quantized_weights = all(
            isinstance(weight, torch.Tensor) and weight.dtype == torch.int8
            for weight in (q_a_proj, q_b_proj, kv_a_proj)
        )
        if (
            not all(
                isinstance(value, int) and value > 0
                for value in (
                    num_heads,
                    qk_nope_head_dim,
                    qk_rope_head_dim,
                    kv_lora_rank,
                    q_lora_rank_arg,
                )
            )
            or not quantized_weights
        ):
            return None
        return [
            SubKernelSpec(
                kernel_type=str(opaque_kernel.get("kernel_type", "mla_preprocess_0_mix_aic")),
                input_shapes=[],
                dtype=dtype_str,
                query_mode=str(opaque_kernel.get("query_mode", "mlapo_preprocess")),
                runtime_params={
                    "num_tokens": int(num_tokens),
                    "hidden_size": int(hidden_size),
                    "local_num_heads": int(num_heads),
                    "q_lora_rank": int(q_lora_rank_arg),
                    "kv_lora_rank": int(kv_lora_rank),
                    "qk_nope_head_dim": int(qk_nope_head_dim),
                    "qk_rope_head_dim": int(qk_rope_head_dim),
                    "block_size": int(opaque_kernel.get("block_size", 128)),
                    "cache_mode": str(opaque_kernel.get("cache_mode", "krope_ctkv")),
                    "quant_mode": str(opaque_kernel.get("quant_mode", "per_tensor_quant_asymm")),
                    "enable_inner_out": bool(opaque_kernel.get("enable_inner_out", True)),
                    "weight_quantized": bool(opaque_kernel.get("weight_quantized", True)),
                    "weight_format": str(opaque_kernel.get("weight_format", "FRACTAL_NZ")),
                },
            )
        ]

    physical_tokens = num_tokens
    tp_size = mapping.get("_runtime_tp_size")
    if runtime_sequence_parallel and decomposer_options.get("projection_token_partition") == "tp":
        if not isinstance(tp_size, int) or tp_size <= 1 or num_tokens % tp_size != 0:
            return None
        physical_tokens = num_tokens // tp_size

    # Fix MISS #1: QuantBatchMatmulV3 activation dtype is INT8 (DynamicQuant runs
    # before QBMV3 on NPU). BF16 path (MatMulV2) keeps the original dtype_str.
    matmul_dtype = "INT8" if matmul_kernel_type == "QuantBatchMatmulV3" else dtype_str

    # Fix MISS #2: For the quant path, SinkSplitPass slices q_b_proj_weight for TP,
    # so args[5].shape is wrong (e.g. (384, q_lora_rank) instead of (3072, q_lora_rank)).
    # Compute the full weight shape from int params args[8]=num_heads, args[9]=qk_head_dim.
    # For the BF16 path (min_args=14), args[8] and args[9] may be None — fall back to
    # the actual tensor shape.
    num_heads_int = args[8]
    qk_head_dim_int = args[9]
    if (
        matmul_kernel_type == "QuantBatchMatmulV3"
        and isinstance(num_heads_int, int)
        and isinstance(qk_head_dim_int, int)
    ):
        physical_num_heads = num_heads_int
        if runtime_sequence_parallel and decomposer_options.get("reconstruct_q_heads_by_tp"):
            if not isinstance(tp_size, int) or tp_size <= 1:
                return None
            physical_num_heads *= tp_size
        q_b_proj_weight_shape = (physical_num_heads * qk_head_dim_int, q_lora_rank)
    else:
        q_b_proj_weight_shape = tuple(q_b_proj.shape)

    q_a_norm_weight = args[4]
    kv_a_norm_weight = args[7]
    if not isinstance(q_a_norm_weight, torch.Tensor) or not isinstance(kv_a_norm_weight, torch.Tensor):
        return None

    specs = []
    if matmul_kernel_type == "QuantBatchMatmulV3":
        specs.append(
            SubKernelSpec(
                kernel_type="AscendQuantV2",
                alternate_kernel_types=["AscendQuantV2Aicore"],
                input_shapes=[(physical_tokens, hidden_size), (hidden_size,), (hidden_size,)],
                dtype=dtype_str,
                tc_input_count=3,
            )
        )
    specs.append(
        SubKernelSpec(
            kernel_type=matmul_kernel_type,
            input_shapes=[(physical_tokens, hidden_size), (fused_proj_dim, hidden_size)],
            dtype=matmul_dtype,
            input_dtypes=[matmul_dtype, matmul_dtype],
            tc_input_count=2,
        )
    )
    specs.append(
        SubKernelSpec(
            kernel_type="RmsNorm",
            input_shapes=[(physical_tokens, q_lora_rank), tuple(q_a_norm_weight.shape)],
            dtype=dtype_str,
            tc_input_count=2,
        )
    )
    if matmul_kernel_type == "QuantBatchMatmulV3":
        specs.append(
            SubKernelSpec(
                kernel_type="AscendQuantV2",
                alternate_kernel_types=["AscendQuantV2Aicore"],
                input_shapes=[(physical_tokens, q_lora_rank), (q_lora_rank,), (q_lora_rank,)],
                dtype=dtype_str,
                tc_input_count=3,
            )
        )
    specs.append(
        SubKernelSpec(
            kernel_type=matmul_kernel_type,
            input_shapes=[(physical_tokens, q_lora_rank), q_b_proj_weight_shape],
            dtype=matmul_dtype,
            input_dtypes=[matmul_dtype, matmul_dtype],
            tc_input_count=2,
        )
    )
    kv_cache_query = decomposer_options.get("kv_cache_query", {})
    kv_cache_query_mode = kv_cache_query.get("mode")
    if kv_cache_query_mode == "pool_dim0_agnostic":
        kv_lora_rank = int(kv_a_norm_weight.shape[0])
        rope_dim = kv_proj_dim - kv_lora_rank
        block_size = kv_cache_query.get("block_size")
        if rope_dim <= 0 or not isinstance(block_size, int) or block_size <= 0:
            return None
        specs.append(
            SubKernelSpec(
                kernel_type="KvRmsNormRopeCache",
                input_shapes=[],
                dtype=dtype_str,
                query_mode="cache_postprocess",
                cache_params={
                    "tokens": physical_tokens,
                    "kv_proj_dim": kv_proj_dim,
                    "kv_lora_rank": kv_lora_rank,
                    "rope_dim": rope_dim,
                    "block_size": block_size,
                },
            )
        )
    elif kv_cache_query_mode is None:
        # Older mappings do not provide semantic cache-query metadata. Keep
        # their original exact-shape lookup instead of constructing an
        # attention query whose required sequence context is unavailable on
        # the MLAPO op.
        specs.append(
            SubKernelSpec(
                kernel_type="KvRmsNormRopeCache",
                input_shapes=[(physical_tokens, 1, 1, kv_proj_dim)],
                dtype=dtype_str,
            )
        )
    else:
        logger.warning("Unsupported MLAPO kv_cache_query mode: %s", kv_cache_query_mode)
        return None
    return specs


def _decompose_mlapo(op_invoke_info: "OpInvokeInfo", mapping: dict) -> Optional[List[SubKernelSpec]]:
    """Decompose mlapo (BF16)."""
    return _decompose_mlapo_common(op_invoke_info, mapping, "MatMulV2", min_args=14)


def _decompose_mlapo_quant(op_invoke_info: "OpInvokeInfo", mapping: dict) -> Optional[List[SubKernelSpec]]:
    """Decompose mlapo_quant."""
    return _decompose_mlapo_common(op_invoke_info, mapping, "QuantBatchMatmulV3", min_args=20)


def _decompose_dsa_indexer(op_invoke_info: "OpInvokeInfo", mapping: dict) -> Optional[List[SubKernelSpec]]:
    """Decompose the GLM5 BF16 DSA indexer without using physical pool capacity.

    Ordinary sub-kernel shapes come only from the semantic op tensors and
    weights. LightningIndexer is queried with the local query work and the
    effective sequence length; its preallocated cache dim0 is deliberately not
    part of the semantic key.
    """

    args = op_invoke_info.args
    if len(args) < 16:
        return None

    hidden_states, qa_normed = args[0], args[1]
    indexer_cache, block_tables, seq_lens = args[4], args[6], args[7]
    query_lens = op_invoke_info.kwargs.get("query_lens")
    wq_b_weight, wk_weight = args[8], args[9]
    weights_proj_weight, k_norm_weight = args[10], args[11]
    num_heads, head_dim, topk_limit = args[12], args[13], args[15]
    tensors = (
        hidden_states,
        qa_normed,
        indexer_cache,
        block_tables,
        seq_lens,
        wq_b_weight,
        wk_weight,
        weights_proj_weight,
        k_norm_weight,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        return None
    if not all(isinstance(value, int) for value in (num_heads, head_dim, topk_limit)):
        return None
    if hidden_states.ndim not in (2, 3) or qa_normed.ndim not in (2, 3):
        return None

    dtype_str = DTYPE_MAP.get(hidden_states.dtype)
    if dtype_str is None:
        return None
    num_tokens = int(hidden_states.numel() // hidden_states.shape[-1])
    hidden_size = hidden_states.shape[-1]
    qa_dim = qa_normed.shape[-1]

    try:
        effective_seq_len = int(seq_lens.float().mean().item())
    except (RuntimeError, ValueError):
        effective_seq_len = None

    phase = _resolve_batch_phase(op_invoke_info, int(seq_lens.shape[0]))
    if phase == "mixed":
        logger.debug(
            "DSA indexer op %s has mixed prefill/decode requests; falling back to analytic model (is_decode_values=%s)",
            op_invoke_info.func,
            op_invoke_info.kwargs.get("is_decode_values"),
        )
        return None
    if phase == "invalid":
        return None
    if phase is None:
        # Compatibility fallback for older call sites without explicit phase.
        # seq_lens is already the effective length (do not add query tokens
        # again), so a forward whose local query work covers that length is
        # prefill; a much smaller query is decode.
        phase = (
            "prefill"
            if effective_seq_len is not None and num_tokens >= effective_seq_len
            else "decode"
            if effective_seq_len is not None
            else None
        )

    indexer_tokens = num_tokens
    runtime_vectors = None
    tp_size = mapping.get("_runtime_tp_size")
    if phase == "prefill" and mapping.get("_runtime_sequence_parallel") and isinstance(tp_size, int) and tp_size > 1:
        indexer_tokens = math.ceil(num_tokens / tp_size)
        runtime_vectors = _rank_zero_sparse_runtime_vectors(
            seq_lens=seq_lens,
            global_query_tokens=num_tokens,
            local_query_tokens=indexer_tokens,
            query_lens=query_lens if isinstance(query_lens, torch.Tensor) else None,
            require_query_lens_for_multiple_requests=True,
        )
        if runtime_vectors is None:
            logger.warning(
                "DSA indexer op %s cannot recover per-request query boundaries for sequence-parallel profiling; "
                "falling back to analytic model",
                op_invoke_info.func,
            )
            return None

    block_size = indexer_cache.shape[-2] if indexer_cache.ndim >= 3 else None
    runtime_kv_lengths = runtime_vectors[1] if runtime_vectors is not None else None
    active_seq_len = max(runtime_kv_lengths) if runtime_kv_lengths else effective_seq_len
    active_cache_blocks = (
        (active_seq_len + block_size - 1) // block_size
        if active_seq_len is not None and isinstance(block_size, int) and block_size > 0
        else None
    )

    kernel_tokens = indexer_tokens
    wk_out = wk_weight.shape[0]
    specs = [
        SubKernelSpec(
            kernel_type="MatMulV2",
            input_shapes=[(kernel_tokens, hidden_size), tuple(wk_weight.shape)],
            dtype=dtype_str,
            tc_input_count=2,
        ),
        SubKernelSpec(
            kernel_type="LayerNormV3",
            input_shapes=[(kernel_tokens, wk_out), tuple(k_norm_weight.shape), tuple(k_norm_weight.shape)],
            dtype="FLOAT",
            input_dtypes=["FLOAT", "FLOAT", "FLOAT"],
            tc_input_count=3,
        ),
        SubKernelSpec(
            kernel_type="Cast",
            input_shapes=[(kernel_tokens, head_dim)],
            dtype=dtype_str,
            tc_input_count=1,
        ),
        SubKernelSpec(
            kernel_type="Cast",
            input_shapes=[(kernel_tokens, head_dim)],
            dtype=dtype_str,
            tc_input_count=1,
        ),
        SubKernelSpec(
            kernel_type="MatMulV2",
            input_shapes=[(kernel_tokens, hidden_size), tuple(weights_proj_weight.shape)],
            dtype=dtype_str,
            tc_input_count=2,
        ),
    ]

    if wq_b_weight.dtype == torch.int8:
        specs.extend(
            [
                SubKernelSpec(
                    kernel_type="AscendQuantV2",
                    alternate_kernel_types=["AscendQuantV2Aicore"],
                    input_shapes=[(kernel_tokens, qa_dim)],
                    dtype=dtype_str,
                    tc_input_count=1,
                ),
                SubKernelSpec(
                    kernel_type="QuantBatchMatmulV3",
                    input_shapes=[(kernel_tokens, qa_dim), tuple(wq_b_weight.shape)],
                    dtype="INT8",
                    input_dtypes=["INT8", "INT8"],
                    tc_input_count=2,
                ),
            ]
        )
    else:
        specs.append(
            SubKernelSpec(
                kernel_type="MatMulV2",
                input_shapes=[(kernel_tokens, qa_dim), tuple(wq_b_weight.shape)],
                dtype=dtype_str,
                tc_input_count=2,
            )
        )

    specs.append(
        SubKernelSpec(
            kernel_type="ScatterNdUpdate",
            alternate_kernel_types=["ScatterNdUpdateAiCore"],
            input_shapes=[
                tuple(indexer_cache.shape),
                (num_tokens, 1),
                (num_tokens, head_dim),
            ],
            dtype=dtype_str,
            input_dtypes=[dtype_str, "INT32", dtype_str],
            query_mode="scatter_cache_write",
            cache_params={"tokens": num_tokens, "feature_dim": head_dim},
        )
    )
    specs.append(
        SubKernelSpec(
            kernel_type=mapping.get("primary_kernel_type", "LightningIndexer"),
            alternate_kernel_types=mapping.get("alternate_kernel_types"),
            input_shapes=[],
            dtype=dtype_str,
            query_mode="attention",
            attention_params={
                **_sparse_runtime_attention_params(
                    work_tokens=indexer_tokens,
                    work_heads=num_heads,
                    head_dim=head_dim,
                    seq_lens=seq_lens,
                    avg_seq_len=effective_seq_len,
                    topk=topk_limit,
                    block_size=block_size,
                    include_sparse_block_size=False,
                    query_lengths_values=runtime_vectors[0] if runtime_vectors is not None else None,
                    kv_lengths_values=runtime_kv_lengths,
                ),
                "active_cache_blocks": active_cache_blocks,
            },
            is_attention=True,
        )
    )
    return specs


COMPOSITE_DECOMPOSERS: Dict[
    str,
    Callable[["OpInvokeInfo", dict], Optional[List[SubKernelSpec]]],
] = {
    # --- Register new decomposers here ---
    # To add a new composite op with dynamic decomposition:
    #   1. Write _decompose_<op>() above (return List[SubKernelSpec] or None)
    #   2. Add "tensor_cast.<op>.default": _decompose_<op> entry below
    #   3. Set composite: true + decomposer: true in op_mapping.yaml
    # See §14 in OP_PLUGIN_MAPPING_TUTORIAL.md for the full SOP.
    "tensor_cast.multihead_latent_attention.default": _decompose_mla,
    "tensor_cast.multihead_latent_attention_quant.default": _decompose_mla_quant,
    "tensor_cast.mla_sparse_attention.default": _decompose_mla_sparse,
    "tensor_cast.mla_sparse_attention_quant.default": _decompose_mla_sparse_quant,
    "tensor_cast.mlapo.default": _decompose_mlapo,
    "tensor_cast.mlapo_quant.default": _decompose_mlapo_quant,
    "tensor_cast.dsa_indexer.default": _decompose_dsa_indexer,
}


def _first_tensor(value: Any) -> Optional[torch.Tensor]:
    """Return the first tensor contained in a tensor/list/tuple value."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _stacked_expert_weight_shape(
    weights: Any,
    hidden_size: int,
    *,
    hidden_axis: int,
) -> Optional[Tuple[int, ...]]:
    """Project per-expert TC weight lists to the DFC physical stacked shape."""
    if not isinstance(weights, (list, tuple)) or not weights:
        return None
    first_weight = _first_tensor(weights[0])
    if first_weight is None or first_weight.ndim != 2:
        return None
    first_shape = tuple(first_weight.shape)
    num_experts = len(weights)
    if hidden_axis == 0 and first_shape[0] == hidden_size:
        return (num_experts, first_shape[0], first_shape[1])
    if hidden_axis == 0 and first_shape[1] == hidden_size:
        return (num_experts, first_shape[1], first_shape[0])
    if hidden_axis == 1 and first_shape[1] == hidden_size:
        return (num_experts, first_shape[0], first_shape[1])
    if hidden_axis == 1 and first_shape[0] == hidden_size:
        return (num_experts, first_shape[1], first_shape[0])
    return (num_experts, *first_shape)


def _project_dispatch_ffn_combine_inputs(
    op_invoke_info: "OpInvokeInfo",
) -> Optional[List[Tuple[Tuple[int, ...], torch.dtype]]]:
    """Project TC DFC semantic args to the seven physical profiling inputs."""
    args = op_invoke_info.args
    if len(args) < 8:
        return None
    x, expert_indices = args[0], args[1]
    if not isinstance(x, torch.Tensor) or not isinstance(expert_indices, torch.Tensor):
        return None
    if x.ndim == 2:
        x_shape = tuple(x.shape)
    elif x.ndim == 3:
        x_shape = (x.shape[0] * x.shape[1], x.shape[2])
    else:
        return None
    if expert_indices.ndim < 1:
        return None

    func_name = _normalize_func_name(op_invoke_info.func)
    if func_name == "tensor_cast.dispatch_ffn_combine.default":
        gmm1_w, gmm2_w = args[2], args[4]
    elif func_name in {
        "tensor_cast.dispatch_ffn_combine_quant.default",
        "tensor_cast.dispatch_ffn_combine_quant_int4.default",
        "tensor_cast.dispatch_ffn_combine_fp8.default",
        "tensor_cast.dispatch_ffn_combine_mxfp4.default",
    }:
        gmm1_w, gmm2_w = args[2], args[7]
    else:
        return None

    hidden_size = x_shape[-1]
    gmm1_shape = _stacked_expert_weight_shape(gmm1_w, hidden_size, hidden_axis=0)
    gmm2_shape = _stacked_expert_weight_shape(gmm2_w, hidden_size, hidden_axis=1)
    gmm1_tensor, gmm2_tensor = _first_tensor(gmm1_w), _first_tensor(gmm2_w)
    if gmm1_shape is None or gmm2_shape is None or gmm1_tensor is None or gmm2_tensor is None:
        return None

    expert_shape = tuple(expert_indices.shape)
    num_experts = gmm1_shape[0]
    return [
        (x_shape, x.dtype),
        (gmm1_shape, gmm1_tensor.dtype),
        (gmm2_shape, gmm2_tensor.dtype),
        (expert_shape, torch.int32),
        ((num_experts * gmm1_shape[-1],), torch.int64),
        ((num_experts * hidden_size,), torch.int64),
        (expert_shape, torch.float32),
    ]


# Checker function type: (row, kernel_type, latency_col) -> Optional[Candidate]
# Used by _find_candidates to unify CSV iteration across all query categories.
CheckerFn = Callable[[pd.Series, str, str], Optional["Candidate"]]


@dataclass
class Candidate:
    """A matched CSV row result from a checker function."""

    latency_us: float
    kernel_type: str
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    shape_match_info: Optional[ShapeMatchInfo] = None
    distance: float = 0.0  # for nearest-neighbor selection (attention)


def _shape_match_distance(rule: str) -> float:
    """Rank semantic-exact shape rewrites ahead of padding approximations."""
    return 1.0 if "padding" in rule else 0.0


@dataclass(frozen=True)
class _FiaColInfo:
    """Detected column names for FIA enriched CSV (cached per kernel_type)."""

    avg_seq_col: str
    has_sparse: bool
    has_kv_heads: bool
    has_layout: bool
    has_phase: bool
    has_topk: bool
    has_block_size: bool
    has_actual_query_values: bool
    has_actual_kv_values: bool
    has_valid_blocks: bool
    has_num_heads: bool
    has_cache_layout: bool
    has_kv_cache_mode: bool
    has_sparse_block_size: bool
    has_sparse_indices_pattern: bool
    has_sparse_indices_valid_count: bool


class ProfilingDataSource(DataSourcePerformanceModel):
    """CSV-backed data source with op_mapping.yaml + FRACTAL_NZ.

    Internally handles all mapping, shape extraction, format conversion.
    The caller (EmpiricalPerformanceModel) only calls lookup(OpInvokeInfo).

    Init args:
        data_dir: path containing op_mapping.yaml + {KernelType}.csv files
        device_profile: DeviceProfile for comm_grid topology_tier resolution.
            Optional — when omitted, communication lookups skip tier filtering.
    """

    def __init__(
        self,
        data_dir: str | Path,
        device_profile: Optional[DeviceProfile] = None,
        parallel_config: Optional["ParallelConfig"] = None,
    ):
        self.data_dir = Path(data_dir)
        self.comm_grid = device_profile.comm_grid if device_profile else None
        self.ep_size = parallel_config.expert_parallel_size if parallel_config else None
        self.tp_size = parallel_config.tensor_parallel_size if parallel_config else None
        self._op_mapping = self._load_op_mapping()
        self._latency_kernel_overrides = self._op_mapping.get("latency_policy", {}).get("kernel_overrides", {})
        self._query_selection_kernel_overrides = self._op_mapping.get("query_selection_policy", {}).get(
            "kernel_overrides", {}
        )
        self._csv_cache: Dict[str, Optional[pd.DataFrame]] = {}
        # Resolve communication data directory from op_mapping communication_data_ref.
        # Falls back to data_dir when the field is absent (legacy layout).
        # NOTE: when _comm_data_dir == data_dir, the fallback in _load_csv is
        # redundant but harmless — kept for clarity over micro-optimization.
        comm_ref = self._op_mapping.get("communication_data_ref")
        if comm_ref:
            self._comm_data_dir = (self.data_dir / comm_ref).resolve()
        else:
            self._comm_data_dir = self.data_dir
        # Set after each lookup() miss to explain why
        self.last_miss_reason: str = ""
        # Set after each lookup() call with shape debug info (HIT or MISS)
        self.last_shape_match_info: Optional[ShapeMatchInfo] = None

    def _load_op_mapping(self) -> dict:
        yaml_path = self.data_dir / "op_mapping.yaml"
        if not yaml_path.exists():
            logger.warning("op_mapping.yaml not found at %s", yaml_path)
            return {}
        with open(yaml_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_csv(self, kernel_type: str) -> Optional[pd.DataFrame]:
        # Convention: comm kernel_types use lowercase hcom_ prefix (e.g. hcom_allReduce_).
        # CamelCase variants (HcomAllReduce) are graph-compiled names and should be
        # listed in alternate_kernel_types, not as primary kernel_type.
        if kernel_type in self._csv_cache:
            return self._csv_cache[kernel_type]
        # Comm kernels: prefer _comm_data_dir (hccl/), fallback to data_dir.
        # This ensures dedicated HCCL benchmark data takes precedence over any
        # comm CSVs that may exist alongside compute kernels in the vllm dir.
        # When _comm_data_dir is None (no communication_data_ref in op_mapping),
        # falls through to the else branch using data_dir directly.
        if self._comm_data_dir and kernel_type.startswith("hcom_"):
            csv_path = self._comm_data_dir / f"{kernel_type}.csv"
            if not csv_path.exists():
                csv_path = self.data_dir / f"{kernel_type}.csv"
        else:
            csv_path = self.data_dir / f"{kernel_type}.csv"
        if not csv_path.exists():
            logger.debug("CSV not found: %s", csv_path)
            self._csv_cache[kernel_type] = None
            return None
        df = pd.read_csv(csv_path)
        self._csv_cache[kernel_type] = df
        return df

    @staticmethod
    def _latency_col(df: pd.DataFrame) -> str:
        """Return the latency column name present in *df*.

        Priority: Average Duration (microbench best) > Profiling Average
        Duration (enriched CSV).  Falls back to the latter when the former
        is absent.
        """
        for col in _LATENCY_COLS:
            if col in df.columns:
                return col
        return _LATENCY_COLS[-1]

    @staticmethod
    def _candidate_latency_cols(preferred_col: str) -> tuple[str, ...]:
        candidate_cols = [preferred_col]
        for col in _LATENCY_COLS:
            if col not in candidate_cols:
                candidate_cols.append(col)
        return tuple(candidate_cols)

    @staticmethod
    def _row_latency_pair(row: pd.Series, preferred_col: str) -> Optional[tuple[str, float]]:
        """Return the latency column and finite positive value for this row."""
        for col in ProfilingDataSource._candidate_latency_cols(preferred_col):
            if col not in row.index:
                continue
            try:
                latency = float(row[col])
            except (TypeError, ValueError):
                continue
            if np.isfinite(latency) and latency > 0:
                return col, latency
        return None

    @staticmethod
    def _row_latency_value(row: pd.Series, preferred_col: str) -> Optional[float]:
        """Return a finite positive latency value for this row."""
        pair = ProfilingDataSource._row_latency_pair(row, preferred_col)
        return None if pair is None else pair[1]

    @staticmethod
    def _row_latency_col(row: pd.Series, preferred_col: str) -> Optional[str]:
        """Return a latency column with a finite positive value for this row."""
        pair = ProfilingDataSource._row_latency_pair(row, preferred_col)
        return None if pair is None else pair[0]

    def _effective_row_latency(
        self,
        row: pd.Series,
        kernel_type: str,
        preferred_col: str,
    ) -> Optional[tuple[str, float, Dict[str, Any]]]:
        """Select a measured row latency with optional versioned quality policies.

        Raw profiling duration can also contain host/profiler scheduling
        outliers even when the AIC/AIV task counters remain stable.  For MIX_AIC
        rows, a second versioned policy can replace such an outlier with the
        concurrent device-core execution envelope.
        """
        pair = self._row_latency_pair(row, preferred_col)
        if pair is None:
            return None
        selected_col, raw_latency_us = pair
        details: Dict[str, Any] = {}

        override = self._latency_kernel_overrides.get(kernel_type, {})
        fallback = override.get("profiling_core_envelope_fallback", {})
        threshold = fallback.get("max_duration_to_core_ratio")
        if (
            threshold is None
            or selected_col != _PROFILING_DURATION_COL
            or str(row.get("Accelerator Core", "")).strip() != "MIX_AIC"
        ):
            return selected_col, raw_latency_us, details

        try:
            aicore_us = float(row[_PROFILING_AICORE_TIME_COL])
            aiv_us = float(row[_PROFILING_AIV_TIME_COL])
            threshold_value = float(threshold)
        except (KeyError, TypeError, ValueError):
            return selected_col, raw_latency_us, details
        if not all(np.isfinite(value) and value > 0 for value in (aicore_us, aiv_us, threshold_value)):
            return selected_col, raw_latency_us, details

        core_envelope_us = max(aicore_us, aiv_us)
        duration_to_core_ratio = raw_latency_us / core_envelope_us
        if duration_to_core_ratio <= threshold_value:
            return selected_col, raw_latency_us, details

        details.update(
            {
                "latency_column": selected_col,
                "latency_selection": "profiling_core_envelope_fallback",
                "raw_latency_us": raw_latency_us,
                "core_envelope_us": core_envelope_us,
                "duration_to_core_ratio": duration_to_core_ratio,
                "max_duration_to_core_ratio": threshold_value,
            }
        )
        return _EFFECTIVE_LATENCY_COL, core_envelope_us, details

    # ---- Unified CSV iteration (PR#123 §3.1: checker pattern) ----

    def _find_candidates(
        self,
        kernel_types: List[str],
        checker_fn: CheckerFn,
        select: str = "first",
    ) -> Optional[Candidate]:
        """Unified CSV iteration loop.

        Iterates *kernel_types* in order, loads each CSV, and calls
        *checker_fn(row, kernel_type, latency_col)* for every row.

        Args:
            kernel_types: Kernel type names to try in order (primary + alternates).
            checker_fn: Category-specific matching function. Returns a Candidate
                on match, None on mismatch.
            select: ``"first"`` returns the first match (compute, moe, elementwise).
                ``"nearest"`` returns the match with smallest ``distance``
                (attention avg_seq_len nearest-neighbor).

        Returns:
            Best Candidate, or None if no match found.
        """
        self.last_miss_reason = ""
        best: Optional[Candidate] = None
        best_score: Optional[tuple[float, int, int]] = None
        best_ties: List[Candidate] = []
        any_csv_loaded = False

        for kernel_index, kernel_type in enumerate(kernel_types):
            df = self._load_csv(kernel_type)
            if df is None:
                continue
            any_csv_loaded = True
            lat_col = self._latency_col(df)

            for _, row in df.iterrows():
                effective_latency = self._effective_row_latency(row, kernel_type, lat_col)
                if effective_latency is None:
                    logger.debug("Skipping %s row with no finite positive latency", kernel_type)
                    continue
                row_lat_col, row_latency_us, latency_details = effective_latency
                checker_row = row
                if row_lat_col == _EFFECTIVE_LATENCY_COL:
                    checker_row = row.copy()
                    checker_row[_EFFECTIVE_LATENCY_COL] = row_latency_us
                candidate = checker_fn(checker_row, kernel_type, row_lat_col)
                if candidate is None:
                    continue
                candidate.details.update(latency_details)
                if select == "first":
                    return candidate

                override = self._latency_kernel_overrides.get(kernel_type, {})
                if override.get("prefer_profiling_rows"):
                    latency_priority = 0 if row_lat_col == _PROFILING_DURATION_COL else 1
                else:
                    latency_priority = (
                        _LATENCY_COLS.index(row_lat_col) if row_lat_col in _LATENCY_COLS else len(_LATENCY_COLS)
                    )
                score = (candidate.distance, kernel_index, latency_priority)
                if best_score is None or score < best_score:
                    best = candidate
                    best_score = score
                    best_ties = [candidate]
                elif score == best_score:
                    best_ties.append(candidate)

        if best is not None and best_ties:
            require_unique = bool(
                self._query_selection_kernel_overrides.get(best.kernel_type, {}).get("require_semantic_unique")
            )
            if require_unique:
                distinct_latencies = {round(candidate.latency_us, 12) for candidate in best_ties}
                if len(distinct_latencies) > 1:
                    self.last_miss_reason = f"ambiguous_semantic_match:{best.kernel_type}"
                    return None

        if best is not None:
            return best

        # MISS: set reason based on whether any CSV was loaded
        if not any_csv_loaded:
            self.last_miss_reason = "csv_not_found"
        return None

    def _query_comm_csv(
        self,
        kernel_type: str,
        message_bytes: int,
        num_devices: int,
        topology_tier: Optional[int],
        interpolation_method: str = "alpha_beta",
    ) -> Optional[Tuple[float, bool]]:
        """Shared comm CSV query with interpolation fallback.

        Tries exact match first. On miss, interpolates linearly on message_bytes
        (num_devices + topology_tier remain exact). Interpolation is default
        behavior because message_bytes is continuous and exact match rarely works.

        Returns (latency_us, is_interpolated) or None on miss.
        Sets self.last_miss_reason on failure.
        """
        df = self._load_csv(kernel_type)
        if df is None:
            self.last_miss_reason = "csv_not_found"
            return None

        required_cols = {"message_bytes", "num_devices"}
        if not required_cols.issubset(df.columns):
            logger.debug(
                "MISS (comm) %s: CSV missing columns %s, need microbenchmark format",
                kernel_type,
                required_cols - set(df.columns),
            )
            self.last_miss_reason = "csv_format_raw"
            return None

        lat_col = self._latency_col(df)

        # --- Exact match ---
        mask = (df["message_bytes"] == message_bytes) & (df["num_devices"] == num_devices)
        if topology_tier is not None and "topology_tier" in df.columns:
            mask = mask & (df["topology_tier"] == topology_tier)

        matched = df[mask]
        if not matched.empty:
            for _, row in matched.iterrows():
                latency = self._row_latency_value(row, lat_col)
                if latency is not None:
                    return (latency, False)

        # --- Interpolation fallback: bracket message_bytes ---
        device_mask = df["num_devices"] == num_devices
        if topology_tier is not None and "topology_tier" in df.columns:
            device_mask = device_mask & (df["topology_tier"] == topology_tier)
        candidates = df[device_mask]
        if not candidates.empty:
            candidates = candidates.copy()
            candidates["_effective_latency_us"] = [
                self._row_latency_value(row, lat_col) for _, row in candidates.iterrows()
            ]
            candidates = candidates.dropna(subset=["_effective_latency_us"])

        if candidates.empty:
            logger.debug(
                "MISS (comm) %s: no rows for num_devices=%d, topology_tier=%s",
                kernel_type,
                num_devices,
                topology_tier,
            )
            self.last_miss_reason = "shape_mismatch"
            return None

        mb_values = candidates["message_bytes"].values
        below = mb_values[mb_values <= message_bytes]
        above = mb_values[mb_values >= message_bytes]

        if len(below) == 0 or len(above) == 0:
            logger.debug(
                "MISS (comm) %s: message_bytes=%d outside range [%d, %d]",
                kernel_type,
                message_bytes,
                int(mb_values.min()),
                int(mb_values.max()),
            )
            self.last_miss_reason = "shape_mismatch"
            return None

        mb_lo, mb_hi = int(below.max()), int(above.min())
        lat_lo = float(candidates.loc[candidates["message_bytes"] == mb_lo, "_effective_latency_us"].iloc[0])
        if mb_lo == mb_hi:
            return (lat_lo, False)  # degenerate bracket = exact

        lat_hi = float(candidates.loc[candidates["message_bytes"] == mb_hi, "_effective_latency_us"].iloc[0])

        # Alpha-beta interpolation: comm latency = alpha + message_bytes / bandwidth
        # Fit from ALL candidate data points (least-squares) rather than just the
        # bracket endpoints. This gives a global alpha-beta model for this
        # (num_devices, topology_tier) group, which handles the latency-dominated →
        # bandwidth-dominated transition more accurately than piecewise linear.
        all_mb = candidates["message_bytes"].values.astype(np.float64)
        all_lat = candidates["_effective_latency_us"].values.astype(np.float64)

        if interpolation_method == "bracket_linear":
            alpha = (message_bytes - mb_lo) / (mb_hi - mb_lo)
            interpolated = lat_lo + alpha * (lat_hi - lat_lo)
        elif interpolation_method == "alpha_beta" and len(all_mb) >= 2:
            A = np.column_stack([np.ones_like(all_mb), all_mb])
            params, _, _, _ = np.linalg.lstsq(A, all_lat, rcond=None)
            interpolated = float(params[0] + params[1] * message_bytes)
        elif interpolation_method == "alpha_beta":
            # Fallback: single-point, use that value
            interpolated = float(all_lat[0])
        else:
            self.last_miss_reason = "invalid_comm_interpolation_method"
            return None

        # Clamp to bracket bounds (safety: don't go below lower or above upper)
        interpolated = max(min(lat_lo, lat_hi), min(interpolated, max(lat_lo, lat_hi)))

        logger.debug(
            "HIT (comm interpolated) %s: message_bytes=%d between "
            "[%d (%.1fus), %d (%.1fus)] → %.1fus (alpha-beta fit from %d points)",
            kernel_type,
            message_bytes,
            mb_lo,
            lat_lo,
            mb_hi,
            lat_hi,
            interpolated,
            len(all_mb),
        )
        return (interpolated, True)

    # ---- Main lookup ----

    def lookup(self, op_invoke_info: "OpInvokeInfo") -> Optional[QueryResult]:
        """Query perf data for an op.

        Dispatch logic:
          func_name -> op_mapping.yaml
            - not found -> return None
            - composite == true -> _lookup_composite()
            - category == "communication" -> _lookup_comm()
            - query_mode == "attention_special" -> _lookup_attention()
            - query_mode == "elementwise" -> _lookup_elementwise()
            - query_mode == "moe_fused" -> _lookup_moe()
            - zero_cost == true -> return QueryResult(0.0)
            - accepted_miss -> return QueryResult(0.0) with note
            - default -> _lookup_compute()

        Extension point: to add a new query_mode, add a branch below
        (before the zero_cost check) and implement _lookup_<mode>().
        See §14 in OP_PLUGIN_MAPPING_TUTORIAL.md for the full SOP.
        """
        self.last_shape_match_info = None
        func_str = _normalize_func_name(op_invoke_info.func)
        mappings = self._op_mapping.get("operator_mappings", {})
        mapping = mappings.get(func_str)
        if mapping is None:
            self.last_miss_reason = "unmapped"
            self.last_shape_match_info = ShapeMatchInfo(
                simulation_shapes=[list(s) for s, _ in self._extract_tensor_inputs(op_invoke_info)],
                kernel_shapes=[],
                shape_match_rule="unmapped",
            )
            return None

        # Composite ops: try decomposition via sub_kernels, else skip
        if mapping.get("composite"):
            return self._lookup_composite(op_invoke_info, mapping)
        if mapping.get("category") == "communication":
            return self._lookup_comm(op_invoke_info, mapping)
        if mapping.get("query_mode") == "attention_special":
            return self._lookup_attention(op_invoke_info, mapping)
        if mapping.get("query_mode") == "elementwise":
            return self._lookup_elementwise(op_invoke_info, mapping)
        if mapping.get("query_mode") == "moe_fused":
            return self._lookup_moe(op_invoke_info, mapping)
        if mapping.get("query_mode") == "mtp_projection":
            return self._lookup_mtp_projection(op_invoke_info, mapping)

        # Zero-cost ops: shape-only operations with no kernel execution
        if mapping.get("zero_cost"):
            _zc_sim_shapes = [list(s) for s, _ in self._extract_tensor_inputs(op_invoke_info)]
            _zc_shape_info = ShapeMatchInfo(
                simulation_shapes=_zc_sim_shapes,
                kernel_shapes=[],
                shape_match_rule="zero_cost",
            )
            self.last_shape_match_info = _zc_shape_info
            return QueryResult(
                latency_us=0.0,
                confidence=1.0,
                source=QuerySource.MEASURED,
                details={
                    "kernel_type": mapping.get("kernel_type", ""),
                    "zero_cost": True,
                },
                shape_match_info=_zc_shape_info,
            )

        # Accepted MISS: TC op has no standalone NPU kernel — its latency
        # is absorbed into another fused kernel (e.g., DFC, KvRmsNormRopeCache).
        # Treated like zero_cost (latency=0, counts as HIT) with documentation.
        accepted = mapping.get("accepted_miss")
        if accepted:
            return QueryResult(
                latency_us=0.0,
                confidence=1.0,
                source=QuerySource.MEASURED,
                details={
                    "kernel_type": "accepted_miss",
                    "zero_cost": True,
                    "note": accepted,
                },
                shape_match_info=ShapeMatchInfo(
                    simulation_shapes=[list(s) for s, _ in self._extract_tensor_inputs(op_invoke_info)],
                    kernel_shapes=[],
                    shape_match_rule="accepted_miss",
                ),
            )

        return self._lookup_compute(op_invoke_info, mapping)

    # ---- Composite op lookup ----

    def _lookup_composite(self, op_invoke_info: "OpInvokeInfo", mapping: dict) -> Optional[QueryResult]:
        """Decompose composite ops and sum sub-kernel latencies.

        For MLA/MLAPO: uses registered decomposer to derive sub-kernel shapes,
        then queries each sub-kernel individually.
        For MC2 (matmul+comm): queries both compute and comm sub-kernels.
        Returns None if any required sub-kernel misses.
        """
        # Check for registered decomposer (MLA/MLAPO)
        func_str = _normalize_func_name(op_invoke_info.func)
        decomposer = COMPOSITE_DECOMPOSERS.get(func_str)
        if decomposer is not None:
            return self._lookup_composite_decomposed(op_invoke_info, mapping, decomposer)

        # Generic composite path (MC2 etc.)
        sub_kernels = mapping.get("sub_kernels", [])
        if not sub_kernels:
            self.last_miss_reason = "no_sub_kernels"
            return None

        tc_inputs = self._extract_tensor_inputs(op_invoke_info)

        # tc_input_count truncation (same as _lookup_compute):
        # quant MC2 ops have 6 tensor args but CSV only needs x + w
        tc_input_count = mapping.get("tc_input_count")
        if tc_input_count is not None:
            tc_inputs = tc_inputs[:tc_input_count]

        # --- Compute sub-kernels: try each until one matches ---
        compute_kernels = [k for k in sub_kernels if not k.startswith("hcom_")]
        compute_hit = self._find_compute_match(compute_kernels, tc_inputs, tc_input_count)

        if compute_hit is None:
            return None

        compute_latency = compute_hit.latency_us
        compute_kernel_hit = compute_hit.kernel_type
        compute_csv_shapes = compute_hit.shape_match_info.kernel_shapes if compute_hit.shape_match_info else []
        compute_rule = compute_hit.shape_match_info.shape_match_rule if compute_hit.shape_match_info else "unknown"
        simulation_shapes = [list(s) for s, _ in tc_inputs]

        # --- Communication sub-kernels ---
        # Convention: comm sub_kernels must use hcom_ prefix (lowercase).
        # CamelCase names (HcomAllReduce) are graph-compiled variants and
        # should only appear in alternate_kernel_types.
        # NOTE: _lookup_comm_for_composite assumes matmul+comm arg layout:
        #   args[0]=mat1, args[1]=mat2, args[-1]=rank_group.
        # This holds for all current MC2 variants (matmul_all_reduce,
        # static_quant_linear_all_reduce, fp8_linear_all_reduce, etc.).
        # If a future composite op has a different arg layout, this will
        # need per-op dispatch or a mapping-driven arg index scheme.
        comm_specs = [
            {
                "kernel_type": kernel_type,
                "message_bytes_mode": "full_output",
                "group_type": None,
                "interpolation_method": "alpha_beta",
            }
            for kernel_type in sub_kernels
            if kernel_type.startswith("hcom_")
        ]
        active_comm_variant = None
        if config.compilation.passes.enable_sequence_parallel:
            variant = mapping.get("runtime_comm_variants", {}).get("sequence_parallel")
            if variant:
                source_kernel_type = variant.get("source_kernel_type")
                target_kernel_type = variant.get("kernel_type")
                message_bytes_mode = variant.get("message_bytes_mode", "full_output")
                group_type = variant.get("group_type")
                interpolation_method = variant.get("interpolation_method", "alpha_beta")
                if (
                    not isinstance(source_kernel_type, str)
                    or not isinstance(target_kernel_type, str)
                    or not target_kernel_type.startswith("hcom_")
                    or message_bytes_mode not in {"full_output", "per_rank_output"}
                    or group_type not in {None, "tensor_parallel"}
                    or interpolation_method not in {"alpha_beta", "bracket_linear"}
                ):
                    self.last_miss_reason = "invalid_runtime_comm_variant"
                    return None
                replaced = False
                for spec in comm_specs:
                    if spec["kernel_type"] == source_kernel_type:
                        spec["kernel_type"] = target_kernel_type
                        spec["message_bytes_mode"] = message_bytes_mode
                        spec["group_type"] = group_type
                        spec["interpolation_method"] = interpolation_method
                        replaced = True
                if not replaced:
                    self.last_miss_reason = "runtime_comm_variant_source_missing"
                    return None
                active_comm_variant = "sequence_parallel"

        comm_latency = 0.0
        has_comm = False
        has_interpolated_comm = False
        communication_sub_kernels = []
        sub_kernel_durations = [(compute_kernel_hit, round(compute_latency, 2))]
        sub_kernel_shapes_info = [
            SubKernelShapeInfo(
                kernel_type=compute_kernel_hit,
                simulation_shapes=simulation_shapes,
                kernel_shapes=compute_csv_shapes,
                shape_match_rule=compute_rule,
            )
        ]
        for comm_spec in comm_specs:
            kernel_type = comm_spec["kernel_type"]
            has_comm = True
            comm_result = self._lookup_comm_for_composite(
                op_invoke_info,
                kernel_type,
                message_bytes_mode=comm_spec["message_bytes_mode"],
                group_type=comm_spec["group_type"],
                interpolation_method=comm_spec["interpolation_method"],
            )
            if comm_result is None:
                self.last_miss_reason = "comm_sub_kernel_miss"
                return None
            lat, comm_details = comm_result
            has_interpolated_comm = has_interpolated_comm or bool(comm_details["interpolated"])
            comm_latency += lat
            sub_kernel_durations.append((kernel_type, round(lat, 2)))
            communication_sub_kernels.append({"kernel_type": kernel_type, **comm_details})
            # Comm ops don't go through _query_by_shapes — record with empty shapes
            sub_kernel_shapes_info.append(
                SubKernelShapeInfo(
                    kernel_type=kernel_type,
                    simulation_shapes=[],
                    kernel_shapes=[],
                    shape_match_rule="comm",
                )
            )
            logger.debug("HIT (composite comm) %s: %.1f us", kernel_type, lat)

        return QueryResult(
            latency_us=compute_latency + comm_latency,
            confidence=0.9 if has_comm else 0.8,
            source=QuerySource.INTERPOLATED if has_interpolated_comm else QuerySource.MEASURED,
            details={
                "kernel_type": compute_kernel_hit,
                "sub_kernel_durations": sub_kernel_durations,
                "composite": True,
                "note": ("compute + comm sub-kernels" if has_comm else "compute sub-kernel only"),
                "compute_latency_details": compute_hit.details,
                "communication_variant": active_comm_variant,
                "communication_sub_kernels": communication_sub_kernels,
            },
            sub_kernel_shapes=sub_kernel_shapes_info,
        )

    def _build_composite_runtime_mapping(self, mapping: dict) -> dict:
        """Inject the runtime TP/SP context required by composite decomposers.

        Exact lookup and interpolation fallback must decompose the same
        ``OpInvokeInfo`` with the same runtime mapping. Otherwise a later
        chunked-prefill forward can use TP/SP-projected shapes for exact lookup
        and then silently fall back to unprojected global shapes while
        interpolating the same composite operator.
        """
        runtime_mapping = dict(mapping)
        runtime_mapping["_runtime_tp_size"] = self.tp_size
        runtime_mapping["_runtime_sequence_parallel"] = bool(config.compilation.passes.enable_sequence_parallel)
        return runtime_mapping

    def _lookup_composite_decomposed(
        self,
        op_invoke_info: "OpInvokeInfo",
        mapping: dict,
        decomposer: Callable,
    ) -> Optional[QueryResult]:
        """Query composite op using registered decomposer (MLA/MLAPO).

        Calls the decomposer to get SubKernelSpec list, then queries each
        sub-kernel via _find_compute_match or _query_by_attn_params.

        Returns:
          - MEASURED if all sub-kernels hit.
          - PARTIAL if non-attention sub-kernels miss but some hit (accumulated latency).
          - PARTIAL on a required attention miss only when the versioned mapping
            explicitly opts in; otherwise None forces analytic fallback. This
            preserves exact child evidence without disguising the missing
            attention latency as a full hit.
          - None if all sub-kernels miss.
        """
        runtime_mapping = self._build_composite_runtime_mapping(mapping)
        specs = decomposer(op_invoke_info, runtime_mapping)
        if not specs:
            visible_regime = mapping.get("decomposer_options", {}).get("visible_kernel_regime", {})
            if visible_regime.get("requires_sequence_parallel") and not runtime_mapping["_runtime_sequence_parallel"]:
                self.last_miss_reason = (
                    "structural_mismatch:mlapo_quant:opaque_mla_preprocess:"
                    "visible_kernel_regime_requires_sequence_parallel"
                )
            else:
                self.last_miss_reason = "decompose_failed"
            return None

        total_latency = 0.0
        hit_kernels = []
        sub_kernel_durations = []
        sub_kernel_shapes_list = []
        missed_kernels = []
        has_interpolated = False

        for spec in specs:
            kernel_types = [spec.kernel_type] + (spec.alternate_kernel_types or [])

            if spec.query_mode == "mlapo_preprocess" and spec.runtime_params:
                hit = self._query_mlapo_preprocess(kernel_types, spec.runtime_params, spec.dtype)
                if hit is not None:
                    total_latency += hit.latency_us
                    hit_kernels.append(hit.kernel_type)
                    sub_kernel_durations.append((hit.kernel_type, round(hit.latency_us, 2)))
                    has_interpolated = has_interpolated or bool(hit.details.get("interpolated"))
                    sub_kernel_shapes_list.append(self._candidate_sub_kernel_shape(hit))
                else:
                    missed_kernels.append(spec.kernel_type)
            elif spec.query_mode == "attention" and spec.attention_params:
                result = self._query_by_attn_params(kernel_types, spec.attention_params, spec.dtype)
                if result is not None:
                    # attention path result is (lat, kernel_type) 2-tuple — no csv_shapes
                    lat, matched_kernel = result
                    total_latency += lat
                    hit_kernels.append(matched_kernel)
                    sub_kernel_durations.append((matched_kernel, round(lat, 2)))
                    sub_kernel_shapes_list.append(
                        SubKernelShapeInfo(
                            kernel_type=matched_kernel,
                            simulation_shapes=[],
                            kernel_shapes=[],
                            shape_match_rule="attention",
                        )
                    )
                else:
                    missed_kernels.append(spec.kernel_type)
                    # Attention sub-kernel miss: SFA/FIA CSV not available.
                    # Return None so analytic fallback produces a complete estimate
                    # rather than a PARTIAL result that silently drops attention latency.
                    semantic_reason = self.last_miss_reason or "semantic_key_mismatch"
                    self.last_miss_reason = (
                        f"attention_sub_kernel_miss:{spec.kernel_type}:{semantic_reason};"
                        f"semantic_key={spec.attention_params};dtype={spec.dtype}"
                    )
                    logger.warning(
                        "attention sub-kernel miss for %s (%s): reason=%s semantic_key=%s dtype=%s; "
                        "falling back from the incomplete measured decomposition.",
                        spec.kernel_type,
                        _normalize_func_name(op_invoke_info.func),
                        semantic_reason,
                        spec.attention_params,
                        spec.dtype,
                    )
                    return None
            elif spec.query_mode == "cache_postprocess" and spec.cache_params:
                hit = self._query_cache_postprocess(kernel_types, spec.cache_params, spec.dtype)
                if hit is not None:
                    total_latency += hit.latency_us
                    hit_kernels.append(hit.kernel_type)
                    sub_kernel_durations.append((hit.kernel_type, round(hit.latency_us, 2)))
                    sub_kernel_shapes_list.append(
                        SubKernelShapeInfo(
                            kernel_type=hit.kernel_type,
                            simulation_shapes=(hit.shape_match_info.simulation_shapes if hit.shape_match_info else []),
                            kernel_shapes=(hit.shape_match_info.kernel_shapes if hit.shape_match_info else []),
                            shape_match_rule=(
                                hit.shape_match_info.shape_match_rule if hit.shape_match_info else "unknown"
                            ),
                        )
                    )
                else:
                    missed_kernels.append(spec.kernel_type)
            elif spec.query_mode == "scatter_cache_write" and spec.cache_params:
                hit = self._query_scatter_cache_write(kernel_types, spec.cache_params, spec.dtype)
                if hit is not None:
                    total_latency += hit.latency_us
                    hit_kernels.append(hit.kernel_type)
                    sub_kernel_durations.append((hit.kernel_type, round(hit.latency_us, 2)))
                    sub_kernel_shapes_list.append(
                        SubKernelShapeInfo(
                            kernel_type=hit.kernel_type,
                            simulation_shapes=(hit.shape_match_info.simulation_shapes if hit.shape_match_info else []),
                            kernel_shapes=(hit.shape_match_info.kernel_shapes if hit.shape_match_info else []),
                            shape_match_rule=(
                                hit.shape_match_info.shape_match_rule if hit.shape_match_info else "unknown"
                            ),
                        )
                    )
                else:
                    missed_kernels.append(spec.kernel_type)
            else:
                profile_dtypes = spec.input_dtypes or [spec.dtype] * len(spec.input_shapes)
                torch_dtypes = []
                for profile_dtype in profile_dtypes:
                    torch_dtype = next((k for k, v in DTYPE_MAP.items() if v == profile_dtype), None)
                    if torch_dtype is None:
                        break
                    torch_dtypes.append(torch_dtype)
                if len(torch_dtypes) != len(spec.input_shapes):
                    logger.debug(
                        "Unknown or incomplete dtypes %s for sub-kernel %s, skipping",
                        profile_dtypes,
                        spec.kernel_type,
                    )
                    missed_kernels.append(spec.kernel_type)
                    continue
                tc_inputs = list(zip(spec.input_shapes, torch_dtypes))
                hit = self._find_compute_match(
                    kernel_types,
                    tc_inputs,
                    spec.tc_input_count,
                    auto_truncate=True,
                )
                if hit is not None:
                    total_latency += hit.latency_us
                    hit_kernels.append(hit.kernel_type)
                    sub_kernel_durations.append((hit.kernel_type, round(hit.latency_us, 2)))
                    sub_kernel_shapes_list.append(
                        SubKernelShapeInfo(
                            kernel_type=hit.kernel_type,
                            simulation_shapes=[list(s) for s in spec.input_shapes],
                            kernel_shapes=(hit.shape_match_info.kernel_shapes if hit.shape_match_info else []),
                            shape_match_rule=(
                                hit.shape_match_info.shape_match_rule if hit.shape_match_info else "unknown"
                            ),
                        )
                    )
                else:
                    missed_kernels.append(spec.kernel_type)
                    if spec.is_attention:
                        # Attention sub-kernel matched in compute mode (e.g. SFA,
                        # whose CSV has no avg_seq_len column). A miss here must
                        # still force analytic fallback — returning PARTIAL would
                        # silently drop the dominant attention latency, the exact
                        # failure mode this composite path guards against.
                        shape_reason = self.last_miss_reason or "shape_mismatch"
                        self.last_miss_reason = (
                            f"attention_sub_kernel_miss:{spec.kernel_type}:{shape_reason};"
                            f"shapes={spec.input_shapes};dtype={profile_dtypes}"
                        )
                        logger.warning(
                            "required sub-kernel miss for %s (%s): reason=%s shapes=%s dtype=%s; "
                            "falling back from the incomplete measured decomposition.",
                            spec.kernel_type,
                            _normalize_func_name(op_invoke_info.func),
                            shape_reason,
                            spec.input_shapes,
                            profile_dtypes,
                        )
                        return None

        if missed_kernels:
            self.last_miss_reason = f"sub_kernel_miss:{','.join(missed_kernels)}"
            if mapping.get("require_all_sub_kernels"):
                # A visible MLAPO regime is only valid when the complete
                # semantic sequence resolves. Returning its few coincidental
                # hits as PARTIAL can double-count kernels owned by another
                # composite (notably Decode DSA).
                return None
            if not hit_kernels:
                # All sub-kernels missed → return None to allow analytic fallback
                return None
            confidence = len(hit_kernels) / len(specs) if specs else 0.0
            return QueryResult(
                latency_us=total_latency,
                confidence=confidence,
                source=QuerySource.PARTIAL,
                details={
                    "hit_kernels": hit_kernels,
                    "missed_kernels": missed_kernels,
                    "sub_kernel_durations": sub_kernel_durations,
                    "composite": True,
                    "partial": True,
                },
                sub_kernel_shapes=sub_kernel_shapes_list,
            )

        logger.debug(
            "HIT (composite decomposed) %s: sub_kernels=%s, total=%.1f us",
            _normalize_func_name(op_invoke_info.func),
            hit_kernels,
            total_latency,
        )
        return QueryResult(
            latency_us=total_latency,
            confidence=0.75 if has_interpolated else 0.8,
            source=QuerySource.INTERPOLATED if has_interpolated else QuerySource.MEASURED,
            details={
                "kernel_type": ",".join(hit_kernels),
                "sub_kernel_durations": sub_kernel_durations,
                "composite": True,
                "note": "decomposed sub-kernels",
            },
            sub_kernel_shapes=sub_kernel_shapes_list,
        )

    def _query_mlapo_preprocess(
        self,
        kernel_types: List[str],
        params: Dict[str, Any],
        dtype_str: str,
    ) -> Optional[Candidate]:
        """Query the opaque GLM-5 decode MLAPO kernel by execution semantics."""
        int_fields = (
            "num_tokens",
            "hidden_size",
            "local_num_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "block_size",
        )
        str_fields = ("cache_mode", "quant_mode", "weight_format")
        bool_fields = ("enable_inner_out", "weight_quantized")
        if any(params.get(field) is None for field in (*int_fields, *str_fields, *bool_fields)):
            self.last_miss_reason = "semantic_context_missing:target:mlapo_preprocess"
            return None
        target = {field: int(params[field]) for field in int_fields}
        target.update({field: str(params[field]) for field in str_fields})
        target.update({field: bool(params[field]) for field in bool_fields})
        if (
            not 1 <= target["num_tokens"] <= _MLA_PREPROCESS_MAX_DECODE_TOKENS
            or min(target[field] for field in int_fields) <= 0
        ):
            self.last_miss_reason = "semantic_key_mismatch:mlapo_preprocess:num_tokens_out_of_range"
            return None
        if _build_mla_preprocess_expected_shapes(**{field: target[field] for field in int_fields}) is None:
            self.last_miss_reason = "semantic_key_mismatch:mlapo_preprocess:fractal_nz_alignment"
            return None
        if dtype_str != "DT_BF16":
            self.last_miss_reason = "semantic_key_mismatch:mlapo_preprocess:dtype"
            return None

        runtime_columns = {field: f"Runtime {field}" for field in (*int_fields, *str_fields, *bool_fields)}
        candidates: list[tuple[int, float, pd.Series, str]] = []
        missing_columns: set[str] = set()
        for kernel_type in kernel_types:
            df = self._load_csv(kernel_type)
            if df is None or df.empty:
                continue
            required_columns = {
                *runtime_columns.values(),
                "Runtime metadata_completeness",
                "Input Shapes",
                "Input Data Types",
                "Input Formats",
                "Output Shapes",
                "Output Data Types",
                "Output Formats",
            }
            absent = required_columns - set(df.columns)
            if absent:
                missing_columns.update(absent)
                continue
            preferred_latency_col = self._latency_col(df)
            for _, row in df.iterrows():
                completeness = (_optional_runtime_str(row.get("Runtime metadata_completeness")) or "legacy").lower()
                if completeness == "legacy":
                    continue
                try:
                    row_ints = {field: int(row[runtime_columns[field]]) for field in int_fields}
                except (TypeError, ValueError):
                    continue
                if any(row_ints[field] != target[field] for field in int_fields if field != "num_tokens"):
                    continue
                if any(
                    (_optional_runtime_str(row.get(runtime_columns[field])) or "") != target[field]
                    for field in str_fields
                ):
                    continue
                row_bools = {
                    field: (_optional_runtime_str(row.get(runtime_columns[field])) or "").lower() in {"true", "1"}
                    for field in bool_fields
                }
                if any(row_bools[field] != target[field] for field in bool_fields):
                    continue
                input_dtypes = _parse_str_list(str(row.get("Input Data Types", "")))
                input_formats = _parse_str_list(str(row.get("Input Formats", "")))
                output_dtypes = _parse_str_list(str(row.get("Output Data Types", "")))
                output_formats = _parse_str_list(str(row.get("Output Formats", "")))
                num_tokens = row_ints["num_tokens"]
                hidden_size = row_ints["hidden_size"]
                local_heads = row_ints["local_num_heads"]
                q_lora_rank = row_ints["q_lora_rank"]
                kv_lora_rank = row_ints["kv_lora_rank"]
                nope_dim = row_ints["qk_nope_head_dim"]
                rope_dim = row_ints["qk_rope_head_dim"]
                block_size = row_ints["block_size"]
                expected_shapes = _build_mla_preprocess_expected_shapes(
                    num_tokens=num_tokens,
                    hidden_size=hidden_size,
                    local_num_heads=local_heads,
                    q_lora_rank=q_lora_rank,
                    kv_lora_rank=kv_lora_rank,
                    qk_nope_head_dim=nope_dim,
                    qk_rope_head_dim=rope_dim,
                    block_size=block_size,
                )
                if expected_shapes is None:
                    continue
                expected_input_shapes, expected_output_shapes = expected_shapes
                input_shapes = _parse_shape_str(str(row.get("Input Shapes", "")))
                output_shapes = _parse_shape_str(str(row.get("Output Shapes", "")))
                if (
                    len(input_dtypes) != _MLA_PREPROCESS_INPUT_COUNT
                    or len(input_formats) != _MLA_PREPROCESS_INPUT_COUNT
                    or len(output_dtypes) != _MLA_PREPROCESS_OUTPUT_COUNT
                    or len(output_formats) != _MLA_PREPROCESS_OUTPUT_COUNT
                    or input_shapes != expected_input_shapes
                    or output_shapes != expected_output_shapes
                    or input_dtypes[0] != dtype_str
                    or input_dtypes[1] not in {"DT_INT8", "INT8"}
                    or input_dtypes[5] not in {"DT_INT8", "INT8"}
                    or input_formats[1] != target["weight_format"]
                    or input_formats[5] != target["weight_format"]
                    or any(dtype != dtype_str for dtype in output_dtypes)
                    or any(fmt != "ND" for fmt in output_formats)
                ):
                    continue
                latency = self._row_latency_value(row, preferred_latency_col)
                if latency is None:
                    continue
                candidates.append((row_ints["num_tokens"], latency, row, kernel_type))

        if not candidates:
            if missing_columns:
                self.last_miss_reason = "semantic_context_missing:csv:" + ",".join(sorted(missing_columns))
            else:
                self.last_miss_reason = "semantic_key_mismatch:mlapo_preprocess"
            return None

        target_tokens = target["num_tokens"]

        def shape_info(row: pd.Series, rule: str) -> ShapeMatchInfo:
            csv_shapes = [list(shape) for shape in _parse_shape_str(str(row.get("Input Shapes", "")))]
            return ShapeMatchInfo(
                simulation_shapes=[[target_tokens, target["hidden_size"]]],
                kernel_shapes=csv_shapes,
                shape_match_rule=rule,
            )

        exact = [candidate for candidate in candidates if candidate[0] == target_tokens]
        if exact:
            _, latency, row, kernel_type = exact[0]
            return Candidate(
                latency_us=latency,
                kernel_type=kernel_type,
                confidence=0.9,
                details={"query_mode": "mlapo_preprocess", "interpolated": False, "semantic_key": target},
                shape_match_info=shape_info(row, "mlapo_runtime_exact"),
            )

        lower = max(
            (candidate for candidate in candidates if candidate[0] < target_tokens),
            default=None,
            key=lambda item: item[0],
        )
        upper = min(
            (candidate for candidate in candidates if candidate[0] > target_tokens),
            default=None,
            key=lambda item: item[0],
        )
        if lower is None or upper is None or lower[0] == upper[0]:
            self.last_miss_reason = "shape_mismatch:mlapo_preprocess:bounded_interpolation_only"
            return None
        ratio = (target_tokens - lower[0]) / (upper[0] - lower[0])
        latency = lower[1] + ratio * (upper[1] - lower[1])
        return Candidate(
            latency_us=latency,
            kernel_type=lower[3],
            confidence=0.75,
            details={
                "query_mode": "mlapo_preprocess",
                "interpolated": True,
                "semantic_key": target,
                "bracket_tokens": [lower[0], upper[0]],
            },
            shape_match_info=shape_info(lower[2], "mlapo_num_tokens_bounded_interpolation"),
        )

    def _query_by_attn_params(
        self,
        kernel_types: List[str],
        params: Dict[str, Any],
        dtype_str: str,
    ) -> Optional[Tuple[float, str]]:
        """Shared attention query core: iterate kernel_types, match FIA params.

        params must contain: q_shape_3d (tuple), avg_seq_len (int).
        Optional: sparse_mode (int), num_kv_heads (int).

        Returns (latency_us, matched_kernel_type) or None.
        """
        q_shape_3d = params.get("q_shape_3d")
        target_avg_seq = params.get("avg_seq_len")
        if q_shape_3d is None:
            self.last_miss_reason = "semantic_context_missing:target:q_shape_3d"
            return None
        if target_avg_seq is None:
            self.last_miss_reason = "semantic_context_missing:target:avg_seq_len"
            return None

        target_sparse = params.get("sparse_mode")
        target_kv_heads = params.get("num_kv_heads")
        target_layout = params.get("input_layout")
        target_phase = params.get("phase")
        target_topk = params.get("topk")
        target_block_size = params.get("block_size")
        target_actual_query_values = params.get("actual_seq_lengths_values")
        target_actual_kv_values = params.get("actual_seq_lengths_kv_values")
        target_valid_blocks = params.get("block_table_valid_blocks")
        target_num_heads = params.get("num_heads")
        target_cache_layout = params.get("cache_layout")
        target_kv_cache_mode = params.get("kv_cache_mode")
        target_sparse_block_size = params.get("sparse_block_size")
        target_sparse_indices_pattern = params.get("sparse_indices_pattern")
        target_sparse_indices_valid_count = params.get("sparse_indices_valid_count")
        required_context_fields = set(params.get("required_context_fields", ()))
        target_values = {
            "avg_seq_len": target_avg_seq,
            "sparse_mode": target_sparse,
            "num_kv_heads": target_kv_heads,
            "input_layout": target_layout,
            "phase": target_phase,
            "topk": target_topk,
            "block_size": target_block_size,
            "actual_seq_lengths_values": target_actual_query_values,
            "actual_seq_lengths_kv_values": target_actual_kv_values,
            "block_table_valid_blocks": target_valid_blocks,
            "num_heads": target_num_heads,
            "cache_layout": target_cache_layout,
            "kv_cache_mode": target_kv_cache_mode,
            "sparse_block_size": target_sparse_block_size,
            "sparse_indices_pattern": target_sparse_indices_pattern,
            "sparse_indices_valid_count": target_sparse_indices_valid_count,
        }
        missing_target_values = sorted(name for name in required_context_fields if target_values.get(name) is None)
        if missing_target_values:
            self.last_miss_reason = "semantic_context_missing:target:" + ",".join(missing_target_values)
            return None
        tc_N, tc_D = q_shape_3d[1], q_shape_3d[2]
        head_dim = tc_D

        # Per-CSV column detection cache: detected once per kernel_type,
        # reused for all rows in that CSV.
        _col_cache: Dict[str, Optional[_FiaColInfo]] = {}
        missing_csv_columns: set[str] = set()

        def _detect_columns(row: pd.Series, kt: str) -> Optional["_FiaColInfo"]:
            """Detect FIA column names from the first row of a kernel CSV."""
            if kt in _col_cache:
                return _col_cache[kt]
            cols = row.index
            if "Runtime avg_seq_len" in cols:
                avg_seq_col = "Runtime avg_seq_len"
            elif "avg_seq_len" in cols:
                avg_seq_col = "avg_seq_len"
            else:
                missing_csv_columns.add("Runtime avg_seq_len")
                _col_cache[kt] = None
                return None
            if "Input Shapes" not in cols:
                _col_cache[kt] = None
                return None
            info = _FiaColInfo(
                avg_seq_col=avg_seq_col,
                has_sparse="Runtime sparse_mode" in cols,
                has_kv_heads="Runtime num_key_value_heads" in cols,
                has_layout="Runtime input_layout" in cols,
                has_phase="Runtime phase" in cols,
                has_topk="Runtime topk" in cols,
                has_block_size="Runtime block_size" in cols,
                has_actual_query_values="Runtime actual_seq_lengths_values" in cols,
                has_actual_kv_values="Runtime actual_seq_lengths_kv_values" in cols,
                has_valid_blocks="Runtime block_table_valid_blocks" in cols,
                has_num_heads="Runtime num_heads" in cols,
                has_cache_layout="Runtime cache_layout" in cols,
                has_kv_cache_mode="Runtime kv_cache_mode" in cols,
                has_sparse_block_size="Runtime sparse_block_size" in cols,
                has_sparse_indices_pattern="Runtime sparse_indices_pattern" in cols,
                has_sparse_indices_valid_count="Runtime sparse_indices_valid_count" in cols,
            )
            _col_cache[kt] = info
            return info

        def checker(row: pd.Series, kt: str, lat_col: str) -> Optional[Candidate]:
            col_info = _detect_columns(row, kt)
            if col_info is None:
                return None

            required_columns = {
                "sparse_mode": (col_info.has_sparse, "Runtime sparse_mode"),
                "num_kv_heads": (col_info.has_kv_heads, "Runtime num_key_value_heads"),
                "input_layout": (col_info.has_layout, "Runtime input_layout"),
                "phase": (col_info.has_phase, "Runtime phase"),
                "topk": (col_info.has_topk, "Runtime topk"),
                "block_size": (col_info.has_block_size, "Runtime block_size"),
                "actual_seq_lengths_values": (
                    col_info.has_actual_query_values,
                    "Runtime actual_seq_lengths_values",
                ),
                "actual_seq_lengths_kv_values": (
                    col_info.has_actual_kv_values,
                    "Runtime actual_seq_lengths_kv_values",
                ),
                "block_table_valid_blocks": (
                    col_info.has_valid_blocks,
                    "Runtime block_table_valid_blocks",
                ),
                "num_heads": (col_info.has_num_heads, "Runtime num_heads"),
                "cache_layout": (col_info.has_cache_layout, "Runtime cache_layout"),
                "kv_cache_mode": (col_info.has_kv_cache_mode, "Runtime kv_cache_mode"),
                "sparse_block_size": (
                    col_info.has_sparse_block_size,
                    "Runtime sparse_block_size",
                ),
                "sparse_indices_pattern": (
                    col_info.has_sparse_indices_pattern,
                    "Runtime sparse_indices_pattern",
                ),
                "sparse_indices_valid_count": (
                    col_info.has_sparse_indices_valid_count,
                    "Runtime sparse_indices_valid_count",
                ),
            }
            legacy_optional_fields = {
                "actual_seq_lengths_values",
                "actual_seq_lengths_kv_values",
                "block_table_valid_blocks",
                "num_heads",
                "cache_layout",
                "kv_cache_mode",
                "sparse_block_size",
                "sparse_indices_pattern",
                "sparse_indices_valid_count",
            }
            missing = [
                column_name
                for field_name, (present, column_name) in required_columns.items()
                if field_name in required_context_fields and field_name not in legacy_optional_fields and not present
            ]
            if missing:
                missing_csv_columns.update(missing)
                return None

            if pd.isna(row[col_info.avg_seq_col]):
                return None
            csv_avg_seq = int(row[col_info.avg_seq_col])
            if csv_avg_seq < 0:
                return None

            shapes_str = str(row.get("Input Shapes", "")).strip('"')
            csv_q_raw = _parse_fia_q_shape(shapes_str)
            if csv_q_raw is None:
                return None
            csv_q_3d = _normalize_fia_q_shape(csv_q_raw, head_dim)
            if csv_q_3d is None:
                return None

            csv_N, csv_D = csv_q_3d[1], csv_q_3d[2]
            csv_dtypes_str = str(row.get("Input Data Types", ""))
            csv_first_dtype = csv_dtypes_str.split(";")[0].strip() if csv_dtypes_str else ""
            if dtype_str != csv_first_dtype:
                return None
            if tc_N != csv_N or tc_D != csv_D:
                return None

            if col_info.has_sparse and target_sparse is not None:
                sparse_val = row["Runtime sparse_mode"]
                if pd.isna(sparse_val) or int(sparse_val) != target_sparse:
                    return None

            if col_info.has_kv_heads and target_kv_heads is not None:
                kv_heads_val = row["Runtime num_key_value_heads"]
                if pd.isna(kv_heads_val) or int(kv_heads_val) != target_kv_heads:
                    return None

            if col_info.has_layout and target_layout is not None:
                csv_layout = str(row.get("Runtime input_layout", "")).strip()
                if csv_layout and csv_layout != target_layout:
                    return None

            if col_info.has_phase and target_phase is not None:
                csv_phase = str(row.get("Runtime phase", "")).strip().lower()
                if not csv_phase or csv_phase != str(target_phase).lower():
                    return None

            if col_info.has_topk and target_topk is not None:
                topk_value = row.get("Runtime topk")
                if pd.isna(topk_value) or int(topk_value) != int(target_topk):
                    return None

            if col_info.has_block_size and target_block_size is not None:
                block_size_value = row.get("Runtime block_size")
                if pd.isna(block_size_value) or int(block_size_value) != int(target_block_size):
                    return None

            runtime_list_checks = (
                (
                    col_info.has_actual_query_values,
                    "Runtime actual_seq_lengths_values",
                    target_actual_query_values,
                ),
                (
                    col_info.has_actual_kv_values,
                    "Runtime actual_seq_lengths_kv_values",
                    target_actual_kv_values,
                ),
                (
                    col_info.has_valid_blocks,
                    "Runtime block_table_valid_blocks",
                    target_valid_blocks,
                ),
            )
            for present, column, target_value in runtime_list_checks:
                if present and target_value is not None:
                    csv_values = _parse_runtime_int_list_cell(row.get(column))
                    if csv_values is not None and csv_values != list(target_value):
                        return None

            runtime_int_checks = (
                (col_info.has_num_heads, "Runtime num_heads", target_num_heads),
                (
                    col_info.has_sparse_block_size,
                    "Runtime sparse_block_size",
                    target_sparse_block_size,
                ),
                (
                    col_info.has_sparse_indices_valid_count,
                    "Runtime sparse_indices_valid_count",
                    target_sparse_indices_valid_count,
                ),
            )
            for present, column, target_value in runtime_int_checks:
                if present and target_value is not None:
                    csv_value = row.get(column)
                    if not pd.isna(csv_value) and str(csv_value).strip() and int(csv_value) != int(target_value):
                        return None

            runtime_str_checks = (
                (col_info.has_cache_layout, "Runtime cache_layout", target_cache_layout),
                (col_info.has_kv_cache_mode, "Runtime kv_cache_mode", target_kv_cache_mode),
                (
                    col_info.has_sparse_indices_pattern,
                    "Runtime sparse_indices_pattern",
                    target_sparse_indices_pattern,
                ),
            )
            for present, column, target_value in runtime_str_checks:
                if present and target_value is not None:
                    csv_value = _optional_runtime_str(row.get(column))
                    if csv_value is not None and csv_value != str(target_value):
                        return None

            avg_seq_gap = abs(target_avg_seq - csv_avg_seq)
            if avg_seq_gap > _AVG_SEQ_LEN_TOLERANCE:
                return None

            tc_T = q_shape_3d[0]
            csv_T = csv_q_3d[0]
            if tc_T != csv_T and not _is_block_padded(tc_T, csv_T) and not _is_block_padded(csv_T, tc_T):
                return None

            completeness = (_optional_runtime_str(row.get("Runtime metadata_completeness")) or "legacy").lower()
            legacy_penalty = 0.5 if completeness in {"", "legacy"} else 0.0
            return Candidate(
                latency_us=float(row[lat_col]),
                kernel_type=kt,
                confidence=0.9,
                distance=float(avg_seq_gap) + legacy_penalty,
            )

        hit = self._find_candidates(kernel_types, checker, select="nearest")
        if hit is None:
            if missing_csv_columns:
                self.last_miss_reason = "semantic_context_missing:csv:" + ",".join(sorted(missing_csv_columns))
            elif not self.last_miss_reason:
                self.last_miss_reason = "shape_mismatch"
            return None

        logger.debug(
            "HIT (attention) %s: params=%s -> %.1f us (avg_seq_len gap=%.0f)",
            hit.kernel_type,
            params,
            hit.latency_us,
            hit.distance,
        )
        return hit.latency_us, hit.kernel_type

    def _query_cache_postprocess(
        self,
        kernel_types: List[str],
        params: Dict[str, Any],
        dtype_str: str,
    ) -> Optional[Candidate]:
        """Match a fused KV postprocess while ignoring only cache-pool capacity.

        The physical pool dim0 is an allocation choice outside the operator's
        semantic work. All token, feature, block-size, head, dtype, and format
        dimensions remain exact query keys.
        """
        required = ("tokens", "kv_proj_dim", "kv_lora_rank", "rope_dim", "block_size")
        if any(params.get(name) is None for name in required):
            self.last_miss_reason = "semantic_context_missing:target:cache_postprocess"
            return None

        tokens = int(params["tokens"])
        kv_proj_dim = int(params["kv_proj_dim"])
        kv_lora_rank = int(params["kv_lora_rank"])
        rope_dim = int(params["rope_dim"])
        block_size = int(params["block_size"])
        if min(tokens, kv_proj_dim, kv_lora_rank, rope_dim, block_size) <= 0:
            self.last_miss_reason = "semantic_key_mismatch:cache_postprocess"
            return None

        expected_prefix = [
            (tokens, 1, 1, kv_proj_dim),
            (kv_lora_rank,),
            (tokens, 1, 1, rope_dim),
            (tokens, 1, 1, rope_dim),
            (tokens,),
        ]
        expected_dtypes = [dtype_str, dtype_str, dtype_str, dtype_str, "INT64", dtype_str, dtype_str]

        def checker(row: pd.Series, kt: str, lat_col: str) -> Optional[Candidate]:
            csv_shapes = _parse_shape_str(str(row.get("Input Shapes", "")))
            csv_dtypes = _parse_str_list(str(row.get("Input Data Types", "")))
            csv_formats = _parse_str_list(str(row.get("Input Formats", "")))
            if len(csv_shapes) < 7 or len(csv_dtypes) < 7 or len(csv_formats) < 7:
                return None
            if csv_shapes[:5] != expected_prefix:
                return None
            cache_rope_shape, cache_latent_shape = csv_shapes[5], csv_shapes[6]
            if len(cache_rope_shape) != 4 or len(cache_latent_shape) != 4:
                return None
            if cache_rope_shape[1:] != (block_size, 1, rope_dim):
                return None
            if cache_latent_shape[1:] != (block_size, 1, kv_lora_rank):
                return None
            if csv_dtypes[:7] != expected_dtypes or any(fmt != "ND" for fmt in csv_formats[:7]):
                return None
            return Candidate(
                latency_us=float(row[lat_col]),
                kernel_type=kt,
                details={
                    "query_mode": "cache_postprocess",
                    "semantic_key": {
                        "tokens": tokens,
                        "kv_proj_dim": kv_proj_dim,
                        "kv_lora_rank": kv_lora_rank,
                        "rope_dim": rope_dim,
                        "block_size": block_size,
                    },
                    "ignored_physical_pool_dim0": [cache_rope_shape[0], cache_latent_shape[0]],
                },
                shape_match_info=ShapeMatchInfo(
                    simulation_shapes=[list(shape) for shape in expected_prefix],
                    kernel_shapes=[list(shape) for shape in csv_shapes[:7]],
                    shape_match_rule="cache_pool_dim0_agnostic",
                ),
            )

        hit = self._find_candidates(kernel_types, checker, select="nearest")
        if hit is None and not self.last_miss_reason:
            self.last_miss_reason = "semantic_key_mismatch:cache_postprocess"
        return hit

    def _query_scatter_cache_write(
        self,
        kernel_types: List[str],
        params: Dict[str, Any],
        dtype_str: str,
    ) -> Optional[Candidate]:
        """Match a cache scatter while ignoring only physical pool capacity."""

        tokens = params.get("tokens")
        feature_dim = params.get("feature_dim")
        if not isinstance(tokens, int) or not isinstance(feature_dim, int) or min(tokens, feature_dim) <= 0:
            self.last_miss_reason = "semantic_context_missing:target:scatter_cache_write"
            return None

        expected_tail = [(tokens, 1), (tokens, feature_dim)]
        expected_dtypes = [dtype_str, "INT32", dtype_str]

        def checker(row: pd.Series, kt: str, lat_col: str) -> Optional[Candidate]:
            csv_shapes = _parse_shape_str(str(row.get("Input Shapes", "")))
            csv_dtypes = _parse_str_list(str(row.get("Input Data Types", "")))
            csv_formats = _parse_str_list(str(row.get("Input Formats", "")))
            output_shapes = _parse_shape_str(str(row.get("Output Shapes", "")))
            output_dtypes = _parse_str_list(str(row.get("Output Data Types", "")))
            output_formats = _parse_str_list(str(row.get("Output Formats", "")))
            if (
                len(csv_shapes) != 3
                or len(csv_shapes[0]) != 2
                or csv_shapes[0][-1] != feature_dim
                or csv_shapes[1:] != expected_tail
                or csv_dtypes != expected_dtypes
                or csv_formats != ["ND", "ND", "ND"]
                or output_shapes != [csv_shapes[0]]
                or output_dtypes != [dtype_str]
                or output_formats != ["ND"]
            ):
                return None
            return Candidate(
                latency_us=float(row[lat_col]),
                kernel_type=kt,
                details={"ignored_physical_pool_dim0": csv_shapes[0][0]},
                shape_match_info=ShapeMatchInfo(
                    simulation_shapes=[
                        [tokens, feature_dim],
                        [tokens, 1],
                        [tokens, feature_dim],
                    ],
                    kernel_shapes=[list(shape) for shape in csv_shapes],
                    shape_match_rule="scatter_cache_pool_dim0_agnostic",
                ),
            )

        hit = self._find_candidates(kernel_types, checker, select="nearest")
        if hit is None and not self.last_miss_reason:
            self.last_miss_reason = "semantic_key_mismatch:scatter_cache_write"
        return hit

    def _lookup_comm_for_composite(
        self,
        op_invoke_info: "OpInvokeInfo",
        kernel_type: str,
        message_bytes_mode: str = "full_output",
        group_type: Optional[str] = None,
        interpolation_method: str = "alpha_beta",
    ) -> Optional[Tuple[float, Dict[str, Any]]]:
        """Look up comm sub-kernel latency for composite ops (e.g., MC2).

        Computes message_bytes from the matmul output shape:
          output = (mat1.shape[0], mat2.shape[-1])
          message_bytes = output_elements * element_size

        Args layout for matmul composites:
          args[0]: mat1, args[1]: mat2, args[-1]: rank_group
        """
        args = op_invoke_info.args
        rank_group = args[-1]
        if not isinstance(rank_group, (list, tuple)):
            return None
        num_devices = len(rank_group)
        if group_type == "tensor_parallel":
            if self.tp_size is None or num_devices != self.tp_size:
                return None

        mat1 = args[0]
        mat2 = args[1]
        if not isinstance(mat1, torch.Tensor) or not isinstance(mat2, torch.Tensor):
            return None
        # Determine output element size for message_bytes calculation.
        # Quant MC2 ops (INT8/FP8/MXFP4 inputs) always accumulate and
        # all_reduce in BF16. Non-quant MC2 (BF16 inputs) keeps the same dtype.
        input_dtype = mat1.dtype
        if input_dtype in (
            torch.int8,
            torch.uint8,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            output_elem_size = 2  # BF16
        else:
            output_elem_size = mat1.element_size()
        message_bytes = mat1.shape[0] * mat2.shape[-1] * output_elem_size
        if message_bytes_mode == "per_rank_output":
            if num_devices <= 1:
                self.last_miss_reason = "invalid_per_rank_output_group_size"
                return None
            if message_bytes % num_devices != 0:
                self.last_miss_reason = (
                    f"nondivisible_per_rank_output:message_bytes={message_bytes};num_devices={num_devices}"
                )
                return None
            message_bytes //= num_devices
        elif message_bytes_mode != "full_output":
            return None

        topology_tier = self._resolve_topology_tier(list(rank_group))

        result = self._query_comm_csv(
            kernel_type,
            message_bytes,
            num_devices,
            topology_tier,
            interpolation_method=interpolation_method,
        )
        if result is None:
            return None
        latency, is_interpolated = result
        return latency, {
            "message_bytes": message_bytes,
            "message_bytes_mode": message_bytes_mode,
            "group_type": group_type,
            "num_devices": num_devices,
            "topology_tier": topology_tier,
            "interpolation_method": interpolation_method,
            "interpolated": is_interpolated,
        }

    # ---- Communication op lookup ----

    def _resolve_topology_tier(self, group: list) -> Optional[int]:
        """Resolve topology_tier from group using CommGrid.

        Returns topology_tier or None if comm_grid is not set.
        """
        if self.comm_grid is None:
            return None
        try:
            return get_topology_tier(self.comm_grid, group)
        except ValueError:
            logger.debug("Could not resolve topology_tier for group %s", group)
            return None

    def _lookup_comm(self, op_invoke_info: "OpInvokeInfo", mapping: dict) -> Optional[QueryResult]:
        """Look up communication op latency by message_bytes + num_devices + topology_tier.

        All TC comm ops have rank_group as the last arg:
          all_reduce(x, rank, rank_group)
          all_gather(x, dim, rank, rank_group)
          reduce_scatter(x, dim, rank, rank_group)
          all_to_all(x, out_splits, in_splits, rank, rank_group)

        Args are expected as (tensor, ..., rank, rank_group) where rank is
        second-to-last and rank_group (list of device ranks) is always last.
        topology_tier is resolved from rank + rank_group via CommGrid when
        comm_grid is set; otherwise the CSV is queried without tier filtering.
        """
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            self.last_miss_reason = "unmapped"
            return None

        # Extract the first tensor arg for message_bytes
        tensor = op_invoke_info.args[0]
        if not isinstance(tensor, torch.Tensor):
            self.last_miss_reason = "invalid_args"
            return None
        message_bytes = tensor.nelement() * tensor.element_size()

        # Extract rank (second-to-last) and rank_group (last)
        rank_group = op_invoke_info.args[-1]
        rank = op_invoke_info.args[-2]  # noqa: F841
        if not isinstance(rank_group, (list, tuple)):
            self.last_miss_reason = "invalid_args"
            return None
        num_devices = len(rank_group)

        # reduce_scatter: TC args[0] is the full input tensor (sendBuf), but
        # bench CSV message_bytes follows HCCL API convention where recvCount
        # is the per-rank output size.  Divide by num_devices to align.
        func_str = _normalize_func_name(op_invoke_info.func)
        if func_str == "tensor_cast.reduce_scatter.default" and num_devices > 1:
            message_bytes = message_bytes // num_devices

        # Resolve topology_tier from group via CommGrid
        topology_tier = self._resolve_topology_tier(list(rank_group))

        result = self._query_comm_csv(kernel_type, message_bytes, num_devices, topology_tier)
        if result is None:
            return None

        latency, is_interpolated = result
        source = QuerySource.INTERPOLATED if is_interpolated else QuerySource.MEASURED
        logger.debug(
            "HIT (comm%s) %s: message_bytes=%d, num_devices=%d, topology_tier=%s -> %.2f us",
            " interpolated" if is_interpolated else "",
            kernel_type,
            message_bytes,
            num_devices,
            topology_tier,
            latency,
        )
        return QueryResult(
            latency_us=latency,
            confidence=0.8 if is_interpolated else 0.9,
            source=source,
            details={"kernel_type": kernel_type, "topology_tier": topology_tier},
            shape_match_info=ShapeMatchInfo(
                simulation_shapes=[[message_bytes]],
                kernel_shapes=[[message_bytes]],
                shape_match_rule="comm",
            ),
        )

    # ---- Attention special lookup ----

    def _lookup_attention(self, op_invoke_info: "OpInvokeInfo", mapping: dict) -> Optional[QueryResult]:
        """Query FIA enriched CSV: extract params from OpInvokeInfo, delegate.

        Extracts Q shape, avg_seq_len, sparse_mode, num_kv_heads from the op,
        builds kernel_types list (primary + alternates), then delegates to
        _query_by_attn_params for the actual CSV matching.
        """
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            self.last_miss_reason = "unmapped"
            return None

        args = op_invoke_info.args
        if len(args) < 7:
            self.last_miss_reason = "insufficient_args"
            return None

        query = args[0]
        key = args[1]
        seq_lens = args[6] if len(args) > 6 else None
        query_lens = args[7] if len(args) > 7 else None

        if not isinstance(query, torch.Tensor):
            self.last_miss_reason = "query_not_tensor"
            return None

        tc_dtype_str = DTYPE_MAP.get(query.dtype)
        if tc_dtype_str is None:
            self.last_miss_reason = "dtype_unmapped"
            return None

        # Get head_dim from key tensor
        head_dim = key.shape[-1] if isinstance(key, torch.Tensor) and key.ndim >= 1 else 0

        # Normalize TC query to 3D
        tc_q_3d = _normalize_fia_q_shape(tuple(query.shape), head_dim)
        if tc_q_3d is None:
            self.last_miss_reason = "q_shape_normalize_failed"
            return None

        # Compute avg_seq_len from seq_lens
        if seq_lens is not None and isinstance(seq_lens, torch.Tensor):
            try:
                tc_avg_seq_len = int(seq_lens.float().mean().item())
            except Exception:
                self.last_miss_reason = "invalid_seq_lens"
                return None
        else:
            self.last_miss_reason = "missing_seq_lens"
            return None

        # Infer sparse_mode from query_lens (TC does not pass it explicitly)
        tc_sparse_mode = _infer_sparse_mode(query_lens)

        # Extract num_kv_heads from key tensor: shape[-2] is kv_head_num
        tc_num_kv_heads = key.shape[-2] if isinstance(key, torch.Tensor) and key.ndim >= 2 else None

        # Derive input_layout from query shape ndim
        input_layout = "TND" if query.ndim == 3 else "BNSD_NBSD" if query.ndim == 4 else None

        # Build params dict
        params = {
            "q_shape_3d": tc_q_3d,
            "avg_seq_len": tc_avg_seq_len,
            "sparse_mode": tc_sparse_mode,
            "num_kv_heads": tc_num_kv_heads,
            "input_layout": input_layout,
        }

        # Build kernel_types list: primary + alternates
        kernel_types = [kernel_type]
        for alt in mapping.get("alternate_kernel_types", []):
            if alt not in kernel_types:
                kernel_types.append(alt)

        result = self._query_by_attn_params(kernel_types, params, tc_dtype_str)
        if result is None:
            # last_miss_reason already set by _query_by_attn_params
            # (csv_not_found or shape_mismatch)
            return None

        lat, matched_kernel = result
        return QueryResult(
            latency_us=lat,
            confidence=0.9,
            source=QuerySource.MEASURED,
            details={
                "kernel_type": matched_kernel,
                "avg_seq_len": tc_avg_seq_len,
                "sparse_mode": tc_sparse_mode,
                "num_kv_heads": tc_num_kv_heads,
            },
            shape_match_info=ShapeMatchInfo(
                simulation_shapes=[list(s) for s, _ in self._extract_tensor_inputs(op_invoke_info)],
                kernel_shapes=[],
                shape_match_rule="attention",
            ),
        )

    # ---- Elementwise op lookup (output-shape matching) ----

    def _lookup_elementwise(self, op_invoke_info: "OpInvokeInfo", mapping: dict) -> Optional[QueryResult]:
        """Look up elementwise op latency by matching output shape.

        Elementwise ops (mul, add, etc.) are bandwidth-bound and their cost
        scales with output size. When the output dtype differs from CSV,
        latency is scaled by the byte-size ratio.

        Falls back to _lookup_compute when output is unavailable.
        """
        out = op_invoke_info.out
        if out is None:
            return self._lookup_compute(op_invoke_info, mapping)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if not isinstance(out, torch.Tensor) or out.ndim == 0 or len(out.shape) < 1:
            return self._lookup_compute(op_invoke_info, mapping)

        tc_output_shape = _strip_batch_dim(tuple(out.shape))
        tc_dtype_str = DTYPE_MAP.get(out.dtype)

        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            self.last_miss_reason = "unmapped"
            return None

        def checker(row: pd.Series, kt: str, lat_col: str) -> Optional[Candidate]:
            csv_out_shapes = _parse_shape_str(str(row.get("Output Shapes", "")))
            csv_out_dtypes = _parse_str_list(str(row.get("Output Data Types", "")))
            if not csv_out_shapes:
                return None

            csv_shape = csv_out_shapes[0]
            csv_shape_stripped = _strip_batch_dim(csv_shape)

            shape_matched = (
                tc_output_shape in (csv_shape, csv_shape_stripped)
                or self._shapes_match_with_padding(tc_output_shape, csv_shape)
                or self._shapes_match_with_padding(tc_output_shape, csv_shape_stripped)
            )
            if not shape_matched and len(tc_output_shape) == 3 and len(csv_shape) == 2:
                flat = (tc_output_shape[0] * tc_output_shape[1], tc_output_shape[2])
                shape_matched = (
                    flat in (csv_shape, csv_shape_stripped)
                    or self._shapes_match_with_padding(flat, csv_shape)
                    or self._shapes_match_with_padding(flat, csv_shape_stripped)
                )
            if not shape_matched:
                return None

            csv_dtype_str = csv_out_dtypes[0] if csv_out_dtypes else None
            latency = float(row[lat_col])
            smi = ShapeMatchInfo(
                simulation_shapes=[list(tc_output_shape)],
                kernel_shapes=[list(csv_shape)],
                shape_match_rule="elementwise",
            )

            if tc_dtype_str and csv_dtype_str and tc_dtype_str == csv_dtype_str:
                return Candidate(
                    latency_us=latency,
                    kernel_type=kt,
                    details={"kernel_type": kt, "query_mode": "elementwise"},
                    shape_match_info=smi,
                )

            tc_bytes = _dtype_byte_size(tc_dtype_str) if tc_dtype_str else 0
            csv_bytes = _dtype_byte_size(csv_dtype_str) if csv_dtype_str else 0
            if tc_bytes > 0 and csv_bytes > 0:
                scale = tc_bytes / csv_bytes
                return Candidate(
                    latency_us=latency * scale,
                    kernel_type=kt,
                    confidence=0.9,
                    details={
                        "kernel_type": kt,
                        "query_mode": "elementwise",
                        "dtype_scale": scale,
                    },
                    shape_match_info=smi,
                )
            return None

        hit = self._find_candidates([kernel_type], checker)
        if hit is None:
            self.last_miss_reason = "elementwise_output_shape_mismatch"
            logger.debug(
                "MISS (elementwise) %s: output=%s dtype=%s",
                kernel_type,
                tc_output_shape,
                tc_dtype_str,
            )
            return None

        logger.debug(
            "HIT (elementwise) %s: output=%s -> %.2f us",
            hit.kernel_type,
            tc_output_shape,
            hit.latency_us,
        )
        return QueryResult(
            latency_us=hit.latency_us,
            confidence=hit.confidence,
            source=QuerySource.MEASURED,
            details=hit.details,
            shape_match_info=hit.shape_match_info,
        )

    # ---- Shared shape-matching via checker (replaces _query_by_shapes) ----

    def _make_compute_checker(
        self,
        tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
        tc_input_count: Optional[int],
        simulation_shapes: Optional[List[List[int]]] = None,
    ) -> CheckerFn:
        """Build a checker closure for compute-style shape matching.

        Args:
            tc_inputs: TC tensor (shape, dtype) pairs to match.
            tc_input_count: Truncate CSV inputs to first N for comparison.
            simulation_shapes: Pre-computed simulation shapes for ShapeMatchInfo.
                If None, derived from tc_inputs on each call.
        """
        sim_shapes = simulation_shapes or [list(s) for s, _ in tc_inputs]

        def checker(row: pd.Series, kt: str, lat_col: str) -> Optional[Candidate]:
            rule = self._inputs_match(tc_inputs, row, kt, tc_input_count)
            if rule is None:
                return None
            csv_shapes = [list(s) for s in _parse_shape_str(str(row.get("Input Shapes", "")))]
            return Candidate(
                latency_us=float(row[lat_col]),
                kernel_type=kt,
                distance=_shape_match_distance(rule),
                shape_match_info=ShapeMatchInfo(
                    simulation_shapes=sim_shapes,
                    kernel_shapes=csv_shapes,
                    shape_match_rule=rule,
                ),
            )

        return checker

    def _find_compute_match(
        self,
        kernel_types: List[str],
        tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
        tc_input_count: Optional[int] = None,
        auto_truncate: bool = False,
    ) -> Optional[Candidate]:
        """Find a compute match using _find_candidates + _inputs_match checker.

        Drop-in replacement for the old _query_by_shapes, used by composite
        lookup paths that need the raw Candidate result.
        """
        effective_tc_input_count = tc_input_count
        if auto_truncate and effective_tc_input_count is None and len(tc_inputs) > 0:
            effective_tc_input_count = len(tc_inputs)

        checker = self._make_compute_checker(tc_inputs, effective_tc_input_count)

        hit = self._find_candidates(kernel_types, checker, select="nearest")
        if hit is None and not self.last_miss_reason:
            # Post-miss diagnosis: distinguish input_count_mismatch from
            # shape_mismatch (restores old _query_by_shapes behavior).
            primary = kernel_types[0] if kernel_types else "unknown"
            df = self._load_csv(primary)
            if df is not None and not df.empty:
                csv_first_shapes = _parse_shape_str(str(df.iloc[0].get("Input Shapes", "")))
                effective_csv_count = len(csv_first_shapes)
                effective_tc_count = len(tc_inputs)
                if effective_tc_input_count is not None:
                    effective_csv_count = min(effective_csv_count, effective_tc_input_count)
                    effective_tc_count = min(effective_tc_count, effective_tc_input_count)
                # SwiGlu: TC 2 inputs → CSV 1 (concat normalization)
                if primary in _SWIGLU_KERNELS and effective_tc_count == 2 and effective_csv_count == 1:
                    effective_tc_count = 1
                # ReshapeAndCache: TC 4 inputs → CSV 5 (split normalization)
                if primary in _RESHAPE_AND_CACHE_KERNELS and effective_tc_count == 4 and effective_csv_count == 5:
                    effective_tc_count = 5
                if effective_tc_count != effective_csv_count:
                    self.last_miss_reason = "input_count_mismatch"
                else:
                    self.last_miss_reason = "shape_mismatch"
            else:
                self.last_miss_reason = "csv_not_found"
        return hit

    def _find_embedded_routing_weight_cast(
        self,
        op_invoke_info: "OpInvokeInfo",
        spec: dict,
    ) -> Optional[Candidate]:
        """Match the cast that prepares DFC floating-point routing weights."""
        kernel_type = spec.get("kernel_type")
        output_dtype = spec.get("output_dtype")
        args = op_invoke_info.args
        if (
            not isinstance(kernel_type, str)
            or not kernel_type
            or not isinstance(output_dtype, str)
            or len(args) < 2
            or not isinstance(args[0], torch.Tensor)
            or not isinstance(args[1], torch.Tensor)
        ):
            return None

        routing_shape = tuple(args[1].shape)
        tc_inputs = [(routing_shape, args[0].dtype)]
        simulation_shapes = [list(routing_shape)]
        base_checker = self._make_compute_checker(tc_inputs, 1, simulation_shapes)

        def checker(row: pd.Series, kt: str, lat_col: str) -> Optional[Candidate]:
            candidate = base_checker(row, kt, lat_col)
            if candidate is None:
                return None
            output_shapes = _parse_shape_str(str(row.get("Output Shapes", "")))
            output_dtypes = _parse_str_list(str(row.get("Output Data Types", "")))
            if output_shapes != [routing_shape] or output_dtypes != [output_dtype]:
                return None
            return candidate

        return self._find_candidates([kernel_type], checker, select="nearest")

    def _find_embedded_residual_norm(
        self,
        activation: torch.Tensor,
        spec: dict,
    ) -> Optional[Candidate]:
        """Project an activation to a residual-add norm physical signature."""
        kernel_type = spec.get("kernel_type")
        if not isinstance(kernel_type, str) or not kernel_type or activation.ndim < 2:
            return None
        activation_shape = tuple(activation.shape)
        if spec.get("flatten_leading_dims") and activation.ndim > 2:
            activation_shape = (math.prod(activation_shape[:-1]), activation_shape[-1])
        if len(activation_shape) != 2:
            return None
        hidden_shape = (activation_shape[-1],)
        tc_inputs = [
            (activation_shape, activation.dtype),
            (activation_shape, activation.dtype),
            (hidden_shape, activation.dtype),
            (hidden_shape, activation.dtype),
        ]
        return self._find_compute_match([kernel_type], tc_inputs, tc_input_count=4)

    @staticmethod
    def _candidate_sub_kernel_shape(candidate: Candidate) -> SubKernelShapeInfo:
        shape_info = candidate.shape_match_info
        return SubKernelShapeInfo(
            kernel_type=candidate.kernel_type,
            simulation_shapes=shape_info.simulation_shapes if shape_info else [],
            kernel_shapes=shape_info.kernel_shapes if shape_info else [],
            shape_match_rule=shape_info.shape_match_rule if shape_info else "unknown",
        )

    # ---- MoE fused op lookup (EP Size matching) ----

    def _moe_embedded_comm_context(
        self,
        op_invoke_info: "OpInvokeInfo",
        spec: dict,
    ) -> Optional[Dict[str, Any]]:
        """Derive an embedded routed-input communication query for fused MoE."""
        args = op_invoke_info.args
        if len(args) < 2:
            return None
        x, expert_indices = args[0], args[1]
        rank_group = args[-1]
        if (
            not isinstance(x, torch.Tensor)
            or not isinstance(expert_indices, torch.Tensor)
            or not isinstance(rank_group, (list, tuple))
        ):
            return None
        if spec.get("message_bytes_mode") != "routed_input":
            return None

        topk = int(expert_indices.shape[-1]) if expert_indices.ndim >= 2 else 1
        if topk <= 0:
            return None
        message_bytes = int(x.nelement() * x.element_size() * topk)
        num_devices = len(rank_group)
        if spec.get("group_type") == "expert_parallel" and (self.ep_size is None or num_devices != self.ep_size):
            return None

        return {
            "message_bytes": message_bytes,
            "message_bytes_mode": "routed_input",
            "topk": topk,
            "num_devices": num_devices,
            "topology_tier": self._resolve_topology_tier(list(rank_group)),
        }

    def _lookup_moe(self, op_invoke_info: "OpInvokeInfo", mapping: dict) -> Optional[QueryResult]:
        """Query DFC CSV: shape match + EP Size exact match."""
        kernel_type = mapping.get("kernel_type")
        if not kernel_type:
            self.last_miss_reason = "unmapped"
            return None

        # Pre-check: if CSV has EP Size column but ep_size not configured, bail
        df = self._load_csv(kernel_type)
        if df is None:
            self.last_miss_reason = "csv_not_found"
            return None
        has_ep_col = "EP Size" in df.columns
        if has_ep_col and self.ep_size is None:
            logger.warning(
                "DFC CSV has EP Size column but ep_size not configured. Pass parallel_config to ProfilingDataSource."
            )
            self.last_miss_reason = "ep_size_not_configured"
            return None

        raw_tc_inputs = self._extract_tensor_inputs(op_invoke_info)
        projected_inputs = None
        if mapping.get("project_full_shape"):
            projected_inputs = _project_dispatch_ffn_combine_inputs(op_invoke_info)
        tc_inputs = projected_inputs or raw_tc_inputs
        tc_input_count = mapping.get("tc_input_count")
        if projected_inputs is not None:
            tc_input_count = None
        elif tc_input_count is not None:
            tc_inputs = tc_inputs[:tc_input_count]
        simulation_shapes = [list(s) for s, _ in tc_inputs]
        ep_size = self.ep_size

        def checker(row: pd.Series, kt: str, lat_col: str) -> Optional[Candidate]:
            rule = self._inputs_match(tc_inputs, row, kt, tc_input_count)
            if rule is None:
                return None
            if projected_inputs is not None and rule == "identity":
                rule = "projected_full_shape"
            if has_ep_col and ep_size is not None:
                ep_val = row["EP Size"]
                if pd.isna(ep_val) or int(ep_val) != ep_size:
                    return None
            csv_shapes = [list(s) for s in _parse_shape_str(str(row.get("Input Shapes", "")))]
            return Candidate(
                latency_us=float(row[lat_col]),
                kernel_type=kt,
                details={"kernel_type": kt, "ep_size": ep_size},
                shape_match_info=ShapeMatchInfo(
                    simulation_shapes=simulation_shapes,
                    kernel_shapes=csv_shapes,
                    shape_match_rule=rule,
                ),
            )

        hit = self._find_candidates([kernel_type], checker)
        if hit is None:
            if not self.last_miss_reason:
                self.last_miss_reason = "shape_mismatch"
            return None

        logger.debug(
            "HIT (moe) %s: ep_size=%s -> %.1f us",
            hit.kernel_type,
            ep_size,
            hit.latency_us,
        )

        embedded_hits: List[Candidate] = []
        embedded_value = mapping.get("embedded_compute")
        embedded_specs = embedded_value if isinstance(embedded_value, list) else [embedded_value]
        activation = _first_tensor(op_invoke_info.args[0]) if op_invoke_info.args else None
        for embedded_spec in embedded_specs:
            if not isinstance(embedded_spec, dict):
                continue
            projection = embedded_spec.get("input_projection")
            embedded_hit = None
            if projection == "residual_norm_from_activation" and activation is not None:
                embedded_hit = self._find_embedded_residual_norm(activation, embedded_spec)
            elif projection == "routing_weight_cast":
                embedded_hit = self._find_embedded_routing_weight_cast(op_invoke_info, embedded_spec)
            if embedded_hit is not None:
                embedded_hits.append(embedded_hit)

        base_latency = hit.latency_us + sum(item.latency_us for item in embedded_hits)
        base_durations = [(hit.kernel_type, round(hit.latency_us, 2))] + [
            (item.kernel_type, round(item.latency_us, 2)) for item in embedded_hits
        ]
        sub_kernel_shapes = [self._candidate_sub_kernel_shape(hit)] + [
            self._candidate_sub_kernel_shape(item) for item in embedded_hits
        ]
        base_details = {
            **hit.details,
            "sub_kernel_durations": base_durations,
        }

        comm_spec = mapping.get("embedded_communication")
        if not isinstance(comm_spec, dict):
            return QueryResult(
                latency_us=base_latency,
                confidence=hit.confidence,
                source=QuerySource.MEASURED,
                details=base_details,
                shape_match_info=hit.shape_match_info,
                sub_kernel_shapes=sub_kernel_shapes,
            )

        comm_kernel_type = comm_spec.get("kernel_type")
        interpolation_method = comm_spec.get("interpolation_method", "alpha_beta")
        if not isinstance(comm_kernel_type, str) or not comm_kernel_type.startswith("hcom_"):
            self.last_miss_reason = "invalid_moe_embedded_communication"
            return None
        if interpolation_method not in {"alpha_beta", "bracket_linear"}:
            self.last_miss_reason = "invalid_moe_embedded_communication"
            return None

        comm_context = self._moe_embedded_comm_context(op_invoke_info, comm_spec)
        if comm_context is None:
            self.last_miss_reason = "moe_embedded_communication_context_mismatch"
            return None
        comm_result = self._query_comm_csv(
            comm_kernel_type,
            comm_context["message_bytes"],
            comm_context["num_devices"],
            comm_context["topology_tier"],
            interpolation_method=interpolation_method,
        )
        if comm_result is None:
            if self.last_miss_reason == "shape_mismatch":
                return QueryResult(
                    latency_us=base_latency,
                    confidence=hit.confidence,
                    source=QuerySource.MEASURED,
                    details={
                        **base_details,
                        "embedded_communication_active": False,
                        "embedded_communication": {
                            **comm_context,
                            "kernel_type": comm_kernel_type,
                            "reason": "outside_measured_range",
                        },
                    },
                    shape_match_info=hit.shape_match_info,
                    sub_kernel_shapes=sub_kernel_shapes,
                )
            partial_miss_reason = f"moe_embedded_communication_miss:{comm_kernel_type}"
            self.last_miss_reason = ""
            return QueryResult(
                latency_us=base_latency,
                confidence=0.5,
                source=QuerySource.PARTIAL,
                details={
                    **base_details,
                    "hit_kernels": [name for name, _latency in base_durations],
                    "missed_kernels": [comm_kernel_type],
                    "partial_miss_reason": partial_miss_reason,
                    "embedded_communication": comm_context,
                },
                shape_match_info=hit.shape_match_info,
                sub_kernel_shapes=sub_kernel_shapes,
            )

        comm_latency, is_interpolated = comm_result
        sub_kernel_shapes.append(
            SubKernelShapeInfo(
                kernel_type=comm_kernel_type,
                simulation_shapes=[[comm_context["message_bytes"]]],
                kernel_shapes=[[comm_context["message_bytes"]]],
                shape_match_rule="comm",
            )
        )
        return QueryResult(
            latency_us=base_latency + comm_latency,
            confidence=0.8 if is_interpolated else 0.9,
            source=QuerySource.INTERPOLATED if is_interpolated else QuerySource.MEASURED,
            details={
                **base_details,
                "sub_kernel_durations": [*base_durations, (comm_kernel_type, round(comm_latency, 2))],
                "embedded_communication": {
                    **comm_context,
                    "kernel_type": comm_kernel_type,
                    "interpolation_method": interpolation_method,
                    "interpolated": is_interpolated,
                },
            },
            shape_match_info=hit.shape_match_info,
            sub_kernel_shapes=sub_kernel_shapes,
        )

    def _lookup_mtp_projection(self, op_invoke_info: "OpInvokeInfo", mapping: dict) -> Optional[QueryResult]:
        """Resolve the 2H-to-H concat projection used by an MTP block."""

        tc_inputs = self._extract_tensor_inputs(op_invoke_info)
        output_tensor = _first_tensor(op_invoke_info.out)
        if len(tc_inputs) < 2 or output_tensor is None:
            return self._lookup_compute(op_invoke_info, mapping)
        activation_shape, _ = tc_inputs[0]
        weight_shape, _ = tc_inputs[1]
        output_shape = tuple(output_tensor.shape)
        if (
            len(activation_shape) != 2
            or len(weight_shape) != 2
            or len(output_shape) != 2
            or activation_shape[0] != output_shape[0]
            or activation_shape[-1] != 2 * output_shape[-1]
            or set(weight_shape) != {activation_shape[-1], output_shape[-1]}
        ):
            return self._lookup_compute(op_invoke_info, mapping)

        projection_kernel_type = mapping.get("mtp_projection_kernel_type")
        if not isinstance(projection_kernel_type, str) or not projection_kernel_type:
            return self._lookup_compute(op_invoke_info, mapping)
        projection_inputs = [
            (activation_shape, output_tensor.dtype),
            (weight_shape, output_tensor.dtype),
        ]
        projection_shapes = [list(shape) for shape, _ in projection_inputs]
        projection_hit = self._find_candidates(
            [projection_kernel_type],
            self._make_compute_checker(projection_inputs, 2, projection_shapes),
            select="nearest",
        )
        if projection_hit is None:
            return self._lookup_compute(op_invoke_info, mapping)
        projection_shape_info = projection_hit.shape_match_info
        projection_result = QueryResult(
            latency_us=projection_hit.latency_us,
            confidence=projection_hit.confidence,
            source=QuerySource.MEASURED,
            details={"kernel_type": projection_hit.kernel_type, **projection_hit.details},
            shape_match_info=projection_shape_info,
        )

        return projection_result

    # ---- Compute op lookup ----

    def _lookup_compute(self, op_invoke_info: "OpInvokeInfo", mapping: dict) -> Optional[QueryResult]:
        if not mapping.get("kernel_type"):
            return None

        kernel_types = [mapping["kernel_type"]]
        for alt in mapping.get("alternate_kernel_types", []):
            if alt not in kernel_types:
                kernel_types.append(alt)

        tc_inputs = self._extract_tensor_inputs(op_invoke_info)
        tc_input_count = mapping.get("tc_input_count")
        if tc_input_count is not None:
            tc_inputs = tc_inputs[:tc_input_count]

        degenerate_bmm_kernel = mapping.get("degenerate_bmm_kernel_type")
        if isinstance(degenerate_bmm_kernel, str) and _uses_glm5_sampling_bmm_mul(tc_inputs):
            if degenerate_bmm_kernel not in kernel_types:
                kernel_types.append(degenerate_bmm_kernel)

        # A versioned graph path may lower a rank-specific TC op to another
        # kernel variant at the same abstraction level. The mapping can reorder
        # candidates by the first semantic input rank; all ordinary shape,
        # dtype, and context checks remain mandatory.
        if tc_inputs:
            rank_map = mapping.get("kernel_types_by_input_rank", {})
            input_rank = len(tc_inputs[0][0])
            rank_candidates = rank_map.get(input_rank, rank_map.get(str(input_rank)))
            if isinstance(rank_candidates, list) and all(isinstance(item, str) for item in rank_candidates):
                kernel_types = list(dict.fromkeys([*rank_candidates, *kernel_types]))

        projection_rule = mapping.get("tp_linear_projection")
        if projection_rule is not None:
            if not isinstance(projection_rule, dict):
                self.last_miss_reason = "invalid_tp_linear_projection_rule"
                return None
            shard_axis = projection_rule.get("shard_axis")
            logical_global_dims = projection_rule.get("logical_global_dims")
            otherwise = projection_rule.get("otherwise", "physical_local")
            if (
                shard_axis not in {"input", "output"}
                or not isinstance(logical_global_dims, list)
                or not logical_global_dims
                or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in logical_global_dims)
                or otherwise != "physical_local"
                or len(tc_inputs) < 2
                or len(tc_inputs[0][0]) != 2
                or len(tc_inputs[1][0]) != 2
            ):
                self.last_miss_reason = "invalid_tp_linear_projection_rule"
                return None

            visible_dim = tc_inputs[0][0][-1] if shard_axis == "input" else tc_inputs[1][0][1]
            if visible_dim in logical_global_dims:
                if self.tp_size == 1:
                    projected = tc_inputs
                elif shard_axis == "input":
                    projected = _project_tp_sharded_linear_inputs(tc_inputs, self.tp_size)
                else:
                    projected = _project_tp_sharded_output_linear_inputs(tc_inputs, self.tp_size)
                if projected is None:
                    self.last_miss_reason = (
                        "tp_projection_invalid:"
                        f"logical_dim={visible_dim};shard_axis={shard_axis};tp_size={self.tp_size}"
                    )
                    self.last_shape_match_info = ShapeMatchInfo(
                        simulation_shapes=[list(shape) for shape, _dtype in tc_inputs],
                        kernel_shapes=[],
                        shape_match_rule=self.last_miss_reason,
                    )
                    return None
                tc_inputs = projected

        simulation_shapes = [list(s) for s, _ in tc_inputs]

        checker = self._make_compute_checker(tc_inputs, tc_input_count, simulation_shapes)

        hit = self._find_candidates(kernel_types, checker, select="nearest")
        if hit is None:
            if not self.last_miss_reason:
                self.last_miss_reason = "shape_mismatch"
            self.last_shape_match_info = ShapeMatchInfo(
                simulation_shapes=simulation_shapes,
                kernel_shapes=[],
                shape_match_rule=self.last_miss_reason,
            )
            return None

        self.last_shape_match_info = hit.shape_match_info
        return QueryResult(
            latency_us=hit.latency_us,
            confidence=hit.confidence,
            source=QuerySource.MEASURED,
            details={"kernel_type": hit.kernel_type, **hit.details},
            shape_match_info=hit.shape_match_info,
        )

    def _extract_tensor_inputs(self, op_invoke_info: "OpInvokeInfo") -> List[Tuple[Tuple[int, ...], torch.dtype]]:
        """Extract (shape, dtype) for each non-scalar tensor arg.

        Scalar tensors (ndim=0, shape=()) are filtered out because profiling
        CSVs never include scalar inputs in their shape strings.
        """
        inputs = []
        for arg in op_invoke_info.args:
            if isinstance(arg, torch.Tensor) and arg.ndim > 0:
                inputs.append((tuple(arg.shape), arg.dtype))
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, torch.Tensor) and item.ndim > 0:
                        inputs.append((tuple(item.shape), item.dtype))
        return inputs

    def _inputs_match(
        self,
        tc_inputs: List[Tuple[Tuple[int, ...], torch.dtype]],
        csv_row: pd.Series,
        kernel_type: str = "",
        tc_input_count: Optional[int] = None,
    ) -> Optional[str]:
        """Match TensorCast input shapes/dtypes against a CSV row.

        Returns the name of the matching rule (e.g. ``"identity"``,
        ``"batch_strip"``, ``"padding"``, ``"transpose"``) on success, or
        ``None`` when no rule fires.

        Handles:
        - FRACTAL_NZ restoration
        - ND weight transpose for matmul kernels (CSV stores (N,K), TC sees (K,N))
        - Block-padding tolerance (TC pads seq to NPU tile alignment)
        """
        csv_shapes = _parse_shape_str(str(csv_row.get("Input Shapes", "")))
        csv_dtypes = _parse_str_list(str(csv_row.get("Input Data Types", "")))
        csv_formats = _parse_str_list(str(csv_row.get("Input Formats", "")))

        # Truncate CSV shapes/dtypes/formats when tc_input_count is set.
        # NPU profiling CSVs may include internal parameters beyond what TC passes;
        # tc_input_count tells us to only compare the first N inputs.
        # NOTE: when tc_input_count is set in a composite mapping, both tc_inputs
        # (truncated above in _lookup_composite) and csv_shapes (truncated here)
        # are shortened — this double truncation is intentional: tc_inputs is
        # pre-filtered to the relevant tensors, csv_shapes is trimmed to match.
        if tc_input_count is not None:
            csv_shapes = csv_shapes[:tc_input_count]
            csv_dtypes = csv_dtypes[:tc_input_count]
            csv_formats = csv_formats[:tc_input_count]

        # RoPE input normalization: swap Q↔K, transpose (B,H,S,D)→(B,S,H,D).
        # Works with both full (4 inputs) and tc_input_count-truncated (2 inputs).
        tc_inputs_normalized = tc_inputs
        if kernel_type in _ROPE_KERNELS and len(tc_inputs) >= 2 and len(csv_shapes) >= 2:
            tc_inputs_normalized = _normalize_rope_inputs(tc_inputs)

        # SwiGlu input normalization: TC sends 2 inputs (gate, up),
        # profiling CSV has 1 fused input concatenated along last dim.
        if kernel_type in _SWIGLU_KERNELS and len(tc_inputs) == 2 and len(csv_shapes) == 1:
            s1, dtype1 = tc_inputs[0]
            s2, dtype2 = tc_inputs[1]
            s1 = _strip_batch_dim(s1)
            s2 = _strip_batch_dim(s2)
            if len(s1) == len(s2) and s1[:-1] == s2[:-1] and dtype1 == dtype2:
                merged_shape = s1[:-1] + (s1[-1] + s2[-1],)
                tc_inputs_normalized = [(merged_shape, dtype1)]

        # ReshapeAndCache input normalization: TC sends 4 inputs
        # (key, value, kv_cache, slot_mapping) with 2D key/value and merged
        # kv_cache; CSV has 5 inputs with 3D key/value and split cache_k/cache_v.
        if kernel_type in _RESHAPE_AND_CACHE_KERNELS and len(tc_inputs) == 4 and len(csv_shapes) == 5:
            normalized = _normalize_reshape_and_cache_inputs(tc_inputs_normalized)
            if normalized is not None:
                tc_inputs_normalized = normalized

        if len(tc_inputs_normalized) != len(csv_shapes):
            return None

        matched_rule = "identity"  # upgraded when a non-identity rule fires

        for i, (tc_shape, tc_dtype) in enumerate(tc_inputs_normalized):
            # Check dtype
            expected_dtype = DTYPE_MAP.get(tc_dtype)
            if expected_dtype is None or i >= len(csv_dtypes):
                return None
            csv_dtype_i = csv_dtypes[i]
            if expected_dtype != csv_dtype_i:
                # Relaxed dtype matching for specific kernels (e.g. RoPE:
                # NPU records K as FLOAT while TC dispatches BF16)
                if kernel_type in _DTYPE_RELAXED_KERNELS:
                    compat_expected = _DTYPE_COMPAT.get(expected_dtype)
                    compat_csv = _DTYPE_COMPAT.get(csv_dtype_i)
                    if compat_expected is None or compat_csv is None or compat_expected != compat_csv:
                        return None
                else:
                    return None

            # Get CSV shape, restore FRACTAL_NZ if needed
            csv_shape = csv_shapes[i]
            fmt = csv_formats[i] if i < len(csv_formats) else "ND"
            if fmt == "FRACTAL_NZ":
                csv_shape = fractal_nz_to_nd(csv_shape)

            if tc_shape == csv_shape:
                continue  # identity — matched_rule stays "identity"

            # Strip leading batch dim=1: TC keeps (1, seq, dim), profiling has (seq, dim)
            tc_shape_stripped = _strip_batch_dim(tc_shape)
            csv_shape_stripped = _strip_batch_dim(csv_shape)
            if tc_shape_stripped == csv_shape_stripped:
                matched_rule = "batch_strip"
                continue
            if tc_shape_stripped == csv_shape:
                matched_rule = "batch_strip"
                continue

            # aclnnIndex reports dim-1 indexing of (1, T) token metadata as
            # a column-shaped (T, 1) kernel input.
            if (
                kernel_type == "Index"
                and len(tc_shape) == 2
                and len(csv_shape) == 2
                and tc_shape[0] == 1
                and csv_shape[1] == 1
                and tc_shape[1] == csv_shape[0]
            ):
                matched_rule = "singleton_axis_transpose"
                continue

            # Weight transpose for matmul: CSV stores (N,K), TC sees (K,N)
            # Applies to both ND format and FRACTAL_NZ-restored shapes.
            # FRACTAL_NZ → ND gives (N,K) via fractal_nz_to_nd(); TC has (K,N).
            if (
                kernel_type in _MATMUL_KERNELS
                and i >= 1
                and len(tc_shape_stripped) == 2
                and len(csv_shape) == 2
                and tc_shape_stripped == (csv_shape[1], csv_shape[0])
            ):
                matched_rule = "transpose"
                continue

            # Grouped kernels stack per-expert weights as (E, K, N) in TC,
            # while restored profiling rows can expose (E, N, K).
            if (
                kernel_type in _MATMUL_KERNELS
                and i >= 1
                and len(tc_shape_stripped) == 3
                and len(csv_shape) == 3
                and tc_shape_stripped[0] == csv_shape[0]
                and tc_shape_stripped[1:] == (csv_shape[2], csv_shape[1])
            ):
                matched_rule = "batched_transpose"
                continue

            # Block-padding tolerance: TC pads to NPU tile alignment
            if self._shapes_match_with_padding(tc_shape_stripped, csv_shape):
                if matched_rule == "identity":
                    matched_rule = "padding"
                continue
            # Also try with both batch dims stripped
            if self._shapes_match_with_padding(tc_shape_stripped, csv_shape_stripped):
                if matched_rule in ("identity", "batch_strip"):
                    matched_rule = "batch_strip+padding"
                continue

            # 3D→2D flatten for quantize/norm kernels
            if kernel_type in _FLATTEN_BATCH_KERNELS and len(csv_shape) == 2:
                # Use original tc_shape (pre-strip) for 3D checks, since
                # _strip_batch_dim may collapse (1,H,D) → (H,D) losing the
                # 3D structure needed for flatten/merge.
                shape_3d = (
                    tc_shape_stripped if len(tc_shape_stripped) == 3 else tc_shape if len(tc_shape) == 3 else None
                )
                if shape_3d is not None:
                    # Flatten first two dims: TC (B, M, D) → CSV (B*M, D)
                    flattened = (
                        shape_3d[0] * shape_3d[1],
                        shape_3d[2],
                    )
                    if flattened == csv_shape:
                        matched_rule = "flatten_3d"
                        continue
                    if self._shapes_match_with_padding(flattened, csv_shape):
                        matched_rule = "flatten_3d+padding"
                        continue

                    # Merge last two dims: TC (T, H, D) → CSV (T, H*D)
                    # Only for MLA quantize kernels where NPU reshapes
                    # per-head to hidden_dim before quantize.
                    if kernel_type in _MERGE_LAST_DIMS_KERNELS:
                        merged = (
                            shape_3d[0],
                            shape_3d[1] * shape_3d[2],
                        )
                        if merged == csv_shape:
                            matched_rule = "merge_last_dims"
                            continue
                        if self._shapes_match_with_padding(merged, csv_shape):
                            matched_rule = "merge_last_dims+padding"
                            continue

            return None  # this input didn't match any rule

        return matched_rule  # all inputs matched

    @staticmethod
    def _shapes_match_with_padding(tc_shape: Tuple[int, ...], csv_shape: Tuple[int, ...]) -> bool:
        """Check if shapes match allowing block-padding on any dimension."""
        if len(tc_shape) != len(csv_shape):
            return False
        has_padding = False
        for tc_dim, csv_dim in zip(tc_shape, csv_shape):
            if tc_dim == csv_dim:
                continue
            if _is_block_padded(tc_dim, csv_dim):
                has_padding = True
                continue
            return False
        return has_padding
