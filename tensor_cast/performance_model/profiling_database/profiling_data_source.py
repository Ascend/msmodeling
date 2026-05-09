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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import yaml

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
            shapes.append(tuple(int(x) for x in part.split(",")))
    return shapes


def _parse_str_list(s: str) -> List[str]:
    """Parse 'A;B;C' -> ['A', 'B', 'C']"""
    s = s.strip().strip('"')
    return [x.strip() for x in s.split(";") if x.strip()]


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


def _normalize_fia_q_shape(
    q_shape: Tuple[int, ...], head_dim: int = 0
) -> Optional[Tuple[int, ...]]:
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
_ROPE_KERNELS = frozenset(
    {"ApplyRotaryPosEmb", "_triton_rope", "split_qkv_rmsnorm_rope_kernel"}
)

# ReshapeAndCache kernel types: TC dispatches (key, value, kv_cache, slot_mapping)
# with key/value as 2D (N, D) and a single merged kv_cache (2, blocks, block_size, heads, D).
# Profiling CSVs store (key, value, cache_k, cache_v, slot_mapping) with key/value
# as 3D (N, 1, D) and separate cache_k/cache_v tensors.
_RESHAPE_AND_CACHE_KERNELS = frozenset(
    {"ReshapeAndCacheNdKernel", "reshape_and_cache_200000000"}
)

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
_DTYPE_RELAXED_KERNELS = (
    _ROPE_KERNELS | _RELAXED_DTYPE_MATMUL_KERNELS | _RELAXED_DTYPE_PAD_KERNELS
)

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
        csv_dim >= min_dim and tc_dim == ((csv_dim + bs - 1) // bs) * bs
        for bs, min_dim in _BLOCK_SIZE_MIN_DIM.items()
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
SUPPORTED_QUERY_MODES: frozenset[str] = frozenset(
    {"attention_special", "elementwise", "moe_fused"}
)

# ---- MLA / MLAPO composite decomposition ----


@dataclass
class SubKernelSpec:
    """Specification for a sub-kernel in composite decomposition."""

    kernel_type: str
    input_shapes: List[Tuple[int, ...]]
    dtype: str  # Profiling dtype string, e.g. "DT_BF16"
    query_mode: str = "compute"  # "compute" | "attention"
    attention_params: Optional[Dict[str, Any]] = field(default=None)
    tc_input_count: Optional[int] = None
    alternate_kernel_types: Optional[List[str]] = None


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


def _decompose_mla_common(
    op_invoke_info: "OpInvokeInfo",
    mapping: dict,
    first_kernel_type: str,
    alternate_kernel_types: Optional[List[str]] = None,
) -> Optional[List[SubKernelSpec]]:
    """Shared MLA decomposition for BF16 and quantized variants.

    Decode: first_kernel_type(q@W_UK_T) + FIA + TransposeBatchMatMul(out@W_UV)
    Prefill: MatMulV2(kv_c@kv_b_proj) + FusedInferAttentionScore (v0.18.0: unified FIA)

    Args:
        first_kernel_type: "BatchMatMulV2" for BF16, "QuantBatchMatmulV3" for quant.
        alternate_kernel_types: Optional fallback kernel types for the
            first decode matmul sub-kernel.
    """
    args = op_invoke_info.args
    if len(args) < 10:
        return None
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

    if _is_decode_mla(args):
        W_UK_T = args[6]  # (num_heads, qk_nope_head_dim, kv_lora_rank)
        W_UV = args[7]  # (num_heads, kv_lora_rank, v_head_dim)
        if W_UK_T is None or W_UV is None:
            return None

        qk_nope_head_dim = W_UK_T.shape[1]
        kv_lora_rank = W_UK_T.shape[2]
        v_head_dim_val = W_UV.shape[2]

        try:
            avg_seq_len = int(seq_lens.float().mean().item())
        except (RuntimeError, ValueError):
            avg_seq_len = 0

        # Fix MISS #5: FIA decode Q only sees kv_lora_rank (512), not full head_dim (576).
        # The rope dim (64) is handled by InterleaveRope separately.
        fia_head_dim = kv_lora_rank  # 512, not head_dim=576
        fia_q_raw = (batch_size, num_heads, 1, fia_head_dim)
        fia_q_normalized = _normalize_fia_q_shape(fia_q_raw, fia_head_dim)

        fia_spec = SubKernelSpec(
            kernel_type="FusedInferAttentionScore",
            input_shapes=[],
            dtype=dtype_str,
            query_mode="attention",
            attention_params={
                "q_shape_3d": fia_q_normalized or (batch_size, num_heads, fia_head_dim),
                "avg_seq_len": avg_seq_len,
                "sparse_mode": 0,  # decode uses no_mask
                "num_kv_heads": 1,  # MLA compressed attention: single KV head
            },
        )

        # QuantBatchMatmulV3 CSV has extra inputs (bias columns) beyond
        # the 2 TC shapes; tc_input_count=2 tells shape matching to only
        # compare the first 2 CSV inputs. BF16 BatchMatMulV2/BatchMatMulNd
        # CSV inputs already match the 2 TC shapes, so no override is needed.
        first_tc_input_count = 2 if first_kernel_type == "QuantBatchMatmulV3" else None

        # Fix MISS #6: NPU BatchMatMulV2/BatchMatMulNd/TransposeBatchMatMul
        # use heads-first layout (H,T,D), not (T,H,D).

        return [
            SubKernelSpec(
                kernel_type=first_kernel_type,
                input_shapes=[
                    (num_heads, num_tokens, qk_nope_head_dim),
                    (num_heads, qk_nope_head_dim, kv_lora_rank),
                ],
                dtype=dtype_str,
                tc_input_count=first_tc_input_count,
                alternate_kernel_types=alternate_kernel_types,
            ),
            fia_spec,
            SubKernelSpec(
                kernel_type="TransposeBatchMatMul",
                input_shapes=[
                    (num_heads, num_tokens, kv_lora_rank),
                    (num_heads, kv_lora_rank, v_head_dim_val),
                ],
                dtype=dtype_str,
            ),
        ]
    else:
        # Prefill: MatMulV2(kv_c@kv_b_proj) + FusedInferAttentionScore
        # vllm-ascend v0.18.0: MLA prefill uses FIA (unified, RING kernel removed)
        kv_b_proj = args[8]  # (kv_lora_rank, num_heads*(qk_nope_head_dim+v_head_dim))
        if kv_b_proj is None:
            logger.debug("MLA prefill: kv_b_proj is None, fallback to analytic")
            return None

        kv_lora_rank = kv_b_proj.shape[0]
        try:
            avg_seq_len = int(seq_lens.float().mean().item())
        except (RuntimeError, ValueError):
            avg_seq_len = 0

        # Fix MISS #4: FIA prefill uses TND layout: (num_tokens, num_heads, qk_nope_head_dim).
        # qk_head_dim = q.shape[2], qk_rope_head_dim = head_dim - kv_lora_rank,
        # qk_nope_head_dim = qk_head_dim - qk_rope_head_dim.
        qk_head_dim = q.shape[2]
        qk_rope_head_dim = head_dim - kv_lora_rank
        qk_nope_head_dim_pf = qk_head_dim - qk_rope_head_dim
        fia_q_shape_3d = (num_tokens, num_heads, qk_nope_head_dim_pf)

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
            SubKernelSpec(
                kernel_type="FusedInferAttentionScore",
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
            ),
        ]


def _decompose_mla(
    op_invoke_info: "OpInvokeInfo", mapping: dict
) -> Optional[List[SubKernelSpec]]:
    """Decompose multihead_latent_attention (BF16)."""
    return _decompose_mla_common(
        op_invoke_info,
        mapping,
        "BatchMatMulV2",
        alternate_kernel_types=["BatchMatMulNd"],
    )


def _decompose_mla_quant(
    op_invoke_info: "OpInvokeInfo", mapping: dict
) -> Optional[List[SubKernelSpec]]:
    """Decompose multihead_latent_attention_quant."""
    return _decompose_mla_common(op_invoke_info, mapping, "QuantBatchMatmulV3")


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

    if (
        hidden_states is None
        or q_a_proj is None
        or q_b_proj is None
        or kv_a_proj is None
    ):
        return None

    dtype_str = DTYPE_MAP.get(hidden_states.dtype)
    if dtype_str is None:
        return None

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    q_lora_rank = q_a_proj.shape[0]
    kv_proj_dim = kv_a_proj.shape[0]
    fused_proj_dim = q_lora_rank + kv_proj_dim

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
        q_b_proj_weight_shape = (num_heads_int * qk_head_dim_int, q_lora_rank)
    else:
        q_b_proj_weight_shape = tuple(q_b_proj.shape)

    return [
        SubKernelSpec(
            kernel_type=matmul_kernel_type,
            input_shapes=[(num_tokens, hidden_size), (fused_proj_dim, hidden_size)],
            dtype=matmul_dtype,
            tc_input_count=2,
        ),
        SubKernelSpec(
            kernel_type=matmul_kernel_type,
            input_shapes=[(num_tokens, q_lora_rank), q_b_proj_weight_shape],
            dtype=matmul_dtype,
            tc_input_count=2,
        ),
        # Fix MISS #3: KvRmsNormRopeCache NPU shape is 4D (T,1,1,D), not 2D (T,D).
        # KvRmsNormRopeCache always uses BF16 (dtype_str), not the quantized matmul dtype.
        SubKernelSpec(
            kernel_type="KvRmsNormRopeCache",
            input_shapes=[(num_tokens, 1, 1, kv_proj_dim)],
            dtype=dtype_str,
        ),
    ]


def _decompose_mlapo(
    op_invoke_info: "OpInvokeInfo", mapping: dict
) -> Optional[List[SubKernelSpec]]:
    """Decompose mlapo (BF16)."""
    return _decompose_mlapo_common(op_invoke_info, mapping, "MatMulV2", min_args=14)


def _decompose_mlapo_quant(
    op_invoke_info: "OpInvokeInfo", mapping: dict
) -> Optional[List[SubKernelSpec]]:
    """Decompose mlapo_quant."""
    return _decompose_mlapo_common(
        op_invoke_info, mapping, "QuantBatchMatmulV3", min_args=20
    )


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
    "tensor_cast.mlapo.default": _decompose_mlapo,
    "tensor_cast.mlapo_quant.default": _decompose_mlapo_quant,
}


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


@dataclass(frozen=True)
class _FiaColInfo:
    """Detected column names for FIA enriched CSV (cached per kernel_type)."""

    avg_seq_col: str
    has_sparse: bool
    has_kv_heads: bool
    has_layout: bool


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
        self._op_mapping = self._load_op_mapping()
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
        for col in (
            "Average Duration(us)",
            "Profiling Average Duration(us)",
        ):
            if col in df.columns:
                return col
        return "Duration(us)"

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
        any_csv_loaded = False

        for kernel_type in kernel_types:
            df = self._load_csv(kernel_type)
            if df is None:
                continue
            any_csv_loaded = True
            lat_col = self._latency_col(df)

            for _, row in df.iterrows():
                candidate = checker_fn(row, kernel_type, lat_col)
                if candidate is None:
                    continue
                if select == "first":
                    return candidate
                # select == "nearest": keep candidate with smallest distance
                if best is None or candidate.distance < best.distance:
                    best = candidate

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
        mask = (df["message_bytes"] == message_bytes) & (
            df["num_devices"] == num_devices
        )
        if topology_tier is not None and "topology_tier" in df.columns:
            mask = mask & (df["topology_tier"] == topology_tier)

        matched = df[mask]
        if not matched.empty:
            return (float(matched.iloc[0][lat_col]), False)

        # --- Interpolation fallback: bracket message_bytes ---
        device_mask = df["num_devices"] == num_devices
        if topology_tier is not None and "topology_tier" in df.columns:
            device_mask = device_mask & (df["topology_tier"] == topology_tier)
        candidates = df[device_mask]

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
        lat_lo = float(
            candidates.loc[candidates["message_bytes"] == mb_lo, lat_col].iloc[0]
        )
        if mb_lo == mb_hi:
            return (lat_lo, False)  # degenerate bracket = exact

        lat_hi = float(
            candidates.loc[candidates["message_bytes"] == mb_hi, lat_col].iloc[0]
        )

        # Alpha-beta interpolation: comm latency = alpha + message_bytes / bandwidth
        # Fit from ALL candidate data points (least-squares) rather than just the
        # bracket endpoints. This gives a global alpha-beta model for this
        # (num_devices, topology_tier) group, which handles the latency-dominated →
        # bandwidth-dominated transition more accurately than piecewise linear.
        all_mb = candidates["message_bytes"].values.astype(np.float64)
        all_lat = candidates[lat_col].values.astype(np.float64)

        if len(all_mb) >= 2:
            A = np.column_stack([np.ones_like(all_mb), all_mb])
            params, _, _, _ = np.linalg.lstsq(A, all_lat, rcond=None)
            interpolated = float(params[0] + params[1] * message_bytes)
        else:
            # Fallback: single-point, use that value
            interpolated = float(all_lat[0])

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
                simulation_shapes=[
                    list(s) for s, _ in self._extract_tensor_inputs(op_invoke_info)
                ],
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

        # Zero-cost ops: shape-only operations with no kernel execution
        if mapping.get("zero_cost"):
            _zc_sim_shapes = [
                list(s) for s, _ in self._extract_tensor_inputs(op_invoke_info)
            ]
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
                    simulation_shapes=[
                        list(s) for s, _ in self._extract_tensor_inputs(op_invoke_info)
                    ],
                    kernel_shapes=[],
                    shape_match_rule="accepted_miss",
                ),
            )

        return self._lookup_compute(op_invoke_info, mapping)

    # ---- Composite op lookup ----

    def _lookup_composite(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
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
            return self._lookup_composite_decomposed(
                op_invoke_info, mapping, decomposer
            )

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
        compute_hit = self._find_compute_match(
            compute_kernels, tc_inputs, tc_input_count
        )

        if compute_hit is None:
            return None

        compute_latency = compute_hit.latency_us
        compute_kernel_hit = compute_hit.kernel_type
        compute_csv_shapes = (
            compute_hit.shape_match_info.kernel_shapes
            if compute_hit.shape_match_info
            else []
        )
        compute_rule = (
            compute_hit.shape_match_info.shape_match_rule
            if compute_hit.shape_match_info
            else "unknown"
        )
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
        comm_latency = 0.0
        has_comm = False
        sub_kernel_durations = [(compute_kernel_hit, round(compute_latency, 2))]
        sub_kernel_shapes_info = [
            SubKernelShapeInfo(
                kernel_type=compute_kernel_hit,
                simulation_shapes=simulation_shapes,
                kernel_shapes=compute_csv_shapes,
                shape_match_rule=compute_rule,
            )
        ]
        for kernel_type in sub_kernels:
            if not kernel_type.startswith("hcom_"):
                continue
            has_comm = True
            lat = self._lookup_comm_for_composite(op_invoke_info, kernel_type)
            if lat is None:
                self.last_miss_reason = "comm_sub_kernel_miss"
                return None
            comm_latency += lat
            sub_kernel_durations.append((kernel_type, round(lat, 2)))
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
            source=QuerySource.MEASURED,
            details={
                "kernel_type": compute_kernel_hit,
                "sub_kernel_durations": sub_kernel_durations,
                "composite": True,
                "note": (
                    "compute + comm sub-kernels"
                    if has_comm
                    else "compute sub-kernel only"
                ),
            },
            sub_kernel_shapes=sub_kernel_shapes_info,
        )

    def _lookup_composite_decomposed(
        self,
        op_invoke_info: "OpInvokeInfo",
        mapping: dict,
        decomposer: Callable,
    ) -> Optional[QueryResult]:
        """Query composite op using registered decomposer (MLA/MLAPO).

        Calls the decomposer to get SubKernelSpec list, then queries each
        sub-kernel via _find_compute_match or _query_by_attn_params.

        Returns PARTIAL if some sub-kernels miss (with accumulated hit latency).
        """
        specs = decomposer(op_invoke_info, mapping)
        if not specs:
            self.last_miss_reason = "decompose_failed"
            return None

        total_latency = 0.0
        hit_kernels = []
        sub_kernel_durations = []
        sub_kernel_shapes_list = []
        missed_kernels = []

        for spec in specs:
            kernel_types = [spec.kernel_type] + (spec.alternate_kernel_types or [])

            if spec.query_mode == "attention" and spec.attention_params:
                result = self._query_by_attn_params(
                    kernel_types, spec.attention_params, spec.dtype
                )
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
            else:
                torch_dtype = None
                for k, v in DTYPE_MAP.items():
                    if v == spec.dtype:
                        torch_dtype = k
                        break
                if torch_dtype is None:
                    logger.debug(
                        "Unknown dtype %s for sub-kernel %s, skipping",
                        spec.dtype,
                        spec.kernel_type,
                    )
                    missed_kernels.append(spec.kernel_type)
                    continue
                tc_inputs = [(shape, torch_dtype) for shape in spec.input_shapes]
                hit = self._find_compute_match(
                    kernel_types,
                    tc_inputs,
                    spec.tc_input_count,
                    auto_truncate=True,
                )
                if hit is not None:
                    total_latency += hit.latency_us
                    hit_kernels.append(hit.kernel_type)
                    sub_kernel_durations.append(
                        (hit.kernel_type, round(hit.latency_us, 2))
                    )
                    sub_kernel_shapes_list.append(
                        SubKernelShapeInfo(
                            kernel_type=hit.kernel_type,
                            simulation_shapes=[list(s) for s in spec.input_shapes],
                            kernel_shapes=(
                                hit.shape_match_info.kernel_shapes
                                if hit.shape_match_info
                                else []
                            ),
                            shape_match_rule=(
                                hit.shape_match_info.shape_match_rule
                                if hit.shape_match_info
                                else "unknown"
                            ),
                        )
                    )
                else:
                    missed_kernels.append(spec.kernel_type)

        if missed_kernels:
            self.last_miss_reason = f"sub_kernel_miss:{','.join(missed_kernels)}"
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
            confidence=0.8,
            source=QuerySource.MEASURED,
            details={
                "kernel_type": ",".join(hit_kernels),
                "sub_kernel_durations": sub_kernel_durations,
                "composite": True,
                "note": "decomposed sub-kernels",
            },
            sub_kernel_shapes=sub_kernel_shapes_list,
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
        if q_shape_3d is None or target_avg_seq is None:
            return None

        target_sparse = params.get("sparse_mode")
        target_kv_heads = params.get("num_kv_heads")
        target_layout = params.get("input_layout")
        tc_N, tc_D = q_shape_3d[1], q_shape_3d[2]
        head_dim = tc_D

        # Per-CSV column detection cache: detected once per kernel_type,
        # reused for all rows in that CSV.
        _col_cache: Dict[str, Optional[_FiaColInfo]] = {}

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
            )
            _col_cache[kt] = info
            return info

        def checker(row: pd.Series, kt: str, lat_col: str) -> Optional[Candidate]:
            col_info = _detect_columns(row, kt)
            if col_info is None:
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
            csv_first_dtype = (
                csv_dtypes_str.split(";")[0].strip() if csv_dtypes_str else ""
            )
            if dtype_str != csv_first_dtype:
                return None
            if tc_N != csv_N or tc_D != csv_D:
                return None

            if (
                col_info.has_sparse
                and target_sparse is not None
                and int(row["Runtime sparse_mode"]) != target_sparse
            ):
                return None
            if (
                col_info.has_kv_heads
                and target_kv_heads is not None
                and int(row["Runtime num_key_value_heads"]) != target_kv_heads
            ):
                return None
            if col_info.has_layout and target_layout is not None:
                csv_layout = str(row.get("Runtime input_layout", "")).strip()
                if csv_layout and csv_layout != target_layout:
                    return None

            avg_seq_gap = abs(target_avg_seq - csv_avg_seq)
            if avg_seq_gap > _AVG_SEQ_LEN_TOLERANCE:
                return None

            tc_T = q_shape_3d[0]
            csv_T = csv_q_3d[0]
            if (
                tc_T != csv_T
                and not _is_block_padded(tc_T, csv_T)
                and not _is_block_padded(csv_T, tc_T)
            ):
                return None

            return Candidate(
                latency_us=float(row[lat_col]),
                kernel_type=kt,
                confidence=0.9,
                distance=float(avg_seq_gap),
            )

        hit = self._find_candidates(kernel_types, checker, select="nearest")
        if hit is None:
            if not self.last_miss_reason:
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

    def _lookup_comm_for_composite(
        self, op_invoke_info: "OpInvokeInfo", kernel_type: str
    ) -> Optional[float]:
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

        topology_tier = self._resolve_topology_tier(list(rank_group))

        result = self._query_comm_csv(
            kernel_type, message_bytes, num_devices, topology_tier
        )
        if result is None:
            return None
        return result[0]  # latency only, caller doesn't need is_interpolated

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

    def _lookup_comm(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
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

        result = self._query_comm_csv(
            kernel_type, message_bytes, num_devices, topology_tier
        )
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

    def _lookup_attention(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
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
        head_dim = (
            key.shape[-1] if isinstance(key, torch.Tensor) and key.ndim >= 1 else 0
        )

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
        tc_num_kv_heads = (
            key.shape[-2] if isinstance(key, torch.Tensor) and key.ndim >= 2 else None
        )

        # Derive input_layout from query shape ndim
        input_layout = (
            "TND" if query.ndim == 3 else "BNSD_NBSD" if query.ndim == 4 else None
        )

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
                simulation_shapes=[
                    list(s) for s, _ in self._extract_tensor_inputs(op_invoke_info)
                ],
                kernel_shapes=[],
                shape_match_rule="attention",
            ),
        )

    # ---- Elementwise op lookup (output-shape matching) ----

    def _lookup_elementwise(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
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
            csv_shapes = [
                list(s) for s in _parse_shape_str(str(row.get("Input Shapes", "")))
            ]
            return Candidate(
                latency_us=float(row[lat_col]),
                kernel_type=kt,
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

        hit = self._find_candidates(kernel_types, checker)
        if hit is None and not self.last_miss_reason:
            # Post-miss diagnosis: distinguish input_count_mismatch from
            # shape_mismatch (restores old _query_by_shapes behavior).
            primary = kernel_types[0] if kernel_types else "unknown"
            df = self._load_csv(primary)
            if df is not None and not df.empty:
                csv_first_shapes = _parse_shape_str(
                    str(df.iloc[0].get("Input Shapes", ""))
                )
                effective_csv_count = len(csv_first_shapes)
                effective_tc_count = len(tc_inputs)
                if effective_tc_input_count is not None:
                    effective_csv_count = min(
                        effective_csv_count, effective_tc_input_count
                    )
                    effective_tc_count = min(
                        effective_tc_count, effective_tc_input_count
                    )
                # SwiGlu: TC 2 inputs → CSV 1 (concat normalization)
                if (
                    primary in _SWIGLU_KERNELS
                    and effective_tc_count == 2
                    and effective_csv_count == 1
                ):
                    effective_tc_count = 1
                # ReshapeAndCache: TC 4 inputs → CSV 5 (split normalization)
                if (
                    primary in _RESHAPE_AND_CACHE_KERNELS
                    and effective_tc_count == 4
                    and effective_csv_count == 5
                ):
                    effective_tc_count = 5
                if effective_tc_count != effective_csv_count:
                    self.last_miss_reason = "input_count_mismatch"
                else:
                    self.last_miss_reason = "shape_mismatch"
            else:
                self.last_miss_reason = "csv_not_found"
        return hit

    # ---- MoE fused op lookup (EP Size matching) ----

    def _lookup_moe(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
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
                "DFC CSV has EP Size column but ep_size not configured. "
                "Pass parallel_config to ProfilingDataSource."
            )
            self.last_miss_reason = "ep_size_not_configured"
            return None

        tc_inputs = self._extract_tensor_inputs(op_invoke_info)
        tc_input_count = mapping.get("tc_input_count")
        if tc_input_count is not None:
            tc_inputs = tc_inputs[:tc_input_count]
        simulation_shapes = [list(s) for s, _ in tc_inputs]
        ep_size = self.ep_size

        def checker(row: pd.Series, kt: str, lat_col: str) -> Optional[Candidate]:
            rule = self._inputs_match(tc_inputs, row, kt, tc_input_count)
            if rule is None:
                return None
            if has_ep_col and ep_size is not None and int(row["EP Size"]) != ep_size:
                return None
            csv_shapes = [
                list(s) for s in _parse_shape_str(str(row.get("Input Shapes", "")))
            ]
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
        return QueryResult(
            latency_us=hit.latency_us,
            confidence=hit.confidence,
            source=QuerySource.MEASURED,
            details=hit.details,
            shape_match_info=hit.shape_match_info,
        )

    # ---- Compute op lookup ----

    def _lookup_compute(
        self, op_invoke_info: "OpInvokeInfo", mapping: dict
    ) -> Optional[QueryResult]:
        kernel_types = [mapping["kernel_type"]]
        for alt in mapping.get("alternate_kernel_types", []):
            if alt not in kernel_types:
                kernel_types.append(alt)

        tc_inputs = self._extract_tensor_inputs(op_invoke_info)
        tc_input_count = mapping.get("tc_input_count")
        if tc_input_count is not None:
            tc_inputs = tc_inputs[:tc_input_count]

        simulation_shapes = [list(s) for s, _ in tc_inputs]

        checker = self._make_compute_checker(
            tc_inputs, tc_input_count, simulation_shapes
        )

        hit = self._find_candidates(kernel_types, checker)
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
            details={"kernel_type": hit.kernel_type},
            shape_match_info=hit.shape_match_info,
        )

    def _extract_tensor_inputs(
        self, op_invoke_info: "OpInvokeInfo"
    ) -> List[Tuple[Tuple[int, ...], torch.dtype]]:
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
        if (
            kernel_type in _ROPE_KERNELS
            and len(tc_inputs) >= 2
            and len(csv_shapes) >= 2
        ):
            tc_inputs_normalized = _normalize_rope_inputs(tc_inputs)

        # SwiGlu input normalization: TC sends 2 inputs (gate, up),
        # profiling CSV has 1 fused input concatenated along last dim.
        if (
            kernel_type in _SWIGLU_KERNELS
            and len(tc_inputs) == 2
            and len(csv_shapes) == 1
        ):
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
        if (
            kernel_type in _RESHAPE_AND_CACHE_KERNELS
            and len(tc_inputs) == 4
            and len(csv_shapes) == 5
        ):
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
                    if (
                        compat_expected is None
                        or compat_csv is None
                        or compat_expected != compat_csv
                    ):
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
                    tc_shape_stripped
                    if len(tc_shape_stripped) == 3
                    else tc_shape
                    if len(tc_shape) == 3
                    else None
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
    def _shapes_match_with_padding(
        tc_shape: Tuple[int, ...], csv_shape: Tuple[int, ...]
    ) -> bool:
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
