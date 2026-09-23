# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""
Replay SparseFlashAttention cases from the performance database on Ascend NPU.

Purpose:
  Read SparseFlashAttention rows from
  profiling_database/data/{device}/vllm_ascend/{version}/SparseFlashAttention.csv,
  rebuild the recorded tensor inputs, construct legal auxiliary tensors,
  then execute the SparseFlashAttention custom operator.

CSV layout (9 inputs, 1 output):
  Input[0]: query         (num_tokens, num_heads, kv_lora_rank)  e.g. (102, 16, 512)         BF16
  Input[1]: key_cache     (total_blocks, block_size, 1, kv_lora_rank) e.g. (1766, 128, 1, 512)  BF16
  Input[2]: value_cache   (total_blocks, block_size, 1, kv_lora_rank) e.g. (1766, 128, 1, 512)  BF16
  Input[3]: topk_indices  (num_tokens, 1, topk)                  e.g. (102, 1, 2048)          INT32
  Input[4]: block_tables  (batch, max_blocks_per_seq)            e.g. (34, 1584)              INT32
  Input[5]: actual_seq_lengths_query (batch,)                    e.g. (1,)                    INT32
  Input[6]: actual_seq_lengths_kv    (batch,)                    e.g. (1,)                    INT32
  Input[7]: q_pe          (num_tokens, num_heads, qk_rope_dim)   e.g. (102, 16, 64)           BF16
  Input[8]: k_pe_cache    (total_blocks, block_size, 1, qk_rope_dim) e.g. (1766, 128, 1, 64)   BF16
  Output[0]: attn_output  (num_tokens, num_heads, kv_lora_rank)  e.g. (102, 16, 512)          BF16

Non-tensor args inferred:
  - softmax_scale: 1.0 / sqrt(kv_lora_rank + qk_rope_dim) (standard attention scale)

Runtime metadata:
  - avg_seq_len builds actual_seq_lengths_kv and bounds sparse indices
  - sparse_mode/input_layout/topk/block_size/num_key_value_heads configure or
    validate the replayed model state

microbench_api: torch.ops._C_ascend.npu_sparse_flash_attention
  (Custom Ascend fused kernel for sparse MLA attention)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    from ..fia_common import parse_runtime_int, parse_runtime_int_list, parse_shape_or_none
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fia_common import parse_runtime_int, parse_runtime_int_list, parse_shape_or_none

try:
    from .common import (
        get_runtime_modules,
        init_runtime,
        parse_list_field,
        parse_shape,
        build_input_tensor,
        normalize_dtype_name,
    )
    from .replay_framework import OpReplay
except ImportError:
    from common import (
        get_runtime_modules,
        init_runtime,
        parse_list_field,
        parse_shape,
        build_input_tensor,
        normalize_dtype_name,
    )
    from replay_framework import OpReplay


RUNTIME_AVG_SEQ_LEN = "Runtime avg_seq_len"
RUNTIME_SPARSE_MODE = "Runtime sparse_mode"
RUNTIME_NUM_KEY_VALUE_HEADS = "Runtime num_key_value_heads"
RUNTIME_INPUT_LAYOUT = "Runtime input_layout"
RUNTIME_TOPK = "Runtime topk"
RUNTIME_BLOCK_SIZE = "Runtime block_size"
RUNTIME_ACTUAL_SEQ_LENGTHS_SHAPE = "Runtime actual_seq_lengths_shape"
RUNTIME_ACTUAL_SEQ_LENGTHS_VALUES = "Runtime actual_seq_lengths_values"
RUNTIME_ACTUAL_SEQ_LENGTHS_KV_SHAPE = "Runtime actual_seq_lengths_kv_shape"
RUNTIME_ACTUAL_SEQ_LENGTHS_KV_VALUES = "Runtime actual_seq_lengths_kv_values"
RUNTIME_BLOCK_TABLE_SHAPE = "Runtime block_table_shape"
RUNTIME_BLOCK_TABLE_VALID_BLOCKS = "Runtime block_table_valid_blocks"
RUNTIME_NUM_HEADS = "Runtime num_heads"
RUNTIME_CACHE_LAYOUT = "Runtime cache_layout"
RUNTIME_KV_CACHE_MODE = "Runtime kv_cache_mode"
RUNTIME_METADATA_COMPLETENESS = "Runtime metadata_completeness"
RUNTIME_SPARSE_BLOCK_SIZE = "Runtime sparse_block_size"
RUNTIME_SPARSE_INDICES_PATTERN = "Runtime sparse_indices_pattern"
RUNTIME_SPARSE_INDICES_VALID_COUNT = "Runtime sparse_indices_valid_count"
RUNTIME_SPARSE_INDICES_SEED = "Runtime sparse_indices_seed"
INT32_INPUT_INDICES = (3, 4, 5, 6)


def _build_block_tables(
    batch: int,
    max_blocks_per_seq: int,
    total_blocks: int,
    valid_blocks: list[int],
):
    """Build a legal block_tables tensor."""
    runtime_torch, _ = get_runtime_modules()
    block_tables = runtime_torch.zeros(
        (batch, max_blocks_per_seq), dtype=runtime_torch.int32
    )
    cursor = 0
    for batch_index, block_count in enumerate(valid_blocks):
        block_tables[batch_index, :block_count] = runtime_torch.arange(
            cursor, cursor + block_count, dtype=runtime_torch.int32
        ).remainder(total_blocks)
        cursor += block_count
    return block_tables.npu()


def _build_int32_tensor(values: list[int]):
    runtime_torch, _ = get_runtime_modules()
    return runtime_torch.tensor(values, dtype=runtime_torch.int32).npu()


def _infer_query_lengths(query_shape: tuple[int, ...], input_layout: str, batch: int) -> list[int]:
    """Mirror vLLM cum_query_lens for TND and per-request lengths for BSND."""
    if batch <= 0:
        raise ValueError(f"SparseFlashAttention batch must be positive, got {batch}")
    if input_layout == "TND":
        if len(query_shape) != 3:
            raise ValueError(f"TND query must be rank 3, got {query_shape}")
        total_tokens = query_shape[0]
        base, extra = divmod(total_tokens, batch)
        lengths = [base + (1 if index < extra else 0) for index in range(batch)]
        cumulative = []
        total = 0
        for length in lengths:
            total += length
            cumulative.append(total)
        return cumulative
    if input_layout == "BSND":
        if len(query_shape) != 4 or query_shape[0] != batch:
            raise ValueError(f"BSND query batch mismatch: query={query_shape}, batch={batch}")
        return [query_shape[1]] * batch
    raise ValueError(f"Unsupported SparseFlashAttention input layout: {input_layout}")


def _build_topk_indices(
    topk_shape: tuple[int, ...],
    query_lengths: list[int],
    kv_lengths: list[int],
    *,
    pattern: str,
    valid_count: int,
    seed: int,
):
    """Build deterministic, context-spanning sparse indices.

    The profiling CSV does not preserve LightningIndexer output values. Evenly
    spacing indices across the recorded runtime sequence length is a stable
    approximation of the model's dispersed sparse reads and avoids the highly
    cache-friendly contiguous-prefix pattern used by the old replay.
    """
    topk = topk_shape[-1]
    if topk <= 0:
        raise ValueError(f"SparseFlashAttention topk must be positive, got {topk}")
    runtime_torch, _ = get_runtime_modules()
    per_sequence_query = [query_lengths[0]] + [
        query_lengths[index] - query_lengths[index - 1]
        for index in range(1, len(query_lengths))
    ]
    rows = []
    for sequence_index, (query_count, kv_length) in enumerate(
        zip(per_sequence_query, kv_lengths)
    ):
        if query_count <= 0:
            continue
        effective_count = min(topk, valid_count, kv_length)
        if effective_count <= 0:
            raise ValueError("SparseFlashAttention sparse indices need at least one valid entry")
        if pattern == "uniform":
            base = runtime_torch.arange(effective_count, dtype=runtime_torch.int32)
            base = base * kv_length // effective_count
        elif pattern == "contiguous":
            base = runtime_torch.arange(effective_count, dtype=runtime_torch.int32)
        elif pattern == "random":
            generator = runtime_torch.Generator(device="cpu")
            generator.manual_seed(seed + sequence_index)
            base = runtime_torch.randperm(kv_length, generator=generator)[:effective_count]
            base = runtime_torch.sort(base).values.to(runtime_torch.int32)
        else:
            raise ValueError(f"Unsupported sparse_indices_pattern: {pattern}")
        if effective_count < topk:
            base = runtime_torch.cat(
                (base, runtime_torch.full((topk - effective_count,), -1, dtype=runtime_torch.int32))
            )
        rows.append(base.reshape(1, 1, topk).expand(query_count, 1, topk))
    if not rows:
        raise ValueError("SparseFlashAttention sparse indices need active query rows")
    return runtime_torch.cat(rows, dim=0).contiguous().npu()


def _is_complete_metadata(row: dict[str, str]) -> bool:
    value = (row.get(RUNTIME_METADATA_COMPLETENESS, "") or "").strip().lower()
    return value not in {"", "legacy"}


def _required_runtime_value(row: dict[str, str], column: str) -> str:
    value = (row.get(column, "") or "").strip()
    if not value:
        raise ValueError(f"Complete SparseFlashAttention row is missing {column}")
    return value


def resolve_case_metadata(
    row: dict[str, str],
    input_shapes: list[tuple[int, ...]],
    input_formats: list[str],
    input_dtypes: list[str],
) -> dict[str, object]:
    """Resolve and validate the runtime state recorded for one SFA row."""
    if not (len(input_shapes) == len(input_formats) == len(input_dtypes) == 9):
        raise ValueError("SparseFlashAttention expects exactly 9 complete inputs")

    for index in INT32_INPUT_INDICES:
        if input_dtypes[index] != "DT_INT32" or input_formats[index] != "ND":
            raise ValueError(
                f"SparseFlashAttention input[{index}] must be INT32/ND, got "
                f"{input_dtypes[index]}/{input_formats[index]}"
            )

    query_shape = input_shapes[0]
    key_shape = input_shapes[1]
    value_shape = input_shapes[2]
    topk_shape = input_shapes[3]
    block_table_shape = input_shapes[4]
    if len(key_shape) != 4 or len(block_table_shape) != 2:
        raise ValueError(f"SFA paged KV shapes are invalid: key={key_shape}, block_table={block_table_shape}")
    if value_shape != key_shape or input_dtypes[2] != input_dtypes[1] or input_formats[2] != input_formats[1]:
        raise ValueError("SFA key/value metadata must match because this adapter aliases the same KV tensor")

    total_blocks, shape_block_size, shape_kv_heads, _ = key_shape
    batch, max_blocks_per_seq = block_table_shape
    input_layout = (row.get(RUNTIME_INPUT_LAYOUT, "") or "").strip() or ("TND" if len(query_shape) == 3 else "BSND")
    complete_metadata = _is_complete_metadata(row)
    if complete_metadata:
        for column in (
            RUNTIME_ACTUAL_SEQ_LENGTHS_SHAPE,
            RUNTIME_ACTUAL_SEQ_LENGTHS_VALUES,
            RUNTIME_ACTUAL_SEQ_LENGTHS_KV_SHAPE,
            RUNTIME_ACTUAL_SEQ_LENGTHS_KV_VALUES,
            RUNTIME_BLOCK_TABLE_SHAPE,
            RUNTIME_BLOCK_TABLE_VALID_BLOCKS,
            RUNTIME_NUM_HEADS,
            RUNTIME_CACHE_LAYOUT,
            RUNTIME_KV_CACHE_MODE,
            RUNTIME_SPARSE_BLOCK_SIZE,
            RUNTIME_SPARSE_INDICES_PATTERN,
            RUNTIME_SPARSE_INDICES_VALID_COUNT,
            RUNTIME_SPARSE_INDICES_SEED,
        ):
            _required_runtime_value(row, column)

    runtime_block_size = parse_runtime_int(row.get(RUNTIME_BLOCK_SIZE))
    runtime_kv_heads = parse_runtime_int(row.get(RUNTIME_NUM_KEY_VALUE_HEADS))
    runtime_topk = parse_runtime_int(row.get(RUNTIME_TOPK))
    runtime_avg_seq_len = parse_runtime_int(row.get(RUNTIME_AVG_SEQ_LEN))
    block_size = shape_block_size if runtime_block_size is None else runtime_block_size
    kv_heads = shape_kv_heads if runtime_kv_heads is None else runtime_kv_heads
    topk = topk_shape[-1] if runtime_topk is None else runtime_topk
    sparse_mode = parse_runtime_int(row.get(RUNTIME_SPARSE_MODE))
    sparse_mode = 3 if sparse_mode is None else sparse_mode
    max_seq_capacity = max_blocks_per_seq * shape_block_size
    avg_seq_len = max_seq_capacity if runtime_avg_seq_len is None else runtime_avg_seq_len

    if block_size != shape_block_size:
        raise ValueError(f"Runtime block_size={block_size} conflicts with key shape {key_shape}")
    if kv_heads != shape_kv_heads:
        raise ValueError(f"Runtime num_key_value_heads={kv_heads} conflicts with key shape {key_shape}")
    if topk_shape[-1] != topk or topk_shape[-2] != kv_heads:
        raise ValueError(f"Runtime topk/KV heads conflict with sparse_indices shape {topk_shape}")
    if sparse_mode not in (0, 3):
        raise ValueError(f"Unsupported SparseFlashAttention sparse_mode: {sparse_mode}")
    if avg_seq_len <= 0 or avg_seq_len > max_seq_capacity:
        raise ValueError(
            f"Runtime avg_seq_len={avg_seq_len} exceeds paged KV capacity {max_seq_capacity}"
        )
    query_lengths = parse_runtime_int_list(row.get(RUNTIME_ACTUAL_SEQ_LENGTHS_VALUES))
    if query_lengths is None:
        query_lengths = _infer_query_lengths(query_shape, input_layout, batch)
    kv_lengths = parse_runtime_int_list(row.get(RUNTIME_ACTUAL_SEQ_LENGTHS_KV_VALUES))
    if kv_lengths is None:
        kv_lengths = [avg_seq_len] * batch
    valid_blocks = parse_runtime_int_list(row.get(RUNTIME_BLOCK_TABLE_VALID_BLOCKS))
    if valid_blocks is None:
        valid_blocks = [math.ceil(length / block_size) for length in kv_lengths]
    runtime_block_table_shape = parse_shape_or_none(row.get(RUNTIME_BLOCK_TABLE_SHAPE))
    runtime_query_shape = parse_runtime_int(row.get(RUNTIME_ACTUAL_SEQ_LENGTHS_SHAPE))
    runtime_kv_shape = parse_runtime_int(row.get(RUNTIME_ACTUAL_SEQ_LENGTHS_KV_SHAPE))
    runtime_num_heads = parse_runtime_int(row.get(RUNTIME_NUM_HEADS))
    cache_layout = (row.get(RUNTIME_CACHE_LAYOUT, "") or "").strip() or "PA_BSND"
    kv_cache_mode = (row.get(RUNTIME_KV_CACHE_MODE, "") or "").strip() or "paged"
    sparse_block_size = parse_runtime_int(row.get(RUNTIME_SPARSE_BLOCK_SIZE)) or 1
    sparse_indices_pattern = (
        (row.get(RUNTIME_SPARSE_INDICES_PATTERN, "") or "").strip().lower() or "uniform"
    )
    sparse_indices_valid_count = (
        parse_runtime_int(row.get(RUNTIME_SPARSE_INDICES_VALID_COUNT)) or min(topk, avg_seq_len)
    )
    sparse_indices_seed = parse_runtime_int(row.get(RUNTIME_SPARSE_INDICES_SEED)) or 0

    if runtime_block_table_shape is not None and runtime_block_table_shape != block_table_shape:
        raise ValueError(
            f"Runtime block_table_shape={runtime_block_table_shape} conflicts with input {block_table_shape}"
        )
    if len(query_lengths) != batch or runtime_query_shape not in (None, batch):
        raise ValueError("Runtime actual query sequence metadata must match block-table batch")
    if len(kv_lengths) != batch or runtime_kv_shape not in (None, batch):
        raise ValueError("Runtime actual KV sequence metadata must match block-table batch")
    if len(valid_blocks) != batch or any(count < 0 or count > max_blocks_per_seq for count in valid_blocks):
        raise ValueError("Runtime block_table_valid_blocks must contain one legal count per request")
    if query_lengths[-1] != query_shape[0] or any(
        query_lengths[index] < (query_lengths[index - 1] if index else 0)
        for index in range(batch)
    ):
        raise ValueError("Runtime actual query sequence values do not cover the TND query tensor")
    if any(length < 0 or length > max_seq_capacity for length in kv_lengths):
        raise ValueError("Runtime actual KV sequence values exceed paged cache capacity")
    previous_query = 0
    for query_end, kv_length in zip(query_lengths, kv_lengths):
        query_count = query_end - previous_query
        if (query_count > 0) != (kv_length > 0):
            raise ValueError("Runtime query and KV sequence activity must match per request")
        previous_query = query_end
    if int(sum(kv_lengths) / len(kv_lengths)) != avg_seq_len:
        raise ValueError("Runtime avg_seq_len conflicts with actual KV sequence values")
    expected_valid_blocks = [math.ceil(length / block_size) for length in kv_lengths]
    if valid_blocks != expected_valid_blocks:
        raise ValueError(
            f"Runtime block_table_valid_blocks={valid_blocks} conflicts with KV lengths {kv_lengths}"
        )
    required_blocks = max(valid_blocks, default=0)
    if required_blocks > total_blocks:
        raise ValueError(
            f"Runtime state needs {required_blocks} KV blocks for one request "
            f"but cache only has {total_blocks}"
        )
    if runtime_num_heads not in (None, query_shape[-2]):
        raise ValueError(f"Runtime num_heads={runtime_num_heads} conflicts with query shape {query_shape}")
    if cache_layout != "PA_BSND" or kv_cache_mode != "paged":
        raise ValueError(f"Unsupported SFA cache state: layout={cache_layout}, mode={kv_cache_mode}")
    if sparse_block_size <= 0:
        raise ValueError(f"Runtime sparse_block_size must be positive, got {sparse_block_size}")
    if not 1 <= sparse_indices_valid_count <= topk:
        raise ValueError(
            f"Runtime sparse_indices_valid_count must be in [1, {topk}], got {sparse_indices_valid_count}"
        )
    if input_shapes[5] != (batch,) or input_shapes[6] != (batch,):
        raise ValueError(
            "SFA sequence-length input shapes must match block-table batch: "
            f"query={input_shapes[5]}, kv={input_shapes[6]}, batch={batch}"
        )

    expected_topk_shape = (
        (query_shape[0], kv_heads, topk)
        if input_layout == "TND"
        else (query_shape[0], query_shape[1], kv_heads, topk)
    )
    if topk_shape != expected_topk_shape:
        raise ValueError(f"SFA sparse_indices shape mismatch: expected {expected_topk_shape}, got {topk_shape}")

    return {
        "avg_seq_len": avg_seq_len,
        "block_size": block_size,
        "cache_layout": cache_layout,
        "input_layout": input_layout,
        "kv_cache_mode": kv_cache_mode,
        "kv_lengths": kv_lengths,
        "kv_heads": kv_heads,
        "query_lengths": query_lengths,
        "sparse_block_size": sparse_block_size,
        "sparse_indices_pattern": sparse_indices_pattern,
        "sparse_indices_seed": sparse_indices_seed,
        "sparse_indices_valid_count": sparse_indices_valid_count,
        "sparse_mode": sparse_mode,
        "topk": topk,
        "valid_blocks": valid_blocks,
    }


def build_case(row: dict[str, str]):
    init_runtime()
    runtime_torch, _ = get_runtime_modules()
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = [
        normalize_dtype_name(item) for item in parse_list_field(row["Input Data Types"])
    ]
    metadata = resolve_case_metadata(row, input_shapes, input_formats, input_dtypes)

    query = build_input_tensor(shape=input_shapes[0], input_format=input_formats[0], dtype_name=input_dtypes[0])
    key_cache = build_input_tensor(shape=input_shapes[1], input_format=input_formats[1], dtype_name=input_dtypes[1])
    value_cache = key_cache
    q_pe = build_input_tensor(shape=input_shapes[7], input_format=input_formats[7], dtype_name=input_dtypes[7])
    k_pe_cache = build_input_tensor(shape=input_shapes[8], input_format=input_formats[8], dtype_name=input_dtypes[8])

    cache_shape = input_shapes[1]
    total_blocks = cache_shape[0]
    kv_lora_rank = cache_shape[-1]
    qk_rope_dim = input_shapes[8][-1]
    bt_shape = input_shapes[4]
    batch = bt_shape[0]
    max_blocks_per_seq = bt_shape[-1]

    block_tables = _build_block_tables(
        batch, max_blocks_per_seq, total_blocks, metadata["valid_blocks"]
    )
    actual_seq_lengths_query = _build_int32_tensor(metadata["query_lengths"])
    actual_seq_lengths_kv = _build_int32_tensor(metadata["kv_lengths"])
    topk_indices = _build_topk_indices(
        input_shapes[3],
        metadata["query_lengths"],
        metadata["kv_lengths"],
        pattern=metadata["sparse_indices_pattern"],
        valid_count=metadata["sparse_indices_valid_count"],
        seed=metadata["sparse_indices_seed"],
    )

    softmax_scale = 1.0 / math.sqrt(kv_lora_rank + qk_rope_dim)

    return {
        "inputs": [
            query,
            key_cache,
            value_cache,
            topk_indices,
            block_tables,
            actual_seq_lengths_query,
            actual_seq_lengths_kv,
            q_pe,
            k_pe_cache,
        ],
        "kwargs": {
            "scale_value": softmax_scale,
            "sparse_block_size": metadata["sparse_block_size"],
            "layout_query": metadata["input_layout"],
            "layout_kv": metadata["cache_layout"],
            "sparse_mode": metadata["sparse_mode"],
        },
        "runtime_metadata": metadata,
        "api": op.resolve_api(),
    }


def run_case(case):
    api = case["api"]
    inputs = case["inputs"]
    kwargs = case["kwargs"]
    return api(
        query=inputs[0],
        key=inputs[1],
        value=inputs[2],
        sparse_indices=inputs[3],
        block_table=inputs[4],
        actual_seq_lengths_query=inputs[5],
        actual_seq_lengths_kv=inputs[6],
        query_rope=inputs[7],
        key_rope=inputs[8],
        **kwargs,
    )


def format_success(csv_path, row_index: int, row: dict[str, str], case, _result) -> str:
    query = case["inputs"][0]
    key_cache = case["inputs"][1]
    topk_indices = case["inputs"][3]
    scale = case["kwargs"]["scale_value"]
    metadata = case["runtime_metadata"]
    return (
        f"[OK] {csv_path}:{row_index} "
        f"query={tuple(query.shape)} kv_cache={tuple(key_cache.shape)} "
        f"topk={tuple(topk_indices.shape)} scale={scale:.6f} "
        f"avg_seq_len={metadata['avg_seq_len']} layout={metadata['input_layout']} "
        f"dtypes={row['Input Data Types']}"
    )


op = OpReplay(
    kernel_type="SparseFlashAttention",
    api_path="torch.ops._C_ascend.npu_sparse_flash_attention",
    description=(
        "Run SparseFlashAttention workload replay on Ascend NPU.\n"
        "Reads SparseFlashAttention.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs input tensors,\n"
        "builds legal block_tables/seq_lens/topk_indices, then runs\n"
        "torch.ops._C_ascend.npu_sparse_flash_attention().\n\n"
        "This operator performs sparse MLA attention:\n"
        "it only attends to the top-K most relevant tokens selected\n"
        "by LightningIndexer, instead of the full KV cache."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/SparseFlashAttention_run.py "
        "--database-path tensor_cast/performance_model/profiling_database/"
        "data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/test",
    ],
    version_help="vLLM-Ascend version, e.g. 0.19.0.",
    build_case=build_case,
    run_case=run_case,
    format_success=format_success,
    runtime_warmup_count=3,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
