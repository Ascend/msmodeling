"""
Replay LightningIndexer cases from the performance database on Ascend NPU.

Purpose:
  Read LightningIndexer rows from
  profiling_database/data/{device}/vllm_ascend/{version}/LightningIndexer.csv,
  rebuild the recorded tensor inputs, construct legal auxiliary tensors,
  then execute the LightningIndexer custom operator.

CSV layout (6 inputs, 2 outputs):
  Input[0]: query         (num_tokens, num_heads, head_dim) e.g. (102, 32, 128) BF16
  Input[1]: indexer_cache  (total_blocks, block_size, 1, head_dim) e.g. (1766, 128, 1, 128) BF16
  Input[2]: weights        (num_tokens, num_heads)           e.g. (102, 32)     BF16
  Input[3]: block_tables   (batch,)                          e.g. (34,)         INT32
  Input[4]: seq_lens       (batch,)                          e.g. (34,)         INT32
  Input[5]: context_lens   (batch, context_len)              e.g. (34, 1584)    INT32
  Output[0]: topk_indices  (num_tokens, 1, topk)             e.g. (102, 1, 2048) INT32
  Output[1]: topk_weights  (num_tokens, 1, topk)             e.g. (102, 1, 2048) BF16

Non-tensor args inferred:
  - index_topk: derived from output shape[-1] (typically 2048)

microbench_api: torch_npu.npu_lightning_indexer
  Maps to aclnnLightningIndexer (ops-transformer, no gSize constraint).
  GLM5 profiling uses this path (sfa_v1.py:448 use_torch_npu_lightning_indexer=True),
  not torch.ops._C_ascend.npu_lightning_indexer (which maps to LightningIndexerVllm
  with a gSize==64 constraint).
"""

from __future__ import annotations

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


def _build_block_tables(batch: int, max_blocks_per_seq: int, total_blocks: int):
    """Build a legal block_tables tensor mapping batch entries to cache blocks."""
    runtime_torch, _ = get_runtime_modules()
    # Fill with sequential block indices, wrapping within total_blocks
    block_tables = runtime_torch.arange(
        0, batch * max_blocks_per_seq, dtype=runtime_torch.int32
    ).remainder(total_blocks).reshape(batch, max_blocks_per_seq)
    return block_tables.npu()


def _build_seq_lengths(
    batch: int, context_len: int, *, num_tokens: int = 0, cumulative: bool = True
):
    """Build actual_seq_lengths tensors.

    When *cumulative* is True the result mirrors cum_query_lens (prefix sum of
    per-request query token counts).  When *cumulative* is False the result
    mirrors seq_lens (absolute per-request KV cache lengths), matching the
    sfa_v1 convention where ``actual_seq_lengths_key = seq_lens``.
    """
    runtime_torch, _ = get_runtime_modules()
    if num_tokens > 0:
        base = num_tokens // batch
        rem = num_tokens % batch
        per_seq = runtime_torch.full((batch,), base, dtype=runtime_torch.int32)
        if rem > 0:
            per_seq[:rem] += 1
        if cumulative:
            return runtime_torch.cumsum(per_seq, dim=0).to(runtime_torch.int32).npu()
        return per_seq.npu()
    return runtime_torch.full(
        (batch,), context_len, dtype=runtime_torch.int32, device="npu"
    )


def build_case(row: dict[str, str]):
    init_runtime()
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = [
        normalize_dtype_name(item) for item in parse_list_field(row["Input Data Types"])
    ]
    output_shapes = [
        parse_shape(item) for item in parse_list_field(row["Output Shapes"])
    ]

    if len(input_shapes) != 6:
        raise ValueError(
            f"LightningIndexer expects exactly 6 inputs, got {len(input_shapes)}"
        )

    query = build_input_tensor(shape=input_shapes[0], input_format=input_formats[0], dtype_name=input_dtypes[0])
    indexer_cache = build_input_tensor(shape=input_shapes[1], input_format=input_formats[1], dtype_name=input_dtypes[1])
    weights = build_input_tensor(shape=input_shapes[2], input_format=input_formats[2], dtype_name=input_dtypes[2])

    cache_shape = input_shapes[1]
    total_blocks = cache_shape[0]
    block_size = cache_shape[1]
    batch = input_shapes[3][0]

    max_blocks_per_seq = input_shapes[5][-1]

    num_tokens = input_shapes[0][0]
    actual_seq_lengths_query = _build_seq_lengths(
        batch, max_blocks_per_seq * block_size, num_tokens=num_tokens
    )
    actual_seq_lengths_key = _build_seq_lengths(
        batch, max_blocks_per_seq * block_size,
        num_tokens=total_blocks * block_size, cumulative=False,
    )
    block_tables = _build_block_tables(batch, max_blocks_per_seq, total_blocks)

    index_topk = output_shapes[0][-1] if output_shapes else 2048

    return {
        "inputs": [
            query,
            indexer_cache,
            weights,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            block_tables,
        ],
        "kwargs": {
            "layout_query": "TND",
            "layout_key": "PA_BSND",
            "sparse_count": index_topk,
            "sparse_mode": 3,
        },
        "api": op.resolve_api(),
    }


def run_case(case):
    api = case["api"]
    inputs = case["inputs"]
    kwargs = case["kwargs"]
    result = api(
        query=inputs[0],
        key=inputs[1],
        weights=inputs[2],
        actual_seq_lengths_query=inputs[3],
        actual_seq_lengths_key=inputs[4],
        block_table=inputs[5],
        **kwargs,
    )
    # torch_npu.npu_lightning_indexer returns (topk_indices, topk_weights).
    # Unwrap the first element.
    return result[0] if isinstance(result, (tuple, list)) else result


def format_success(csv_path, row_index: int, row: dict[str, str], case, _result) -> str:
    query = case["inputs"][0]
    cache = case["inputs"][1]
    topk = case["kwargs"]["sparse_count"]
    return (
        f"[OK] {csv_path}:{row_index} "
        f"query={tuple(query.shape)} cache={tuple(cache.shape)} "
        f"sparse_count={topk} "
        f"dtypes={row['Input Data Types']}"
    )


op = OpReplay(
    kernel_type="LightningIndexer",
    api_path="torch_npu.npu_lightning_indexer",
    description=(
        "Run LightningIndexer workload replay on Ascend NPU.\n"
        "Reads LightningIndexer.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs input tensors,\n"
        "builds legal block_tables and seq_lens, then runs\n"
        "torch_npu.npu_lightning_indexer() (GLM5 profiling path).\n\n"
        "This is the fused DSA indexer kernel from ops-transformer:\n"
        "it computes Q*K scores, applies ReLU + scaling, reduces,\n"
        "and selects top-K indices for sparse attention."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/LightningIndexer_run.py "
        "--database-path tensor_cast/performance_model/profiling_database/"
        "data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/test",
    ],
    version_help="vLLM-Ascend version, e.g. 0.19.0.",
    build_case=build_case,
    run_case=run_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
