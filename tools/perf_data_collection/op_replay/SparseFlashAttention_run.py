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
  Input[5]: batch_size    scalar                                 e.g. (34,)                   INT32
  Input[6]: seq_lens      scalar                                 e.g. (34,)                   INT32
  Input[7]: q_pe          (num_tokens, num_heads, qk_rope_dim)   e.g. (102, 16, 64)           BF16
  Input[8]: k_pe_cache    (total_blocks, block_size, 1, qk_rope_dim) e.g. (1766, 128, 1, 64)   BF16
  Output[0]: attn_output  (num_tokens, num_heads, kv_lora_rank)  e.g. (102, 16, 512)          BF16

Non-tensor args inferred:
  - softmax_scale: 1.0 / sqrt(kv_lora_rank + qk_rope_dim) (standard attention scale)

microbench_api: torch.ops._C_ascend.npu_sparse_flash_attention
  (Custom Ascend fused kernel for sparse MLA attention in GLM5)
"""

from __future__ import annotations

import math

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
    """Build a legal block_tables tensor."""
    runtime_torch, _ = get_runtime_modules()
    # Fill with sequential block indices, wrapping within total_blocks
    block_tables = runtime_torch.arange(
        0, batch * max_blocks_per_seq, dtype=runtime_torch.int32
    ).remainder(total_blocks).reshape(batch, max_blocks_per_seq)
    return block_tables.npu()


def _build_seq_lengths(batch: int, max_seq_len: int, *, num_tokens: int = 0):
    """Build actual_seq_lengths tensors for SparseFlashAttention.

    In vLLM (sfa_v1.py), actual_seq_lengths_query is cum_query_lens (prefix
    sum of per-request query token counts) and actual_seq_lengths_kv is
    seq_lens (per-request KV cache lengths).  Using max_seq_len for query
    lengths tells the kernel to read millions of query tokens from a tensor
    that only holds num_tokens entries.
    """
    runtime_torch, _ = get_runtime_modules()
    if num_tokens > 0:
        base = num_tokens // batch
        rem = num_tokens % batch
        per_seq = runtime_torch.full((batch,), base, dtype=runtime_torch.int32)
        if rem > 0:
            per_seq[:rem] += 1
        return runtime_torch.cumsum(per_seq, dim=0).to(runtime_torch.int32).npu()
    return runtime_torch.full(
        (batch,), max_seq_len, dtype=runtime_torch.int32, device="npu"
    )


def _build_topk_indices(
    num_tokens: int,
    topk: int,
    max_seq_len: int,
):
    """Build sorted topk token-position indices for sparse flash attention.

    The kernel requires sorted, valid token positions in [0, max_seq_len).
    Random indices cause aicore exceptions because the kernel makes OOB
    memory accesses when gathering KV cache entries in unsorted order.
    """
    if topk <= 0:
        raise ValueError(f"SparseFlashAttention topk must be positive, got {topk}")
    if max_seq_len <= 0:
        raise ValueError(
            f"SparseFlashAttention max_seq_len must be positive, got {max_seq_len}"
        )
    effective_topk = min(topk, max_seq_len)
    runtime_torch, _ = get_runtime_modules()
    base = runtime_torch.arange(0, effective_topk, dtype=runtime_torch.int32)
    if effective_topk < topk:
        padding = runtime_torch.full(
            (topk - effective_topk,),
            effective_topk - 1,
            dtype=runtime_torch.int32,
        )
        base = runtime_torch.cat((base, padding))
    topk_indices = base.unsqueeze(0).unsqueeze(1).expand(num_tokens, 1, topk).contiguous()
    return topk_indices.npu()


def build_case(row: dict[str, str]):
    init_runtime()
    runtime_torch, _ = get_runtime_modules()
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = [
        normalize_dtype_name(item) for item in parse_list_field(row["Input Data Types"])
    ]

    if len(input_shapes) != 9:
        raise ValueError(
            f"SparseFlashAttention expects exactly 9 inputs, got {len(input_shapes)}"
        )

    query = build_input_tensor(shape=input_shapes[0], input_format=input_formats[0], dtype_name=input_dtypes[0])
    key_cache = build_input_tensor(shape=input_shapes[1], input_format=input_formats[1], dtype_name=input_dtypes[1])
    value_cache = build_input_tensor(shape=input_shapes[2], input_format=input_formats[2], dtype_name=input_dtypes[2])
    q_pe = build_input_tensor(shape=input_shapes[7], input_format=input_formats[7], dtype_name=input_dtypes[7])
    k_pe_cache = build_input_tensor(shape=input_shapes[8], input_format=input_formats[8], dtype_name=input_dtypes[8])

    cache_shape = input_shapes[1]
    total_blocks = cache_shape[0]
    block_size = cache_shape[1]
    kv_lora_rank = cache_shape[-1]
    qk_rope_dim = input_shapes[8][-1]
    num_tokens = input_shapes[0][0]

    topk_shape = input_shapes[3]
    topk = topk_shape[-1]

    bt_shape = input_shapes[4]
    batch = bt_shape[0]
    max_blocks_per_seq = bt_shape[-1]

    block_tables = _build_block_tables(batch, max_blocks_per_seq, total_blocks)
    max_seq_len = max_blocks_per_seq * block_size
    actual_seq_lengths_query = _build_seq_lengths(batch, max_seq_len, num_tokens=num_tokens)
    actual_seq_lengths_kv = _build_seq_lengths(batch, max_seq_len)
    topk_indices = _build_topk_indices(num_tokens, topk, max_seq_len)

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
            "sparse_block_size": 1,
            "layout_query": "TND",
            "layout_kv": "PA_BSND",
            "sparse_mode": 3,
        },
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
    return (
        f"[OK] {csv_path}:{row_index} "
        f"query={tuple(query.shape)} kv_cache={tuple(key_cache.shape)} "
        f"topk={tuple(topk_indices.shape)} scale={scale:.6f} "
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
        "This operator performs sparse MLA attention in GLM5:\n"
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
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
