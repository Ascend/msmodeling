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

"""Replay the DeepSeek-V4 W8A8 lightning indexer main kernel."""

from __future__ import annotations

try:
    from .common import get_runtime_modules
    from .replay_framework import OpReplay
except ImportError:
    from common import get_runtime_modules
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    from tensor_cast.performance_model.profiling_database.backend_projector import (
        v4_profile_shapes,
        v4_full_prefill_metadata_matches,
        validate_v4_indexer_physical_shapes,
    )

    shapes = v4_profile_shapes(row["Input Shapes"])
    output_shapes = v4_profile_shapes(row["Output Shapes"])
    if not validate_v4_indexer_physical_shapes(shapes, output_shapes):
        raise ValueError("VllmQuantLightningIndexer requires a verified single-request full-prefill ABI")
    if not v4_full_prefill_metadata_matches(row, shapes[0][0]):
        raise ValueError("VllmQuantLightningIndexer replay only supports single-request full prefill")
    runtime_torch, _ = get_runtime_modules()
    tensors = op.build_inputs(row)
    if len(tensors) != 9:
        raise ValueError("VllmQuantLightningIndexer expects nine recorded tensor inputs")

    query, key, weights, query_scale, key_scale = tensors[:5]
    query_tokens = int(query.shape[0])
    actual_q = runtime_torch.tensor([query_tokens], dtype=runtime_torch.int32, device="npu")
    actual_k = runtime_torch.tensor([query_tokens], dtype=runtime_torch.int32, device="npu")
    block_table = runtime_torch.arange(shapes[7][-1], dtype=runtime_torch.int32, device="npu").reshape(shapes[7])
    sparse_count = int(output_shapes[0][-1])
    metadata_api = runtime_torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata
    metadata = metadata_api(
        num_heads_q=int(query.shape[1]),
        num_heads_k=int(key.shape[-2]),
        head_dim=int(query.shape[-1]),
        query_quant_mode=0,
        key_quant_mode=0,
        actual_seq_lengths_query=actual_q,
        actual_seq_lengths_key=actual_k,
        batch_size=1,
        max_seqlen_q=query_tokens,
        max_seqlen_k=query_tokens,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=sparse_count,
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        cmp_ratio=4,
        device="npu",
    )
    return {
        "inputs": [query, key, weights, query_scale, key_scale],
        "kwargs": {
            "query_quant_mode": 0,
            "key_quant_mode": 0,
            "actual_seq_lengths_query": actual_q,
            "actual_seq_lengths_key": actual_k,
            "block_table": block_table,
            "metadata": metadata,
            "layout_query": "TND",
            "layout_key": "PA_BSND",
            "sparse_count": sparse_count,
            "sparse_mode": 3,
            "pre_tokens": (1 << 63) - 1,
            "next_tokens": (1 << 63) - 1,
            "cmp_ratio": 4,
            "return_value": False,
        },
        "api": op.resolve_api(),
    }


op = OpReplay(
    kernel_type="VllmQuantLightningIndexer",
    api_path="torch.ops._C_ascend.npu_vllm_quant_lightning_indexer",
    description="Replay DeepSeek-V4 quantized lightning indexer kernels.",
    usage_examples=[
        (
            "python tools/perf_data_collection/op_replay/VllmQuantLightningIndexer_run.py "
            "--database-path <vllm0.23.0 database directory>"
        ),
    ],
    version_help="vLLM-Ascend version, e.g. 0.23.0.",
    input_count=9,
    build_case=build_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
