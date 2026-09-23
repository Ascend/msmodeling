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
Replay ScatterNdUpdate cases from the performance database on Ascend NPU.

Purpose:
  Read ScatterNdUpdate rows from
  profiling_database/data/{device}/vllm_ascend/{version}/ScatterNdUpdate.csv,
  rebuild the recorded tensor inputs, construct a legal indices tensor,
  then execute torch_npu.npu_scatter_nd_update() (aclnnScatterNdUpdate).

CSV layout (3 inputs, 1 output):
  Input[0]: data tensor   (total_slots, head_dim)   e.g. (226048, 128)  BF16
  Input[1]: indices tensor (num_tokens, 1)           e.g. (102, 1)      INT32
  Input[2]: updates tensor (num_tokens, head_dim)    e.g. (102, 128)    BF16
  Output[0]: updated data  (total_slots, head_dim)   same as Input[0]

Non-tensor args: None (in-place scatter update by indices).

microbench_api: torch_npu.npu_scatter_nd_update (maps to aclnnScatterNdUpdate)

Paged cache data: vLLM-Ascend's DSA indexer (sfa_v1.py) allocates kv_cache[2]
as a multi-dimensional paged tensor (num_blocks, block_size, [num_kv_heads,]
head_dim) and flattens it via ``.view(-1, head_dim)`` immediately before
calling ``npu_scatter_nd_update_``.  The profiling/composite-leaf diagnostic
records the pre-view paged shape (for example, (2976, 128, 128)),
so ``total_slots`` must be the product of all leading dims (num_blocks *
block_size [* num_kv_heads]) rather than ``data_shape[0]`` alone, mirroring
``ReshapeAndCacheNdKernel_run.py``'s ``key_cache_shape[0] * key_cache_shape[1]``.
"""

from __future__ import annotations

from functools import reduce
from operator import mul

try:
    from .common import get_runtime_modules, parse_list_field, parse_shape
    from .replay_framework import OpReplay
except ImportError:
    from common import get_runtime_modules, parse_list_field, parse_shape
    from replay_framework import OpReplay


def _total_slots(data_shape: tuple[int, ...]) -> int:
    """Return the flat slot count of a scatter data tensor.

    For 2D ``(total_slots, head_dim)`` this is ``data_shape[0]`` (decode
    cache write).  For 3D/4D paged ``(num_blocks, block_size, [heads,]
    head_dim)`` this is the product of all leading dims, matching the
    ``.view(-1, head_dim)`` flattening vLLM-Ascend applies before the op.
    """
    if len(data_shape) < 2:
        raise ValueError(f"data tensor rank must be >= 2, got shape={data_shape}")
    return reduce(mul, data_shape[:-1], 1)


def build_indices_tensor(
    indices_shape: tuple[int, ...],
    data_shape: tuple[int, ...],
):
    """Build a legal indices tensor with values in [0, total_slots)."""
    runtime_torch, _ = get_runtime_modules()
    if len(indices_shape) != 2 or indices_shape[1] != 1:
        raise ValueError(f"indices must be (N, 1), got shape={indices_shape}")

    num_tokens = indices_shape[0]
    total_slots = _total_slots(data_shape)
    if num_tokens > total_slots:
        raise ValueError(
            f"num_tokens ({num_tokens}) exceeds total_slots ({total_slots}) "
            f"for data shape {data_shape}"
        )

    return runtime_torch.arange(
        num_tokens,
        dtype=runtime_torch.int32,
        device="npu",
    ).unsqueeze(1)


def build_scatter_case(replay: OpReplay, row: dict[str, str]):
    inputs = replay.build_inputs(row)
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    if len(inputs) != 3 or len(input_shapes) != 3:
        raise ValueError("ScatterNdUpdate expects exactly three inputs")

    # Replace indices with legal values
    return {
        "inputs": [
            inputs[0],
            build_indices_tensor(
                indices_shape=input_shapes[1],
                data_shape=input_shapes[0],
            ),
            inputs[2],
        ],
        "kwargs": {},
        "api": replay.resolve_api(),
    }


def build_case(row: dict[str, str]):
    return build_scatter_case(op, row)


def format_success(csv_path, row_index: int, row: dict[str, str], case, _result) -> str:
    data = case["inputs"][0]
    indices = case["inputs"][1]
    updates = case["inputs"][2]
    return (
        f"[OK] {csv_path}:{row_index} "
        f"data={tuple(data.shape)} indices={tuple(indices.shape)} "
        f"updates={tuple(updates.shape)} "
        f"dtypes={row['Input Data Types']}"
    )


op = OpReplay(
    kernel_type="ScatterNdUpdate",
    api_path="torch_npu.npu_scatter_nd_update",
    description=(
        "Run ScatterNdUpdate workload replay on Ascend NPU.\n"
        "Reads ScatterNdUpdate.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs input tensors,\n"
        "builds a legal indices tensor, then runs\n"
        "torch_npu.npu_scatter_nd_update(data, indices, updates).\n\n"
        "This operator performs sparse-indexer cache updates:\n"
        "it writes new token K vectors into the indexer cache at\n"
        "the positions specified by indices."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/ScatterNdUpdate_run.py "
        "--database-path tensor_cast/performance_model/profiling_database/"
        "data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/test",
    ],
    version_help="vLLM-Ascend version, e.g. 0.19.0.",
    build_case=build_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
