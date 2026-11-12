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

"""Replay ScatterNdUpdateV2 cache writes from their recorded 4-D ABI."""

from __future__ import annotations

try:
    from .common import get_runtime_modules, parse_list_field, parse_shape
    from .replay_framework import OpReplay
except ImportError:
    from common import get_runtime_modules, parse_list_field, parse_shape
    from replay_framework import OpReplay


def _legal_cache_indices(indices_shape: tuple[int, ...], cache_shape: tuple[int, ...]):
    runtime_torch, _ = get_runtime_modules()
    if len(indices_shape) != 2 or indices_shape[1] != 2:
        raise ValueError(f"ScatterNdUpdateV2 indices must be (N, 2), got {indices_shape}")
    if len(cache_shape) < 3:
        raise ValueError(f"ScatterNdUpdateV2 cache must be rank >= 3, got {cache_shape}")
    count = indices_shape[0]
    capacity = cache_shape[0] * cache_shape[1]
    if count > capacity:
        raise ValueError(f"update count {count} exceeds cache capacity {capacity}")
    flat = runtime_torch.arange(count, dtype=runtime_torch.int32, device="npu")
    return runtime_torch.stack((flat // cache_shape[1], flat % cache_shape[1]), dim=1)


def build_case(row: dict[str, str]):
    cache, _indices, update = op.build_inputs(row)
    shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    indices = _legal_cache_indices(shapes[1], shapes[0])
    return {
        "inputs": [cache, indices, update],
        "kwargs": {},
        "api": op.resolve_api(),
    }


def run_case(case):
    case["api"](*case["inputs"])
    return case["inputs"][0]


op = OpReplay(
    kernel_type="ScatterNdUpdateV2",
    api_path="torch.ops._C_ascend.npu_scatter_nd_update_v2",
    description="Replay DeepSeek-V4 paged cache writes with legal (block, offset) indices.",
    usage_examples=[
        (
            "python tools/perf_data_collection/op_replay/ScatterNdUpdateV2_run.py "
            "--database-path <vllm0.23.0 database directory>"
        ),
    ],
    version_help="vLLM-Ascend version, e.g. 0.23.0.",
    input_count=3,
    build_case=build_case,
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
