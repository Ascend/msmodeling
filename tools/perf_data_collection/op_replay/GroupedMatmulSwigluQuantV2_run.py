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
Replay GroupedMatmulSwigluQuantV2 cases from the performance database on Ascend NPU.

Purpose:
  Read GroupedMatmulSwigluQuantV2 rows (W4A8 MoE grouped
  GEMM + SwiGlu + quant) from the profiling database,
  rebuild the recorded (x, x_scale, group_list, weight)
  tensors, then execute torch_npu.npu_grouped_matmul_swiglu_quant_v2()
  with the recorded weight_scale.

  The weight tensor is recorded in FRACTAL_NZ layout with a
  DT_INT4 element type; it is rebuilt as a logical
  (groups, K, 2N) int8 tensor and format-cast to FRACTAL_NZ.
"""

from __future__ import annotations

try:
    from .common import (
        FRACTAL_NZ_FORMAT_ID,
        build_host_tensor,
        get_runtime_modules,
        init_runtime,
        normalize_dtype_name,
        parse_shape,
        resolve_runtime_dtype,
        split_metadata_field,
    )
    from .replay_framework import OpReplay
except ImportError:
    from common import (
        FRACTAL_NZ_FORMAT_ID,
        build_host_tensor,
        get_runtime_modules,
        init_runtime,
        normalize_dtype_name,
        parse_shape,
        resolve_runtime_dtype,
        split_metadata_field,
    )
    from replay_framework import OpReplay


def _build_weight_tensor(groups: int, k_dim: int, n_dim: int, runtime_torch_npu):
    """Build a (groups, K, N) weight tensor in FRACTAL_NZ layout."""
    dtype = resolve_runtime_dtype("DT_INT8")
    tensor = build_host_tensor((groups, k_dim, n_dim), dtype).npu()
    return runtime_torch_npu.npu_format_cast(tensor, FRACTAL_NZ_FORMAT_ID)


def build_case(row: dict[str, str]):
    init_runtime()
    runtime_torch, runtime_torch_npu = get_runtime_modules()
    shapes = [parse_shape(item) for item in split_metadata_field(row["Input Shapes"])]
    dtypes = [normalize_dtype_name(item) for item in split_metadata_field(row["Input Data Types"])]
    if len(shapes) < 6:
        raise ValueError(
            "GroupedMatmulSwigluQuantV2 expects x, x_scale, group_list, weight, ..., "
            "weight_scale slots (at least six recorded slots)"
        )
    if not shapes[0]:
        raise ValueError("GroupedMatmulSwigluQuantV2 requires a non-empty x slot")

    x = build_host_tensor(shapes[0], resolve_runtime_dtype(dtypes[0])).npu()
    # Scales must be finite: uninitialized (empty) scale values can be
    # NaN/Inf and make the aclnn kernel fail internally.
    x_scale = runtime_torch.ones(shapes[1], dtype=runtime_torch.float32).npu()
    group_count = shapes[2][0]
    base = x.shape[0] // group_count
    remainder = x.shape[0] % group_count
    cumulative = []
    running = 0
    for _ in range(group_count):
        running += base
        if remainder:
            running += 1
            remainder -= 1
        cumulative.append(running)
    group_list = runtime_torch.tensor(cumulative, dtype=runtime_torch.int64).npu()

    k_dim = x.shape[1]
    n_dim = shapes[5][1] if len(shapes) >= 6 and shapes[5] else None
    if n_dim is None:
        raise ValueError(
            "GroupedMatmulSwigluQuantV2 requires the weight_scale slot to infer N"
        )
    # Cross-check the recorded weight slot when it is a plain 3-D
    # (groups, K, N) shape: its last dim must match the N inferred from
    # weight_scale, otherwise the rebuilt GEMM shape would silently
    # diverge from the recorded row.
    if shapes[3] and len(shapes[3]) == 3 and shapes[3][2] != n_dim:
        raise ValueError(
            f"Rebuilt weight N ({n_dim}) mismatches recorded weight slot {shapes[3]}"
        )
    weight = _build_weight_tensor(group_count, k_dim, n_dim, runtime_torch_npu)

    return {
        "kwargs": {
            "x": x,
            "weight": [weight],
            "weight_scale": [
                runtime_torch.ones(shapes[5], dtype=runtime_torch.float32).npu()
            ],
            "x_scale": x_scale,
            "group_list": group_list,
        },
        "api": op.resolve_api(),
    }


def run_case(case):
    # Returns (quantized_output, swiglu_out_scale).
    result = case["api"](**case["kwargs"])
    return result[:2]


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    output = result[0] if isinstance(result, (tuple, list)) and result else result
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(output.shape)}"
    )


op = OpReplay(
    kernel_type="GroupedMatmulSwigluQuantV2",
    api_path="torch_npu.npu_grouped_matmul_swiglu_quant_v2",
    description=(
        "Run GroupedMatmulSwigluQuantV2 workload replay on Ascend NPU.\n"
        "The script reads GroupedMatmulSwigluQuantV2.csv under the\n"
        "selected device and vllm_ascend version directory,\n"
        "rebuilds the (x, x_scale, group_list, weight) tensors,\n"
        "then runs torch_npu.npu_grouped_matmul_swiglu_quant_v2()."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/GroupedMatmulSwigluQuantV2_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.23.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.23.0.",
    build_case=build_case,
    run_case=run_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
