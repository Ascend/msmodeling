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
Replay GroupedMatmulSwigluQuant cases from the performance database on Ascend NPU.

Purpose:
  Read GroupedMatmulSwigluQuant rows from the profiling database,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch_npu.npu_grouped_matmul_swiglu_quant().

Usage:
  python tools/perf_data_collection/op_replay/GroupedMatmulSwigluQuant_run.py ^
    --device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.18.0
"""

from __future__ import annotations

import json
import math

try:
    from .common import (
        FRACTAL_NZ_FORMAT_ID,
        build_host_tensor,
        build_input_tensor,
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
        build_input_tensor,
        get_runtime_modules,
        init_runtime,
        normalize_dtype_name,
        parse_shape,
        resolve_runtime_dtype,
        split_metadata_field,
    )
    from replay_framework import OpReplay


def _build_scale_tensor(shape, dtype_name: str, runtime_torch):
    dtype = resolve_runtime_dtype(dtype_name)
    return runtime_torch.ones(shape, dtype=dtype).npu()


def _build_group_list_tensor(
    total_tokens: int,
    num_groups: int,
    dtype_name: str,
    runtime_torch,
    values=None,
):
    dtype = resolve_runtime_dtype(dtype_name)
    if values is not None:
        if (
            not isinstance(values, list)
            or len(values) != num_groups
            or any(type(value) is not int or value < 0 for value in values)
            or values[-1] != total_tokens
            or any(a > b for a, b in zip(values, values[1:]))
        ):
            raise ValueError("GroupedMatmulSwigluQuant requires cumulative boundaries conserving packed tokens")
        return runtime_torch.tensor(values, dtype=dtype).npu()
    base = total_tokens // num_groups
    remainder = total_tokens % num_groups
    cumulative = []
    running_total = 0
    for idx in range(num_groups):
        running_total += base + (1 if idx < remainder else 0)
        cumulative.append(running_total)
    return runtime_torch.tensor(cumulative, dtype=dtype).npu()


def _build_weight_tensor(shape, dtype_name: str, input_format: str, runtime_torch_npu):
    if input_format == "FRACTAL_NZ" and len(shape) == 5:
        *batch_dims, h_dim, w_dim, block_h, block_w = shape
        logical_shape = (*batch_dims, w_dim * block_h, h_dim * block_w)
        dtype = resolve_runtime_dtype(dtype_name)
        tensor = build_host_tensor(logical_shape, dtype).npu()
        return runtime_torch_npu.npu_format_cast(tensor, FRACTAL_NZ_FORMAT_ID)
    return build_input_tensor(shape, input_format, dtype_name)


def build_case(row: dict[str, str]):
    contract = (row.get("Runtime grouped_contract") or "").strip()
    limit_text = (row.get("Runtime swiglu_limit") or "").strip()
    if contract or limit_text:
        if contract != "ascend_v4_grouped_v1":
            raise ValueError("V4 grouped replay requires Runtime grouped_contract=ascend_v4_grouped_v1")
        try:
            limit = float(limit_text)
        except ValueError as exc:
            raise ValueError("V4 grouped replay requires a finite nonnegative Runtime swiglu_limit") from exc
        if not math.isfinite(limit) or limit < 0:
            raise ValueError("V4 grouped replay requires a finite nonnegative Runtime swiglu_limit")
    init_runtime()
    runtime_torch, runtime_torch_npu = get_runtime_modules()
    shapes = [parse_shape(item) for item in split_metadata_field(row["Input Shapes"])]
    dtypes = [normalize_dtype_name(item) for item in split_metadata_field(row["Input Data Types"])]
    formats = split_metadata_field(row["Input Formats"])

    x = build_input_tensor(shapes[0], formats[0], dtypes[0])
    weight = _build_weight_tensor(shapes[1], dtypes[1], formats[1], runtime_torch_npu)
    weight_scale = _build_scale_tensor(shapes[2], dtypes[2], runtime_torch)
    x_scale = _build_scale_tensor(shapes[3], dtypes[3], runtime_torch)
    if int(row.get("Runtime group_list_type") or 0) != 0:
        raise ValueError("GroupedMatmulSwigluQuant requires cumulative group lists")
    values_text = row.get("Runtime group_list_values", "")
    values = json.loads(values_text) if values_text and str(values_text).strip().lower() != "nan" else None
    group_list = _build_group_list_tensor(x.shape[0], shapes[4][0], dtypes[4], runtime_torch, values=values)
    kwargs = {
        "x": x,
        "weight": weight,
        "group_list": group_list,
        "weight_scale": weight_scale,
        "x_scale": x_scale,
    }
    if contract == "ascend_v4_grouped_v1":
        # A3 vLLM-Ascend 0.23.0 BaseDeviceAdaptor uses this custom fused
        # kernel, including its clamp scalar, rather than torch_npu's ABI.
        api = runtime_torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz
        kwargs["bias"] = None
        kwargs["swiglu_limit"] = limit
    else:
        api = op.resolve_api()
    return {
        "inputs": [x, weight, weight_scale, x_scale, group_list],
        "kwargs": kwargs,
        "api": api,
    }


def run_case(case):
    # npu_grouped_matmul_swiglu_quant returns (result, scale, pertoken_scale).
    # The profiling database records only the first two outputs.
    result = case["api"](**case["kwargs"])
    return result[:2]


op = OpReplay(
    kernel_type="GroupedMatmulSwigluQuant",
    api_path="torch_npu.npu_grouped_matmul_swiglu_quant",
    description=(
        "Run GroupedMatmulSwigluQuant workload replay on Ascend NPU.\n"
        "Fused grouped matmul + SwiGlu + quantization for MoE FFN layers."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/GroupedMatmulSwigluQuant_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.18.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    build_case=build_case,
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
