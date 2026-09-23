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

"""Replay CANN Fill cases from the performance database on Ascend NPU."""

from __future__ import annotations

try:
    from .common import (
        get_runtime_modules,
        parse_shape,
        resolve_device_type,
        resolve_runtime_dtype,
        split_metadata_field,
    )
    from .replay_framework import OpReplay
except ImportError:
    from common import (
        get_runtime_modules,
        parse_shape,
        resolve_device_type,
        resolve_runtime_dtype,
        split_metadata_field,
    )
    from replay_framework import OpReplay


def _fill_value(dtype_name: str):
    normalized = dtype_name.strip().upper()
    if "BOOL" in normalized:
        return True
    if "FLOAT" in normalized or "BF16" in normalized or "DOUBLE" in normalized:
        return 1.0
    return 1


def build_case(row: dict[str, str]):
    input_shapes = split_metadata_field(row["Input Shapes"])
    input_dtypes = split_metadata_field(row["Input Data Types"])
    input_formats = split_metadata_field(row["Input Formats"])
    output_shapes = split_metadata_field(row["Output Shapes"])
    output_dtypes = split_metadata_field(row["Output Data Types"])

    if len(input_shapes) != 2 or len(input_dtypes) != 2 or len(input_formats) != 2:
        raise ValueError("Fill expects shape-metadata and scalar-value input slots")
    if input_shapes[1]:
        raise ValueError("Fill value input must be a scalar slot")
    if len(output_shapes) != 1 or not output_shapes[0]:
        raise ValueError("Fill requires exactly one non-scalar output shape")
    if len(output_dtypes) != 1:
        raise ValueError("Fill requires exactly one output dtype")

    output_shape = parse_shape(output_shapes[0])
    metadata_shape = parse_shape(input_shapes[0])
    if metadata_shape != (len(output_shape),):
        raise ValueError(
            f"Fill shape-metadata length must match output rank: metadata={metadata_shape}, output={output_shape}"
        )

    runtime_torch, _ = get_runtime_modules()
    output_dtype = resolve_runtime_dtype(output_dtypes[0])
    return {
        "api": runtime_torch.full,
        "output_shape": output_shape,
        "output_dtype": output_dtype,
        "fill_value": _fill_value(output_dtypes[0]),
        "device": resolve_device_type(runtime_torch),
    }


def run_case(case):
    return case["api"](
        case["output_shape"],
        case["fill_value"],
        dtype=case["output_dtype"],
        device=case["device"],
    )


op = OpReplay(
    kernel_type="Fill",
    description=("Replay Fill.csv shape-metadata plus scalar-value cases with torch.full on Ascend NPU."),
    usage_examples=["python tools/perf_data_collection/op_replay/Fill_run.py --database-path /path/to/database"],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    build_case=build_case,
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
