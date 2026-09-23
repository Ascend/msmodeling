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

"""Replay the API variants that lower to the Mul NPU kernel."""

from __future__ import annotations

try:
    from .common import (
        build_input_tensor,
        get_runtime_modules,
        parse_list_field,
        parse_shape,
        split_metadata_field,
    )
    from .replay_framework import OpReplay
except ImportError:
    from common import (
        build_input_tensor,
        get_runtime_modules,
        parse_list_field,
        parse_shape,
        split_metadata_field,
    )
    from replay_framework import OpReplay


def classify_mul_replay(row: dict[str, str]) -> str:
    """Select the source API semantics represented by one Mul database row."""
    shapes = [parse_shape(value) for value in parse_list_field(row["Input Shapes"])]
    if len(shapes) == 1:
        return "scalar"
    if len(shapes) != 2:
        raise ValueError(f"Mul expects one or two tensor inputs, got {len(shapes)}")

    lhs, rhs = shapes
    output_shapes = [parse_shape(value) for value in parse_list_field(row.get("Output Shapes", ""))]
    is_degenerate_bmm = (
        len(lhs) == 3
        and len(rhs) == 3
        and lhs[0] == rhs[0]
        and lhs[-1] == rhs[-2]
        and bool(output_shapes)
        and output_shapes[0] == (lhs[0], lhs[-2], rhs[-1])
    )
    return "bmm" if is_degenerate_bmm else "elementwise"


# The real scalar value cannot be recovered from the profiling CSV, which has
# no column for the scalar operand of a tensor-by-scalar Mul.  The default 1 is
# a safe neutral element for kernel timing; the source label makes this
# assumption traceable if future kernels depend on the scalar value.
SCALAR_DEFAULT_SOURCE = "default_1_unrecoverable_from_csv"


def infer_scalar_value(row: dict[str, str]) -> int | float:
    """Return the placeholder scalar for tensor-by-scalar Mul replay.

    The CSV does not store the actual scalar value, so a neutral default is
    used.  Callers should consult ``SCALAR_DEFAULT_SOURCE`` for traceability.
    """
    output_dtypes = [value.upper() for value in parse_list_field(row.get("Output Data Types", ""))]
    if output_dtypes and ("INT" in output_dtypes[0] or "BOOL" in output_dtypes[0]):
        return 1
    return 1.0


def build_case(row: dict[str, str]):
    inputs = op.build_inputs(row)
    replay_mode = classify_mul_replay(row)
    if replay_mode == "scalar":
        dtypes = split_metadata_field(row.get("Input Data Types", ""))
        formats = split_metadata_field(row.get("Input Formats", ""))
        if len(dtypes) > 1 and dtypes[1].upper() not in {"", "DT_UNDEFINED", "UNDEFINED"}:
            inputs.append(
                build_input_tensor(
                    shape=(),
                    input_format=formats[1] if len(formats) > 1 else "ND",
                    dtype_name=dtypes[1],
                )
            )
    runtime_torch, _ = get_runtime_modules()
    api = runtime_torch.bmm if replay_mode == "bmm" else runtime_torch.mul
    return {
        "inputs": inputs,
        "mode": replay_mode,
        "scalar": infer_scalar_value(row),
        "scalar_source": SCALAR_DEFAULT_SOURCE,
        "api": api,
    }


def run_case(case):
    if case["mode"] == "scalar":
        rhs = case["inputs"][1] if len(case["inputs"]) > 1 else case["scalar"]
        return case["api"](case["inputs"][0], rhs)
    return case["api"](case["inputs"][0], case["inputs"][1])


op = OpReplay(
    kernel_type="Mul",
    api_path="torch.mul",
    description=(
        "Replay Mul.csv rows through tensor-by-scalar, broadcast elementwise, "
        "or degenerate BMM semantics. All three paths lower "
        "to the profiled Mul kernel."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/Mul_run.py "
        "--database-path /path/to/database"
    ],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    build_case=build_case,
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
