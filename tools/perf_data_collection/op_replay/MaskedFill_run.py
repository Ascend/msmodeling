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
Replay MaskedFill cases from the performance database on Ascend NPU.

Purpose:
  Read MaskedFill rows from
  profiling_database/data/{device}/vllm_ascend/{version}/MaskedFill.csv,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch.Tensor.masked_fill_().

Usage:
  python tools/perf_data_collection/op_replay/MaskedFill_run.py ^
    --device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.15.0

Arguments:
  --device                Selects the device directory under profiling_database/data.
  --vllm-version   Selects the version directory under {device}/vllm_ascend.
"""

from __future__ import annotations

try:
    from .common import parse_list_field, scalar_fill_value_for_dtype
    from .replay_framework import OpReplay
except ImportError:
    from common import parse_list_field, scalar_fill_value_for_dtype
    from replay_framework import OpReplay


def build_case(row):
    """Rebuild the full MaskedFill case, including the scalar value slot.

    MaskedFill rows store two tensor inputs plus one scalar value operand:
    the value slot has an empty shape with an active dtype in
    ``Input Data Types`` (issue #510). ``OpReplay.build_inputs`` only
    materializes slots with a shape, so the value is re-attached here from
    its recorded dtype instead of being dropped.
    """
    case = {
        "inputs": op.build_inputs(row),
        "kwargs": {},
        "api": None,
    }
    dtypes = parse_list_field(row["Input Data Types"])
    if len(dtypes) >= 3:
        case["kwargs"]["value"] = scalar_fill_value_for_dtype(dtypes[2])
    else:
        # Fallback for legacy rows without a recorded value dtype: keep the
        # pre-fix literal 0 — an int converts into both integer and float
        # tensors, whereas 0.0 raises RuntimeError on integer dtypes.
        case["kwargs"]["value"] = 0
    return case


def run_case(case):
    tensor, mask_tensor = case["inputs"][:2]
    tensor.masked_fill_(mask_tensor, case["kwargs"].get("value", 0))
    return tensor


op = OpReplay(
    kernel_type="MaskedFill",
    description=(
        "Run MaskedFill workload replay on Ascend NPU/CUDA/CPU.\n"
        "The script reads MaskedFill.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs input tensors from\n"
        "Input Shapes / Input Formats / Input Data Types, then runs\n"
        "tensor.masked_fill_(mask, value)."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/MaskedFill_run.py --device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.15.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.15.0.",
    input_count=2,
    input_dtype_overrides={1: "DT_BOOL"},
    build_case=build_case,
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
