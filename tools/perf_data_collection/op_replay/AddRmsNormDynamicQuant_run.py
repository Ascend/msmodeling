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
Replay AddRmsNormDynamicQuant cases from the performance database on Ascend NPU.

Purpose:
  Read AddRmsNormDynamicQuant rows from the profiling database,
  rebuild the recorded (rms_norm_input, residual, rms_norm_weight,
  beta) tensors, then execute
  torch.ops.npu.npu_add_rms_norm_dynamic_quant() exactly like the
  vLLM-Ascend norm_quant_fusion_pass replacement does.
"""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay

API_PATH = "torch.ops.npu.npu_add_rms_norm_dynamic_quant"


def build_case(row: dict[str, str]):
    inputs = op.build_inputs(row)
    if len(inputs) < 3:
        raise ValueError(
            "AddRmsNormDynamicQuant expects rms_norm_input, residual, and rms_norm_weight"
        )
    rms_norm_input, residual, rms_norm_weight = inputs[:3]
    beta = inputs[3] if len(inputs) >= 4 else None
    kwargs = {
        "epsilon": 1e-6,
        "output_mask": [True, True],
    }
    if beta is not None:
        kwargs["beta"] = beta
    return {
        "inputs": [rms_norm_input, residual, rms_norm_weight],
        "kwargs": kwargs,
        "api": op.resolve_api(),
    }


def run_case(case):
    return case["api"](*case["inputs"], **case["kwargs"])


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    output = result[0] if isinstance(result, (tuple, list)) and result else result
    output_shape = tuple(output.shape) if hasattr(output, "shape") else str(output)
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={output_shape} "
        f"outputs={len(result) if isinstance(result, (tuple, list)) else 1}"
    )


op = OpReplay(
    kernel_type="AddRmsNormDynamicQuant",
    api_path=API_PATH,
    description=(
        "Run AddRmsNormDynamicQuant workload replay on Ascend NPU.\n"
        "The script reads AddRmsNormDynamicQuant.csv under the selected\n"
        "device and vllm_ascend version directory, rebuilds the\n"
        "(rms_norm_input, residual, rms_norm_weight[, beta]) tensors,\n"
        "then runs torch.ops.npu.npu_add_rms_norm_dynamic_quant()\n"
        "with epsilon=1e-6 and output_mask=[True, True]."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/AddRmsNormDynamicQuant_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.18.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    build_case=build_case,
    run_case=run_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
