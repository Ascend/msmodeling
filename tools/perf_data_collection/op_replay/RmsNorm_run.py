"""
Replay RmsNorm cases from the performance database on Ascend NPU.

Purpose:
  Read RmsNorm rows from
  profiling_database/data/{device}/vllm_ascend/{version}/RmsNorm.csv,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch_npu.npu_rms_norm().
"""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay


def format_success(csv_path, row_index: int, row: dict[str, str], _case, result) -> str:
    output, rstd = result
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output0={tuple(output.shape)} "
        f"output1={tuple(rstd.shape)}"
    )


op = OpReplay(
    kernel_type="RmsNorm",
    api_path="torch_npu.npu_rms_norm",
    description=(
        "Run RmsNorm workload replay on Ascend NPU.\n"
        "The script reads RmsNorm.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs input tensors from\n"
        "Input Shapes / Input Formats / Input Data Types, then runs\n"
        "torch_npu.npu_rms_norm()."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/RmsNorm_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.13.0.",
    input_count=2,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
