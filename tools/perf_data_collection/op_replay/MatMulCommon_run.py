"""
Replay MatMulCommon cases from the performance database on Ascend NPU.

Purpose:
  Read MatMulCommon rows from
  profiling_database/data/{device}/vllm_ascend/{version}/MatMulCommon.csv,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch.mm() with the recorded transpose convention.
"""

from __future__ import annotations

try:
    from .common import build_matmul_case
    from .replay_framework import OpReplay
except ImportError:
    from common import build_matmul_case
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    case = build_matmul_case(row, kernel_type="MatMulCommon", require_exact_inputs=False)
    case["api"] = op.resolve_api()
    return case


op = OpReplay(
    kernel_type="MatMulCommon",
    api_path="torch.mm",
    description=(
        "Run MatMulCommon workload replay on Ascend NPU.\n"
        "The script reads MatMulCommon.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs input tensors from\n"
        "Input Shapes / Input Formats / Input Data Types, then runs torch.mm()."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/MatMulCommon_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.15.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.15.0.",
    build_case=build_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
