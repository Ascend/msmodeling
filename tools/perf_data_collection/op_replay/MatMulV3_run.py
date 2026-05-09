"""
Replay MatMulV3 cases from the performance database on Ascend NPU.

Purpose:
  Read MatMulV3 rows from
  profiling_database/data/{device}/vllm_ascend/{version}/MatMulV3.csv,
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
    case = build_matmul_case(row, kernel_type="MatMulV3", require_exact_inputs=True)
    case["api"] = op.resolve_api()
    return case


op = OpReplay(
    kernel_type="MatMulV3",
    api_path="torch.mm",
    description=(
        "Run MatMulV3 workload replay on Ascend NPU.\n"
        "The script reads MatMulV3.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs input tensors from\n"
        "Input Shapes / Input Formats / Input Data Types, then runs torch.mm()."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/MatMulV3_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.13.0.",
    build_case=build_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
