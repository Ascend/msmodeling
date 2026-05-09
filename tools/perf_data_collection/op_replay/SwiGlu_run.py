"""
Replay SwiGlu cases from the performance database on Ascend NPU.

Purpose:
  Read SwiGlu rows from
  profiling_database/data/{device}/vllm_ascend/{version}/SwiGlu.csv,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch_npu.npu_swiglu() with dim fixed to -1.

Usage:
  python tools/perf_data_collection/op_replay/SwiGlu_run.py ^
    --device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0

Arguments:
  --device                Selects the device directory under profiling_database/data.
  --vllm-version   Selects the version directory under {device}/vllm_ascend.
"""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay


def format_success(csv_path, row_index: int, row: dict[str, str], _case, result) -> str:
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(result.shape)} dim=-1"
    )


op = OpReplay(
    kernel_type="SwiGlu",
    api_path="torch_npu.npu_swiglu",
    description=(
        "Run SwiGlu workload replay on Ascend NPU.\n"
        "The script reads SwiGlu.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs input tensors from\n"
        "Input Shapes / Input Formats / Input Data Types, then runs\n"
        "torch_npu.npu_swiglu() with dim=-1."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/SwiGlu_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0",
        "python tools/perf_data_collection/op_replay/SwiGlu_run.py "
        "--device TEST_DEVICE --vllm-version 0.9.2",
    ],
    version_help="vLLM-Ascend version, e.g. 0.9.2.",
    input_count=1,
    fixed_kwargs={"dim": -1},
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()

