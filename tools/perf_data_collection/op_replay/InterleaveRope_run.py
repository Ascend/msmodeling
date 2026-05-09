"""
Replay InterleaveRope cases from the performance database on Ascend NPU.

Purpose:
  Read InterleaveRope rows from
  profiling_database/data/{device}/vllm_ascend/{version}/InterleaveRope.csv,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch_npu.npu_interleave_rope().
"""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay


op = OpReplay(
    kernel_type="InterleaveRope",
    api_path="torch_npu.npu_interleave_rope",
    description=(
        "Run InterleaveRope workload replay on Ascend NPU.\n"
        "The script reads InterleaveRope.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs input tensors from\n"
        "Input Shapes / Input Formats / Input Data Types, then runs\n"
        "torch_npu.npu_interleave_rope()."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/InterleaveRope_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.13.0.",
    input_count=3,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
