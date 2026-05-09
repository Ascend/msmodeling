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
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay


def run_case(case):
    tensor, mask_tensor = case["inputs"][:2]
    tensor.masked_fill_(mask_tensor, 0)
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
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()

