"""
Replay MoeTokenPermute cases from the performance database on Ascend NPU.

Purpose:
  Read MoeTokenPermute rows from the profiling database,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch_npu.npu_moe_token_permute().

Usage:
  python tools/perf_data_collection/op_replay/MoeTokenPermute_run.py ^
    --device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.18.0
"""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    case = {
        "inputs": op.build_inputs(row),
        "kwargs": {},
        "api": op.resolve_api(),
    }
    if len(case["inputs"]) < 2:
        raise ValueError("MoeTokenPermute expects at least 2 inputs")
    return case


def run_case(case):
    return case["api"](case["inputs"][0], case["inputs"][1])


op = OpReplay(
    kernel_type="MoeTokenPermute",
    api_path="torch_npu.npu_moe_token_permute",
    description=(
        "Run MoeTokenPermute workload replay on Ascend NPU.\n"
        "Permutes tokens by expert assignment for MoE all-to-all dispatching."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/MoeTokenPermute_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.18.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    build_case=build_case,
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
