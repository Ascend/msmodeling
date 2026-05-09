"""
Replay GatherV2 cases from the performance database on Ascend NPU.

Purpose:
  Read GatherV2 rows from
  profiling_database/data/{device}/vllm_ascend/{version}/GatherV2.csv,
  rebuild the recorded weight and index tensors, then execute
  torch.nn.functional.embedding(indices, weight).
"""

from __future__ import annotations

try:
    from .replay_framework import OpReplay
except ImportError:
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    inputs = op.build_inputs(row)
    if len(inputs) < 2:
        raise ValueError("GatherV2 expects at least two inputs (weight, indices)")
    return {
        "inputs": inputs[:2],
        "kwargs": {},
        "api": op.resolve_api(),
    }


def run_case(case):
    weight_tensor, indices_tensor = case["inputs"]
    return case["api"](indices_tensor, weight_tensor)


op = OpReplay(
    kernel_type="GatherV2",
    api_path="torch.nn.functional.embedding",
    description=(
        "Run GatherV2 workload replay on Ascend NPU.\n"
        "The script reads GatherV2.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs the recorded weight\n"
        "and index tensors, then runs torch.nn.functional.embedding()."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/GatherV2_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.15.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.15.0.",
    build_case=build_case,
    run_case=run_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
