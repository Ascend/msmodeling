"""
Replay Slice cases from the performance database on Ascend NPU.

Purpose:
  Read Slice rows from
  profiling_database/data/{device}/vllm_ascend/{version}/Slice.csv,
  rebuild the recorded source tensor, infer zero offsets from the current
  dataset, then execute torch_npu.npu_slice().
"""

from __future__ import annotations

try:
    from .common import parse_list_field, parse_shape
    from .replay_framework import OpReplay
except ImportError:
    from common import parse_list_field, parse_shape
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    inputs = op.build_inputs(row)
    if len(inputs) < 1:
        raise ValueError("Slice requires at least one recorded input tensor")

    output_shapes = [parse_shape(item) for item in parse_list_field(row["Output Shapes"])]
    if not output_shapes:
        raise ValueError("Slice requires at least one recorded output shape")

    output_shape = output_shapes[0]
    return {
        "inputs": [inputs[0]],
        "kwargs": {
            "offsets": [0] * len(output_shape),
            "sizes": list(output_shape),
        },
        "api": op.resolve_api(),
    }


def run_case(case):
    source = case["inputs"][0]
    return case["api"](source, case["kwargs"]["offsets"], case["kwargs"]["sizes"])


def format_success(csv_path, row_index: int, _row: dict[str, str], case, result) -> str:
    source = case["inputs"][0]
    offsets = case["kwargs"]["offsets"]
    sizes = case["kwargs"]["sizes"]
    return (
        f"[OK] {csv_path}:{row_index} "
        f"source={tuple(source.shape)} output={tuple(result.shape)} "
        f"offsets={offsets} sizes={sizes}"
    )


op = OpReplay(
    kernel_type="Slice",
    api_path="torch_npu.npu_slice",
    description=(
        "Run Slice workload replay on Ascend NPU.\n"
        "The script reads Slice.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs the source tensor,\n"
        "infers zero-based offsets from the current dataset, then runs\n"
        "torch_npu.npu_slice(input, offsets, sizes)."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/Slice_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.13.0.",
    build_case=build_case,
    run_case=run_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
