"""
Replay PadV3 cases from the performance database on Ascend NPU.

Purpose:
  Read PadV3 rows, rebuild the recorded input tensor, infer paddings from
  the input/output shape delta, then execute torch.nn.functional.pad().
"""

from __future__ import annotations

try:
    from .common import build_input_tensor, normalize_dtype_name, parse_list_field, parse_shape
    from .replay_framework import OpReplay
except ImportError:
    from common import build_input_tensor, normalize_dtype_name, parse_list_field, parse_shape
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = parse_list_field(row["Input Data Types"])
    if not input_shapes:
        raise ValueError("PadV3 requires at least one recorded input tensor")

    input_tensor = build_input_tensor(
        shape=input_shapes[0],
        input_format=input_formats[0] if input_formats else "ND",
        dtype_name=normalize_dtype_name(input_dtypes[0] if input_dtypes else "DT_FLOAT"),
    )

    output_shapes = [parse_shape(item) for item in parse_list_field(row["Output Shapes"])]
    if not output_shapes:
        raise ValueError("PadV3 requires at least one recorded output shape")

    paddings: list[int] = []
    for input_dim, output_dim in zip(reversed(tuple(input_tensor.shape)), reversed(output_shapes[0])):
        paddings.extend([0, output_dim - input_dim])

    return {
        "inputs": [input_tensor],
        "kwargs": {"pad": paddings, "mode": "constant", "value": 0.0},
        "api": op.resolve_api(),
    }


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(result.shape)} "
        f"paddings={case['kwargs']['pad']}"
    )


op = OpReplay(
    kernel_type="PadV3",
    api_path="torch.nn.functional.pad",
    description=(
        "Run PadV3 workload replay on Ascend NPU.\n"
        "The script reads PadV3.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs the input tensor,\n"
        "infers paddings from the recorded input/output shapes, then runs\n"
        "torch.nn.functional.pad()."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/PadV3_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.16.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.16.0.",
    input_count=1,
    build_case=build_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
