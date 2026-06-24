"""
Replay BatchMatMulV2 cases from the performance database on Ascend NPU.

The database contains both legacy 2D rows whose second input is recorded as
[N, K], and batched rows whose second input is usually recorded as [B, K, N].
Use the recorded output shape to choose the least surprising torch API.
"""

from __future__ import annotations

try:
    from .common import (
        build_input_tensor,
        build_matmul_case,
        normalize_dtype_name,
        parse_list_field,
        parse_shape,
    )
    from .replay_framework import OpReplay
except ImportError:
    from common import (
        build_input_tensor,
        build_matmul_case,
        normalize_dtype_name,
        parse_list_field,
        parse_shape,
    )
    from replay_framework import OpReplay


def _build_batched_case(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = [normalize_dtype_name(item) for item in parse_list_field(row["Input Data Types"])]
    output_shapes = [parse_shape(item) for item in parse_list_field(row["Output Shapes"])]
    output_shape = output_shapes[0] if output_shapes else None

    if len(input_shapes) < 2 or len(input_formats) < 2 or len(input_dtypes) < 2:
        raise ValueError("BatchMatMulV2 expects at least two inputs")
    if output_shape is None or len(output_shape) != 3:
        raise ValueError("BatchMatMulV2 batched replay expects a 3D output shape")

    a_shape, b_shape = input_shapes[0], input_shapes[1]
    if a_shape is None or b_shape is None or len(a_shape) != 3 or len(b_shape) != 3:
        raise ValueError("BatchMatMulV2 batched replay expects two 3D inputs")

    batch, m, k = a_shape
    out_batch, out_m, out_n = output_shape
    if batch != out_batch or m != out_m:
        raise ValueError(
            f"BatchMatMulV2 output {output_shape} is not compatible with first input {a_shape}"
        )

    transpose_b = False
    if b_shape == (batch, k, out_n):
        transpose_b = False
    elif b_shape == (batch, out_n, k):
        transpose_b = True
    else:
        raise ValueError(
            f"BatchMatMulV2 second input {b_shape} is not compatible with "
            f"first input {a_shape} and output {output_shape}"
        )

    return {
        "inputs": [
            build_input_tensor(
                shape=a_shape,
                input_format=input_formats[0],
                dtype_name=input_dtypes[0],
                transpose=False,
            ),
            build_input_tensor(
                shape=b_shape,
                input_format=input_formats[1],
                dtype_name=input_dtypes[1],
                transpose=False,
            ).transpose(-2, -1)
            if transpose_b
            else build_input_tensor(
                shape=b_shape,
                input_format=input_formats[1],
                dtype_name=input_dtypes[1],
                transpose=False,
            ),
        ],
        "kwargs": {},
    }


def build_case(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    first_shape = input_shapes[0] if input_shapes else None
    if first_shape is not None and len(first_shape) == 2:
        case = build_matmul_case(row, kernel_type="BatchMatMulV2", require_exact_inputs=True)
    else:
        case = _build_batched_case(row)
    case["api"] = op.resolve_api()
    return case


op = OpReplay(
    kernel_type="BatchMatMulV2",
    api_path="torch.matmul",
    description=(
        "Run BatchMatMulV2 workload replay on Ascend NPU.\n"
        "The script reads BatchMatMulV2.csv and reconstructs 2D or batched\n"
        "matmul inputs from the recorded shapes, formats, and dtypes."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/BatchMatMulV2_run.py "
        "--database-path tensor_cast/performance_model/profiling_database/data/"
        "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5",
    ],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    build_case=build_case,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
