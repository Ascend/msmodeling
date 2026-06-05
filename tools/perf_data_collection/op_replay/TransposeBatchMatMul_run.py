"""
Replay TransposeBatchMatMul cases from the performance database on Ascend NPU.

Purpose:
  Read TransposeBatchMatMul rows from the selected profiling database directory,
  rebuild the recorded input tensors, then execute
  torch_npu.npu_transpose_batchmatmul() so msprof captures the fused
  TransposeBatchMatMul kernel instead of separate BatchMatMulV2 and Transpose
  kernels.
"""

from __future__ import annotations

try:
    from .common import build_input_tensor, get_runtime_modules, normalize_dtype_name, parse_list_field, parse_shape
    from .replay_framework import OpReplay
except ImportError:
    from common import build_input_tensor, get_runtime_modules, normalize_dtype_name, parse_list_field, parse_shape
    from replay_framework import OpReplay


def _validate_input_metadata(
    input_shapes: list[tuple[int, ...]],
    input_formats: list[str],
    input_dtypes: list[str],
) -> None:
    if len(input_shapes) != 2:
        raise ValueError(f"TransposeBatchMatMul expects exactly two inputs, got {len(input_shapes)}")
    if len(input_formats) != 2 or len(input_dtypes) != 2:
        raise ValueError(
            "TransposeBatchMatMul input metadata length mismatch: "
            f"shapes={len(input_shapes)}, formats={len(input_formats)}, dtypes={len(input_dtypes)}"
        )
    if len(input_shapes[0]) != 3 or len(input_shapes[1]) != 3:
        raise ValueError(f"TransposeBatchMatMul currently supports only 3D inputs, got {input_shapes}")
    if input_formats != ["ND", "ND"]:
        raise ValueError(f"TransposeBatchMatMul currently supports only ND inputs, got {input_formats}")

    lhs_batch, _, lhs_k = input_shapes[0]
    rhs_batch, rhs_k, rhs_n = input_shapes[1]
    if lhs_batch != rhs_batch or lhs_k != rhs_k:
        raise ValueError(f"TransposeBatchMatMul input shapes are not bmm-compatible: {input_shapes}")
    if lhs_k % 128 != 0 or rhs_n % 128 != 0:
        raise ValueError(
            "TransposeBatchMatMul requires K and N to be divisible by 128, "
            f"got K={lhs_k}, N={rhs_n}"
        )


def _validate_output_layout(
    input_shapes: list[tuple[int, ...]],
    output_shape: tuple[int, ...],
) -> None:
    lhs_batch, lhs_m, _ = input_shapes[0]
    _, _, rhs_n = input_shapes[1]
    transposed_output_shape = (lhs_m, lhs_batch, rhs_n)

    if output_shape == transposed_output_shape:
        return
    raise ValueError(
        "TransposeBatchMatMul replay requires the fused kernel output layout "
        f"(M, B, N). Got output={output_shape}, expected={transposed_output_shape}"
    )


def build_case(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = [normalize_dtype_name(item) for item in parse_list_field(row["Input Data Types"])]
    output_shapes = [parse_shape(item) for item in parse_list_field(row["Output Shapes"])]

    _validate_input_metadata(input_shapes, input_formats, input_dtypes)
    if len(output_shapes) != 1:
        raise ValueError(f"TransposeBatchMatMul expects exactly one output, got {len(output_shapes)}")

    _, runtime_torch_npu = get_runtime_modules()
    _validate_output_layout(input_shapes, output_shapes[0])

    return {
        "inputs": [
            build_input_tensor(input_shapes[0], input_formats[0], input_dtypes[0]),
            build_input_tensor(input_shapes[1], input_formats[1], input_dtypes[1]),
        ],
        "api_kwargs": {
            "bias": None,
            "scale": None,
            "perm_x1": [0, 1, 2],
            "perm_x2": [0, 1, 2],
            "perm_y": [1, 0, 2],
            "batch_split_factor": 1,
        },
        "metadata": {
            "output_shape": output_shapes[0],
        },
        "api": runtime_torch_npu.npu_transpose_batchmatmul,
    }


def run_case(case):
    return case["api"](*case["inputs"], **case["api_kwargs"])


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(result.shape)} "
        f"perm_x1={case['api_kwargs']['perm_x1']} "
        f"perm_x2={case['api_kwargs']['perm_x2']} "
        f"perm_y={case['api_kwargs']['perm_y']}"
    )


op = OpReplay(
    kernel_type="TransposeBatchMatMul",
    input_count=2,
    description=(
        "Run TransposeBatchMatMul workload replay on Ascend NPU.\n"
        "The script reads TransposeBatchMatMul.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs 3D ND input tensors from\n"
        "Input Shapes / Input Formats / Input Data Types, then runs\n"
        "torch_npu.npu_transpose_batchmatmul() with perm_y=[1, 0, 2] to\n"
        "produce the recorded (M, B, N) output layout."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/TransposeBatchMatMul_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.18.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.18.0.",
    build_case=build_case,
    run_case=run_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
