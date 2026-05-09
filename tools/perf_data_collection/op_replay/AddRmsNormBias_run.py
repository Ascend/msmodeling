"""
Replay AddRmsNormBias cases from the performance database on Ascend NPU.

Purpose:
  Read AddRmsNormBias rows from
  profiling_database/data/{device}/vllm_ascend/{version}/AddRmsNormBias.csv,
  rebuild the recorded tensor inputs, then execute the exact microbench API.
"""

from __future__ import annotations

try:
    from .common import (
        build_input_tensor,
        get_runtime_modules,
        init_runtime,
        normalize_dtype_name,
        parse_shape_or_none,
        split_metadata_field,
    )
    from .replay_framework import OpReplay
except ImportError:
    from common import (
        build_input_tensor,
        get_runtime_modules,
        init_runtime,
        normalize_dtype_name,
        parse_shape_or_none,
        split_metadata_field,
    )
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    init_runtime()
    input_shapes = [parse_shape_or_none(item) for item in split_metadata_field(row["Input Shapes"])]
    input_formats = [item if item else "NULL" for item in split_metadata_field(row["Input Formats"])]
    input_dtypes = [normalize_dtype_name(item) for item in split_metadata_field(row["Input Data Types"])]
    output_shapes = [parse_shape_or_none(item) for item in split_metadata_field(row["Output Shapes"])]

    if not (len(input_shapes) == len(input_formats) == len(input_dtypes) == 4):
        raise ValueError(
            "AddRmsNormBias expects four input metadata slots, got "
            f"shapes={len(input_shapes)} formats={len(input_formats)} dtypes={len(input_dtypes)}"
        )

    x1_shape = input_shapes[0]
    x2_shape = input_shapes[1]
    gamma_shape = input_shapes[2]
    beta_shape = input_shapes[3]

    if any(input_shapes[index] is None for index in (0, 1, 2)):
        raise ValueError("AddRmsNormBias requires x1, x2, and gamma inputs")
    if len(x1_shape) != 2 or len(x2_shape) != 2:
        raise ValueError(f"AddRmsNormBias only supports 2D x inputs, got x1={x1_shape}, x2={x2_shape}")
    if x1_shape != x2_shape:
        raise ValueError(f"x1/x2 shapes must match, got x1={x1_shape}, x2={x2_shape}")
    if len(gamma_shape) != 1 or gamma_shape[0] != x1_shape[1]:
        raise ValueError(f"gamma must match hidden dim, got gamma={gamma_shape}, x={x1_shape}")
    if beta_shape is not None and (len(beta_shape) != 1 or beta_shape[0] != x1_shape[1]):
        raise ValueError(f"beta must match hidden dim, got beta={beta_shape}, x={x1_shape}")

    beta_is_absent = beta_shape is None or input_dtypes[3] == "DT_UNDEFINED" or input_formats[3] == "NULL"
    beta_tensor = None
    if not beta_is_absent:
        beta_tensor = build_input_tensor(beta_shape, input_formats[3], input_dtypes[3])

    return {
        "inputs": [
            build_input_tensor(x1_shape, input_formats[0], input_dtypes[0]),
            build_input_tensor(x2_shape, input_formats[1], input_dtypes[1]),
            build_input_tensor(gamma_shape, input_formats[2], input_dtypes[2]),
        ],
        "beta_tensor": beta_tensor,
        "expected_output_shapes": output_shapes,
        "epsilon": 1e-6,
        "kwargs": {},
        "api": None,
    }


def run_case(case):
    runtime_torch, runtime_torch_npu = get_runtime_modules()
    try:
        result = runtime_torch.ops._C_ascend.npu_add_rms_norm_bias(
            case["inputs"][0],
            case["inputs"][1],
            case["inputs"][2],
            case["beta_tensor"],
            case["epsilon"],
        )
        case["api_name"] = "torch.ops._C_ascend.npu_add_rms_norm_bias"
        return result
    except RuntimeError as exc:
        if "does not support opType [AddRmsNormBias]" not in str(exc):
            raise
        y, rstd, x = runtime_torch_npu.npu_add_rms_norm(
            case["inputs"][0],
            case["inputs"][1],
            case["inputs"][2],
            case["epsilon"],
        )
        if case["beta_tensor"] is not None:
            y = y.add(case["beta_tensor"])
        case["api_name"] = "torch_npu.npu_add_rms_norm(+bias fallback)"
        return y, rstd, x


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    y, rstd, x = result
    expected_shapes = case["expected_output_shapes"]
    actual_shapes = [tuple(y.shape), tuple(rstd.shape), tuple(x.shape)]
    for actual, expected, name in zip(actual_shapes, expected_shapes, ("y", "rstd", "x")):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} shape mismatch: actual={actual} expected={expected}")
    return (
        f"[OK] {csv_path}:{row_index} "
        f"api={case['api_name']} "
        f"shapes={row['Input Shapes']} dtypes={row['Input Data Types']} "
        f"beta={'present' if case['beta_tensor'] is not None else 'absent'} "
        f"y={tuple(y.shape)} rstd={tuple(rstd.shape)} x={tuple(x.shape)}"
    )


op = OpReplay(
    kernel_type="AddRmsNormBias",
    description=(
        "Run AddRmsNormBias microbenchmark rows on Ascend NPU.\n"
        "The script reads raw AddRmsNormBias.csv profiling rows,\n"
        "reconstructs the current 2D/1D workload shape, then executes\n"
        "the exact microbench API torch.ops._C_ascend.npu_add_rms_norm_bias()."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/AddRmsNormBias_run.py "
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
