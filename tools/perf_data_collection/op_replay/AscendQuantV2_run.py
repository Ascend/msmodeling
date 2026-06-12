"""
Replay AscendQuantV2 cases from the performance database on Ascend NPU.

Purpose:
  Read AscendQuantV2 rows from
  profiling_database/data/{device}/vllm_ascend/{version}/AscendQuantV2.csv,
  rebuild the recorded tensor inputs, then execute torch_npu.npu_quantize().
"""

from __future__ import annotations

try:
    from .common import (
        build_input_tensor,
        get_runtime_modules,
        normalize_dtype_name,
        parse_list_field,
        parse_shape,
    )
    from .replay_framework import OpReplay
except ImportError:
    from common import (
        build_input_tensor,
        get_runtime_modules,
        normalize_dtype_name,
        parse_list_field,
        parse_shape,
    )
    from replay_framework import OpReplay


def to_quantized_dtype(dtype_name: str):
    runtime_torch, _ = get_runtime_modules()
    normalized = normalize_dtype_name(dtype_name)
    dtype = {
        "DT_INT8": runtime_torch.qint8,
        "DT_UINT8": runtime_torch.quint8,
        "DT_INT32": runtime_torch.int32,
    }.get(normalized)
    if dtype is None and hasattr(runtime_torch, "quint4x2"):
        dtype = {"DT_INT4": runtime_torch.quint4x2}.get(normalized)
    if dtype is None:
        raise ValueError(f"Unsupported AscendQuantV2 output dtype: {dtype_name}")
    return dtype


def build_case(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = [normalize_dtype_name(item) for item in parse_list_field(row["Input Data Types"])]
    output_dtypes = [normalize_dtype_name(item) for item in parse_list_field(row["Output Data Types"])]

    if len(input_shapes) != len(input_formats) or len(input_shapes) != len(input_dtypes):
        raise ValueError("AscendQuantV2 input metadata length mismatch")
    if len(input_shapes) < 2:
        raise ValueError("AscendQuantV2 expects at least input and scales tensors")
    if len(output_dtypes) != 1:
        raise ValueError("AscendQuantV2 expects exactly one output dtype")

    inputs = [
        build_input_tensor(
            shape=input_shapes[0],
            input_format=input_formats[0],
            dtype_name=input_dtypes[0],
        ),
        build_input_tensor(
            shape=input_shapes[1],
            input_format=input_formats[1],
            dtype_name=input_dtypes[1],
        ),
    ]
    if len(input_shapes) >= 3:
        inputs.append(
            build_input_tensor(
                shape=input_shapes[2],
                input_format=input_formats[2],
                dtype_name=input_dtypes[2],
            )
        )

    return {
        "inputs": inputs,
        "kwargs": {
            "quant_dtype": to_quantized_dtype(output_dtypes[0]),
        },
        "api": op.resolve_api(),
    }


def run_case(case):
    zero_points = case["inputs"][2] if len(case["inputs"]) >= 3 else None
    return case["api"](
        case["inputs"][0],
        case["inputs"][1],
        zero_points,
        case["kwargs"]["quant_dtype"],
        -1,
        False,
    )


def format_success(csv_path, row_index: int, row: dict[str, str], _case, result) -> str:
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(result.shape)} "
        f"output_dtype={row['Output Data Types']}"
    )


op = OpReplay(
    kernel_type="AscendQuantV2",
    api_path="torch_npu.npu_quantize",
    description=(
        "Run AscendQuantV2 workload replay on Ascend NPU.\n"
        "The script reads AscendQuantV2.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs the recorded tensor\n"
        "inputs, then runs torch_npu.npu_quantize(axis=-1, div_mode=False)."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/AscendQuantV2_run.py "
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
