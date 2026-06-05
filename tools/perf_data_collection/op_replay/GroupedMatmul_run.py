"""
Replay GroupedMatmul cases from the performance database on Ascend NPU.

Purpose:
  Read GroupedMatmul rows from the profiling database,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch_npu.npu_grouped_matmul().

Usage:
  python tools/perf_data_collection/op_replay/GroupedMatmul_run.py ^
    --device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.18.0
"""

from __future__ import annotations

try:
    from .common import (
        FRACTAL_NZ_FORMAT_ID,
        build_host_tensor,
        build_input_tensor,
        get_runtime_modules,
        init_runtime,
        normalize_dtype_name,
        parse_shape_or_none,
        resolve_runtime_dtype,
        split_metadata_field,
    )
    from .replay_framework import OpReplay
except ImportError:
    from common import (
        FRACTAL_NZ_FORMAT_ID,
        build_host_tensor,
        build_input_tensor,
        get_runtime_modules,
        init_runtime,
        normalize_dtype_name,
        parse_shape_or_none,
        resolve_runtime_dtype,
        split_metadata_field,
    )
    from replay_framework import OpReplay


def _build_scale_tensor(shape, dtype_name: str, input_format: str):
    runtime_torch, runtime_torch_npu = get_runtime_modules()
    dtype = resolve_runtime_dtype(dtype_name)
    tensor = runtime_torch.ones(shape, dtype=dtype).npu()
    if input_format == "FRACTAL_NZ":
        return runtime_torch_npu.npu_format_cast(tensor, FRACTAL_NZ_FORMAT_ID)
    return tensor


def _build_group_list_tensor(total_tokens: int, num_groups: int, dtype_name: str):
    runtime_torch, _ = get_runtime_modules()
    dtype = resolve_runtime_dtype(dtype_name)
    base = total_tokens // num_groups
    remainder = total_tokens % num_groups
    cumulative = []
    running_total = 0
    for idx in range(num_groups):
        running_total += base + (1 if idx < remainder else 0)
        cumulative.append(running_total)
    return runtime_torch.tensor(cumulative, dtype=dtype).npu()


def _build_grouped_weight_tensor(shape, dtype_name: str, input_format: str):
    if input_format == "FRACTAL_NZ" and len(shape) == 3:
        _, runtime_torch_npu = get_runtime_modules()
        dtype = resolve_runtime_dtype(dtype_name)
        tensor = build_host_tensor(shape, dtype).npu()
        return runtime_torch_npu.npu_format_cast(tensor, FRACTAL_NZ_FORMAT_ID)
    return build_input_tensor(shape=shape, input_format=input_format, dtype_name=dtype_name)


def build_case(row: dict[str, str]):
    """Build kwargs for torch_npu.npu_grouped_matmul.

    CSV layout (positions 0-8, some empty):  x(0), weight(1), (2:empty),
    scale(3), (4-6:empty), group_list(7), per_token_scale(8).
    """
    init_runtime()

    shapes = split_metadata_field(row["Input Shapes"])
    dtypes = [normalize_dtype_name(d) for d in split_metadata_field(row["Input Data Types"])]
    formats = split_metadata_field(row["Input Formats"])
    output_dtypes = [
        normalize_dtype_name(d)
        for d in split_metadata_field(row.get("Output Data Types", ""))
        if d.strip()
    ]

    def _tensor(pos):
        s = parse_shape_or_none(shapes[pos]) if pos < len(shapes) else None
        if s is None:
            return None
        dt = dtypes[pos] if pos < len(dtypes) else "DT_BF16"
        fmt = formats[pos] if pos < len(formats) else "ND"
        return build_input_tensor(shape=s, input_format=fmt, dtype_name=dt)

    x = _tensor(0)
    weight_shape = parse_shape_or_none(shapes[1]) if len(shapes) > 1 else None
    if weight_shape is None:
        raise ValueError("GroupedMatmul requires a non-empty weight shape")
    weight = _build_grouped_weight_tensor(weight_shape, dtypes[1], formats[1])
    scale_shape = parse_shape_or_none(shapes[3]) if len(shapes) > 3 else None
    scale = (
        _build_scale_tensor(scale_shape, dtypes[3], formats[3])
        if scale_shape is not None
        else None
    )
    group_list_shape = parse_shape_or_none(shapes[7]) if len(shapes) > 7 else None
    group_list = (
        _build_group_list_tensor(x.shape[0], group_list_shape[0], dtypes[7])
        if group_list_shape is not None
        else None
    )
    per_token_scale_shape = parse_shape_or_none(shapes[8]) if len(shapes) > 8 else None
    per_token_scale = (
        _build_scale_tensor(per_token_scale_shape, dtypes[8], formats[8])
        if per_token_scale_shape is not None
        else None
    )

    api = op.resolve_api()
    kw: dict = {
        "x": [x],
        "weight": [weight],
        "split_item": 2,
        "group_list_type": 0,
        "group_type": 0,
        "group_list": group_list,
    }
    if scale is not None:
        kw["scale"] = [scale]
    if per_token_scale is not None:
        kw["per_token_scale"] = [per_token_scale]
    if output_dtypes:
        kw["output_dtype"] = resolve_runtime_dtype(output_dtypes[0])
    return {"inputs": [x, weight], "kwargs": kw, "api": api}


def run_case(case):
    return case["api"](**case["kwargs"])


op = OpReplay(
    kernel_type="GroupedMatmul",
    api_path="torch_npu.npu_grouped_matmul",
    description=(
        "Run GroupedMatmul workload replay on Ascend NPU.\n"
        "Grouped matrix multiplication for MoE weighted-sum output projection."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/GroupedMatmul_run.py "
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
