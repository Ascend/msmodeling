"""
Replay QuantBatchMatmulV3 cases from the performance database on Ascend NPU.

Purpose:
  Read QuantBatchMatmulV3 rows from
  profiling_database/data/{device}/vllm_ascend/{version}/QuantBatchMatmulV3.csv,
  rebuild tensors from the recorded Input Shapes / Input Data Types /
  Input Formats, then execute torch_npu.npu_quant_matmul() with the same
  tensor metadata layout as the profiled row.

Notes:
  - The current atlas_800_a3_752t_128g_die/v0.13.0 dataset contains INT8
    QuantBatchMatmulV3 cases only.
  - FRACTAL_NZ inputs expand the recorded NZ shape to its ND equivalent,
    then cast that logical KxN tensor to internal FRACTAL_NZ format on NPU.
"""

from __future__ import annotations

from functools import lru_cache

try:
    from .common import (
        build_input_tensor,
        build_standard_argparser,
        ensure_npu_available,
        get_replay_repeat_count,
        get_runtime_modules,
        get_target_data_dir,
        normalize_dtype_name,
        parse_list_field,
        parse_shape,
        print_invalid_replay_summary,
        process_replay_csvs,
    )
except ImportError:
    from common import (
        build_input_tensor,
        build_standard_argparser,
        ensure_npu_available,
        get_replay_repeat_count,
        get_runtime_modules,
        get_target_data_dir,
        normalize_dtype_name,
        parse_list_field,
        parse_shape,
        print_invalid_replay_summary,
        process_replay_csvs,
    )

def to_output_dtype(dtype_name: str):
    runtime_torch, _ = get_runtime_modules()
    dtype = {
        "DT_FLOAT": runtime_torch.float32,
        "DT_FLOAT16": runtime_torch.float16,
        "DT_BF16": runtime_torch.bfloat16,
        "DT_INT8": runtime_torch.int8,
        "DT_INT32": runtime_torch.int32,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported QuantBatchMatmulV3 output dtype: {dtype_name}")
    return dtype


def shape_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def infer_optional_input_role(
    x_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    optional_shape: tuple[int, ...],
    optional_dtype: str,
) -> str:
    m_dim = x_shape[-2] if len(x_shape) >= 2 else x_shape[-1]
    n_dim = output_shape[-1]
    optional_numel = shape_numel(optional_shape)

    if optional_dtype == "DT_INT32":
        return "bias"

    if optional_dtype == "DT_FLOAT":
        if optional_numel == m_dim:
            return "pertoken_scale"
        if optional_numel in {1, n_dim}:
            return "offset"

    raise ValueError(
        "Unable to infer QuantBatchMatmulV3 optional input role for "
        f"shape={optional_shape}, dtype={optional_dtype}, x_shape={x_shape}, "
        f"output_shape={output_shape}"
    )


class QuantMatmulGraphModule:
    def __init__(
        self,
        output_dtype_name: str,
        has_bias: bool,
        has_offset: bool,
        has_pertoken_scale: bool,
    ):
        runtime_torch, runtime_torch_npu = get_runtime_modules()
        output_dtype = to_output_dtype(output_dtype_name)

        class _Module(runtime_torch.nn.Module):
            def forward(self, x, weight, scale, bias, offset, pertoken_scale):
                kwargs = {"output_dtype": output_dtype}
                if has_bias:
                    kwargs["bias"] = bias
                if has_offset:
                    kwargs["offset"] = offset
                if has_pertoken_scale:
                    kwargs["pertoken_scale"] = pertoken_scale
                return runtime_torch_npu.npu_quant_matmul(x, weight, scale, **kwargs)

        self._module = _Module().npu()


@lru_cache(maxsize=None)
def get_quant_matmul_graph_runner(
    dtype_signature: tuple[str, str, str, str, str, str, str, bool, bool, bool],
):
    runtime_torch, _ = get_runtime_modules()
    output_dtype_name = dtype_signature[6]
    has_bias = dtype_signature[7]
    has_offset = dtype_signature[8]
    has_pertoken_scale = dtype_signature[9]
    try:
        import torchair as tng
        from torchair.configs.compiler_config import CompilerConfig
    except ImportError as exc:
        raise RuntimeError(
            "FRACTAL_NZ QuantBatchMatmulV3 replay requires torchair graph mode"
        ) from exc

    config = CompilerConfig()
    backend = tng.get_npu_backend(compiler_config=config)
    model = QuantMatmulGraphModule(
        output_dtype_name=output_dtype_name,
        has_bias=has_bias,
        has_offset=has_offset,
        has_pertoken_scale=has_pertoken_scale,
    )
    return runtime_torch.compile(model._module, backend=backend, dynamic=True)


def build_quant_matmul_case(
    input_shapes: list[tuple[int, ...]],
    input_formats: list[str],
    input_dtypes: list[str],
    output_shape: tuple[int, ...],
    output_dtype_name: str,
):
    if len(input_shapes) not in {3, 4}:
        raise ValueError("QuantBatchMatmulV3 replay currently supports three or four inputs")

    case = {
        "weight_format": input_formats[1],
        "x_tensor": build_input_tensor(
            shape=input_shapes[0],
            input_format=input_formats[0],
            dtype_name=input_dtypes[0],
            transpose=False,
        ),
        "weight_tensor": build_input_tensor(
            shape=input_shapes[1],
            input_format=input_formats[1],
            dtype_name=input_dtypes[1],
            transpose=False,
        ),
        "scale_tensor": build_input_tensor(
            shape=input_shapes[2],
            input_format=input_formats[2],
            dtype_name=input_dtypes[2],
            transpose=False,
        ),
        "bias_tensor": None,
        "offset_tensor": None,
        "pertoken_scale_tensor": None,
        "output_dtype_name": output_dtype_name,
    }

    if len(input_shapes) == 4:
        optional_role = infer_optional_input_role(
            x_shape=input_shapes[0],
            output_shape=output_shape,
            optional_shape=input_shapes[3],
            optional_dtype=input_dtypes[3],
        )
        case[f"{optional_role}_tensor"] = build_input_tensor(
            shape=input_shapes[3],
            input_format=input_formats[3],
            dtype_name=input_dtypes[3],
            transpose=False,
        )

    return case


def build_row_tensors(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = [normalize_dtype_name(item) for item in parse_list_field(row["Input Data Types"])]
    output_shapes = [parse_shape(item) for item in parse_list_field(row["Output Shapes"])]
    output_dtypes = [normalize_dtype_name(item) for item in parse_list_field(row["Output Data Types"])]

    if len(output_shapes) != 1 or len(output_dtypes) != 1:
        raise ValueError("QuantBatchMatmulV3 expects exactly one output")
    if len(input_shapes) != len(input_formats) or len(input_shapes) != len(input_dtypes):
        raise ValueError("QuantBatchMatmulV3 input metadata length mismatch")
    if input_dtypes[:2] != ["DT_INT8", "DT_INT8"]:
        raise ValueError(
            "QuantBatchMatmulV3 replay currently supports the INT8 x INT8 rows "
            f"recorded in the CSV, got {input_dtypes[:2]}"
        )

    return build_quant_matmul_case(
        input_shapes=input_shapes,
        input_formats=input_formats,
        input_dtypes=input_dtypes,
        output_shape=output_shapes[0],
        output_dtype_name=output_dtypes[0],
    )


def run_quant_matmul(
    x_tensor,
    weight_tensor,
    scale_tensor,
    bias_tensor,
    offset_tensor,
    pertoken_scale_tensor,
    output_dtype_name: str,
    use_graph_mode: bool,
):
    runtime_torch_npu = get_runtime_modules()[1]

    kwargs = {"output_dtype": to_output_dtype(output_dtype_name)}
    if bias_tensor is not None:
        kwargs["bias"] = bias_tensor
    if offset_tensor is not None:
        kwargs["offset"] = offset_tensor
    if pertoken_scale_tensor is not None:
        kwargs["pertoken_scale"] = pertoken_scale_tensor

    if not use_graph_mode:
        return runtime_torch_npu.npu_quant_matmul(
            x_tensor,
            weight_tensor,
            scale_tensor,
            **kwargs,
        )

    dtype_signature = (
        str(x_tensor.dtype),
        str(weight_tensor.dtype),
        str(scale_tensor.dtype),
        str(bias_tensor.dtype) if bias_tensor is not None else "None",
        str(offset_tensor.dtype) if offset_tensor is not None else "None",
        str(pertoken_scale_tensor.dtype) if pertoken_scale_tensor is not None else "None",
        output_dtype_name,
        bias_tensor is not None,
        offset_tensor is not None,
        pertoken_scale_tensor is not None,
    )
    runner = get_quant_matmul_graph_runner(dtype_signature)
    return runner(
        x_tensor,
        weight_tensor,
        scale_tensor,
        bias_tensor,
        offset_tensor,
        pertoken_scale_tensor,
    )


def build_argparser():
    return build_standard_argparser(
        description=(
            "Run QuantBatchMatmulV3 workload replay on Ascend NPU.\n"
            "The script reads QuantBatchMatmulV3.csv under the selected device and\n"
            "vllm_ascend version directory, rebuilds input tensors from\n"
            "Input Shapes / Input Formats / Input Data Types, infers optional\n"
            "inputs from the recorded metadata, then runs torch_npu.npu_quant_matmul().\n"
            "FRACTAL_NZ weight rows are replayed in graph mode."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/QuantBatchMatmulV3_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0",
            "python tools/perf_data_collection/op_replay/QuantBatchMatmulV3_run.py "
            "--device TEST_DEVICE --vllm-version 0.9.2",
        ],
        version_help="vLLM-Ascend version, e.g. 0.13.0.",
    )


def run_row(csv_path, row_index: int, row: dict[str, str]) -> None:
    runtime_torch, _ = get_runtime_modules()
    replay_case = build_row_tensors(row)
    use_graph_mode = replay_case["weight_format"] == "FRACTAL_NZ"

    result = run_quant_matmul(
        replay_case["x_tensor"],
        replay_case["weight_tensor"],
        replay_case["scale_tensor"],
        replay_case["bias_tensor"],
        replay_case["offset_tensor"],
        replay_case["pertoken_scale_tensor"],
        replay_case["output_dtype_name"],
        use_graph_mode,
    )

    runtime_torch.npu.synchronize()

    print(
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} output={tuple(result.shape)}"
    )


def main() -> None:
    args = build_argparser().parse_args()
    repeat_count = get_replay_repeat_count(args.repeat_count)
    ensure_npu_available()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_version,
        database_path=args.database_path,
        torch_version=args.torch_version,
        cann_version=args.cann_version,
    )
    csv_paths = sorted(target_data_dir.rglob("QuantBatchMatmulV3.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No QuantBatchMatmulV3.csv found under {target_data_dir}")

    total_rows, invalid_rows, _, skipped_rows = process_replay_csvs(
        kernel_type="QuantBatchMatmulV3",
        csv_paths=csv_paths,
        repeat_count=repeat_count,
        run_row_fn=run_row,
        update_mode=args.update_mode,
    )

    print(
        f"Processed {total_rows} QuantBatchMatmulV3 rows from {len(csv_paths)} csv file(s) "
        f"under {target_data_dir}."
    )
    if args.update_mode == "missing-only":
        print(f"[SUMMARY] QuantBatchMatmulV3: skipped {skipped_rows} row(s) due to missing-only mode.")
    print_invalid_replay_summary(invalid_rows, label="QuantBatchMatmulV3")


if __name__ == "__main__":
    main()

