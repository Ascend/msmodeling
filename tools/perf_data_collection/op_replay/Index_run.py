"""
Replay Index cases from the performance database on Ascend NPU.

Purpose:
  Read Index rows from
  profiling_database/data/{device}/vllm_ascend/{version}/Index.csv,
  rebuild the recorded source tensor plus a legal 1D index tensor, then
  execute the Python indexing path that lowers to aten.index.Tensor /
  aclnnIndex on NPU.

Notes:
  - Current profiling rows are all the simple "select rows on dim 0" shape:
    `source;1;2;index_len -> output(index_len, ...)`.
  - The trailing scalar slots are recorded by the kernel CSV but are not
    needed for replay. We only use them to validate the expected shape pattern.
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


def _parse_row_metadata(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_dtypes = [
        normalize_dtype_name(item) for item in parse_list_field(row["Input Data Types"])
    ]
    input_formats = parse_list_field(row["Input Formats"])
    output_shapes = [parse_shape(item) for item in parse_list_field(row["Output Shapes"])]
    return input_shapes, input_dtypes, input_formats, output_shapes


def build_case(row: dict[str, str]):
    runtime_torch, _ = get_runtime_modules()
    input_shapes, input_dtypes, input_formats, output_shapes = _parse_row_metadata(row)

    if len(input_shapes) != 4 or len(input_dtypes) != 4:
        raise ValueError(
            "Index expects four recorded inputs: source + three scalar metadata slots"
        )
    if not output_shapes:
        raise ValueError("Index replay requires a recorded output shape")

    source_shape = input_shapes[0]
    if not source_shape:
        raise ValueError("Index replay requires a non-scalar source tensor")

    output_shape = output_shapes[0]
    if not output_shape:
        raise ValueError("Index replay requires a parseable output shape")

    index_len = output_shape[0]
    if index_len == 0:
        raise ValueError(f"Index replay requires non-empty output: {output_shape}")
    if index_len < 0:
        raise ValueError(f"Invalid output shape for Index: {output_shape}")
    if source_shape[0] < index_len:
        raise ValueError(
            f"Index output rows exceed source capacity: source={source_shape}, output={output_shape}"
        )

    source_tensor = build_input_tensor(
        shape=source_shape,
        input_format=input_formats[0] if input_formats else "ND",
        dtype_name=input_dtypes[0],
    )

    # Slot 3 records the index-length scalar; its dtype matches the runtime
    # index tensor dtype used by aten.index.Tensor replay.
    index_dtype_name = input_dtypes[3]
    if index_dtype_name not in {"DT_INT32", "DT_INT64"}:
        raise ValueError(f"Unsupported Index dtype: {index_dtype_name}")
    index_dtype = (
        runtime_torch.int64 if index_dtype_name == "DT_INT64" else runtime_torch.int32
    )
    index_tensor = runtime_torch.arange(index_len, dtype=index_dtype).npu()

    return {
        "inputs": [source_tensor, index_tensor],
        "kwargs": {},
        "source_shape": source_shape,
        "output_shape": output_shape,
        "index_len": index_len,
        "index_dtype": index_dtype_name,
        "api": None,
    }


def run_case(case):
    source_tensor, index_tensor = case["inputs"]
    return source_tensor[index_tensor]


def format_success(csv_path: str, row_index: int, row: dict[str, str], case: dict, result) -> str:
    output_shape = tuple(result.shape) if hasattr(result, "shape") else str(result)
    return (
        f"[OK] {csv_path}:{row_index} "
        f"source={case['source_shape']} index_len={case['index_len']} "
        f"index_dtype={case['index_dtype']} output={output_shape}"
    )


op = OpReplay(
    kernel_type="Index",
    description=(
        "Run Index workload replay on Ascend NPU.\n"
        "The script reads Index.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs the recorded source\n"
        "tensor, builds a legal 1D row index tensor, then runs source[index]."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/Index_run.py "
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
