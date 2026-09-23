# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Replay runtime-described quantized MLA preprocess cases on Ascend NPU."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

try:
    from .common import (
        build_standard_argparser,
        case_belongs_to_shard,
        ensure_npu_available,
        get_runtime_modules,
        get_target_data_dir,
        normalize_dtype_name,
        parse_list_field,
        parse_shape,
        print_invalid_replay_summary,
        process_replay_csvs,
        record_runtime_replay_case,
    )
except ImportError:
    from common import (
        build_standard_argparser,
        case_belongs_to_shard,
        ensure_npu_available,
        get_runtime_modules,
        get_target_data_dir,
        normalize_dtype_name,
        parse_list_field,
        parse_shape,
        print_invalid_replay_summary,
        process_replay_csvs,
        record_runtime_replay_case,
    )

try:
    from .mla_preprocess_schema import (
        MLA_PREPROCESS_INPUT_DTYPES,
        MLA_PREPROCESS_INPUT_FORMATS,
        MLA_PREPROCESS_KERNEL,
        MLA_PREPROCESS_OUTPUT_DTYPES,
        MLA_PREPROCESS_OUTPUT_FORMATS,
        MlaPreprocessRuntime,
    )
    from .operator_metadata import get_operator_metadata
except ImportError:
    from mla_preprocess_schema import (
        MLA_PREPROCESS_INPUT_DTYPES,
        MLA_PREPROCESS_INPUT_FORMATS,
        MLA_PREPROCESS_KERNEL,
        MLA_PREPROCESS_OUTPUT_DTYPES,
        MLA_PREPROCESS_OUTPUT_FORMATS,
        MlaPreprocessRuntime,
    )
    from operator_metadata import get_operator_metadata


DEFAULT_WARMUP_COUNT = 3
DEFAULT_MEASURE_REPEAT_COUNT = 10
FRACTAL_NZ_FORMAT_ID = 29


def resolve_case_metadata(row: dict[str, str]) -> dict[str, Any]:
    runtime = MlaPreprocessRuntime.from_row(row)

    input_shapes = [
        parse_shape(value) for value in parse_list_field(row.get("Input Shapes", ""))
    ]
    output_shapes = [
        parse_shape(value) for value in parse_list_field(row.get("Output Shapes", ""))
    ]
    input_dtypes = tuple(
        normalize_dtype_name(value)
        for value in parse_list_field(row.get("Input Data Types", ""))
    )
    output_dtypes = tuple(
        normalize_dtype_name(value)
        for value in parse_list_field(row.get("Output Data Types", ""))
    )
    input_formats = tuple(parse_list_field(row.get("Input Formats", "")))
    output_formats = tuple(parse_list_field(row.get("Output Formats", "")))
    expected_inputs, expected_outputs = runtime.shapes()
    if input_shapes != expected_inputs or output_shapes != expected_outputs:
        raise ValueError("MLAPO Shape columns conflict with the Runtime metadata")
    if (
        input_dtypes != MLA_PREPROCESS_INPUT_DTYPES
        or input_formats != MLA_PREPROCESS_INPUT_FORMATS
    ):
        raise ValueError(
            "MLAPO input dtype/format columns do not match the replay contract"
        )
    if (
        output_dtypes != MLA_PREPROCESS_OUTPUT_DTYPES
        or output_formats != MLA_PREPROCESS_OUTPUT_FORMATS
    ):
        raise ValueError(
            "MLAPO output dtype/format columns do not match the replay contract"
        )

    return {
        "case_id": runtime.case_id,
        "num_tokens": runtime.num_tokens,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "input_dtypes": input_dtypes,
        "input_formats": input_formats,
        "cache_mode": runtime.cache_mode,
        "quant_mode": runtime.quant_mode,
        "enable_inner_out": runtime.enable_inner_out,
    }


def _build_tensor(shape: tuple[int, ...], dtype_name: str, input_format: str):
    runtime_torch, runtime_torch_npu = get_runtime_modules()
    dtype_map = {
        "DT_BF16": runtime_torch.bfloat16,
        "DT_FLOAT": runtime_torch.float32,
        "DT_INT8": runtime_torch.int8,
        "DT_INT32": runtime_torch.int32,
    }
    dtype = dtype_map[dtype_name]
    if dtype in {runtime_torch.bfloat16, runtime_torch.float32}:
        tensor = runtime_torch.randn(shape, dtype=dtype).npu()
    else:
        tensor = runtime_torch.randint(0, 8, shape, dtype=dtype).npu()
    if input_format == "FRACTAL_NZ":
        tensor = runtime_torch_npu.npu_format_cast(
            tensor.contiguous(), FRACTAL_NZ_FORMAT_ID
        )
    return tensor


def build_case(row: dict[str, str]) -> dict[str, Any]:
    metadata = resolve_case_metadata(row)
    tensors = [
        _build_tensor(shape, dtype_name, input_format)
        for shape, dtype_name, input_format in zip(
            metadata["input_shapes"],
            metadata["input_dtypes"],
            metadata["input_formats"],
        )
    ]
    runtime_torch, _ = get_runtime_modules()
    cache_capacity = tensors[11].shape[0] * tensors[11].shape[1]
    tensors[13] = (
        runtime_torch.arange(metadata["num_tokens"], dtype=runtime_torch.int32)
        .remainder(cache_capacity)
        .npu()
    )
    output_shapes = metadata["output_shapes"]
    outputs = [
        runtime_torch.empty(
            output_shapes[0], dtype=runtime_torch.bfloat16, device="npu"
        ),
        tensors[11],
        runtime_torch.empty(
            output_shapes[2], dtype=runtime_torch.bfloat16, device="npu"
        ),
        tensors[12],
        runtime_torch.empty(
            output_shapes[4], dtype=runtime_torch.bfloat16, device="npu"
        ),
    ]
    metadata["inputs"] = tensors
    metadata["outputs"] = outputs
    return metadata


def run_case(case: dict[str, Any]):
    runtime_torch, _ = get_runtime_modules()
    inputs = case["inputs"]
    outputs = case["outputs"]
    return runtime_torch.ops._C_ascend.mla_preprocess(
        *inputs[:14],
        quant_scale0=inputs[14],
        quant_offset0=inputs[15],
        bias0=inputs[16],
        quant_scale1=inputs[17],
        quant_offset1=inputs[18],
        bias1=inputs[19],
        ctkv_scale=inputs[20],
        q_nope_scale=inputs[21],
        cache_mode=case["cache_mode"],
        quant_mode=case["quant_mode"],
        enable_inner_out=case["enable_inner_out"],
        q_out0=outputs[0],
        kv_cache_out0=outputs[1],
        q_out1=outputs[2],
        kv_cache_out1=outputs[3],
        inner_out=outputs[4],
    )


def _cleanup_case() -> None:
    runtime_torch, _ = get_runtime_modules()
    gc.collect()
    runtime_torch.npu.empty_cache()


def main() -> None:
    parser = build_standard_argparser(
        description="Replay runtime-described quantized mla_preprocess cases.",
        usage_examples=[
            "python mla_preprocess_0_mix_aic_run.py --database-path /path/to/database"
        ],
        version_help="vLLM-Ascend version containing mla_preprocess, e.g. 0.18.0.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Replay only the selected Runtime case_id. May be repeated.",
    )
    args = parser.parse_args()
    repeat_count = args.repeat_count or DEFAULT_MEASURE_REPEAT_COUNT
    if repeat_count <= 0:
        raise ValueError("--repeat-count must be positive")
    if (
        args.case_shard_count <= 0
        or not 0 <= args.case_shard_index < args.case_shard_count
    ):
        raise ValueError("case shard index must be in [0, case shard count)")
    ensure_npu_available()
    get_runtime_modules()

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_version,
        database_path=args.database_path,
        torch_version=args.torch_version,
        cann_version=args.cann_version,
    )
    csv_paths = sorted(target_data_dir.rglob(f"{MLA_PREPROCESS_KERNEL}.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No {MLA_PREPROCESS_KERNEL}.csv found under {target_data_dir}"
        )

    def replay_row(csv_path: Path, row_index: int, row: dict[str, str]) -> None:
        case = build_case(row)
        runtime_torch, _ = get_runtime_modules()
        for _ in range(DEFAULT_WARMUP_COUNT):
            run_case(case)
            runtime_torch.npu.synchronize()
        for _ in range(repeat_count):
            run_case(case)
            runtime_torch.npu.synchronize()
        operator_metadata = get_operator_metadata(MLA_PREPROCESS_KERNEL)
        record_runtime_replay_case(
            kernel_type=MLA_PREPROCESS_KERNEL,
            case_id=case["case_id"],
            csv_path=csv_path,
            row_index=row_index,
            warmup_count=DEFAULT_WARMUP_COUNT,
            repeat_count=repeat_count,
            aggregation="mean",
            require_task_start_time=True,
            kernel_name_prefix=operator_metadata.profiler_kernel_prefix,
            profile_kernel_type=operator_metadata.aliases[0] if operator_metadata.aliases else operator_metadata.canonical_name,
            expected_task_type=operator_metadata.profiler_task_type,
        )
        print(
            f"[OK] {csv_path}:{row_index} case_id={case['case_id']} "
            f"warmup={DEFAULT_WARMUP_COUNT} repeat={repeat_count}"
        )

    selected_case_ids = set(args.case_id or [])

    def should_skip_row(_csv_path: Path, _row_index: int, row: dict[str, str]) -> bool:
        case_id = (row.get("Runtime case_id", "") or "").strip()
        if selected_case_ids and case_id not in selected_case_ids:
            return True
        return not case_belongs_to_shard(
            case_id, args.case_shard_count, args.case_shard_index
        )

    total_rows, invalid_rows, _, skipped_rows = process_replay_csvs(
        kernel_type=MLA_PREPROCESS_KERNEL,
        csv_paths=csv_paths,
        repeat_count=1,
        run_row_fn=replay_row,
        update_mode=args.update_mode,
        should_skip_row=should_skip_row,
        on_row_finally=_cleanup_case,
    )
    print(
        f"Processed {total_rows} {MLA_PREPROCESS_KERNEL} case(s); "
        f"skipped {skipped_rows} row(s)."
    )
    print_invalid_replay_summary(invalid_rows, label=MLA_PREPROCESS_KERNEL)


if __name__ == "__main__":
    main()
