"""
Replay KvRmsNormRopeCache cases from the performance database on Ascend NPU.

Purpose:
  Read KvRmsNormRopeCache rows from
  profiling_database/data/{device}/vllm_ascend/{version}/KvRmsNormRopeCache.csv,
  rebuild the recorded tensor inputs, infer the required scalar arguments,
  then execute torch_npu.npu_kv_rmsnorm_rope_cache().
"""

from __future__ import annotations

try:
    from .common import (
        build_input_tensor,
        get_runtime_modules,
        parse_shape_or_none,
        split_metadata_field,
    )
    from .replay_framework import OpReplay
except ImportError:
    from common import (
        build_input_tensor,
        get_runtime_modules,
        parse_shape_or_none,
        split_metadata_field,
    )
    from replay_framework import OpReplay


def build_index_tensor(index_shape: tuple[int, ...], cache_shape: tuple[int, ...]):
    runtime_torch, _ = get_runtime_modules()
    if len(index_shape) != 1:
        raise ValueError(f"index must be 1D, got shape={index_shape}")
    if len(cache_shape) < 2:
        raise ValueError(f"cache rank must be >= 2, got shape={cache_shape}")

    token_count = index_shape[0]
    total_slots = cache_shape[0] * cache_shape[1]
    if token_count > total_slots:
        raise ValueError(
            "index token count exceeds cache capacity: "
            f"tokens={token_count}, total_slots={total_slots}, cache_shape={cache_shape}"
        )

    return runtime_torch.arange(token_count, dtype=runtime_torch.int64, device="npu")


def infer_cache_mode(k_cache_shape: tuple[int, ...], index_shape: tuple[int, ...]) -> str:
    if len(k_cache_shape) != 4:
        raise ValueError(f"Unsupported k_cache shape for cache_mode inference: {k_cache_shape}")
    if len(index_shape) != 1:
        raise ValueError(f"Unsupported index shape for cache_mode inference: {index_shape}")
    return "PA_BNSD"


def build_case(row: dict[str, str]):
    input_shapes = [parse_shape_or_none(item) for item in split_metadata_field(row["Input Shapes"])]
    input_dtypes = [item.strip() for item in split_metadata_field(row["Input Data Types"])]
    input_formats = [item.strip() if item.strip() else "NULL" for item in split_metadata_field(row["Input Formats"])]

    if not (len(input_shapes) == len(input_dtypes) == len(input_formats) == 12):
        raise ValueError(
            "KvRmsNormRopeCache expects 12 input metadata slots, got "
            f"shapes={len(input_shapes)} dtypes={len(input_dtypes)} formats={len(input_formats)}"
        )

    return {
        "inputs": [
            build_input_tensor(input_shapes[0], input_formats[0], input_dtypes[0]),
            build_input_tensor(input_shapes[1], input_formats[1], input_dtypes[1]),
            build_input_tensor(input_shapes[2], input_formats[2], input_dtypes[2]),
            build_input_tensor(input_shapes[3], input_formats[3], input_dtypes[3]),
            build_index_tensor(input_shapes[4], input_shapes[5]),
            build_input_tensor(input_shapes[5], input_formats[5], input_dtypes[5]),
            build_input_tensor(input_shapes[6], input_formats[6], input_dtypes[6]),
        ],
        "kwargs": {
            "k_rope_scale": None,
            "c_kv_scale": None,
            "k_rope_offset": None,
            "c_kv_offset": None,
            "v": None,
            "epsilon": 1e-5,
            "cache_mode": infer_cache_mode(input_shapes[5], input_shapes[4]),
            "is_output_kv": True,
        },
        "api": op.resolve_api(),
    }


def run_case(case):
    return case["api"](
        *case["inputs"],
        **case["kwargs"],
    )


def format_success(csv_path, row_index: int, _row: dict[str, str], case, result) -> str:
    k_cache, ckv_cache, k_rope, c_kv = result
    return (
        f"[OK] {csv_path}:{row_index} "
        f"cache_mode={case['kwargs']['cache_mode']} is_output_kv={case['kwargs']['is_output_kv']} "
        f"k_cache={tuple(k_cache.shape)} ckv_cache={tuple(ckv_cache.shape)} "
        f"k_rope={tuple(k_rope.shape)} c_kv={tuple(c_kv.shape)}"
    )


op = OpReplay(
    kernel_type="KvRmsNormRopeCache",
    api_path="torch_npu.npu_kv_rmsnorm_rope_cache",
    description=(
        "Run KvRmsNormRopeCache workload replay on Ascend NPU.\n"
        "The script reads KvRmsNormRopeCache.csv under the selected device\n"
        "and vllm_ascend version directory, reconstructs the recorded tensor\n"
        "inputs, infers cache_mode from the cache layout, then runs\n"
        "torch_npu.npu_kv_rmsnorm_rope_cache()."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/KvRmsNormRopeCache_run.py "
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
