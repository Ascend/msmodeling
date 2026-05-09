"""
Replay ReshapeAndCacheNdKernel cases from the performance database on Ascend NPU.

Purpose:
  Read ReshapeAndCacheNdKernel rows from
  profiling_database/data/{device}/vllm_ascend/{version}/ReshapeAndCacheNdKernel.csv,
  rebuild the recorded tensor inputs, construct a legal slot_mapping tensor,
  then execute torch_npu._npu_reshape_and_cache().
"""

from __future__ import annotations

try:
    from .common import get_runtime_modules, parse_list_field, parse_shape
    from .replay_framework import OpReplay
except ImportError:
    from common import get_runtime_modules, parse_list_field, parse_shape
    from replay_framework import OpReplay


def build_slot_mapping_tensor(
    slot_mapping_shape: tuple[int, ...],
    key_cache_shape: tuple[int, ...],
):
    runtime_torch, _ = get_runtime_modules()
    if len(slot_mapping_shape) != 1:
        raise ValueError(f"slot_mapping must be 1D, got shape={slot_mapping_shape}")
    if len(key_cache_shape) < 2:
        raise ValueError(f"key_cache rank must be >= 2, got shape={key_cache_shape}")

    token_count = slot_mapping_shape[0]
    total_slots = key_cache_shape[0] * key_cache_shape[1]
    if token_count > total_slots:
        raise ValueError(
            "slot_mapping token count exceeds cache capacity: "
            f"tokens={token_count}, total_slots={total_slots}, key_cache_shape={key_cache_shape}"
        )

    permutation = runtime_torch.randperm(total_slots, dtype=runtime_torch.int64)[:token_count]
    return permutation.to(dtype=runtime_torch.int32, device="npu")


def build_case(row: dict[str, str]):
    inputs = op.build_inputs(row)
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    if len(inputs) != 5 or len(input_shapes) != 5:
        raise ValueError("ReshapeAndCacheNdKernel expects exactly five inputs")

    return {
        "inputs": inputs[:4] + [
            build_slot_mapping_tensor(
                slot_mapping_shape=input_shapes[4],
                key_cache_shape=input_shapes[2],
            )
        ],
        "kwargs": {},
        "api": op.resolve_api(),
    }


def format_success(csv_path, row_index: int, row: dict[str, str], case, _result) -> str:
    key_cache = case["inputs"][2]
    value_cache = case["inputs"][3]
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} "
        f"key_cache={tuple(key_cache.shape)} value_cache={tuple(value_cache.shape)}"
    )


op = OpReplay(
    kernel_type="ReshapeAndCacheNdKernel",
    api_path="torch_npu._npu_reshape_and_cache",
    description=(
        "Run ReshapeAndCacheNdKernel workload replay on Ascend NPU.\n"
        "The script reads ReshapeAndCacheNdKernel.csv under the selected\n"
        "device and vllm_ascend version directory, reconstructs input\n"
        "tensors from Input Shapes / Input Formats / Input Data Types,\n"
        "builds a legal slot_mapping tensor, then runs\n"
        "torch_npu._npu_reshape_and_cache()."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/ReshapeAndCacheNdKernel_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.13.0.",
    build_case=build_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
