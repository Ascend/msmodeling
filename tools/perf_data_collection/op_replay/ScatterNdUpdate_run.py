"""
Replay ScatterNdUpdate cases from the performance database on Ascend NPU.

Purpose:
  Read ScatterNdUpdate rows from
  profiling_database/data/{device}/vllm_ascend/{version}/ScatterNdUpdate.csv,
  rebuild the recorded tensor inputs, construct a legal indices tensor,
  then execute torch_npu.npu_scatter_nd_update() (aclnnScatterNdUpdate).

CSV layout (3 inputs, 1 output):
  Input[0]: data tensor   (total_slots, head_dim)   e.g. (226048, 128)  BF16
  Input[1]: indices tensor (num_tokens, 1)           e.g. (102, 1)      INT32
  Input[2]: updates tensor (num_tokens, head_dim)    e.g. (102, 128)    BF16
  Output[0]: updated data  (total_slots, head_dim)   same as Input[0]

Non-tensor args: None (in-place scatter update by indices).

microbench_api: torch_npu.npu_scatter_nd_update (maps to aclnnScatterNdUpdate)
"""

from __future__ import annotations

try:
    from .common import get_runtime_modules, parse_list_field, parse_shape
    from .replay_framework import OpReplay
except ImportError:
    from common import get_runtime_modules, parse_list_field, parse_shape
    from replay_framework import OpReplay


def build_indices_tensor(
    indices_shape: tuple[int, ...],
    data_shape: tuple[int, ...],
):
    """Build a legal indices tensor with values in [0, total_slots)."""
    runtime_torch, _ = get_runtime_modules()
    if len(indices_shape) != 2 or indices_shape[1] != 1:
        raise ValueError(f"indices must be (N, 1), got shape={indices_shape}")

    num_tokens = indices_shape[0]
    total_slots = data_shape[0]
    if num_tokens > total_slots:
        raise ValueError(
            f"num_tokens ({num_tokens}) exceeds total_slots ({total_slots})"
        )

    # Generate unique random indices within legal range
    permutation = runtime_torch.randperm(total_slots, dtype=runtime_torch.int64)[
        :num_tokens
    ]
    return permutation.to(dtype=runtime_torch.int32, device="npu").unsqueeze(1)


def build_case(row: dict[str, str]):
    inputs = op.build_inputs(row)
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    if len(inputs) != 3 or len(input_shapes) != 3:
        raise ValueError("ScatterNdUpdate expects exactly three inputs")

    # Replace indices with legal values
    return {
        "inputs": [
            inputs[0],
            build_indices_tensor(
                indices_shape=input_shapes[1],
                data_shape=input_shapes[0],
            ),
            inputs[2],
        ],
        "kwargs": {},
        "api": op.resolve_api(),
    }


def format_success(csv_path, row_index: int, row: dict[str, str], case, _result) -> str:
    data = case["inputs"][0]
    indices = case["inputs"][1]
    updates = case["inputs"][2]
    return (
        f"[OK] {csv_path}:{row_index} "
        f"data={tuple(data.shape)} indices={tuple(indices.shape)} "
        f"updates={tuple(updates.shape)} "
        f"dtypes={row['Input Data Types']}"
    )


op = OpReplay(
    kernel_type="ScatterNdUpdate",
    api_path="torch_npu.npu_scatter_nd_update",
    description=(
        "Run ScatterNdUpdate workload replay on Ascend NPU.\n"
        "Reads ScatterNdUpdate.csv under the selected device and\n"
        "vllm_ascend version directory, reconstructs input tensors,\n"
        "builds a legal indices tensor, then runs\n"
        "torch_npu.npu_scatter_nd_update(data, indices, updates).\n\n"
        "This operator performs DSA indexer cache updates in GLM5:\n"
        "it writes new token K vectors into the indexer cache at\n"
        "the positions specified by indices."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/ScatterNdUpdate_run.py "
        "--database-path tensor_cast/performance_model/profiling_database/"
        "data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/test",
    ],
    version_help="vLLM-Ascend version, e.g. 0.19.0.",
    build_case=build_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
