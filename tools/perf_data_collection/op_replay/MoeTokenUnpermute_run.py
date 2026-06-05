"""
Replay MoeTokenUnpermute cases from the performance database on Ascend NPU.

Purpose:
  Read MoeTokenUnpermute rows from the profiling database,
  rebuild input tensors from the recorded shapes, formats, and dtypes,
  then execute torch_npu.npu_moe_token_unpermute().

Usage:
  python tools/perf_data_collection/op_replay/MoeTokenUnpermute_run.py ^
    --device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.18.0
"""

from __future__ import annotations

try:
    from .common import build_input_tensor, init_runtime, normalize_dtype_name, parse_list_field, parse_shape_or_none
    from .replay_framework import OpReplay
except ImportError:
    from common import build_input_tensor, init_runtime, normalize_dtype_name, parse_list_field, parse_shape_or_none
    from replay_framework import OpReplay


def build_case(row: dict[str, str]):
    """CSV has 3 inputs; 3rd may be empty/DT_UNDEFINED → skip."""
    init_runtime()
    shapes = parse_list_field(row["Input Shapes"])
    dtypes = [normalize_dtype_name(d) for d in parse_list_field(row["Input Data Types"])]
    formats = parse_list_field(row["Input Formats"])

    tensors = []
    for i in range(len(shapes)):
        s = parse_shape_or_none(shapes[i])
        if s is None:
            continue
        dt = dtypes[i] if i < len(dtypes) else "DT_BF16"
        if dt in ("DT_UNDEFINED", "UNDEFINED"):
            continue
        fmt = formats[i] if i < len(formats) else "ND"
        tensors.append(build_input_tensor(shape=s, input_format=fmt, dtype_name=dt))

    if len(tensors) < 2:
        raise ValueError("MoeTokenUnpermute expects at least 2 inputs")

    api = op.resolve_api()
    return {"inputs": tensors, "kwargs": {}, "api": api}


def run_case(case):
    return case["api"](*case["inputs"])


op = OpReplay(
    kernel_type="MoeTokenUnpermute",
    api_path="torch_npu.npu_moe_token_unpermute",
    description=(
        "Run MoeTokenUnpermute workload replay on Ascend NPU.\n"
        "Reverses MoE token permutation, restoring tokens to their "
        "original batch positions."
    ),
    usage_examples=[
        "python tools/perf_data_collection/op_replay/MoeTokenUnpermute_run.py "
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
