"""
Replay split_qkv_rmsnorm_rope_kernel cases from the performance database on Ascend NPU.

Purpose:
  Read split_qkv_rmsnorm_rope_kernel rows from
  profiling_database/data/{device}/vllm_ascend/{version}/split_qkv_rmsnorm_rope_kernel.csv,
  rebuild the recorded tensor inputs, infer the required scalar arguments,
  then execute torch.ops.vllm.qkv_rmsnorm_rope().
"""

from __future__ import annotations

from math import gcd
import os
from pathlib import Path
import sys

try:
    from .common import build_input_tensor, get_runtime_modules, parse_list_field, parse_shape
    from .replay_framework import OpReplay
except ImportError:
    from common import build_input_tensor, get_runtime_modules, parse_list_field, parse_shape
    from replay_framework import OpReplay


DEFAULT_EPS = 1e-6
VLLM_ASCEND_REPO = Path(__file__).resolve().parents[4] / "vllm-ascend"


def ensure_vllm_ascend_available() -> None:
    try:
        import vllm_ascend.ops.register_custom_ops  # noqa: F401
        from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    except ImportError:
        candidate_paths = []
        env_path = os.environ.get("VLLM_ASCEND_PATH")
        if env_path:
            candidate_paths.append(Path(env_path))
        candidate_paths.append(VLLM_ASCEND_REPO)

        for candidate in candidate_paths:
            if not candidate.exists():
                continue
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            try:
                import vllm_ascend.ops.register_custom_ops  # noqa: F401
                from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
                break
            except ImportError:
                continue
        else:
            searched_paths = [str(path) for path in candidate_paths]
            raise ImportError(
                "Unable to import vllm_ascend. Either install vllm-ascend into the "
                "current Python environment or set VLLM_ASCEND_PATH to the repo root. "
                f"Searched: {searched_paths}"
            )

    init_device_properties_triton()


def infer_head_dim(q_hidden_size: int, kv_hidden_size: int, rope_dim: int) -> int:
    if q_hidden_size <= 0 or kv_hidden_size <= 0 or rope_dim <= 0:
        raise ValueError(
            "q_hidden_size, kv_hidden_size, and rope_dim must be positive, got "
            f"q_hidden_size={q_hidden_size}, kv_hidden_size={kv_hidden_size}, {rope_dim}"
        )

    common_divisor = gcd(q_hidden_size, kv_hidden_size)
    valid_divisors = []
    for divisor in range(1, common_divisor + 1):
        if common_divisor % divisor != 0:
            continue
        if divisor < rope_dim:
            continue
        if q_hidden_size % divisor == 0 and kv_hidden_size % divisor == 0:
            valid_divisors.append(divisor)

    if not valid_divisors:
        raise ValueError(
            "Unable to infer head_dim from q_hidden_size / kv_hidden_size / rope_dim: "
            f"{q_hidden_size}, {kv_hidden_size}, {rope_dim}"
        )

    if rope_dim in valid_divisors:
        return rope_dim
    return min(valid_divisors)


def build_positions_tensor(shape: tuple[int, ...], max_position_embeddings: int):
    runtime_torch, _ = get_runtime_modules()
    if len(shape) != 1:
        raise ValueError(f"positions must be 1D, got shape={shape}")
    return runtime_torch.randint(
        low=0,
        high=max(1, max_position_embeddings),
        size=shape,
        dtype=runtime_torch.int64,
        device="npu",
    )


def build_weight_tensor(length: int, dtype_name: str):
    runtime_torch, _ = get_runtime_modules()
    dtype = {
        "DT_FLOAT": runtime_torch.float32,
        "DT_FLOAT16": runtime_torch.float16,
        "DT_BF16": runtime_torch.bfloat16,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported weight dtype: {dtype_name}")
    return runtime_torch.randn((length,), dtype=dtype, device="npu")


def build_case(row: dict[str, str]):
    input_shapes = [parse_shape(item) for item in parse_list_field(row["Input Shapes"])]
    input_formats = parse_list_field(row["Input Formats"])
    input_dtypes = parse_list_field(row["Input Data Types"])
    output_shapes = [parse_shape(item) for item in parse_list_field(row["Output Shapes"])]

    if len(output_shapes) not in (2, 3):
        raise ValueError("split_qkv_rmsnorm_rope_kernel expects two legacy outputs or three recorded outputs")
    if len(input_shapes) not in (2, 3):
        raise ValueError("split_qkv_rmsnorm_rope_kernel expects two profiled inputs or three replay inputs")
    if not (len(input_shapes) == len(input_formats) == len(input_dtypes)):
        raise ValueError(
            "split_qkv_rmsnorm_rope_kernel input metadata length mismatch: "
            f"shapes={len(input_shapes)} formats={len(input_formats)} dtypes={len(input_dtypes)}"
        )

    q_hidden_size = output_shapes[0][-1]
    kv_hidden_size = output_shapes[1][-1]
    tokens = input_shapes[0][0]
    if len(input_shapes) == 2:
        rope_dim = input_shapes[1][-1]
        cos_sin_cache_shape = (max(tokens, 2048), rope_dim)
        positions_shape = (tokens,)
    else:
        rope_dim = input_shapes[1][-1]
        cos_sin_cache_shape = input_shapes[1]
        positions_shape = input_shapes[2]
    head_dim = infer_head_dim(q_hidden_size, kv_hidden_size, rope_dim)

    return {
        "inputs": [],
        "kwargs": {
            "input": build_input_tensor(input_shapes[0], input_formats[0], input_dtypes[0]),
            "cos_sin_cache": build_input_tensor(cos_sin_cache_shape, input_formats[1], input_dtypes[1]),
            "positions": build_positions_tensor(positions_shape, cos_sin_cache_shape[0]),
            "q_weight": build_weight_tensor(head_dim, input_dtypes[0]),
            "k_weight": build_weight_tensor(head_dim, input_dtypes[0]),
            "q_hidden_size": q_hidden_size,
            "kv_hidden_size": kv_hidden_size,
            "head_dim": head_dim,
            "eps": DEFAULT_EPS,
        },
        "api": op.resolve_api(),
    }


def run_case(case):
    return case["api"](**case["kwargs"])


def format_success(csv_path, row_index: int, row: dict[str, str], case, result) -> str:
    q_out, k_out, v_out = result
    return (
        f"[OK] {csv_path}:{row_index} "
        f"shapes={row['Input Shapes']} formats={row['Input Formats']} "
        f"dtypes={row['Input Data Types']} q={tuple(q_out.shape)} "
        f"k={tuple(k_out.shape)} v={tuple(v_out.shape)} "
        f"head_dim={case['kwargs']['head_dim']} eps={DEFAULT_EPS}"
    )


op = OpReplay(
    kernel_type="split_qkv_rmsnorm_rope_kernel",
    api_path="torch.ops.vllm.qkv_rmsnorm_rope",
    description=(
        "Run split_qkv_rmsnorm_rope_kernel workload replay on Ascend NPU.\n"
        "The script reads split_qkv_rmsnorm_rope_kernel.csv under the\n"
        "selected device and vllm_ascend version directory, reconstructs\n"
        "the recorded tensor inputs, infers q_hidden_size / kv_hidden_size /\n"
        "head_dim from the CSV metadata, then runs the vLLM-Ascend Triton\n"
        "custom op torch.ops.vllm.qkv_rmsnorm_rope()."
    ),
    usage_examples=[
        "py -3 tools/perf_data_collection/op_replay/split_qkv_rmsnorm_rope_kernel_run.py "
        "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.13.0",
    ],
    version_help="vLLM-Ascend version, e.g. 0.13.0.",
    prepare=ensure_vllm_ascend_available,
    build_case=build_case,
    run_case=run_case,
    format_success=format_success,
)


def main() -> None:
    op.main()


if __name__ == "__main__":
    main()
