"""
Replay DispatchFFNCombine rows from the perf database on Ascend NPU.

This script now follows the same replay contract as other operator scripts:
- rebuild tensors from DispatchFFNCombine.csv
- execute the operator on NPU
- let the outer `msprof + start_microbench.py` pipeline collect op_summary data

It keeps EP-related launch logic because this operator may require an EP-style
distributed environment, but it no longer owns profiler capture or writes an
extra output CSV by itself.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from typing import Any

try:
    from .common import (
        MICROBENCH_DURATION,
        build_host_tensor,
        build_standard_argparser,
        csv_has_complete_microbench,
        ensure_npu_available,
        get_replay_repeat_count,
        get_runtime_modules,
        get_target_data_dir,
        init_runtime,
        load_csv_rows,
        maybe_cast_internal_format,
        normalize_dtype_name,
        parse_shape_or_none,
        print_invalid_replay_summary,
        process_replay_csvs,
        resolve_runtime_dtype,
        split_metadata_field,
    )
except ImportError:
    from common import (
        MICROBENCH_DURATION,
        build_host_tensor,
        build_standard_argparser,
        csv_has_complete_microbench,
        ensure_npu_available,
        get_replay_repeat_count,
        get_runtime_modules,
        get_target_data_dir,
        init_runtime,
        load_csv_rows,
        maybe_cast_internal_format,
        normalize_dtype_name,
        parse_shape_or_none,
        print_invalid_replay_summary,
        process_replay_csvs,
        resolve_runtime_dtype,
        split_metadata_field,
    )


# ============================================================================
# DFC-specific constants
# ============================================================================
# Default repeat count for DFC benchmarking. The best timing is kept.
DEFAULT_DFC_REPEAT_COUNT = 20

DEFAULT_EP_SIZE = 16
EP_RANK: int = 0
EP_GROUP = None
HCOMM_INFO: str | None = None
MAX_OUTPUT_SIZE = 65536
EP_SIZE: int = DEFAULT_EP_SIZE
ENABLE_BALANCED: bool = True

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch_torchrun_and_wait(ep_size: int, args: list[str]) -> None:
    torchrun_cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={ep_size}",
        f"--master_port={find_free_port()}",
        __file__,
        *args,
    ]
    print(f"[Auto EP] Launching torchrun with {ep_size} ranks...")
    env = os.environ.copy()
    env["_DFC_AUTO_TORCHRUN"] = "1"
    subprocess.run(torchrun_cmd, env=env, check=True)


def init_ep_process_group(ep_size: int, ep_rank: int, master_addr: str, master_port: int):
    global EP_SIZE, EP_RANK, EP_GROUP, HCOMM_INFO

    runtime_torch, runtime_torch_npu = get_runtime_modules()
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import _get_default_group

    EP_SIZE = ep_size
    EP_RANK = ep_rank

    device_index = ep_rank % runtime_torch.npu.device_count()
    runtime_torch_npu.npu.set_device(device_index)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="hccl",
            rank=ep_rank,
            world_size=ep_size,
            init_method=f"tcp://{master_addr}:{master_port}",
        )

    default_pg = _get_default_group()
    EP_GROUP = default_pg
    if runtime_torch.__version__ > "2.0.1":
        backend = default_pg._get_backend(runtime_torch.device("npu"))
        HCOMM_INFO = backend.get_hccl_comm_name(ep_rank)
    else:
        HCOMM_INFO = default_pg.get_hccl_comm_name(ep_rank)
    return HCOMM_INFO


def get_default_hccl_group_name() -> str:
    global HCOMM_INFO

    if HCOMM_INFO is not None:
        return HCOMM_INFO

    if EP_SIZE > 1:
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return init_ep_process_group(world_size, rank, master_addr, master_port)

    runtime_torch, runtime_torch_npu = get_runtime_modules()
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import _get_default_group

    device_index = runtime_torch.npu.current_device()
    runtime_torch_npu.npu.set_device(device_index)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="hccl",
            rank=0,
            world_size=1,
            init_method=f"tcp://127.0.0.1:{find_free_port()}",
        )

    default_pg = _get_default_group()
    if runtime_torch.__version__ > "2.0.1":
        backend = default_pg._get_backend(runtime_torch.device("npu"))
        HCOMM_INFO = backend.get_hccl_comm_name(0)
    else:
        HCOMM_INFO = default_pg.get_hccl_comm_name(0)
    return HCOMM_INFO


def build_npu_tensor(shape: tuple[int, ...], dtype_name: str, tensor_format: str):
    dtype = resolve_runtime_dtype(dtype_name)
    tensor = build_host_tensor(shape, dtype).npu()
    return maybe_cast_internal_format(tensor, tensor_format)


def build_expert_idx_tensor(shape: tuple[int, ...], num_experts: int):
    runtime_torch, _ = get_runtime_modules()
    return runtime_torch.randint(0, num_experts, shape, dtype=runtime_torch.int32).npu()


def build_balanced_expert_idx_tensor(shape: tuple[int, ...], num_experts: int):
    runtime_torch, _ = get_runtime_modules()
    num_tokens, topk = shape
    total_slots = num_tokens * topk
    flat_ids = runtime_torch.arange(total_slots, dtype=runtime_torch.int32) % num_experts
    return flat_ids.reshape(num_tokens, topk).npu()


def build_uniform_probs_tensor(shape: tuple[int, ...], topk: int):
    runtime_torch, _ = get_runtime_modules()
    return runtime_torch.full(shape, 1.0 / topk, dtype=runtime_torch.float32).npu()


def build_scale_tensor(flattened_shape: tuple[int, ...], expected_shape: tuple[int, int], dtype_name: str):
    runtime_torch, _ = get_runtime_modules()
    dtype = resolve_runtime_dtype(dtype_name)
    if len(flattened_shape) == 2:
        if flattened_shape != expected_shape:
            raise ValueError(
                f"scale shape mismatch: actual={flattened_shape} expected={expected_shape}"
            )
        return runtime_torch.zeros(flattened_shape, dtype=dtype).npu()
    if len(flattened_shape) != 1:
        raise ValueError(f"scale tensor must be 1D or 2D, got {flattened_shape}")
    flat_size = flattened_shape[0]
    if flat_size != expected_shape[0] * expected_shape[1]:
        raise ValueError(
            f"flattened scale size mismatch: actual={flat_size} expected={expected_shape[0] * expected_shape[1]}"
        )
    return runtime_torch.zeros(expected_shape, dtype=dtype).reshape(-1).npu()


def build_output_tensor(shape: tuple[int, ...], dtype_name: str, tensor_format: str):
    dtype = resolve_runtime_dtype(dtype_name)
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"invalid output shape: {shape}")
    tensor = build_host_tensor(shape, dtype).npu()
    return maybe_cast_internal_format(tensor, tensor_format)


def build_row_case(row: dict[str, str], balanced: bool = True) -> dict[str, Any]:
    init_runtime()
    input_shapes = [parse_shape_or_none(item) for item in split_metadata_field(row["Input Shapes"])]
    input_dtypes = [normalize_dtype_name(item) for item in split_metadata_field(row["Input Data Types"])]
    input_formats = [item if item else "NULL" for item in split_metadata_field(row["Input Formats"])]
    output_shapes = [parse_shape_or_none(item) for item in split_metadata_field(row["Output Shapes"])]
    output_dtypes = [normalize_dtype_name(item) for item in split_metadata_field(row["Output Data Types"])]
    output_formats = [item if item else "NULL" for item in split_metadata_field(row["Output Formats"])]

    if not (len(input_shapes) == len(input_dtypes) == len(input_formats) == 7):
        raise ValueError(
            "DispatchFFNCombine expects seven input metadata slots, got "
            f"shapes={len(input_shapes)} dtypes={len(input_dtypes)} formats={len(input_formats)}"
        )
    if not (len(output_shapes) == len(output_dtypes) == len(output_formats) == 2):
        raise ValueError(
            "DispatchFFNCombine expects two output metadata slots, got "
            f"shapes={len(output_shapes)} dtypes={len(output_dtypes)} formats={len(output_formats)}"
        )
    if any(item is None for item in input_shapes + output_shapes):
        raise ValueError("DispatchFFNCombine metadata contains unexpected empty shape slots")

    x_shape, weight1_shape, weight2_shape, expert_idx_shape, scale1_shape, scale2_shape, probs_shape = input_shapes
    out_shape, expert_token_nums_shape = output_shapes

    if len(x_shape) != 2:
        raise ValueError(f"x must be 2D, got {x_shape}")
    if len(weight1_shape) != 3 or len(weight2_shape) != 3:
        raise ValueError(f"weight tensors must be 3D, got w1={weight1_shape} w2={weight2_shape}")
    if len(expert_idx_shape) != 2 or len(probs_shape) != 2:
        raise ValueError(f"expert_idx/probs must be 2D, got idx={expert_idx_shape} probs={probs_shape}")

    num_experts, hidden_size, inter_size = weight1_shape
    if weight2_shape[0] != num_experts:
        raise ValueError(f"expert count mismatch between weight1 and weight2: {weight1_shape} vs {weight2_shape}")
    if weight2_shape[2] != hidden_size:
        raise ValueError(f"hidden size mismatch between x/weight2: x={x_shape} w2={weight2_shape}")
    if x_shape[1] != hidden_size:
        raise ValueError(f"x hidden size mismatch: x={x_shape} weight1={weight1_shape}")
    if expert_idx_shape != probs_shape:
        raise ValueError(f"expert_idx/probs shape mismatch: idx={expert_idx_shape} probs={probs_shape}")
    if expert_idx_shape[0] != x_shape[0]:
        raise ValueError(f"token count mismatch between x and expert_idx: x={x_shape} idx={expert_idx_shape}")
    if out_shape != x_shape:
        raise ValueError(f"output shape must match x shape, got out={out_shape} x={x_shape}")
    if expert_token_nums_shape not in {(num_experts,), (1, num_experts)}:
        raise ValueError(
            f"expert_token_nums shape must be ({num_experts},) or (1, {num_experts}), got {expert_token_nums_shape}"
        )

    scale1_expected_shape = (num_experts, inter_size)
    scale2_expected_shape = (num_experts, hidden_size)

    x = build_npu_tensor(x_shape, input_dtypes[0], input_formats[0])
    weight1 = build_npu_tensor(weight1_shape, input_dtypes[1], input_formats[1])
    weight2 = build_npu_tensor(weight2_shape, input_dtypes[2], input_formats[2])

    ep_size_str = row.get("EP Size", "") or ""
    if ep_size_str.strip():
        try:
            expert_idx_num_experts = num_experts * int(ep_size_str.strip())
        except ValueError:
            expert_idx_num_experts = num_experts * EP_SIZE
    else:
        expert_idx_num_experts = num_experts * EP_SIZE

    topk = expert_idx_shape[1]
    if balanced:
        expert_idx = build_balanced_expert_idx_tensor(expert_idx_shape, expert_idx_num_experts)
        probs = build_uniform_probs_tensor(probs_shape, topk)
    else:
        expert_idx = build_expert_idx_tensor(expert_idx_shape, expert_idx_num_experts)
        probs = build_npu_tensor(probs_shape, input_dtypes[6], input_formats[6])

    scale1 = build_scale_tensor(scale1_shape, scale1_expected_shape, input_dtypes[4])
    scale2 = build_scale_tensor(scale2_shape, scale2_expected_shape, input_dtypes[5])
    out = build_output_tensor(out_shape, output_dtypes[0], output_formats[0])
    expert_token_nums = build_output_tensor(
        expert_token_nums_shape,
        output_dtypes[1],
        output_formats[1],
    )

    return {
        "x": x,
        "weight1_list": [weight1],
        "weight2_list": [weight2],
        "expert_idx": expert_idx,
        "scale1_list": [scale1],
        "scale2_list": [scale2],
        "probs": probs,
        "group": get_default_hccl_group_name(),
        "max_output_size": MAX_OUTPUT_SIZE,
        "out": out,
        "expert_token_nums": expert_token_nums,
        "expected_output_shapes": output_shapes,
        "weight_kind": input_dtypes[1],
        "num_experts": num_experts,
        "topk": topk,
    }


def build_argparser():
    parser = build_standard_argparser(
        description=(
            "Replay DispatchFFNCombine rows on Ascend NPU.\n"
            "EP mode: use --ep-size to control expert-parallel world size.\n"
            "EP_SIZE=1 runs in a single process; EP_SIZE>1 auto-launches torchrun.\n"
            "Profiling is owned by the outer start_microbench/msprof pipeline."
        ),
        usage_examples=[
            "py -3 tools/perf_data_collection/op_replay/DispatchFFNCombine_run.py "
            "--device ATLAS_800_A3_752T_128G_DIE --vllm-version 0.20.0 --ep-size 1",
            "py -3 tools/perf_data_collection/op_replay/DispatchFFNCombine_run.py "
            "--database-path tensor_cast/performance_model/profiling_database/data/"
            "ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.20.0_torch2.9.0_cann8.5 --ep-size 8",
        ],
        version_help="vLLM-Ascend version, e.g. 0.20.0.",
    )
    parser.add_argument(
        "--ep-size",
        type=int,
        default=DEFAULT_EP_SIZE,
        help=(
            f"EP size, equals world_size/rank count. "
            f"EP_SIZE=1: single-process. EP_SIZE>1: auto-launch torchrun. "
            f"Default: {DEFAULT_EP_SIZE}."
        ),
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=True,
        help="Use balanced expert distribution (round-robin). Default: True.",
    )
    parser.add_argument(
        "--no-balanced",
        action="store_false",
        dest="balanced",
        help="Use random expert distribution instead of balanced.",
    )
    return parser


def execute_dfc_op(case: dict[str, Any]) -> tuple:
    runtime_torch, _ = get_runtime_modules()

    try:
        out, expert_token_nums = runtime_torch.ops._C_ascend.dispatch_ffn_combine(
            x=case["x"],
            weight1=case["weight1_list"],
            weight2=case["weight2_list"],
            expert_idx=case["expert_idx"],
            scale1=case["scale1_list"],
            scale2=case["scale2_list"],
            probs=case["probs"],
            group=case["group"],
            max_output_size=case["max_output_size"],
            out=case["out"],
            expert_token_nums=case["expert_token_nums"],
        )
        return out, expert_token_nums, False
    except RuntimeError as exc:
        if "does not support opType [DispatchFFNCombine]" not in str(exc):
            raise

    out, expert_token_nums = runtime_torch.ops._C_ascend.dispatch_gmm_combine_decode(
        x=case["x"],
        expert_ids=case["expert_idx"],
        gmm1_permuted_weight=case["weight1_list"],
        gmm1_permuted_weight_scale=[tensor.to(runtime_torch.float32) for tensor in case["scale1_list"]],
        gmm2_weight=case["weight2_list"],
        gmm2_weight_scale=[tensor.to(runtime_torch.float32) for tensor in case["scale2_list"]],
        expert_scales=case["probs"],
        expert_smooth_scales=None,
        x_active_mask=None,
        group_ep=case["group"],
        ep_rank_size=EP_SIZE,
        ep_rank_id=EP_RANK,
        moe_expert_num=case["num_experts"],
        shared_expert_num=1,
        shared_expert_rank_num=0,
        quant_mode=0,
        global_bs=case["x"].shape[0],
    )
    if case["expert_token_nums"].ndim == 2 and expert_token_nums.ndim == 1:
        expert_token_nums = expert_token_nums.unsqueeze(0)
    expert_token_nums = expert_token_nums.to(case["expert_token_nums"].dtype)
    return out, expert_token_nums, True


def run_row(csv_path, row_index: int, row: dict[str, str], *, balanced: bool) -> None:
    runtime_torch, _ = get_runtime_modules()
    case = build_row_case(row, balanced=balanced)
    out, expert_token_nums, used_fallback = execute_dfc_op(case)
    runtime_torch.npu.synchronize()

    actual_shapes = [tuple(out.shape), tuple(expert_token_nums.shape)]
    expected_shapes = case["expected_output_shapes"]
    if actual_shapes[0] != expected_shapes[0]:
        raise ValueError(f"out shape mismatch: actual={actual_shapes[0]} expected={expected_shapes[0]}")
    if actual_shapes[1] != expected_shapes[1]:
        raise ValueError(
            f"expert_token_nums shape mismatch: actual={actual_shapes[1]} expected={expected_shapes[1]}"
        )

    if EP_RANK == 0:
        balance_tag = " balanced" if balanced else ""
        ep_tag = f" EP={EP_SIZE}" if EP_SIZE > 1 else ""
        api_name = "dispatch_gmm_combine_decode" if used_fallback else "dispatch_ffn_combine"
        print(
            f"[OK]{balance_tag}{ep_tag} {csv_path}:{row_index} "
            f"api={api_name} "
            f"x={tuple(case['x'].shape)} "
            f"w1={tuple(case['weight1_list'][0].shape)} "
            f"w2={tuple(case['weight2_list'][0].shape)} "
            f"topk={case['topk']} experts={case['num_experts']} "
            f"weight_kind={case['weight_kind']} "
            f"out={tuple(out.shape)} expert_token_nums={tuple(expert_token_nums.shape)}"
        )


def main() -> None:
    global EP_SIZE, EP_RANK, ENABLE_BALANCED

    args = build_argparser().parse_args()
    
    # DFC uses its own default repeat count.
    if args.repeat_count is None:
        args.repeat_count = DEFAULT_DFC_REPEAT_COUNT
    
    repeat_count = get_replay_repeat_count(args.repeat_count)
    EP_SIZE = args.ep_size
    ENABLE_BALANCED = args.balanced

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    is_auto_torchrun = os.environ.get("_DFC_AUTO_TORCHRUN", "0") == "1"

    target_data_dir = get_target_data_dir(
        device=args.device,
        vllm_ascend_version=args.vllm_version,
        database_path=args.database_path,
        torch_version=args.torch_version,
        cann_version=args.cann_version,
    )
    csv_paths = sorted(target_data_dir.rglob("DispatchFFNCombine.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No DispatchFFNCombine.csv found under {target_data_dir}")

    if args.update_mode == "missing-only":
        all_csvs_have_microbench = True
        total_skipped_rows = 0
        for csv_path in csv_paths:
            _, rows = load_csv_rows(csv_path)
            if not csv_has_complete_microbench(rows):
                all_csvs_have_microbench = False
                break
            total_skipped_rows += len(rows)
        if all_csvs_have_microbench:
            print(
                f"[SKIP] DispatchFFNCombine all {total_skipped_rows} row(s) already have "
                "usable Average/Profiling durations."
            )
            print(
                f"Processed 0 DispatchFFNCombine rows from {len(csv_paths)} csv file(s) "
                f"under {target_data_dir}."
            )
            print(
                f"[SUMMARY] DispatchFFNCombine: skipped {total_skipped_rows} row(s) due to missing-only mode."
            )
            print_invalid_replay_summary([], label="DispatchFFNCombine")
            return

    if EP_SIZE > 1 and env_world_size == 1 and not is_auto_torchrun:
        cli_args = []
        if args.database_path is not None:
            cli_args.extend(["--database-path", str(args.database_path)])
        else:
            cli_args.extend(["--device", args.device])
            if args.vllm_version:
                cli_args.extend(["--vllm-version", args.vllm_version])
            if args.torch_version:
                cli_args.extend(["--torch-version", args.torch_version])
            if args.cann_version:
                cli_args.extend(["--cann-version", args.cann_version])
        cli_args.extend(["--repeat-count", str(repeat_count)])
        cli_args.extend(["--update-mode", args.update_mode])
        cli_args.extend(["--ep-size", str(EP_SIZE)])
        if not ENABLE_BALANCED:
            cli_args.append("--no-balanced")
        launch_torchrun_and_wait(EP_SIZE, cli_args)
        return

    ensure_npu_available()

    if env_world_size > 1:
        EP_SIZE = env_world_size
        EP_RANK = env_rank
        get_default_hccl_group_name()
        if EP_RANK == 0:
            print(f"[EP Mode] EP_SIZE={EP_SIZE}, EP_RANK={EP_RANK}")
    else:
        print(f"[Single-process Mode] EP_SIZE={EP_SIZE} (no EP communication)")

    def barrier() -> None:
        if EP_SIZE > 1:
            import torch.distributed as dist

            dist.barrier()

    total_rows, invalid_rows, _, skipped_rows = process_replay_csvs(
        kernel_type="DispatchFFNCombine",
        csv_paths=csv_paths,
        repeat_count=repeat_count,
        run_row_fn=lambda csv_path, row_index, row: run_row(
            csv_path,
            row_index,
            row,
            balanced=ENABLE_BALANCED,
        ),
        update_mode=args.update_mode,
        on_row_finally=barrier,
        can_write_cleanup=lambda: EP_RANK == 0,
        on_cleanup_written=barrier,
    )

    if EP_RANK == 0:
        print(
            f"Processed {total_rows} DispatchFFNCombine rows "
            f"from {len(csv_paths)} csv file(s) under {target_data_dir}."
        )
        if args.update_mode == "missing-only":
            print(f"[SUMMARY] DispatchFFNCombine: skipped {skipped_rows} row(s) due to missing-only mode.")
        print_invalid_replay_summary(invalid_rows, label="DispatchFFNCombine")

    if EP_SIZE > 1:
        import torch.distributed as dist

        dist.barrier()


if __name__ == "__main__":
    main()
