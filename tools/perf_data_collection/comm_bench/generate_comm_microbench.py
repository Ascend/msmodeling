"""Run HCCL communication-operator microbenchmarks and collect CSV data.

Benchmarks torch.distributed communication ops for HCCL profiling:
- all_reduce, all_gather, reduce_scatter, all_to_all
- topology_tier is derived from rank + group via CommGrid logic (not manually specified)
- Outputs CSV in the format required by ProfilingDataSource

CSV output format:
    message_bytes,num_devices,dtype,topology_tier,Duration(us),bandwidth_gbps

topology_tier semantics (mirrors CommAnalyticModel._get_topology_idx_for_group):
    Determined by the outermost grid dimension where ranks in the group differ.
    For ATLAS_800_A3 with grid shape [48, 8, 2] (48 pods x 8 nodes x 2 dies):
        tier 0 = inter_pod  (ranks span multiple pods, stride=16)
        tier 1 = intra_pod  (ranks within one pod, span multiple nodes, stride=2)
        tier 2 = die_level  (ranks within one node, 2 dies, stride=1)
        e.g. TP=16 uses ranks 0..15 (pod0, all 8 nodes x 2 dies) -> tier=1

    The group_ranks argument controls which ranks participate, which determines
    the tier automatically. Use --grid-shape to match your hardware topology.

Usage examples:
    # Run all ops + all tiers in ONE torchrun session (recommended)
    torchrun --nproc_per_node=16 generate_comm_microbench.py \\
        --do-run --output-dir ./hccl_data \\
        --ops all_reduce all_gather reduce_scatter all_to_all \\
        --grid-shape 48 8 2 --num-devices 16 2

    # Run with event mode (fast, hcom_kernel only)
    torchrun --nproc_per_node=16 generate_comm_microbench.py \\
        --do-run --bench-mode event --output-dir ./hccl_data \\
        --ops all_reduce all_gather --grid-shape 48 8 2
"""

import argparse
import csv
import glob
import math
import os
import shutil
import statistics
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from textwrap import dedent
from typing import Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cli.logo import print_logo

WARMUP_ITERS = 20
BENCH_ITERS = 100

# Profiler bench mode constants
PROFILER_WARMUP_ITERS = 5   # profiler-internal warmup steps (separate from op warmup)
PROFILER_ACTIVE_ITERS = 10  # active profiling steps -> 10 Duration samples, take median
PROFILER_WAIT_ITERS = 0
PROFILER_ACTIVE_ITERS_LARGE = 1  # single active iter per session to minimise profiler ring-buffer pressure
PROFILER_LARGE_MSG_SESSIONS = 10  # repeat separate sessions and take median (compensates active=1)
PROFILER_LARGE_MSG_THRESHOLD = 524288  # 512KB: above this, use per-msg separate sessions

_BENCH_MODES = ["event", "kernel"]

_COMM_OPS = ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"]

# Fixed dispatch overhead corrections (us) applied to bench results before writing CSV.
#
# Background:
#   In vLLM production, each HCCL collective call passes through a dispatch chain:
#     scheduler -> c10d wrapper -> HCCL group lookup -> stream sync
#   This chain adds a fixed per-call latency that is independent of message size.
#   The bench runs ops in isolation (no scheduler, no c10d wrapper), so this overhead
#   is absent from raw bench measurements.  Adding it back aligns bench CSV values
#   with what ProfilingDataSource will observe when predicting production latency.
#
# Measurement methodology:
#   Overhead = median(vLLM step_trace Communication) - median(bench kernel Duration)
#   measured at small message sizes where the overhead fraction is largest.
#   Hardware: ATLAS_800_A3_752T_128G_DIE
#   vLLM version: v0.18.0 (vllm-ascend)
#   CANN version: 8.5
#
# Coverage:
#   Only (op, nd) combinations that were explicitly measured are listed.
#   Missing entries default to 0.0 (no correction applied).
#   all_to_all and nd=4/nd=2 entries are absent -- not yet measured.
#
# Maintenance:
#   These values are coupled to the vLLM dispatch implementation and CANN driver.
#   Re-measure after: vLLM version upgrades, CANN driver upgrades, or hardware changes.
#   The overhead is small (1.2~14.6 us) relative to large-message latency but
#   significant for small messages (e.g. 14.6 us on a 50 us all_gather is ~29%).
#
# Key: (op_type, num_devices) -> overhead_us
_DISPATCH_OVERHEAD: Dict[Tuple[str, int], float] = {
    ("all_reduce", 16): 7.7,
    ("all_gather", 16): 14.6,
    ("all_gather", 8): 1.2,
    ("reduce_scatter", 16): 14.6,
    ("reduce_scatter", 8): 2.0,
}

# Maps op_type -> kernel_details.csv Type prefix for hcom_* kernels
_OP_TO_KERNEL_TYPE = {
    "all_reduce": "hcom_allReduce_",
    "all_gather": "hcom_allGather_",
    "reduce_scatter": "hcom_reduceScatter_",
    "all_to_all": "hcom_alltoallv_",
}

# Maps op_type -> canonical CSV filename expected by ProfilingDataSource / op_mapping.yaml
_OP_TO_CSV_FILENAME = {
    "all_reduce": "hcom_allReduce_.csv",
    "all_gather": "hcom_allGather_.csv",
    "reduce_scatter": "hcom_reduceScatter_.csv",
    "all_to_all": "hcom_alltoallv_.csv",  # kernel_type in op_mapping.yaml is hcom_alltoallv_
}

# message_bytes grid: 1KB ~ 512MB, powers of 2 (dense grid for interpolation)
_DEFAULT_BYTES_GRID = [
    1024,        # 1 KB
    2048,        # 2 KB
    4096,        # 4 KB
    8192,        # 8 KB
    16384,       # 16 KB
    32768,       # 32 KB
    65536,       # 64 KB
    131072,      # 128 KB
    262144,      # 256 KB
    524288,      # 512 KB
    1048576,     # 1 MB
    2097152,     # 2 MB
    4194304,     # 4 MB
    8388608,     # 8 MB
    16777216,    # 16 MB
    33554432,    # 32 MB
    67108864,    # 64 MB
    134217728,   # 128 MB
    268435456,   # 256 MB
    536870912,   # 512 MB
]

_DTYPE_ELEM_SIZE = {
    "torch.bfloat16": 2,
    "torch.float16": 2,
    "torch.float32": 4,
    "torch.int8": 1,
}

_DTYPE_TO_CSV = {
    "torch.bfloat16": "DT_BF16",
    "torch.float16": "DT_FP16",
    "torch.float32": "DT_FLOAT",
    "torch.int8": "DT_INT8",
}

# CSV columns per S4.7
_CSV_COLUMNS = ["message_bytes", "num_devices", "dtype", "topology_tier", "Duration(us)", "bandwidth_gbps"]


# ============================================================================
# Topology tier resolution (mirrors CommAnalyticModel._get_topology_idx_for_group)
# ============================================================================

def _rank_to_coord(rank: int, grid_shape: List[int]) -> List[int]:
    coord = []
    temp = rank
    for dim_size in reversed(grid_shape):
        coord.insert(0, temp % dim_size)
        temp //= dim_size
    return coord


def resolve_topology_tier(group_ranks: List[int], grid_shape: List[int]) -> int:
    """Determine topology_tier for a group, matching CommAnalyticModel logic.

    Finds the outermost grid dimension where ranks differ, then returns the
    largest start_dim <= diff_dim (most specific topology that covers the span).

    For ATLAS_800_A3 grid_shape=[pods, nodes, dies]:
        All ranks same node  -> diff_dim=2 -> tier=2 (SIO / die-level)
        Ranks span nodes     -> diff_dim=1 -> tier=1 (1-level CLOS / intra-pod)
        Ranks span pods      -> diff_dim=0 -> tier=0 (2-level CLOS / inter-pod)
    """
    ndim = len(grid_shape)
    coords = [_rank_to_coord(r, grid_shape) for r in group_ranks]

    diff_dim = -1
    for dim_idx in range(ndim):
        first = coords[0][dim_idx]
        if any(c[dim_idx] != first for c in coords[1:]):
            diff_dim = dim_idx
            break

    if diff_dim == -1:
        return ndim - 1  # all same rank (shouldn't happen), use fastest

    # Most specific topology: largest start_dim <= diff_dim
    for start_dim in range(ndim - 1, -1, -1):
        if start_dim <= diff_dim:
            return start_dim

    return 0


def build_group_for_tier(
    rank: int, num_devices: int, topology_tier: int, grid_shape: List[int]
) -> List[int]:
    """Build a contiguous group of num_devices ranks at the given topology_tier.

    The group is anchored to rank's position in the grid: all ranks in the
    group share the same coordinates in dimensions > topology_tier, and span
    contiguously within the tier dimension.

    Example (grid_shape=[3,8,2], rank=5, num_devices=16, tier=1):
        rank 5 coord = [0, 2, 1]
        tier=1 means we span dims [1,2] -> group size per pod = 8*2=16
        group = ranks 0..15 (pod 0, all nodes, all dies)
    """
    ndim = len(grid_shape)
    coord = _rank_to_coord(rank, grid_shape)

    # Compute the stride and size for each dimension
    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * grid_shape[i + 1]

    # The group spans dims [topology_tier .. ndim-1]
    # Fix the prefix (dims 0 .. topology_tier-1) to rank's own coordinates
    # and enumerate all combinations within the span
    span_dims = list(range(topology_tier, ndim))
    span_sizes = [grid_shape[d] for d in span_dims]
    total_in_span = math.prod(span_sizes)

    if num_devices > total_in_span:
        raise ValueError(
            f"num_devices={num_devices} exceeds span size {total_in_span} "
            f"for tier={topology_tier}, grid_shape={grid_shape}"
        )

    # Base rank: fix prefix dims, set span dims to 0
    base_rank = sum(coord[d] * strides[d] for d in range(topology_tier))

    # Enumerate num_devices consecutive ranks within the span
    group = [base_rank + i for i in range(num_devices)]
    return group


# Direct run mode (--do-run)
# ============================================================================

def _build_run_op(
    op_type: str, message_bytes: int, dtype_str: str, device: str, group, group_ranks: List[int],
) -> Callable:
    """Build a run_op closure for the given comm op and message size."""
    import torch
    import torch.distributed as dist

    dtype = getattr(torch, dtype_str.replace("torch.", ""))
    elem_size = _DTYPE_ELEM_SIZE.get(dtype_str, 2)
    num_elements = message_bytes // elem_size
    num_devices = len(group_ranks)

    if op_type == "all_reduce":
        tensor = torch.randn(num_elements, dtype=dtype, device=device)
        def run_op(): dist.all_reduce(tensor, group=group)
    elif op_type == "all_gather":
        # Use tensor-based API so kernel_details records hcom_allGather_
        local_tensor = torch.randn(num_elements, dtype=dtype, device=device)
        output_tensor = torch.empty(num_elements * num_devices, dtype=dtype, device=device)
        def run_op(): dist.all_gather_into_tensor(output_tensor, local_tensor, group=group)
    elif op_type == "reduce_scatter":
        # Use tensor-based API so kernel_details records hcom_reduceScatter_
        input_tensor = torch.randn(num_elements * num_devices, dtype=dtype, device=device)
        output_tensor = torch.empty(num_elements, dtype=dtype, device=device)
        def run_op(): dist.reduce_scatter_tensor(output_tensor, input_tensor, group=group)
    elif op_type == "all_to_all":
        per_rank = max(1, num_elements // num_devices)
        input_list = [torch.randn(per_rank, dtype=dtype, device=device) for _ in group_ranks]
        output_list = [torch.empty(per_rank, dtype=dtype, device=device) for _ in group_ranks]
        def run_op(): dist.all_to_all(output_list, input_list, group=group)
    else:
        raise ValueError(f"Unknown op_type: {op_type}")
    return run_op


def run_benchmark(
    op_type: str,
    message_bytes: int,
    group_ranks: List[int],
    topology_tier: int,
    dtype_str: str,
    output_csv: Optional[str],
    group=None,
    bench_mode: str = "kernel",
) -> Optional[dict]:
    """Run a single benchmark directly in the current process.

    Args:
        group: pre-created dist.ProcessGroup. If None, creates one internally
               (only safe when called once per group_ranks combination).
        bench_mode: "kernel" (profiler -> kernel_details hcom_* Duration, aligns Communication),
                    "event" (per-iteration NPU Event timing, hcom_kernel only).

    Note:
        In kernel mode, only the first rank in group_ranks runs the profiler.
        Other ranks use event mode to participate in collective communication
        without starting their own profiler (avoids resource contention and /tmp bloat).
    """
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        print("ERROR: torch not available", file=sys.stderr)
        return None

    try:
        import torch_npu  # noqa: F401
        is_npu = True
    except ImportError:
        is_npu = False

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # Bind each rank to its own NPU device (critical for HCCL init)
    if is_npu:
        torch.npu.set_device(local_rank)
        device = f"npu:{local_rank}"
    else:
        device = "cpu"

    if group is None:
        group = dist.new_group(ranks=group_ranks)
    if rank not in group_ranks:
        return None

    dtype = getattr(torch, dtype_str.replace("torch.", ""))
    elem_size = _DTYPE_ELEM_SIZE.get(dtype_str, 2)
    num_elements = message_bytes // elem_size
    num_devices = len(group_ranks)

    if op_type == "all_reduce":
        tensor = torch.randn(num_elements, dtype=dtype, device=device)
        def run_op(): dist.all_reduce(tensor, group=group)
    elif op_type == "all_gather":
        local_tensor = torch.randn(num_elements, dtype=dtype, device=device)
        output_tensor = torch.empty(num_elements * num_devices, dtype=dtype, device=device)
        def run_op(): dist.all_gather_into_tensor(output_tensor, local_tensor, group=group)
    elif op_type == "reduce_scatter":
        input_tensor = torch.randn(num_elements * num_devices, dtype=dtype, device=device)
        output_tensor = torch.empty(num_elements, dtype=dtype, device=device)
        def run_op(): dist.reduce_scatter_tensor(output_tensor, input_tensor, group=group)
    elif op_type == "all_to_all":
        per_rank = max(1, num_elements // num_devices)
        input_list = [torch.randn(per_rank, dtype=dtype, device=device) for _ in group_ranks]
        output_list = [torch.empty(per_rank, dtype=dtype, device=device) for _ in group_ranks]
        def run_op(): dist.all_to_all(output_list, input_list, group=group)
    else:
        raise ValueError(f"Unknown op_type: {op_type}")

    # ---- Measurement: dispatch by bench_mode ----
    # In kernel mode, only the first rank in the group runs the profiler to
    # avoid resource contention (multiple profilers writing /tmp simultaneously).
    # All ranks must execute the same number of run_op() calls to avoid hangs.
    if bench_mode == "kernel" and is_npu:
        is_leader = (rank == group_ranks[0])
        duration_us = _run_bench_kernel(run_op, op_type, is_npu, is_leader=is_leader)
        if duration_us is None:
            return None  # follower ranks don't report results
    else:
        # event mode (or non-NPU fallback)
        duration_us = _run_bench_event(run_op, is_npu)

    bandwidth_gbps = message_bytes / (duration_us * 1e-6) / 1e9

    result = {
        "message_bytes": message_bytes,
        "num_devices": num_devices,
        "dtype": _DTYPE_TO_CSV.get(dtype_str, "DT_BF16"),
        "topology_tier": topology_tier,
        "Duration(us)": round(duration_us, 2),
        "bandwidth_gbps": round(bandwidth_gbps, 2),
    }

    if rank == group_ranks[0]:
        print(
            f"op={op_type}  bytes={message_bytes}  devices={num_devices}"
            f"  tier={topology_tier}  rank={rank}  group={group_ranks}"
            f"  duration={duration_us:.2f}us  bw={bandwidth_gbps:.2f}GB/s"
        )
        if output_csv:
            _append_csv(output_csv, result, op_type=op_type)

    return result


def _apply_dispatch_overhead(row: dict, op_type: str) -> dict:
    """Return a new row with dispatch overhead added if applicable.

    Overhead is applied uniformly to all message sizes. In the unified kernel
    mode, all data comes from the same profiler source (kernel_details), so
    there is no alternating/kernel source distinction. The overhead values
    are small (1.2-14.6us) and represent fixed dispatch latency that is
    independent of message size.
    """
    nd = row.get("num_devices", 0)
    overhead = _DISPATCH_OVERHEAD.get((op_type, nd), 0.0)
    if overhead <= 0:
        return row
    r = dict(row)
    r["Duration(us)"] = round(r["Duration(us)"] + overhead, 2)
    msg_bytes = r["message_bytes"]
    dur_us = r["Duration(us)"]
    r["bandwidth_gbps"] = round(msg_bytes / (dur_us * 1e-6) / 1e9, 2) if dur_us > 0 else 0.0
    return r


def _append_csv(path: str, row: dict, op_type: str = "") -> None:
    """Append a result row to CSV, applying dispatch overhead if op_type given."""
    if op_type:
        row = _apply_dispatch_overhead(row, op_type)
    p = Path(path)
    write_header = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ============================================================================
# Bench helpers: warmup, profiler loop, batched profiler session
# ============================================================================

def _warmup_and_sync(run_op, is_npu: bool) -> None:
    """Shared warmup: run WARMUP_ITERS iterations then sync."""
    import torch

    for _ in range(WARMUP_ITERS):
        run_op()
    if is_npu:
        torch.npu.synchronize()


def _profiler_loop_steps(run_op, is_npu: bool) -> None:
    """Execute the profiler-mode iteration pattern (total_steps with per-step sync).

    Used by both the profiler leader (inside profiler context) and followers
    (without profiler) to keep collective communication in lockstep.
    """
    import torch

    total_steps = PROFILER_WAIT_ITERS + PROFILER_WARMUP_ITERS + PROFILER_ACTIVE_ITERS
    for _ in range(total_steps):
        run_op()
        if is_npu:
            torch.npu.synchronize()


def _active_iters_for_msg(msg_bytes: int) -> int:
    """Return profiler active iterations based on message size.

    Large messages (>=512KB) use active=1 per session to minimise profiler
    ring-buffer pressure.  The caller compensates by running multiple
    separate sessions (PROFILER_LARGE_MSG_SESSIONS) and taking the median.
    """
    if msg_bytes >= PROFILER_LARGE_MSG_THRESHOLD:
        return PROFILER_ACTIVE_ITERS_LARGE
    return PROFILER_ACTIVE_ITERS


def _run_bench_profiler_batch(
    op_type: str,
    msg_bytes_list: List[int],
    dtype_str: str,
    device: str,
    group,
    group_ranks: List[int],
    is_npu: bool,
    is_leader: bool,
    parse_fn: Optional[Callable[[str, str], List[float]]] = None,
    no_sync: bool = False,
) -> Optional[Dict[int, float]]:
    """Run ONE profiler session for all msg_sizes, return {msg_bytes: median_us}.

    CANN profiler cannot be started/stopped repeatedly in the same process.
    This function batches all msg_sizes into a single profiler session to avoid
    the crash. Each msg_size gets a per-size number of active iterations
    (reduced for large messages >=512KB to limit profiler overhead); durations
    are split by position in the parsed CSV.

    When no_sync=True, skips torch.npu.synchronize() between iterations,
    allowing HCCL pipeline overlap (matches production behavior).

    All ranks in group_ranks must call this function together (collective ops).
    Only is_leader=True rank runs the profiler; others run the same call pattern.
    Returns None for non-leader ranks.
    """
    import torch

    # Build run_op for each msg_size, with per-size active iters
    run_ops: List[Tuple[int, Callable, int]] = [
        (mb, _build_run_op(op_type, mb, dtype_str, device, group, group_ranks),
         _active_iters_for_msg(mb))
        for mb in msg_bytes_list
    ]

    # Warmup all msg_sizes (outside profiler)
    for _, run_op, _ in run_ops:
        for _ in range(WARMUP_ITERS):
            run_op()
    if is_npu:
        torch.npu.synchronize()

    total_active = sum(n_iters for _, _, n_iters in run_ops)

    if not is_leader:
        # Follower: match exact call pattern without profiler
        first_run_op = run_ops[0][1]
        for _ in range(PROFILER_WAIT_ITERS + PROFILER_WARMUP_ITERS):
            first_run_op()
            if is_npu and not no_sync:
                torch.npu.synchronize()
        for _, run_op, n_iters in run_ops:
            for _ in range(n_iters):
                run_op()
                if is_npu and not no_sync:
                    torch.npu.synchronize()
        if is_npu and no_sync:
            torch.npu.synchronize()
        return None

    prof_dir = tempfile.mkdtemp(prefix="comm_bench_prof_")

    try:
        import torch_npu  # noqa: F811

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=True,
        )

        first_run_op = run_ops[0][1]

        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=PROFILER_WAIT_ITERS,
                warmup=PROFILER_WARMUP_ITERS,
                active=total_active,
                repeat=1,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
            experimental_config=experimental_config,
        ) as prof:
            for _ in range(PROFILER_WAIT_ITERS + PROFILER_WARMUP_ITERS):
                first_run_op()
                if is_npu and not no_sync:
                    torch.npu.synchronize()
                prof.step()
            for _, run_op, n_iters in run_ops:
                for _ in range(n_iters):
                    run_op()
                    if is_npu and not no_sync:
                        torch.npu.synchronize()
                    prof.step()
            if is_npu and no_sync:
                torch.npu.synchronize()

        _parse = parse_fn or _parse_kernel_comm_duration
        durations = _parse(prof_dir, op_type)

        if not durations:
            print(
                f"WARNING: No duration entries found for {op_type} in {prof_dir} "
                f"(parse_fn={_parse.__name__}). Returning empty results.",
                file=sys.stderr,
            )
            return {}

        expected = total_active
        if len(durations) < expected:
            print(
                f"WARNING: Expected {expected} entries but got {len(durations)} "
                f"for {op_type} in {prof_dir}",
                file=sys.stderr,
            )

        results: Dict[int, float] = {}
        offset = 0
        for msg_bytes, _, n_iters in run_ops:
            chunk = durations[offset:offset + n_iters]
            offset += n_iters
            if chunk:
                results[msg_bytes] = statistics.median(chunk)
            else:
                print(f"WARNING: No duration data for {op_type} msg_bytes={msg_bytes}", file=sys.stderr)

        return results

    finally:
        try:
            shutil.rmtree(prof_dir)
        except OSError as e:
            print(f"WARNING: Failed to clean up profiler dir {prof_dir}: {e}", file=sys.stderr)


# ============================================================================
# Bench mode: kernel (hcom_* kernel Duration, excluding AivKernel)
# ============================================================================

def _parse_kernel_comm_duration(prof_dir: str, op_type: str) -> List[float]:
    """Parse kernel_details.csv, extract hcom_* Duration excluding AivKernel.

    This measures the same physical quantity as Communication in step_trace:
        Communication = Sum kernel_details hcom_* Duration (excluding AivKernel)

    Deduplication: kernel_details.csv records each HCCL op twice --
    once as hcom_* (Stream ID = NaN) and once as AivKernel (on HCCL stream).
    We filter by: Type starts with 'hcom_' AND Name does NOT contain 'AivKernel'.
    """
    target_type = _OP_TO_KERNEL_TYPE[op_type]
    durations: List[float] = []

    pattern = os.path.join(prof_dir, "**", "kernel_details.csv")
    csv_files = glob.glob(pattern, recursive=True)

    if not csv_files:
        print(f"WARNING: No kernel_details.csv found in {prof_dir}", file=sys.stderr)
        return durations

    csv_path = sorted(csv_files)[0]  # rank 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_type = row.get("Type", "")
            row_name = row.get("Name", "")
            if row_type.startswith(target_type) and "AivKernel" not in row_name:
                dur = float(row.get("Duration(us)", "0"))
                if dur > 0:
                    durations.append(dur)

    return durations


def _run_bench_kernel(run_op, op_type: str, is_npu: bool, *, is_leader: bool = True,
                      no_sync: bool = False) -> Optional[float]:
    """Run bench via torch_npu.profiler, return median hcom_* kernel Duration (us).

    Parses kernel_details.csv, extracting only hcom_* Duration (excluding
    AivKernel duplicates). This aligns exactly with:
        Communication (step_trace) = Sum kernel_details hcom_* Duration (excluding AivKernel)

    When no_sync=True, skips torch.npu.synchronize() between iterations during
    the profiler active phase, allowing HCCL calls to pipeline on the device
    stream (matches production behavior).

    When is_leader=False, executes the same call pattern without starting a
    profiler (follower mode for non-leader ranks in collective communication).
    """
    _warmup_and_sync(run_op, is_npu)

    if not is_leader:
        _profiler_loop_steps(run_op, is_npu)
        return None

    prof_dir = tempfile.mkdtemp(prefix="comm_bench_kernel_")

    try:
        import torch
        import torch_npu  # noqa: F811

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=True,
        )

        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=PROFILER_WAIT_ITERS,
                warmup=PROFILER_WARMUP_ITERS,
                active=PROFILER_ACTIVE_ITERS,
                repeat=1,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
            experimental_config=experimental_config,
        ) as prof:
            total_steps = PROFILER_WAIT_ITERS + PROFILER_WARMUP_ITERS + PROFILER_ACTIVE_ITERS
            for _step in range(total_steps):
                run_op()
                if is_npu and not no_sync:
                    torch.npu.synchronize()
                prof.step()
            # When no_sync, ensure all device ops complete before reading results
            if is_npu and no_sync:
                torch.npu.synchronize()

        durations = _parse_kernel_comm_duration(prof_dir, op_type)
        if not durations:
            print(
                f"WARNING: No {_OP_TO_KERNEL_TYPE[op_type]} entries found in "
                f"kernel_details.csv under {prof_dir}. Returning None.",
                file=sys.stderr,
            )
            return None

        return statistics.median(durations)

    finally:
        try:
            shutil.rmtree(prof_dir)
        except OSError as e:
            print(f"WARNING: Failed to clean up profiler dir {prof_dir}: {e}", file=sys.stderr)


# ============================================================================
# Bench mode: event (event mode -- measures hcom_kernel only, no AicpuKernel)
# ============================================================================

def _run_bench_event(run_op, is_npu: bool) -> float:
    """Run bench via per-iteration NPU Event timing, return median Duration (us).

    NPU Events measure Device-side hcom_kernel execution time. This does NOT
    include AicpuKernel overhead, so it underestimates Comm_NO on models like
    Qwen3 where AicpuKernel > 0. Use as a fast sanity check alongside profiler mode.
    """
    import torch

    _warmup_and_sync(run_op, is_npu)

    durations_us: List[float] = []
    if is_npu:
        # Reuse a single pair of Event objects across iterations
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        for _ in range(BENCH_ITERS):
            torch.npu.synchronize()
            start_event.record()
            run_op()
            end_event.record()
            torch.npu.synchronize()
            durations_us.append(start_event.elapsed_time(end_event) * 1000)  # ms -> us
    else:
        for _ in range(BENCH_ITERS):
            t0 = time.perf_counter()
            run_op()
            durations_us.append((time.perf_counter() - t0) * 1e6)

    return statistics.median(durations_us)


# ============================================================================
# CLI
# ============================================================================

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HCCL communication microbenchmarks and collect CSV data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            Examples:
              # Run all ops + all tiers in ONE torchrun session (recommended)
              # tier=1 (intra_pod): 16 devices; tier=2 (die_level): 2 devices
              # Writes hcom_allReduce_.csv / hcom_allGather_.csv / etc. to --output-dir
              torchrun --nproc_per_node=16 generate_comm_microbench.py \\
                  --do-run --output-dir ./hccl_data \\
                  --ops all_reduce all_gather reduce_scatter all_to_all \\
                  --grid-shape 48 8 2 --num-devices 16 2

              # Single op, single CSV (legacy)
              torchrun --nproc_per_node=16 generate_comm_microbench.py \\
                  --do-run --output-csv ./hccl_v8.5/hcom_allReduce_.csv \\
                  --ops all_reduce --grid-shape 48 8 2
        """),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write per-op CSV files "
            "(hcom_allReduce_.csv, hcom_allGather_.csv, etc.). "
            "Ignored when --output-csv is given."
        ),
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=_COMM_OPS,
        choices=_COMM_OPS,
        help="Communication ops to benchmark (default: all 4)",
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        nargs="+",
        default=[16],
        help="Number of devices per communicator group (default: 16)",
    )
    parser.add_argument(
        "--topology-tier",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Topology tier(s) to benchmark: 0=inter_pod 1=intra_pod 2=die_level. "
            "Default: auto-resolve all tiers from --grid-shape and --num-devices."
        ),
    )
    parser.add_argument(
        "--grid-shape",
        type=int,
        nargs="+",
        default=[48, 8, 2],
        help=(
            "Hardware grid shape (outermost to innermost), e.g. '48 8 2' for "
            "ATLAS_800_A3 (48 pods x 8 nodes x 2 dies, stride=[16,2,1]). "
            "Used to resolve topology_tier from group composition. (default: 48 8 2)"
        ),
    )
    parser.add_argument(
        "--dtype",
        default="torch.bfloat16",
        choices=list(_DTYPE_ELEM_SIZE.keys()),
        help="Tensor dtype (default: torch.bfloat16)",
    )
    parser.add_argument(
        "--bytes-grid",
        type=int,
        nargs="+",
        default=None,
        help="Custom message_bytes grid (default: 1KB~512MB powers-of-4)",
    )
    parser.add_argument(
        "--do-run",
        action="store_true",
        dest="run",
        help="Run benchmarks directly (requires torchrun)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="CSV file to append results to (used with --do-run, format per S4.7)",
    )
    parser.add_argument(
        "--bench-mode",
        default="kernel",
        choices=_BENCH_MODES,
        help=(
            "Measurement mode (default: kernel). "
            "'kernel': torch_npu.profiler -> kernel_details hcom_* Duration "
            "(excludes AivKernel, aligns Communication in step_trace). "
            "'event': per-iteration NPU Event timing (hcom_kernel only, fast)."
        ),
    )
    return parser


def _iter_configs(
    ops: List[str],
    num_devices_list: List[int],
    topology_tiers: Optional[List[int]],
    grid_shape: List[int],
    bytes_grid: List[int],
) -> List[Tuple]:
    """Yield (op_type, message_bytes, num_devices, topology_tier, group_ranks) tuples.

    If topology_tiers is None, auto-resolve tier from group composition.
    Uses rank=0 as the anchor rank for group construction.
    """
    configs = []
    anchor_rank = 0
    for op_type in ops:
        for num_devices in num_devices_list:
            tiers_to_run = topology_tiers
            if tiers_to_run is None:
                # Build group anchored at rank 0 spanning the full num_devices,
                # then resolve tier from the group composition.
                try:
                    group = list(range(num_devices))
                    tier = resolve_topology_tier(group, grid_shape)
                    tiers_to_run = [tier]
                except ValueError:
                    tiers_to_run = [len(grid_shape) - 1]

            for tier in tiers_to_run:
                try:
                    group_ranks = build_group_for_tier(anchor_rank, num_devices, tier, grid_shape)
                except ValueError as e:
                    print(f"WARNING: skipping tier={tier}, num_devices={num_devices}: {e}", file=sys.stderr)
                    continue
                for msg_bytes in bytes_grid:
                    configs.append((op_type, msg_bytes, num_devices, tier, group_ranks))
    return configs


def main() -> None:
    args = build_argparser().parse_args()
    print_logo()
    bytes_grid = args.bytes_grid or _DEFAULT_BYTES_GRID
    grid_shape = args.grid_shape

    # Validate: --output-csv only supports a single op to avoid silent data mixing
    if args.output_csv and len(args.ops) > 1:
        print(
            "ERROR: --output-csv only supports a single --ops value. "
            "Use --output-dir for multiple ops.",
            file=sys.stderr,
        )
        sys.exit(1)

    configs = _iter_configs(
        args.ops, args.num_devices, args.topology_tier,
        grid_shape, bytes_grid,
    )

    if args.run:
        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group(backend="hccl" if _has_torch_npu() else "gloo")
        except Exception as e:
            print(f"ERROR: Failed to initialize distributed: {e}", file=sys.stderr)
            print("Hint: run with `torchrun --nproc_per_node=N generate_comm_microbench.py --do-run ...`",
                  file=sys.stderr)
            sys.exit(1)

        import torch
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if _has_torch_npu():
            import torch_npu  # noqa: F401
            torch.npu.set_device(local_rank)

        # Resolve per-op output CSV paths.
        # Priority: --output-csv (single file, legacy) > --output-dir (per-op files) > None
        if args.output_csv:
            def _csv_for_op(op_type: str) -> Optional[str]:
                return args.output_csv
        elif args.output_dir:
            run_out_dir = Path(args.output_dir)
            run_out_dir.mkdir(parents=True, exist_ok=True)
            def _csv_for_op(op_type: str) -> Optional[str]:
                return str(run_out_dir / _OP_TO_CSV_FILENAME[op_type])
        else:
            def _csv_for_op(op_type: str) -> Optional[str]:
                return None

        # Pre-create one process group per unique group_ranks to avoid
        # repeated hcclCommInitRootInfoConfig calls (HCCL error code 1).
        # dist.new_group() must be called by ALL ranks in the world even if
        # they are not in the group -- so we call it unconditionally here.
        group_cache: dict = {}
        unique_groups = []
        seen = set()
        for _, _, _, _, group_ranks in configs:
            key = tuple(group_ranks)
            if key not in seen:
                seen.add(key)
                unique_groups.append(group_ranks)
        for group_ranks in unique_groups:
            key = tuple(group_ranks)
            group_cache[key] = dist.new_group(ranks=list(group_ranks))

        # Global warmup: run each (op, group, msg_bytes) combination once to
        # trigger HCCL JIT compilation for ALL message sizes before benchmarking.
        # HCCL compiles different internal kernels per message size; warming up
        # only the smallest size leaves mid-range sizes (1~5MB) un-compiled,
        # causing the first real measurement to include JIT overhead.
        # Always use event mode for warmup (fast, no profiler overhead).
        if rank == 0:
            print("Running global warmup to trigger HCCL JIT compilation "
                  f"({len(configs)} configs)...")
        warmed = set()
        for op_type, msg_bytes, _, _, group_ranks in configs:
            wkey = (op_type, msg_bytes, tuple(group_ranks))
            if wkey not in warmed:
                warmed.add(wkey)
                run_benchmark(
                    op_type, msg_bytes, group_ranks,
                    resolve_topology_tier(list(group_ranks), grid_shape),
                    args.dtype, output_csv=None,
                    group=group_cache[tuple(group_ranks)],
                    bench_mode="event",
                )

        if rank == 0:
            print(f"Global warmup done. Starting benchmarks (mode={args.bench_mode})...\n")


        if args.bench_mode == "kernel" and _has_torch_npu():
            # Kernel mode: profiler -> kernel_details.csv, no inter-iteration sync.
            # To avoid profiler ring-buffer pressure on large messages, split:
            #   - Small messages (<512KB): batched into ONE profiler session
            #   - Large messages (>=512KB): each msg_bytes gets its OWN profiler
            #     session with active=1, repeated N times, then take median.
            batched: OrderedDict = OrderedDict()
            for op_type, msg_bytes, num_devices, tier, group_ranks in configs:
                key = (op_type, tuple(group_ranks))
                if key not in batched:
                    batched[key] = []
                batched[key].append((msg_bytes, num_devices, tier))

            is_npu = True
            device = f"npu:{local_rank}"

            total_batches = len(batched)
            for batch_idx, ((op_type, gr_tuple), items) in enumerate(batched.items()):
                group_ranks = list(gr_tuple)
                group = group_cache[gr_tuple]
                is_member = rank in group_ranks

                if is_member:
                    is_leader = rank == group_ranks[0]

                    # Split into small and large msg_bytes
                    small_items = [(mb, nd, t) for mb, nd, t in items
                                   if mb < PROFILER_LARGE_MSG_THRESHOLD]
                    large_items = [(mb, nd, t) for mb, nd, t in items
                                   if mb >= PROFILER_LARGE_MSG_THRESHOLD]

                    if is_leader:
                        print(f"\n[kernel] batch {batch_idx+1}/{total_batches}  "
                              f"op={op_type}  group={group_ranks}  "
                              f"small(<{PROFILER_LARGE_MSG_THRESHOLD//1024}KB)="
                              f"{len(small_items)}(batch, active={PROFILER_ACTIVE_ITERS})  "
                              f"large(>={PROFILER_LARGE_MSG_THRESHOLD//1024}KB)="
                              f"{len(large_items)}(per-msg session"
                              f"\u00d7{PROFILER_LARGE_MSG_SESSIONS}, "
                              f"active={PROFILER_ACTIVE_ITERS_LARGE})")

                    # 1) Small messages: one batch session
                    if small_items:
                        small_msg_list = [mb for mb, _, _ in small_items]
                        try:
                            results = _run_bench_profiler_batch(
                                op_type, small_msg_list, args.dtype, device,
                                group, group_ranks, is_npu, is_leader,
                                parse_fn=_parse_kernel_comm_duration,
                                no_sync=True,
                            )
                        except Exception as e:
                            if is_leader:
                                print(f"  ERROR [small-batch] op={op_type}: {e}",
                                      file=sys.stderr)
                            results = None

                        if results and is_leader:
                            for msg_bytes, nd, tier in small_items:
                                if msg_bytes not in results:
                                    print(f"  SKIP op={op_type} bytes={msg_bytes} "
                                          f"(no data in batch result)", file=sys.stderr)
                                    continue
                                duration_us = results[msg_bytes]
                                bandwidth_gbps = msg_bytes / (duration_us * 1e-6) / 1e9
                                row = {
                                    "message_bytes": msg_bytes,
                                    "num_devices": nd,
                                    "dtype": _DTYPE_TO_CSV.get(args.dtype, "DT_BF16"),
                                    "topology_tier": tier,
                                    "Duration(us)": round(duration_us, 2),
                                    "bandwidth_gbps": round(bandwidth_gbps, 2),
                                }
                                print(
                                    f"  op={op_type}  bytes={msg_bytes}  devices={nd}"
                                    f"  tier={tier}  duration={duration_us:.2f}us"
                                    f"  bw={bandwidth_gbps:.2f}GB/s  [small-batch]"
                                )
                                csv_path = _csv_for_op(op_type)
                                if csv_path:
                                    _append_csv(csv_path, row, op_type=op_type)

                    # 2) Large messages: per-msg separate profiler sessions
                    for large_idx, (msg_bytes, nd, tier) in enumerate(large_items):
                        if is_leader:
                            print(f"  [large {large_idx+1}/{len(large_items)}] "
                                  f"op={op_type}  bytes={msg_bytes}  "
                                  f"sessions=0/{PROFILER_LARGE_MSG_SESSIONS}...",
                                  end="", flush=True)

                        durations_across_sessions: List[float] = []
                        session_errors = 0
                        for repeat_idx in range(PROFILER_LARGE_MSG_SESSIONS):
                            try:
                                result = _run_bench_profiler_batch(
                                    op_type, [msg_bytes], args.dtype, device,
                                    group, group_ranks, is_npu, is_leader,
                                    parse_fn=_parse_kernel_comm_duration,
                                    no_sync=True,
                                )
                                if result and is_leader and msg_bytes in result:
                                    durations_across_sessions.append(result[msg_bytes])
                            except Exception as e:
                                session_errors += 1
                                if is_leader:
                                    print(f"\n    WARN session {repeat_idx+1} failed: {e}",
                                          file=sys.stderr, end="", flush=True)

                        if is_leader:
                            ok = len(durations_across_sessions)
                            fail = session_errors
                            print(f"\r  [large {large_idx+1}/{len(large_items)}] "
                                  f"op={op_type}  bytes={msg_bytes}  "
                                  f"sessions={ok}/{ok+fail}"
                                  f"{f' ({fail} failed)' if fail else ''}",
                                  end="")

                        if durations_across_sessions and is_leader:
                            duration_us = statistics.median(durations_across_sessions)
                            bandwidth_gbps = msg_bytes / (duration_us * 1e-6) / 1e9
                            row = {
                                "message_bytes": msg_bytes,
                                "num_devices": nd,
                                "dtype": _DTYPE_TO_CSV.get(args.dtype, "DT_BF16"),
                                "topology_tier": tier,
                                "Duration(us)": round(duration_us, 2),
                                "bandwidth_gbps": round(bandwidth_gbps, 2),
                            }
                            print(
                                f"  duration={duration_us:.2f}us"
                                f"  bw={bandwidth_gbps:.2f}GB/s"
                            )
                            csv_path = _csv_for_op(op_type)
                            if csv_path:
                                _append_csv(csv_path, row, op_type=op_type)
                        elif is_leader:
                            print(f"  FAILED (0 valid sessions)", file=sys.stderr)

                # World barrier: ALL ranks must participate (including non-members)
                # to prevent non-member ranks from racing ahead to the next batch.
                dist.barrier()
        else:
            # Event mode: per-point measurement (no profiler session needed)
            for op_type, msg_bytes, num_devices, tier, group_ranks in configs:
                run_benchmark(
                    op_type, msg_bytes, group_ranks, tier, args.dtype,
                    _csv_for_op(op_type),
                    group=group_cache[tuple(group_ranks)],
                    bench_mode="event",
                )

        dist.destroy_process_group()
        if rank == 0:
            n = len(configs)
            out_info = args.output_csv or args.output_dir or "(no output)"
            print(f"\nCompleted {n} benchmarks. Results saved to {out_info}")
    else:
        print("ERROR: --do-run is required. Script generation mode has been removed.", file=sys.stderr)
        print("Use: torchrun --nproc_per_node=N generate_comm_microbench.py --do-run ...", file=sys.stderr)
        sys.exit(1)


def _has_torch_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
