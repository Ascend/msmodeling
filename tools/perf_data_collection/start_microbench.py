"""Profile replay scripts and write results back to the database.

Usage:
DATA_DIR="$(pwd)/tensor_cast/performance_model/profiling_database/data/ATLAS_800_A3_752T_128G_DIE/vllm_ascend/vllm0.18.0_torch2.9.0_cann8.5"

export ASCEND_CUSTOM_OPP_PATH=/pathto/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/pathto/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/:${LD_LIBRARY_PATH}

python3 tools/perf_data_collection/start_microbench.py \
    --device ATLAS_800_A3_752T_128G_DIE \
    --vllm-version 0.18.0 \
    --torch-version 2.9.0 \
    --cann-version 8.5 \
    --prune-empty-duration-rows
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any

# =============================================================================
# Path Configuration
# =============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
OP_REPLAY_DIR = CURRENT_DIR / "op_replay"
RUN_ALL_SCRIPT = OP_REPLAY_DIR / "run_all_op.py"

if str(OP_REPLAY_DIR) not in sys.path:
    sys.path.insert(0, str(OP_REPLAY_DIR))

from common import (
    DEFAULT_DEVICE, SUPPORTED_DEVICES, build_database_cli_args, check_version,
    csv_has_complete_microbench, ensure_npu_available, get_target_data_dir,
    load_csv_rows, parse_float, row_has_valid_duration,
    row_has_only_invalid_durations,
)
from signature_utils import get_sig, is_matmul_family, normalize_op_name

# =============================================================================
# CSV Column Names
# =============================================================================
BASE_COLS = [
    "OP State", "Accelerator Core", "Input Shapes", "Input Data Types",
    "Input Formats", "Output Shapes", "Output Data Types", "Output Formats",
]
MATCH_COLS = [
    "Input Shapes", "Input Data Types", "Input Formats",
    "Output Shapes", "Output Data Types",
]

# Duration column names
LEGACY_MB_DUR = "MicroBench Duration(us)"
MB_DUR = "Average Duration(us)"
PROF_AVG_DUR = "Profiling Average Duration(us)"
PROF_MED_DUR = "Profiling Median Duration(us)"
PROF_STD_DUR = "Profiling Std Duration(us)"

# Profiling columns mapping: {source_col: display_col}
PROF_COLS = {
    k: f"Profiling Average {k}"
    for k in [
        "aicore_time(us)", "aic_total_cycles", "aic_mac_time(us)",
        "aic_mac_ratio", "aic_scalar_time(us)", "aic_scalar_ratio",
        "aic_mte1_time(us)", "aic_mte1_ratio", "aic_mte2_time(us)",
        "aic_mte2_ratio", "aic_fixpipe_time(us)", "aic_fixpipe_ratio",
        "aic_icache_miss_rate", "aiv_time(us)", "aiv_total_cycles",
        "aiv_vec_time(us)", "aiv_vec_ratio", "aiv_scalar_time(us)",
        "aiv_scalar_ratio", "aiv_mte2_time(us)", "aiv_mte2_ratio",
        "aiv_mte3_time(us)", "aiv_mte3_ratio", "aiv_icache_miss_rate",
        "cube_utilization(%)",
    ]
}
MB_EXTRA_COLS = {
    src: "MicroBench " + disp.removeprefix("Profiling Average ")
    for src, disp in PROF_COLS.items()
}

# =============================================================================
# Operator Configuration
# =============================================================================
DISPATCH_FFN_OP = "DispatchFFNCombine"
PROFILE_OP_ALIASES = {
    "MatMulV2": ("MatMulV3", "MatMulCommon"),
    "MatMulV3": ("MatMulV2", "MatMulCommon"),
    "MatMulCommon": ("MatMulV2", "MatMulV3"),
}

# Operators requiring custom OPP environment
CUSTOM_OPP_OPS = {
    "AddRmsNormBias",
    "DispatchFFNCombine",
    "KvRmsNormRopeCache",
    "RINGMLAPrefillBF16Kernel",
    "split_qkv_rmsnorm_rope_kernel",
}

# =============================================================================
# Report Configuration
# =============================================================================
GAP_BOUNDS = (0.8, 1.2)  # Ratio bounds for detecting duration gap hotspots
MAX_GAP_DISPLAY = 20  # Max hotspot rows to display
MAX_EXAMPLE_DISPLAY = 3  # Max examples to show in summary tables


@dataclass
class GapRecord:
    """Record for duration gap between microbench and profiling."""

    op_type: str
    csv_name: str
    signature: str
    mb_us: float
    prof_us: float
    diff_us: float
    ratio: float


@dataclass
class UpdateResult:
    """Result of updating a CSV file with profiling data."""

    csv_path: Path
    updated: int = 0
    added: int = 0
    unchanged: int = 0
    missing: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    duplicates: list[tuple[str, int]] = field(default_factory=list)
    gaps: list[GapRecord] = field(default_factory=list)


# =============================================================================
# CLI Argument Parsing
# =============================================================================
def list_ops() -> list[str]:
    """List available operator names from run scripts.

    Returns:
        Sorted list of normalized operator names.
    """
    return sorted(normalize_op_name(p.stem) for p in OP_REPLAY_DIR.glob("*_run.py"))


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    ops = ", ".join(list_ops())
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"Available operators: {ops}\n\nUpdate modes:\n"
               "  all: Update all matched rows\n"
               "  missing-only: Only fill rows with invalid durations")
    parser.add_argument("--database-path", type=Path)
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=SUPPORTED_DEVICES)
    parser.add_argument("--vllm-version", dest="vllm_version", type=check_version)
    parser.add_argument("--torch-version", type=check_version)
    parser.add_argument("--cann-version", type=check_version)
    parser.add_argument("--prof-path", help="Existing PROF_* directory")
    parser.add_argument("--op", nargs="+", help=f"Operators. Available: {ops}")
    parser.add_argument(
        "--dispatch-ffn-combine-ep-size",
        type=int,
        default=16,
        help="Expert-parallel size for DispatchFFNCombine replay. Default: 16.",
    )
    parser.add_argument(
        "--dispatch-ffn-combine-nproc-per-node",
        type=int,
        default=None,
        help=(
            "torchrun processes per node when launching DispatchFFNCombine "
            "in EP mode. Default: equal to EP_SIZE for single-node; "
            "must be set explicitly for multi-node."
        ),
    )
    parser.add_argument(
        "--dispatch-ffn-combine-nnodes",
        type=int,
        default=1,
        help="torchrun node count for DispatchFFNCombine EP mode. Default: 1.",
    )
    parser.add_argument(
        "--dispatch-ffn-combine-node-rank",
        type=int,
        default=0,
        help="torchrun node rank for DispatchFFNCombine EP mode. Default: 0.",
    )
    parser.add_argument(
        "--dispatch-ffn-combine-master-addr",
        default="127.0.0.1",
        help=(
            "torchrun master address for DispatchFFNCombine EP mode. "
            "Default: 127.0.0.1 (localhost)."
        ),
    )
    parser.add_argument(
        "--dispatch-ffn-combine-master-port",
        type=int,
        default=None,
        help=(
            "torchrun master port for DispatchFFNCombine EP mode. "
            "Default: auto-selected by torchrun on node 0."
        ),
    )
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--update-mode", choices=("all", "missing-only"), default="all")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--prune-empty-duration-rows", action="store_true")

    return parser


def validate_ops(selected: list[str] | None) -> list[str] | None:
    """Validate and normalize operator names.

    Args:
        selected: List of operator names from CLI, or None.

    Returns:
        Normalized operator names, or None if no selection.

    Raises:
        ValueError: If any operator name is invalid.
    """
    if not selected:
        return None

    available = set(list_ops())
    normalized = [normalize_op_name(op) for op in selected]
    invalid = sorted(op for op in normalized if op not in available)

    if invalid:
        raise ValueError(f"Unsupported --op: {', '.join(invalid)}. "
                         f"Available: {', '.join(sorted(available))}")

    return normalized


# =============================================================================
# Environment Setup
# =============================================================================
def ensure_custom_opp_env(selected_ops: list[str] | None) -> None:
    """Ensure environment variables for custom OPP operators are set.

    Args:
        selected_ops: Selected operator names, or None to check all.

    Raises:
        RuntimeError: If required environment variables are missing.
    """
    required = [op for op in (selected_ops or list_ops()) if op in CUSTOM_OPP_OPS]
    env_vars = ("ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH")

    if not required or all(os.environ.get(e, "").strip() for e in env_vars):
        return

    try:
        p = Path(import_module("vllm_ascend").__file__).resolve().parent
        custom_opp = f"{p}/_cann_ops_custom/vendors/vllm-ascend:$ASCEND_CUSTOM_OPP_PATH"
        ld_lib = (
            f"{p}/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/:$LD_LIBRARY_PATH")
    except (ImportError, AttributeError, OSError, TypeError):
        custom_opp = (
            "<vllm-ascend>/_cann_ops_custom/vendors/vllm-ascend:"
            "$ASCEND_CUSTOM_OPP_PATH")
        ld_lib = (
            "<vllm-ascend>/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/:"
            "$LD_LIBRARY_PATH")

    raise RuntimeError(f"Missing env vars for operators: {', '.join(required)}.\n"
                       f"Please run:\n  export ASCEND_CUSTOM_OPP_PATH={custom_opp}\n"
                       f"  export LD_LIBRARY_PATH={ld_lib}")


# =============================================================================
# Profiling Execution
# =============================================================================
def build_msprof_cmd(
    profiler_root: Path,
    args: argparse.Namespace,
    selected_ops: list[str] | None,
) -> list[str]:
    """Build the msprof command for one profiling run."""
    cmd = ["msprof", f"--output={profiler_root}", "python", str(RUN_ALL_SCRIPT),
           "--execution-mode", "inprocess"]

    if not args.fail_fast:
        cmd.append("--continue-on-error")

    cmd.extend(build_database_cli_args(
        database_path=args.database_path, device=args.device,
        vllm_ascend_version=args.vllm_version, torch_version=args.torch_version,
        cann_version=args.cann_version))

    if args.repeat_count:
        cmd += ["--repeat-count", str(args.repeat_count)]

    cmd += ["--update-mode", args.update_mode]

    if selected_ops:
        cmd += ["--op"] + selected_ops

    if args.dispatch_ffn_combine_ep_size:
        cmd += ["--dispatch-ffn-combine-ep-size",
                str(args.dispatch_ffn_combine_ep_size)]
    if args.dispatch_ffn_combine_nproc_per_node is not None:
        cmd += ["--dispatch-ffn-combine-nproc-per-node",
                str(args.dispatch_ffn_combine_nproc_per_node)]
    if args.dispatch_ffn_combine_nnodes is not None:
        cmd += ["--dispatch-ffn-combine-nnodes",
                str(args.dispatch_ffn_combine_nnodes)]
    if args.dispatch_ffn_combine_node_rank is not None:
        cmd += ["--dispatch-ffn-combine-node-rank",
                str(args.dispatch_ffn_combine_node_rank)]
    if args.dispatch_ffn_combine_master_addr:
        cmd += ["--dispatch-ffn-combine-master-addr",
                args.dispatch_ffn_combine_master_addr]
    if args.dispatch_ffn_combine_master_port is not None:
        cmd += ["--dispatch-ffn-combine-master-port",
                str(args.dispatch_ffn_combine_master_port)]

    return cmd


def run_msprof_cmd(profiler_root: Path, cmd: list[str]) -> tuple[int, set[Path]]:
    """Execute msprof command and return its code plus generated PROF dirs."""
    try:
        result = subprocess.run(cmd, check=False, cwd=REPO_ROOT)
    except FileNotFoundError as e:
        raise RuntimeError(
            "msprof not found. Activate Ascend toolkit environment.") from e

    prof_dirs = {p for p in profiler_root.rglob("PROF_*") if p.is_dir()}
    return result.returncode, prof_dirs


def run_msprof_per_op_fallback(
    profiler_root: Path,
    args: argparse.Namespace,
    selected_ops: list[str],
) -> set[Path]:
    """Profile selected operators one-by-one after a combined msprof failure."""
    all_prof_dirs: set[Path] = set()

    print(
        "[WARN] Combined msprof run produced no op_summary data. "
        "Retrying each selected operator in a separate msprof run."
    )
    for op in selected_ops:
        op_root = profiler_root / f"per_op_{op}"
        op_root.mkdir(parents=True, exist_ok=True)
        op_cmd = build_msprof_cmd(op_root, args, [op])
        returncode, prof_dirs = run_msprof_cmd(op_root, op_cmd)
        summary_files = find_summary_files(prof_dirs, raise_if_missing=False)
        if returncode != 0 and not summary_files:
            raise RuntimeError(
                f"msprof exited with {returncode} while profiling {op}; "
                f"profiling data kept at {op_root}: {subprocess.list2cmdline(op_cmd)}")
        if returncode != 0:
            print(
                f"[WARN] msprof exited with {returncode} while profiling {op}, "
                f"but {len(summary_files)} op_summary file(s) were generated."
            )
        all_prof_dirs.update(prof_dirs)

    return all_prof_dirs


def run_msprof(target_dir: Path, args: argparse.Namespace,
               selected_ops: list[str] | None,
               allow_per_op_fallback: bool = True) -> tuple[Path, set[Path]]:
    """Run msprof to profile operator execution.

    Args:
        target_dir: Directory for profiler output.
        args: Parsed CLI arguments.
        selected_ops: Selected operator names, or None for all.
        allow_per_op_fallback: Whether to rerun operators one-by-one when the
            combined msprof command fails without summary data.

    Returns:
        Tuple of (profiler_root_path, set_of_PROF_directories).

    Raises:
        RuntimeError: If msprof is not found or exits with error.
        FileNotFoundError: If no PROF_* directories are created.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    profiler_root = Path(tempfile.mkdtemp(prefix="msprof_run_", dir=target_dir))

    cmd = build_msprof_cmd(profiler_root, args, selected_ops)
    returncode, prof_dirs = run_msprof_cmd(profiler_root, cmd)

    if returncode != 0:
        summary_files = find_summary_files(prof_dirs, raise_if_missing=False)
        if summary_files:
            print(
                f"[WARN] msprof exited with {returncode}, but "
                f"{len(summary_files)} op_summary file(s) were generated under "
                f"{profiler_root}. Continuing with generated profiling data."
            )
        else:
            if selected_ops is None or not allow_per_op_fallback:
                raise RuntimeError(
                    "combined msprof failed without op_summary data; rerun "
                    "with --op to enable per-op fallback. Profiling data kept "
                    f"at {profiler_root}: {subprocess.list2cmdline(cmd)}")
            fallback_ops = selected_ops or list_ops()
            if len(fallback_ops) <= 1 or args.fail_fast:
                raise RuntimeError(
                    f"msprof exited with {returncode}; profiling data kept at "
                    f"{profiler_root}: {subprocess.list2cmdline(cmd)}")
            prof_dirs = run_msprof_per_op_fallback(
                profiler_root, args, fallback_ops)

    if not prof_dirs:
        raise FileNotFoundError(f"No PROF_* directories under {profiler_root}")

    return profiler_root, prof_dirs


def find_summary_files(
    prof_dirs: set[Path],
    *,
    raise_if_missing: bool = True,
) -> list[Path]:
    """Find op_summary CSV files in profiler output directories.

    Args:
        prof_dirs: Set of PROF_* directory paths.
        raise_if_missing: Raise if no summary files are found.

    Returns:
        Sorted list of op_summary_*.csv file paths.

    Raises:
        FileNotFoundError: If no summary files are found.
    """
    files = [
        f for d in sorted(prof_dirs)
        for f in sorted((d / "mindstudio_profiler_output").glob("op_summary_*.csv"))]

    if not files and raise_if_missing:
        raise FileNotFoundError("No op_summary_*.csv found")

    return files


# =============================================================================
# Data Processing
# =============================================================================
def read_status() -> dict[str, Any] | None:
    """Read operator execution status from JSON file.

    Returns:
        Status dict with success/failed/skipped lists, or None if file missing.
    """
    p = OP_REPLAY_DIR / "run_all_op_status.json"

    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (json.JSONDecodeError, OSError):
        return None


def aggregate_summary(files: list[Path], ep_size: int | None
                      ) -> dict[str, list[dict[str, str]]]:
    """Aggregate op_summary CSV files by operator type.

    Args:
        files: List of op_summary CSV file paths.
        ep_size: Expert parallel size for DispatchFFNCombine.

    Returns:
        Dict mapping operator type to list of aggregated row dicts.
    """
    grouped: dict[tuple[str, tuple], dict] = {}

    # Read and group rows by (op_type, signature)
    for csv_path in files:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                op_type = (row.get("OP Type", "") or "").strip()
                if not op_type:
                    continue

                key = (op_type, get_sig(row, op_name=op_type))
                item = grouped.setdefault(key, {"count": 0, "row": row, "min_dur": None,
                                                "sums": defaultdict(float)})
                dur = parse_float(row.get("Task Duration(us)", ""))
                item["count"] += 1

                if item["min_dur"] is None or dur < item["min_dur"]:
                    item["min_dur"] = dur

                for src in PROF_COLS:
                    item["sums"][src] += parse_float(row.get(src, ""))

    # Build aggregated results
    result: dict[str, list[dict[str, str]]] = defaultdict(list)

    for (op_type, _), item in grouped.items():
        cnt, src_row = item["count"], dict(item["row"])
        agg = {k: (src_row.get(k, "") or "").strip() for k in BASE_COLS}
        agg[MB_DUR] = f"{item['min_dur']:.6f}"
        agg["Accelerator Core"] = (src_row.get("Task Type", "") or "").strip()

        if normalize_op_name(op_type) == normalize_op_name(DISPATCH_FFN_OP) and ep_size:
            agg["EP Size"] = str(ep_size)

        for src, mb_col in MB_EXTRA_COLS.items():
            agg[mb_col] = f"{item['sums'][src] / cnt:.6f}"

        result[op_type].append(agg)

    return result


# =============================================================================
# CSV Update
# =============================================================================
def get_cols(fieldnames: list[str] | None) -> list[str]:
    """Get column list, handling legacy column names and inserting missing cols.

    Args:
        fieldnames: Original column names from CSV, or None for default schema.

    Returns:
        Processed column name list.
    """
    exclude = {"MicroBench Task Duration(us)", "MicroBench Kernel Duration(us)"}

    # Return full default schema for new CSV files
    if fieldnames is None:
        cols = list(BASE_COLS)
        cols.append(MB_DUR)
        cols.append(PROF_AVG_DUR)
        cols.append(PROF_MED_DUR)
        cols.append(PROF_STD_DUR)
        for prof_col in PROF_COLS.values():
            mb_col = "MicroBench " + prof_col.removeprefix("Profiling Average ")
            cols.append(mb_col)
            cols.append(prof_col)
        return cols

    cols = [c for c in fieldnames if c not in exclude]

    # Handle legacy column name
    if LEGACY_MB_DUR in cols and MB_DUR not in cols:
        cols[cols.index(LEGACY_MB_DUR)] = MB_DUR

    # Ensure MB_DUR column exists
    if MB_DUR not in cols:
        cols = BASE_COLS + [MB_DUR] + [c for c in cols if c not in BASE_COLS]

    # Insert MicroBench columns before Profiling columns
    for prof_col in PROF_COLS.values():
        mb_col = "MicroBench " + prof_col.removeprefix("Profiling Average ")
        if prof_col in cols and mb_col not in cols:
            cols.insert(cols.index(prof_col), mb_col)

    return cols


def update_csv(csv_path: Path, rows_to_merge: list[dict[str, str]],
               mode: str, prune: bool,
               match_only_rows: list[dict[str, str]] | None = None) -> UpdateResult:
    """Update CSV file with new profiling data rows.

    Args:
        csv_path: Path to CSV file.
        rows_to_merge: New rows to merge into existing data.
        mode: Update mode - "all" or "missing-only".
        prune: If True, remove rows with only invalid durations.

    Returns:
        UpdateResult with update statistics.
    """
    result = UpdateResult(csv_path=csv_path)
    existing_rows, columns = [], get_cols(None)

    # Read existing data
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            columns = get_cols(list(reader.fieldnames or []))
            existing_rows = list(reader)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # For new CSV, include extra columns from rows_to_merge (e.g., EP Size)
        if rows_to_merge:
            extra_cols = []
            for row in rows_to_merge:
                for k in row:
                    if k not in columns and k not in extra_cols:
                        extra_cols.append(k)
            columns = columns + extra_cols

    if normalize_op_name(csv_path.stem) == normalize_op_name(DISPATCH_FFN_OP) and "EP Size" not in columns:
        incoming_ep_sizes = {
            (row.get("EP Size", "") or "").strip()
            for row in [*(rows_to_merge or []), *((match_only_rows or []))]
            if (row.get("EP Size", "") or "").strip()
        }
        if len(incoming_ep_sizes) == 1:
            legacy_ep_size = next(iter(incoming_ep_sizes))
            columns.append("EP Size")
            for row in existing_rows:
                row["EP Size"] = legacy_ep_size
            print(
                f"[WARN] {csv_path.name} has no EP Size column; "
                f"using incoming EP Size={legacy_ep_size} for legacy row matching."
            )

    # Build signature index and detect duplicates
    sig_idx: dict[tuple, int] = {}
    dup_counts: dict[tuple, int] = {}

    for i, row in enumerate(existing_rows):
        s = get_sig(row, op_name=csv_path.stem)
        if s in sig_idx:
            dup_counts[s] = dup_counts.get(s, 1) + 1
        else:
            sig_idx[s] = i
            dup_counts.setdefault(s, 1)

    result.duplicates = [(get_sig(existing_rows[sig_idx[s]], True), c)
                       for s, c in dup_counts.items() if c > 1]
    result.duplicates.sort(key=lambda x: (-x[1], x[0]))

    # Merge new rows
    for row in existing_rows:
        if LEGACY_MB_DUR in row and MB_DUR not in row:
            row[MB_DUR] = row.get(LEGACY_MB_DUR, "")

    original = [{c: row.get(c, "") for c in columns} for row in existing_rows]

    def merge_row(
        new_row: dict[str, str],
        *,
        allow_add: bool,
        record_missing: bool,
        record_unchanged: bool,
    ) -> None:
        s = get_sig(new_row, op_name=csv_path.stem)

        if s not in sig_idx:
            if record_missing:
                result.missing.append(get_sig(new_row, True))
            if not allow_add:
                print(
                    f"[WARN] match-only profiling row did not match {csv_path.name}: "
                    f"{get_sig(new_row, True)}"
                )
            if allow_add and mode == "all":
                existing_rows.append(new_row)
                sig_idx[s] = len(existing_rows) - 1
                result.added += 1
            return

        row = existing_rows[sig_idx[s]]
        can_update = mode == "all" or not row_has_valid_duration(row)

        if not can_update:
            if record_unchanged:
                result.unchanged += 1
            else:
                return
        else:
            old_mb = row.get(MB_DUR, "")
            row[MB_DUR] = new_row.get(MB_DUR, "")

            for mb_col in MB_EXTRA_COLS.values():
                if mb_col in new_row:
                    row[mb_col] = new_row[mb_col]

            result.updated += (
                1 if (old_mb or "").strip() != (row.get(MB_DUR) or "").strip() else 0)

        # Record gap between microbench and profiling durations
        mb_us = parse_float(row.get(MB_DUR, ""))
        prof_us = parse_float(row.get(PROF_AVG_DUR, ""))

        if mb_us > 0 and prof_us > 0:
            result.gaps.append(GapRecord(
                csv_path.stem, csv_path.name, get_sig(row, True),
                mb_us, prof_us, abs(mb_us - prof_us), mb_us / prof_us))

    for new_row in rows_to_merge:
        merge_row(
            new_row,
            allow_add=True,
            record_missing=True,
            record_unchanged=True,
        )

    for new_row in match_only_rows or []:
        merge_row(
            new_row,
            allow_add=False,
            record_missing=False,
            record_unchanged=False,
        )

    # Prune invalid rows
    kept, norm_orig = [], [{c: r.get(c, "") for c in columns} for r in original]

    for row in existing_rows:
        norm_row = {c: row.get(c, "") for c in columns}

        if prune and row_has_only_invalid_durations(norm_row):
            result.deleted.append(get_sig(norm_row, True))
        else:
            kept.append(norm_row)

    # Write if changed
    if not csv_path.exists() or kept != norm_orig:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(kept)

    return result


def update_db(target_dir: Path, aggregated: dict[str, list[dict[str, str]]],
              selected: list[str] | None, mode: str, prune: bool) -> list[UpdateResult]:
    """Update database CSV files with aggregated profiling data.

    Args:
        target_dir: Target database directory.
        aggregated: Dict mapping operator type to aggregated rows.
        selected: Selected operator names, or None for all.
        mode: Update mode - "all" or "missing-only".
        prune: If True, remove rows with only invalid durations.

    Returns:
        List of UpdateResult for each updated CSV file.
    """
    if selected:
        csv_paths = [
            p for op in selected for p in sorted(target_dir.rglob(f"{op}.csv"))]
    else:
        avail = set(list_ops())
        csv_paths = sorted(
            p for p in target_dir.glob("*.csv") if normalize_op_name(p.stem) in avail)

    csv_by_op = {p.stem: p for p in csv_paths} if csv_paths else {
        op: target_dir / f"{op}.csv" for op in aggregated}

    results = []
    for op, path in sorted(csv_by_op.items()):
        rows_to_merge = list(aggregated.get(op, []))
        match_only_rows = []
        for alias in PROFILE_OP_ALIASES.get(op, ()):
            match_only_rows.extend(aggregated.get(alias, []))
        if is_matmul_family(op):
            match_only_rows = rows_to_merge + match_only_rows
            rows_to_merge = []
        results.append(update_csv(path, rows_to_merge, mode, prune, match_only_rows))
    return results


def get_visible_npu_count() -> int:
    try:
        runtime_torch = import_module("torch")
        npu = getattr(runtime_torch, "npu", None)
        if npu is None or not npu.is_available():
            return 0
        return int(npu.device_count())
    except Exception:
        return 0


def should_skip_dispatch_ffn_msprof(
    selected_ops: list[str] | None,
    *,
    ep_size: int,
    nproc_per_node: int | None,
    visible_devices: int,
    update_mode: str,
    has_prof_path: bool,
) -> bool:
    if has_prof_path or update_mode != "missing-only" or ep_size <= 1:
        return False
    local_required = nproc_per_node or ep_size
    if visible_devices <= 0 or visible_devices >= local_required:
        return False
    if not selected_ops:
        return True
    return all(normalize_op_name(op) == normalize_op_name(DISPATCH_FFN_OP)
               for op in selected_ops)


# =============================================================================
# Report Generation
# =============================================================================
def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format data as Markdown table.

    Args:
        headers: Column header names.
        rows: Table data rows.

    Returns:
        Markdown formatted table string.
    """
    if not rows:
        return "_None_"

    widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
              for i, h in enumerate(headers)]
    fmt = lambda c: "| " + " | ".join(x.ljust(w) for x, w in zip(c, widths)) + " |"

    return "\n".join(
        [fmt(headers), "| " + " | ".join("-" * w for w in widths) + " |"] +
        [fmt(r) for r in rows])


def collect_gaps(results: list[UpdateResult]) -> list[GapRecord]:
    """Collect duration gap hotspots from update results.

    Args:
        results: List of UpdateResult from database updates.

    Returns:
        Sorted list of GapRecord with ratio outside GAP_BOUNDS.
    """
    lower, upper = GAP_BOUNDS

    return sorted(
        [g for r in results for g in r.gaps if g.ratio < lower or g.ratio > upper],
        key=lambda x: (max(x.ratio, 1 / x.ratio), x.diff_us, x.op_type), reverse=True)


def _build_table_rows(results: list[UpdateResult], attr: str, sort_key=None
                      ) -> list[list[str]]:
    """Build table rows for report display.

    Args:
        results: List of UpdateResult.
        attr: Attribute name to extract (missing, deleted, duplicates).
        sort_key: Optional sort key function.

    Returns:
        List of table rows.
    """
    items = (
        sorted([r for r in results if getattr(r, attr)], key=sort_key)
        if sort_key else [r for r in results if getattr(r, attr)])
    rows = []

    for r in items:
        vals = getattr(r, attr)
        if attr == "duplicates":
            ex = "; ".join(f"{s} (x{c})" for s, c in vals[:MAX_EXAMPLE_DISPLAY])
        else:
            ex = "; ".join(vals[:MAX_EXAMPLE_DISPLAY]) or "-"
        rows.append([r.csv_path.stem, str(len(vals)), ex])

    return rows


def print_report(results: list[UpdateResult], gaps: list[GapRecord],
                 status: dict[str, Any] | None, to_file: Path | None = None
                 ) -> tuple[Path, Path] | None:
    """Print update summary and write report files.

    Args:
        results: List of UpdateResult from database updates.
        gaps: List of GapRecord hotspots.
        status: Operator execution status dict, or None.
        to_file: If set, write report files to this directory.

    Returns:
        Tuple of (report_path, gap_csv_path) if to_file set, else None.
    """
    # Build tables
    update_rows = [
        [r.csv_path.stem, str(r.updated), str(r.added), str(len(r.deleted)),
         str(r.unchanged), "; ".join(r.missing[:MAX_EXAMPLE_DISPLAY]) or "-"]
        for r in sorted(results, key=lambda x: (-x.added, -x.updated, x.csv_path.name))]
    missing_rows = _build_table_rows(
        results, "missing", lambda x: (-len(x.missing), x.csv_path.name))
    deleted_rows = _build_table_rows(
        results, "deleted", lambda x: (-len(x.deleted), x.csv_path.name))
    duplicates_rows = _build_table_rows(
        results, "duplicates", lambda x: (-len(x.duplicates), x.csv_path.name))
    gap_rows = [
        [g.op_type, f"{g.mb_us:.6f}", f"{g.prof_us:.6f}",
         f"{g.diff_us:.6f}", f"{g.ratio:.2f}x", g.signature]
        for g in gaps[:MAX_GAP_DISPLAY]]

    lower, upper = GAP_BOUNDS
    overview = [
        ["CSV files touched", str(len(results))],
        ["Rows updated", str(sum(r.updated for r in results))],
        ["Rows added", str(sum(r.added for r in results))],
        ["Rows deleted", str(sum(len(r.deleted) for r in results))],
        ["Rows unchanged", str(sum(r.unchanged for r in results))],
        ["Duplicate signatures", str(sum(len(r.duplicates) for r in results))],
        ["Missing shapes", str(sum(len(r.missing) for r in results))],
        ["Hotspots", str(len(gaps))],
        ["Gap threshold", f"microbench/profiling not in [{lower:.2f}x, {upper:.2f}x]"]]

    # Build report content (shared for console and file)
    lines = ["# Profile Update Report"]
    if status:
        lines.append(f"\n## Operator Execution Status\n"
                     f"- Success: {len(status.get('success', []))}\n"
                     f"- Failed: {len(status.get('failed', []))}\n"
                     f"- Skipped: {len(status.get('skipped', []))}")
        for item in status.get("failed", []):
            lines.append(f"  - FAILED: {item['op']}: {item['reason']}")

    lines.append("\n## Overview\n" + md_table(["Metric", "Value"], overview))
    lines.append("\n## Update Summary\n" + md_table(
        ["Operator", "Updated", "Added", "Deleted", "Unchanged", "Missing Samples"],
        update_rows))
    lines.append("\n## Missing Shapes\n" + md_table(
        ["Operator", "Count", "Examples"], missing_rows))
    lines.append("\n## Deleted Empty Rows\n" + md_table(
        ["Operator", "Count", "Examples"], deleted_rows))
    lines.append("\n## Duplicate Signatures\n" + md_table(
        ["Operator", "Count", "Examples"], duplicates_rows))
    lines.append("\n## Duration Gap Hotspots\n" + md_table(
        ["Operator", "MicroBench(us)", "Profiling(us)", "Abs Diff(us)", "MB/Profile",
         "Shape"], gap_rows))
    if status and status.get("failed"):
        lines.append("\n## Failures\n> [!WARNING]\n> Operators that failed.\n\n" +
                     md_table(["Operator", "Reason"],
                              [[x["op"], x["reason"]] for x in status["failed"]]))
    report_content = "\n".join(lines) + "\n"

    # Print to console
    print(report_content)

    if not to_file:
        return None

    # Write report files
    report_dir = to_file / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Write gap CSV
    gap_csv = report_dir / f"duration_gap_hotspots_full_{ts}.csv"

    with gap_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["Operator", "CSV Name", "MicroBench(us)", "Profiling(us)",
                           "Abs Diff(us)", "MB/Profile", "Shape"])
        w.writeheader()

        for g in gaps:
            w.writerow({
                "Operator": g.op_type, "CSV Name": g.csv_name,
                "MicroBench(us)": f"{g.mb_us:.6f}", "Profiling(us)": f"{g.prof_us:.6f}",
                "Abs Diff(us)": f"{g.diff_us:.6f}", "MB/Profile": f"{g.ratio:.6f}",
                "Shape": g.signature})

    # Write report markdown
    report_path = report_dir / f"profile_update_report_{ts}.md"
    with report_path.open("w", encoding="utf-8", newline="") as f:
        f.write(report_content)
        f.write(f"Full hotspot CSV: {gap_csv.name}\n")

    return report_path, gap_csv


def main() -> None:
    """Main entry point for profile replay and database update.

    Parses CLI arguments, runs profiling, aggregates results, and updates
    the database CSV files with new microbench duration data.
    """
    args = build_argparser().parse_args()
    selected_ops = validate_ops(args.op)
    target_dir = get_target_data_dir(
        args.device, args.vllm_version, database_path=args.database_path,
        torch_version=args.torch_version, cann_version=args.cann_version)

    # Early exit if all CSVs already have usable durations
    if not args.prof_path and args.update_mode == "missing-only":
        csv_paths = (
            [p for op in selected_ops for p in sorted(target_dir.rglob(f"{op}.csv"))]
            if selected_ops else
            sorted(p for p in target_dir.glob("*.csv")
                   if normalize_op_name(p.stem) in set(list_ops())))

        if csv_paths and all(
                csv_has_complete_microbench(load_csv_rows(p)[1]) for p in csv_paths):
            for p in csv_paths:
                print(f"[SKIP] {p} already has usable durations.")
            print(
                "[SUMMARY] All target CSV files already have usable replay durations.")
            return

    profiling_ops = selected_ops
    visible_devices = get_visible_npu_count()
    if should_skip_dispatch_ffn_msprof(
        selected_ops,
        ep_size=args.dispatch_ffn_combine_ep_size,
        nproc_per_node=args.dispatch_ffn_combine_nproc_per_node,
        visible_devices=visible_devices,
        update_mode=args.update_mode,
        has_prof_path=bool(args.prof_path),
    ):
        skip_ops = selected_ops or [DISPATCH_FFN_OP]
        csv_paths = [
            p for op in skip_ops for p in sorted(target_dir.rglob(f"{op}.csv"))
        ]
        skipped_rows = sum(
            1 for p in csv_paths for row in load_csv_rows(p)[1]
            if not row_has_valid_duration(row)
        )
        print(
            f"[SKIP] {DISPATCH_FFN_OP} requires ep-size "
            f"{args.dispatch_ffn_combine_nproc_per_node or args.dispatch_ffn_combine_ep_size} "
            "local rank(s), but only "
            f"{visible_devices} visible NPU device(s) are available."
        )
        print(
            f"[SUMMARY] {DISPATCH_FFN_OP}: skipped {skipped_rows} row(s) "
            "because ep-size exceeds visible NPU count in missing-only mode."
        )
        if selected_ops is None:
            profiling_ops = [
                op for op in list_ops()
                if normalize_op_name(op) != normalize_op_name(DISPATCH_FFN_OP)
            ]
            if not profiling_ops:
                return
            print(f"[SKIP] Continuing full run without {DISPATCH_FFN_OP}.")
        else:
            profiling_ops = selected_ops
            results = update_db(
                target_dir, {}, selected_ops, args.update_mode,
                args.prune_empty_duration_rows)
            gaps = collect_gaps(results)
            report_result = print_report(results, gaps, None, target_dir)
            if report_result:
                print(f"\n[REPORT] {report_result[0]}\n[REPORT] {report_result[1]}")
            return
    # Ensure environment and NPU availability
    if not args.prof_path:
        ensure_custom_opp_env(profiling_ops)
        ensure_npu_available()

    # Run profiling and update database
    succeeded = False
    profiler_root = None

    try:
        prof_dirs = {Path(args.prof_path)} if args.prof_path else set()

        if args.prof_path and not Path(args.prof_path).exists():
            raise FileNotFoundError(f"PROF path does not exist: {args.prof_path}")
        elif not args.prof_path:
            profiler_root, prof_dirs = run_msprof(
                target_dir,
                args,
                profiling_ops,
                allow_per_op_fallback=selected_ops is not None,
            )

        aggregated = aggregate_summary(
            find_summary_files(prof_dirs), args.dispatch_ffn_combine_ep_size)

        if selected_ops:
            sel_set = set(selected_ops)
            for op in list(sel_set):
                sel_set.update(PROFILE_OP_ALIASES.get(op, ()))
            aggregated = {
                op: rows for op, rows in aggregated.items()
                if normalize_op_name(op) in sel_set}

        results = update_db(
            target_dir, aggregated, selected_ops, args.update_mode,
            args.prune_empty_duration_rows)
        gaps = collect_gaps(results)
        status = read_status()
        report_result = print_report(results, gaps, status, target_dir)

        if report_result:
            print(f"\n[REPORT] {report_result[0]}\n[REPORT] {report_result[1]}")

        succeeded = True
    finally:
        status_path = OP_REPLAY_DIR / "run_all_op_status.json"
        if status_path.exists():
            try:
                status_path.unlink()
            except OSError:
                pass

        if not args.prof_path and succeeded and profiler_root is not None:
            shutil.rmtree(profiler_root)
            print(f"[CLEAN] removed {profiler_root}")
        elif not args.prof_path and profiler_root is not None:
            print(f"[PRESERVE] profiling data kept for debugging: {profiler_root}")


if __name__ == "__main__":
    main()
