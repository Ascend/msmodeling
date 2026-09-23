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

"""Multi-device orchestration for the public microbench CLI.

Users select how many local NPUs to use.  Stable case-shard details stay an
internal worker protocol so one public invocation always launches every shard.
"""

from __future__ import annotations

import os
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence, TextIO

try:
    from .merge_shard_results import merge_shard_directories
    from .operator_metadata import supports_case_sharding
except ImportError:
    from merge_shard_results import merge_shard_directories
    from operator_metadata import supports_case_sharding


@dataclass(frozen=True)
class ParallelRunResult:
    """Artifacts produced by one multi-device microbench run."""

    work_root: Path
    merged_snapshot: Path
    device_ids: tuple[int, ...]


def resolve_worker_device_ids(
    num_devices: int,
    visible_device_count: int,
) -> tuple[int, ...]:
    """Select the first ``num_devices`` local IDs visible to this process."""
    if num_devices <= 0:
        raise ValueError("--num-devices must be a positive integer")

    if num_devices > visible_device_count:
        raise ValueError(
            f"requested {num_devices} devices, but only {visible_device_count} Ascend NPU device(s) are available"
        )
    return tuple(range(num_devices))


def build_parallel_worker_command(
    *,
    start_script: Path,
    database_path: Path,
    shard_count: int,
    shard_index: int,
    selected_ops: Sequence[str] | None,
    repeat_count: int,
    update_mode: str,
    fail_fast: bool,
    device: str | None = None,
    vllm_ascend_version: str | None = None,
    torch_version: str | None = None,
    cann_version: str | None = None,
) -> list[str]:
    """Build one internal worker invocation.

    Device and version parameters are forwarded so the worker resolves the
    database and custom-OPP paths identically to the main process, even when
    ``--database-path`` is used.
    """
    command = [
        sys.executable,
        str(start_script),
        "--database-path",
        str(database_path),
    ]
    if device is not None:
        command += ["--device", device]
    if vllm_ascend_version is not None:
        command += ["--vllm-version", vllm_ascend_version]
    if torch_version is not None:
        command += ["--torch-version", torch_version]
    if cann_version is not None:
        command += ["--cann-version", cann_version]
    command += [
        "--repeat-count",
        str(repeat_count),
        "--update-mode",
        update_mode,
        "--case-shard-count",
        str(shard_count),
        "--case-shard-index",
        str(shard_index),
    ]
    if selected_ops:
        command.extend(["--ops", *selected_ops])
    if fail_fast:
        command.append("--fail-fast")
    return command


def partition_worker_ops(
    selected_ops: Sequence[str],
    worker_count: int,
) -> list[list[str]]:
    """Share shard-aware operators and distribute manual adapters once."""
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    assignments = [[] for _ in range(worker_count)]
    next_manual_worker = 0
    for operator_name in selected_ops:
        if supports_case_sharding(operator_name):
            for worker_ops in assignments:
                worker_ops.append(operator_name)
            continue
        assignments[next_manual_worker].append(operator_name)
        next_manual_worker = (next_manual_worker + 1) % worker_count
    return assignments


def _close_logs(
    processes: list[tuple[int, subprocess.Popen, TextIO, Path]],
) -> None:
    for _, _, log_file, _ in processes:
        if not log_file.closed:
            log_file.close()


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Recursively terminate *process* and all of its descendants.

    Workers spawn msprof, run_all_op, and operator sub-processes.  Terminating
    only the direct Popen child leaves those grandchildren as orphans that keep
    holding NPU devices and temp directories.
    """
    if process.poll() is not None:
        return
    if os.name == "nt":
        # Windows: ``taskkill /T`` walks the entire process tree.
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                capture_output=True,
                timeout=15,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            process.kill()
    else:
        # Unix: try process-group termination first, then fall back.
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _terminate_running_processes(
    processes: list[tuple[int, subprocess.Popen, TextIO, Path]],
) -> None:
    """Best-effort cleanup for workers when orchestration cannot continue."""
    for _, process, _, _ in processes:
        if process.poll() is None:
            _terminate_process_tree(process)


MAX_MERGED_SNAPSHOTS = 3


def _prune_old_snapshots(database_dir: Path, keep: int = MAX_MERGED_SNAPSHOTS) -> None:
    """Delete oldest ``{name}_merged_*`` snapshots beyond *keep*."""
    pattern = f"{database_dir.name}_merged_*"
    snapshots = sorted(database_dir.parent.glob(pattern), key=lambda p: p.name, reverse=True)
    for old in snapshots[keep:]:
        shutil.rmtree(old, ignore_errors=True)


def run_parallel_microbench(
    *,
    start_script: Path,
    database_path: Path,
    device_ids: Sequence[int],
    selected_ops: Sequence[str] | None,
    repeat_count: int,
    update_mode: str,
    fail_fast: bool,
    device: str | None = None,
    vllm_ascend_version: str | None = None,
    torch_version: str | None = None,
    cann_version: str | None = None,
    keep_artifacts: bool = False,
) -> ParallelRunResult:
    """Run one isolated worker per device, merge its CSVs, and write them back."""
    resolved_database = database_path.resolve()
    if not resolved_database.is_dir():
        raise FileNotFoundError(f"database directory does not exist: {resolved_database}")
    if not device_ids:
        raise ValueError("parallel microbench requires at least one device")

    work_root = Path(tempfile.mkdtemp(prefix="msmodeling_microbench_shards_"))
    shard_dirs: list[Path] = []
    processes: list[tuple[int, subprocess.Popen, TextIO, Path]] = []
    shard_count = len(device_ids)
    worker_ops = partition_worker_ops(selected_ops or (), shard_count)
    manual_assignments = {
        operator_name: worker_index
        for worker_index, assigned_ops in enumerate(worker_ops)
        for operator_name in assigned_ops
        if not supports_case_sharding(operator_name)
    }

    try:
        for shard_index in range(shard_count):
            shard_dir = work_root / f"shard_{shard_index}"
            shutil.copytree(resolved_database, shard_dir, copy_function=shutil.copy2)
            shard_dirs.append(shard_dir)

        for shard_index, device_id in enumerate(device_ids):
            if not worker_ops[shard_index]:
                print(
                    f"[parallel] worker {shard_index}/{shard_count - 1}: "
                    f"NPU {device_id} has no assigned operator and remains idle"
                )
                continue
            shard_dir = shard_dirs[shard_index]
            log_path = work_root / f"shard_{shard_index}.log"
            log_file = log_path.open("w", encoding="utf-8")
            command = build_parallel_worker_command(
                start_script=start_script,
                database_path=shard_dir,
                shard_count=shard_count,
                shard_index=shard_index,
                selected_ops=worker_ops[shard_index],
                repeat_count=repeat_count,
                update_mode=update_mode,
                fail_fast=fail_fast,
                device=device,
                vllm_ascend_version=vllm_ascend_version,
                torch_version=torch_version,
                cann_version=cann_version,
            )
            environment = os.environ.copy()
            environment["MB_DEVICE_ID"] = str(device_id)
            try:
                process = subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception:
                log_file.close()
                raise
            processes.append((shard_index, process, log_file, log_path))
            print(f"[parallel] worker {shard_index}/{shard_count - 1}: NPU {device_id}, log={log_path}")

        failures: list[str] = []
        for shard_index, process, _, log_path in processes:
            return_code = process.wait()
            if return_code:
                failures.append(f"shard {shard_index} exited with {return_code}; log={log_path}")
        _close_logs(processes)
        if failures:
            raise RuntimeError(
                f"parallel microbench worker failure; intermediate data kept at {work_root}: {'; '.join(failures)}"
            )

        merged_dir = work_root / "merged"
        merge_shard_directories(
            shard_dirs,
            merged_dir,
            case_shard_count=shard_count,
            operator_shard_assignments=manual_assignments,
            operators=selected_ops,
        )
        merged_csvs = sorted(merged_dir.glob("*.csv"))
        if not merged_csvs:
            raise RuntimeError(f"parallel microbench produced no merged CSV files; artifacts kept at {work_root}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        merged_snapshot = resolved_database.parent / f"{resolved_database.name}_merged_{timestamp}"
        shutil.copytree(merged_dir, merged_snapshot)
        _prune_old_snapshots(resolved_database)
        for merged_csv in merged_csvs:
            shutil.copy2(merged_csv, resolved_database / merged_csv.name)

        if not keep_artifacts:
            shutil.rmtree(work_root, ignore_errors=True)
            work_root = merged_snapshot  # point callers at the durable snapshot

        return ParallelRunResult(
            work_root=work_root,
            merged_snapshot=merged_snapshot,
            device_ids=tuple(device_ids),
        )
    except Exception:
        _terminate_running_processes(processes)
        raise
    finally:
        _close_logs(processes)
