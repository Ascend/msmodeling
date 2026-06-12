from __future__ import annotations

import logging
import subprocess
import threading
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

from .parsers import parse_result
from .time_tracker import get_tracker

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .result_store import ResultStore
    from .schemas import ExperimentResult, ExperimentTask

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _decode_stream(data: bytes | None) -> str:
    if not data:
        return ""
    for encoding in ("utf-8", "gb18030", "cp936"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


class ExperimentRunner:
    def __init__(self, store: ResultStore, max_workers: int = 2):
        self.store = store
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self._active_processes: set[subprocess.Popen] = set()
        self._stop_requested = False

    def reset_stop_flag(self) -> None:
        with self._lock:
            self._stop_requested = False

    def stop_all(self) -> int:
        with self._lock:
            self._stop_requested = True
            processes = list(self._active_processes)
        for proc in processes:
            if proc.poll() is not None:
                continue
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            except OSError:
                continue
        return len(processes)

    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested

    def _run_task(self, task: ExperimentTask) -> ExperimentResult:
        if self.stop_requested():
            result = parse_result(
                task,
                "Run cancelled before execution.",
                "failed",
                "Cancelled by user",
            )
            self.store.save_result(result)
            return result

        cached = self.store.get_cached_result(task)
        if cached is not None and cached.status != "failed":
            # Cache hit: don't record time (not actual execution)
            return cached

        cmd_str = " ".join(task.command)
        logger.info(f"Executing command: {cmd_str}")
        print(f"[Executing] {cmd_str}")

        # Get estimate before running (for logging)
        estimate = get_tracker().get_estimate(task)
        if estimate > 10:
            logger.info(f"Estimated time for {task.params.get('model_id', 'unknown')}: ~{estimate:.0f}s")

        start = time.time()
        proc = subprocess.Popen(
            task.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            cwd=str(PROJECT_ROOT),
        )
        with self._lock:
            self._active_processes.add(proc)
        stdout, stderr = proc.communicate()
        with self._lock:
            self._active_processes.discard(proc)
        duration = time.time() - start

        log = _decode_stream(stdout) + _decode_stream(stderr)
        error = (
            None
            if proc.returncode == 0
            else (
                "Cancelled by user"
                if self.stop_requested() and proc.returncode != 0
                else f"Process exited with code {proc.returncode}"
            )
        )
        result = parse_result(
            task,
            log,
            "success" if proc.returncode == 0 else "failed",
            error,
        )
        self.store.save_result(result)

        # Record actual time for future estimates
        get_tracker().record(task, duration)
        logger.info(f"Task completed in {duration:.1f}s (estimate was {estimate:.0f}s)")

        return result

    def run_matrix(self, tasks: list[ExperimentTask]) -> Iterable[tuple[int, int, ExperimentResult]]:
        total = len(tasks)
        self.reset_stop_flag()
        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as pool:
            futures = {pool.submit(self._run_task, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                yield completed, total, future.result()
                if self.stop_requested():
                    break


def summarize_rows(results: list[ExperimentResult]) -> list[dict]:
    return [res.to_row() for res in results]
