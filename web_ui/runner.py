from __future__ import annotations

import subprocess
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

from .parsers import parse_result

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

    def _run_task(self, task: ExperimentTask) -> ExperimentResult:
        cached = self.store.get_cached_result(task)
        if cached is not None and cached.status != "failed":
            return cached
        proc = subprocess.run(
            task.command, capture_output=True, text=False, cwd=str(PROJECT_ROOT)
        )
        log = _decode_stream(proc.stdout) + _decode_stream(proc.stderr)
        error = (
            None
            if proc.returncode == 0
            else f"Process exited with code {proc.returncode}"
        )
        result = parse_result(
            task, log, "success" if proc.returncode == 0 else "failed", error
        )
        self.store.save_result(result)
        return result

    def run_matrix(
        self, tasks: list[ExperimentTask]
    ) -> Iterable[tuple[int, int, ExperimentResult]]:
        total = len(tasks)
        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as pool:
            futures = {pool.submit(self._run_task, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                yield completed, total, future.result()


def summarize_rows(results: list[ExperimentResult]) -> list[dict]:
    return [res.to_row() for res in results]
