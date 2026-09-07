"""Shared subprocess spawner for runner adapters.

Each adapter's run() spawns runners._worker as a subprocess. The worker's
stdout is streamed into the job log, and the process can be killed for cancel.
The structured result comes from a JSON file, never from parsing logs.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from models.entities import ResultRecord

from runners._cli_command import build_cli_command_string

# Emit at INFO so the reference-CLI-command log line reaches the capture handler
# (the root logger defaults to WARNING; without this the line is dropped at the
# logger level before any handler sees it).
logging.getLogger("runners").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# web/backend/ — the worker is invoked with this cwd so `runners`/`models`/
# `services` and the repo-root `tensor_cast`/`serving_cast` are all importable.
_WEB_BACKEND_DIR = Path(__file__).resolve().parents[1]


def _tree_kill(pid: int) -> None:
    """Kill subprocess and its children (throughput spawns ProcessPoolExecutor)."""
    if os.name == "nt":
        # /T = whole tree, /F = force (same pattern as web/scripts/stop.sh)
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],  # nosec B607
            capture_output=True,
        )
    else:
        import signal

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)  # pylint: disable=no-member
        except OSError:
            # Process/group already gone or no permission — best-effort tree kill.
            pass


def _open_stdout_at(stdout_path: str, offset: int):
    """Reopen worker's stdout file for reading at offset.

    Args:
        stdout_path: Path to stdout file.
        offset: Byte offset to seek to.

    Returns:
        Open binary handle or None if file can't be opened.
    """
    try:
        fh = open(stdout_path, "rb")  # noqa: SIM115  # caller owns the handle
    except (OSError, IOError):
        return None
    if offset:
        try:
            fh.seek(offset)
        except (OSError, IOError):
            try:
                fh.close()
            except OSError:
                pass
            return None
    return fh


def _safe_read_all(fh) -> bytes:
    """Read all remaining bytes from ``fh``; return b"" on a transient error."""
    try:
        return fh.read() or b""
    except (OSError, IOError):
        return b""


def _stream_and_watch(
    proc: subprocess.Popen,
    stdout_path: str,
    cancel_flag: Callable[[], bool] | None,
) -> bool:
    """Tail worker's stdout file, writing to sys.stdout (job log).

    A watcher thread polls cancel_flag and tree-kills on cancel. Uses a temp
    file (not PIPE) to avoid deadlock with ProcessPoolExecutor children.

    Args:
        proc: Subprocess to watch.
        stdout_path: Path to stdout file.
        cancel_flag: Callable returning True if cancel requested.

    Returns:
        True if cancel was requested.
    """
    import threading
    import time

    cancelled = threading.Event()

    def cancel_watcher() -> None:
        if cancel_flag is None:
            return
        while proc.poll() is None:  # process still alive
            if cancel_flag():
                _tree_kill(proc.pid)
                cancelled.set()
                return
            time.sleep(0.3)

    watcher = threading.Thread(target=cancel_watcher, daemon=True)
    watcher.start()

    total_bytes = 0
    post_exit_read_errors = 0
    f = _open_stdout_at(stdout_path, 0)
    try:
        while True:
            if cancelled.is_set():
                break
            if f is None:
                # File not (yet) openable. Stop only once the worker is gone —
                # otherwise wait and retry (it may just not exist yet).
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                f = _open_stdout_at(stdout_path, total_bytes)
                continue
            try:
                chunk = f.read1(65536)  # available data, no newline wait
            except (OSError, IOError) as e:
                # Transient read error ([Errno 22]). The data is still on disk;
                # reopen a FRESH handle at the last good offset and keep going.
                # Do NOT abandon the stream — that truncated the job log.
                logger.debug("Transient stdout read error at %d bytes (%s); reopening", total_bytes, e)
                try:
                    f.close()
                except OSError:
                    pass
                f = None
                if proc.poll() is not None:
                    # Worker gone — the file is stable, so reads should now
                    # succeed; if they keep failing, give up after a few tries
                    # rather than spin forever (the case_log still has the full
                    # output as a fallback).
                    post_exit_read_errors += 1
                    if post_exit_read_errors > 20:
                        logger.warning(
                            "Giving up reading subprocess stdout at %d bytes after %d post-exit errors: %s",
                            total_bytes,
                            post_exit_read_errors,
                            e,
                        )
                        break
                time.sleep(0.05)
                f = _open_stdout_at(stdout_path, total_bytes)
                continue
            if chunk:
                post_exit_read_errors = 0
                total_bytes += len(chunk)
                # decode with errors="replace" never raises (bad bytes → U+FFFD).
                sys.stdout.write(chunk.decode("utf-8", "replace"))
                sys.stdout.flush()
            else:
                # No new data right now — if the process has fully exited, its
                # stdout is flushed to disk: drain any final bytes then stop.
                # Otherwise poll again shortly.
                if proc.poll() is not None:
                    remaining = _safe_read_all(f)
                    if remaining:
                        total_bytes += len(remaining)
                        sys.stdout.write(remaining.decode("utf-8", "replace"))
                        sys.stdout.flush()
                    break
                time.sleep(0.1)
        logger.info("Streamed %d bytes from subprocess stdout", total_bytes)
    except (OSError, IOError) as e:
        # Defensive only — the per-read handling above absorbs transient errors.
        logger.warning("Error reading subprocess stdout after %d bytes: %s", total_bytes, e)
    finally:
        if f is not None:
            try:
                f.close()
            except OSError:
                pass
    watcher.join(timeout=2)
    return cancelled.is_set()


def _build_popen_kwargs(stdout_fd: int) -> dict[str, Any]:
    """Build Popen kwargs with platform-appropriate process-group setup.

    Args:
        stdout_fd: File descriptor for stdout.

    Returns:
        Kwargs dict for Popen.
    """
    kwargs: dict[str, Any] = {
        "stdout": stdout_fd,
        "stderr": stdout_fd,  # merge -> banner (stderr) lands in the log too
        "cwd": str(_WEB_BACKEND_DIR),
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True  # own process group -> killpg
    return kwargs


def run_module_subprocess(
    module_id: str,
    params: dict[str, Any],
    *,
    job_id: str,
    on_progress: Callable[[int | None, str | None], None] | None = None,
    cancel_flag: Callable[[], bool] | None = None,
    cached_hashes: set[str] | None = None,
    form_schema_version: str | None = None,
    provided: set[str] | None = None,
) -> tuple[list[ResultRecord], list[str]]:
    """Spawn runner subprocess, stream output, tree-kill on cancel.

    Args:
        module_id: Module identifier.
        params: Form params.
        job_id: Job ID.
        on_progress: Progress callback.
        cancel_flag: Cancel flag callable.
        cached_hashes: Cached case hashes for dedup.
        form_schema_version: Schema version.
        provided: Set of explicitly provided field names (for wants_provided validators).

    Returns:
        Tuple of (records, skipped_hashes).
    """
    # Reference command for the job's ORIGINAL params (before any multi-case
    # expansion in the worker). For runners that split into per-case subprocess
    # runs (e.g. throughput_optimizer with multi-device), the ACTUAL per-case
    # CLI commands are logged by the worker itself — see the job log for the
    # per-case "[case i/n] CLI:" lines.
    # Synthesize chrome_trace path if enabled (so the reference command shows the actual path, not <auto>)
    ref_params = dict(params)
    if ref_params.get("chrome-trace-file") is True:
        from runners._multicase import compute_case_hash
        from services.trace_store import legacy_hash_path

        case_hash = compute_case_hash(module_id, form_schema_version, ref_params)
        if case_hash and job_id:
            ref_params["chrome-trace-file"] = str(legacy_hash_path(job_id, case_hash))
    logger.info("CLI (reference, unexpanded): %s", build_cli_command_string(module_id, ref_params))
    logger.info(
        "case-dedup: passing cached_hashes=%d form_schema_version=%r to worker",
        len(cached_hashes or []),
        form_schema_version,
    )
    if on_progress:
        on_progress(None, "Starting CLI subprocess")

    params_fd, params_path = tempfile.mkstemp(suffix=".json", prefix="msm_params_")
    result_fd, result_path = tempfile.mkstemp(suffix=".json", prefix="msm_result_")
    stdout_fd, stdout_path = tempfile.mkstemp(suffix=".log", prefix="msm_stdout_")
    os.close(params_fd)
    os.close(result_fd)
    try:
        # Case-dedup metadata rides along in params.json (the worker pops these
        # out before calling execute; they are NOT form fields).
        params_with_meta = {
            **params,
            "_cached_case_hashes": sorted(cached_hashes or []),
            "_form_schema_version": form_schema_version,
            "_job_id": job_id,
            "_provided": sorted(provided or []),
        }
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params_with_meta, f, ensure_ascii=False)

        popen_kwargs = _build_popen_kwargs(stdout_fd)

        proc = subprocess.Popen(
            [sys.executable, "-m", "runners._worker", module_id, params_path, result_path],
            **popen_kwargs,
        )
        # Close the parent's copy of the stdout fd — the subprocess has its own
        # inherited copy and writes there. Keeping the parent's copy open would
        # prevent detecting when the subprocess is done (for pipes); for a file
        # it's just an fd leak.
        os.close(stdout_fd)
        cancelled = _stream_and_watch(proc, stdout_path, cancel_flag)
        if cancelled:
            _tree_kill(proc.pid)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            return [], []

        proc.wait()
        # On an abnormal worker exit (OOM/segfault/crash) no result.json is
        # written — a direct json.load would raise JSONDecodeError/FileNotFoundError,
        # surfacing upstairs as FAILED but with an error_detail that is a JSON parse
        # error rather than the true root cause. Construct the error explicitly for
        # easier diagnosis (the cancel path already returned above).
        if proc.returncode != 0:
            raise RuntimeError(
                f"worker exited with code {proc.returncode} without producing a result; see job log tail for details"
            )
        # Windows file-locking race: the subprocess may not have fully released
        # the result file handle by the time proc.wait() returns, causing
        # [Errno 22] Invalid argument on open(). Retry a few times with a short
        # delay to let the OS release the lock.
        import time

        last_err = None
        for attempt in range(5):
            try:
                with open(result_path, encoding="utf-8") as f:
                    result = json.load(f)
                break
            except (OSError, IOError) as e:
                last_err = e
                if attempt < 4:
                    time.sleep(0.2 * (attempt + 1))  # 0.2s, 0.4s, 0.6s, 0.8s
                continue
        else:
            raise RuntimeError(f"failed to read result file {result_path} after 5 attempts: {last_err}") from last_err
        records = result.get("records", []) if isinstance(result, dict) else result
        skipped = result.get("skipped", []) if isinstance(result, dict) else []
        logger.info("case-dedup: worker returned records=%d skipped=%d", len(records), len(skipped))
        return (
            [
                ResultRecord(
                    job_id="",
                    seq=0,
                    config=r["config"],
                    summary=r["summary"],
                    tables=r.get("tables", {}),
                    rank=r.get("rank"),
                    case_hash=r.get("case_hash"),
                    case_log=r.get("case_log"),
                )
                for r in records
            ],
            list(skipped),
        )
    finally:
        # stdout_fd: close if still open. On the success path line 305 already
        # closed it (parent's copy handed to the child via Popen), so a second
        # close raises EBADF — expected, log at debug. On any exception between
        # mkstemp (line 280) and line 305 (json.dump / Popen failure / EMFILE),
        # this reclaims the fd; without it a long-running service leaking on
        # every failed spawn would exhaust the process fd table.
        try:
            os.close(stdout_fd)
        except OSError as exc:
            logger.debug("stdout_fd already closed (success path): %s", exc)
        for p in (params_path, result_path, stdout_path):
            try:
                os.remove(p)
            except OSError:
                pass
