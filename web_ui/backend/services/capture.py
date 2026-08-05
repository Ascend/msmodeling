"""Progress + log capture.

Per-job ``logging.Handler`` + ``redirect_stdout`` capturing runner output to
``.msmodeling_ui/logs/{job_id}.log`` and an in-memory ring buffer (for tail
endpoints / error_detail). Handles the Windows console encodings
(utf-8/gb18030/cp936) that the simulation stack emits.

A ``CaptureContext`` is entered around a runner call: stdout/stderr redirection
+ a temporary logging handler attached to the runner's loggers. On exit the
handler is detached and streams restored.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# Logs live next to the SQLite DB (gitignored .msmodeling_ui/).
def msmodeling_ui_dir() -> Path:
    """Get the .msmodeling_ui directory from env or default."""
    return Path(
        os.environ.get(
            "MSMODELING_UI_DIR",
            Path(__file__).resolve().parents[3] / ".msmodeling_ui",
        )
    )


_DEFAULT_LOG_DIR = msmodeling_ui_dir() / "logs"

# Runners log under these names (tensor_cast / serving_cast root loggers).
_RUNNER_LOGGER_NAMES = (
    "tensor_cast",
    "serving_cast",
    "cli",
    "msmodeling",
    "root",
)

# Encodings to try when decoding bytes from stdout (Windows consoles emit
# gb18030/cp936; everything else is utf-8).
_ENCODINGS = ("utf-8", "gb18030", "cp936", "latin-1")


def _decode(chunk: bytes) -> str:
    """Best-effort decode a stdout chunk, trying utf-8 then Windows console
    encodings (gb18030/cp936); falls back to loss-tolerant utf-8 on total failure.
    """
    for enc in _ENCODINGS:
        try:
            return chunk.decode(enc)
        except UnicodeDecodeError:
            continue
    return chunk.decode("utf-8", errors="replace")


class RingBuffer:
    """Bounded in-memory line buffer for tail/error_detail."""

    def __init__(self, capacity: int = 500):
        self._lines: deque[str] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        """Append text, split into lines, into the bounded ring (thread-safe)."""
        if not text:
            return
        with self._lock:
            for line in text.splitlines():
                self._lines.append(line)

    def tail(self, n: int = 200) -> list[str]:
        """Return up to the last ``n`` lines (all lines when ``n`` is falsy)."""
        with self._lock:
            items = list(self._lines)
        return items[-n:] if n else items

    def get_all(self) -> list[str]:
        """Return a snapshot of every buffered line."""
        with self._lock:
            return list(self._lines)


class _CaptureStream(io.TextIOBase):
    """A minimal text stream that fans writes out to multiple sinks."""

    def __init__(self, sinks: list):
        self._sinks = sinks

    def write(self, data: str) -> int:  # type: ignore[override]
        """Fan ``data`` out to every sink; a failing sink is silently skipped."""
        if not data:
            return 0
        for sink in self._sinks:
            try:
                sink.write(data)
            except Exception:
                logger.debug("Failed to write to sink", exc_info=True)
        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        """Flush every sink; a failing sink is silently skipped."""
        for sink in self._sinks:
            try:
                sink.flush()
            except Exception:
                logger.debug("Failed to flush sink", exc_info=True)


class _FileLikeBuffer:
    """Encodes writes to bytes for a binary log file, decoding back for the ring."""

    def __init__(self, file_handle, ring: RingBuffer):
        self._file = file_handle
        self._ring = ring

    def write(self, text: str) -> int:
        """Write text utf-8-encoded to the binary log file AND mirror it to the ring."""
        if text:
            self._file.write(text.encode("utf-8", errors="replace"))
            self._ring.write(text)
        return len(text) if text else 0

    def flush(self) -> None:
        """Flush the underlying file handle (failures swallowed)."""
        try:
            self._file.flush()
        except OSError:
            pass


class _OwnerThreadFilter(logging.Filter):
    """Emit only records produced by the thread that owns this job's capture.

    Handlers sit on SHARED loggers (tensor_cast/serving_cast/...); without this
    filter, concurrent jobs cross-contaminate each other's logs. ``record.thread``
    is the emitting thread id; each job runs in exactly one worker thread.
    """

    def __init__(self, thread_id: int) -> None:
        super().__init__()
        self._thread_id = thread_id

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        return getattr(record, "thread", None) == self._thread_id


# ---------------------------------------------------------------------------
# Concurrency: route each job-thread's stdout/stderr to its own sink instead
# of racing on a single process-global ``sys.stdout``. Installed once at import.
# ---------------------------------------------------------------------------

_capture_state = threading.local()
_REAL_STDOUT, _REAL_STDERR = sys.stdout, sys.stderr


class _ThreadRouter:
    """A ``sys.stdout``/``sys.stderr`` stand-in that writes to the current
    thread's sink when a ``capture_job`` is active, else to the real stream.
    Lets concurrent jobs each capture their own ``print()`` output.
    """

    def __init__(self, real, attr: str) -> None:
        self._real = real
        self._attr = attr

    def _sink(self):
        return getattr(_capture_state, self._attr, None)

    def write(self, data: str) -> int:
        sink = self._sink()
        (sink if sink is not None else self._real).write(data)
        return len(data) if data else 0

    def flush(self) -> None:
        sink = self._sink()
        if sink is not None:
            try:
                sink.flush()
            except Exception:
                logger.debug("Failed to flush stream sink", exc_info=True)
        self._real.flush()

    def __getattr__(self, name: str):
        # Delegate isatty/encoding/fileno/reconfigure/… to the real stream.
        return getattr(self._real, name)


def _install_thread_routers() -> None:
    """Install the per-thread stdout/stderr routers (idempotent).

    Re-installs when sys.stdout has been swapped out from underneath us (e.g.
    pytest's capsys replaces sys.stdout with ``EncodedFile`` AFTER import time).
    The _ThreadRouter's ``_real`` always points at the CURRENT outer stream so
    writes that aren't captured by a case/job still reach the active sink.
    """
    global _REAL_STDOUT, _REAL_STDERR
    if isinstance(sys.stdout, _ThreadRouter):
        # Router is still in place; nothing to do.
        pass
    else:
        # sys.stdout was replaced since import (e.g. pytest capsys). Snapshot
        # the current outer stream so the new router forwards to it.
        _REAL_STDOUT = sys.stdout
        sys.stdout = _ThreadRouter(_REAL_STDOUT, "out")  # type: ignore[assignment]
    if isinstance(sys.stderr, _ThreadRouter):
        pass
    else:
        _REAL_STDERR = sys.stderr
        sys.stderr = _ThreadRouter(_REAL_STDERR, "err")  # type: ignore[assignment]


_install_thread_routers()


class JobLogHandler(logging.Handler):
    """A ``logging.Handler`` that mirrors records into the file + ring buffer."""

    def __init__(self, sink: "_FileLikeBuffer"):
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        """Format the record and write it to the file+ring sink (failures swallowed)."""
        try:
            msg = self.format(record) + "\n"
            self._sink.write(msg)
        except Exception:
            logger.debug("Failed to write log record", exc_info=True)


@contextmanager
def capture_job(job_id: str, log_dir: Path | None = None) -> Iterator[RingBuffer]:
    """Capture stdout/stderr + runner logging for one job into a log file + ring.

    Usage::

        with capture_job(job_id) as ring:
            runner.run(...)   # stdout + logging land in {job_id}.log
    """
    log_dir = log_dir or _DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_id}.log"
    ring = RingBuffer()

    fh = log_path.open("ab")
    file_buffer = _FileLikeBuffer(fh, ring)

    handler = JobLogHandler(file_buffer)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    # Concurrency: only capture records emitted from THIS job's thread.
    handler.addFilter(_OwnerThreadFilter(threading.get_ident()))

    attached_loggers: list[logging.Logger] = []
    for name in _RUNNER_LOGGER_NAMES:
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        attached_loggers.append(logger)

    # Route THIS thread's stdout/stderr to the job's file+ring (thread-local, so
    # concurrent jobs don't clobber the process-global sys.stdout). Routers are
    # installed once at import; this just (de)activates the per-thread sink.
    _install_thread_routers()
    _capture_state.out = file_buffer
    _capture_state.err = file_buffer
    try:
        yield ring
    finally:
        _capture_state.out = None
        _capture_state.err = None
        for logger in attached_loggers:
            logger.removeHandler(handler)
        try:
            fh.flush()
            fh.close()
        except Exception:
            logger.debug("Failed to flush/close log file", exc_info=True)


_LOG_TAIL_BLOCK = 8192
# Ceiling on a single tail read: a pathological log can't force the polling
# endpoint to allocate/decode the whole file. ~4 MiB covers very large tails.
_LOG_TAIL_MAX_BYTES = 4 * 1024 * 1024


def read_log_tail(job_id: str, tail: int = 200, log_dir: Path | None = None) -> str:
    """Read the last N lines of a job's log file (for GET /api/jobs/{id}/log).

    Reads backward from the file end in fixed-size blocks, so a long-running
    job's full log is never loaded into memory — only enough blocks to cover
    ``tail`` lines (bounded by ``_LOG_TAIL_MAX_BYTES``). ``tail <= 0`` returns
    up to ``_LOG_TAIL_MAX_BYTES`` bytes from the end.
    """
    log_dir = log_dir or _DEFAULT_LOG_DIR
    log_path = log_dir / f"{job_id}.log"
    if not log_path.exists():
        return ""

    want = tail if tail and tail > 0 else None
    collected: list[str] = []
    pos = log_path.stat().st_size
    carry = b""  # a partial line straddling the current block's right edge
    read_bytes = 0
    with log_path.open("rb") as fh:
        while pos > 0 and (want is None or len(collected) < want):
            read_size = min(_LOG_TAIL_BLOCK, pos)
            pos -= read_size
            fh.seek(pos)
            # ``carry`` is the partial first-line of the block to the right; join
            # it here so the line spanning the boundary is reassembled.
            parts = (fh.read(read_size) + carry).splitlines(keepends=True)
            read_bytes += read_size
            carry = parts[0]  # may extend further left on the next iteration
            for line in reversed(parts[1:]):
                collected.append(line.decode("utf-8", errors="replace"))
                if want is not None and len(collected) >= want:
                    break
            if read_bytes >= _LOG_TAIL_MAX_BYTES:
                break
    if carry and (want is None or len(collected) < want):
        collected.append(carry.decode("utf-8", errors="replace"))
    collected.reverse()
    return "".join(collected)


# ---------------------------------------------------------------------------
# Per-case log capture (replaces regex-splitting of {job_id}.log)
# ---------------------------------------------------------------------------
# Each case's CLI output is tee'd into an in-memory buffer (alongside the normal
# stdout that flows into the job log), then carried on the result record as
# ``case_log`` to the main process, which persists it to case_logs + a file.

_CASE_LOG_DIR = msmodeling_ui_dir() / "logs" / "cases"


def case_log_path(case_hash: str) -> Path:
    """Filesystem path for a case's standalone log (``.msmodeling_ui/logs/cases/``)."""
    return _CASE_LOG_DIR / f"{case_hash}.log"


def write_case_log_file(case_hash: str, content: str) -> None:
    """Mirror a case log to ``{case_hash}.log`` (redundant on-disk copy for
    streaming / inspection outside the DB).

    I/O failures are NOT swallowed here: they propagate to the caller (the job
    runner's case loop / write queue), where they are recorded at the job
    boundary. Mirroring a case log is best-effort at the system level, not a
    per-call guarantee.
    """
    if not case_hash or content is None:
        return
    _CASE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    case_log_path(case_hash).write_text(content, encoding="utf-8")


def read_case_log_file(case_hash: str) -> str:
    """Read a case's standalone log file (empty string if absent).

    Defense-in-depth against path traversal: ``case_hash`` is validated at the
    router (64 hex chars), and the resolved path is confirmed to stay inside
    ``_CASE_LOG_DIR`` before reading. A transient read failure (e.g.
    permissions) still propagates to the API caller.
    """
    if not case_hash:
        return ""
    path = case_log_path(case_hash)
    # Reject traversal (e.g. "../" or encoded "..%2F" in case_hash): the resolved
    # path must remain inside the case-log directory.
    if not path.resolve().is_relative_to(_CASE_LOG_DIR.resolve()):
        return ""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


class _Tee(io.TextIOBase):
    """Fan ``write``/``flush`` out to multiple text sinks (used in the worker to
    mirror a case's stdout into a buffer while still forwarding to the real
    stdout → job log).
    """

    def __init__(self, sinks: list):
        self._sinks = sinks

    def write(self, data: str) -> int:  # type: ignore[override]
        if not data:
            return 0
        for sink in self._sinks:
            try:
                sink.write(data)
            except Exception:
                logger.debug("Failed to write to sink", exc_info=True)
        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        for sink in self._sinks:
            try:
                sink.flush()
            except Exception:
                logger.debug("Failed to flush sink", exc_info=True)


class _BufferLogHandler(logging.Handler):
    """A logging handler that mirrors records into a text buffer (used to fold a
    case's log records into the same per-case capture as its stdout).
    """

    def __init__(self, sink):
        super().__init__()
        self._sink = sink
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        try:
            self._sink.write(self.format(record) + "\n")
        except Exception:
            logger.debug("Failed to write to buffer", exc_info=True)


@contextmanager
def capture_case_log() -> Iterator[io.StringIO]:
    """Capture ONE case's stdout + logging into a StringIO buffer, while still
    forwarding everything to the current ``sys.stdout`` (→ the job log).

    Used inside a runner's case loop (``_multicase.run_cases`` /
    ``text_generate.execute``) wrapped around ``run_one_case``. The returned
    buffer's text is attached to the record as ``case_log`` and persisted by the
    main process. A temporary root-logger handler mirrors log records into the
    buffer too (the worker's ``basicConfig`` handler holds the original stream
    object, so reassigning ``sys.stdout`` alone would miss logging records).

    Concurrency (#33): the previous implementation replaced the process-global
    ``sys.stdout`` with a tee (``sys.stdout = tee``), which broke the per-thread
    isolation provided by ``_ThreadRouter`` — concurrent cases in other threads
    had their stdout rerouted through THIS case's tee until the context exited.
    The fix updates the THREAD-LOCAL ``_capture_state`` instead: the existing
    ``_ThreadRouter`` (already installed at import) reads
    ``_capture_state.out`` on every write, so swapping it here only affects
    THIS thread's ``print()`` calls. The tee's sinks are ``(prev_out, buf)``:
    ``prev_out`` is the job log's file_buffer (set by the enclosing
    ``capture_job``); if absent (tests, one-off use), the tee writes only to
    ``buf``. The prior sink is restored on exit so nesting / sequential cases
    each see only their own buffer.
    """
    buf = io.StringIO()
    handler = _BufferLogHandler(buf)
    # Concurrency: only capture records emitted from THIS thread. The handler
    # sits on the ROOT logger (shared across all jobs/cases), so without a filter
    # concurrent cases cross-contaminate each other's case_log (mirrors
    # capture_job's _OwnerThreadFilter).
    handler.addFilter(_OwnerThreadFilter(threading.get_ident()))
    root_logger = logging.getLogger()

    # Save the current thread's sink so we can restore it on exit.
    prev_out = getattr(_capture_state, "out", None)
    prev_err = getattr(_capture_state, "err", None)
    # Ensure the per-thread routers are installed (re-install if pytest swapped
    # sys.stdout since import; see _install_thread_routers docstring).
    _install_thread_routers()
    # Fan-out: this thread's stdout goes to (current sink + case buffer).
    # ``prev_out`` is the enclosing capture_job's file_buffer (when present);
    # if no capture_job is active (e.g. in the subprocess worker where there's
    # no capture_job but stdout is piped to the main process), fall back to the
    # real stdout so the output still reaches the job log via the pipe.
    if prev_out is not None:
        out_sinks = [prev_out, buf]
    else:
        out_sinks = [s for s in (_REAL_STDOUT, buf) if s is not None]
    if prev_err is not None:
        err_sinks = [prev_err, buf]
    else:
        err_sinks = [s for s in (_REAL_STDERR, buf) if s is not None]
    _capture_state.out = _Tee(out_sinks)
    _capture_state.err = _Tee(err_sinks)
    root_logger.addHandler(handler)
    try:
        yield buf
    finally:
        _capture_state.out = prev_out
        _capture_state.err = prev_err
        root_logger.removeHandler(handler)
        try:
            buf.flush()
        except Exception:
            logger.debug("Failed to flush case buffer", exc_info=True)
