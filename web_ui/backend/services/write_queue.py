"""Single-writer serialisation for SQLite (shared-deploy write coalescing).

SQLite WAL permits concurrent readers but only ONE writer at a time. With 8
worker threads all calling ``job_repo.update`` / ``result_repo.add_many`` /
``CaseLogRepository.upsert_many`` from inside ``run_job``, concurrent writes
hit ``database is locked`` under sustained load (especially with the
throughput-optimiser's ProcessPoolExecutor competing for the same DB).

``WriteQueue`` solves this without changing repository signatures: callers
enqueue a thunk (``Callable[[], T]``) and get back a ``Future[T]``. A single
dedicated writer thread drains the queue, executing each thunk in order. Reads
still go direct to SQLite (WAL readers never block).

Usage (worker thread)::

    future = write_queue.enqueue(lambda: job_repo.update(job_id, status=RUNNING))
    result = future.result(timeout=5)  # optional — blocks caller until written

For fire-and-forget writes the caller can discard the future.

The writer thread lives for the process lifetime (started at app boot, stopped
at shutdown). It is NOT a connection pool — the queue enforces ordering, but
each thunk opens its own session via ``session_scope`` (same as today). The
isolation is **serial execution** of writes, not a shared transaction.

WRITE ROUTING (which writes go through the queue):
- **Worker-thread writes** (``run_job`` via ``_w``) — ALWAYS queued (8 concurrent workers).
- **Heavy request-handler writes** — queued via ``job_manager.write_async`` when the
  handler's write may touch many rows or overlap with worker writes.
- **Light single-row handler writes** (``submit``, feedback, telemetry) — DIRECT; WAL ``busy_timeout`` absorbs overlap.
Reads are never queued (WAL readers never block).
"""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import Future

logger = logging.getLogger(__name__)


class WriteQueue:
    """Single-writer serialisation queue backed by a dedicated thread.

    Enqueued callables execute sequentially in the writer thread. The caller
    gets a ``Future[T]`` that resolves when the write completes (or raises if
    the thunk raises).
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[Future, object]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Guards the _stop check+put in enqueue against a concurrent shutdown so
        # no item can slip in AFTER shutdown set _stop but BEFORE enqueue's put
        # (which would orphan its future — the writer thread is already exiting).
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Launch the writer thread (idempotent — no-op if already running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._drain, name="write-queue", daemon=True)
        self._thread.start()
        logger.info("WriteQueue writer thread started")

    def shutdown(self, *, wait: bool = True) -> None:
        """Signal the writer thread to stop and (optionally) join it.

        Any queued-but-not-yet-started items are CANCELLED — their futures get a
        ``RuntimeError("WriteQueue shutting down; write cancelled")`` — so a
        caller waiting on ``.result(timeout=…)`` fails fast instead of hanging
        for the full timeout. The caller should drain all workers BEFORE shutting
        down the write queue so in-flight jobs can persist their final state.
        Idempotent.
        """
        with self._lock:
            if self._stop.is_set():
                return
            self._stop.set()
        # Wake the writer thread so it observes the stop event.
        self._queue.put((_sentinel_future, _sentinel_value))
        if wait and self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)
            logger.info("WriteQueue writer thread joined")
        # Cancel any futures the writer thread never reached (put after the
        # sentinel, or simply not drained before exit) so no caller hangs.
        cancelled = 0
        while True:
            try:
                fut, _item = self._queue.get_nowait()
            except queue.Empty:
                break
            if fut is not _sentinel_future and not fut.done():
                fut.set_exception(RuntimeError("WriteQueue shutting down; write cancelled"))
                cancelled += 1
        if cancelled:
            logger.info("WriteQueue cancelled %d pending write(s) at shutdown", cancelled)

    # -- enqueue -------------------------------------------------------------

    def enqueue(self, thunk) -> "Future":
        """Enqueue ``thunk()`` to run in the writer thread.

        Returns a ``Future`` that resolves to the thunk's return value, or
        raises if the thunk raises. The caller can ``.result(timeout=…)`` to
        block until the write completes, or discard the future for fire-and-
        forget writes.
        """
        future: Future = Future()
        # Atomic check+put under _lock: a concurrent shutdown cannot set _stop
        # between our check and our put, so we never enqueue an item whose future
        # would be orphaned by a simultaneously-exiting writer thread.
        with self._lock:
            if self._stop.is_set():
                future.set_exception(RuntimeError("WriteQueue is stopped; write rejected"))
                return future
            self._queue.put((future, thunk))
        return future

    # -- internals -----------------------------------------------------------

    def _drain(self) -> None:
        """Writer-thread body: dequeue + execute until stopped."""
        while not self._stop.is_set():
            try:
                future, thunk = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if future is _sentinel_future:
                # Wake-up sentinel from shutdown; loop will see _stop and exit.
                continue
            try:
                result = thunk()
            except Exception as exc:
                # The thunk raised — propagate to the caller via the future.
                # The writer thread MUST NOT die on a single bad write.
                logger.exception("WriteQueue thunk failed")
                future.set_exception(exc)
            else:
                future.set_result(result)

    @property
    def pending(self) -> int:
        """Approximate count of items waiting to be written (for monitoring)."""
        return self._queue.qsize()


# Sentinel objects for the wake-up-on-shutdown path.
_sentinel_future = Future()
_sentinel_value = object()
