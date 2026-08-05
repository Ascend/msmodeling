"""JobManager — the multi-worker async job engine with a FIFO queue.

``max_workers`` (default 8) worker slots run jobs concurrently. The internal queue
is a heapq ordered by (seq, job): jobs run in submission order, FIFO.

The actual run loop (claim -> runner -> capture -> persist -> transition) lives
in ``services/job_runner.py``; this module owns the concurrency
primitives + the submit/cancel entry points. ``run_job`` is pluggable so tests
can inject a fake.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable

from models.entities import Job
from models.enums import JobStatus
from services.repositories import JobRepository
from services.write_queue import WriteQueue

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# Type of the worker callable: takes a Job, runs the simulation, returns None
# (status transitions + result persistence happen inside it via the repos).
RunJobFn = Callable[[Job], None]


class JobManager:
    """Multi-worker async job manager with a FIFO queue."""

    def __init__(
        self,
        job_repository: JobRepository,
        *,
        run_job: RunJobFn | None = None,
        max_workers: int = 8,
    ):
        self._jobs = job_repository
        self._run_job = run_job  # wired post-build via set_run_job() (job_runner.build_run_job)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._max_workers = max_workers
        # In-flight job cap (pending + running): prevent unbounded queue growth
        # on local DoS (malicious/accidental batch-submit). Default 2×max_workers
        # allows some queuing while bounding resource consumption.
        self._max_inflight = max_workers * 2
        self.write_queue = WriteQueue()
        self.write_queue.start()
        # FIFO queue: (seq, job). seq (from itertools.count) is globally unique,
        # so jobs always dequeue in submission order.
        self._pq: list[tuple[int, Job]] = []
        self._pq_lock = threading.Lock()
        self._ctr = itertools.count()
        # module -> Lock (guards runner construction / device registry reads)
        self._module_locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()
        # job_id -> set-once cancel flag (cooperative)
        self._cancel_flags: dict[str, Callable[[], bool]] = {}
        # Shutdown sentinel (#36): once True, submit/_schedule refuse new work
        # so a late submit_async during teardown can't schedule onto a shut-down
        # executor. Set by shutdown(); read under _pq_lock.
        self._shutting_down = False

    def set_run_job(self, run_job: RunJobFn) -> None:
        """Inject the run loop. Called once at startup after the manager
        exists — ``run_job`` closes over this manager (cancel_flag / module_lock),
        so it can't be passed to ``__init__`` without a circular dependency.
        """
        self._run_job = run_job

    # -- locks ---------------------------------------------------------------

    def module_lock(self, module_id: str) -> threading.Lock:
        """Return the (lazily created, per-module) lock guarding runner
        construction / device-registry reads for ``module_id``.
        """
        with self._registry_lock:
            lock = self._module_locks.get(module_id)
            if lock is None:
                lock = threading.Lock()
                self._module_locks[module_id] = lock
            return lock

    # -- cancel flags --------------------------------------------------------

    def _make_cancel_flag(self, job_id: str) -> Callable[[], bool]:
        """Create a thread-safe set-once cancel flag pair for ``job_id``.

        Returns the *poll* callable (``is_cancelled``) and stashes the matching
        *request* callable under ``{job_id}__request`` so the cancel endpoint can
        flip the flag cooperatively (no Runner exposes a hard cancel hook in v1).
        """
        flag = {"requested": False}
        lock = threading.Lock()

        def is_cancelled() -> bool:
            with lock:
                return flag["requested"]

        def request() -> None:
            with lock:
                flag["requested"] = True

        self._cancel_flags[job_id] = is_cancelled
        # stash the requester on the closure for the cancel endpoint
        self._cancel_flags[f"{job_id}__request"] = request  # type: ignore[assignment]
        return is_cancelled

    def cancel_flag(self, job_id: str) -> Callable[[], bool]:
        """Return the poll callable for ``job_id`` (always-false if unknown)."""
        return self._cancel_flags.get(job_id, lambda: False)

    def request_cancel(self, job_id: str) -> bool:
        """Cooperatively request cancellation for ``job_id``.

        Returns ``True`` if the job has a live flag (the in-flight call will
        notice and discard its result), ``False`` if the job is unknown/finished.
        """
        requester = self._cancel_flags.get(f"{job_id}__request")
        if requester is None:
            return False
        requester()  # type: ignore[operator]
        return True

    # -- submit --------------------------------------------------------------

    def _inflight_total(self) -> int:
        """Count pending + running jobs (the "in-flight" total).

        Used by submit guards to enforce the in-flight cap — prevents unbounded
        queue growth from batch-submit (local DoS on multi-user hosts). Reads
        from the repository (two SQL counts), so it's eventually consistent
        with concurrent submits — acceptable for a defensive rate limit.
        """
        return self._jobs.count_jobs(status=JobStatus.PENDING) + self._jobs.count_jobs(status=JobStatus.RUNNING)

    class InflightLimitExceeded(Exception):
        """Raised when submit would exceed the in-flight job cap."""

    def submit(self, job: Job) -> Job:
        """Persist the job as ``pending`` and schedule it on the worker.

        Raises ``InflightLimitExceeded`` if the in-flight total (pending +
        running) would exceed ``_max_inflight`` — defensive rate limit against
        unbounded queue growth.
        """
        if self._inflight_total() >= self._max_inflight:
            raise self.InflightLimitExceeded(f"in-flight job limit exceeded ({self._max_inflight}); retry later")
        job.status = JobStatus.PENDING
        job.created_at = job.created_at or _utcnow_iso()
        self._jobs.add(job)
        self._schedule(job)
        return job

    def submit_many(self, jobs: list[Job]) -> list[Job]:
        """Persist multiple jobs (already persisted via add_many) and schedule
        them all on the worker pool, in submission order.

        Callers do add_many first, then submit_many to enqueue.

        Raises ``InflightLimitExceeded`` if adding all jobs would exceed the
        in-flight cap. Partial submits are NOT rolled back — the caller should
        check the cap before calling if atomic semantics are required.
        """
        current = self._inflight_total()
        if current + len(jobs) > self._max_inflight:
            raise self.InflightLimitExceeded(
                f"in-flight job limit exceeded ({self._max_inflight}, "
                f"current={current}, requested={len(jobs)}); retry later"
            )
        for job in jobs:
            self._make_cancel_flag(job.id)
            self._schedule(job)
        return jobs

    def _schedule(self, job: Job) -> None:
        """Push ``job`` into the FIFO queue and drain one slot."""
        if self._run_job is None:
            # Defensive: run_job is wired at startup; if not yet set, leave the job pending.
            return
        # Refuse to schedule during shutdown (#36): a late submit_async during
        # lifespan teardown would otherwise push onto a shut-down executor and
        # either raise RuntimeError or hang the caller on a never-drained queue.
        with self._pq_lock:
            if self._shutting_down:
                logger.warning(
                    "Job %s: submit refused during shutdown; leaving PENDING for next boot",
                    job.id,
                )
                return
            self._make_cancel_flag(job.id)
            heapq.heappush(self._pq, (next(self._ctr), job))
        self._executor.submit(self._drain_one)

    def _drain_one(self) -> None:
        """Pop the next job (FIFO) from the queue and run it.

        If the queue is empty this is a no-op (idle drain call).
        """
        with self._pq_lock:
            try:
                seq, job = heapq.heappop(self._pq)
            except IndexError:
                return
        try:
            self._run_job(job)  # type: ignore[misc]
        except Exception as e:
            # run_job has its own outer guard, so reaching here means something
            # escaped it. Preserve the traceback (A4) so the real cause isn't
            # hidden behind a generic message, then mark the job failed so it
            # isn't stuck in RUNNING forever. The status write is itself guarded
            # — a DB error here must not kill the worker thread and strand the job.
            logger.exception("run_job escaped its guard for job %s", job.id)
            try:
                # This FAILED write is what keeps the job from being stranded in
                # RUNNING — wait for it (briefly) rather than fire-and-forget, so
                # a queue-stopped/timeout rejection is observable instead of
                # silently dropped at shutdown. (review: medium)
                fut = self.write_queue.enqueue(
                    lambda jid=job.id, e=str(e): self._jobs.update(
                        jid,
                        status=JobStatus.FAILED,
                        error="Job worker raised an unhandled exception",
                        error_detail=e,
                    )
                )
                fut.result(timeout=10)
            except Exception:
                logger.exception("Failed to mark job %s FAILED after run_job escaped", job.id)
        finally:
            # Job is terminal (succeeded/failed/cancelled/interrupted): drop the
            # in-memory cancel flag so subsequent polls don't see a stale
            # ``cancel_requested=True`` (#89). The flag's source of truth is now
            # the worker's state, not the DB.
            self._cancel_flags.pop(job.id, None)
            self._cancel_flags.pop(f"{job.id}__request", None)

    # -- async bridge (used by the routers) ----------------------------------

    async def submit_async(self, job: Job) -> Job:
        """Async-friendly ``submit`` — persist + schedule via the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.submit(job))

    async def submit_many_async(self, jobs: list[Job]) -> list[Job]:
        """Async-friendly ``submit_many`` — schedule already-persisted jobs."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.submit_many(jobs))

    async def write_async(self, thunk, *, timeout: float = 30.0):
        """Run a write thunk on the single writer thread — awaitable for handlers.

        Heavy request-handler writes (e.g. batch ``add_many`` of up to 1000 rows)
        route through here so they serialise with worker writes via the WriteQueue
        instead of contending for SQLite's single writer slot. The blocking
        ``enqueue().result()`` runs in the default executor so it never stalls the
        event loop. Light single-row handler writes (submit, feedback) may still
        go direct — their write-lock hold time is negligible.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.write_queue.enqueue(thunk).result(timeout=timeout))

    def shutdown(self, *, wait: bool = True, worker_timeout: float = 30.0) -> None:
        """Stop the worker pool, then drain the write queue.

        ``wait=True`` (default): block until workers finish OR ``worker_timeout``
        seconds elapse, then drain the write queue so every in-flight job's
        final state persists. The bounded wait prevents a stuck worker from
        hanging server shutdown indefinitely; jobs still running at timeout
        remain ``running`` in the DB and the next boot's startup sweep marks
        them ``interrupted``.

        ``wait=False``: return immediately — only safe in tests. Production
        callers (main.py lifespan) should use ``wait=True``.

        The write queue is ALWAYS drained with ``wait=True`` — writes are
        short SQLite ops and must complete so every in-flight job's final
        status persists, regardless of the worker-pool wait mode.

        Ordering (#36): the previous implementation called both
        ``executor.shutdown(wait=False)`` and ``write_queue.shutdown(wait=False)``
        from lifespan, so a worker that was mid-write at shutdown time would
        find the write queue already stopped and its final status write would
        be lost. The new sequence is: (1) refuse new submits, (2) wait for
        running workers, (3) drain write queue.
        """
        # 1. Stop accepting new work. _schedule() reads this under _pq_lock.
        with self._pq_lock:
            self._shutting_down = True
        # 2. Wait for workers to finish (bounded). executor.shutdown(wait=True)
        # has no timeout parameter; wrap it in a helper thread + Event.wait()
        # so a stuck worker can't block shutdown forever.
        if wait:
            done = threading.Event()

            def _join_pool() -> None:
                try:
                    self._executor.shutdown(wait=True, cancel_futures=False)
                finally:
                    done.set()

            t = threading.Thread(target=_join_pool, name="jm-shutdown", daemon=True)
            t.start()
            if not done.wait(timeout=worker_timeout):
                logger.warning(
                    "JobManager: worker pool shutdown timed out after %.1fs; "
                    "in-flight jobs will be swept as 'interrupted' on next boot",
                    worker_timeout,
                )
        else:
            self._executor.shutdown(wait=False, cancel_futures=False)
        # 3. Always drain the write queue so every in-flight job's final status
        # persists. (If step 2 timed out, any worker still running will find
        # its next write rejected by the stopped queue — the startup sweep on
        # next boot marks those jobs interrupted.)
        self.write_queue.shutdown(wait=True)

    # -- cancel state query (used by routers to surface in-memory state) -----

    def is_cancel_requested(self, job_id: str) -> bool:
        """True iff the in-memory cancel flag for ``job_id`` exists AND is set.

        The cancel flag is held in memory only (not persisted) — it lives for
        the duration of the job's run and is cleaned up in ``_drain_one``'s
        finally block once the job reaches terminal state. The router queries
        this to surface accurate ``cancel_requested`` in GET /jobs/{id} polls
        (#89) — previously the poll response was always False because it read
        from the DB, which never saw the in-memory flag.
        """
        return self.cancel_flag(job_id)()
