"""Job execution flow. Orchestrates JobManager → runner → capture → persist.

This module implements the ``run_job`` callable injected into JobManager.
The flow: claim job → runner factory → adapter → capture structured result → persist
``result_records`` + transition status.

Per ``contracts/rest-api.md``: the Runner validates internally and raises on invalid
params → the job is marked ``failed`` (no backend field validation).
"""

from __future__ import annotations

import logging
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import TYPE_CHECKING

from models.enums import JobStatus
from services.capture import capture_job, write_case_log_file
from services.repositories import CaseLogRepository, JobRepository, ResultRepository
from services.ranking import assign_optimizer_ranks

if TYPE_CHECKING:  # pragma: no cover - type-checker-only import; TYPE_CHECKING is False at runtime
    from services.job_manager import JobManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_run_job(job_manager: "JobManager") -> "JobManager | None":
    """Factory: build the ``run_job`` callable and wire it into JobManager.

    The returned ``run_job`` function is the core job execution loop:
    1. Claim the job (transition to ``running``)
    2. Resolve the runner adapter via registry
    3. Capture logs via ``capture_job``
    4. Run the adapter (polls cancel_flag, captures progress)
    5. Persist ``result_records`` (adapter returns list[ResultRecord])
    6. Transition job status (``succeeded``/``failed``/``cancelled``)
    """

    def run_job(job) -> None:
        """Execute a job from start to finish (runs in worker pool).

        Owns all status transitions + persistence; exceptions are caught and
        translated to ``failed`` status (never escape the pool). All DB writes
        are serialised through ``job_manager.write_queue`` so concurrent workers
        never contend for the SQLite write lock; reads go direct (WAL readers
        never block).
        """
        wq = job_manager.write_queue

        # Track consecutive write timeouts to detect WriteQueue saturation.
        # If WriteQueue is continuously blocked (e.g. disk IO saturation), all
        # 8 workers could be stuck waiting 30s each → thread pool starvation
        # deadlock. Counter + reduced timeout (10s) + fail-fast after 2
        # consecutive timeouts prevents this.
        write_timeout_count = 0
        MAX_WRITE_TIMEOUTS = 2

        def _w(thunk, *, timeout: float = 10.0):
            """Enqueue *thunk* on the writer thread, block until done, return result.

            Preserves synchronous control-flow — every DB write appears
            synchronous to the caller while SQL runs on the single writer thread.

            Timeout reduced from 30s to 10s to prevent thread pool starvation.
            Retries up to MAX_WRITE_TIMEOUTS times on timeout; after MAX_WRITE_TIMEOUTS
            consecutive failures, raises RuntimeError to fail the job instead of
            leaving it RUNNING forever.

            Idempotent enqueue (#14): the thunk is enqueued EXACTLY ONCE. On
            timeout we re-await the same Future rather than enqueuing a second
            copy — the previous implementation re-enqueued on each retry, and
            the original Future was never cancelled, so the writer thread would
            eventually execute BOTH copies → duplicate records / duplicate
            status transitions / primary-key conflicts. The thunk is wrapped in
            a ``skip``-flag guard: if all retries exhaust, we flip ``skip`` so
            the pending (still-queued) thunk becomes a no-op when the writer
            thread finally picks it up, preventing a late status-transition
            write from racing with the saturation handler's FAILED-mark write.
            """
            nonlocal write_timeout_count
            # Guarded wrapper: lets us neuter the pending thunk if we time out
            # and give up. List (not bare bool) so the closure can mutate it.
            skip = [False]

            def guarded():
                if skip[0]:
                    return None  # writer thread: caller gave up, no-op
                return thunk()

            future = wq.enqueue(guarded)
            attempt = 0
            last_exception = None
            while attempt < MAX_WRITE_TIMEOUTS:
                try:
                    # Re-await the SAME future on each retry — no re-enqueue.
                    # Future.result(timeout=N) is safe to call repeatedly; it
                    # just blocks the caller, never re-runs the thunk.
                    result = future.result(timeout=timeout)
                    write_timeout_count = 0  # reset on success
                    return result
                except FuturesTimeoutError as e:
                    last_exception = e
                    write_timeout_count += 1
                    attempt += 1
                    if attempt < MAX_WRITE_TIMEOUTS:
                        # Log the retry attempt
                        logger.warning(
                            "Job %s: write timeout (attempt %d/%d, pending=%d), retrying...",
                            job.id,
                            write_timeout_count,
                            MAX_WRITE_TIMEOUTS,
                            wq.pending,
                        )
                    # If this was the last attempt, let the loop exit naturally
                    # and raise RuntimeError below

            # Loop exited normally (attempt >= MAX_WRITE_TIMEOUTS)
            # WriteQueue is saturated — fail fast instead of blocking all workers.
            # Neuter the pending thunk so its late execution is a no-op (the
            # caller's outer handler is about to enqueue a FAILED-mark write;
            # the pending thunk must NOT overwrite that with a stale status).
            skip[0] = True
            raise RuntimeError(
                f"WriteQueue saturation detected ({write_timeout_count} consecutive "
                f"timeouts, pending={wq.pending}). Failing job to prevent thread pool "
                "starvation deadlock."
            ) from last_exception

        job_repo = JobRepository()
        result_repo = ResultRepository()
        cancel_flag = job_manager.cancel_flag(job.id)

        try:
            # 1. Claim the job (transition to running)
            logger.info(f"Starting job {job.id} (module: {job.module_id})")
            _w(lambda: job_repo.update(job.id, status=JobStatus.RUNNING, started_at=_utcnow()))

            # 1b. Phase C cache check: if an identical (module_id, params) run
            # already succeeded, reuse its result records + CLI log verbatim —
            # skip the worker entirely.
            from services.params_hash import compute_params_hash

            params_hash = compute_params_hash(job.module_id, job.form_schema_version, job.params)
            cached = job_repo.find_succeeded_by_params_hash(job.module_id, params_hash)
            if cached is not None:
                n_cloned = _w(lambda: result_repo.clone_records(cached.id, job.id))
                # Chrome trace: Phase C reuses records + log but NOT the on-disk
                # trace artifacts. When chrome_trace was enabled, copy the source
                # job's trace files too (seq is preserved by clone_records, so
                # case_{seq}.json names line up 1:1) — otherwise the result page
                # would hide the download table for a cache-hit job.
                if job.params.get("chrome_trace") is True:
                    from services.trace_store import copy_all_traces

                    copy_all_traces(cached.id, job.id)
                _w(
                    lambda: job_repo.update(
                        job.id,
                        status=JobStatus.SUCCEEDED,
                        params_hash=params_hash,
                        log_text=cached.log_text,
                        completed_at=_utcnow(),
                    )
                )
                logger.info(
                    f"Job {job.id}: cache hit — reused {n_cloned} record(s) + log "
                    f"from job {cached.id} (params_hash={params_hash[:12]}…)"
                )
                return

            # 1c. Case-level dedup: collect the case_hashes already succeeded for
            # this module so the worker can SKIP those cases (the main process
            # clones their records after the run instead of re-running them).
            cached_case_hashes = result_repo.succeeded_case_hashes_for_module(job.module_id)
            logger.info(
                f"Job {job.id}: case-dedup cached_hashes={len(cached_case_hashes)} "
                f"for module={job.module_id} version={job.form_schema_version}"
            )

            # 2. Resolve the runner adapter
            from runners.registry import create_runner

            try:
                runner = create_runner(job.module_id)
            except Exception as e:
                _w(
                    lambda e=str(e): job_repo.update(
                        job.id,
                        status=JobStatus.FAILED,
                        error="Runner instantiation failed",
                        error_detail=e,
                        completed_at=_utcnow(),
                    )
                )
                logger.error(f"Job {job.id}: runner creation failed: {e}")
                return

            def on_progress(progress: int | None, text: str | None) -> None:
                """Update job progress (called by runner adapter)."""
                if progress is not None:
                    _w(lambda: job_repo.update(job.id, progress=progress, progress_text=text or ""))
                else:
                    _w(lambda: job_repo.update(job.id, progress_text=text or ""))

            # 3 + 4. Capture logs + run the adapter inside the capture context.
            # No per-module lock: runner construction is thread-safe (atomic
            # imports, read-only device-registry access) and jobs may run
            # concurrently (ThreadPoolExecutor in JobManager).
            try:
                # Warm the sim stack in ONE thread before any concurrent run():
                # adapters import tensor_cast/transformers lazily, and parallel
                # lazy imports race on half-initialized modules.
                from services.sim_warmup import ensure_sim_stack_warmed

                ensure_sim_stack_warmed()
                with capture_job(job.id) as ring:
                    logger.info(f"Job {job.id} params={job.params}")
                    records, skipped_hashes = runner.run(
                        job.params,
                        on_progress=on_progress,
                        cancel_flag=cancel_flag,
                        cached_hashes=cached_case_hashes,
                        form_schema_version=job.form_schema_version,
                        job_id=job.id,
                    )
                    logger.info(
                        f"Job {job.id}: case-dedup worker returned records={len(records)} skipped={len(skipped_hashes)}"
                    )
                    # Reused cases didn't run, so no CLI table was emitted for them.
                    # Pull their records now (still inside the capture scope) and echo
                    # the case's CLI log (looked up by case_hash in case_logs — no more
                    # regex-splitting of the source job's log) so the job log shows the
                    # reused-case output.
                    cached_records = []
                    case_log_repo = CaseLogRepository()
                    for ch in skipped_hashes:
                        recs = result_repo.get_succeeded_records_by_case_hash(ch)
                        cached_records.extend(recs)
                        if recs:
                            _src_job_id = recs[0].job_id
                            _src_log = case_log_repo.get(ch)
                            print(
                                f"\n===== reused case (hash {ch[:8]}…): replaying CLI output "
                                f"from source job {_src_job_id} =====",
                                flush=True,
                            )
                            print(_src_log or f"  (no source log; {len(recs)} record(s))", flush=True)
            except Exception as e:
                # Runner raised — job failed
                _w(
                    lambda e=str(e): job_repo.update(
                        job.id,
                        status=JobStatus.FAILED,
                        error="Runner execution failed",
                        error_detail=e,
                        completed_at=_utcnow(),
                    )
                )
                logger.error(f"Job {job.id}: runner execution failed: {e}")
                return

            # Check for cooperative cancel
            if cancel_flag and cancel_flag():
                _w(lambda: job_repo.update(job.id, status=JobStatus.CANCELLED, completed_at=_utcnow()))
                logger.info(f"Job {job.id}: cancelled by user")
                return

            # 5. Persist result_records
            # Case-level dedup: pull each skipped (cached) case's records from the
            # most recent succeeded job sharing that case_hash, then merge with the
            # freshly-run records. Fresh ids are assigned to EVERY record (cached
            # rows keep their config/summary/tables but get new id/job_id/seq) so
            # there is no PK collision and the global rank is recomputed across all.
            from uuid import uuid4

            # cached_records was populated + its summary printed inside the capture
            # scope above (so reused cases still appear in the job log even though
            # they didn't run).
            logger.info(
                f"Job {job.id}: case-dedup reused {len(cached_records)} cached records "
                f"from {len(skipped_hashes)} skipped cases; fresh={len(records)}"
            )
            all_records = cached_records + records

            if not all_records:
                # Runner returned nothing and nothing cached (defensive)
                _w(
                    lambda: job_repo.update(
                        job.id,
                        status=JobStatus.FAILED,
                        error="Runner produced no results",
                        completed_at=_utcnow(),
                    )
                )
                logger.error(f"Job {job.id}: runner produced no results")
                return

            # Chrome trace file handling: rename fresh records' {case_hash}.json to
            # case_{seq}.json, copy cached records' traces from source job.
            if job.params.get("chrome_trace") is True:
                from services.trace_store import materialize_traces

                materialize_traces(job.id, all_records, set(skipped_hashes))

            for i, record in enumerate(all_records):
                record.id = uuid4().hex  # fresh id (cached rows had source ids)
                record.job_id = job.id
                record.seq = i

            # Capture-time ranking for throughput_optimizer. This is the
            # sole authoritative rank computation: the global rank across all cases
            # (fresh + reused), persisted as authoritative. The worker does NOT
            # compute rank (it would be overwritten — wasted work); result_view
            # recomputes a per-case rank on demand when rendering multi_case (for
            # the best_config lookup; the persisted value is never mutated).
            if job.module_id == "throughput_optimizer":
                ranks = assign_optimizer_ranks(all_records)
                for record, rank in zip(all_records, ranks):
                    record.rank = rank

            # Persist records (repository builds ResultRecordRow from each entity)
            _w(lambda: result_repo.add_many(all_records))

            # Persist per-case CLI logs (fresh records carry ``case_log``; cached
            # records already have theirs in the table — keyed by case_hash). Mirror
            # each to {case_hash}.log too for streaming/inspection outside the DB.
            case_logs = {r.case_hash: r.case_log for r in all_records if r.case_hash and r.case_log}
            if case_logs:
                _w(lambda: CaseLogRepository().upsert_many(case_logs))
                for ch, content in case_logs.items():
                    write_case_log_file(ch, content)

            # 6. Success — persist params_hash + log_text so a future identical
            # submission reuses this run's results + CLI log (Phase C cache source).
            log_text = "\n".join(ring.get_all()) if ring else None
            _w(
                lambda: job_repo.update(
                    job.id,
                    status=JobStatus.SUCCEEDED,
                    params_hash=params_hash,
                    log_text=log_text,
                    completed_at=_utcnow(),
                )
            )
            logger.info(f"Job {job.id}: succeeded with {len(records)} result record(s)")

        except Exception as e:
            # WriteQueue saturation (detected by _w after MAX_WRITE_TIMEOUTS):
            # mark job FAILED to prevent thread pool starvation deadlock. The
            # saturation error means multiple consecutive writes timed out, so
            # the writer thread is likely blocked (disk IO / lock contention).
            # Attempting one last FAILED write with a short timeout; if it also
            # times out, the job stays RUNNING for the interrupted-sweep.
            if isinstance(e, RuntimeError) and "WriteQueue saturation" in str(e):
                logger.error(
                    "Job %s: WriteQueue saturation detected — marking FAILED to prevent "
                    "thread pool starvation (pending=%d)",
                    job.id,
                    wq.pending,
                )
                try:
                    # Attempt to mark FAILED with a short timeout (5s)
                    wq.enqueue(
                        lambda e=str(e): job_repo.update(
                            job.id,
                            status=JobStatus.FAILED,
                            error="WriteQueue saturation — job failed to prevent system deadlock",
                            error_detail=e,
                            completed_at=_utcnow(),
                        )
                    ).result(timeout=5)
                except Exception:
                    # FAILED write also timed out — leave RUNNING for interrupted-sweep
                    logger.error(
                        "Job %s: FAILED write also timed out; leaving RUNNING for startup sweep",
                        job.id,
                    )
                return
            # Defensive: any other unexpected error → mark job failed.
            logger.exception(f"Job {job.id}: unexpected error in run_job: {e}")
            _w(
                lambda e=str(e): job_repo.update(
                    job.id,
                    status=JobStatus.FAILED,
                    error="Job execution failed unexpectedly",
                    error_detail=e,
                    completed_at=_utcnow(),
                )
            )

    # Wire the run_job into JobManager via the public setter (run_job closes
    # over the manager for cancel_flag/module_lock, so injected post-build).
    job_manager.set_run_job(run_job)
    return job_manager


def _utcnow() -> str:
    """Current UTC timestamp in ISO format."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
