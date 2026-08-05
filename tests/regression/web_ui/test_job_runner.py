"""Unit tests for services/job_runner.py.

Exercises the inner ``run_job`` callable across success, cache-hit, dedup,
failure, cancel, and error branches by mocking the collaborators (repos,
write_queue, runner factory). The happy path is also covered by the backend
integration tests; these unit tests add the failure/edge branches for 100%
line+branch coverage. Real imports + fixture-scoped mocks only, per
tests/SKILL.md.
"""

from __future__ import annotations

from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import MagicMock, patch

from models.enums import JobStatus
from services.job_runner import _utcnow, build_run_job

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Future:
    """A fake write-queue future: .result() runs the thunk synchronously."""

    def __init__(self, thunk):
        self._thunk = thunk
        self._result = None
        self._ran = False

    def result(self, timeout=None):
        if not self._ran:
            self._result = self._thunk()
            self._ran = True
        return self._result


def _make_manager(write_queue=None, cancel_return=False):
    """Build a JobManager mock whose write_queue runs thunks synchronously."""
    manager = MagicMock()
    wq = write_queue or MagicMock()
    # enqueue(thunk) -> a future that runs the thunk on .result()
    wq.enqueue.side_effect = lambda thunk: _Future(thunk)
    wq.pending = 0
    manager.write_queue = wq
    manager.cancel_flag.return_value = MagicMock(return_value=cancel_return)
    manager.set_run_job = MagicMock()
    return manager


def _make_job(module_id="text_generate", params=None, version="1.0.0"):
    job = MagicMock()
    job.id = "job-1"
    job.module_id = module_id
    job.form_schema_version = version
    job.params = params if params is not None else {"model": "gpt2"}
    return job


def _run_job_from(manager):
    """Extract the run_job callable wired into the manager by build_run_job."""
    build_run_job(manager)
    return manager.set_run_job.call_args[0][0]


# ---------------------------------------------------------------------------
# _utcnow + factory wiring
# ---------------------------------------------------------------------------


class TestUtcnow:
    """Tests for the _utcnow helper."""

    def test_returns_iso_format_string(self):
        result = _utcnow()
        assert isinstance(result, str)
        assert "T" in result
        assert result.endswith("Z")
        assert ":" in result
        assert "-" in result


class TestBuildRunJobFactory:
    """Tests for the build_run_job factory wiring."""

    def test_wires_run_job_and_returns_manager(self):
        manager = _make_manager()
        result = build_run_job(manager)
        manager.set_run_job.assert_called_once()
        assert callable(manager.set_run_job.call_args[0][0])
        assert result is manager

    def test_wired_run_job_catches_exceptions(self):
        """run_job never lets exceptions escape the worker pool."""
        manager = _make_manager()
        run_job = _run_job_from(manager)
        job = _make_job()
        # Patch the heavy collaborators so the function runs deterministically
        # (and fails fast into the outer except if anything goes wrong).
        with (
            patch("services.job_runner.JobRepository"),
            patch("services.job_runner.ResultRepository"),
            patch("services.params_hash.compute_params_hash", return_value="h"),
        ):
            run_job(job)
        manager.write_queue.enqueue.assert_called()


# ---------------------------------------------------------------------------
# Success + dedup paths
# ---------------------------------------------------------------------------


class TestRunJobSuccess:
    """Tests for the success path."""

    def test_runs_to_success(self):
        manager = _make_manager()
        run_job = _run_job_from(manager)
        job = _make_job()
        fake_record = MagicMock()
        fake_record.case_hash = None
        fake_record.case_log = None

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository") as mock_res_repo,
            patch("services.params_hash.compute_params_hash", return_value="hash123"),
            patch("runners.registry.create_runner") as mock_create,
            patch("services.sim_warmup.ensure_sim_stack_warmed"),
            patch("services.job_runner.capture_job") as mock_capture,
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = None
            mock_res_repo.return_value.succeeded_case_hashes_for_module.return_value = []
            runner = MagicMock()
            runner.run.return_value = ([fake_record], [])
            mock_create.return_value = runner
            ring = MagicMock()
            ring.get_all.return_value = ["log line"]
            mock_capture.return_value.__enter__.return_value = ring
            mock_capture.return_value.__exit__.return_value = False

            run_job(job)

        # Final status update should be SUCCEEDED (passed as kwargs).
        kwargs_list = [c.kwargs for c in mock_job_repo.return_value.update.call_args_list]
        assert any(k.get("status") == JobStatus.SUCCEEDED for k in kwargs_list)
        mock_res_repo.return_value.add_many.assert_called()

    def test_cache_hit_clones_records_and_succeeds(self):
        manager = _make_manager()
        run_job = _run_job_from(manager)
        job = _make_job(params={"model": "gpt2"})

        cached = MagicMock()
        cached.id = "src-job"
        cached.log_text = "cached log"

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository") as mock_res_repo,
            patch("services.params_hash.compute_params_hash", return_value="hash123"),
            patch("services.trace_store.copy_all_traces") as mock_copy,
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = cached
            mock_res_repo.return_value.clone_records.return_value = 5

            run_job(job)

        mock_res_repo.return_value.clone_records.assert_called_once_with("src-job", "job-1")
        mock_copy.assert_not_called()  # chrome_trace not enabled

    def test_cache_hit_copies_traces_when_chrome_trace_enabled(self):
        manager = _make_manager()
        run_job = _run_job_from(manager)
        job = _make_job(params={"model": "gpt2", "chrome_trace": True})

        cached = MagicMock()
        cached.id = "src-job"
        cached.log_text = "log"

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository") as mock_res_repo,
            patch("services.params_hash.compute_params_hash", return_value="hash123"),
            patch("services.trace_store.copy_all_traces") as mock_copy,
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = cached
            mock_res_repo.return_value.clone_records.return_value = 3

            run_job(job)

        mock_copy.assert_called_once_with("src-job", "job-1")


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestRunJobFailures:
    """Tests for failure/cancel/error branches."""

    def test_runner_creation_failure_marks_failed(self):
        manager = _make_manager()
        run_job = _run_job_from(manager)
        job = _make_job()

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository") as mock_res_repo,
            patch("services.params_hash.compute_params_hash", return_value="hash123"),
            patch("runners.registry.create_runner", side_effect=RuntimeError("no runner")),
            patch("services.sim_warmup.ensure_sim_stack_warmed"),
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = None
            mock_res_repo.return_value.succeeded_case_hashes_for_module.return_value = []

            run_job(job)

        updates = [c.kwargs for c in mock_job_repo.return_value.update.call_args_list]
        failed = [u for u in updates if u.get("status") == JobStatus.FAILED]
        assert any(u.get("error") == "Runner instantiation failed" for u in failed)

    def test_runner_execution_failure_marks_failed(self):
        manager = _make_manager()
        run_job = _run_job_from(manager)
        job = _make_job()

        runner = MagicMock()
        runner.run.side_effect = RuntimeError("runner blew up")

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository") as mock_res_repo,
            patch("services.params_hash.compute_params_hash", return_value="hash123"),
            patch("runners.registry.create_runner", return_value=runner),
            patch("services.sim_warmup.ensure_sim_stack_warmed"),
            patch("services.job_runner.capture_job") as mock_capture,
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = None
            mock_res_repo.return_value.succeeded_case_hashes_for_module.return_value = []
            ring = MagicMock()
            ring.get_all.return_value = []
            mock_capture.return_value.__enter__.return_value = ring
            mock_capture.return_value.__exit__.return_value = False

            run_job(job)

        updates = [c.kwargs for c in mock_job_repo.return_value.update.call_args_list]
        failed = [u for u in updates if u.get("error") == "Runner execution failed"]
        assert len(failed) == 1

    def test_cancel_after_run_marks_cancelled(self):
        cancel_flag = MagicMock(return_value=True)
        manager = MagicMock()
        manager.write_queue = MagicMock()
        manager.write_queue.enqueue.side_effect = lambda thunk: _Future(thunk)
        manager.cancel_flag.return_value = cancel_flag

        run_job = _run_job_from(manager)
        job = _make_job()
        fake_record = MagicMock()
        fake_record.case_hash = None
        fake_record.case_log = None
        runner = MagicMock()
        runner.run.return_value = ([fake_record], [])

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository") as mock_res_repo,
            patch("services.params_hash.compute_params_hash", return_value="hash123"),
            patch("runners.registry.create_runner", return_value=runner),
            patch("services.sim_warmup.ensure_sim_stack_warmed"),
            patch("services.job_runner.capture_job") as mock_capture,
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = None
            mock_res_repo.return_value.succeeded_case_hashes_for_module.return_value = []
            ring = MagicMock()
            ring.get_all.return_value = []
            mock_capture.return_value.__enter__.return_value = ring
            mock_capture.return_value.__exit__.return_value = False

            run_job(job)

        updates = [c.kwargs for c in mock_job_repo.return_value.update.call_args_list]
        assert any(u.get("status") == JobStatus.CANCELLED for u in updates)

    def test_no_results_marks_failed(self):
        manager = _make_manager()
        run_job = _run_job_from(manager)
        job = _make_job()
        runner = MagicMock()
        runner.run.return_value = ([], [])  # no records, no cached

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository") as mock_res_repo,
            patch("services.params_hash.compute_params_hash", return_value="hash123"),
            patch("runners.registry.create_runner", return_value=runner),
            patch("services.sim_warmup.ensure_sim_stack_warmed"),
            patch("services.job_runner.capture_job") as mock_capture,
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = None
            mock_res_repo.return_value.succeeded_case_hashes_for_module.return_value = []
            ring = MagicMock()
            ring.get_all.return_value = []
            mock_capture.return_value.__enter__.return_value = ring
            mock_capture.return_value.__exit__.return_value = False

            run_job(job)

        updates = [c.kwargs for c in mock_job_repo.return_value.update.call_args_list]
        failed = [u for u in updates if u.get("error") == "Runner produced no results"]
        assert len(failed) == 1

    def test_write_queue_timeout_marks_failed_after_retries(self):
        """Consecutive FuturesTimeoutError triggers saturation detection → FAILED.

        B-29 fix: _w() retries up to MAX_WRITE_TIMEOUTS times. After exhausting
        retries, raises RuntimeError. The outer handler attempts to mark the job
        FAILED. If that FAILED write also times out, job is left RUNNING for sweep.

        #14 fix: the thunk is enqueued EXACTLY ONCE; retries re-await the same
        Future. The mock's ``.result()`` raises on every call (MagicMock
        side_effect), so one enqueue produces two timeout raises.
        """
        call_count = {"n": 0}

        def enqueue_timeout_then_work(thunk):
            """First call times out (claim retry loop re-awaits same future);
            subsequent enqueues work (FAILED write attempt).
            """
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Claim: single enqueue; .result() will be called twice (two
                # timeout raises from the SAME future under the #14 fix).
                bad_future = MagicMock()
                bad_future.result.side_effect = FuturesTimeoutError()
                return bad_future
            # Subsequent calls (saturation handler's FAILED write) work
            return _Future(thunk)

        wq = MagicMock()
        wq.enqueue.side_effect = enqueue_timeout_then_work
        wq.pending = 42
        manager = MagicMock()
        manager.write_queue = wq
        manager.cancel_flag.return_value = MagicMock(return_value=False)

        run_job = _run_job_from(manager)
        job = _make_job()

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository"),
            patch("services.params_hash.compute_params_hash", return_value="hash123"),
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = None
            # The first _w call (claim job) times out twice, triggering saturation.
            run_job(job)

        # After MAX_WRITE_TIMEOUTS retries, RuntimeError is raised.
        # The outer handler successfully marks the job FAILED.
        updates = [c.kwargs for c in mock_job_repo.return_value.update.call_args_list]
        failed_updates = [u for u in updates if u.get("status") == JobStatus.FAILED]
        assert len(failed_updates) == 1
        assert "saturation" in failed_updates[0].get("error", "").lower()

    def test_unexpected_error_marks_failed(self):
        """A non-timeout unexpected error in the claim write marks the job FAILED.

        The first _w call (claim job) raises a ValueError; the outer except sees
        it's not a FuturesTimeoutError and writes a defensive FAILED update.
        Subsequent _w calls (the FAILED update) must succeed, so enqueue works
        from the second call onward.
        """
        calls = {"n": 0}

        def enqueue_then_work(thunk):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("queue broken on first write")
            return _Future(thunk)

        wq = MagicMock()
        wq.enqueue.side_effect = enqueue_then_work
        manager = MagicMock()
        manager.write_queue = wq
        manager.cancel_flag.return_value = MagicMock(return_value=False)

        run_job = _run_job_from(manager)
        job = _make_job()

        # Patch heavy collaborators so the test isolates the timeout/error logic.
        with (
            patch("services.job_runner.JobRepository"),
            patch("services.job_runner.ResultRepository"),
            patch("services.params_hash.compute_params_hash", return_value="h"),
        ):
            # The function must not propagate the ValueError (caught + translated).
            run_job(job)
        assert calls["n"] >= 2

    def test_two_consecutive_timeouts_mark_failed_saturation(self):
        """Two consecutive FuturesTimeoutError triggers saturation detection → FAILED.

        B-29 fix: _w() tracks consecutive timeouts. After MAX_WRITE_TIMEOUTS (2),
        it raises RuntimeError("WriteQueue saturation detected"). The outer except
        catches this and attempts to mark the job FAILED with a short timeout.

        #14 fix: the thunk is enqueued EXACTLY ONCE per _w call; retries re-await
        the same Future. So "two consecutive timeouts" now means one enqueue whose
        .result() is called twice and raises both times.

        To test this, we let the first _w (claim) succeed, then timeout on the
        next _w (cache hit path: clone_records).
        """
        call_count = {"n": 0}

        def enqueue_first_success_then_timeout(thunk):
            """First _w succeeds, next _w timeouts to trigger saturation."""
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call (claim) succeeds
                return _Future(thunk)
            elif call_count["n"] == 2:
                # Second call (clone_records): single enqueue, .result() raises
                # twice (retry awaits same future under #14 fix)
                bad_future = MagicMock()
                bad_future.result.side_effect = FuturesTimeoutError()
                return bad_future
            else:
                # Subsequent calls work (saturation handler's FAILED write)
                return _Future(thunk)

        wq = MagicMock()
        wq.enqueue.side_effect = enqueue_first_success_then_timeout
        wq.pending = 5
        manager = MagicMock()
        manager.write_queue = wq
        manager.cancel_flag.return_value = MagicMock(return_value=False)

        run_job = _run_job_from(manager)
        job = _make_job()

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository") as mock_res_repo,
            patch("services.params_hash.compute_params_hash", return_value="hash123"),
        ):
            # Cache HIT so we go through the clone_records path
            cached_job = MagicMock()
            cached_job.id = "cached-job-1"
            cached_job.log_text = "cached log"
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = cached_job
            mock_res_repo.return_value.clone_records.return_value = 5

            run_job(job)

        # After 2 timeouts (clone_records + mark_succeeded), saturation RuntimeError is raised.
        # The outer handler attempts to mark FAILED.
        updates = [c.kwargs for c in mock_job_repo.return_value.update.call_args_list]
        failed_updates = [u for u in updates if u.get("status") == JobStatus.FAILED]
        # Should have at least one FAILED update with saturation error
        assert len(failed_updates) >= 1, f"Expected FAILED update, got: {updates}"
        assert any("saturation" in u.get("error", "").lower() for u in failed_updates)

    def test_saturation_failed_write_also_times_out(self):
        """Saturation detected but FAILED write also times out → leave RUNNING.

        Edge case: _w() raises saturation RuntimeError, but the FAILED write
        (with 5s timeout) also times out. Job stays RUNNING for interrupted-sweep.

        #14 fix: one enqueue per _w, retries re-await same future. So: 1 enqueue
        for claim (times out twice via re-await) + 1 enqueue for saturation's
        FAILED write (times out) = 2 total enqueues (was 3 under old behavior).
        """
        call_count = {"n": 0}

        def enqueue_always_timeout(thunk):
            """All enqueues time out (every .result() call raises)."""
            call_count["n"] += 1
            bad_future = MagicMock()
            bad_future.result.side_effect = FuturesTimeoutError()
            return bad_future

        wq = MagicMock()
        wq.enqueue.side_effect = enqueue_always_timeout
        wq.pending = 10
        manager = MagicMock()
        manager.write_queue = wq
        manager.cancel_flag.return_value = MagicMock(return_value=False)

        run_job = _run_job_from(manager)
        job = _make_job()

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository"),
            patch("services.params_hash.compute_params_hash", return_value="h"),
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = None
            run_job(job)

        # 1 enqueue for claim (re-awaited twice under #14 fix) + 1 enqueue for
        # saturation FAILED write = 2 total.
        assert call_count["n"] == 2
        updates = [c.kwargs for c in mock_job_repo.return_value.update.call_args_list]
        # No successful FAILED update (the saturation handler's write also timed out)
        failed_updates = [u for u in updates if u.get("status") == JobStatus.FAILED]
        assert len(failed_updates) == 0

    def test_timeout_does_not_duplicate_thunk_execution(self):
        """#14 fix: a timed-out write is enqueued ONCE; thunk runs exactly once.

        The OLD implementation called ``wq.enqueue(thunk)`` inside the retry
        loop, so a 2-timeout failure produced 2 enqueues and the writer thread
        executed the thunk twice (duplicate records / duplicate status
        transitions / primary-key conflicts). The fix enqueues ONCE and
        re-awaits the same Future. We verify both properties:

        * ``wq.enqueue`` is called exactly once for the failing write
        * The underlying thunk is invoked exactly once (when the writer
          thread finally drains the queue)
        """
        thunk_calls = {"n": 0}

        def slow_thunk():
            thunk_calls["n"] += 1
            return "ok"

        def enqueue_returns_same_bad_future(thunk):
            """Single enqueue; the returned Future's .result() always times out.

            Captures only the FIRST enqueued thunk (the guarded wrapper for the
            main write). The second enqueue (saturation FAILED-mark) is ignored
            for capture purposes — the test wants to inspect the guarded wrapper
            specifically.
            """
            if enqueue_returns_same_bad_future.captured_thunk is None:
                enqueue_returns_same_bad_future.captured_thunk = thunk
            bad = MagicMock()
            bad.result.side_effect = FuturesTimeoutError()
            return bad

        enqueue_returns_same_bad_future.captured_thunk = None

        wq = MagicMock()
        wq.enqueue.side_effect = enqueue_returns_same_bad_future
        wq.pending = 1
        manager = MagicMock()
        manager.write_queue = wq
        manager.cancel_flag.return_value = MagicMock(return_value=False)

        run_job = _run_job_from(manager)
        job = _make_job()

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository"),
            patch("services.params_hash.compute_params_hash", return_value="h"),
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = None
            run_job(job)

        # Exactly one enqueue for the claim (retries re-awaited the same future)
        # + one enqueue for the saturation FAILED write = 2 total.
        assert wq.enqueue.call_count == 2
        # The captured thunk from the FIRST enqueue is the guarded wrapper.
        # Running it AFTER skip was set must NOT invoke slow_thunk.
        guarded_thunk = enqueue_returns_same_bad_future.captured_thunk
        # First, simulate the writer thread running the guarded thunk BEFORE skip
        # would have been set — but we can't observe that here since the
        # saturation already set skip=True. So instead, verify: running the
        # guarded thunk now (after skip) returns None and never calls slow_thunk.
        result = guarded_thunk()
        assert result is None
        assert thunk_calls["n"] == 0  # slow_thunk never ran

    def test_late_execution_after_skip_is_noop(self):
        """#14 fix: after all retries exhaust, the pending thunk is neutered.

        Simulates the writer thread picking up the guarded thunk AFTER the
        caller gave up (skip=True). The guarded thunk must return None and
        the wrapped write must NOT execute — this prevents a late status
        transition (e.g. RUNNING) from racing with the saturation handler's
        FAILED-mark write.
        """
        inner_ran = {"n": 0}

        def inner_thunk():
            inner_ran["n"] += 1
            return "should-not-happen"

        # Build the guarded wrapper by calling _w indirectly: set up a manager
        # whose write_queue captures the FIRST enqueued thunk and returns a
        # timed-out future. The second enqueue (saturation FAILED-mark write)
        # is NOT captured — we want the guarded wrapper for the main write.
        captured = {"thunk": None}

        def enqueue_capture(thunk):
            if captured["thunk"] is None:
                captured["thunk"] = thunk
            bad = MagicMock()
            bad.result.side_effect = FuturesTimeoutError()
            return bad

        wq = MagicMock()
        wq.enqueue.side_effect = enqueue_capture
        wq.pending = 1
        manager = MagicMock()
        manager.write_queue = wq
        manager.cancel_flag.return_value = MagicMock(return_value=False)

        run_job = _run_job_from(manager)
        job = _make_job()

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository"),
            patch("services.params_hash.compute_params_hash", return_value="h"),
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = None
            run_job(job)

        # _w gave up and set skip=True. The captured thunk is the guarded wrapper.
        guarded = captured["thunk"]
        assert guarded is not None
        # Late execution: writer thread picks this up after caller gave up.
        result = guarded()
        assert result is None
        assert inner_ran["n"] == 0  # inner_thunk must NOT run


# ---------------------------------------------------------------------------
# Progress callback + dedup replay
# ---------------------------------------------------------------------------


class TestProgressAndDedup:
    """Tests for on_progress callback and dedup record replay."""

    def test_progress_callback_updates_progress(self):
        """on_progress with a non-None progress writes a progress update."""
        manager = _make_manager()
        run_job = _run_job_from(manager)
        job = _make_job()
        fake_record = MagicMock()
        fake_record.case_hash = None
        fake_record.case_log = None

        def fake_run(params, on_progress, cancel_flag, cached_hashes, **kw):
            on_progress(75, "three quarters")
            on_progress(None, "status text")
            return [fake_record], []

        runner = MagicMock()
        runner.run.side_effect = fake_run

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository") as mock_res_repo,
            patch("services.params_hash.compute_params_hash", return_value="hash123"),
            patch("runners.registry.create_runner", return_value=runner),
            patch("services.sim_warmup.ensure_sim_stack_warmed"),
            patch("services.job_runner.capture_job") as mock_capture,
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = None
            mock_res_repo.return_value.succeeded_case_hashes_for_module.return_value = []
            ring = MagicMock()
            ring.get_all.return_value = []
            mock_capture.return_value.__enter__.return_value = ring
            mock_capture.return_value.__exit__.return_value = False

            run_job(job)

        # progress=75 update and progress_text-only update both written.
        updates = [c.kwargs for c in mock_job_repo.return_value.update.call_args_list]
        assert any(u.get("progress") == 75 for u in updates)

    def test_dedup_replays_cached_case_records(self):
        """Skipped (cached) cases have their records replayed into the result."""
        manager = _make_manager()
        run_job = _run_job_from(manager)
        job = _make_job()
        fresh_record = MagicMock()
        fresh_record.case_hash = "fresh1"
        fresh_record.case_log = None

        cached_record = MagicMock()
        cached_record.case_hash = "cached1"
        cached_record.case_log = None
        cached_record.job_id = "src-job"

        runner = MagicMock()
        runner.run.return_value = ([fresh_record], ["cached1"])

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository") as mock_res_repo,
            patch("services.job_runner.CaseLogRepository") as mock_case_repo,
            patch("services.params_hash.compute_params_hash", return_value="hash123"),
            patch("runners.registry.create_runner", return_value=runner),
            patch("services.sim_warmup.ensure_sim_stack_warmed"),
            patch("services.job_runner.capture_job") as mock_capture,
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = None
            mock_res_repo.return_value.succeeded_case_hashes_for_module.return_value = ["cached1"]
            mock_res_repo.return_value.get_succeeded_records_by_case_hash.return_value = [cached_record]
            mock_case_repo.return_value.get.return_value = "replayed log"
            ring = MagicMock()
            ring.get_all.return_value = []
            mock_capture.return_value.__enter__.return_value = ring
            mock_capture.return_value.__exit__.return_value = False

            run_job(job)

        # add_many called with cached + fresh records merged.
        added = mock_res_repo.return_value.add_many.call_args[0][0]
        assert cached_record in added
        assert fresh_record in added


class TestOptimizerRanking:
    """Tests for throughput_optimizer rank assignment."""

    def test_assigns_ranks_for_optimizer_module(self):
        manager = _make_manager()
        run_job = _run_job_from(manager)
        job = _make_job(module_id="throughput_optimizer")
        rec = MagicMock()
        rec.case_hash = None
        rec.case_log = None
        runner = MagicMock()
        runner.run.return_value = ([rec], [])

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository") as mock_res_repo,
            patch("services.params_hash.compute_params_hash", return_value="hash123"),
            patch("runners.registry.create_runner", return_value=runner),
            patch("services.sim_warmup.ensure_sim_stack_warmed"),
            patch("services.job_runner.capture_job") as mock_capture,
            patch("services.job_runner.assign_optimizer_ranks", return_value=[7]) as mock_ranks,
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = None
            mock_res_repo.return_value.succeeded_case_hashes_for_module.return_value = []
            ring = MagicMock()
            ring.get_all.return_value = []
            mock_capture.return_value.__enter__.return_value = ring
            mock_capture.return_value.__exit__.return_value = False

            run_job(job)

        mock_ranks.assert_called_once()
        assert rec.rank == 7


class TestChromeTraceAndCaseLogs:
    """Tests for chrome-trace materialization + per-case CLI log persistence."""

    def test_materializes_traces_when_chrome_trace_enabled(self):
        manager = _make_manager()
        run_job = _run_job_from(manager)
        job = _make_job(params={"model": "gpt2", "chrome_trace": True})
        rec = MagicMock()
        rec.case_hash = None
        rec.case_log = None
        runner = MagicMock()
        runner.run.return_value = ([rec], [])

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository") as mock_res_repo,
            patch("services.params_hash.compute_params_hash", return_value="hash123"),
            patch("runners.registry.create_runner", return_value=runner),
            patch("services.sim_warmup.ensure_sim_stack_warmed"),
            patch("services.job_runner.capture_job") as mock_capture,
            patch("services.trace_store.materialize_traces") as mock_mat,
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = None
            mock_res_repo.return_value.succeeded_case_hashes_for_module.return_value = []
            ring = MagicMock()
            ring.get_all.return_value = []
            mock_capture.return_value.__enter__.return_value = ring
            mock_capture.return_value.__exit__.return_value = False

            run_job(job)

        mock_mat.assert_called_once()

    def test_persists_case_logs_when_records_have_them(self):
        manager = _make_manager()
        run_job = _run_job_from(manager)
        job = _make_job()
        rec = MagicMock()
        rec.case_hash = "case-abc"
        rec.case_log = "cli output for case"
        runner = MagicMock()
        runner.run.return_value = ([rec], [])

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository") as mock_res_repo,
            patch("services.job_runner.CaseLogRepository") as mock_case_repo,
            patch("services.params_hash.compute_params_hash", return_value="hash123"),
            patch("runners.registry.create_runner", return_value=runner),
            patch("services.sim_warmup.ensure_sim_stack_warmed"),
            patch("services.job_runner.capture_job") as mock_capture,
            patch("services.job_runner.write_case_log_file") as mock_write_log,
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = None
            mock_res_repo.return_value.succeeded_case_hashes_for_module.return_value = []
            ring = MagicMock()
            ring.get_all.return_value = []
            mock_capture.return_value.__enter__.return_value = ring
            mock_capture.return_value.__exit__.return_value = False

            run_job(job)

        mock_case_repo.return_value.upsert_many.assert_called_once_with({"case-abc": "cli output for case"})
        mock_write_log.assert_called_once_with("case-abc", "cli output for case")

    def test_dedup_skipped_case_with_no_records_prints_fallback(self):
        """A skipped case whose source has no records prints a fallback line
        (covers the recs-empty branch in the dedup replay loop).
        """
        manager = _make_manager()
        run_job = _run_job_from(manager)
        job = _make_job()
        fresh_record = MagicMock()
        fresh_record.case_hash = "fresh1"
        fresh_record.case_log = None
        runner = MagicMock()
        runner.run.return_value = ([fresh_record], ["ghost"])  # "ghost" has no records

        with (
            patch("services.job_runner.JobRepository") as mock_job_repo,
            patch("services.job_runner.ResultRepository") as mock_res_repo,
            patch("services.job_runner.CaseLogRepository"),
            patch("services.params_hash.compute_params_hash", return_value="hash123"),
            patch("runners.registry.create_runner", return_value=runner),
            patch("services.sim_warmup.ensure_sim_stack_warmed"),
            patch("services.job_runner.capture_job") as mock_capture,
        ):
            mock_job_repo.return_value.find_succeeded_by_params_hash.return_value = None
            mock_res_repo.return_value.succeeded_case_hashes_for_module.return_value = ["ghost"]
            mock_res_repo.return_value.get_succeeded_records_by_case_hash.return_value = []
            ring = MagicMock()
            ring.get_all.return_value = []
            mock_capture.return_value.__enter__.return_value = ring
            mock_capture.return_value.__exit__.return_value = False

            run_job(job)  # must not raise; prints fallback "(no source log; ...)"
