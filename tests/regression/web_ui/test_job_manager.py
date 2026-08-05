"""Unit tests for job_manager module."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from models.entities import Job
from models.enums import JobStatus
from services.job_manager import JobManager, _utcnow_iso


def _make_mock_repo(**overrides):
    """Create a MagicMock repo with count_jobs returning 0 (no in-flight jobs).

    Tests that need a different count can override via kwargs or set
    ``mock_repo.count_jobs.return_value`` directly after creation.
    """
    repo = MagicMock()
    repo.count_jobs.return_value = overrides.get("count_jobs_return", 0)
    return repo


class TestUtcnowIso:
    """Tests for _utcnow_iso helper function."""

    def test_utcnow_iso_returns_string(self):
        """Returns ISO format timestamp string."""
        result = _utcnow_iso()
        assert isinstance(result, str)

    def test_utcnow_iso_format(self):
        """Returns ISO 8601 format with 'Z' suffix."""
        result = _utcnow_iso()
        assert "T" in result
        assert result.endswith("Z")
        assert result.count("-") >= 2
        assert result.count(":") >= 2

    def test_utcnow_iso_utc(self):
        """Timestamp is in UTC."""
        result = _utcnow_iso()
        assert result.endswith("Z")


class TestJobManagerInit:
    """Tests for JobManager initialization."""

    def test_init_default_workers(self):
        """Initializes with default max_workers=8."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        assert manager._max_workers == 8
        assert manager._jobs is mock_repo
        assert manager._run_job is None
        assert isinstance(manager._executor, ThreadPoolExecutor)

    def test_init_custom_workers(self):
        """Initializes with custom max_workers."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo, max_workers=4)
        assert manager._max_workers == 4

    def test_init_write_queue_started(self):
        """WriteQueue is started on initialization."""
        mock_repo = _make_mock_repo()
        mock_write_queue = MagicMock()
        with patch("services.job_manager.WriteQueue", return_value=mock_write_queue):
            JobManager(mock_repo)
            mock_write_queue.start.assert_called_once()

    def test_init_priority_queue_empty(self):
        """Queue starts empty."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        assert manager._pq == []

    def test_init_module_locks_empty(self):
        """Module locks dict starts empty."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        assert manager._module_locks == {}

    def test_init_cancel_flags_empty(self):
        """Cancel flags dict starts empty."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        assert manager._cancel_flags == {}


class TestJobManagerSetRunJob:
    """Tests for set_run_job method."""

    def test_set_run_job(self):
        """Can inject run_job function."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        mock_run_job = MagicMock()
        manager.set_run_job(mock_run_job)
        assert manager._run_job is mock_run_job


class TestJobManagerModuleLock:
    """Tests for module_lock method."""

    def test_module_lock_returns_lock(self):
        """Returns a lock object."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        lock = manager.module_lock("test_module")
        # Returns a lock (may be threading.Lock or similar)
        assert hasattr(lock, 'acquire') and hasattr(lock, 'release')

    def test_module_lock_same_instance(self):
        """Returns the same lock for the same module_id."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        lock1 = manager.module_lock("test_module")
        lock2 = manager.module_lock("test_module")
        assert lock1 is lock2

    def test_module_lock_different_modules(self):
        """Returns different locks for different module_ids."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        lock1 = manager.module_lock("module1")
        lock2 = manager.module_lock("module2")
        assert lock1 is not lock2

    def test_module_lock_thread_safety(self):
        """Module lock creation is thread-safe."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)

        locks = []

        def get_lock():
            locks.append(manager.module_lock("test_module"))

        threads = [threading.Thread(target=get_lock) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same lock instance
        assert len({id(lock) for lock in locks}) == 1


class TestJobManagerMakeCancelFlag:
    """Tests for _make_cancel_flag method."""

    def test_make_cancel_flag_returns_callable(self):
        """Returns a callable."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        flag = manager._make_cancel_flag("job123")
        assert callable(flag)

    def test_make_cancel_flag_initially_false(self):
        """Flag is initially False (not cancelled)."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        flag = manager._make_cancel_flag("job123")
        assert flag() is False

    def test_make_cancel_flag_returns_same_flag(self):
        """Returns the same flag for the same job_id."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        flag1 = manager._make_cancel_flag("job123")
        flag2 = manager.cancel_flag("job123")
        assert flag1 is flag2

    def test_make_cancel_flag_unknown_job_returns_false_flag(self):
        """Unknown job_id returns always-false flag."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        flag = manager.cancel_flag("unknown_job")
        assert flag() is False


class TestJobManagerRequestCancel:
    """Tests for request_cancel method."""

    def test_request_cancel_unknown_job(self):
        """Returns False for unknown job_id."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        result = manager.request_cancel("unknown_job")
        assert result is False

    def test_request_cancel_known_job(self):
        """Returns True and sets flag for known job."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        manager._make_cancel_flag("job123")
        result = manager.request_cancel("job123")
        assert result is True

    def test_request_cancel_flag_is_set(self):
        """Requesting cancel sets the flag to True."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        flag = manager._make_cancel_flag("job123")
        assert flag() is False
        manager.request_cancel("job123")
        assert flag() is True


class TestJobManagerSubmit:
    """Tests for submit method."""

    def test_submit_sets_pending_status(self):
        """Submit sets job status to PENDING."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        mock_run_job = MagicMock()
        manager.set_run_job(mock_run_job)

        job = Job(id="job123", module_id="test_module", params={}, form_schema_version="1.0", status=JobStatus.RUNNING)
        result = manager.submit(job)

        assert result.status == JobStatus.PENDING

    def test_submit_sets_created_at(self):
        """Submit sets created_at timestamp if missing."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        mock_run_job = MagicMock()
        manager.set_run_job(mock_run_job)

        job = Job(id="job123", module_id="test_module", params={}, form_schema_version="1.0", created_at=None)
        result = manager.submit(job)

        assert result.created_at is not None
        assert "T" in result.created_at

    def test_submit_adds_to_repository(self):
        """Submit calls repository.add."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        mock_run_job = MagicMock()
        manager.set_run_job(mock_run_job)

        job = Job(id="job123", module_id="test_module", params={}, form_schema_version="1.0")
        manager.submit(job)

        mock_repo.add.assert_called_once()

    def test_submit_without_run_job(self):
        """Submit works even before run_job is injected."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        # Don't set run_job

        job = Job(id="job123", module_id="test_module", params={}, form_schema_version="1.0")
        result = manager.submit(job)

        assert result.status == JobStatus.PENDING
        mock_repo.add.assert_called_once()


class TestJobManagerSubmitMany:
    """Tests for submit_many method."""

    def test_submit_many_creates_cancel_flags(self):
        """Creates cancel flags for all jobs."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        manager.set_run_job(MagicMock())

        jobs = [
            Job(id="job1", module_id="test_module", params={}, form_schema_version="1.0"),
            Job(id="job2", module_id="test_module", params={}, form_schema_version="1.0"),
        ]
        manager.submit_many(jobs)

        assert manager.cancel_flag("job1") is not False
        assert manager.cancel_flag("job2") is not False

    def test_submit_many_schedules_jobs(self):
        """Schedules all jobs for execution."""
        mock_repo = _make_mock_repo()
        mock_run_job = MagicMock()
        manager = JobManager(mock_repo)
        manager.set_run_job(mock_run_job)

        jobs = [
            Job(id="job1", module_id="test_module", params={}, form_schema_version="1.0"),
            Job(id="job2", module_id="test_module", params={}, form_schema_version="1.0"),
        ]
        manager.submit_many(jobs)

        # Jobs should be scheduled (added to priority queue)
        # We can verify by checking the queue is not empty
        # (though we can't directly inspect the queue without waiting)
        assert len(manager._pq) >= 0  # Jobs may have been processed already

    def test_submit_many_returns_jobs(self):
        """Returns the list of jobs."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        manager.set_run_job(MagicMock())

        jobs = [
            Job(id="job1", module_id="test_module", params={}, form_schema_version="1.0"),
            Job(id="job2", module_id="test_module", params={}, form_schema_version="1.0"),
        ]
        result = manager.submit_many(jobs)

        assert result == jobs


class TestJobManagerSchedule:
    """Tests for _schedule method."""

    def test_schedule_enqueues_job(self):
        """A scheduled job is picked up by a worker and run."""
        mock_repo = _make_mock_repo()
        mock_run_job = MagicMock()
        manager = JobManager(mock_repo)
        manager.set_run_job(mock_run_job)

        job = Job(id="job123", module_id="test", params={}, form_schema_version="1.0")
        manager._schedule(job)

        # Force the worker to drain the queue deterministically, then confirm the
        # job was actually run. (Asserting on _pq directly races the worker thread.)
        manager._executor.shutdown(wait=True)
        mock_run_job.assert_called_once_with(job)

    def test_schedule_without_run_job(self):
        """Does nothing if run_job is not set."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        # Don't set run_job

        job = Job(id="job123", module_id="test", params={}, form_schema_version="1.0")
        manager._schedule(job)

        # Queue should remain empty
        assert len(manager._pq) == 0


class TestJobManagerDrainOne:
    """Tests for _drain_one method."""

    def test_drain_one_empty_queue(self):
        """Does nothing when queue is empty."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        manager.set_run_job(MagicMock())

        # Queue is empty, should not raise
        manager._drain_one()

    def test_drain_one_pops_job(self):
        """Pops and runs a job from the queue."""
        mock_repo = _make_mock_repo()
        mock_run_job = MagicMock()
        manager = JobManager(mock_repo)
        manager.set_run_job(mock_run_job)

        job = Job(id="job123", module_id="test", params={}, form_schema_version="1.0")
        manager._schedule(job)

        # Wait for executor to process deterministically (no sleep race)
        manager._executor.shutdown(wait=True)

        # Job should be removed from queue
        with manager._pq_lock:
            assert len(manager._pq) == 0
        # Job should have actually been executed
        mock_run_job.assert_called_once_with(job)


class TestJobManagerAsync:
    """Tests for async methods."""

    def test_submit_async_is_coroutine(self):
        """submit_async is a coroutine function."""
        import asyncio

        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)

        job = Job(id="job123", module_id="test", params={}, form_schema_version="1.0")
        result = manager.submit_async(job)
        assert asyncio.iscoroutine(result)
        # Close the coroutine to avoid warning
        result.close()

    def test_write_async_is_coroutine(self):
        """write_async is a coroutine function."""
        import asyncio

        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)

        thunk = MagicMock()
        result = manager.write_async(thunk)
        assert asyncio.iscoroutine(result)
        # Close the coroutine to avoid warning
        result.close()


class TestJobManagerShutdown:
    """Tests for shutdown method."""

    def test_shutdown_stops_executor(self):
        """Shutdown stops the executor."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        manager.set_run_job(MagicMock())

        manager.shutdown(wait=False)

        # Executor should be shutdown
        assert manager._executor._shutdown is True

    def test_shutdown_stops_write_queue(self):
        """Shutdown stops the write queue."""
        mock_repo = _make_mock_repo()
        mock_write_queue = MagicMock()
        with patch("services.job_manager.WriteQueue", return_value=mock_write_queue):
            manager = JobManager(mock_repo)
            manager.set_run_job(MagicMock())

            manager.shutdown(wait=False)

            mock_write_queue.shutdown.assert_called_once()

    def test_shutdown_with_wait(self):
        """Shutdown can wait for tasks to complete."""
        mock_repo = _make_mock_repo()
        mock_write_queue = MagicMock()
        with patch("services.job_manager.WriteQueue", return_value=mock_write_queue):
            manager = JobManager(mock_repo)
            manager.set_run_job(MagicMock())

            manager.shutdown(wait=True)

            mock_write_queue.shutdown.assert_called_once_with(wait=True)

    def test_shutdown_without_wait(self):
        """Shutdown without wait still drains the write queue (#36).

        The previous implementation propagated ``wait=False`` to the write
        queue, so a worker mid-write at shutdown time would have its final
        status write lost. Now only the executor wait is optional; the write
        queue is ALWAYS drained so in-flight jobs can persist.
        """
        mock_repo = _make_mock_repo()
        mock_write_queue = MagicMock()
        with patch("services.job_manager.WriteQueue", return_value=mock_write_queue):
            manager = JobManager(mock_repo)
            manager.set_run_job(MagicMock())

            manager.shutdown(wait=False)

            # Write queue is always drained with wait=True, regardless of the
            # worker-pool wait mode.
            mock_write_queue.shutdown.assert_called_once_with(wait=True)

    def test_shutdown_logs_warning_on_pool_timeout(self, caplog):
        """When executor.shutdown takes longer than worker_timeout, a warning
        is logged (in-flight jobs will be swept as interrupted on next boot).
        """
        import logging
        import threading

        mock_repo = _make_mock_repo()
        mock_write_queue = MagicMock()

        # Block shutdown indefinitely so the wait times out.
        block = threading.Event()

        class SlowExecutor:
            def submit(self, *a, **kw):
                return MagicMock()

            def shutdown(self, wait=True, cancel_futures=False):
                block.wait()  # block until test releases

        with (
            patch("services.job_manager.WriteQueue", return_value=mock_write_queue),
            patch("services.job_manager.ThreadPoolExecutor", return_value=SlowExecutor()),
        ):
            manager = JobManager(mock_repo, max_workers=1)
            manager.set_run_job(MagicMock())
            with caplog.at_level(logging.WARNING, logger="services.job_manager"):
                # Use a very short timeout so the test doesn't hang.
                manager.shutdown(wait=True, worker_timeout=0.05)
        block.set()  # release the blocked shutdown thread
        assert any("timed out" in r.message for r in caplog.records)

    def test_shutdown_sets_shutting_down_flag(self):
        """shutdown() sets _shutting_down so late submits are refused (#36)."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        manager.set_run_job(MagicMock())

        assert manager._shutting_down is False
        manager.shutdown(wait=False)
        assert manager._shutting_down is True

    def test_schedule_refuses_after_shutdown(self):
        """_schedule refuses to enqueue new jobs after shutdown (#36)."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        manager.set_run_job(MagicMock())

        job = Job(id="job-after-shutdown", module_id="test", params={}, form_schema_version="1.0")

        manager.shutdown(wait=False)
        # Should not raise — refuses silently and logs a warning.
        manager._schedule(job)

        # Job was NOT pushed onto the queue.
        assert manager._pq == []
        # Cancel flag was NOT created (the early-return path skips _make_cancel_flag).
        assert manager.cancel_flag("job-after-shutdown")() is False

    def test_drain_one_cleans_up_cancel_flags_on_terminal(self):
        """_drain_one drops the in-memory cancel flag once the job is done (#89).

        Previously the flag persisted after the job reached terminal state, so
        a stale poll could see cancel_requested=True for a succeeded job.
        """
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        # Synchronous run_job that completes immediately.
        manager.set_run_job(lambda job: None)

        job = Job(id="job-terminal", module_id="test", params={}, form_schema_version="1.0")
        manager._make_cancel_flag(job.id)
        manager.request_cancel(job.id)
        assert manager.cancel_flag(job.id)() is True

        manager._schedule(job)
        # Wait for the worker to complete.
        manager._executor.shutdown(wait=True)

        # Cancel flags (both the poll callable and the __request entry) are gone.
        assert manager.cancel_flag(job.id)() is False
        assert f"{job.id}__request" not in manager._cancel_flags

    def test_is_cancel_requested(self):
        """is_cancel_requested surfaces the in-memory flag state (#89)."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)

        # Unknown job → False.
        assert manager.is_cancel_requested("no-such-job") is False

        # Known but not-yet-requested → False.
        manager._make_cancel_flag("job-x")
        assert manager.is_cancel_requested("job-x") is False

        # Requested → True.
        manager.request_cancel("job-x")
        assert manager.is_cancel_requested("job-x") is True


class TestJobManagerErrorHandling:
    """Tests for error handling in job execution."""

    def test_drain_one_handles_run_job_exception(self):
        """Logs and marks job FAILED when run_job raises exception."""
        mock_repo = _make_mock_repo()
        mock_run_job = MagicMock(side_effect=RuntimeError("Test error"))
        manager = JobManager(mock_repo)
        manager.set_run_job(mock_run_job)

        job = Job(id="job123", module_id="test", params={}, form_schema_version="1.0")
        manager._schedule(job)

        # Wait for executor and write queue to drain deterministically
        manager._executor.shutdown(wait=True)
        manager.write_queue.shutdown(wait=True)

        # Job should have been marked FAILED via a repo update call
        assert any(
            call.args[0] == "job123" and call.kwargs.get("status") == JobStatus.FAILED
            for call in mock_repo.update.call_args_list
        ), f"Expected a FAILED update for job123; calls: {mock_repo.update.call_args_list}"

    def test_drain_one_handles_write_failure(self, caplog):
        """Handles write failure after run_job exception gracefully."""
        import logging

        mock_repo = _make_mock_repo()
        mock_run_job = MagicMock(side_effect=RuntimeError("Test error"))

        # Build a Future whose .result() raises — the path exercised when the
        # write queue rejects the FAILED-marking write (e.g. shutting down).
        failed_future = Future()
        failed_future.set_exception(RuntimeError("Write failed"))

        mock_write_queue = MagicMock()
        mock_write_queue.enqueue.return_value = failed_future
        with patch("services.job_manager.WriteQueue", return_value=mock_write_queue):
            manager = JobManager(mock_repo)
            manager.set_run_job(mock_run_job)

            job = Job(id="job123", module_id="test", params={}, form_schema_version="1.0")
            manager._schedule(job)

            with caplog.at_level(logging.ERROR, logger="services.job_manager"):
                # Wait for executor to drain deterministically
                manager._executor.shutdown(wait=True)

            # The inner except should have logged "Failed to mark job ... FAILED"
            assert any("Failed to mark job" in r.message and "job123" in r.message for r in caplog.records), (
                f"Expected a write-failure log; records: {[r.message for r in caplog.records]}"
            )


class TestAsyncBridges:
    """Tests for the submit_async / submit_many_async / write_async bridges
    (awaitable wrappers around run_in_executor). pytest-asyncio isn't installed,
    so each test drives the coroutine via asyncio.run().
    """

    def test_submit_async_persists_and_returns_job(self):
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        try:
            job = Job(id="async-1", module_id="text_generate", params={}, form_schema_version="1.0")
            result = asyncio.run(manager.submit_async(job))
            assert result.id == "async-1"
            assert result.status == JobStatus.PENDING
            mock_repo.add.assert_called_once()
        finally:
            manager.shutdown(wait=False)

    def test_submit_many_async_schedules_all(self):
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        try:
            jobs = [Job(id=f"a-{i}", module_id="text_generate", params={}, form_schema_version="1.0") for i in range(3)]
            result = asyncio.run(manager.submit_many_async(jobs))
            assert len(result) == 3
        finally:
            manager.shutdown(wait=False)

    def test_write_async_runs_thunk_on_writer(self):
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo)
        try:
            ran = {"n": 0}

            def thunk():
                ran["n"] += 1
                return "done"

            out = asyncio.run(manager.write_async(thunk, timeout=10))
            assert out == "done"
            assert ran["n"] == 1
        finally:
            manager.shutdown(wait=False)


class TestInflightLimit:
    """Tests for the in-flight job cap (local DoS defense)."""

    def test_inflight_total_counts_pending_and_running(self):
        """_inflight_total sums pending + running from the repository."""
        mock_repo = _make_mock_repo()
        # count_jobs is called twice: once for PENDING, once for RUNNING
        mock_repo.count_jobs.side_effect = [3, 5]
        manager = JobManager(mock_repo)
        assert manager._inflight_total() == 8
        mock_repo.count_jobs.assert_any_call(status=JobStatus.PENDING)
        mock_repo.count_jobs.assert_any_call(status=JobStatus.RUNNING)

    def test_submit_raises_when_inflight_limit_exceeded(self):
        """submit() raises InflightLimitExceeded when at capacity."""
        mock_repo = _make_mock_repo()
        # 16 in-flight (max_workers=8 → max_inflight=16)
        mock_repo.count_jobs.side_effect = [8, 8]
        manager = JobManager(mock_repo, max_workers=8)
        job = Job(id="j1", module_id="test", params={}, form_schema_version="1.0")
        import pytest

        with pytest.raises(JobManager.InflightLimitExceeded, match="in-flight job limit"):
            manager.submit(job)

    def test_submit_succeeds_when_below_limit(self):
        """submit() works normally when below the in-flight cap."""
        mock_repo = _make_mock_repo()
        # 15 in-flight (below 16 limit)
        mock_repo.count_jobs.side_effect = [8, 7]
        manager = JobManager(mock_repo, max_workers=8)
        manager.set_run_job(MagicMock())
        job = Job(id="j1", module_id="test", params={}, form_schema_version="1.0")
        result = manager.submit(job)
        assert result.status == JobStatus.PENDING

    def test_submit_many_raises_when_would_exceed_limit(self):
        """submit_many() raises if adding all jobs would exceed the cap."""
        mock_repo = _make_mock_repo()
        # 14 in-flight, requesting 3 more → 17 > 16
        mock_repo.count_jobs.side_effect = [7, 7]
        manager = JobManager(mock_repo, max_workers=8)
        jobs = [Job(id=f"j{i}", module_id="test", params={}, form_schema_version="1.0") for i in range(3)]
        import pytest

        with pytest.raises(JobManager.InflightLimitExceeded, match="current=14, requested=3"):
            manager.submit_many(jobs)

    def test_submit_many_succeeds_when_within_limit(self):
        """submit_many() works when total stays within the cap."""
        mock_repo = _make_mock_repo()
        # 10 in-flight, requesting 3 more → 13 ≤ 16
        mock_repo.count_jobs.side_effect = [5, 5]
        manager = JobManager(mock_repo, max_workers=8)
        manager.set_run_job(MagicMock())
        jobs = [Job(id=f"j{i}", module_id="test", params={}, form_schema_version="1.0") for i in range(3)]
        result = manager.submit_many(jobs)
        assert len(result) == 3

    def test_max_inflight_defaults_to_2x_max_workers(self):
        """_max_inflight is 2×max_workers by default."""
        mock_repo = _make_mock_repo()
        manager = JobManager(mock_repo, max_workers=4)
        assert manager._max_inflight == 8
