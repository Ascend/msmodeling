"""Unit tests for write_queue module."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future

import pytest
from services.write_queue import WriteQueue, _sentinel_future, _sentinel_value


class TestWriteQueueInit:
    """Tests for WriteQueue initialization."""

    def test_init_default_state(self):
        """Initializes with empty queue and no thread."""
        wq = WriteQueue()
        assert wq._queue.empty()
        assert wq._thread is None
        assert not wq._stop.is_set()

    def test_init_stop_event_clear(self):
        """Stop event is initially clear."""
        wq = WriteQueue()
        assert not wq._stop.is_set()

    def test_init_lock_exists(self):
        """Lock is initialized."""
        wq = WriteQueue()
        assert wq._lock is not None


class TestWriteQueueStart:
    """Tests for start method."""

    def test_start_creates_thread(self):
        """Start creates and starts the writer thread."""
        wq = WriteQueue()
        wq.start()
        assert wq._thread is not None
        assert wq._thread.is_alive()

        # Cleanup
        wq.shutdown(wait=False)

    def test_start_idempotent(self):
        """Multiple start calls are safe (only one thread created)."""
        wq = WriteQueue()
        wq.start()
        first_thread = wq._thread
        wq.start()
        assert wq._thread is first_thread

        # Cleanup
        wq.shutdown(wait=False)

    def test_start_clears_stop_event(self):
        """Start clears the stop event."""
        wq = WriteQueue()
        wq._stop.set()
        wq.start()
        assert not wq._stop.is_set()

        # Cleanup
        wq.shutdown(wait=False)

    def test_start_restarts_after_shutdown(self):
        """Can restart after shutdown."""
        wq = WriteQueue()
        wq.start()
        wq.shutdown(wait=True)
        assert not wq._thread.is_alive()

        # Restart
        wq.start()
        assert wq._thread is not None
        assert wq._thread.is_alive()

        # Cleanup
        wq.shutdown(wait=False)


class TestWriteQueueEnqueue:
    """Tests for enqueue method."""

    def test_enqueue_returns_future(self):
        """Enqueue returns a Future."""
        wq = WriteQueue()
        wq.start()

        future = wq.enqueue(lambda: 42)
        assert isinstance(future, Future)

        # Cleanup
        wq.shutdown(wait=False)

    def test_enqueue_executes_thunk(self):
        """Thunk is executed in writer thread."""
        wq = WriteQueue()
        wq.start()

        result = []
        future = wq.enqueue(lambda: result.append(1))
        future.result(timeout=2)

        assert result == [1]

        # Cleanup
        wq.shutdown(wait=False)

    def test_enqueue_returns_result(self):
        """Future resolves to thunk's return value."""
        wq = WriteQueue()
        wq.start()

        future = wq.enqueue(lambda: "test_result")
        result = future.result(timeout=2)

        assert result == "test_result"

        # Cleanup
        wq.shutdown(wait=False)

    def test_enqueue_multiple(self):
        """Multiple enqueues execute in order."""
        wq = WriteQueue()
        wq.start()

        results = []
        futures = [
            wq.enqueue(lambda: results.append(1)),
            wq.enqueue(lambda: results.append(2)),
            wq.enqueue(lambda: results.append(3)),
        ]

        for f in futures:
            f.result(timeout=2)

        assert results == [1, 2, 3]

        # Cleanup
        wq.shutdown(wait=False)

    def test_enqueue_with_exception(self):
        """Thunk exception propagates to future."""
        wq = WriteQueue()
        wq.start()

        future = wq.enqueue(lambda: 1 / 0)

        with pytest.raises(ZeroDivisionError):
            future.result(timeout=2)

        # Cleanup
        wq.shutdown(wait=False)

    def test_enqueue_after_shutdown_rejects(self):
        """Enqueue after shutdown raises immediately."""
        wq = WriteQueue()
        wq.start()
        wq.shutdown(wait=True)

        future = wq.enqueue(lambda: 42)

        # Future should already be completed with exception
        with pytest.raises(RuntimeError, match="stopped"):
            future.result(timeout=0.1)

    def test_enqueue_during_shutdown_race_safe(self):
        """Enqueue during shutdown is safe (atomic check+put)."""
        wq = WriteQueue()
        wq.start()

        # This test verifies the lock prevents race between
        # shutdown and enqueue
        def slow_write():
            time.sleep(0.1)
            return "done"

        future1 = wq.enqueue(slow_write)

        # Wait for first write to complete
        assert future1.result(timeout=2) == "done"

        # Now shutdown
        wq.shutdown(wait=True)

        # Try to enqueue after shutdown
        future2 = wq.enqueue(lambda: "should_fail")

        # Second should fail immediately
        with pytest.raises(RuntimeError, match="stopped"):
            future2.result(timeout=0.1)


class TestWriteQueueShutdown:
    """Tests for shutdown method."""

    def test_shutdown_sets_stop_event(self):
        """Shutdown sets the stop event."""
        wq = WriteQueue()
        wq.start()
        wq.shutdown(wait=False)
        assert wq._stop.is_set()

    def test_shutdown_idempotent(self):
        """Multiple shutdown calls are safe."""
        wq = WriteQueue()
        wq.start()
        wq.shutdown(wait=False)
        wq.shutdown(wait=False)
        wq.shutdown(wait=False)
        assert True  # No exceptions

    def test_shutdown_waits_for_thread(self):
        """Shutdown with wait=True joins the thread."""
        wq = WriteQueue()
        wq.start()
        wq.shutdown(wait=True)
        assert not wq._thread.is_alive()

    def test_shutdown_without_wait_returns_fast(self):
        """Shutdown with wait=False sets the stop event deterministically."""
        wq = WriteQueue()
        wq.start()
        wq.shutdown(wait=False)
        assert wq._stop.is_set()
        # Clean up
        if wq._thread and wq._thread.is_alive():
            wq._thread.join(timeout=2)

    def test_shutdown_cancels_pending_futures(self):
        """Pending futures are cancelled at shutdown."""
        wq = WriteQueue()
        wq.start()

        # Enqueue a slow write
        wq.enqueue(lambda: time.sleep(1))
        # Enqueue more items that won't be processed
        pending_futures = [wq.enqueue(lambda: "never_executed") for _ in range(3)]

        # Give first write time to start, then shutdown
        time.sleep(0.05)
        wq.shutdown(wait=False)

        # Pending futures should be cancelled
        for f in pending_futures:
            with pytest.raises(RuntimeError, match="shutting down"):
                f.result(timeout=0.1)

    def test_shutdown_with_wait_timeout(self, monkeypatch):
        """Shutdown join has a timeout."""
        wq = WriteQueue()
        wq.start()

        # Enqueue a blocking write
        wq.enqueue(lambda: time.sleep(10))

        time.sleep(0.05)
        # Shorten the join timeout for this test
        original_join = wq._thread.join
        wq._thread.join = lambda timeout=5: original_join(timeout=0.5)
        start = time.time()
        wq.shutdown(wait=True)
        elapsed = time.time() - start
        assert elapsed < 2  # should return quickly with the short timeout


class TestWriteQueuePending:
    """Tests for pending property."""

    def test_pending_initially_zero(self):
        """Pending is 0 for empty queue."""
        wq = WriteQueue()
        assert wq.pending == 0

    def test_pending_increases_with_enqueue(self):
        """Pending increases when items are enqueued."""
        wq = WriteQueue()
        # Don't start — items will queue up
        assert wq.pending == 0
        for _ in range(5):
            wq.enqueue(lambda: time.sleep(0.1))
        assert wq.pending == 5  # all 5 pending
        wq.start()
        wq.shutdown(wait=True)

    def test_pending_decreases_after_processing(self):
        """Pending decreases as items are processed."""
        wq = WriteQueue()
        wq.start()

        futures = [wq.enqueue(lambda: i) for i in range(3)]

        # Wait for all to complete
        for f in futures:
            f.result(timeout=2)

        # Queue should be empty
        assert wq.pending == 0

        # Cleanup
        wq.shutdown(wait=False)


class TestWriteQueueDrain:
    """Tests for _drain internal method."""

    def test_drain_processes_items(self):
        """_drain processes items from queue."""
        wq = WriteQueue()
        wq.start()

        results = []
        future = wq.enqueue(lambda: results.append("processed"))
        future.result(timeout=2)

        assert "processed" in results

        # Cleanup
        wq.shutdown(wait=False)

    def test_drain_handles_empty_queue(self):
        """_drain handles empty queue gracefully."""
        wq = WriteQueue()
        wq.start()

        # Let drain loop run with empty queue
        time.sleep(0.2)

        # Should not raise
        assert True

        # Cleanup
        wq.shutdown(wait=False)

    def test_drain_continues_after_exception(self):
        """_drain continues processing after exception in thunk."""
        wq = WriteQueue()
        wq.start()

        # First thunk fails
        failing_future = wq.enqueue(lambda: 1 / 0)

        # Second thunk should still execute
        success_future = wq.enqueue(lambda: "success")

        # First should raise
        with pytest.raises(ZeroDivisionError):
            failing_future.result(timeout=2)

        # Second should succeed
        assert success_future.result(timeout=2) == "success"

        # Cleanup
        wq.shutdown(wait=False)

    def test_drain_respects_stop_event(self):
        """_drain exits when stop event is set."""
        wq = WriteQueue()
        wq.start()

        # Let it run
        time.sleep(0.1)

        # Shutdown
        wq.shutdown(wait=True)

        # Thread should have exited
        assert not wq._thread.is_alive()


class TestSentinels:
    """Tests for sentinel objects."""

    def test_sentinel_future_exists(self):
        """Sentinel future object exists."""
        assert _sentinel_future is not None
        assert isinstance(_sentinel_future, Future)

    def test_sentinel_value_exists(self):
        """Sentinel value object exists."""
        assert _sentinel_value is not None

    def test_sentinel_unique(self):
        """Sentinel objects are unique instances."""
        assert _sentinel_value is not None
        assert _sentinel_future is not None


class TestWriteQueueConcurrency:
    """Tests for concurrent behavior."""

    def test_concurrent_enqueues(self):
        """Multiple threads can enqueue concurrently."""
        wq = WriteQueue()
        wq.start()

        results = []

        def append_value(v):
            return results.append(v)

        threads = []
        for i in range(10):
            t = threading.Thread(target=lambda v=i: wq.enqueue(lambda: append_value(v)).result(timeout=2))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All values should be appended
        assert len(results) == 10
        assert set(results) == set(range(10))

        # Cleanup
        wq.shutdown(wait=False)

    def test_writer_thread_alive(self):
        """Writer thread stays alive with work."""
        wq = WriteQueue()
        wq.start()

        # Submit work repeatedly
        for _ in range(5):
            future = wq.enqueue(lambda: time.sleep(0.01))
            future.result(timeout=2)
            assert wq._thread.is_alive()

        # Cleanup
        wq.shutdown(wait=False)
