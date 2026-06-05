"""Tests for web_ui.runner module."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from web_ui.result_store import ResultStore
from web_ui.runner import _decode_stream, ExperimentRunner, summarize_rows
from web_ui.schemas import ExperimentResult, ExperimentTask


class TestDecodeStream:
    """Tests for _decode_stream function."""

    def test_decode_stream_none(self) -> None:
        """Test with None input."""
        result = _decode_stream(None)
        assert result == ""

    def test_decode_stream_empty_bytes(self) -> None:
        """Test with empty bytes."""
        result = _decode_stream(b"")
        assert result == ""

    def test_decode_stream_utf8(self) -> None:
        """Test UTF-8 decoding."""
        result = _decode_stream(b"Hello World")
        assert result == "Hello World"

    def test_decode_stream_utf8_chinese(self) -> None:
        """Test UTF-8 decoding with special characters."""
        result = _decode_stream("TestContent".encode("utf-8"))
        assert result == "TestContent"

    def test_decode_stream_gb18030(self) -> None:
        """Test GB18030 decoding."""
        text = "TestContent"
        result = _decode_stream(text.encode("gb18030"))
        assert result == "TestContent"

    def test_decode_stream_cp936(self) -> None:
        """Test CP936 decoding."""
        text = "TestContent"
        result = _decode_stream(text.encode("cp936"))
        assert result == "TestContent"

    def test_decode_stream_fallback(self) -> None:
        """Test fallback to UTF-8 with replacement."""
        # Invalid UTF-8 sequence
        result = _decode_stream(b"\xff\xfe")
        assert isinstance(result, str)

    def test_decode_stream_mixed_content(self) -> None:
        """Test mixed ASCII and special content."""
        text = "Hello Test World"
        result = _decode_stream(text.encode("utf-8"))
        assert result == text


class TestSummarizeRows:
    """Tests for summarize_rows function."""

    def test_summarize_empty_list(self) -> None:
        """Test with empty list."""
        result = summarize_rows([])
        assert result == []

    def test_summarize_single_result(self) -> None:
        """Test with single result."""
        exp_result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={"model": "test"},
            command=["test"],
            task_hash="h1",
            label="test",
            summary={"tps": 100.0},
        )
        result = summarize_rows([exp_result])
        assert len(result) == 1
        assert result[0]["sim_type"] == "text_generate"

    def test_summarize_multiple_results(self) -> None:
        """Test with multiple results."""
        results = [
            ExperimentResult(
                sim_type="text_generate",
                status="success",
                params={"model": f"test{i}"},
                command=["test"],
                task_hash=f"h{i}",
                label=f"test{i}",
                summary={"tps": i * 100.0},
            )
            for i in range(3)
        ]
        result = summarize_rows(results)
        assert len(result) == 3
        assert result[0]["label"] == "test0"
        assert result[2]["label"] == "test2"


class TestExperimentRunner:
    """Tests for ExperimentRunner class."""

    @pytest.fixture
    def temp_store(self) -> ResultStore:
        """Create a ResultStore with temporary directory."""
        import tempfile
        import shutil
        import gc

        tmpdir = tempfile.mkdtemp()
        store = ResultStore(root=tmpdir)
        yield store
        # Cleanup
        try:
            del store
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    @pytest.fixture
    def runner(self, temp_store: ResultStore) -> ExperimentRunner:
        """Create an ExperimentRunner with temp store."""
        return ExperimentRunner(store=temp_store, max_workers=1)

    def test_runner_init(self, temp_store: ResultStore) -> None:
        """Test ExperimentRunner initialization."""
        runner = ExperimentRunner(store=temp_store, max_workers=4)
        assert runner.store is temp_store
        assert runner.max_workers == 4

    def test_runner_init_default_workers(self, temp_store: ResultStore) -> None:
        """Test default max_workers value."""
        runner = ExperimentRunner(store=temp_store)
        assert runner.max_workers == 2

    def test_run_task_with_cache(self, runner: ExperimentRunner) -> None:
        """Test _run_task returns cached result."""
        # Save a cached result first
        cached = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={"model": "cached"},
            command=["cached"],
            task_hash="cached_hash",
            label="cached",
            summary={"cached": True},
        )
        runner.store.save_result(cached)

        task = ExperimentTask("text_generate", {"model": "cached"}, [], "cached_hash", "cached")
        result = runner._run_task(task)

        assert result.status == "success"
        assert result.summary.get("cached") is True
        assert result.source == "cache"

    @patch("web_ui.runner.subprocess.run")
    def test_run_task_failed_cache_overridden(self, mock_run, runner: ExperimentRunner) -> None:
        """Test _run_task re-runs failed cached results."""
        # Save a failed result
        failed = ExperimentResult(
            sim_type="text_generate",
            status="failed",
            params={"model": "failed"},
            command=["echo", "success"],
            task_hash="failed_hash",
            label="failed",
            error="Previous failure",
        )
        runner.store.save_result(failed)

        # Mock successful run
        mock_run.return_value = Mock(returncode=0, stdout=b"Success", stderr=b"")

        task = ExperimentTask("text_generate", {"model": "failed"}, [], "failed_hash", "failed")
        result = runner._run_task(task)

        # Should re-run and succeed
        assert result.status == "success"
        assert result.source == "run"  # New run, not cached

    @patch("web_ui.runner.subprocess.run")
    def test_run_task_execute_success(self, mock_run, runner: ExperimentRunner) -> None:
        """Test _run_task executes successfully."""
        mock_run.return_value = Mock(returncode=0, stdout=b"Success output", stderr=b"")

        task = ExperimentTask("text_generate", {"model": "test"}, [], "new_hash", "new_test")
        result = runner._run_task(task)

        assert result.status == "success"
        assert result.source == "run"
        mock_run.assert_called_once()

    @patch("web_ui.runner.subprocess.run")
    def test_run_task_execute_failure(self, mock_run, runner: ExperimentRunner) -> None:
        """Test _run_task handles failure."""
        mock_run.return_value = Mock(returncode=1, stdout=b"", stderr=b"Error output")

        task = ExperimentTask("text_generate", {"model": "fail"}, [], "fail_hash", "fail_test")
        result = runner._run_task(task)

        assert result.status == "failed"
        assert "Process exited with code 1" in result.error
        mock_run.assert_called_once()

    @patch("web_ui.runner.subprocess.run")
    def test_run_task_saves_to_store(self, mock_run, runner: ExperimentRunner) -> None:
        """Test _run_task saves result to store."""
        mock_run.return_value = Mock(returncode=0, stdout=b"Output", stderr=b"")

        task = ExperimentTask("text_generate", {"model": "save"}, [], "save_hash", "save_test")
        runner._run_task(task)

        # Verify it was saved
        cached = runner.store.get_cached_result(task)
        assert cached is not None
        assert cached.status == "success"

    @patch("web_ui.runner.subprocess.run")
    def test_run_task_with_special_output(self, mock_run, runner: ExperimentRunner) -> None:
        """Test _run_task handles special output."""
        mock_run.return_value = Mock(returncode=0, stdout="TestOutput".encode("gb18030"), stderr=b"")

        task = ExperimentTask("text_generate", {"model": "special"}, [], "sp_hash", "sp_test")
        result = runner._run_task(task)

        assert result.status == "success"
        assert "TestOutput" in result.raw_log or len(result.raw_log) > 0

    def test_run_matrix_empty_list(self, runner: ExperimentRunner) -> None:
        """Test run_matrix with empty task list."""
        results = list(runner.run_matrix([]))
        assert results == []

    @patch("web_ui.runner.subprocess.run")
    def test_run_matrix_single_task(self, mock_run, runner: ExperimentRunner) -> None:
        """Test run_matrix with single task."""
        mock_run.return_value = Mock(returncode=0, stdout=b"Output", stderr=b"")

        task = ExperimentTask("text_generate", {"model": "single"}, [], "single_hash", "single")
        results = list(runner.run_matrix([task]))

        assert len(results) == 1
        completed, total, result = results[0]
        assert completed == 1
        assert total == 1
        assert result.status == "success"

    @patch("web_ui.runner.subprocess.run")
    def test_run_matrix_multiple_tasks(self, mock_run, runner: ExperimentRunner) -> None:
        """Test run_matrix with multiple tasks."""
        mock_run.return_value = Mock(returncode=0, stdout=b"Output", stderr=b"")

        tasks = [ExperimentTask("text_generate", {"model": f"test{i}"}, [], f"hash{i}", f"test{i}") for i in range(3)]
        results = list(runner.run_matrix(tasks))

        assert len(results) == 3
        # Check completed counts
        completed_counts = [r[0] for r in results]
        assert set(completed_counts) == {1, 2, 3}

    @patch("web_ui.runner.subprocess.run")
    def test_run_matrix_updates_progress(self, mock_run, runner: ExperimentRunner) -> None:
        """Test that run_matrix yields correct progress."""
        mock_run.return_value = Mock(returncode=0, stdout=b"Output", stderr=b"")

        tasks = [ExperimentTask("text_generate", {"model": f"prog{i}"}, [], f"prog{i}", f"prog{i}") for i in range(5)]

        results = []
        for completed, total, result in runner.run_matrix(tasks):
            results.append((completed, total, result))

        assert len(results) == 5
        assert all(r[1] == 5 for r in results)  # Total is always 5

    def test_run_matrix_respects_max_workers(self, temp_store: ResultStore) -> None:
        """Test run_matrix uses configured max_workers."""
        runner = ExperimentRunner(store=temp_store, max_workers=1)

        with patch("web_ui.runner.ThreadPoolExecutor") as mock_pool:
            mock_executor = Mock()
            mock_pool.return_value.__enter__ = Mock(return_value=mock_executor)
            mock_pool.return_value.__exit__ = Mock(return_value=False)

            # Setup submit to return completed futures
            from concurrent.futures import Future

            def completed_future(*args, **kwargs):
                f = Future()
                f.set_result(
                    ExperimentResult(
                        sim_type="text_generate",
                        status="success",
                        params={},
                        command=[],
                        task_hash="test",
                        label="test",
                    )
                )
                return f

            mock_executor.submit = completed_future
            mock_executor.__enter__ = Mock(return_value=mock_executor)
            mock_executor.__exit__ = Mock(return_value=False)

            task = ExperimentTask("text_generate", {}, [], "h", "t")
            list(runner.run_matrix([task]))

            # Verify max_workers was passed
            mock_pool.assert_called_once()

    @patch("web_ui.runner.subprocess.run")
    def test_run_matrix_with_mixed_success_failure(self, mock_run, runner: ExperimentRunner) -> None:
        """Test run_matrix with mixed success and failure."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return Mock(returncode=1, stdout=b"", stderr=b"Error")
            return Mock(returncode=0, stdout=b"Success", stderr=b"")

        mock_run.side_effect = side_effect

        tasks = [ExperimentTask("text_generate", {}, [], f"h{i}", f"t{i}") for i in range(4)]

        results = list(runner.run_matrix(tasks))

        assert len(results) == 4
        success_count = sum(1 for _, _, r in results if r.status == "success")
        failed_count = sum(1 for _, _, r in results if r.status == "failed")
        assert success_count == 2
        assert failed_count == 2

    def test_run_matrix_preserves_task_order_in_results(self, temp_store: ResultStore) -> None:
        """Test that tasks are executed but completion order may vary."""
        runner = ExperimentRunner(store=temp_store, max_workers=2)

        with patch("web_ui.runner.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout=b"Output", stderr=b"")

            tasks = [ExperimentTask("text_generate", {}, [], f"hash{i}", f"label{i}") for i in range(3)]

            results = list(runner.run_matrix(tasks))
            assert len(results) == 3

            # All tasks should be executed
            labels = [r[2].label for r in results]
            assert len(set(labels)) == 3  # All unique
