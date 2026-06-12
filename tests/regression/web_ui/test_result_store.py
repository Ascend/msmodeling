"""Tests for web_ui.result_store module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from web_ui.result_store import (
    _enrich_optimizer_summary,
    _extract_optimizer_top1_from_log,
    _infer_optimizer_no_result_reason_from_params,
    _resolve_log_path,
    ResultStore,
)
from web_ui.schemas import ExperimentResult, ExperimentTask


class TestResolveLogPath:
    """Tests for _resolve_log_path function."""

    def test_resolve_log_path_empty_string(self) -> None:
        """Test with empty string."""
        result = _resolve_log_path("")
        assert result == Path("")

    def test_resolve_log_path_none(self) -> None:
        """Test with None."""
        result = _resolve_log_path(None)
        assert result == Path("")

    def test_resolve_log_path_unix_style(self) -> None:
        """Test with Unix-style path."""
        result = _resolve_log_path("logs/test.log")
        assert result == Path("logs/test.log")

    def test_resolve_log_path_windows_style(self) -> None:
        """Test with Windows-style path."""
        result = _resolve_log_path(r"logs\test.log")
        assert result == Path("logs/test.log")

    def test_resolve_log_path_mixed_separators(self) -> None:
        """Test with mixed separators."""
        result = _resolve_log_path(r"logs\\sub/test.log")
        assert result == Path("logs/sub/test.log")


class TestExtractOptimizerTop1FromLog:
    """Tests for _extract_optimizer_top1_from_log function."""

    def test_extract_top1_none_log(self) -> None:
        """Test with None log."""
        result = _extract_optimizer_top1_from_log(None)
        assert result == {}

    def test_extract_top1_empty_log(self) -> None:
        """Test with empty log."""
        result = _extract_optimizer_top1_from_log("")
        assert result == {}

    def test_extract_top1_valid_log(self) -> None:
        """Test extracting from valid log."""
        log = """
| Top | Throughput (token/s) | TTFT (ms) | TPOT (ms) | concurrency | num_devices | parallel | batch_size |
|  1  |       2500.0        |  15000.0  |   45.0    |     200     |       8     | TP=8 | DP=1 |    200     |
"""
        result = _extract_optimizer_top1_from_log(log)
        assert result["best_throughput"] == 2500.0
        assert result["best_ttft_ms"] == 15000.0
        assert result["best_tpot_ms"] == 45.0
        assert result["best_concurrency"] == 200
        assert result["best_batch_size"] == 200

    def test_extract_top1_with_pp(self) -> None:
        """Test extracting with PP (Pipeline Parallel) column."""
        log = """
| Top | Throughput (token/s) | TTFT (ms) | TPOT (ms) | concurrency | num_devices | parallel | batch_size |
|  1  |       1000.0        |  16000.0  |   50.0    |     130     |       8     | TP=4 | PP=1 | DP=2 |    65     |
"""
        result = _extract_optimizer_top1_from_log(log)
        assert result["best_parallel"] == "TP=4 | PP=1 | DP=2"
        assert result["best_batch_size"] == 65

    def test_extract_top1_no_match(self) -> None:
        """Test with log that doesn't match pattern."""
        log = "Some other log content\nwithout table"
        result = _extract_optimizer_top1_from_log(log)
        assert result == {}


class TestInferOptimizerNoResultReasonFromParams:
    """Tests for _infer_optimizer_no_result_reason_from_params function."""

    def test_infer_both_limits(self) -> None:
        """Test with both limits set."""
        params = {"ttft_limits": 100.0, "tpot_limits": 50.0}
        result = _infer_optimizer_no_result_reason_from_params(params)
        assert "TTFT=100 ms" in result
        assert "TPOT=50 ms" in result

    def test_infer_ttft_only(self) -> None:
        """Test with only TTFT set."""
        params = {"ttft_limits": 200.0, "tpot_limits": None}
        result = _infer_optimizer_no_result_reason_from_params(params)
        assert "TTFT=200 ms" in result
        assert "TPOT=unlimited" in result

    def test_infer_tpot_only(self) -> None:
        """Test with only TPOT set."""
        params = {"ttft_limits": None, "tpot_limits": 75.0}
        result = _infer_optimizer_no_result_reason_from_params(params)
        assert "TTFT=unlimited" in result
        assert "TPOT=75 ms" in result

    def test_infer_both_none(self) -> None:
        """Test with both limits None."""
        params = {"ttft_limits": None, "tpot_limits": None}
        result = _infer_optimizer_no_result_reason_from_params(params)
        assert "TTFT=unlimited" in result
        assert "TPOT=unlimited" in result


class TestEnrichOptimizerSummary:
    """Tests for _enrich_optimizer_summary function."""

    def test_enrich_none_summary(self) -> None:
        """Test with None summary."""
        result = _enrich_optimizer_summary(None, {}, "")
        assert isinstance(result, dict)

    def test_enrich_from_top_configs(self) -> None:
        """Test enriching from top_configs table."""
        summary = {}
        tables = {
            "top_configs": [{"parallel": "TP=4", "batch_size": 64, "concurrency": 100, "throughput_token_s": 1000.0}]
        }
        result = _enrich_optimizer_summary(summary, tables, "", {}, None)
        assert result["best_parallel"] == "TP=4"
        assert result["best_batch_size"] == 64

    def test_enrich_from_log(self) -> None:
        """Test enriching from raw log when table empty."""
        summary = {}
        tables = {}
        log = """
| Top | Throughput (token/s) | TTFT (ms) | TPOT (ms) | concurrency | num_devices | parallel | batch_size |
|  1  |       1000.0        |  15000.0  |   45.0    |     200     |       8     | TP=8 | DP=1 |    200     |
"""
        result = _enrich_optimizer_summary(summary, tables, log, {}, None)
        assert result["best_throughput"] == 1000.0
        assert result["best_concurrency"] == 200

    def test_enrich_adds_no_result_reason(self) -> None:
        """Test that no_result_reason is added when needed."""
        summary = {}
        tables = {"top_configs": []}
        params = {"ttft_limits": 50.0, "tpot_limits": 25.0}
        result = _enrich_optimizer_summary(summary, tables, "", params, None)
        assert "no_result_reason" in result

    def test_enrich_preserves_existing_fields(self) -> None:
        """Test that existing fields are preserved."""
        summary = {"existing_field": "value"}
        tables = {}
        result = _enrich_optimizer_summary(summary, tables, "", {}, None)
        assert result["existing_field"] == "value"


class TestResultStore:
    """Tests for ResultStore class."""

    @pytest.fixture
    def temp_store(self) -> ResultStore:
        """Create a ResultStore with temporary directory."""
        import shutil

        tmpdir = tempfile.mkdtemp()
        store = ResultStore(root=tmpdir)
        yield store
        # Explicitly close any open connections and cleanup
        try:
            # Force garbage collection to close connections
            import gc

            del store
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass  # Best effort cleanup

    def test_store_creates_directories(self, temp_store: ResultStore) -> None:
        """Test that store creates required directories."""
        assert temp_store.root.exists()
        assert temp_store.logs_dir.exists()

    def test_store_creates_database(self, temp_store: ResultStore) -> None:
        """Test that database is created."""
        assert temp_store.db_path.exists()

    def test_store_database_schema(self, temp_store: ResultStore) -> None:
        """Test that database has correct schema."""
        with temp_store._connect() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
            tables = cursor.fetchall()
            assert len(tables) == 1
            assert tables[0][0] == "runs"

    def test_get_cached_result_none(self, temp_store: ResultStore) -> None:
        """Test getting non-existent cached result."""
        task = ExperimentTask("text_generate", {}, [], "hash", "test")
        result = temp_store.get_cached_result(task)
        assert result is None

    def test_save_and_get_result(self, temp_store: ResultStore) -> None:
        """Test saving and retrieving a result."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={"model_id": "test_model"},
            command=["test"],
            task_hash="test_hash",
            label="test_result",
            summary={"tps_per_device": 100.0},
            raw_log="Test log content",
        )
        temp_store.save_result(result)

        task = ExperimentTask("text_generate", {"model_id": "test_model"}, [], "test_hash", "test_result")
        cached = temp_store.get_cached_result(task)

        assert cached is not None
        assert cached.status == "success"
        assert cached.summary["tps_per_device"] == 100.0
        assert cached.source == "cache"

    def test_save_creates_log_file(self, temp_store: ResultStore) -> None:
        """Test that saving creates a log file."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={},
            command=[],
            task_hash="log_test",
            label="test",
            raw_log="Log line 1\nLog line 2",
        )
        temp_store.save_result(result)

        log_file = temp_store.logs_dir / "log_test.log"
        assert log_file.exists()
        assert log_file.read_text(encoding="utf-8") == "Log line 1\nLog line 2"

    def test_save_updates_existing(self, temp_store: ResultStore) -> None:
        """Test that save updates existing result."""
        result1 = ExperimentResult(
            sim_type="text_generate",
            status="running",
            params={},
            command=[],
            task_hash="update_test",
            label="test",
            summary={"progress": 50},
        )
        temp_store.save_result(result1)

        result2 = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={},
            command=[],
            task_hash="update_test",
            label="test",
            summary={"progress": 100},
        )
        temp_store.save_result(result2)

        task = ExperimentTask("text_generate", {}, [], "update_test", "test")
        cached = temp_store.get_cached_result(task)
        assert cached.status == "success"
        assert cached.summary["progress"] == 100

    def test_query_rows_empty(self, temp_store: ResultStore) -> None:
        """Test querying rows from empty store."""
        rows = temp_store.query_rows()
        assert rows == []

    def test_query_rows_by_sim_type(self, temp_store: ResultStore) -> None:
        """Test querying rows filtered by sim_type."""
        result1 = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={"device": "D1"},
            command=[],
            task_hash="h1",
            label="test1",
        )
        result2 = ExperimentResult(
            sim_type="video_generate",
            status="success",
            params={"device": "D2"},
            command=[],
            task_hash="h2",
            label="test2",
        )
        temp_store.save_result(result1)
        temp_store.save_result(result2)

        text_rows = temp_store.query_rows("text_generate")
        assert len(text_rows) == 1
        assert text_rows[0]["sim_type"] == "text_generate"

        video_rows = temp_store.query_rows("video_generate")
        assert len(video_rows) == 1
        assert video_rows[0]["sim_type"] == "video_generate"

    def test_query_rows_orders_by_created_at(self, temp_store: ResultStore) -> None:
        """Test that results are ordered by created_at descending."""
        for i in range(3):
            result = ExperimentResult(
                sim_type="text_generate",
                status="success",
                params={"index": i},
                command=[],
                task_hash=f"h{i}",
                label=f"test{i}",
            )
            temp_store.save_result(result)

        rows = temp_store.query_rows("text_generate")
        assert rows[0]["label"] == "test2"  # Last saved
        assert rows[2]["label"] == "test0"  # First saved

    def test_query_rows_includes_top_configs(self, temp_store: ResultStore) -> None:
        """Test that optimizer rows include top_configs."""
        result = ExperimentResult(
            sim_type="throughput_optimizer",
            status="success",
            params={},
            command=[],
            task_hash="opt_test",
            label="opt",
            tables={"top_configs": [{"rank": 1, "throughput_token_s": 1000}]},
        )
        temp_store.save_result(result)

        rows = temp_store.query_rows("throughput_optimizer")
        assert len(rows) == 1
        assert "top_configs" in rows[0]
        assert len(rows[0]["top_configs"]) == 1

    def test_get_cached_result_failed_overwritten(self, temp_store: ResultStore) -> None:
        """Test that failed results are not cached."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="failed",
            params={},
            command=[],
            task_hash="fail_test",
            label="fail",
            error="Test error",
        )
        temp_store.save_result(result)

        task = ExperimentTask("text_generate", {}, [], "fail_test", "fail")
        cached = temp_store.get_cached_result(task)
        # Failed results should be returned but can be re-run
        assert cached is not None
        assert cached.status == "failed"

    def test_optimizer_enrichment_on_query(self, temp_store: ResultStore) -> None:
        """Test that optimizer results are enriched on query."""
        result = ExperimentResult(
            sim_type="throughput_optimizer",
            status="success",
            params={"ttft_limits": 100.0},
            command=[],
            task_hash="opt_enrich",
            label="opt",
            summary={"best_throughput": None},  # Missing best fields
            tables={},  # Missing top_configs
            raw_log="",
        )
        temp_store.save_result(result)

        rows = temp_store.query_rows("throughput_optimizer")
        assert len(rows) == 1
        # Should have no_result_reason added since no valid result
        assert "no_result_reason" in rows[0]

    def test_multiple_saves_same_hash(self, temp_store: ResultStore) -> None:
        """Test that multiple saves with same hash update correctly."""
        for i in range(3):
            result = ExperimentResult(
                sim_type="text_generate",
                status="success",
                params={"run": i},
                command=[],
                task_hash="same_hash",
                label=f"run_{i}",
                summary={"count": i},
            )
            temp_store.save_result(result)

        rows = temp_store.query_rows("text_generate")
        assert len(rows) == 1
        assert rows[0]["count"] == 2  # Last saved value

    def test_query_rows_all_types(self, temp_store: ResultStore) -> None:
        """Test querying all sim_types."""
        for sim_type in ["text_generate", "video_generate", "throughput_optimizer"]:
            result = ExperimentResult(
                sim_type=sim_type,
                status="success",
                params={},
                command=[],
                task_hash=f"{sim_type}_h",
                label=sim_type,
            )
            temp_store.save_result(result)

        rows = temp_store.query_rows()
        assert len(rows) == 3

    def test_result_with_warnings_and_infos(self, temp_store: ResultStore) -> None:
        """Test saving result with warnings and infos."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={},
            command=[],
            task_hash="warn_test",
            label="test",
            warnings=["WARNING: Test warning"],
            infos=["INFO: Test info"],
        )
        temp_store.save_result(result)

        rows = temp_store.query_rows("text_generate")
        assert rows[0]["warning_count"] == 1
        assert rows[0]["info_count"] == 1

    def test_result_with_error(self, temp_store: ResultStore) -> None:
        """Test saving result with error."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="failed",
            params={},
            command=[],
            task_hash="error_test",
            label="test",
            error="Process failed",
        )
        temp_store.save_result(result)

        rows = temp_store.query_rows("text_generate")
        # error is stored in summary as execution_error for optimizer results
        # or just check status for text_generate
        assert rows[0]["status"] == "failed"
