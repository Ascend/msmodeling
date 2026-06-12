"""Tests for web_ui.schemas module."""

from __future__ import annotations


from web_ui.schemas import ExperimentResult, ExperimentTask


class TestExperimentTask:
    """Tests for ExperimentTask dataclass."""

    def test_experiment_task_creation(self) -> None:
        """Test creating an ExperimentTask with all fields."""
        task = ExperimentTask(
            sim_type="text_generate",
            params={"model_id": "Qwen/Qwen3-32B", "device": "test_device"},
            command=["python", "-m", "cli.inference.text_generate"],
            task_hash="abc123",
            label="test_task",
        )
        assert task.sim_type == "text_generate"
        assert task.params["model_id"] == "Qwen/Qwen3-32B"
        assert task.command == ["python", "-m", "cli.inference.text_generate"]
        assert task.task_hash == "abc123"
        assert task.label == "test_task"

    def test_experiment_task_with_nested_params(self) -> None:
        """Test ExperimentTask with nested parameter structures."""
        task = ExperimentTask(
            sim_type="video_generate",
            params={
                "model_id": "test_model",
                "device": "D1",
                "cache_params": {"range": "10,20", "interval": 5},
            },
            command=["video_cmd"],
            task_hash="hash1",
            label="video_task",
        )
        assert task.params["cache_params"]["range"] == "10,20"


class TestExperimentResult:
    """Tests for ExperimentResult dataclass."""

    def test_experiment_result_creation(self) -> None:
        """Test creating an ExperimentResult with all fields."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={"model_id": "test_model"},
            command=["test", "cmd"],
            task_hash="hash123",
            label="test_result",
        )
        assert result.sim_type == "text_generate"
        assert result.status == "success"
        assert result.summary == {}
        assert result.tables == {}
        assert result.warnings == []
        assert result.infos == []
        assert result.raw_log == ""
        assert result.error is None
        assert result.source == "run"

    def test_experiment_result_with_summary(self) -> None:
        """Test ExperimentResult with summary data."""
        result = ExperimentResult(
            sim_type="optimizer",
            status="success",
            params={"device": "D1"},
            command=["opt"],
            task_hash="h",
            label="opt_result",
            summary={"best_throughput": 1000.0, "best_ttft_ms": 50.0},
        )
        assert result.summary["best_throughput"] == 1000.0
        assert result.summary["best_ttft_ms"] == 50.0

    def test_experiment_result_with_tables(self) -> None:
        """Test ExperimentResult with tables data."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={},
            command=[],
            task_hash="h",
            label="result",
            tables={
                "op_breakdown": [
                    {"name": "matmul", "analytic_total_us": 1000.0},
                    {"name": "add", "analytic_total_us": 500.0},
                ]
            },
        )
        assert len(result.tables["op_breakdown"]) == 2
        assert result.tables["op_breakdown"][0]["name"] == "matmul"

    def test_experiment_result_with_warnings_and_infos(self) -> None:
        """Test ExperimentResult with warnings and info messages."""
        result = ExperimentResult(
            sim_type="video_generate",
            status="success",
            params={},
            command=[],
            task_hash="h",
            label="result",
            warnings=["WARNING: Low memory", "WARNING: High latency"],
            infos=["INFO: Model loaded", "INFO: Cache enabled"],
        )
        assert len(result.warnings) == 2
        assert len(result.infos) == 2
        assert "Low memory" in result.warnings[0]
        assert "Model loaded" in result.infos[0]

    def test_experiment_result_with_error(self) -> None:
        """Test ExperimentResult with error status."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="failed",
            params={},
            command=[],
            task_hash="h",
            label="failed_result",
            error="Process exited with code 1",
        )
        assert result.status == "failed"
        assert result.error == "Process exited with code 1"

    def test_experiment_result_to_row(self) -> None:
        """Test to_row method merges params and summary."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={"model_id": "test_model", "device": "D1"},
            command=["cmd"],
            task_hash="h",
            label="test",
            summary={"tps_per_device": 500.0, "analytic_total_time_s": 1.5},
            warnings=["WARN"],
            infos=["INFO"],
        )
        row = result.to_row()
        assert row["sim_type"] == "text_generate"
        assert row["status"] == "success"
        assert row["model_id"] == "test_model"
        assert row["device"] == "D1"
        assert row["tps_per_device"] == 500.0
        assert row["analytic_total_time_s"] == 1.5
        assert row["warning_count"] == 1
        assert row["info_count"] == 1

    def test_experiment_result_to_dict(self) -> None:
        """Test to_dict method returns all fields as dict."""
        result = ExperimentResult(
            sim_type="optimizer",
            status="success",
            params={"model": "m1"},
            command=["opt"],
            task_hash="h",
            label="opt",
            summary={"best_throughput": 1000},
            tables={"top_configs": []},
        )
        d = result.to_dict()
        assert d["sim_type"] == "optimizer"
        assert d["status"] == "success"
        assert d["summary"]["best_throughput"] == 1000
        assert d["tables"]["top_configs"] == []

    def test_experiment_result_source_default(self) -> None:
        """Test default source is 'run'."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={},
            command=[],
            task_hash="h",
            label="test",
        )
        assert result.source == "run"

    def test_experiment_result_source_cache(self) -> None:
        """Test setting source to 'cache'."""
        result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={},
            command=[],
            task_hash="h",
            label="test",
            source="cache",
        )
        assert result.source == "cache"

    def test_experiment_result_with_raw_log(self) -> None:
        """Test ExperimentResult with raw log content."""
        log_content = "Starting simulation...\nModel loaded\nComplete"
        result = ExperimentResult(
            sim_type="text_generate",
            status="success",
            params={},
            command=[],
            task_hash="h",
            label="test",
            raw_log=log_content,
        )
        assert "Starting simulation" in result.raw_log
        assert "Model loaded" in result.raw_log
        assert "Complete" in result.raw_log

    def test_experiment_result_empty_tables(self) -> None:
        """Test ExperimentResult with empty tables."""
        result = ExperimentResult(
            sim_type="video_generate",
            status="success",
            params={},
            command=[],
            task_hash="h",
            label="test",
            tables={},
        )
        assert result.tables == {}
        assert not result.tables.get("op_breakdown")

    def test_experiment_result_nested_summary_values(self) -> None:
        """Test ExperimentResult with nested summary values."""
        result = ExperimentResult(
            sim_type="optimizer",
            status="success",
            params={},
            command=[],
            task_hash="h",
            label="test",
            summary={
                "best_parallel": "TP=4 | DP=2",
                "limits": {"ttft": 100.0, "tpot": 50.0},
            },
        )
        assert result.summary["best_parallel"] == "TP=4 | DP=2"
        assert result.summary["limits"]["ttft"] == 100.0
