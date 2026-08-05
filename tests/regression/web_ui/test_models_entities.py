"""Real unit tests for models/entities.py module.

Tests domain entities using real module imports.
"""

from __future__ import annotations

import pytest
from models.entities import (
    CapabilityModule,
    Job,
    ResultRecord,
    _new_id,
)
from models.enums import IllegalJobTransitionError, JobStatus


class TestNewId:
    """Tests for _new_id helper function."""

    def test_returns_string(self):
        """_new_id returns a string."""
        result = _new_id()
        assert isinstance(result, str)

    def test_returns_hex(self):
        """_new_id returns hexadecimal string."""
        result = _new_id()
        # UUID4 hex is 32 characters
        assert len(result) == 32

    def test_returns_unique(self):
        """Each call returns a unique ID."""
        id1 = _new_id()
        id2 = _new_id()
        assert id1 != id2


class TestCapabilityModule:
    """Tests for CapabilityModule entity."""

    def test_create_minimal(self):
        """CapabilityModule can be created with required fields."""
        module = CapabilityModule(id="text_generate", display_name="Text Generation", runner_class="ModelRunner")
        assert module.id == "text_generate"
        assert module.display_name == "Text Generation"
        assert module.runner_class == "ModelRunner"
        assert module.description is None

    def test_create_with_description(self):
        """CapabilityModule can have optional description."""
        module = CapabilityModule(
            id="video_generate",
            display_name="Video Generation",
            runner_class="VideoGenerateRunner",
            description="Generates videos from text",
        )
        assert module.description == "Generates videos from text"

    def test_all_modules(self):
        """Create all three capability modules."""
        modules = [
            CapabilityModule("text_generate", "Text Generation", "ModelRunner"),
            CapabilityModule("video_generate", "Video Generation", "VideoGenerateRunner"),
            CapabilityModule("throughput_optimizer", "Throughput Optimizer", "ParallelRunner"),
        ]
        assert len(modules) == 3
        assert modules[0].id == "text_generate"
        assert modules[1].id == "video_generate"
        assert modules[2].id == "throughput_optimizer"


class TestJobEntity:
    """Tests for Job entity."""

    def test_create_minimal(self):
        """Job can be created with required fields."""
        job = Job(module_id="text_generate", params={"model_id": "gpt2"}, form_schema_version="1.0.0")
        assert job.module_id == "text_generate"
        assert job.params == {"model_id": "gpt2"}
        assert job.form_schema_version == "1.0.0"
        assert job.status == JobStatus.PENDING
        assert job.id is not None

    def test_default_status(self):
        """Job status defaults to PENDING."""
        job = Job(module_id="test", params={}, form_schema_version="1.0")
        assert job.status == JobStatus.PENDING

    def test_default_cancel_requested(self):
        """Job cancel_requested defaults to False."""
        job = Job(module_id="test", params={}, form_schema_version="1.0")
        assert job.cancel_requested is False

    def test_custom_id(self):
        """Job can be created with custom ID."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", id="custom_job_id")
        assert job.id == "custom_job_id"

    def test_custom_status(self):
        """Job can be created with custom status."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", status=JobStatus.RUNNING)
        assert job.status == JobStatus.RUNNING

    def test_with_label(self):
        """Job can have a label."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", label="My Test Job")
        assert job.label == "My Test Job"

    def test_with_progress(self):
        """Job can have progress fields."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", progress=50, progress_text="Processing...")
        assert job.progress == 50
        assert job.progress_text == "Processing..."

    def test_with_error(self):
        """Job can have error fields."""
        job = Job(
            module_id="test", params={}, form_schema_version="1.0", error="Job failed", error_detail="Out of memory"
        )
        assert job.error == "Job failed"
        assert job.error_detail == "Out of memory"

    def test_with_timestamps(self):
        """Job can have timestamp fields."""
        job = Job(
            module_id="test",
            params={},
            form_schema_version="1.0",
            created_at="2024-01-01T00:00:00Z",
            started_at="2024-01-01T00:01:00Z",
            completed_at="2024-01-01T00:02:00Z",
        )
        assert job.created_at == "2024-01-01T00:00:00Z"
        assert job.started_at == "2024-01-01T00:01:00Z"
        assert job.completed_at == "2024-01-01T00:02:00Z"

    def test_with_params_hash(self):
        """Job can have params_hash for deduplication."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", params_hash="abc123")
        assert job.params_hash == "abc123"

    def test_with_log_text(self):
        """Job can have log_text for cached logs."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", log_text="line1\nline2\nline3")
        assert job.log_text == "line1\nline2\nline3"

    def test_transition_pending_to_running(self):
        """Job can transition from PENDING to RUNNING."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", status=JobStatus.PENDING)
        job.transition(JobStatus.RUNNING)
        assert job.status == JobStatus.RUNNING

    def test_transition_running_to_succeeded(self):
        """Job can transition from RUNNING to SUCCEEDED."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", status=JobStatus.RUNNING)
        job.transition(JobStatus.SUCCEEDED)
        assert job.status == JobStatus.SUCCEEDED

    def test_transition_running_to_failed(self):
        """Job can transition from RUNNING to FAILED."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", status=JobStatus.RUNNING)
        job.transition(JobStatus.FAILED)
        assert job.status == JobStatus.FAILED

    def test_transition_raises_on_illegal(self):
        """Job transition raises on illegal transition."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", status=JobStatus.SUCCEEDED)
        with pytest.raises(IllegalJobTransitionError):
            job.transition(JobStatus.RUNNING)

    def test_is_terminal_pending(self):
        """is_terminal returns False for PENDING."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", status=JobStatus.PENDING)
        assert not job.is_terminal

    def test_is_terminal_running(self):
        """is_terminal returns False for RUNNING."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", status=JobStatus.RUNNING)
        assert not job.is_terminal

    def test_is_terminal_succeeded(self):
        """is_terminal returns True for SUCCEEDED."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", status=JobStatus.SUCCEEDED)
        assert job.is_terminal

    def test_is_terminal_failed(self):
        """is_terminal returns True for FAILED."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", status=JobStatus.FAILED)
        assert job.is_terminal

    def test_is_terminal_cancelled(self):
        """is_terminal returns True for CANCELLED."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", status=JobStatus.CANCELLED)
        assert job.is_terminal

    def test_is_terminal_interrupted(self):
        """is_terminal returns True for INTERRUPTED."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", status=JobStatus.INTERRUPTED)
        assert job.is_terminal

    def test_result_ready_succeeded(self):
        """result_ready returns True for SUCCEEDED."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", status=JobStatus.SUCCEEDED)
        assert job.result_ready

    def test_result_ready_not_succeeded(self):
        """result_ready returns False for non-SUCCEEDED."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", status=JobStatus.RUNNING)
        assert not job.result_ready

    def test_result_ready_failed(self):
        """result_ready returns False for FAILED."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", status=JobStatus.FAILED)
        assert not job.result_ready

    def test_cancel_requested_set(self):
        """Job can have cancel_requested set to True."""
        job = Job(module_id="test", params={}, form_schema_version="1.0", cancel_requested=True)
        assert job.cancel_requested is True


class TestResultRecord:
    """Tests for ResultRecord entity."""

    def test_create_minimal(self):
        """ResultRecord can be created with required fields."""
        result = ResultRecord(job_id="job123", seq=0, config={"model": "gpt2"}, summary={"loss": 0.5})
        assert result.job_id == "job123"
        assert result.seq == 0
        assert result.config == {"model": "gpt2"}
        assert result.summary == {"loss": 0.5}
        assert result.id is not None

    def test_default_tables(self):
        """ResultRecord tables defaults to empty dict."""
        result = ResultRecord(job_id="job123", seq=0, config={}, summary={})
        assert result.tables == {}

    def test_with_tables(self):
        """ResultRecord can have tables."""
        tables = {"metrics": [[1, 2, 3], [4, 5, 6]]}
        result = ResultRecord(job_id="job123", seq=0, config={}, summary={}, tables=tables)
        assert result.tables == tables

    def test_with_rank(self):
        """ResultRecord can have rank."""
        result = ResultRecord(job_id="job123", seq=0, config={}, summary={}, rank=1)
        assert result.rank == 1

    def test_with_case_hash(self):
        """ResultRecord can have case_hash."""
        result = ResultRecord(job_id="job123", seq=0, config={}, summary={}, case_hash="abc123")
        assert result.case_hash == "abc123"

    def test_custom_id(self):
        """ResultRecord can have custom ID."""
        result = ResultRecord(job_id="job123", seq=0, config={}, summary={}, id="custom_result_id")
        assert result.id == "custom_result_id"

    def test_with_created_at(self):
        """ResultRecord can have created_at."""
        result = ResultRecord(job_id="job123", seq=0, config={}, summary={}, created_at="2024-01-01T00:00:00Z")
        assert result.created_at == "2024-01-01T00:00:00Z"

    def test_with_case_log(self):
        """ResultRecord can have case_log."""
        result = ResultRecord(job_id="job123", seq=0, config={}, summary={}, case_log="CLI output line1\nline2")
        assert result.case_log == "CLI output line1\nline2"

    def test_seq_ordering(self, tmp_path):
        """ResultRecords fetched from the repository come back ordered by seq."""
        import db
        from models import orm  # noqa: F401  register table models
        from models.entities import Job
        from models.enums import JobStatus
        from services.repositories import JobRepository, ResultRepository

        # Set up an isolated DB with real schema.
        db.reset_engine()
        db.init_db(str(tmp_path / "seq.db"))

        try:
            # Insert the parent job (result_records.job_id has a FK to jobs.id).
            JobRepository().add(
                Job(
                    id="job1",
                    module_id="text_generate",
                    params={},
                    form_schema_version="1.0.0",
                    status=JobStatus.PENDING,
                )
            )
            repo = ResultRepository()
            # Insert records with non-sequential seq values (out of order).
            repo.add(ResultRecord(job_id="job1", seq=2, config={}, summary={}))
            repo.add(ResultRecord(job_id="job1", seq=0, config={}, summary={}))
            repo.add(ResultRecord(job_id="job1", seq=1, config={}, summary={}))

            # Fetch via the real repository (which uses ORDER BY seq).
            fetched = repo.list_for_job("job1")
            assert [r.seq for r in fetched] == [0, 1, 2]
        finally:
            db.reset_engine()

    def test_multiple_results_per_job(self):
        """Multiple ResultRecords can exist for the same job."""
        job_id = "job123"
        result1 = ResultRecord(job_id=job_id, seq=0, config={}, summary={})
        result2 = ResultRecord(job_id=job_id, seq=1, config={}, summary={})
        assert result1.job_id == job_id
        assert result2.job_id == job_id
        assert result1.seq != result2.seq
        assert result1.id != result2.id

    def test_complex_config(self):
        """ResultRecord config can have complex nested data."""
        config = {"model": "gpt2", "batch_size": 32, "device": "cuda", "options": {"opt1": True, "opt2": False}}
        result = ResultRecord(job_id="job123", seq=0, config=config, summary={})
        assert result.config["options"]["opt1"] is True

    def test_complex_summary(self):
        """ResultRecord summary can have complex nested data."""
        summary = {"metrics": {"loss": 0.5, "accuracy": 0.9}, "timing": {"forward": 1.0, "backward": 0.5}}
        result = ResultRecord(job_id="job123", seq=0, config={}, summary=summary)
        assert result.summary["metrics"]["accuracy"] == 0.9

    def test_rank_none_vs_zero(self):
        """ResultRecord distinguishes None rank from 0."""
        result1 = ResultRecord(job_id="job1", seq=0, config={}, summary={}, rank=None)
        result2 = ResultRecord(job_id="job2", seq=0, config={}, summary={}, rank=0)
        assert result1.rank is None
        assert result2.rank == 0
