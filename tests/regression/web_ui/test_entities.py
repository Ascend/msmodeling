"""Unit tests for entities module."""

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

    def test_new_id_returns_string(self):
        """Returns a string ID."""
        result = _new_id()
        assert isinstance(result, str)

    def test_new_id_is_uuid_hex(self):
        """Returns a hexadecimal UUID string."""
        result = _new_id()
        assert len(result) == 32  # UUID4 hex length
        int(result, 16)  # Should not raise

    def test_new_id_unique(self):
        """Each call returns a unique ID."""
        id1 = _new_id()
        id2 = _new_id()
        assert id1 != id2


class TestCapabilityModule:
    """Tests for CapabilityModule dataclass."""

    def test_capability_module_creation(self):
        """Can create a capability module."""
        module = CapabilityModule(
            id="text_generate",
            display_name="Text Generation",
            runner_class="ModelRunner",
        )
        assert module.id == "text_generate"
        assert module.display_name == "Text Generation"
        assert module.runner_class == "ModelRunner"
        assert module.description is None

    def test_capability_module_with_description(self):
        """Can include optional description."""
        module = CapabilityModule(
            id="video_generate",
            display_name="Video Generation",
            runner_class="VideoGenerateRunner",
            description="Video generation profiling",
        )
        assert module.description == "Video generation profiling"

    def test_capability_module_immutability(self):
        """Dataclass fields are mutable unless frozen."""
        module = CapabilityModule(
            id="test",
            display_name="Test",
            runner_class="TestRunner",
        )
        # Can modify (not frozen)
        module.description = "New description"
        assert module.description == "New description"


class TestJob:
    """Tests for Job dataclass."""

    def test_job_creation_minimal(self):
        """Can create job with minimal required fields."""
        job = Job(
            module_id="text_generate",
            params={"model": "gpt2"},
            form_schema_version="1.0.0",
        )
        assert job.module_id == "text_generate"
        assert job.params == {"model": "gpt2"}
        assert job.form_schema_version == "1.0.0"
        assert job.status == JobStatus.PENDING  # default
        assert job.id is not None  # auto-generated

    def test_job_creation_with_all_fields(self):
        """Can create job with all fields."""
        job = Job(
            module_id="video_generate",
            params={"frames": 100},
            form_schema_version="2.0.0",
            status=JobStatus.RUNNING,
            id="custom_id",
            label="Test Job",
            progress=50,
            progress_text="Processing",
            error=None,
            error_detail=None,
            created_at="2024-01-01T00:00:00Z",
            started_at="2024-01-01T00:01:00Z",
            completed_at=None,
            cancel_requested=False,
            params_hash="abc123",
            log_text="Job started",
        )
        assert job.module_id == "video_generate"
        assert job.progress == 50

    def test_job_default_status(self):
        """Default status is PENDING."""
        job = Job(
            module_id="test",
            params={},
            form_schema_version="1.0",
        )
        assert job.status == JobStatus.PENDING

    def test_job_auto_generates_id(self):
        """Job ID is auto-generated if not provided."""
        job = Job(
            module_id="test",
            params={},
            form_schema_version="1.0",
        )
        assert job.id is not None
        assert len(job.id) == 32  # UUID hex length

    def test_job_custom_id(self):
        """Can provide custom job ID."""
        job = Job(
            module_id="test",
            params={},
            form_schema_version="1.0",
            id="my_custom_id",
        )
        assert job.id == "my_custom_id"

    def test_job_transition_valid(self):
        """Can transition through valid state edges."""
        job = Job(
            module_id="test",
            params={},
            form_schema_version="1.0",
            status=JobStatus.PENDING,
        )
        job.transition(JobStatus.RUNNING)
        assert job.status == JobStatus.RUNNING

    def test_job_transition_invalid_raises(self):
        """Invalid transition raises IllegalJobTransitionError."""
        job = Job(
            module_id="test",
            params={},
            form_schema_version="1.0",
            status=JobStatus.SUCCEEDED,
        )
        with pytest.raises(IllegalJobTransitionError):
            job.transition(JobStatus.RUNNING)

    def test_job_is_terminal_property(self):
        """is_terminal property reflects state."""
        job = Job(
            module_id="test",
            params={},
            form_schema_version="1.0",
        )

        job.status = JobStatus.PENDING
        assert job.is_terminal is False

        job.status = JobStatus.RUNNING
        assert job.is_terminal is False

        job.status = JobStatus.SUCCEEDED
        assert job.is_terminal is True

        job.status = JobStatus.FAILED
        assert job.is_terminal is True

    def test_job_result_ready_property(self):
        """result_ready is True only for SUCCEEDED jobs."""
        job = Job(
            module_id="test",
            params={},
            form_schema_version="1.0",
        )

        job.status = JobStatus.RUNNING
        assert job.result_ready is False

        job.status = JobStatus.SUCCEEDED
        assert job.result_ready is True

        job.status = JobStatus.FAILED
        assert job.result_ready is False

    def test_job_phase_c_fields(self):
        """Job can have Phase C cache fields."""
        job = Job(
            module_id="test",
            params={},
            form_schema_version="1.0",
            params_hash="hash123",
            log_text="Cached log content",
        )
        assert job.params_hash == "hash123"
        assert job.log_text == "Cached log content"

    def test_job_cancel_requested_default(self):
        """cancel_requested defaults to False."""
        job = Job(
            module_id="test",
            params={},
            form_schema_version="1.0",
        )
        assert job.cancel_requested is False


class TestResultRecord:
    """Tests for ResultRecord dataclass."""

    def test_result_record_minimal(self):
        """Can create result record with minimal fields."""
        record = ResultRecord(
            job_id="job123",
            seq=0,
            config={"model": "gpt2"},
            summary={"throughput": 100},
        )
        assert record.job_id == "job123"
        assert record.seq == 0
        assert record.config == {"model": "gpt2"}
        assert record.summary == {"throughput": 100}
        assert record.tables == {}
        assert record.rank is None
        assert record.id is not None  # auto-generated

    def test_result_record_with_all_fields(self):
        """Can create result record with all fields."""
        record = ResultRecord(
            job_id="job123",
            seq=1,
            config={"devices": ["A100"]},
            summary={"throughput_token_s": 500},
            tables={"summary": "data"},
            rank=1,
            case_hash="hash123",
            id="custom_id",
            created_at="2024-01-01T00:00:00Z",
            case_log="Output log",
        )
        assert record.rank == 1
        assert record.case_hash == "hash123"
        assert record.id == "custom_id"
        assert record.case_log == "Output log"

    def test_result_record_auto_generates_id(self):
        """Record ID is auto-generated if not provided."""
        record = ResultRecord(
            job_id="job123",
            seq=0,
            config={},
            summary={},
        )
        assert record.id is not None
        assert len(record.id) == 32

    def test_result_record_tables_default(self):
        """tables field defaults to empty dict."""
        record = ResultRecord(
            job_id="job123",
            seq=0,
            config={},
            summary={},
        )
        assert record.tables == {}

    def test_result_record_rank_optional(self):
        """rank is optional (None for text/video)."""
        record = ResultRecord(
            job_id="job123",
            seq=0,
            config={},
            summary={},
        )
        assert record.rank is None

    def test_result_record_case_hash_optional(self):
        """case_hash is optional."""
        record = ResultRecord(
            job_id="job123",
            seq=0,
            config={},
            summary={},
        )
        assert record.case_hash is None

    def test_result_record_case_log_optional(self):
        """case_log is optional."""
        record = ResultRecord(
            job_id="job123",
            seq=0,
            config={},
            summary={},
        )
        assert record.case_log is None

    def test_result_record_seq_stable_sort(self):
        """seq provides stable sort order within job."""
        records = [
            ResultRecord(job_id="job1", seq=2, config={}, summary={}),
            ResultRecord(job_id="job1", seq=0, config={}, summary={}),
            ResultRecord(job_id="job1", seq=1, config={}, summary={}),
        ]
        sorted_records = sorted(records, key=lambda r: r.seq)
        assert [r.seq for r in sorted_records] == [0, 1, 2]

    def test_result_record_optimizer_rank(self):
        """rank is computed for optimizer results."""
        record = ResultRecord(
            job_id="optimizer_job",
            seq=0,
            config={},
            summary={},
            rank=1,  # Best configuration
        )
        assert record.rank == 1
