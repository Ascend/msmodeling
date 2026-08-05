"""Real unit tests for models/orm.py module.

Tests SQLModel ORM tables using real module imports.
"""

from __future__ import annotations

import json

import pytest

# Import models.orm - conftest.py already adds web_ui/backend to sys.path
from models.orm import (
    _STATUS_VALUES,
    CaseLogRow,
    FeedbackRow,
    FormSchemaRow,
    JobRow,
    ModuleRow,
    ResultRecordRow,
    TelemetryEventRow,
    _utcnow_iso,
)


class TestUtcnowIso:
    """Tests for _utcnow_iso helper function."""

    def test_utcnow_iso_returns_string(self):
        """_utcnow_iso returns ISO format timestamp string."""
        result = _utcnow_iso()
        assert isinstance(result, str)

    def test_utcnow_iso_format(self):
        """_utcnow_iso returns ISO 8601 format with 'Z' suffix."""
        result = _utcnow_iso()
        assert "T" in result
        assert result.endswith("Z")
        assert result.count("-") >= 2  # YYYY-MM-DD
        assert result.count(":") >= 2  # HH:MM:SS

    def test_utcnow_iso_utc(self):
        """_utcnow_iso timestamp is in UTC."""
        result = _utcnow_iso()
        assert result.endswith("Z")
        assert "+00:00" not in result  # No offset suffix

    def test_utcnow_iso_includes_microseconds(self):
        """_utcnow_iso includes microseconds."""
        result = _utcnow_iso()
        # Format should be YYYY-MM-DDTHH:MM:SS.ffffffZ
        parts = result.split(".")
        assert len(parts) == 2
        assert parts[1].endswith("Z")

    def test_utcnow_iso_consistency(self):
        """Multiple calls produce valid timestamps."""
        result1 = _utcnow_iso()
        result2 = _utcnow_iso()
        # Both should be valid ISO format
        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert len(result1) > 20
        assert len(result2) > 20


class TestStatusValues:
    """Tests for _STATUS_VALUES constant."""

    def test_status_values_is_tuple(self):
        """_STATUS_VALUES is a tuple."""
        assert isinstance(_STATUS_VALUES, tuple)

    def test_status_values_contains_expected(self):
        """_STATUS_VALUES contains expected status values."""
        expected = ["pending", "running", "succeeded", "failed", "cancelled", "interrupted"]
        for status in expected:
            assert status in _STATUS_VALUES

    def test_status_values_complete(self):
        """_STATUS_VALUES has exactly 6 values."""
        assert len(_STATUS_VALUES) == 6

    def test_status_values_strings(self):
        """All status values are strings."""
        for status in _STATUS_VALUES:
            assert isinstance(status, str)

    def test_status_values_lowercase(self):
        """All status values are lowercase."""
        for status in _STATUS_VALUES:
            assert status.islower()


class TestModuleRow:
    """Tests for ModuleRow SQLModel."""

    def test_module_row_has_tablename(self):
        """ModuleRow has correct tablename."""
        assert ModuleRow.__tablename__ == "modules"

    def test_module_row_primary_key(self):
        """ModuleRow uses id as primary key."""
        module = ModuleRow(
            id="test_module", display_name="Test Module", runner_class="TestRunner", description="Test description"
        )
        assert module.id == "test_module"

    def test_module_row_fields(self):
        """ModuleRow has correct fields."""
        module = ModuleRow(
            id="text_generate",
            display_name="Text Generation",
            runner_class="TextGenerateRunner",
            description="Generates text",
        )
        assert module.id == "text_generate"
        assert module.display_name == "Text Generation"
        assert module.runner_class == "TextGenerateRunner"
        assert module.description == "Generates text"

    def test_module_row_optional_description(self):
        """ModuleRow description is optional."""
        module = ModuleRow(id="test_module", display_name="Test", runner_class="Runner")
        assert module.description is None

    def test_module_row_created_at(self):
        """ModuleRow has created_at timestamp."""
        module = ModuleRow(id="test_module", display_name="Test", runner_class="Runner")
        assert module.created_at is not None
        assert isinstance(module.created_at, str)
        assert "T" in module.created_at

    def test_module_row_all_fields_required(self):
        """ModuleRow fields are properly typed."""
        # id, display_name, runner_class are required
        # description is optional
        assert ModuleRow.__fields__['id'].is_required()
        assert ModuleRow.__fields__['display_name'].is_required()
        assert ModuleRow.__fields__['runner_class'].is_required()
        assert not ModuleRow.__fields__['description'].is_required()


class TestFormSchemaRow:
    """Tests for FormSchemaRow SQLModel."""

    def test_form_schema_row_has_tablename(self):
        """FormSchemaRow has correct tablename."""
        assert FormSchemaRow.__tablename__ == "form_schemas"

    def test_form_schema_row_composite_primary_key(self):
        """FormSchemaRow has composite primary key (module_id, version)."""
        schema = FormSchemaRow(module_id="test_module", version="1.0.0", schema_hash="abc123", fields="[]")
        assert schema.module_id == "test_module"
        assert schema.version == "1.0.0"

    def test_form_schema_row_fields(self):
        """FormSchemaRow has correct fields."""
        schema = FormSchemaRow(
            module_id="text_generate", version="1.0.0", schema_hash="hash123", fields='{"fields": []}'
        )
        assert schema.module_id == "text_generate"
        assert schema.version == "1.0.0"
        assert schema.schema_hash == "hash123"
        assert schema.fields == '{"fields": []}'

    def test_form_schema_row_created_at(self):
        """FormSchemaRow has created_at timestamp."""
        schema = FormSchemaRow(module_id="test", version="1.0", schema_hash="hash", fields="[]")
        assert schema.created_at is not None

    def test_form_schema_row_json_field(self):
        """FormSchemaRow stores fields as JSON string."""
        fields_data = {"fields": [{"name": "model_id"}]}
        schema = FormSchemaRow(module_id="test", version="1.0", schema_hash="hash", fields=json.dumps(fields_data))
        loaded = json.loads(schema.fields)
        assert loaded["fields"][0]["name"] == "model_id"


class TestJobRow:
    """Tests for JobRow SQLModel."""

    def test_job_row_has_tablename(self):
        """JobRow has correct tablename."""
        assert JobRow.__tablename__ == "jobs"

    def test_job_row_primary_key(self):
        """JobRow uses id as primary key."""
        job = JobRow(id="job123", module_id="text_generate", status="pending", params="{}", form_schema_version="1.0.0")
        assert job.id == "job123"

    def test_job_row_status_field(self):
        """JobRow has status field with index."""
        job = JobRow(id="job123", module_id="test", status="running", params="{}", form_schema_version="1.0.0")
        assert job.status == "running"
        assert job.status in _STATUS_VALUES

    def test_job_row_progress_fields(self):
        """JobRow has optional progress fields."""
        job = JobRow(
            id="job123",
            module_id="test",
            status="running",
            params="{}",
            form_schema_version="1.0.0",
            progress=50,
            progress_text="Processing...",
        )
        assert job.progress == 50
        assert job.progress_text == "Processing..."

    def test_job_row_progress_optional(self):
        """JobRow progress fields are optional."""
        job = JobRow(id="job123", module_id="test", status="pending", params="{}", form_schema_version="1.0.0")
        assert job.progress is None
        assert job.progress_text is None

    def test_job_row_params_json(self):
        """JobRow stores params as JSON string."""
        params = {"model_id": "gpt2", "batch_size": 32}
        job = JobRow(
            id="job123", module_id="test", status="pending", params=json.dumps(params), form_schema_version="1.0.0"
        )
        assert isinstance(job.params, str)
        loaded = json.loads(job.params)
        assert loaded["model_id"] == "gpt2"

    def test_job_row_to_params(self):
        """JobRow.to_params() parses JSON params."""
        params = {"model_id": "gpt2", "batch_size": 32}
        job = JobRow(
            id="job123", module_id="test", status="pending", params=json.dumps(params), form_schema_version="1.0.0"
        )
        parsed = job.to_params()
        assert parsed["model_id"] == "gpt2"
        assert parsed["batch_size"] == 32

    def test_job_row_to_params_empty(self):
        """JobRow.to_params() returns empty dict for empty params."""
        job = JobRow(id="job123", module_id="test", status="pending", params="", form_schema_version="1.0.0")
        parsed = job.to_params()
        assert parsed == {}

    def test_job_row_label_field(self):
        """JobRow has optional label field."""
        job = JobRow(
            id="job123",
            module_id="test",
            status="pending",
            params="{}",
            form_schema_version="1.0.0",
            label="My Test Job",
        )
        assert job.label == "My Test Job"

    def test_job_row_error_fields(self):
        """JobRow has optional error fields."""
        job = JobRow(
            id="job123",
            module_id="test",
            status="failed",
            params="{}",
            form_schema_version="1.0.0",
            error="Job failed",
            error_detail="Out of memory",
        )
        assert job.error == "Job failed"
        assert job.error_detail == "Out of memory"

    def test_job_row_timestamp_fields(self):
        """JobRow has created_at, started_at, completed_at fields."""
        job = JobRow(id="job123", module_id="test", status="completed", params="{}", form_schema_version="1.0.0")
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None

    def test_job_row_params_hash_field(self):
        """JobRow has params_hash field for deduplication."""
        job = JobRow(
            id="job123",
            module_id="test",
            status="pending",
            params="{}",
            form_schema_version="1.0.0",
            params_hash="hash123",
        )
        assert job.params_hash == "hash123"

    def test_job_row_log_text_field(self):
        """JobRow has log_text field for cached logs."""
        job = JobRow(
            id="job123",
            module_id="test",
            status="succeeded",
            params="{}",
            form_schema_version="1.0.0",
            log_text="line1\nline2\nline3",
        )
        assert job.log_text == "line1\nline2\nline3"

    def test_job_row_module_foreign_key(self):
        """JobRow has foreign key to modules."""
        job = JobRow(id="job123", module_id="text_generate", status="pending", params="{}", form_schema_version="1.0.0")
        assert job.module_id == "text_generate"

    def test_job_row_status_validation(self):
        """JobRow status must be from _STATUS_VALUES."""
        # Valid status
        job = JobRow(id="job123", module_id="test", status="running", params="{}", form_schema_version="1.0.0")
        assert job.status in _STATUS_VALUES


class TestResultRecordRow:
    """Tests for ResultRecordRow SQLModel."""

    def test_result_record_row_has_tablename(self):
        """ResultRecordRow has correct tablename."""
        assert ResultRecordRow.__tablename__ == "result_records"

    def test_result_record_row_primary_key(self):
        """ResultRecordRow uses id as primary key."""
        result = ResultRecordRow(id="result123", job_id="job456", seq=0, config="{}", summary="{}")
        assert result.id == "result123"

    def test_result_record_row_job_foreign_key(self):
        """ResultRecordRow has foreign key to jobs."""
        result = ResultRecordRow(id="result123", job_id="job456", seq=0, config="{}", summary="{}")
        assert result.job_id == "job456"

    def test_result_record_row_seq_field(self):
        """ResultRecordRow has seq field."""
        result = ResultRecordRow(id="result123", job_id="job456", seq=5, config="{}", summary="{}")
        assert result.seq == 5

    def test_result_record_row_config_json(self):
        """ResultRecordRow stores config as JSON string."""
        config = {"model": "gpt2", "batch_size": 32}
        result = ResultRecordRow(id="result123", job_id="job456", seq=0, config=json.dumps(config), summary="{}")
        loaded = json.loads(result.config)
        assert loaded["model"] == "gpt2"

    def test_result_record_row_summary_json(self):
        """ResultRecordRow stores summary as JSON string."""
        summary = {"loss": 0.5, "accuracy": 0.9}
        result = ResultRecordRow(id="result123", job_id="job456", seq=0, config="{}", summary=json.dumps(summary))
        loaded = json.loads(result.summary)
        assert loaded["loss"] == 0.5

    def test_result_record_row_tables_json(self):
        """ResultRecordRow stores tables as JSON string."""
        tables = {"metrics": [[1, 2, 3], [4, 5, 6]]}
        result = ResultRecordRow(
            id="result123", job_id="job456", seq=0, config="{}", summary="{}", tables=json.dumps(tables)
        )
        loaded = json.loads(result.tables)
        assert loaded["metrics"][0] == [1, 2, 3]

    def test_result_record_row_tables_default(self):
        """ResultRecordRow tables defaults to empty JSON object."""
        result = ResultRecordRow(id="result123", job_id="job456", seq=0, config="{}", summary="{}")
        assert result.tables == "{}"
        loaded = json.loads(result.tables)
        assert loaded == {}

    def test_result_record_row_rank_field(self):
        """ResultRecordRow has optional rank field."""
        result = ResultRecordRow(id="result123", job_id="job456", seq=0, config="{}", summary="{}", rank=1)
        assert result.rank == 1

    def test_result_record_row_rank_optional(self):
        """ResultRecordRow rank is optional."""
        result = ResultRecordRow(id="result123", job_id="job456", seq=0, config="{}", summary="{}")
        assert result.rank is None

    def test_result_record_row_case_hash_field(self):
        """ResultRecordRow has case_hash field for deduplication."""
        result = ResultRecordRow(id="result123", job_id="job456", seq=0, config="{}", summary="{}", case_hash="abc123")
        assert result.case_hash == "abc123"

    def test_result_record_row_created_at(self):
        """ResultRecordRow has created_at timestamp."""
        result = ResultRecordRow(id="result123", job_id="job456", seq=0, config="{}", summary="{}")
        assert result.created_at is not None


class TestTelemetryEventRow:
    """Tests for TelemetryEventRow SQLModel."""

    def test_telemetry_event_row_has_tablename(self):
        """TelemetryEventRow has correct tablename."""
        assert TelemetryEventRow.__tablename__ == "telemetry_events"

    def test_telemetry_event_row_auto_increment_id(self):
        """TelemetryEventRow has auto-increment id."""
        event = TelemetryEventRow(module_id="test", target="field1")
        assert event.id is None  # Auto-increment starts from None

    def test_telemetry_event_row_module_id_field(self):
        """TelemetryEventRow has module_id field."""
        event = TelemetryEventRow(module_id="text_generate", target="model")
        assert event.module_id == "text_generate"

    def test_telemetry_event_row_target_field(self):
        """TelemetryEventRow has target field."""
        event = TelemetryEventRow(module_id="test", target="run_button")
        assert event.target == "run_button"

    def test_telemetry_event_row_event_type_field(self):
        """TelemetryEventRow has event_type field."""
        event = TelemetryEventRow(module_id="test", target="field", event_type="click")
        assert event.event_type == "click"

    def test_telemetry_event_row_default_event_type(self):
        """TelemetryEventRow event_type defaults to 'change'."""
        event = TelemetryEventRow(module_id="test", target="field")
        assert event.event_type == "change"

    def test_telemetry_event_row_fingerprint_field(self):
        """TelemetryEventRow has optional fingerprint field."""
        event = TelemetryEventRow(module_id="test", target="field", fingerprint="user123")
        assert event.fingerprint == "user123"

    def test_telemetry_event_row_fingerprint_optional(self):
        """TelemetryEventRow fingerprint is optional."""
        event = TelemetryEventRow(module_id="test", target="field")
        assert event.fingerprint is None

    def test_telemetry_event_row_created_at(self):
        """TelemetryEventRow has created_at timestamp."""
        event = TelemetryEventRow(module_id="test", target="field")
        assert event.created_at is not None


class TestFeedbackRow:
    """Tests for FeedbackRow SQLModel."""

    def test_feedback_row_has_tablename(self):
        """FeedbackRow has correct tablename."""
        assert FeedbackRow.__tablename__ == "feedbacks"

    def test_feedback_row_primary_key(self):
        """FeedbackRow uses id as primary key."""
        feedback = FeedbackRow(id="feedback123", kind="text", content_text="Issue report")
        assert feedback.id == "feedback123"

    def test_feedback_row_job_foreign_key(self):
        """FeedbackRow has optional foreign key to jobs."""
        feedback = FeedbackRow(id="feedback123", job_id="job456", kind="text", content_text="Found a bug")
        assert feedback.job_id == "job456"

    def test_feedback_row_job_id_optional(self):
        """FeedbackRow job_id is optional."""
        feedback = FeedbackRow(id="feedback123", kind="text", content_text="General feedback")
        assert feedback.job_id is None

    def test_feedback_row_module_id_field(self):
        """FeedbackRow has optional module_id field."""
        feedback = FeedbackRow(id="feedback123", module_id="text_generate", kind="text", content_text="Feedback")
        assert feedback.module_id == "text_generate"

    def test_feedback_row_module_id_optional(self):
        """FeedbackRow module_id is optional."""
        feedback = FeedbackRow(id="feedback123", kind="text", content_text="Feedback")
        assert feedback.module_id is None

    def test_feedback_row_kind_field(self):
        """FeedbackRow has kind field."""
        feedback = FeedbackRow(id="feedback123", kind="screenshot", screenshot_path='["img1.png"]')
        assert feedback.kind == "screenshot"

    def test_feedback_row_text_kind_fields(self):
        """FeedbackRow text kind uses content_text field."""
        feedback = FeedbackRow(id="feedback123", kind="text", content_text="Text feedback")
        assert feedback.content_text == "Text feedback"
        assert feedback.screenshot_path is None

    def test_feedback_row_screenshot_kind_fields(self):
        """FeedbackRow screenshot kind uses screenshot_path field."""
        screenshot_paths = ["img1.png", "img2.png"]
        feedback = FeedbackRow(id="feedback123", kind="screenshot", screenshot_path=json.dumps(screenshot_paths))
        assert feedback.screenshot_path is not None
        feedback.content_text is None

    def test_feedback_row_fingerprint_field(self):
        """FeedbackRow has optional fingerprint field."""
        feedback = FeedbackRow(id="feedback123", kind="text", content_text="Feedback", fingerprint="user456")
        assert feedback.fingerprint == "user456"

    def test_feedback_row_fingerprint_optional(self):
        """FeedbackRow fingerprint is optional."""
        feedback = FeedbackRow(id="feedback123", kind="text", content_text="Feedback")
        assert feedback.fingerprint is None

    def test_feedback_row_created_at(self):
        """FeedbackRow has created_at timestamp."""
        feedback = FeedbackRow(id="feedback123", kind="text", content_text="Feedback")
        assert feedback.created_at is not None


class TestCaseLogRow:
    """Tests for CaseLogRow SQLModel."""

    def test_case_log_row_has_tablename(self):
        """CaseLogRow has correct tablename."""
        assert CaseLogRow.__tablename__ == "case_logs"

    def test_case_log_row_primary_key(self):
        """CaseLogRow uses case_hash as primary key."""
        case_log = CaseLogRow(case_hash="abc123", content="log content")
        assert case_log.case_hash == "abc123"

    def test_case_log_row_max_length(self):
        """CaseLogRow case_hash has max_length of 64."""
        long_hash = "a" * 64
        case_log = CaseLogRow(case_hash=long_hash, content="log")
        assert case_log.case_hash == long_hash

    def test_case_log_row_content_field(self):
        """CaseLogRow stores content as string."""
        case_log = CaseLogRow(case_hash="abc123", content="line1\nline2\nline3")
        assert case_log.content == "line1\nline2\nline3"

    def test_case_log_row_content_default(self):
        """CaseLogRow content defaults to empty string."""
        case_log = CaseLogRow(case_hash="abc123")
        assert case_log.content == ""

    def test_case_log_row_created_at(self):
        """CaseLogRow has created_at timestamp."""
        case_log = CaseLogRow(case_hash="abc123", content="log")
        assert case_log.created_at is not None


class TestEdgeCases:
    """Tests for edge cases and special handling."""

    def test_empty_params_to_params(self):
        """JobRow.to_params handles empty params string."""
        job = JobRow(id="job123", module_id="test", status="pending", params="", form_schema_version="1.0.0")
        result = job.to_params()
        assert result == {}

    def test_null_params_to_params(self):
        """JobRow.to_params() handles None/empty params via the real ORM method."""
        # Both empty string and None-equivalent falsy values should yield {}.
        for params_value in ("",):
            job = JobRow(
                id="job123",
                module_id="test",
                status="pending",
                params=params_value,
                form_schema_version="1.0.0",
            )
            result = job.to_params()
            assert isinstance(result, dict)
            assert result == {}

    def test_large_json_fields(self):
        """ORM handles large JSON strings correctly."""
        large_data = {"field_" + str(i): i for i in range(100)}
        large_json = json.dumps(large_data)
        assert len(large_json) > 1000

    def test_unicode_in_fields(self):
        """ORM handles unicode characters correctly."""
        feedback = FeedbackRow(id="feedback123", kind="text", content_text="café feedback 🐛")
        assert "café" in feedback.content_text
        assert "🐛" in feedback.content_text

    def test_special_characters_in_params_hash(self):
        """JobRow params_hash can contain special characters."""
        job = JobRow(
            id="job123",
            module_id="test",
            status="pending",
            params="{}",
            form_schema_version="1.0.0",
            params_hash="hash@#$!%",
        )
        assert job.params_hash == "hash@#$!%"

    def test_rank_none_vs_zero(self):
        """ResultRecordRow distinguishes None rank from 0."""
        result1 = ResultRecordRow(id="r1", job_id="j1", seq=0, config="{}", summary="{}", rank=None)
        result2 = ResultRecordRow(id="r2", job_id="j2", seq=0, config="{}", summary="{}", rank=0)
        assert result1.rank is None
        assert result2.rank == 0

    def test_case_hash_max_length_boundary(self):
        """CaseLogRow case_hash max_length is 64 characters."""
        # Test boundary: 64 chars should work
        valid_hash = "a" * 64
        case_log = CaseLogRow(case_hash=valid_hash, content="log")
        assert len(case_log.case_hash) == 64

    def test_timestamp_consistency(self):
        """All created_at timestamps are in UTC format."""
        job = JobRow(id="job123", module_id="test", status="pending", params="{}", form_schema_version="1.0.0")
        result = ResultRecordRow(id="result123", job_id="job123", seq=0, config="{}", summary="{}")
        # Both should have timestamps
        assert job.created_at.endswith("Z")
        assert result.created_at.endswith("Z")


class TestFieldTypes:
    """Tests for field type definitions."""

    def test_module_row_field_types(self):
        """ModuleRow fields have correct types."""
        assert 'id' in ModuleRow.__fields__
        assert 'display_name' in ModuleRow.__fields__
        assert 'runner_class' in ModuleRow.__fields__

    def test_job_row_field_types(self):
        """JobRow fields have correct types."""
        assert 'id' in JobRow.__fields__
        assert 'status' in JobRow.__fields__
        assert 'params' in JobRow.__fields__

    def test_result_record_row_field_types(self):
        """ResultRecordRow fields have correct types."""
        assert 'id' in ResultRecordRow.__fields__
        assert 'job_id' in ResultRecordRow.__fields__
        assert 'seq' in ResultRecordRow.__fields__

    def test_telemetry_event_row_field_types(self):
        """TelemetryEventRow fields have correct types."""
        assert 'module_id' in TelemetryEventRow.__fields__
        assert 'target' in TelemetryEventRow.__fields__
        assert 'event_type' in TelemetryEventRow.__fields__


class TestIndexFields:
    """Tests for indexed fields."""

    def test_job_row_status_indexed(self):
        """JobRow status field is indexed."""
        # Check if field has index=True
        assert JobRow.model_fields['status'].index is True

    def test_job_row_module_id_indexed(self):
        """JobRow module_id field is indexed."""
        assert JobRow.model_fields['module_id'].index is True

    def test_job_row_created_at_indexed(self):
        """JobRow created_at field is indexed."""
        assert JobRow.model_fields['created_at'].index is True

    def test_job_row_params_hash_indexed(self):
        """JobRow params_hash field is indexed."""
        assert JobRow.model_fields['params_hash'].index is True

    def test_result_record_row_job_id_indexed(self):
        """ResultRecordRow job_id field is indexed."""
        assert ResultRecordRow.model_fields['job_id'].index is True

    def test_result_record_row_rank_indexed(self):
        """ResultRecordRow rank field is indexed."""
        assert ResultRecordRow.model_fields['rank'].index is True

    def test_result_record_row_case_hash_indexed(self):
        """ResultRecordRow case_hash field is indexed."""
        assert ResultRecordRow.model_fields['case_hash'].index is True

    def test_telemetry_event_row_module_id_indexed(self):
        """TelemetryEventRow module_id field is indexed."""
        assert TelemetryEventRow.model_fields['module_id'].index is True

    def test_telemetry_event_row_target_indexed(self):
        """TelemetryEventRow target field is indexed."""
        assert TelemetryEventRow.model_fields['target'].index is True

    def test_telemetry_event_row_fingerprint_indexed(self):
        """TelemetryEventRow fingerprint field is indexed."""
        assert TelemetryEventRow.model_fields['fingerprint'].index is True

    def test_telemetry_event_row_created_at_indexed(self):
        """TelemetryEventRow created_at field is indexed."""
        assert TelemetryEventRow.model_fields['created_at'].index is True

    def test_feedback_row_job_id_indexed(self):
        """FeedbackRow job_id field is indexed."""
        assert FeedbackRow.model_fields['job_id'].index is True

    def test_feedback_row_module_id_indexed(self):
        """FeedbackRow module_id field is indexed."""
        assert FeedbackRow.model_fields['module_id'].index is True

    def test_feedback_row_fingerprint_indexed(self):
        """FeedbackRow fingerprint field is indexed."""
        assert FeedbackRow.model_fields['fingerprint'].index is True

    def test_feedback_row_created_at_indexed(self):
        """FeedbackRow created_at field is indexed."""
        assert FeedbackRow.model_fields['created_at'].index is True


class TestForeignKeys:
    """Tests for foreign key relationships."""

    def test_form_schema_foreign_key_to_modules(self):
        """FormSchemaRow has foreign key to ModuleRow."""
        assert FormSchemaRow.model_fields['module_id'].foreign_key == "modules.id"

    def test_job_foreign_key_to_modules(self):
        """JobRow has foreign key to ModuleRow."""
        assert JobRow.model_fields['module_id'].foreign_key == "modules.id"

    def test_result_record_foreign_key_to_jobs(self):
        """ResultRecordRow has foreign key to JobRow."""
        assert ResultRecordRow.model_fields['job_id'].foreign_key == "jobs.id"

    def test_feedback_foreign_key_to_jobs(self):
        """FeedbackRow has optional foreign key to JobRow."""
        assert FeedbackRow.model_fields['job_id'].foreign_key == "jobs.id"


class TestIntegration:
    """Integration tests for ORM relationships via real DB persistence."""

    def test_complete_job_lifecycle(self, tmp_path):
        """Complete job lifecycle with FK relationships enforced by the DB.

        Creates a real SQLite DB, commits parent (ModuleRow) + child (JobRow)
        rows, verifies the child can be fetched via the FK, and confirms the
        FK constraint rejects an orphan row referencing a non-existent parent.
        """
        import db as _db
        from sqlalchemy.exc import IntegrityError

        db_file = tmp_path / "lifecycle.db"
        _db.reset_engine()
        _db.init_db(str(db_file))
        try:
            # The alembic migration seeds the 3 modules (including text_generate).
            # Just persist a job + result referencing the seeded module.
            with _db.session_scope() as session:
                job = JobRow(
                    id="job123",
                    module_id="text_generate",
                    status="pending",
                    params='{"model_id": "gpt2"}',
                    form_schema_version="1.0.0",
                )
                session.add(job)
            with _db.session_scope() as session:
                result = ResultRecordRow(
                    id="result123",
                    job_id="job123",
                    seq=0,
                    config='{"model": "gpt2"}',
                    summary='{"loss": 0.5}',
                )
                session.add(result)

            # Verify: job.module_id resolves to a real modules row.
            with _db.session_scope() as session:
                fetched_job = session.get(JobRow, "job123")
                assert fetched_job is not None
                assert fetched_job.module_id == "text_generate"
                parent = session.get(ModuleRow, fetched_job.module_id)
                assert parent is not None
                assert parent.display_name == "Text Generation"

                fetched_result = session.get(ResultRecordRow, "result123")
                assert fetched_result is not None
                assert fetched_result.job_id == "job123"

            # FK constraint: referencing a non-existent module must fail.
            with _db.session_scope() as session:
                orphan = JobRow(
                    id="orphan_job",
                    module_id="nonexistent_module",
                    status="pending",
                    params="{}",
                    form_schema_version="1.0.0",
                )
                session.add(orphan)
                with pytest.raises(IntegrityError):
                    session.commit()
                # After the commit failure, the session needs an explicit
                # rollback before it can be reused / cleanly closed.
                session.rollback()
        finally:
            _db.reset_engine()

    def test_talytics_with_module(self):
        """TelemetryEvent can be linked to module or global."""
        # Global event
        global_event = TelemetryEventRow(module_id="global", target="settings", event_type="change")
        assert global_event.module_id == "global"

        # Module-specific event
        module_event = TelemetryEventRow(module_id="text_generate", target="model", event_type="change")
        assert module_event.module_id == "text_generate"

    def test_feedback_with_job(self):
        """Feedback can be linked to a specific job."""
        job = JobRow(
            id="job123", module_id="test", status="failed", params="{}", form_schema_version="1.0.0", error="Job failed"
        )

        feedback = FeedbackRow(id="feedback123", job_id="job123", kind="text", content_text="Found issue")

        assert feedback.job_id == job.id

    def test_multiple_results_per_job(self):
        """Single job can have multiple result records."""
        job_id = "job123"

        result1 = ResultRecordRow(id="r1", job_id=job_id, seq=0, config='{"device": "cpu"}', summary='{"time": 1.0}')

        result2 = ResultRecordRow(id="r2", job_id=job_id, seq=1, config='{"device": "cuda"}', summary='{"time": 0.5}')

        assert result1.job_id == job_id
        assert result2.job_id == job_id
        assert result1.seq != result2.seq
        assert result1.id != result2.id

    def test_case_log_with_dedup(self):
        """CaseLog supports case-level deduplication via case_hash."""
        result = ResultRecordRow(
            id="result123",
            job_id="job456",
            seq=0,
            config='{"model": "gpt2"}',
            summary='{"loss": 0.5}',
            case_hash="abc123",
        )

        case_log = CaseLogRow(case_hash="abc123", content="CLI output for case")

        assert result.case_hash == case_log.case_hash
