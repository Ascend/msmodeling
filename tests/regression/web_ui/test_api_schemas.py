"""Real unit tests for api/schemas.py.

Single test file for the shared Pydantic DTOs. Pure schema validation — no
mocks needed. Per tests/SKILL.md.
"""

from __future__ import annotations

from api.schemas import (
    JobListItem,
    JobResponse,
    JobStatusResponse,
    ModuleOut,
    OptionItem,
)


class TestOptionItem:
    """Tests for OptionItem schema."""

    def test_creates_with_value(self):
        item = OptionItem(value="foo")
        assert item.value == "foo"
        assert item.label is None

    def test_creates_with_label(self):
        item = OptionItem(value="foo", label="Foo")
        assert item.label == "Foo"

    def test_accepts_numeric_value(self):
        item = OptionItem(value=42)
        assert item.value == 42

    def test_accepts_value_any_type(self):
        item = OptionItem(value=[1, 2, 3])
        assert item.value == [1, 2, 3]


class TestModuleOut:
    """Tests for ModuleOut schema."""

    def test_creates_with_required_fields(self):
        m = ModuleOut(id="m", display_name="M", runner_class="TextRunner")
        assert m.id == "m"
        assert m.display_name == "M"
        assert m.runner_class == "TextRunner"
        assert m.description is None

    def test_creates_with_description(self):
        m = ModuleOut(id="m", display_name="M", runner_class="TextRunner", description="desc")
        assert m.description == "desc"


class TestJobResponse:
    """Tests for JobResponse (POST /api/jobs)."""

    def test_creates_response(self):
        r = JobResponse(job_id="j1", status="pending")
        assert r.job_id == "j1"
        assert r.status == "pending"


class TestJobStatusResponse:
    """Tests for JobStatusResponse (GET /api/jobs/{id})."""

    def test_creates_with_minimal_fields(self):
        r = JobStatusResponse(job_id="j1", module_id="m", status="running")
        assert r.result_ready is False
        assert r.cancel_requested is False
        assert r.progress is None
        assert r.params is None

    def test_creates_with_all_fields(self):
        r = JobStatusResponse(
            job_id="j1",
            module_id="m",
            status="succeeded",
            progress=100,
            label="test",
            params={"model": "gpt2"},
        )
        assert r.progress == 100
        assert r.params == {"model": "gpt2"}

    def test_result_ready_flag(self):
        r = JobStatusResponse(job_id="j1", module_id="m", status="succeeded", result_ready=True)
        assert r.result_ready is True

    def test_cancel_requested_flag(self):
        r = JobStatusResponse(job_id="j1", module_id="m", status="running", cancel_requested=True)
        assert r.cancel_requested is True


class TestJobListItem:
    """Tests for JobListItem schema."""

    def test_creates_with_required_fields(self):
        item = JobListItem(job_id="j1", module_id="m", status="running")
        assert item.label is None

    def test_creates_with_all_fields(self):
        item = JobListItem(job_id="j1", module_id="m", status="succeeded", label="task", progress=100)
        assert item.label == "task"
        assert item.progress == 100
