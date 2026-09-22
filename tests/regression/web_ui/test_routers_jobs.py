"""Real unit tests for api/routers/jobs.py.

Single test file for the jobs router. Async handlers are awaited via
``asyncio.run()`` (``pytest-asyncio`` is declared but not installed). Real
imports + fixture-scoped mocks only, per tests/SKILL.md.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from api.routers.jobs import (
    DEFAULT_POLL_INTERVAL_MS,
    JobListResponse,
    JobResultResponse,
    JobSubmitRequest,
    _expand_job_cases,
    cancel_job,
    create_job,
    get_job,
    get_job_log,
    get_job_manager,
    get_job_result,
    get_job_trace,
    list_jobs,
    router,
)
from fastapi import HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from models.enums import JobStatus
from services.job_manager import JobManager


def _run(coro):
    """Await an async coroutine synchronously (no pytest-asyncio needed)."""
    return asyncio.run(coro)


def _job(status=JobStatus.RUNNING, params=None, module_id="text_generate", job_id="job-1", log_text=None):
    """Build a Job-like mock with a real JobStatus and all attrs the router reads."""
    job = MagicMock()
    job.id = job_id
    job.module_id = module_id
    job.form_schema_version = "1.0.0"
    job.params = params if params is not None else {"model": "gpt2"}
    job.status = status
    job.progress = 50
    job.progress_text = "halfway"
    job.label = "a task"
    job.created_at = "2026-01-01T00:00:00Z"
    job.started_at = "2026-01-01T00:00:01Z"
    job.completed_at = None
    job.error = None
    job.error_detail = None
    job.log_text = log_text
    return job


def _record(seq=0):
    rec = MagicMock()
    rec.seq = seq
    rec.rank = 1
    rec.config = {"model": "gpt2"}
    rec.summary = {"loss": 0.1}
    rec.tables = []
    rec.case_hash = "hash0"
    return rec


# ---------------------------------------------------------------------------
# Router config & helpers
# ---------------------------------------------------------------------------


class TestRouterConfiguration:
    """Tests for router setup and module-level helpers."""

    def test_router_prefix(self):
        assert router.prefix == "/api/jobs"

    def test_router_tag(self):
        assert router.tags == ["jobs"]

    def test_default_poll_interval(self):
        assert DEFAULT_POLL_INTERVAL_MS == 1500

    def test_get_job_manager_resolves_from_state(self):
        request = MagicMock()
        assert get_job_manager(request) is request.app.state.job_manager


# ---------------------------------------------------------------------------
# list_jobs
# ---------------------------------------------------------------------------


class TestListJobs:
    """Tests for the list_jobs endpoint."""

    def test_returns_paginated_items_and_total(self):
        repo = MagicMock()
        repo.list_jobs.return_value = [_job(JobStatus.SUCCEEDED, job_id="j1")]
        repo.count_jobs.return_value = 1
        result = _run(list_jobs(repo))
        assert isinstance(result, JobListResponse)
        assert result.total == 1
        assert result.items[0].job_id == "j1"
        assert result.items[0].status == "succeeded"

    def test_passes_filters_to_repository(self):
        repo = MagicMock()
        repo.list_jobs.return_value = []
        repo.count_jobs.return_value = 0
        _run(list_jobs(repo, module_id="m", status=JobStatus.RUNNING, limit=10, offset=5))
        repo.list_jobs.assert_called_once_with(module_id="m", status=JobStatus.RUNNING, limit=10, offset=5)
        repo.count_jobs.assert_called_once_with(module_id="m", status=JobStatus.RUNNING)

    def test_empty_result(self):
        repo = MagicMock()
        repo.list_jobs.return_value = []
        repo.count_jobs.return_value = 0
        result = _run(list_jobs(repo))
        assert result.items == []
        assert result.total == 0


# ---------------------------------------------------------------------------
# create_job
# ---------------------------------------------------------------------------


class TestCreateJob:
    """Tests for the create_job endpoint."""

    def test_raises_400_when_schema_not_found(self):
        with patch("api.routers.jobs.SchemaRegistry") as mock_reg:
            mock_reg.return_value.get_form_schema.return_value = None
            req = JobSubmitRequest(module_id="m", form_schema_version="9.9.9", params={"x": 1})
            with pytest.raises(HTTPException) as exc:
                _run(create_job(req, MagicMock(), AsyncMock(), MagicMock()))
        assert exc.value.status_code == 400
        assert "9.9.9" in exc.value.detail

    def test_submits_job_and_returns_body(self):
        submitted = _job(JobStatus.PENDING, params={"model-id": "test-model", "num-queries": 1, "query-length": 128})
        job_manager = AsyncMock()
        job_manager.submit_async.return_value = submitted
        job_repo = MagicMock()
        job_repo.find_succeeded_by_params_hash.return_value = None
        job_repo.find_inflight_by_params_hash.return_value = None
        with patch("api.routers.jobs.SchemaRegistry") as mock_reg:
            mock_reg.return_value.get_form_schema.return_value = {"fields": []}
            # Provide valid params including all required fields
            req = JobSubmitRequest(
                module_id="text_generate",
                form_schema_version="1.0.0",
                params={"model-id": "test-model", "num-queries": 1, "query-length": 128},
            )
            result = _run(create_job(req, job_repo, job_manager, MagicMock()))
        job_manager.submit_async.assert_awaited_once()
        # Verify the Job entity that was passed to submit_async has the right fields.
        call_args = job_manager.submit_async.call_args
        submitted_job = call_args.args[0] if call_args.args else call_args.kwargs.get("job")
        assert submitted_job.module_id == "text_generate"
        assert submitted_job.form_schema_version == "1.0.0"
        assert submitted_job.params == {"model-id": "test-model", "num-queries": 1, "query-length": 128}
        assert submitted_job.status == JobStatus.PENDING
        assert result.job_id == "job-1"
        assert result.status == "pending"

    def test_passes_explicitly_touched_to_job_entity(self):
        """explicitly_touched from the API request must propagate to the Job entity.

        Regression test: the runner reads ``job.explicitly_touched`` to compute the
        ``provided`` set for wants_provided=True validators. If the API router
        forgets to pass it through, the runner crashes with AttributeError at
        execution time — even though the Job dataclass declares the field.
        """
        submitted = _job(JobStatus.PENDING, params={"model-id": "test-model"})
        job_manager = AsyncMock()
        job_manager.submit_async.return_value = submitted
        job_repo = MagicMock()
        job_repo.find_succeeded_by_params_hash.return_value = None
        job_repo.find_inflight_by_params_hash.return_value = None
        with patch("api.routers.jobs.SchemaRegistry") as mock_reg:
            mock_reg.return_value.get_form_schema.return_value = {"fields": []}
            req = JobSubmitRequest(
                module_id="text_generate",
                form_schema_version="1.0.0",
                params={"model-id": "test-model"},
                explicitly_touched=["model-id", "num-queries"],
            )
            _run(create_job(req, job_repo, job_manager, MagicMock()))
        call_args = job_manager.submit_async.call_args
        submitted_job = call_args.args[0] if call_args.args else call_args.kwargs.get("job")
        assert submitted_job.explicitly_touched == ["model-id", "num-queries"]

    def test_explicitly_touched_defaults_to_none_when_absent(self):
        """When the frontend doesn't send explicitly_touched, Job gets None."""
        submitted = _job(JobStatus.PENDING, params={"model-id": "test-model"})
        job_manager = AsyncMock()
        job_manager.submit_async.return_value = submitted
        job_repo = MagicMock()
        job_repo.find_succeeded_by_params_hash.return_value = None
        job_repo.find_inflight_by_params_hash.return_value = None
        with patch("api.routers.jobs.SchemaRegistry") as mock_reg:
            mock_reg.return_value.get_form_schema.return_value = {"fields": []}
            req = JobSubmitRequest(
                module_id="text_generate",
                form_schema_version="1.0.0",
                params={"model-id": "test-model"},
            )
            _run(create_job(req, job_repo, job_manager, MagicMock()))
        call_args = job_manager.submit_async.call_args
        submitted_job = call_args.args[0] if call_args.args else call_args.kwargs.get("job")
        assert submitted_job.explicitly_touched is None

    def test_raises_400_when_case_expansion_fails(self):
        """_expand_job_cases_strict ValueError → 400 (catches oversized/invalid
        multi-value fields before the job is submitted).
        """
        with patch("api.routers.jobs._expand_job_cases_strict", side_effect=ValueError("too many cases")):
            with patch("api.routers.jobs.SchemaRegistry") as mock_reg:
                mock_reg.return_value.get_form_schema.return_value = {"fields": []}
                req = JobSubmitRequest(
                    module_id="text_generate",
                    form_schema_version="1.0.0",
                    params={"model-id": "test-model", "num-queries": 1, "query-length": 128},
                )
                with pytest.raises(HTTPException) as exc:
                    _run(create_job(req, MagicMock(), AsyncMock(), MagicMock()))
        assert exc.value.status_code == 400
        assert "too many cases" in exc.value.detail

    def test_raises_429_when_inflight_limit_exceeded(self):
        """JobManager.InflightLimitExceeded → 429 (local DoS defense)."""
        from services.job_manager import JobManager

        job_manager = AsyncMock()
        job_manager.InflightLimitExceeded = JobManager.InflightLimitExceeded
        job_manager.submit_async.side_effect = JobManager.InflightLimitExceeded("worker pool saturated")
        job_repo = MagicMock()
        job_repo.find_succeeded_by_params_hash.return_value = None
        job_repo.find_inflight_by_params_hash.return_value = None
        with patch("api.routers.jobs.SchemaRegistry") as mock_reg:
            mock_reg.return_value.get_form_schema.return_value = {"fields": []}
            # Provide valid params including all required fields
            req = JobSubmitRequest(
                module_id="text_generate",
                form_schema_version="1.0.0",
                params={"model-id": "test-model", "num-queries": 1, "query-length": 128},
            )
            with pytest.raises(HTTPException) as exc:
                _run(create_job(req, job_repo, job_manager, MagicMock()))
        assert exc.value.status_code == 429
        assert "saturated" in exc.value.detail

    def test_chrome_trace_file_string_normalized_and_preserved(self):
        """String chrome-trace-file is normalized to True for Job, request.params unchanged.

        Regression test: params_hash must be computed on normalized params (True),
        not the raw string, so idempotent dedup works across retries with the same
        original payload. The Job.params should store the normalized True, not the
        original string.
        """
        submitted = _job(JobStatus.PENDING, params={"model-id": "test-model", "chrome-trace-file": True})
        job_manager = AsyncMock()
        job_manager.submit_async.return_value = submitted
        job_repo = MagicMock()
        job_repo.find_succeeded_by_params_hash.return_value = None
        job_repo.find_inflight_by_params_hash.return_value = None
        with patch("api.routers.jobs.SchemaRegistry") as mock_reg:
            mock_reg.return_value.get_form_schema.return_value = {"fields": []}
            req = JobSubmitRequest(
                module_id="text_generate",
                form_schema_version="1.0.0",
                params={"model-id": "test-model", "chrome-trace-file": "true"},  # Valid boolean-like string
            )
            original_params = dict(req.params)  # snapshot before call
            _run(create_job(req, job_repo, job_manager, MagicMock()))
        # request.params should NOT be mutated (preserved for audit trail)
        assert req.params == original_params
        assert req.params["chrome-trace-file"] == "true"
        # Job.params should have normalized value (True, not the string)
        call_args = job_manager.submit_async.call_args
        submitted_job = call_args.args[0] if call_args.args else call_args.kwargs.get("job")
        assert submitted_job.params["chrome-trace-file"] is True
        # params_hash should be computed on normalized params (same as if client sent True)
        from services.params_hash import compute_params_hash

        expected_hash = compute_params_hash(
            "text_generate", "1.0.0", {"model-id": "test-model", "chrome-trace-file": True}
        )
        assert submitted_job.params_hash == expected_hash

    def test_chrome_trace_file_invalid_string_rejected(self):
        """Invalid chrome-trace-file strings are rejected with 400.

        Security test: arbitrary file paths like "/etc/passwd" must be rejected,
        not silently normalized to True. This prevents arbitrary file write attacks
        via direct HTTP requests.
        """
        job_manager = AsyncMock()
        job_repo = MagicMock()
        job_repo.find_succeeded_by_params_hash.return_value = None
        job_repo.find_inflight_by_params_hash.return_value = None
        with patch("api.routers.jobs.SchemaRegistry") as mock_reg:
            mock_reg.return_value.get_form_schema.return_value = {"fields": []}
            req = JobSubmitRequest(
                module_id="text_generate",
                form_schema_version="1.0.0",
                params={"model-id": "test-model", "chrome-trace-file": "/etc/passwd"},
            )
            with pytest.raises(HTTPException) as exc:
                _run(create_job(req, job_repo, job_manager, MagicMock()))
        assert exc.value.status_code == 400
        assert "chrome-trace-file must be a boolean-like value" in exc.value.detail

    def test_integrity_error_returns_existing_inflight_job(self):
        """IntegrityError from partial unique index returns existing in-flight job.

        Regression test: when two concurrent requests with identical params both
        pass the application-level duplicate check, the database's partial unique
        index (uq_jobs_inflight_params_hash) rejects the second INSERT. The API
        should catch IntegrityError and return the existing job instead of failing.
        """
        from sqlalchemy.exc import IntegrityError

        existing_job = _job(JobStatus.RUNNING, job_id="existing-job-id")
        job_manager = AsyncMock()
        job_manager.submit_async.side_effect = IntegrityError("mock", "mock", Exception())
        job_manager.InflightLimitExceeded = JobManager.InflightLimitExceeded
        job_repo = MagicMock()
        job_repo.find_succeeded_by_params_hash.return_value = None
        # First call (pre-submission check) returns None so execution proceeds
        # to submit_async; second call (post-IntegrityError re-check) returns
        # the existing job that won the race.
        job_repo.find_inflight_by_params_hash.side_effect = [None, existing_job]

        with patch("api.routers.jobs.SchemaRegistry") as mock_reg:
            mock_reg.return_value.get_form_schema.return_value = {"fields": []}
            req = JobSubmitRequest(
                module_id="text_generate",
                form_schema_version="1.0.0",
                params={"model-id": "test-model", "num-queries": 1, "query-length": 128},
            )
            result = _run(create_job(req, job_repo, job_manager, MagicMock()))

        # Should return the existing job, not raise an error
        assert result.job_id == "existing-job-id"
        assert result.status == "running"
        # submit_async must have been called (proves the pre-submission check
        # didn't short-circuit) and raised IntegrityError on the way.
        job_manager.submit_async.assert_awaited_once()
        # Both calls accounted for: pre-submission check + post-IntegrityError re-check.
        assert job_repo.find_inflight_by_params_hash.call_count == 2

    def test_integrity_error_returns_cached_succeeded_job(self):
        """IntegrityError followed by missing in-flight returns cached succeeded job.

        Edge case: if the in-flight job completes (SUCCEEDED) between the INSERT
        failure and the SELECT, we should fall back to the succeeded cache.
        """
        from sqlalchemy.exc import IntegrityError

        cached_job = _job(JobStatus.SUCCEEDED, job_id="cached-job-id")
        job_manager = AsyncMock()
        job_manager.submit_async.side_effect = IntegrityError("mock", "mock", Exception())
        job_manager.InflightLimitExceeded = JobManager.InflightLimitExceeded
        job_repo = MagicMock()
        # First call (pre-submission cache check) returns None so execution
        # proceeds to submit_async; second call (post-IntegrityError fallback)
        # returns the cached job.
        job_repo.find_succeeded_by_params_hash.side_effect = [None, cached_job]
        # Both in-flight queries return None: the pre-submission check and the
        # post-IntegrityError re-check (job completed between INSERT and SELECT).
        job_repo.find_inflight_by_params_hash.side_effect = [None, None]

        with patch("api.routers.jobs.SchemaRegistry") as mock_reg:
            mock_reg.return_value.get_form_schema.return_value = {"fields": []}
            req = JobSubmitRequest(
                module_id="text_generate",
                form_schema_version="1.0.0",
                params={"model-id": "test-model", "num-queries": 1, "query-length": 128},
            )
            result = _run(create_job(req, job_repo, job_manager, MagicMock()))

        # Should return the cached job
        assert result.job_id == "cached-job-id"
        assert result.status == "succeeded"
        # submit_async must have been called (proves the pre-submission cache
        # didn't short-circuit) and raised IntegrityError on the way.
        job_manager.submit_async.assert_awaited_once()


# ---------------------------------------------------------------------------
# get_job_trace
# ---------------------------------------------------------------------------


class TestGetJobTrace:
    """Tests for the get_job_trace endpoint."""

    def test_raises_404_when_job_missing(self):
        repo = MagicMock()
        repo.get.return_value = None
        with pytest.raises(HTTPException) as exc:
            _run(get_job_trace("nope", 0, repo))
        assert exc.value.status_code == 404

    def test_raises_404_when_trace_file_missing(self):
        repo = MagicMock()
        repo.get.return_value = _job()
        with patch("services.trace_store.trace_path") as mock_tp:
            mock_tp.return_value = Path("/nonexistent/trace.json")
            with pytest.raises(HTTPException) as exc:
                _run(get_job_trace("job-1", 2, repo))
        assert exc.value.status_code == 404
        assert "case 2" in exc.value.detail

    def test_returns_file_response_when_trace_exists(self, tmp_path):
        trace = tmp_path / "trace.json"
        trace.write_text("{}")
        repo = MagicMock()
        repo.get.return_value = _job()
        with patch("services.trace_store.trace_path", return_value=trace):
            response = _run(get_job_trace("job-1", 3, repo))
        assert isinstance(response, FileResponse)
        assert Path(response.path) == trace


# ---------------------------------------------------------------------------
# get_job
# ---------------------------------------------------------------------------


class TestGetJob:
    """Tests for the get_job endpoint."""

    def test_raises_404_when_job_missing(self):
        repo = MagicMock()
        repo.get.return_value = None
        with pytest.raises(HTTPException) as exc:
            _run(get_job("nope", repo, MagicMock()))
        assert exc.value.status_code == 404

    def test_returns_status_response_with_command(self):
        repo = MagicMock()
        repo.get.return_value = _job(JobStatus.RUNNING)
        job_manager = MagicMock()
        job_manager.is_cancel_requested.return_value = False
        with patch("runners._cli_command.build_cli_command_string", return_value="cli cmd"):
            result = _run(get_job("job-1", repo, job_manager))
        assert result.job_id == "job-1"
        assert result.status == "running"
        assert result.result_ready is False
        assert result.command == "cli cmd"
        assert result.poll_interval_ms == DEFAULT_POLL_INTERVAL_MS
        assert result.cancel_requested is False

    def test_result_ready_when_succeeded(self):
        repo = MagicMock()
        repo.get.return_value = _job(JobStatus.SUCCEEDED)
        job_manager = MagicMock()
        job_manager.is_cancel_requested.return_value = False
        with patch("runners._cli_command.build_cli_command_string", return_value="c"):
            result = _run(get_job("job-1", repo, job_manager))
        assert result.result_ready is True

    def test_cancel_requested_surfaced_from_manager(self):
        """get_job surfaces the in-memory cancel state from JobManager (#89).

        Previously the response read cancel_requested from the DB (always False)
        — the in-memory flag was invisible to polls. Now the route queries
        JobManager.is_cancel_requested() so the flag is visible until the
        worker cleans it up on terminal transition.
        """
        repo = MagicMock()
        repo.get.return_value = _job(JobStatus.RUNNING)
        job_manager = MagicMock()
        job_manager.is_cancel_requested.return_value = True  # flag set
        with patch("runners._cli_command.build_cli_command_string", return_value="c"):
            result = _run(get_job("job-1", repo, job_manager))
        job_manager.is_cancel_requested.assert_called_once_with("job-1")
        assert result.cancel_requested is True

    def test_command_none_when_no_params(self):
        repo = MagicMock()
        repo.get.return_value = _job(params={})
        with patch("runners._cli_command.build_cli_command_string") as mock_cmd:
            result = _run(get_job("job-1", repo, MagicMock()))
        mock_cmd.assert_not_called()
        assert result.command is None

    def test_commands_single_case_collapses_to_command(self):
        """Single-case job returns `commands` with one element equal to `command`."""
        repo = MagicMock()
        # Use a single (non-list) device so the original command equals the expanded one.
        repo.get.return_value = _job(
            module_id="video_generate",
            params={"model_id": "m", "device": "A100", "ulysses_size": "4"},
        )
        with patch("runners._cli_command.build_cli_command_string", side_effect=lambda m, p: f"{m}:{p.get('device')}"):
            result = _run(get_job("job-1", repo, MagicMock()))
        # Single device → one case → commands[0] == command
        assert result.command is not None
        assert result.commands == [result.command]
        assert result.commands[0] == "video_generate:A100"

    def test_commands_multi_case_returns_per_case_list(self):
        """Multi-case job (multi-device) returns `commands` with one entry per case."""
        repo = MagicMock()
        repo.get.return_value = _job(
            module_id="video_generate",
            params={"model_id": "m", "device": ["A100", "H100"], "ulysses_size": "4"},
        )
        with patch("runners._cli_command.build_cli_command_string", side_effect=lambda m, p: f"{m}:{p.get('device')}"):
            result = _run(get_job("job-1", repo, MagicMock()))
        assert result.commands is not None
        assert len(result.commands) == 2
        # Each per-case command shows the SINGLE device for that case (not the list).
        assert result.commands[0] == "video_generate:A100"
        assert result.commands[1] == "video_generate:H100"
        # The `command` reference still carries the ORIGINAL (multi-device) params.
        assert result.command == "video_generate:['A100', 'H100']"

    def test_chrome_trace_path_synthesized_when_enabled(self):
        """When chrome-trace-file=True, the command shows the actual trace path, not <auto>."""
        repo = MagicMock()
        repo.get.return_value = _job(
            module_id="text_generate",
            params={"model_id": "m", "chrome-trace-file": True},
        )
        with patch("runners._cli_command.build_cli_command_string") as mock_cmd:
            mock_cmd.return_value = "python -m cli.inference.text_generate m --chrome-trace /path/to/trace.json"
            with patch("runners._multicase.compute_case_hash", return_value="hash123"):
                with patch("services.trace_store.legacy_hash_path", return_value=Path("/path/to/trace.json")):
                    result = _run(get_job("job-1", repo, MagicMock()))
        # Verify build_cli_command_string was called with the synthesized path
        call_args = mock_cmd.call_args
        assert call_args[0][0] == "text_generate"
        params_passed = call_args[0][1]
        # Path conversion may change slashes on Windows, so check the string representation
        assert "trace.json" in str(params_passed["chrome-trace-file"])
        assert result.command == "python -m cli.inference.text_generate m --chrome-trace /path/to/trace.json"

    def test_chrome_trace_not_synthesized_when_case_hash_is_none(self):
        """When chrome-trace-file=True but case_hash is None, path stays True (not synthesized)."""
        repo = MagicMock()
        repo.get.return_value = _job(
            module_id="text_generate",
            params={"model_id": "m", "chrome-trace-file": True},
        )
        with patch("runners._cli_command.build_cli_command_string") as mock_cmd:
            mock_cmd.return_value = "cli"
            with patch("runners._multicase.compute_case_hash", return_value=None):
                _run(get_job("job-1", repo, MagicMock()))
        # Verify chrome-trace-file stays True (not synthesized to a path)
        call_args = mock_cmd.call_args
        params_passed = call_args[0][1]
        assert params_passed["chrome-trace-file"] is True


class TestExpandJobCases:
    """Tests for the per-module `_expand_job_cases` helper used to compute
    per-case commands without DB schema changes.
    """

    def test_video_generate_multi_device(self):
        cases = _expand_job_cases("video_generate", {"device": ["A", "B"], "ulysses-size": "4"})
        assert len(cases) == 2
        assert cases[0]["device"] == "A"
        assert cases[1]["device"] == "B"
        # ulysses_size is parsed by _parse_int_list -> int.
        assert cases[0]["ulysses-size"] == 4
        assert cases[1]["ulysses-size"] == 4

    def test_text_generate_cartesian_product(self):
        cases = _expand_job_cases(
            "text_generate",
            {"device": ["A", "B"], "num-queries": "2 3", "quantize-linear-action": "W8A8_DYNAMIC"},
        )
        # 2 devices x 2 num_queries x 1 quantize = 4 cases
        assert len(cases) == 4
        devices = sorted(c["device"] for c in cases)
        assert devices == ["A", "A", "B", "B"]

    def test_throughput_optimizer_multi_device(self):
        cases = _expand_job_cases(
            "throughput_optimizer",
            {"device": ["A", "B"], "tpot-limit": "100", "tp-sizes": "1 2"},
        )
        # throughput_optimizer's _THROUGHPUT_MULTI_FIELDS includes device + tpot/tpot_limits
        # but not tp_sizes (that's parsed by argparser, not by expand_cases). So 2 devices.
        assert len(cases) == 2
        assert cases[0]["device"] == "A"
        assert cases[1]["device"] == "B"

    def test_unknown_module_falls_back_to_single_case(self):
        cases = _expand_job_cases("unknown_module", {"foo": "bar"})
        # Unknown module → no expansion → original params as single case
        assert cases == [{"foo": "bar"}]

    def test_expansion_error_falls_back_to_single_case(self, caplog):
        """If per-module expansion raises, fall back to the original params as
        a single case (defensive — never crash the job detail API).
        """
        import logging

        with patch("runners.text_generate._expand_cases", side_effect=RuntimeError("boom")):
            with caplog.at_level(logging.WARNING):
                cases = _expand_job_cases("text_generate", {"model_id": "m", "device": "A"})
        assert cases == [{"model_id": "m", "device": "A"}]
        assert any("Failed to expand cases" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# get_job_log
# ---------------------------------------------------------------------------


class TestGetJobLog:
    """Tests for the get_job_log endpoint."""

    def test_raises_404_when_job_missing(self):
        repo = MagicMock()
        repo.get.return_value = None
        with pytest.raises(HTTPException) as exc:
            _run(get_job_log("nope", repo, 200))
        assert exc.value.status_code == 404

    def test_returns_log_file_content(self):
        repo = MagicMock()
        repo.get.return_value = _job()
        with patch("api.routers.jobs.read_log_tail", return_value="log line"):
            response = _run(get_job_log("job-1", repo, 200))
        assert isinstance(response, PlainTextResponse)
        assert response.body == b"log line"

    def test_falls_back_to_cached_log_text_when_file_empty(self):
        repo = MagicMock()
        repo.get.return_value = _job(log_text="c1\nc2\nc3\nc4")
        with patch("api.routers.jobs.read_log_tail", return_value=""):
            response = _run(get_job_log("job-1", repo, 2))
        assert response.body.decode() == "c3\nc4"

    def test_cached_log_full_when_tail_zero(self):
        repo = MagicMock()
        repo.get.return_value = _job(log_text="a\nb\nc")
        with patch("api.routers.jobs.read_log_tail", return_value=""):
            response = _run(get_job_log("job-1", repo, 0))
        assert response.body.decode() == "a\nb\nc"

    def test_empty_response_when_no_log_anywhere(self):
        repo = MagicMock()
        repo.get.return_value = _job(log_text=None)
        with patch("api.routers.jobs.read_log_tail", return_value=""):
            response = _run(get_job_log("job-1", repo, 200))
        assert response.body == b""


# ---------------------------------------------------------------------------
# cancel_job
# ---------------------------------------------------------------------------


class TestCancelJob:
    """Tests for the cancel_job endpoint."""

    def test_raises_404_when_job_missing(self):
        repo = MagicMock()
        repo.get.return_value = None
        with pytest.raises(HTTPException) as exc:
            _run(cancel_job("nope", repo, AsyncMock()))
        assert exc.value.status_code == 404

    def test_requests_cancel_and_returns_actual_cancel_state(self):
        """cancel_job surfaces the ACTUAL in-memory cancel state (#89).

        Previously the response hardcoded ``cancel_requested=True`` — even for
        finished jobs — lying to the client. Now we re-read the job (so a
        terminal transition that raced with the POST is visible) and surface
        the actual ``is_cancel_requested`` value. A succeeded job whose flag
        has been cleaned up returns ``cancel_requested=False, result_ready=True``.
        """
        repo = MagicMock()
        # First GET returns RUNNING (in-flight); second GET (after cancel) returns SUCCEEDED.
        repo.get.side_effect = [
            _job(JobStatus.RUNNING),
            _job(JobStatus.SUCCEEDED),
        ]
        job_manager = MagicMock()
        job_manager.request_cancel.return_value = True
        # The worker already cleaned up the flag (job reached terminal state
        # between the cancel POST and the response build).
        job_manager.is_cancel_requested.return_value = False
        with patch("runners._cli_command.build_cli_command_string", return_value="c"):
            result = _run(cancel_job("job-1", repo, job_manager))
        job_manager.request_cancel.assert_called_once_with("job-1")
        job_manager.is_cancel_requested.assert_called_once_with("job-1")
        # Accurate state: flag was cleaned up, result IS ready.
        assert result.cancel_requested is False
        assert result.result_ready is True

    def test_cancel_requested_true_while_flag_active(self):
        """When the in-memory flag is active, response surfaces cancel_requested=True."""
        repo = MagicMock()
        repo.get.return_value = _job(JobStatus.RUNNING)
        job_manager = MagicMock()
        job_manager.request_cancel.return_value = True
        job_manager.is_cancel_requested.return_value = True  # flag still active
        with patch("runners._cli_command.build_cli_command_string", return_value="c"):
            result = _run(cancel_job("job-1", repo, job_manager))
        assert result.cancel_requested is True
        assert result.result_ready is False  # RUNNING → not ready

    def test_logs_info_when_no_active_cancel_flag(self):
        """Requesting cancel on a finished/unknown job logs info and returns
        accurate cancel_requested=False (#89) — no more lying to the client.
        """
        repo = MagicMock()
        repo.get.return_value = _job(JobStatus.SUCCEEDED)
        job_manager = MagicMock()
        job_manager.request_cancel.return_value = False  # no in-memory flag
        job_manager.is_cancel_requested.return_value = False
        with (
            patch("runners._cli_command.build_cli_command_string", return_value="c"),
            patch("api.routers.jobs.logger") as mock_logger,
        ):
            result = _run(cancel_job("job-1", repo, job_manager))
        mock_logger.info.assert_called_once()
        # Accurate state: cancel didn't apply (job finished), result IS ready.
        assert result.cancel_requested is False
        assert result.result_ready is True


# ---------------------------------------------------------------------------
# get_job_result
# ---------------------------------------------------------------------------


class TestGetJobResult:
    """Tests for the get_job_result endpoint."""

    def test_raises_404_when_job_missing(self):
        repo = MagicMock()
        repo.get.return_value = None
        with pytest.raises(HTTPException) as exc:
            _run(get_job_result("nope", repo, MagicMock()))
        assert exc.value.status_code == 404

    def test_assembles_result_with_schema(self):
        repo = MagicMock()
        repo.get.return_value = _job()
        result_repo = MagicMock()
        result_repo.list_for_job.return_value = [_record(0), _record(1)]
        with (
            patch("api.routers.jobs.SchemaRegistry") as mock_reg,
            patch("api.routers.jobs.assemble_result_envelope", return_value={"envelope": True}),
        ):
            mock_reg.return_value.get_form_schema.return_value = {"fields": []}
            result = _run(get_job_result("job-1", repo, result_repo))
        assert isinstance(result, JobResultResponse)
        assert len(result.records) == 2
        assert result.records[0]["case_hash"] == "hash0"
        assert result.form_schema == {"fields": []}
        assert result.result == {"envelope": True}

    def test_falls_back_to_empty_schema_when_missing(self):
        repo = MagicMock()
        repo.get.return_value = _job()
        result_repo = MagicMock()
        result_repo.list_for_job.return_value = []
        with (
            patch("api.routers.jobs.SchemaRegistry") as mock_reg,
            patch("api.routers.jobs.assemble_result_envelope", return_value={}),
            patch("api.routers.jobs.logger") as mock_logger,
        ):
            mock_reg.return_value.get_form_schema.return_value = None
            result = _run(get_job_result("job-1", repo, result_repo))
        assert result.form_schema == {}
        mock_logger.warning.assert_called_once()
