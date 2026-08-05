"""Jobs router.

POST /api/jobs — submit a job
GET /api/jobs/{id} — poll job status/progress
GET /api/jobs/{id}/log — fetch captured logs (text/plain)
POST /api/jobs/{id}/cancel — cooperative cancel
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, FileResponse
from pydantic import BaseModel, Field

from api.schemas import JobResponse, JobStatusResponse, JobListItem
from models.enums import JobStatus
from services.repositories import JobRepository, ResultRepository
from services.capture import read_log_tail
from services.schema_registry import SchemaRegistry
from services.result_view import assemble_result_envelope

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/jobs", tags=["jobs"])

#: Server-suggested polling interval (ms) for non-terminal jobs.
#: Clients drive polling off this hint and fall back to a local default if absent.
DEFAULT_POLL_INTERVAL_MS = 1500


def get_job_manager(request: Request):
    """Resolve the app-wide JobManager wired on startup (main.py lifespan)."""
    return request.app.state.job_manager


def _expand_job_cases_strict(module_id: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a job's params into per-case dicts, RAISING on bad/oversized input.

    Mirrors the worker's expansion. Raises ``ValueError`` when a multi-value
    field is unparseable or the cartesian product exceeds the case cap
    (``runners._multicase.MAX_CASES``). Used for submit-time preflight so the
    error surfaces as a 4xx instead of being swallowed.
    """
    if module_id == "text_generate":
        from runners.text_generate import _expand_cases

        return _expand_cases(params)
    if module_id == "video_generate":
        from runners._multicase import expand_cases
        from runners.video_generate import _VIDEO_MULTI_FIELDS

        return expand_cases(params, _VIDEO_MULTI_FIELDS)
    if module_id == "throughput_optimizer":
        from runners._multicase import expand_cases
        from runners.throughput_optimizer import _THROUGHPUT_MULTI_FIELDS

        return expand_cases(params, _THROUGHPUT_MULTI_FIELDS)
    return [params]


def _expand_job_cases(module_id: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a job's params into per-case param dicts (mirrors the worker's expansion).

    Each runner has its own multi-case field definitions; dispatch here so the
    API can compute the actual per-case CLI commands on demand (without storing
    them in the DB). Single-value submits collapse to one case (matching the
    worker's backward-compatible single-case path). Expansion failures fall back
    to ``[params]`` so status polling for an already-submitted job never breaks.
    """
    try:
        return _expand_job_cases_strict(module_id, params)
    except Exception as exc:
        logger.warning("Failed to expand cases for module %s: %s", module_id, exc)
    return [params]


def _to_status_response(job, **overrides):
    """Build JobStatusResponse from a Job entity, computing command/params.

    ``overrides`` are keyword arguments that replace the default values
    (e.g. ``result_ready``, ``cancel_requested``). Used by both ``get_job``
    and ``cancel_job`` to avoid duplication. Overrides take precedence over the
    computed defaults (cancel_job forces ``result_ready=False``).
    """
    # Lazy import: the command reconstruction is pure but lives in the runners
    # package; importing it here keeps the router import-light.
    from runners._cli_command import build_cli_command_string

    # Synthesize chrome_trace path if enabled (so the command shows the actual path, not <auto>)
    params_for_command = dict(job.params) if job.params else None
    if params_for_command and params_for_command.get("chrome_trace") is True:
        from runners._multicase import compute_case_hash
        from services.trace_store import legacy_hash_path

        case_hash = compute_case_hash(job.module_id, job.form_schema_version, params_for_command)
        if case_hash and job.id:
            params_for_command["chrome_trace"] = str(legacy_hash_path(job.id, case_hash))

    command = build_cli_command_string(job.module_id, params_for_command) if params_for_command else None
    # Per-case commands (mirrors what the worker actually executes after multi-
    # case expansion). Single-case jobs collapse to [command].
    commands = None
    if params_for_command:
        cases = _expand_job_cases(job.module_id, params_for_command)
        commands = [build_cli_command_string(job.module_id, case) for case in cases]

    fields = dict(
        job_id=job.id,
        module_id=job.module_id,
        status=job.status.value,
        progress=job.progress,
        progress_text=job.progress_text or "",
        label=job.label,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
        error_detail=job.error_detail,
        result_ready=job.status == JobStatus.SUCCEEDED,
        poll_interval_ms=DEFAULT_POLL_INTERVAL_MS,
        command=command,
        commands=commands,
        params=job.params,
    )
    fields.update(overrides)
    return JobStatusResponse(**fields)


# --- Pydantic schemas (request/response) --------------------------------


class JobSubmitRequest(BaseModel):
    """Request schema for POST /api/jobs."""

    module_id: str = Field(..., description="Module identifier (text_generate, video_generate, throughput_optimizer)")
    form_schema_version: str = Field(..., description="Form schema version (semver)")
    params: dict = Field(..., description="Form field values (validated client-side; runner validates internally)")


class JobListQuery(BaseModel):
    """Query parameters for GET /api/jobs."""

    module_id: str | None = None
    status: JobStatus | None = None
    limit: int = 50
    offset: int = 0


class JobResultResponse(BaseModel):
    """Response schema for GET /api/jobs/{id}/result."""

    job_id: str
    module_id: str
    form_schema: dict[str, Any]
    records: list[dict[str, Any]]
    result: dict[str, Any]


# --- GET /api/jobs ----------------------------------------------------------------


class JobListResponse(BaseModel):
    """Paginated job list: the page items + the total count (for pagination UI)."""

    items: list[JobListItem]
    total: int


@router.get("", response_model=JobListResponse)
async def list_jobs(
    job_repo: Annotated[JobRepository, Depends()],
    module_id: str | None = None,
    status: JobStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> JobListResponse:
    """List jobs with optional filtering and pagination.

    Returns the page items plus the total count (matching the filter, ignoring
    limit/offset) so the UI can render pagination controls.
    """
    jobs = job_repo.list_jobs(
        module_id=module_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    total = job_repo.count_jobs(module_id=module_id, status=status)
    return JobListResponse(
        items=[
            JobListItem(
                job_id=job.id,
                module_id=job.module_id,
                label=job.label,
                status=job.status.value,
                progress=job.progress,
                created_at=job.created_at,
                completed_at=job.completed_at,
            )
            for job in jobs
        ],
        total=total,
    )


# --- POST /api/jobs ---------------------------------------------------------------


@router.post("", response_model=JobResponse, status_code=201)
async def create_job(
    request: JobSubmitRequest,
    job_repo: Annotated[JobRepository, Depends()],
    job_manager: Annotated[Any, Depends(get_job_manager)],
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Submit a new modeling job.

    The job is persisted as ``pending`` and scheduled on the single-worker pool.
    Returns 201 on acceptance (job may be queued if another is running).
    """
    from models.entities import Job
    from models.enums import JobStatus

    # Reject unknown form-schema versions early: a missing snapshot would later
    # leave the job's result page unable to re-render the form.
    schema_registry = SchemaRegistry()
    if schema_registry.get_form_schema(request.module_id, request.form_schema_version) is None:
        raise HTTPException(
            status_code=400,
            detail=(f"Form schema version {request.form_schema_version} not found for module {request.module_id}"),
        )

    # Preflight the multi-case expansion: reject oversized fan-out (e.g. many
    # comma-list values multiplying past the case cap) and unparseable multi-value
    # fields BEFORE scheduling — otherwise the worker (or the state-polling expand
    # via ``_expand_job_cases``) would swallow the error into a single case.
    try:
        _expand_job_cases_strict(request.module_id, request.params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Create the Job entity (id auto-generated by the Job dataclass;
    # created_at is stamped by the repository on insert).
    job = Job(
        module_id=request.module_id,
        form_schema_version=request.form_schema_version,
        params=request.params,
        status=JobStatus.PENDING,
        progress=0,
        progress_text="",
        label=f"{request.module_id} task",
    )

    # Submit to JobManager (which persists and schedules)
    try:
        submitted = await job_manager.submit_async(job)
    except job_manager.InflightLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    return JobResponse(
        job_id=submitted.id,
        status=submitted.status.value,
    )


# --- GET /api/jobs/{id}/trace/{seq} (Trace download) -----------------------
# NOTE: This route must be defined BEFORE /{job_id} to avoid matching "trace" as a job_id


@router.get("/{job_id}/trace/{seq}", response_class=FileResponse)
async def get_job_trace(
    job_id: str,
    seq: int,
    job_repo: Annotated[JobRepository, Depends()],
) -> FileResponse:
    """Download a specific case's Chrome trace JSON file.

    Returns the Chrome trace file for case {seq} of job {job_id}.
    Raises 404 if the job/seq doesn't exist or the trace file is missing.
    """
    job = job_repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    from services.trace_store import trace_path

    trace_file = trace_path(job_id, seq)
    if not trace_file.exists():
        raise HTTPException(status_code=404, detail=f"Trace file for job {job_id} case {seq} not found")

    return FileResponse(
        trace_file,
        media_type="application/json",
        filename=f"chrome_trace_case_{seq}.json",
        headers={"Content-Disposition": f'attachment; filename="chrome_trace_case_{seq}.json"'},
    )


# --- GET /api/jobs/{id} ----------------------------------------------------------


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: str,
    job_repo: Annotated[JobRepository, Depends()],
    job_manager: Annotated[Any, Depends(get_job_manager)],
) -> JobStatusResponse:
    """Poll job status/progress.

    Returns the job record with status, progress, progress_text, and error details.
    Raises 404 if unknown job_id.

    ``cancel_requested`` is surfaced from the JobManager's in-memory flag
    (#89) — the flag is held in memory only (not persisted to DB) because it's
    scoped to the job's worker lifetime; the poll response would otherwise
    always read False from the DB.
    """
    job = job_repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return _to_status_response(
        job,
        cancel_requested=job_manager.is_cancel_requested(job_id),
    )


# --- GET /api/jobs/{id}/log --------------------------------------------------


@router.get("/{job_id}/log", response_class=PlainTextResponse)
async def get_job_log(
    job_id: str,
    job_repo: Annotated[JobRepository, Depends()],
    tail: Annotated[int, Query] = 200,
) -> PlainTextResponse:
    """Fetch captured runner stdout+logging as text/plain.

    Returns the last ``tail`` lines (default 200) of the job's captured log.
    Raises 404 if unknown job_id.
    """
    job = job_repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    log_content = read_log_tail(job_id, tail)
    if not log_content and job.log_text:
        # Cache-hit jobs (Phase C) have no log file — no worker ran — so serve
        # the cached CLI log from the DB, applying the same tail.
        lines = job.log_text.splitlines()
        log_content = "\n".join(lines[-tail:] if tail and tail > 0 else lines)
    return PlainTextResponse(content=log_content)


# --- POST /api/jobs/{id}/cancel ------------------------------------------------


@router.post("/{job_id}/cancel", response_model=JobStatusResponse)
async def cancel_job(
    job_id: str,
    job_repo: Annotated[JobRepository, Depends()],
    job_manager: Annotated[Any, Depends(get_job_manager)],
) -> JobStatusResponse:
    """Cooperatively request cancellation (research Area 3).

    Sets ``cancel_requested`` (cooperative flag). The worker polls the flag
    and, if true, discards its result and transitions to ``cancelled`` (if still
    running). If the job already finished, this is a no-op (the final status
    remains unchanged).

    Returns the CURRENT status response with the actual ``cancel_requested``
    state (#89). The previous implementation hardcoded ``cancel_requested=True``
    even when the request was a no-op (unknown/finished job), which lied to the
    client. We re-read the job AFTER the cancel request so a terminal transition
    that raced with the POST is visible to the caller.
    """
    job = job_repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    was_requested = job_manager.request_cancel(job_id)
    if not was_requested:
        # No in-memory flag — the job is unknown to the manager (finished,
        # cancelled, or the server restarted). Surface the truth rather than
        # pretending the cancel took effect.
        logger.info(
            "Cancel requested for job %s with no active in-memory flag (likely already terminal or server-restarted)",
            job_id,
        )

    # Re-read the job so the response reflects any terminal transition that
    # raced with the cancel request (e.g. the worker just finished between
    # the first GET and this POST).
    job = job_repo.get(job_id)
    return _to_status_response(
        job,
        cancel_requested=job_manager.is_cancel_requested(job_id),
    )


# --- GET /api/jobs/{id}/result ------------------------------------------------


@router.get("/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(
    job_id: str,
    job_repo: Annotated[JobRepository, Depends()],
    result_repo: Annotated[ResultRepository, Depends()],
) -> JobResultResponse:
    """Fetch a job's assembled result envelope.

    Returns:
    * ``result``: the assembled envelope built from the job's ``result_records[]``
      (module-specific structure per ``contracts/result-rendering.md``)
    * ``records[]``: raw persisted result records (with ``seq``, ``rank``, ``config``,
      ``summary``, ``tables``) for advanced inspection
    * ``form_schema``: the pinned form-schema snapshot used to submit this job
      (deterministic re-render)

    Raises 404 if unknown job_id.
    """
    # 1. Fetch the job
    job = job_repo.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # 2. Fetch result records
    raw_records = result_repo.list_for_job(job_id)
    records = [
        {
            "seq": r.seq,
            "rank": r.rank,
            "config": r.config,
            "summary": r.summary,
            "tables": r.tables,
            "case_hash": r.case_hash,
        }
        for r in raw_records
    ]

    # 3. Fetch the pinned form schema snapshot
    schema_registry = SchemaRegistry()
    form_schema = schema_registry.get_form_schema(job.module_id, job.form_schema_version)
    if form_schema is None:
        # Fallback: return empty schema if snapshot missing (shouldn't happen)
        logger.warning(f"Form schema snapshot missing for {job.module_id}:{job.form_schema_version}")
        form_schema = {}

    # 4. Assemble the result envelope
    result = assemble_result_envelope(
        module_id=job.module_id,
        records=records,
        form_schema_version=job.form_schema_version,
        input_config=job.params,
        job_id=job.id,
    )

    return JobResultResponse(
        job_id=job.id,
        module_id=job.module_id,
        form_schema=form_schema,
        records=records,
        result=result,
    )
