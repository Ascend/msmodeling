"""Pydantic v2 request/response DTOs.

Pure inbound-adapter shapes (``contracts/rest-api.md``). No business rules.
``params`` is deliberately ``dict[str, Any]`` and is NOT field-validated here —
the form validates client-side and the Runner validates internally
(invalid -> job ``failed``); the backend forwards params unchecked.

Note: DTOs used by only a single router (e.g. submit/result) have been inlined
into ``api/routers/jobs.py`` (JobSubmitRequest / JobListQuery / JobResultResponse
/ JobListResponse) to avoid maintaining them in two places. This file keeps only
the DTOs shared across routers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --- Option sources ---------------------------------------------------------


class OptionItem(BaseModel):
    value: Any
    label: str | None = None


# --- Modules ----------------------------------------------------------------


class ModuleOut(BaseModel):
    id: str
    display_name: str
    runner_class: str
    description: str | None = None


# --- Jobs -------------------------------------------------------------------


class JobResponse(BaseModel):
    """Response for POST /api/jobs (job acceptance)."""

    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    """Response for GET /api/jobs/{id} and POST /api/jobs/{id}/cancel."""

    job_id: str
    module_id: str
    status: str
    progress: int | None = None
    progress_text: str | None = None
    label: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    error_detail: str | None = None
    result_ready: bool = False
    cancel_requested: bool = False
    poll_interval_ms: int | None = Field(
        default=None,
        description="Server-suggested polling interval (ms) for non-terminal jobs; "
        "clients fall back to a local default when absent.",
    )
    command: str | None = Field(
        default=None,
        description="Reconstructed CLI command string for the job's ORIGINAL params "
        "(reference only — see `commands` for the actual per-case commands when the "
        "job was expanded into multiple cases).",
    )
    commands: list[str] | None = Field(
        default=None,
        description="Per-case CLI commands actually executed by the worker. For "
        "single-case jobs, contains one element equal to `command`. For multi-case "
        "jobs (multi-device / multi-quantize / etc.), contains one command per case "
        "in the order they ran.",
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="Submitted job parameters (as provided in the original request).",
    )


class JobListItem(BaseModel):
    job_id: str
    module_id: str
    label: str | None = None
    status: str
    progress: int | None = None
    created_at: str | None = None
    completed_at: str | None = None
