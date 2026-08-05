"""SQLModel ORM tables — persistence shape.

NOTE: SQLModel is imported at module
top-level here (it IS the persistence framework); this module is only imported
by ``db.init_db`` / the repositories (lazy via ``db.get_engine``), so the app
boots without it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# Valid statuses mirror the domain JobStatus enum + the CHECK constraint.
_STATUS_VALUES = (
    "pending",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
)


class ModuleRow(SQLModel, table=True):
    """``modules`` — seeded capability registry (exactly 3 rows)."""

    __tablename__ = "modules"

    id: str = Field(primary_key=True)
    display_name: str
    runner_class: str
    description: str | None = None
    created_at: str = Field(default_factory=_utcnow_iso)


class FormSchemaRow(SQLModel, table=True):
    """``form_schemas`` — snapshot registry, PK (module_id, version)."""

    __tablename__ = "form_schemas"

    module_id: str = Field(foreign_key="modules.id", primary_key=True)
    version: str = Field(primary_key=True)
    schema_hash: str
    fields: str  # JSON snapshot of the FULL form-schema envelope (fields[], formValidation[], optionSourceRegistry, title) — hashed + pinned for faithful reopen
    created_at: str = Field(default_factory=_utcnow_iso)


class JobRow(SQLModel, table=True):
    """``jobs`` — one submission. status CHECK + FK to the form-schema snapshot."""

    __tablename__ = "jobs"

    id: str = Field(primary_key=True)
    module_id: str = Field(foreign_key="modules.id", index=True)
    status: str = Field(index=True)  # CHECK enforced at app layer (see _STATUS_VALUES)
    progress: int | None = Field(default=None)
    progress_text: str | None = None
    params: str  # JSON
    form_schema_version: str
    label: str | None = None
    error: str | None = None
    error_detail: str | None = None
    created_at: str = Field(default_factory=_utcnow_iso, index=True)
    started_at: str | None = None
    completed_at: str | None = None
    # Phase C result+log caching: stable hash of (module_id, params) for dedup,
    # and the full CLI log text (so a cache hit can reuse the log too).
    params_hash: str | None = Field(default=None, index=True)
    log_text: str | None = None

    def to_params(self) -> dict:
        return json.loads(self.params) if self.params else {}


class ResultRecordRow(SQLModel, table=True):
    """``result_records`` — 1..N per job (text/video=1; optimizer=N per device).

    ``rank`` is capture-time (domain/services/ranking.py); ``tables`` holds ALL
    structured payloads as JSON TEXT (no sidecar files).
    """

    __tablename__ = "result_records"

    id: str = Field(primary_key=True)
    job_id: str = Field(foreign_key="jobs.id", index=True)
    seq: int
    rank: int | None = Field(default=None, index=True)
    config: str  # JSON
    summary: str  # JSON (flat scalar metrics)
    tables: str = Field(default="{}")  # JSON
    case_hash: str | None = Field(default=None, index=True)  # case-level dedup key (hash of the case's concrete params)
    created_at: str = Field(default_factory=_utcnow_iso)


class TelemetryEventRow(SQLModel, table=True):
    """``telemetry_events`` — UI interaction events (field changes + button clicks).

    Aggregated by (module_id, target) to surface high-frequency form fields and
    controls, informing form-layout optimization (e.g. tier-based visibility).
    Local single-user console: no auth, and NO field values are stored — only
    the fact that an interaction happened.
    """

    __tablename__ = "telemetry_events"

    id: int | None = Field(default=None, primary_key=True)
    module_id: str = Field(index=True)  # module id, or "global" for cross-module actions
    target: str = Field(index=True)  # field_id, or button name (tab:*, run, view_result, group:*)
    event_type: str = Field(default="change")  # change / click / submit / toggle
    fingerprint: str | None = Field(
        default=None, index=True
    )  # visitor browser fingerprint (for DISTINCT-user counting)
    created_at: str = Field(default_factory=_utcnow_iso, index=True)


class FeedbackRow(SQLModel, table=True):
    """``feedbacks`` — user text/screenshot feedback (physically isolated from
    telemetry; stores content values).

    For kind=text only content_text is set; for kind=screenshot, screenshot_path
    holds a JSON array (a list of paths relative to ``.msmodeling_ui``).
    job_id/module_id are optional and locate the feedback's origin.
    """

    __tablename__ = "feedbacks"

    id: str = Field(primary_key=True)  # uuid4 hex
    job_id: str | None = Field(default=None, foreign_key="jobs.id", index=True)
    module_id: str | None = Field(default=None, index=True)  # 'global' or a concrete module id
    kind: str  # 'text' | 'screenshot'
    content_text: str | None = None
    screenshot_path: str | None = None  # JSON array of rel-paths
    fingerprint: str | None = Field(default=None, index=True)
    created_at: str = Field(default_factory=_utcnow_iso, index=True)


class CaseLogRow(SQLModel, table=True):
    """``case_logs`` — per-case CLI output, keyed by case_hash.

    Replaces the fragile regex-splitting of ``{job_id}.log`` by ``===== Case i/N
    =====`` headers (which broke when separators appeared in case bodies). Each
    case's output is captured at run time and stored here + mirrored to
    ``.msmodeling_ui/logs/cases/{case_hash}.log``. Case-level dedup reuses a
    prior run's log by case_hash lookup (no re-extraction needed).
    """

    __tablename__ = "case_logs"

    case_hash: str = Field(primary_key=True, max_length=64)
    content: str = Field(default="")  # the case's CLI output (print + logging)
    created_at: str = Field(default_factory=_utcnow_iso)
