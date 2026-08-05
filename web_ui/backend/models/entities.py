"""Domain entities: ``CapabilityModule``, ``Job``, ``ResultRecord``.

Pure domain (Constitution Principle I). No FastAPI / SQLModel / Pydantic /
torch. These are plain dataclasses representing the core business objects.
Persistence shape lives in ``infrastructure/persistence/orm``;
these entities are the language of the application/domain layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from models.enums import JobStatus, assert_transition


def _new_id() -> str:
    return uuid4().hex


@dataclass
class CapabilityModule:
    """A capability module row (exactly 3 seeded)."""

    id: str  # text_generate | video_generate | throughput_optimizer
    display_name: str
    runner_class: str  # ModelRunner | VideoGenerateRunner | ParallelRunner
    description: str | None = None


@dataclass
class Job:
    """One submission. Status drives the 6-state machine."""

    module_id: str
    params: dict[str, Any]
    form_schema_version: str
    status: JobStatus = JobStatus.PENDING
    id: str = field(default_factory=_new_id)
    label: str | None = None
    progress: int | None = None
    progress_text: str | None = None
    error: str | None = None
    error_detail: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    cancel_requested: bool = False
    # Phase C: params_hash dedups runs (cache key); log_text caches the CLI log
    # so a cache hit reuses both the result and the log without re-running.
    params_hash: str | None = None
    log_text: str | None = None

    def transition(self, to_status: JobStatus) -> None:
        """Validate + apply a state-machine edge."""
        assert_transition(self.status, to_status)
        self.status = to_status

    @property
    def is_terminal(self) -> bool:
        """True once the job is in a terminal state (no further transitions)."""
        return self.status.is_terminal()

    @property
    def result_ready(self) -> bool:
        """True only for SUCCEEDED jobs (a result exists to read)."""
        return self.status == JobStatus.SUCCEEDED


@dataclass
class ResultRecord:
    """A normalized result row. 1 per text/video; N per optimizer job.

    ``rank`` is computed at CAPTURE time for the optimizer
    (max ``throughput_token_s``; ties -> lower ttft then tpot then device);
    ``1`` is the overall best across devices. ``None`` for text/video.
    """

    job_id: str
    seq: int  # 0-based stable sort key within the job
    config: dict[str, Any]
    summary: dict[str, Any]
    tables: dict[str, Any] = field(default_factory=dict)
    rank: int | None = None
    case_hash: str | None = None  # case-level dedup key (compute_params_hash of the case's concrete params)
    id: str = field(default_factory=_new_id)
    created_at: str | None = None
    # Worker-captured CLI output for THIS case only (banner/excluded). Carried via
    # result.json to the main process, which persists it to the case_logs table +
    # {case_hash}.log file. NOT persisted to result_records (metadata-only).
    case_log: str | None = None
