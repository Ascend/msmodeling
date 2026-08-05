"""Domain enums + value objects: ``JobStatus`` + ``FieldSchema`` (form fields).

Flattened from the former DDD ``domain/value_objects`` package. Pure domain
(Constitution Principle I): NO FastAPI / SQLModel / Pydantic / torch imports.
``JobStatus`` is authoritative and matches the ``jobs.status`` CHECK constraint
and the REST status list:
pending, running, succeeded, failed, cancelled, interrupted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class JobStatus(str, Enum):
    """The 6-state job lifecycle.

    Terminal states: ``succeeded``, ``failed``, ``cancelled``, ``interrupted``.
    Re-entry to a terminal state happens only via a NEW ``job_id`` (a re-run).
    ``interrupted`` is written ONLY by the FastAPI lifespan startup sweep
    (a job left ``pending``/``running`` from a crashed server).
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

    @classmethod
    def terminal(cls) -> frozenset[JobStatus]:
        """The terminal states (no outbound transitions; re-entry = new job_id)."""
        return frozenset({cls.SUCCEEDED, cls.FAILED, cls.CANCELLED, cls.INTERRUPTED})

    @classmethod
    def active(cls) -> frozenset[JobStatus]:
        """States swept to ``interrupted`` on startup (server died mid-run)."""
        return frozenset({cls.PENDING, cls.RUNNING})

    def is_terminal(self) -> bool:
        """Instance convenience for ``self in JobStatus.terminal()``."""
        return self in self.terminal()


class IllegalJobTransitionError(ValueError):
    """Raised when a job transitions through a disallowed edge."""

    def __init__(self, from_status: JobStatus, to_status: JobStatus):
        super().__init__(f"Illegal job status transition: {from_status.value} -> {to_status.value}")
        self.from_status = from_status
        self.to_status = to_status


# Allowed transitions for the worker/submit/lifespan drivers. The lifespan
# sweep (pending/running -> interrupted) is intentionally included here.
_ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.PENDING: frozenset({JobStatus.RUNNING, JobStatus.CANCELLED, JobStatus.INTERRUPTED}),
    JobStatus.RUNNING: frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.INTERRUPTED}),
    # terminal states have no outbound edges (re-entry = new job_id)
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
    JobStatus.INTERRUPTED: frozenset(),
}


def assert_transition(from_status: JobStatus, to_status: JobStatus) -> None:
    """Validate a single state-machine edge; raise on illegal moves."""
    if to_status not in _ALLOWED_TRANSITIONS.get(from_status, frozenset()):
        raise IllegalJobTransitionError(from_status, to_status)


def can_transition(from_status: JobStatus, to_status: JobStatus) -> bool:
    """Non-raising predicate companion to :func:`assert_transition`."""
    return to_status in _ALLOWED_TRANSITIONS.get(from_status, frozenset())


# === Value objects (FieldSchema) ======================================================


@dataclass(frozen=True)
class FieldOption:
    """A single option in an inline option list (value/label pair)."""

    value: Any
    label: str | None = None


@dataclass(frozen=True)
class OptionSource:
    """``inline`` (frozen enum) or ``dynamic`` (named backend source)."""

    type: str  # "inline" | "dynamic"
    name: str | None = None  # only when type == "dynamic"
    values: tuple[FieldOption, ...] = field(default_factory=tuple)

    def is_dynamic(self) -> bool:
        """True when the options come from a named backend source (not inline)."""
        return self.type == "dynamic"


@dataclass(frozen=True)
class LocalizedText:
    """A bilingual display string: ``{zh, en}``, ``en`` optional.

    Accepts a plain string (locale-neutral shorthand, treated as ``zh``) or a
    ``{ "zh": ..., "en": ... }`` mapping. The backend stores/hashes it only — it
    never renders; the frontend resolves it by the active locale (``useLocale``).
    """

    zh: str | None = None
    en: str | None = None

    @classmethod
    def from_value(cls, value: str | Mapping[str, str] | LocalizedText | None) -> LocalizedText | None:
        """Coerce a raw schema value (``str`` | ``{zh,en}`` | ``None``)."""
        if value is None or isinstance(value, LocalizedText):
            return value  # type: ignore[return-value]
        if isinstance(value, str):
            return cls(zh=value)
        return cls(zh=value.get("zh"), en=value.get("en"))

    def get(self, locale: str = "zh") -> str | None:
        """Resolve for ``locale``; fall back to ``zh`` then any present value."""
        if locale == "en" and self.en is not None:
            return self.en
        return self.zh or self.en


@dataclass(frozen=True)
class ValidationRule:
    """One async-validator-shaped rule from the form schema.

    ``rule == "validator"`` names a frontend-registry function whose input is the
    owning field value **plus the whole form model** (cross-field validation).
    ``depends_on`` lists sibling field ids whose change re-triggers this
    rule (reactive cross-field re-validation). ``message`` is localized.
    """

    rule: str  # required | pattern | min | max | len | enum | validator
    message: str | LocalizedText | None = None
    value: Any = None  # validator name (str) | pattern | bound | enum list
    type: str | None = None
    trigger: tuple[str, ...] = field(default_factory=tuple)
    depends_on: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Conditions:
    """visible/enabled/required predicate trees (raw, opaque here)."""

    visible: dict[str, Any] | None = None
    enabled: dict[str, Any] | None = None
    required: dict[str, Any] | None = None


@dataclass(frozen=True)
class FieldSchema:
    """A single form field."""

    id: str
    label: str | LocalizedText  # zh primary; {zh,en} when bilingual
    control: str = "text"  # text|number|select|multi-select|switch|slider
    data_type: str = "string"  # string|integer|number|boolean|string[]|integer[]
    default: Any = None
    group: str | None = None
    tooltip: str | LocalizedText | None = None
    placeholder: str | LocalizedText | None = None
    option_source: OptionSource | None = None
    validation: tuple[ValidationRule, ...] = field(default_factory=tuple)
    conditions: Conditions | None = None

    @property
    def is_required(self) -> bool:
        """True if any validation rule marks the field ``required``."""
        return any(r.rule == "required" for r in self.validation)
