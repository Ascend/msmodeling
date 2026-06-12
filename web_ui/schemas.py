from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExperimentTask:
    sim_type: str
    params: dict[str, Any]
    command: list[str]
    task_hash: str
    label: str


@dataclass
class ExperimentResult:
    sim_type: str
    status: str
    params: dict[str, Any]
    command: list[str]
    task_hash: str
    label: str
    summary: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)
    raw_log: str = ""
    error: str | None = None
    source: str = "run"  # run | cache

    def to_row(self) -> dict[str, Any]:
        row = {
            "sim_type": self.sim_type,
            "status": self.status,
            "label": self.label,
            "task_hash": self.task_hash,
            **self.params,
            **self.summary,
            "warning_count": len(self.warnings),
            "info_count": len(self.infos),
            "source": self.source,
            "error": self.error,
        }
        return row

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
