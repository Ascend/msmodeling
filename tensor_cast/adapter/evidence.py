import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclasses.dataclass(frozen=True)
class ExpectedValue:
    time_s: float
    rel_tolerance: float = 0.2
    abs_tolerance_s: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_rel_tolerance: float) -> "ExpectedValue":
        if "time_s" not in data:
            raise ValueError("Expected time value must include 'time_s'.")
        return cls(
            time_s=float(data["time_s"]),
            rel_tolerance=float(data.get("rel_tolerance", default_rel_tolerance)),
            abs_tolerance_s=(None if data.get("abs_tolerance_s") is None else float(data["abs_tolerance_s"])),
        )

    def matches(self, actual_s: float) -> bool:
        tolerance = abs(self.time_s) * self.rel_tolerance
        if self.abs_tolerance_s is not None:
            tolerance = max(tolerance, self.abs_tolerance_s)
        return abs(actual_s - self.time_s) <= tolerance


@dataclasses.dataclass(frozen=True)
class ExpectedOp:
    name: str
    count: Optional[int] = None
    count_min: Optional[int] = None
    count_max: Optional[int] = None
    total_time: Optional[ExpectedValue] = None
    rel_tolerance: float = 0.3
    confidence: str = "high"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpectedOp":
        if "name" not in data:
            raise ValueError("Each major op expectation must include 'name'.")
        count = data.get("count")
        count_min = data.get("count_min")
        count_max = data.get("count_max")
        if isinstance(count, dict):
            count_min = count.get("min")
            count_max = count.get("max")
            count = None
        rel_tolerance = float(data.get("rel_tolerance", 0.3))
        total_time = None
        if "total_time_s" in data:
            total_time = ExpectedValue(
                time_s=float(data["total_time_s"]),
                rel_tolerance=rel_tolerance,
                abs_tolerance_s=(None if data.get("abs_tolerance_s") is None else float(data["abs_tolerance_s"])),
            )
        return cls(
            name=str(data["name"]),
            count=None if count is None else int(count),
            count_min=None if count_min is None else int(count_min),
            count_max=None if count_max is None else int(count_max),
            total_time=total_time,
            rel_tolerance=rel_tolerance,
            confidence=str(data.get("confidence", "high")),
        )

    def count_matches(self, actual_count: int) -> bool:
        if self.count is not None:
            return actual_count == self.count
        if self.count_min is not None and actual_count < self.count_min:
            return False
        if self.count_max is not None and actual_count > self.count_max:
            return False
        return True


@dataclasses.dataclass(frozen=True)
class EvidenceCase:
    name: str
    input: Dict[str, Any]
    total_forward: Optional[ExpectedValue]
    major_ops: List[ExpectedOp]
    notes: List[str] = dataclasses.field(default_factory=list)
    accepted_gaps: List[str] = dataclasses.field(default_factory=list)
    observed_kernels: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    shape_hints: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceCase":
        expected = data.get("expected", {})
        total_forward = None
        if expected.get("total_forward") is not None:
            total_forward = ExpectedValue.from_dict(expected["total_forward"], 0.2)
        return cls(
            name=str(data.get("name", "default")),
            input=dict(data.get("input", {})),
            total_forward=total_forward,
            major_ops=[ExpectedOp.from_dict(op) for op in expected.get("major_ops", [])],
            notes=list(data.get("notes", [])),
            accepted_gaps=list(data.get("accepted_gaps", [])),
            observed_kernels=list(data.get("observed_kernels", [])),
            shape_hints=dict(data.get("shape_hints", {})),
        )


@dataclasses.dataclass(frozen=True)
class EvidenceDocument:
    version: int
    model: Dict[str, Any]
    cases: List[EvidenceCase]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceDocument":
        version = int(data.get("version", 1))
        if version != 1:
            raise ValueError(f"Unsupported profiling evidence version: {version}")
        cases = [EvidenceCase.from_dict(case) for case in data.get("cases", [])]
        if not cases:
            raise ValueError("Profiling evidence must contain at least one case.")
        return cls(version=version, model=dict(data.get("model", {})), cases=cases)


def load_evidence(path: Union[str, Path]) -> EvidenceDocument:
    evidence_path = Path(path)
    with evidence_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Profiling evidence root must be a mapping.")
    return EvidenceDocument.from_dict(data)
