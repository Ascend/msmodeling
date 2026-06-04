import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclasses.dataclass(frozen=True)
class UserHint:
    kind: str
    data: Dict[str, Any]
    confidence: str = "medium"
    note: Optional[str] = None
    source: str = "user"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserHint":
        if "kind" not in data:
            raise ValueError("Each user hint must include 'kind'.")
        payload = dict(data)
        kind = str(payload.pop("kind"))
        confidence = str(payload.pop("confidence", "medium"))
        note = payload.pop("note", None)
        source = str(payload.pop("source", "user"))
        return cls(kind=kind, data=payload, confidence=confidence, note=note, source=source)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "kind": self.kind,
            "confidence": self.confidence,
            "source": self.source,
            **self.data,
        }
        if self.note is not None:
            data["note"] = self.note
        return data


@dataclasses.dataclass(frozen=True)
class HintConflict:
    category: str
    message: str
    severity: str
    hint: Dict[str, Any]
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class HintLedger:
    hints: List[UserHint]
    model_id: Optional[str] = None
    version: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HintLedger":
        version = int(data.get("version", 1))
        hints = [UserHint.from_dict(item) for item in data.get("hints", [])]
        return cls(
            version=version,
            model_id=data.get("model_id"),
            hints=hints,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "model_id": self.model_id,
            "hints": [hint.to_dict() for hint in self.hints],
        }

    def conflicts_with_raw_insight(self, raw_insight: Any) -> List[HintConflict]:
        if raw_insight is None:
            return []
        counts = {kernel.normalized_name: kernel.occurrences for kernel in getattr(raw_insight, "kernels", [])}
        conflicts: List[HintConflict] = []
        for hint in self.hints:
            if hint.kind == "profiling_op_observation":
                op_name = hint.data.get("op") or hint.data.get("profiling_op")
                if not op_name:
                    continue
                observed_count = counts.get(str(op_name))
                hinted_count = hint.data.get("count")
                if observed_count is None:
                    conflicts.append(
                        HintConflict(
                            category="HINT_RAW_KERNEL_MISSING",
                            message="User hint references a profiling op that is absent from raw Insight.",
                            severity="warning",
                            hint=hint.to_dict(),
                            expected=str(op_name),
                            actual=None,
                        )
                    )
                hinted_count_int = _optional_int(hinted_count)
                if hinted_count is not None and hinted_count_int is None:
                    conflicts.append(
                        HintConflict(
                            category="HINT_COUNT_INVALID",
                            message="User hinted profiling op count is not an integer.",
                            severity="warning",
                            hint=hint.to_dict(),
                            expected="integer count",
                            actual=hinted_count,
                        )
                    )
                elif hinted_count_int is not None and hinted_count_int != observed_count:
                    conflicts.append(
                        HintConflict(
                            category="HINT_COUNT_CONFLICT",
                            message="User hinted profiling op count differs from raw Insight occurrences.",
                            severity="warning",
                            hint=hint.to_dict(),
                            expected=hinted_count_int,
                            actual=observed_count,
                        )
                    )
            elif hint.kind == "op_mapping_hint":
                profiling_op = hint.data.get("profiling_op")
                if profiling_op and str(profiling_op) not in counts:
                    conflicts.append(
                        HintConflict(
                            category="HINT_MAPPING_SOURCE_MISSING",
                            message="User mapping hint references a profiling op absent from raw Insight.",
                            severity="warning",
                            hint=hint.to_dict(),
                            expected=str(profiling_op),
                            actual=None,
                        )
                    )
        return conflicts


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_hints(path: Union[str, Path]) -> HintLedger:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Hint ledger root must be a mapping.")
    return HintLedger.from_dict(data)
