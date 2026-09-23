# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Candidate:
    candidate_id: str
    params: Dict[str, Any]
    rationale: str = ""
    expected_effect: str = ""
    risk: str = ""
    source: str = "agent"

    @classmethod
    def from_dict(cls, data: Dict[str, Any], index: int = 0) -> "Candidate":
        cid = str(data.get("candidate_id") or data.get("id") or f"candidate-{index + 1}")
        return cls(
            candidate_id=cid,
            params=dict(data.get("params", {})),
            rationale=str(data.get("rationale", "")),
            expected_effect=str(data.get("expected_effect", "")),
            risk=str(data.get("risk", "")),
            source=str(data.get("source", "agent")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "params": self.params,
            "rationale": self.rationale,
            "expected_effect": self.expected_effect,
            "risk": self.risk,
            "source": self.source,
        }


@dataclass
class CandidateValidation:
    candidate: Candidate
    valid: bool
    params: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "valid": self.valid,
            "params": self.params,
            "errors": self.errors,
            "warnings": self.warnings,
            "candidate": self.candidate.to_dict(),
        }


@dataclass
class TrialFailure:
    """Structured failure information for a trial."""

    category: str = "unknown"
    sub_category: str = ""
    message: str = ""
    evidence_line: str = ""
    offending_params: Dict[str, Any] = field(default_factory=dict)
    suggested_action: str = ""
    suggested_bounds: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "sub_category": self.sub_category,
            "message": self.message,
            "evidence_line": self.evidence_line,
            "offending_params": self.offending_params,
            "suggested_action": self.suggested_action,
            "suggested_bounds": self.suggested_bounds,
        }


@dataclass
class TrialResult:
    params: Dict[str, Any]
    fitness: float
    performance: Any = None
    candidate_id: Optional[str] = None
    error: Optional[str] = None
    source: str = "unknown"
    elapsed_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    early_exit: bool = False
    would_early_exit: bool = False
    failure: Optional[TrialFailure] = None
    slo_violated: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        performance = None
        if self.performance is not None:
            if hasattr(self.performance, "model_dump"):
                performance = self.performance.model_dump()
            elif hasattr(self.performance, "to_dict"):
                performance = self.performance.to_dict()
            else:
                try:
                    performance = dict(self.performance)
                except (TypeError, ValueError):
                    performance = str(self.performance)
        result = {
            "candidate_id": self.candidate_id,
            "params": self.params,
            "fitness": self.fitness,
            "performance": performance,
            "error": str(self.error) if self.error is not None else None,
            "source": self.source,
            "elapsed_seconds": self.elapsed_seconds,
            "early_exit": self.early_exit,
            "would_early_exit": self.would_early_exit,
            "metadata": self.metadata,
            "slo_violated": self.slo_violated,
        }
        if self.failure:
            result["failure"] = self.failure.to_dict()
        return result


@dataclass
class SearchResult:
    best_params: Dict[str, Any]
    best_fitness: float
    trials: List['TrialResult'] = field(default_factory=list)
    best_candidate: Optional[Candidate] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_fitness": self.best_fitness,
            "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            "trials": [t.to_dict() for t in self.trials],
        }
