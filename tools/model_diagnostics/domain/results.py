# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Immutable diagnostic findings and aggregate result values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from tools.model_diagnostics.domain.artifact import ProducerInfo
from tools.model_diagnostics.domain.models import ExecutionPhase, SourceKind, TensorSlot

DiagnosticValue = str | int | float | bool | tuple[int, ...] | None


class FindingStatus(Enum):
    PASS = "pass"  # nosec B105 - result status, not a credential
    FAIL = "fail"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"
    SKIP = "skip"


@dataclass(frozen=True)
class EvidenceRef:
    source_kind: SourceKind
    call_index: int
    stage_call_position: int
    operator_name: str
    tensor_slot: TensorSlot | None = None
    source_reference: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("call_index", "stage_call_position"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if not self.operator_name.strip():
            raise ValueError("operator_name must not be empty")


@dataclass(frozen=True)
class Finding:
    region_id: str
    layer_index: int | None
    stage_id: str
    rule_id: str
    comparison_kind: str
    status: FindingStatus
    message_code: str
    message: str
    expected: DiagnosticValue = None
    actual: DiagnosticValue = None
    left_evidence: tuple[EvidenceRef, ...] = ()
    right_evidence: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("region_id", "stage_id", "rule_id", "comparison_kind", "message_code", "message"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.layer_index is not None:
            if isinstance(self.layer_index, bool) or not isinstance(self.layer_index, int):
                raise TypeError("layer_index must be an integer")
            if self.layer_index < 0:
                raise ValueError("layer_index must be non-negative")


@dataclass(frozen=True)
class DiagnosticsSummary:
    overall_status: FindingStatus
    counts_by_status: Mapping[FindingStatus, int]

    def __post_init__(self) -> None:
        counts = {status: self.counts_by_status.get(status, 0) for status in FindingStatus}
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts.values()):
            raise ValueError("finding status counts must be non-negative integers")
        object.__setattr__(self, "counts_by_status", MappingProxyType(counts))


@dataclass(frozen=True)
class ModelRunContextSummary:
    model_name: str
    entrypoint: str | None
    phase: ExecutionPhase | None
    batch_size: int | None
    query_length: int | None
    context_length: int | None
    tensor_parallel_size: int | None = None


@dataclass(frozen=True)
class SourceDescription:
    source_kind: SourceKind
    artifact_reference: str | None = None
    producer: ProducerInfo | None = None


@dataclass(frozen=True)
class Limitation:
    code: str
    message: str

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.message.strip():
            raise ValueError("limitation code and message must not be empty")


@dataclass(frozen=True)
class DiagnosticsResult:
    schema_version: str
    spec_id: str
    spec_version: str
    context: ModelRunContextSummary
    left_source: SourceDescription
    right_source: SourceDescription
    selected_layers: Mapping[str, tuple[int, ...]]
    selected_stage_regions: tuple[str, ...]
    findings: tuple[Finding, ...]
    summary: DiagnosticsSummary
    limitations: tuple[Limitation, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("schema_version", "spec_id", "spec_version"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        object.__setattr__(
            self,
            "selected_layers",
            MappingProxyType(dict(self.selected_layers)),
        )


_STATUS_PRECEDENCE = (
    FindingStatus.FAIL,
    FindingStatus.INCOMPLETE,
    FindingStatus.UNSUPPORTED,
    FindingStatus.SKIP,
    FindingStatus.PASS,
)


def summarize_findings(findings: tuple[Finding, ...]) -> DiagnosticsSummary:
    counts = {status: 0 for status in FindingStatus}
    for finding in findings:
        counts[finding.status] += 1
    overall = next((status for status in _STATUS_PRECEDENCE if counts[status]), FindingStatus.PASS)
    return DiagnosticsSummary(overall_status=overall, counts_by_status=counts)
