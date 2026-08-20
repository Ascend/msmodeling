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
"""Core domain values with no simulation or rendering dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from tools.model_diagnostics.domain.selection import (
    normalize_selected_layers,
    normalize_selected_stage_regions,
)
from tools.model_diagnostics.errors import InvalidDiagnosticsRequest

TensorShape = tuple[int, ...]
DType = str


def _freeze_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(values))


def _require_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def validate_expert_parallel_features(
    expert_parallel_size: int,
    *,
    enable_external_shared_experts: bool,
    enable_redundant_experts: bool,
) -> None:
    """Require expert-parallel execution for features that shard expert ownership."""

    if expert_parallel_size <= 1 and (enable_external_shared_experts or enable_redundant_experts):
        raise ValueError(
            "enable_external_shared_experts / enable_redundant_experts require expert_parallel_size > 1"
        )


class ExecutionPhase(Enum):
    """Simulation phase that affects expected tensor relationships."""

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class ParallelContext:
    """Parallel degrees used to resolve model specifications.

    MoE tensor parallel (--moe-tp-size / MTPt) is not supported by this module:
    it is fixed at 1 internally. Do not re-add a configurable MTPt field unless
    the MTPt>1 Runtime structure difference is modeled.
    """

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    # MoE data parallel (--moe-dp-size / MDP). Defaults to 1 when omitted.
    moe_data_parallel_size: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "expert_parallel_size",
            "moe_data_parallel_size",
        ):
            _require_positive_integer(getattr(self, field_name), field_name)


@dataclass(frozen=True)
class ModelRunContext:
    """Stable business-run information shared by all evidence sources.

    ``entrypoint`` identifies the user-visible model workflow, such as
    ``text_generate``. Evidence adapters must not replace it with their capture
    function name.
    """

    model_name: str
    entrypoint: str | None
    phase: ExecutionPhase | None
    batch_size: int | None
    query_length: int | None
    context_length: int | None
    parallel: ParallelContext
    model_config: Mapping[str, object]
    quantization_config: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.entrypoint is not None and not self.entrypoint.strip():
            raise ValueError("entrypoint must be None or a non-empty business entrypoint")
        for field_name in ("batch_size", "query_length"):
            value = getattr(self, field_name)
            if value is not None:
                _require_positive_integer(value, field_name)
        if self.context_length is not None:
            if isinstance(self.context_length, bool) or not isinstance(self.context_length, int):
                raise TypeError("context_length must be an integer")
            if self.context_length < 0:
                raise ValueError("context_length must be non-negative")
        object.__setattr__(self, "model_config", _freeze_mapping(self.model_config))
        object.__setattr__(self, "quantization_config", _freeze_mapping(self.quantization_config))


@dataclass(frozen=True)
class DiagnosticsRequest:
    """Explicit region and physical-layer selection for one diagnostic run."""

    context: ModelRunContext
    selected_layers: Mapping[str, tuple[int, ...]]
    selected_stage_regions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_layers = normalize_selected_layers(self.selected_layers)
        normalized_regions = normalize_selected_stage_regions(self.selected_stage_regions)
        if not normalized_layers and not normalized_regions:
            raise InvalidDiagnosticsRequest("at least one region or layer selection is required")

        object.__setattr__(self, "selected_layers", normalized_layers)
        object.__setattr__(self, "selected_stage_regions", normalized_regions)


class SourceKind(Enum):
    """Evidence source participating in a pairwise diagnostic."""

    THEORY = "theory"
    RUNTIME = "runtime"


class TensorDirection(Enum):
    """Direction of a tensor relative to an operator call."""

    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


@dataclass(frozen=True)
class TensorSlot:
    """Stable tensor identity within one operator call."""

    direction: TensorDirection
    index: int
    name: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError("tensor slot index must be an integer")
        if self.index < 0:
            raise ValueError("tensor slot index must be non-negative")


class _TensorSlots:
    def __init__(self, direction: TensorDirection) -> None:
        self._direction = direction

    def __getitem__(self, index: int) -> TensorSlot:
        return TensorSlot(self._direction, index)


INPUT = _TensorSlots(TensorDirection.INPUT)
OUTPUT = _TensorSlots(TensorDirection.OUTPUT)


@dataclass(frozen=True)
class TensorInfo:
    """Shape and dtype evidence for one operator tensor slot."""

    slot: TensorSlot
    shape: TensorShape | None
    dtype: DType | None

    def __post_init__(self) -> None:
        if self.shape is not None:
            if any(isinstance(dimension, bool) or not isinstance(dimension, int) for dimension in self.shape):
                raise TypeError("tensor shape dimensions must be integers")
            if any(dimension < 0 for dimension in self.shape):
                raise ValueError("tensor shape dimensions must be non-negative")
            object.__setattr__(self, "shape", tuple(self.shape))


@dataclass(frozen=True)
class OperatorCallRecord:
    """One ordered semantic-operator call from a source."""

    call_index: int
    operator_name: str
    original_operator_name: str | None
    tensors: tuple[TensorInfo, ...]
    source_reference: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.call_index, bool) or not isinstance(self.call_index, int):
            raise TypeError("call_index must be an integer")
        if self.call_index < 0:
            raise ValueError("call_index must be non-negative")
        if not self.operator_name.strip():
            raise ValueError("operator_name must not be empty")
        object.__setattr__(self, "tensors", tuple(self.tensors))


@dataclass(frozen=True)
class ModelExecutionRecord:
    """Ordered calls emitted by one evidence source for a shared context."""

    source_kind: SourceKind
    run_context: ModelRunContext
    operator_calls: tuple[OperatorCallRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator_calls", tuple(self.operator_calls))
        indices = tuple(call.call_index for call in self.operator_calls)
        if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
            raise ValueError("operator call indices must be unique and ordered")
