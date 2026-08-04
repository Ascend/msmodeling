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
"""Immutable model specification values."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, TypeVar

from tools.model_diagnostics.domain.models import SourceKind, TensorSlot

_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


def _freeze_mapping(values: Mapping[_Key, _Value]) -> Mapping[_Key, _Value]:
    return MappingProxyType(dict(values))


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _reject_duplicate_ids(values: tuple[object, ...], attribute: str) -> None:
    identifiers = tuple(getattr(value, attribute) for value in values)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate {attribute}")


@dataclass(frozen=True)
class ShapeExpr:
    expression: str

    def __post_init__(self) -> None:
        _require_text(self.expression, "shape expression")


@dataclass(frozen=True)
class DTypeExpr:
    expression: str

    def __post_init__(self) -> None:
        _require_text(self.expression, "dtype expression")


@dataclass(frozen=True)
class TheoryTensorSpec:
    shape: ShapeExpr | None = None
    dtype: DTypeExpr | None = None

    def __post_init__(self) -> None:
        if self.shape is None and self.dtype is None:
            raise ValueError("theory tensor spec must declare shape or dtype")


@dataclass(frozen=True)
class TheoryOperatorSpec:
    operator_name: str
    tensors: Mapping[TensorSlot, TheoryTensorSpec] = field(default_factory=dict)
    activation: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.operator_name, "theory operator_name")
        object.__setattr__(self, "tensors", _freeze_mapping(self.tensors))
        if self.activation is not None:
            _require_text(self.activation, "theory operator activation")


@dataclass(frozen=True)
class TheoryStageOptions:
    operators: tuple[TheoryOperatorSpec, ...]

    def __post_init__(self) -> None:
        if not self.operators:
            raise ValueError("theory operators must not be empty")


@dataclass(frozen=True)
class RuntimeStageOptions:
    boundary_operators: tuple[str, ...]
    ignored_operators: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.boundary_operators:
            raise ValueError("runtime boundary_operators must not be empty")
        for operator in self.boundary_operators + self.ignored_operators:
            _require_text(operator, "runtime operator")


SourceStageOptions = TheoryStageOptions | RuntimeStageOptions


class TensorMappingMode(Enum):
    POSITIONAL = "positional"
    EXPLICIT = "explicit"
    COMPOSITE = "composite"


@dataclass(frozen=True)
class TensorSlotPair:
    left_call_index: int
    left_slot: TensorSlot
    right_call_index: int
    right_slot: TensorSlot


@dataclass(frozen=True)
class TensorSlotRef:
    call_index: int
    slot: TensorSlot


@dataclass(frozen=True)
class TensorRelation:
    left: tuple[TensorSlotRef, ...]
    right: tuple[TensorSlotRef, ...]
    operation: str
    axis: int | None = None


@dataclass(frozen=True)
class TensorMapping:
    mode: TensorMappingMode
    pairs: tuple[TensorSlotPair, ...] = ()
    relations: tuple[TensorRelation, ...] = ()


@dataclass(frozen=True)
class OneToOneOptions:
    mapping: TensorMapping


@dataclass(frozen=True)
class ConcatOptions:
    mapping: TensorMapping
    axis: int


@dataclass(frozen=True)
class BoundaryEqualOptions:
    mapping: TensorMapping


ComparisonOptions = OneToOneOptions | ConcatOptions | BoundaryEqualOptions


@dataclass(frozen=True)
class ComparisonSpec:
    strategy_id: str
    options: ComparisonOptions

    def __post_init__(self) -> None:
        _require_text(self.strategy_id, "comparison strategy_id")


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    source_options: Mapping[SourceKind, SourceStageOptions]
    comparisons: Mapping[tuple[SourceKind, SourceKind], ComparisonSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.stage_id, "stage_id")
        for source_kind, options in self.source_options.items():
            expected_type = TheoryStageOptions if source_kind is SourceKind.THEORY else RuntimeStageOptions
            if not isinstance(options, expected_type):
                raise TypeError(f"source options do not match {source_kind.value}")
        object.__setattr__(self, "source_options", _freeze_mapping(self.source_options))
        object.__setattr__(self, "comparisons", _freeze_mapping(self.comparisons))


@dataclass(frozen=True)
class LayerSpec:
    layer_kind: str
    stages: tuple[StageSpec, ...]

    def __post_init__(self) -> None:
        _require_text(self.layer_kind, "layer_kind")
        _reject_duplicate_ids(self.stages, "stage_id")


@dataclass(frozen=True)
class RegionSpec:
    region_id: str
    stages: tuple[StageSpec, ...] = ()
    layer_layout: tuple[str, ...] = ()
    layer_specs: Mapping[str, LayerSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.region_id, "region_id")
        _reject_duplicate_ids(self.stages, "stage_id")
        for layer_kind, layer_spec in self.layer_specs.items():
            if layer_kind != layer_spec.layer_kind:
                raise ValueError("layer_specs key must match layer_kind")
        unknown_layer_kinds = set(self.layer_layout).difference(self.layer_specs)
        if unknown_layer_kinds:
            names = ", ".join(sorted(unknown_layer_kinds))
            raise ValueError(f"unknown layer_kind in layer_layout: {names}")
        object.__setattr__(self, "layer_specs", _freeze_mapping(self.layer_specs))


@dataclass(frozen=True)
class SpecMatchCriteria:
    entrypoints: tuple[str, ...] = ()
    model_types: tuple[str, ...] = ()
    required_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (self.entrypoints or self.model_types or self.required_features):
            raise ValueError("at least one match criterion is required")


@dataclass(frozen=True)
class ModelDiagnosticsSpec:
    schema_version: str
    spec_id: str
    spec_version: str
    model_category: str
    matches: SpecMatchCriteria
    regions: tuple[RegionSpec, ...]
    operator_aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("schema_version", "spec_id", "spec_version", "model_category"):
            _require_text(getattr(self, field_name), field_name)
        if not self.regions:
            raise ValueError("regions must not be empty")
        _reject_duplicate_ids(self.regions, "region_id")
        for key, value in self.operator_aliases.items():
            _require_text(key, "operator_aliases key")
            _require_text(value, "operator_aliases value")
        object.__setattr__(self, "operator_aliases", _freeze_mapping(self.operator_aliases))
