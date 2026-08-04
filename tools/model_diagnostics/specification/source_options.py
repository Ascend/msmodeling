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
"""Injectable parsers for source-specific stage options."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Protocol

from tools.model_diagnostics.domain.models import INPUT, OUTPUT, SourceKind, TensorSlot
from tools.model_diagnostics.domain.specification import (
    DTypeExpr,
    RuntimeStageOptions,
    ShapeExpr,
    SourceStageOptions,
    TheoryOperatorSpec,
    TheoryStageOptions,
    TheoryTensorSpec,
)
from tools.model_diagnostics.schema_utils import SchemaGuard
from tools.model_diagnostics.specification.errors import SpecificationLoadError

if TYPE_CHECKING:
    from tools.model_diagnostics.specification.theory_fragments import TheoryFragmentRegistry

_SCHEMA = SchemaGuard(
    error=SpecificationLoadError,
    accept_any_mapping=True,
    kind_mapping="mapping",
    kind_list="list",
    text_non_empty=True,
    require_string_keys=True,
)
_exact_keys = _SCHEMA.exact_keys
_require_mapping = _SCHEMA.mapping
_require_list = _SCHEMA.sequence
_as_str = _SCHEMA.text


class SourceOptionsParser(Protocol):
    source_kind: SourceKind
    yaml_key: str

    def parse(self, raw: Mapping[str, object]) -> SourceStageOptions: ...


@dataclass(frozen=True)
class TheorySourceOptionsParser:
    source_kind = SourceKind.THEORY
    yaml_key = "theory"
    fragment_registry: TheoryFragmentRegistry | None = None

    def parse(self, raw: Mapping[str, object]) -> TheoryStageOptions:
        _exact_keys(
            raw,
            required=set(),
            optional={"modules", "include_module_groups", "include_stages"},
            label="theory options",
        )
        operators: list[TheoryOperatorSpec] = []
        if "modules" in raw:
            operators.extend(
                parse_theory_operator(item, index)
                for index, item in enumerate(_require_list(raw.get("modules"), "theory.modules"))
            )
        if "include_module_groups" in raw:
            if self.fragment_registry is None:
                raise SpecificationLoadError(
                    "theory.include_module_groups requires a TheoryFragmentRegistry"
                )
            for index, item in enumerate(
                _require_list(raw.get("include_module_groups"), "theory.include_module_groups")
            ):
                location = f"theory.include_module_groups[{index}]"
                ref = _require_mapping(item, location)
                _exact_keys(ref, required={"fragment", "group"}, label=location)
                fragment_id = _as_str(ref.get("fragment"), f"{location}.fragment")
                group_id = _as_str(ref.get("group"), f"{location}.group")
                operators.extend(self.fragment_registry.get(fragment_id).group(group_id))
        if "include_stages" in raw:
            if self.fragment_registry is None:
                raise SpecificationLoadError(
                    "theory.include_stages requires a TheoryFragmentRegistry"
                )
            for index, item in enumerate(
                _require_list(raw.get("include_stages"), "theory.include_stages")
            ):
                location = f"theory.include_stages[{index}]"
                ref = _require_mapping(item, location)
                _exact_keys(ref, required={"fragment", "stage"}, label=location)
                fragment_id = _as_str(ref.get("fragment"), f"{location}.fragment")
                stage_id = _as_str(ref.get("stage"), f"{location}.stage")
                operators.extend(self.fragment_registry.get(fragment_id).stage(stage_id).operators)
        if not operators:
            raise SpecificationLoadError(
                "theory options must declare modules, include_module_groups, and/or include_stages"
            )
        return TheoryStageOptions(operators=tuple(operators))


@dataclass(frozen=True)
class RuntimeSourceOptionsParser:
    source_kind = SourceKind.RUNTIME
    yaml_key = "runtime"

    def parse(self, raw: Mapping[str, object]) -> RuntimeStageOptions:
        _exact_keys(
            raw,
            required={"boundary_operators"},
            optional={"ignored_operators"},
            label="runtime options",
        )
        boundaries = tuple(
            _as_str(item, "runtime boundary operator")
            for item in _require_list(
                raw.get("boundary_operators"),
                "runtime.boundary_operators",
            )
        )
        ignored = tuple(
            _as_str(item, "runtime ignored operator")
            for item in _require_list(
                raw.get("ignored_operators", []),
                "runtime.ignored_operators",
            )
        )
        return RuntimeStageOptions(
            boundary_operators=boundaries,
            ignored_operators=ignored,
        )


def create_builtin_source_options_parsers(
    *,
    fragment_registry: TheoryFragmentRegistry | None = None,
) -> Mapping[str, SourceOptionsParser]:
    from tools.model_diagnostics.specification.theory_fragments import (
        load_builtin_theory_fragment_registry,
    )

    registry = fragment_registry if fragment_registry is not None else load_builtin_theory_fragment_registry()
    parsers: tuple[SourceOptionsParser, ...] = (
        TheorySourceOptionsParser(fragment_registry=registry),
        RuntimeSourceOptionsParser(),
    )
    return {parser.yaml_key: parser for parser in parsers}


_TENSOR_SLOT_KEY = re.compile(r"^(INPUT|OUTPUT)\[(0|[1-9][0-9]*)\]$")


def _tensor_slot_key(value: object, label: str) -> TensorSlot:
    key = _as_str(value, label)
    match = _TENSOR_SLOT_KEY.fullmatch(key)
    if match is None:
        raise SpecificationLoadError(f"{label} has invalid Tensor slot syntax: {key!r}")
    index = int(match.group(2))
    return INPUT[index] if match.group(1) == "INPUT" else OUTPUT[index]


def parse_theory_operator(value: object, index: int) -> TheoryOperatorSpec:
    location = f"theory.modules[{index}]"
    raw = _require_mapping(value, location)
    _exact_keys(raw, required={"name", "tensors"}, optional={"activation"}, label=location)
    tensors_raw = _require_mapping(raw.get("tensors"), f"{location}.tensors")
    tensors: dict[TensorSlot, TheoryTensorSpec] = {}
    for key, tensor in tensors_raw.items():
        slot = _tensor_slot_key(key, f"{location}.tensor key")
        if slot in tensors:
            raise SpecificationLoadError(f"{location}.tensors has duplicate Tensor slot: {key!r}")
        tensors[slot] = _theory_tensor(tensor, f"{location}.tensors[{key}]")
    return TheoryOperatorSpec(
        operator_name=_as_str(raw.get("name"), f"{location}.name"),
        tensors=tensors,
        activation=(
            _as_str(raw["activation"], f"{location}.activation") if "activation" in raw else None
        ),
    )


# Back-compat alias for fragment loader.
_theory_operator = parse_theory_operator


def _theory_tensor(value: object, location: str) -> TheoryTensorSpec:
    raw = _require_mapping(value, location)
    _exact_keys(
        raw,
        required=set(),
        optional={"shape", "dtype"},
        label=location,
    )
    if not raw:
        raise SpecificationLoadError(f"{location} must declare shape or dtype")
    return TheoryTensorSpec(
        shape=(ShapeExpr(_as_str(raw["shape"], f"{location}.shape")) if "shape" in raw else None),
        dtype=(DTypeExpr(_as_str(raw["dtype"], f"{location}.dtype")) if "dtype" in raw else None),
    )
