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
"""Strict parsers from YAML mappings to comparison options."""

from __future__ import annotations

from collections.abc import Mapping

from tools.model_diagnostics.domain import (
    BoundaryEqualOptions,
    ConcatOptions,
    OneToOneOptions,
    TensorDirection,
    TensorMapping,
    TensorMappingMode,
    TensorRelation,
    TensorSlot,
    TensorSlotPair,
    TensorSlotRef,
)
from tools.model_diagnostics.schema_utils import SchemaGuard


class ComparisonOptionParseError(ValueError):
    """Typed comparison options cannot be built from the supplied mapping."""


_SCHEMA = SchemaGuard(
    error=ComparisonOptionParseError,
    accept_any_mapping=False,
    kind_mapping="object",
    kind_list="array",
    text_non_empty=True,
    require_string_keys=False,
)
_object = _SCHEMA.mapping
_array = _SCHEMA.sequence
_text = _SCHEMA.text
_optional_text = _SCHEMA.optional_text
_integer = _SCHEMA.integer
_optional_integer = _SCHEMA.optional_integer


def _exact_keys(raw: Mapping[str, object], expected: set[str], location: str) -> None:
    _SCHEMA.exact_keys(raw, required=expected, label=location)


class OneToOneOptionParser:
    def parse(self, raw: Mapping[str, object]) -> OneToOneOptions:
        if not raw:
            return OneToOneOptions(mapping=TensorMapping(mode=TensorMappingMode.POSITIONAL))
        _exact_keys(raw, {"mapping"}, "one_to_one options")
        return OneToOneOptions(
            mapping=_mapping(_object(raw["mapping"], "mapping")),
        )


class BoundaryEqualOptionParser:
    def parse(self, raw: Mapping[str, object]) -> BoundaryEqualOptions:
        _exact_keys(raw, {"mapping"}, "boundary_equal options")
        return BoundaryEqualOptions(mapping=_mapping(_object(raw["mapping"], "mapping")))


class ConcatOptionParser:
    def parse(self, raw: Mapping[str, object]) -> ConcatOptions:
        if not raw:
            return ConcatOptions(
                mapping=TensorMapping(mode=TensorMappingMode.COMPOSITE),
                axis=-1,
            )
        _SCHEMA.exact_keys(raw, required=set(), optional={"mapping", "axis"}, label="concat options")
        return ConcatOptions(
            mapping=(
                _mapping(_object(raw["mapping"], "mapping"))
                if "mapping" in raw
                else TensorMapping(mode=TensorMappingMode.COMPOSITE)
            ),
            axis=_integer(raw.get("axis", -1), "axis"),
        )


def _mapping(raw: Mapping[str, object]) -> TensorMapping:
    _exact_keys(raw, {"mode", "pairs", "relations"}, "mapping")
    mode_raw = _text(raw["mode"], "mapping.mode")
    try:
        mode = TensorMappingMode(mode_raw)
    except ValueError as error:
        raise ComparisonOptionParseError(f"unsupported mapping.mode {mode_raw!r}") from error
    pairs = tuple(
        _pair(_object(item, f"mapping.pairs[{index}]"), index)
        for index, item in enumerate(_array(raw["pairs"], "mapping.pairs"))
    )
    relations = tuple(
        _relation(_object(item, f"mapping.relations[{index}]"), index)
        for index, item in enumerate(_array(raw["relations"], "mapping.relations"))
    )
    if mode is TensorMappingMode.POSITIONAL and (pairs or relations):
        raise ComparisonOptionParseError("positional mapping cannot contain pairs or relations")
    if mode is TensorMappingMode.EXPLICIT and (not pairs or relations):
        raise ComparisonOptionParseError("explicit mapping requires pairs only")
    if mode is TensorMappingMode.COMPOSITE and (pairs or not relations):
        raise ComparisonOptionParseError("composite mapping requires relations only")
    return TensorMapping(mode=mode, pairs=pairs, relations=relations)


def _pair(raw: Mapping[str, object], index: int) -> TensorSlotPair:
    location = f"mapping.pairs[{index}]"
    _exact_keys(
        raw,
        {"left_call_index", "left_slot", "right_call_index", "right_slot"},
        location,
    )
    return TensorSlotPair(
        left_call_index=_integer(raw["left_call_index"], f"{location}.left_call_index"),
        left_slot=_slot(_object(raw["left_slot"], f"{location}.left_slot"), f"{location}.left_slot"),
        right_call_index=_integer(raw["right_call_index"], f"{location}.right_call_index"),
        right_slot=_slot(
            _object(raw["right_slot"], f"{location}.right_slot"),
            f"{location}.right_slot",
        ),
    )


def _relation(raw: Mapping[str, object], index: int) -> TensorRelation:
    location = f"mapping.relations[{index}]"
    _exact_keys(raw, {"left", "right", "operation", "axis"}, location)
    return TensorRelation(
        left=tuple(
            _slot_ref(_object(item, f"{location}.left[{position}]"), f"{location}.left[{position}]")
            for position, item in enumerate(_array(raw["left"], f"{location}.left"))
        ),
        right=tuple(
            _slot_ref(
                _object(item, f"{location}.right[{position}]"),
                f"{location}.right[{position}]",
            )
            for position, item in enumerate(_array(raw["right"], f"{location}.right"))
        ),
        operation=_text(raw["operation"], f"{location}.operation"),
        axis=_optional_integer(raw["axis"], f"{location}.axis"),
    )


def _slot_ref(raw: Mapping[str, object], location: str) -> TensorSlotRef:
    _exact_keys(raw, {"call_index", "slot"}, location)
    return TensorSlotRef(
        call_index=_integer(raw["call_index"], f"{location}.call_index"),
        slot=_slot(_object(raw["slot"], f"{location}.slot"), f"{location}.slot"),
    )


def _slot(raw: Mapping[str, object], location: str) -> TensorSlot:
    _exact_keys(raw, {"direction", "index", "name"}, location)
    direction_raw = _text(raw["direction"], f"{location}.direction")
    try:
        direction = TensorDirection(direction_raw)
    except ValueError as error:
        raise ComparisonOptionParseError(f"unsupported {location}.direction {direction_raw!r}") from error
    return TensorSlot(
        direction=direction,
        index=_integer(raw["index"], f"{location}.index"),
        name=_optional_text(raw["name"], f"{location}.name"),
    )
