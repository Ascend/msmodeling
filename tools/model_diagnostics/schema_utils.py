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

"""Strict YAML/JSON boundary helpers shared by Spec, comparison, and Artifact codecs."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Reject explicit duplicate keys while preserving YAML merge precedence."""

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._checked_mappings: set[MappingNode] = set()

    def flatten_mapping(self, node: MappingNode) -> None:
        # Aliases can revisit a node after its inherited keys have been inserted.
        if node in self._checked_mappings:
            return
        key_marks: dict[object, object] = {}
        for key_node, _ in node.value:
            if key_node.tag in {"tag:yaml.org,2002:merge", "tag:yaml.org,2002:value"}:
                key = key_node.value
            else:
                key = self.construct_object(key_node, deep=True)
            try:
                duplicate = key in key_marks
            except TypeError as error:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from error
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    key_marks[key],
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            key_marks[key] = key_node.start_mark
        self._checked_mappings.add(node)
        # SafeLoader recursively calls this override for merged mappings too.
        super().flatten_mapping(node)


def load_yaml_strict(text: str) -> object:
    """Safely parse YAML while rejecting duplicate keys at every depth."""

    return yaml.load(text, Loader=_UniqueKeySafeLoader)


@dataclass(frozen=True)
class SchemaGuard:
    """Bound error type and wording for one wire/schema boundary."""

    error: type[BaseException]
    accept_any_mapping: bool = False
    kind_mapping: str = "object"
    kind_list: str = "array"
    text_non_empty: bool = True
    require_string_keys: bool = False

    def exact_keys(
        self,
        raw: Mapping[str, Any],
        *,
        required: Set[str],
        optional: Set[str] = frozenset(),
        label: str,
    ) -> None:
        if self.require_string_keys and any(not isinstance(key, str) for key in raw):
            raise self.error(f"{label} field names must be strings")
        actual = {str(key) for key in raw} if self.require_string_keys else set(raw)
        missing = required.difference(actual)
        unknown = actual.difference(set(required).union(optional))
        if missing or unknown:
            # Sort by repr so mixed key types (when require_string_keys is False)
            # cannot raise TypeError and bypass self.error.
            raise self.error(
                f"{label} fields mismatch: "
                f"missing={sorted(missing, key=repr)}, unknown={sorted(unknown, key=repr)}"
            )

    def mapping(self, value: object, label: str) -> Mapping[str, Any]:
        if self.accept_any_mapping:
            if not isinstance(value, Mapping):
                raise self.error(f"{label} must be a {self.kind_mapping}")
            return value
        if not isinstance(value, dict):
            raise self.error(f"{label} must be a {self.kind_mapping}")
        return value

    def sequence(self, value: object, label: str) -> list[Any]:
        if not isinstance(value, list):
            raise self.error(f"{label} must be a {self.kind_list}")
        return value

    def text(self, value: object, label: str) -> str:
        if not isinstance(value, str):
            raise self.error(f"{label} must be a string")
        if self.text_non_empty and not value.strip():
            raise self.error(f"{label} must be a non-empty string")
        return value

    def optional_text(self, value: object, label: str) -> str | None:
        if value is None:
            return None
        return self.text(value, label)

    def integer(self, value: object, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise self.error(f"{label} must be an integer")
        return value

    def optional_integer(self, value: object, label: str) -> int | None:
        if value is None:
            return None
        return self.integer(value, label)
