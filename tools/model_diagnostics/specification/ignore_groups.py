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
"""Reusable, semantic Runtime ignored-operator groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import yaml

from tools.model_diagnostics.schema_utils import SchemaGuard, load_yaml_strict
from tools.model_diagnostics.specification.errors import SpecificationLoadError

_DEFAULT_GROUPS_PATH = Path(__file__).resolve().parents[1] / "specs" / "runtime" / "ignore_groups.yaml"
_SCHEMA = SchemaGuard(
    error=SpecificationLoadError,
    accept_any_mapping=True,
    kind_mapping="mapping",
    kind_list="list",
    require_string_keys=True,
)


@lru_cache(maxsize=1)
def load_builtin_ignore_groups() -> Mapping[str, tuple[str, ...]]:
    """Load the single builtin registry and reject ambiguous definitions."""

    try:
        raw = load_yaml_strict(_DEFAULT_GROUPS_PATH.read_text(encoding="utf-8"))
    except OSError as error:
        raise SpecificationLoadError(
            f"cannot read Runtime ignore groups: {_DEFAULT_GROUPS_PATH}"
        ) from error
    except yaml.YAMLError as error:
        raise SpecificationLoadError(
            f"invalid Runtime ignore groups YAML: {_DEFAULT_GROUPS_PATH}: {error}"
        ) from error
    payload = _SCHEMA.mapping(raw, "Runtime ignore groups")
    _SCHEMA.exact_keys(
        payload,
        required={"schema_version", "groups"},
        label="Runtime ignore groups",
    )
    if _SCHEMA.text(payload["schema_version"], "schema_version") != "1":
        raise SpecificationLoadError("unsupported Runtime ignore groups schema_version")
    groups_raw = _SCHEMA.mapping(payload["groups"], "Runtime ignore groups.groups")
    if not groups_raw:
        raise SpecificationLoadError("Runtime ignore groups.groups must not be empty")
    groups: dict[str, tuple[str, ...]] = {}
    for group_id, values in groups_raw.items():
        name = _SCHEMA.text(group_id, "Runtime ignore group id")
        operators = _operator_names(
            _SCHEMA.sequence(values, f"Runtime ignore group {name!r}"),
            f"Runtime ignore group {name!r}",
        )
        groups[name] = operators
    return MappingProxyType(groups)


def expand_ignore_groups(
    group_ids: Sequence[object],
    *,
    registry: Mapping[str, tuple[str, ...]],
    label: str,
) -> tuple[str, ...]:
    """Expand named groups in declaration order without widening scope."""

    expanded: list[str] = []
    seen_groups: set[str] = set()
    for index, value in enumerate(group_ids):
        group_id = _SCHEMA.text(value, f"{label}[{index}]")
        if group_id in seen_groups:
            raise SpecificationLoadError(
                f"{label} contains duplicate ignore group {group_id!r}"
            )
        seen_groups.add(group_id)
        try:
            expanded.extend(registry[group_id])
        except KeyError as error:
            raise SpecificationLoadError(
                f"{label}[{index}] references unknown ignore group {group_id!r}"
            ) from error
    return merge_operator_names(expanded)


def merge_operator_names(*collections: Sequence[str]) -> tuple[str, ...]:
    """Stable union used for group expansion and stage-local additions."""

    return tuple(dict.fromkeys(name for names in collections for name in names))


def _operator_names(values: Sequence[object], label: str) -> tuple[str, ...]:
    operators = tuple(
        _SCHEMA.text(value, f"{label}[{index}]")
        for index, value in enumerate(values)
    )
    if not operators:
        raise SpecificationLoadError(f"{label} must not be empty")
    if len(operators) != len(set(operators)):
        raise SpecificationLoadError(f"{label} contains duplicate operators")
    return operators
