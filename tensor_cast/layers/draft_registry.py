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

"""Draft plugin contracts, independent of concrete model implementations.

Only trusted registration code supplies implementation paths. Config files
select registered family keys and never supply Python import paths.
"""

from collections.abc import Callable
from importlib import import_module

_CONFIG_PARSERS: dict[str, Callable[[dict], dict]] = {}
_DSPARK_FAMILIES: dict[str, tuple[str, str]] = {}


def register_draft_config_parser(architecture: str, parser: Callable[[dict], dict]) -> None:
    previous = _CONFIG_PARSERS.get(architecture)
    if previous is not None and previous is not parser:
        raise ValueError(f"Draft config parser already registered: {architecture}")
    _CONFIG_PARSERS[architecture] = parser


def register_dspark_family(family: str, draft_type: str, wrapper_type: str) -> None:
    implementation = (draft_type, wrapper_type)
    previous = _DSPARK_FAMILIES.get(family)
    if previous is not None and previous != implementation:
        raise ValueError(f"DSpark family already registered: {family}")
    _DSPARK_FAMILIES[family] = implementation


def normalize_draft_config(source: dict) -> dict:
    parsers = [_CONFIG_PARSERS[name] for name in source.get("architectures", []) if name in _CONFIG_PARSERS]
    if not parsers:
        return source
    if any(parser is not parsers[0] for parser in parsers[1:]):
        raise ValueError("Draft config matches multiple architecture parsers")
    return parsers[0](source)


def _resolve_type(path: str) -> type:
    module, name = path.split(":", maxsplit=1)
    implementation = getattr(import_module(module), name)
    if not isinstance(implementation, type):
        raise TypeError(f"Registered draft implementation is not a class: {path}")
    return implementation


def get_dspark_family(family: str | None, *, default: tuple[type, type]) -> tuple[type, type]:
    if family is None:
        return default
    try:
        draft_type, wrapper_type = _DSPARK_FAMILIES[family]
    except KeyError as exc:
        raise ValueError(f"Unregistered DSpark draft family: {family}") from exc
    return _resolve_type(draft_type), _resolve_type(wrapper_type)
