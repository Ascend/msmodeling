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

"""Query helpers for test_node -> source_file -> symbols test_map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from scripts.helpers.ci_gate.models import CiGatePolicy

from scripts.helpers.ci_gate.gate_policy import is_config_path, is_source_path, is_test_path

TestMap = dict[str, dict[str, list[str]]]


@dataclass(frozen=True, slots=True)
class TouchedPaths:
    """Map-relevant paths changed between two authoritative map commits."""

    tests: frozenset[str]
    sources: frozenset[str]
    configs: frozenset[str]


@dataclass(frozen=True, slots=True)
class TestMapIndex:
    """Reverse indexes for O(1) source/symbol watcher lookups."""

    _source_watchers: dict[str, frozenset[str]]
    _symbol_watchers: dict[tuple[str, str], frozenset[str]]

    def source_watchers(self, source_file: str) -> frozenset[str]:
        return self._source_watchers.get(source_file, frozenset())

    def symbol_watchers(self, source_file: str, symbol: str) -> frozenset[str]:
        return self._symbol_watchers.get((source_file, symbol), frozenset())


def build_test_map_index(test_map: TestMap) -> TestMapIndex:
    """Build reverse indexes from a node-oriented test_map (one O(N) pass)."""
    by_source: dict[str, set[str]] = {}
    by_symbol: dict[tuple[str, str], set[str]] = {}
    for node, sources in test_map.items():
        for src_path, symbols in sources.items():
            by_source.setdefault(src_path, set()).add(node)
            for symbol in symbols:
                by_symbol.setdefault((src_path, symbol), set()).add(node)
    return TestMapIndex(
        _source_watchers={path: frozenset(nodes) for path, nodes in by_source.items()},
        _symbol_watchers={key: frozenset(nodes) for key, nodes in by_symbol.items()},
    )


def classify_touched_paths(paths: Iterable[str], policy: CiGatePolicy) -> TouchedPaths:
    """Classify changed paths using the same policy as the PR CI gate."""
    touched = frozenset(paths)
    return TouchedPaths(
        tests=frozenset(path for path in touched if is_test_path(path, policy)),
        sources=frozenset(path for path in touched if is_source_path(path, policy)),
        configs=frozenset(path for path in touched if is_config_path(path, policy)),
    )


def select_resync_targets(
    test_map: TestMap,
    touched: TouchedPaths,
    *,
    repo_root: Path,
) -> tuple[str, ...]:
    """Return existing changed test files and nodes watching changed sources."""
    index = build_test_map_index(test_map)
    targets = {path for path in touched.tests if (repo_root / path).is_file()}
    for source_file in touched.sources:
        if (repo_root / source_file).is_file():
            targets.update(index.source_watchers(source_file))
    return tuple(sorted(targets))


def has_unmapped_existing_source(
    test_map: TestMap,
    touched: TouchedPaths,
    *,
    repo_root: Path,
) -> bool:
    """Return whether a changed, still-existing source has no old-map watcher."""
    index = build_test_map_index(test_map)
    return any(
        (repo_root / source_file).is_file() and not index.source_watchers(source_file)
        for source_file in touched.sources
    )


def nodes_for_test_file(test_map: TestMap, test_file: str) -> frozenset[str]:
    prefix = f"{test_file}::"
    return frozenset(node for node in test_map if node.startswith(prefix))


def symbol_watchers(
    test_map: TestMap,
    source_file: str,
    symbol: str,
    *,
    index: TestMapIndex | None = None,
) -> frozenset[str]:
    """Return test nodes watching *symbol* on *source_file*.

    When *index* is omitted, scans the full map (O(N) per call). Pass a
    :class:`TestMapIndex` from :func:`build_test_map_index` when querying repeatedly.
    """
    if index is not None:
        return index.symbol_watchers(source_file, symbol)
    watchers: set[str] = set()
    for node, sources in test_map.items():
        if symbol in sources.get(source_file, ()):
            watchers.add(node)
    return frozenset(watchers)


def source_watchers(
    test_map: TestMap,
    source_file: str,
    *,
    index: TestMapIndex | None = None,
) -> frozenset[str]:
    """Return test nodes that reference *source_file*.

    When *index* is omitted, scans the full map (O(N) per call). Pass a
    :class:`TestMapIndex` from :func:`build_test_map_index` when querying repeatedly.
    """
    if index is not None:
        return index.source_watchers(source_file)
    watchers: set[str] = set()
    for node, sources in test_map.items():
        if sources.get(source_file):
            watchers.add(node)
    return frozenset(watchers)


def is_source_file_mapped(test_map: TestMap, source_file: str) -> bool:
    return bool(source_watchers(test_map, source_file))


def is_symbol_mapped(test_map: TestMap, source_file: str, symbol: str) -> bool:
    return bool(symbol_watchers(test_map, source_file, symbol))


def symbols_mapped_for_source(test_map: TestMap, source_file: str) -> frozenset[str]:
    symbols: set[str] = set()
    for sources in test_map.values():
        symbols.update(sources.get(source_file, ()))
    return frozenset(symbols)


def prune_deleted_sources(test_map: TestMap, deleted: Iterable[str]) -> TestMap:
    deleted_set = set(deleted)
    pruned: TestMap = {}
    for node, sources in test_map.items():
        kept = {path: syms for path, syms in sources.items() if path not in deleted_set}
        if kept:
            pruned[node] = kept
    return pruned
