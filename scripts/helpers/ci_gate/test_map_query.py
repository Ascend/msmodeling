"""Query helpers for test_node -> source_file -> symbols test_map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

TestMap = dict[str, dict[str, list[str]]]


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
