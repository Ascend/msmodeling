"""Tests for ci_gate.test_map_query."""

from __future__ import annotations

from scripts.helpers.ci_gate.test_map_query import (
    build_test_map_index,
    is_source_file_mapped,
    is_symbol_mapped,
    prune_deleted_sources,
    source_watchers,
    symbol_watchers,
    symbols_mapped_for_source,
)


def test_symbol_watchers_matches_canonical_symbol() -> None:
    test_map = {
        "tests/regression/cli/test_run.py::test_run": {
            "cli/main.py": ["Widget::run"],
        },
    }
    assert symbol_watchers(test_map, "cli/main.py", "Widget::run") == frozenset(
        {"tests/regression/cli/test_run.py::test_run"}
    )


def test_is_symbol_mapped_false_for_unrelated_symbol() -> None:
    test_map = {
        "tests/regression/cli/test_run.py::test_run": {
            "cli/main.py": ["Widget::run"],
        },
    }
    assert is_symbol_mapped(test_map, "cli/main.py", "Widget::stop") is False


def test_is_source_file_mapped_true_when_any_symbol_present() -> None:
    test_map = {
        "tests/regression/cli/test_run.py::test_run": {
            "cli/main.py": ["Widget::run"],
        },
    }
    assert is_source_file_mapped(test_map, "cli/main.py") is True
    assert is_source_file_mapped(test_map, "tensor_cast/ops.py") is False


def test_symbols_mapped_for_source_collects_unique_symbols() -> None:
    test_map = {
        "tests/regression/cli/test_run.py::test_run": {
            "cli/main.py": ["Widget::run", "Widget::stop"],
        },
        "tests/regression/cli/test_other.py::test_x": {
            "cli/main.py": ["Widget::run"],
            "tensor_cast/ops.py": ["add"],
        },
    }
    assert symbols_mapped_for_source(test_map, "cli/main.py") == frozenset({"Widget::run", "Widget::stop"})
    assert symbols_mapped_for_source(test_map, "missing.py") == frozenset()


def test_build_test_map_index_matches_scan_helpers() -> None:
    test_map = {
        "tests/regression/cli/test_run.py::test_run": {
            "cli/main.py": ["Widget::run", "Widget::stop"],
            "tensor_cast/ops.py": ["add"],
        },
        "tests/regression/tensor_cast/test_ops.py::test_add": {
            "tensor_cast/ops.py": ["add"],
        },
    }
    index = build_test_map_index(test_map)
    assert symbol_watchers(test_map, "cli/main.py", "Widget::run", index=index) == symbol_watchers(
        test_map,
        "cli/main.py",
        "Widget::run",
    )
    assert source_watchers(test_map, "tensor_cast/ops.py", index=index) == source_watchers(
        test_map,
        "tensor_cast/ops.py",
    )


def test_prune_deleted_sources_drops_deleted_paths() -> None:
    test_map = {
        "tests/a.py::test_x": {
            "keep.py": ["fn"],
            "drop.py": ["fn"],
        },
    }
    pruned = prune_deleted_sources(test_map, ("drop.py",))
    assert pruned == {"tests/a.py::test_x": {"keep.py": ["fn"]}}
