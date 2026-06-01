"""Tests for common.ast_utils."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from scripts.helpers.common.ast_utils import (
    DefinitionSpan,
    filter_executable_lines,
    iter_qualified_definition_spans,
    symbol_for_line,
    symbols_for_lines,
    top_level_definitions,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spans(sample_py_file: Path) -> list[DefinitionSpan]:
    return iter_qualified_definition_spans(sample_py_file)


# ---------------------------------------------------------------------------
# top_level_definitions
# ---------------------------------------------------------------------------


def test_top_level_definitions_includes_functions_and_classes(
    sample_py_file: Path,
) -> None:
    names = top_level_definitions(sample_py_file)
    assert "foo" in names
    assert "Bar" in names
    assert "baz" in names


# ---------------------------------------------------------------------------
# iter_qualified_definition_spans
# ---------------------------------------------------------------------------


def test_spans_includes_top_level_and_methods(spans: list[DefinitionSpan]) -> None:
    qualified = {s.qualified_name for s in spans}
    assert "foo" in qualified
    assert "baz" in qualified
    assert "Bar.method" in qualified


def test_spans_line_ranges_are_valid(spans: list[DefinitionSpan]) -> None:
    foo = next(s for s in spans if s.qualified_name == "foo")
    assert foo.start_line <= foo.end_line


# ---------------------------------------------------------------------------
# symbol_for_line
# ---------------------------------------------------------------------------


def test_symbol_for_line_inside_function_returns_name(
    spans: list[DefinitionSpan],
) -> None:
    foo = next(s for s in spans if s.qualified_name == "foo")
    mid = (foo.start_line + foo.end_line) // 2
    assert symbol_for_line(spans, mid) == "foo"


def test_symbol_for_line_module_docstring_returns_none(
    spans: list[DefinitionSpan],
) -> None:
    assert symbol_for_line(spans, 1) is None


def test_symbol_for_line_method_body_returns_qualified_name(
    spans: list[DefinitionSpan],
) -> None:
    method = next(s for s in spans if s.qualified_name == "Bar.method")
    assert symbol_for_line(spans, method.start_line + 1) == "Bar.method"


# ---------------------------------------------------------------------------
# filter_executable_lines
# ---------------------------------------------------------------------------


def test_filter_docstring_line_excluded(sample_py_file: Path) -> None:
    tree = ast.parse(sample_py_file.read_text(encoding="utf-8"))
    doc_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and node.value.value == "Module docstring."
    )
    result = filter_executable_lines(sample_py_file, {doc_line})
    assert result == set()


def test_filter_all_assignment_line_excluded(sample_py_file: Path) -> None:
    tree = ast.parse(sample_py_file.read_text(encoding="utf-8"))
    all_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
    )
    result = filter_executable_lines(sample_py_file, {all_line})
    assert result == set()


def test_filter_type_only_annotation_line_excluded(sample_py_file: Path) -> None:
    tree = ast.parse(sample_py_file.read_text(encoding="utf-8"))
    ann_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and node.value is None
        and isinstance(node.target, ast.Name)
        and node.target.id == "CLASS_VAR"
    )
    result = filter_executable_lines(sample_py_file, {ann_line})
    assert result == set()


def test_filter_executable_line_kept(sample_py_file: Path) -> None:
    tree = ast.parse(sample_py_file.read_text(encoding="utf-8"))
    exec_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "x" for t in node.targets)
    )
    result = filter_executable_lines(sample_py_file, {exec_line})
    assert result == {exec_line}


def test_filter_empty_changed_lines_returns_empty(sample_py_file: Path) -> None:
    assert filter_executable_lines(sample_py_file, set()) == set()


def test_filter_syntax_error_file_falls_back_to_text_filter(tmp_path: Path) -> None:
    path = tmp_path / "broken.py"
    path.write_text("def foo(\n  pass\n", encoding="utf-8")
    # Line 1 "def foo(" — non-empty, non-comment → executable
    # Line 2 "  pass" — non-empty, non-comment → executable
    result = filter_executable_lines(path, {1, 2})
    assert result == {1, 2}


# ---------------------------------------------------------------------------
# symbols_for_lines
# ---------------------------------------------------------------------------


def test_symbols_for_lines_out_of_range_returns_empty(sample_py_file: Path) -> None:
    syms = symbols_for_lines(sample_py_file, {1000})
    assert syms == set()


def test_symbols_for_lines_empty_input_returns_empty(sample_py_file: Path) -> None:
    assert symbols_for_lines(sample_py_file, set()) == set()


# ---------------------------------------------------------------------------
# AST cache
# ---------------------------------------------------------------------------


def test_cache_reuses_parse_result(sample_py_file: Path) -> None:
    spans1 = iter_qualified_definition_spans(sample_py_file)
    spans2 = iter_qualified_definition_spans(sample_py_file)
    assert len(spans1) == len(spans2)
