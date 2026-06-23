"""Tests for common.ast_utils."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
from scripts.helpers.common.ast_utils import (
    MODULE_SYMBOL,
    CoverageChecks,
    DefinitionSpan,
    ShadowWarning,
    assert_canonical_symbol,
    canonical_symbol_for_line,
    canonical_symbol_for_path_line,
    canonical_symbols_for_lines,
    collect_file_symbols,
    collect_shadow_warnings,
    coverage_checks_for_definition,
    executable_lines_for_canonical_symbol,
    filter_executable_lines,
    gated_coverage_symbols,
    import_symbol_for_definition,
    iter_canonical_definition_spans,
    iter_qualified_definition_spans,
    symbol_for_line,
    symbols_for_lines,
    top_level_definitions,
    touched_definition_symbols,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spans(sample_py_file: Path) -> list[DefinitionSpan]:
    return iter_canonical_definition_spans(sample_py_file)


@pytest.fixture(scope="module")
def legacy_spans(sample_py_file: Path) -> list[DefinitionSpan]:
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
# canonical spans
# ---------------------------------------------------------------------------


def test_canonical_spans_use_class_double_colon(spans: list[DefinitionSpan]) -> None:
    qualified = {span.qualified_name for span in spans}
    assert "foo" in qualified
    assert "baz" in qualified
    assert "Bar::method" in qualified


def test_qualified_definition_spans_use_canonical_names(
    legacy_spans: list[DefinitionSpan],
) -> None:
    qualified = {span.qualified_name for span in legacy_spans}
    assert "Bar::method" in qualified


def test_assert_canonical_symbol_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        assert_canonical_symbol("")


def test_assert_canonical_symbol_accepts_canonical_method() -> None:
    assert_canonical_symbol("Bar::method")
    assert_canonical_symbol("Bar.method")
    assert_canonical_symbol("run")
    assert_canonical_symbol(MODULE_SYMBOL)


def test_canonical_symbol_for_line_module_fallback(sample_py_file: Path) -> None:
    symbols = collect_file_symbols(sample_py_file)
    assert canonical_symbol_for_line(symbols, 1) == MODULE_SYMBOL


def test_canonical_symbol_for_line_class_body_fallback(sample_py_file: Path) -> None:
    tree = ast.parse(sample_py_file.read_text(encoding="utf-8"))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    ann_line = next(
        node.lineno
        for node in ast.walk(class_node)
        if isinstance(node, ast.AnnAssign)
        and node.value is None
        and isinstance(node.target, ast.Name)
        and node.target.id == "CLASS_VAR"
    )
    symbols = collect_file_symbols(sample_py_file)
    assert canonical_symbol_for_line(symbols, ann_line) == "Bar::%"


def test_canonical_symbol_for_path_line_inside_function(sample_py_file: Path) -> None:
    tree = ast.parse(sample_py_file.read_text(encoding="utf-8"))
    exec_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "x" for t in node.targets)
    )
    assert canonical_symbol_for_path_line(sample_py_file, exec_line) == "foo"


def test_different_mangled_underscore_defs_both_retained(tmp_path: Path) -> None:
    path = tmp_path / "dup.py"
    path.write_text(
        "\n".join(
            [
                "def deco(op):",
                "    def wrap(fn):",
                "        return fn",
                "    return wrap",
                "",
                "@deco('a')",
                "def _():",
                "    return 1",
                "",
                "@deco('b')",
                "def _():",
                "    return 2",
            ]
        ),
        encoding="utf-8",
    )
    names = {span.qualified_name for span in iter_canonical_definition_spans(path)}
    underscore_names = {name for name in names if name.startswith("_@")}
    assert underscore_names == {'_@deco("a")', '_@deco("b")'}


def test_underscore_def_with_decorator_uses_suffix_without_quotes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "decorated.py"
    path.write_text(
        "\n".join(
            [
                "def deco(op):",
                "    def wrap(fn):",
                "        return fn",
                "    return wrap",
                "",
                "@deco('a')",
                "def _():",
                "    return 1",
            ]
        ),
        encoding="utf-8",
    )
    names = {span.qualified_name for span in iter_canonical_definition_spans(path)}
    underscore_names = {name for name in names if name.startswith("_")}
    assert underscore_names == {'_@deco("a")'}


def test_underscore_def_without_decorator_uses_bare_name(tmp_path: Path) -> None:
    path = tmp_path / "bare.py"
    path.write_text("def _():\n    return 1\n", encoding="utf-8")
    names = {span.qualified_name for span in iter_canonical_definition_spans(path)}
    assert names == {"_"}


def test_duplicate_defs_last_wins_for_canonical_symbol(tmp_path: Path) -> None:
    path = tmp_path / "dup_foo.py"
    path.write_text(
        "\n".join(
            [
                "def foo():",
                "    return 1",
                "",
                "def foo():",
                "    return 2",
            ]
        ),
        encoding="utf-8",
    )
    spans = iter_canonical_definition_spans(path)
    assert len(spans) == 1
    assert spans[0].qualified_name == "foo"
    symbols = collect_file_symbols(path)
    assert canonical_symbol_for_line(symbols, spans[0].start_line) == "foo"


def test_collect_shadow_warnings_reports_non_underscore_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shadow.py"
    path.write_text(
        "\n".join(
            [
                "def foo():",
                "    return 1",
                "",
                "def foo():",
                "    return 2",
            ]
        ),
        encoding="utf-8",
    )
    warnings = collect_shadow_warnings(path)
    assert warnings == (ShadowWarning(str(path), 1, "foo", 4),)


def test_collect_shadow_warnings_reports_underscore_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "shadow_underscore.py"
    path.write_text(
        "\n".join(
            [
                "def _():",
                "    return 1",
                "",
                "def _():",
                "    return 2",
            ]
        ),
        encoding="utf-8",
    )
    warnings = collect_shadow_warnings(path)
    assert warnings == (ShadowWarning(str(path), 1, "_", 4),)


def test_collect_shadow_warnings_different_mangled_no_warning(tmp_path: Path) -> None:
    path = tmp_path / "shadow_mangled.py"
    path.write_text(
        "\n".join(
            [
                "def deco(op):",
                "    def wrap(fn):",
                "        return fn",
                "    return wrap",
                "",
                "@deco('a')",
                "def _():",
                "    return 1",
                "",
                "@deco('b')",
                "def _():",
                "    return 2",
            ]
        ),
        encoding="utf-8",
    )
    assert collect_shadow_warnings(path) == ()


def test_collect_shadow_warnings_same_mangled_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "shadow_same_mangled.py"
    path.write_text(
        "\n".join(
            [
                "def deco(op):",
                "    def wrap(fn):",
                "        return fn",
                "    return wrap",
                "",
                "@deco('a')",
                "def _():",
                "    return 1",
                "",
                "@deco('a')",
                "def _():",
                "    return 2",
            ]
        ),
        encoding="utf-8",
    )
    warnings = collect_shadow_warnings(path)
    assert warnings == (ShadowWarning(str(path), 7, '_@deco("a")', 11),)


# ---------------------------------------------------------------------------
# symbol_for_line (canonical)
# ---------------------------------------------------------------------------


def test_symbol_for_line_inside_function_returns_name(
    legacy_spans: list[DefinitionSpan],
) -> None:
    foo = next(span for span in legacy_spans if span.qualified_name == "foo")
    mid = (foo.start_line + foo.end_line) // 2
    assert symbol_for_line(legacy_spans, mid) == "foo"


def test_symbol_for_line_module_docstring_returns_none(
    legacy_spans: list[DefinitionSpan],
) -> None:
    assert symbol_for_line(legacy_spans, 1) is None


def test_symbol_for_line_method_body_returns_qualified_name(
    legacy_spans: list[DefinitionSpan],
) -> None:
    method = next(span for span in legacy_spans if span.qualified_name == "Bar::method")
    assert symbol_for_line(legacy_spans, method.start_line + 1) == "Bar::method"


def test_symbols_for_lines_returns_canonical_method_name(sample_py_file: Path) -> None:
    spans = iter_qualified_definition_spans(sample_py_file)
    method = next(span for span in spans if span.qualified_name == "Bar::method")
    assert symbols_for_lines(sample_py_file, {method.start_line + 1}) == {"Bar::method"}


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


def test_canonical_symbols_for_lines_returns_canonical_names(
    sample_py_file: Path,
) -> None:
    tree = ast.parse(sample_py_file.read_text(encoding="utf-8"))
    exec_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "x" for t in node.targets)
    )
    assert canonical_symbols_for_lines(sample_py_file, {exec_line}) == {"foo"}


def test_gated_coverage_symbols_requires_each_top_level_function(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mod.py"
    path.write_text(
        "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n",
        encoding="utf-8",
    )
    assert gated_coverage_symbols(path) == frozenset({"alpha", "beta"})


def test_gated_coverage_symbols_skips_protocol_ellipsis_stubs(tmp_path: Path) -> None:
    path = tmp_path / "protocol_mod.py"
    path.write_text(
        "from typing import Protocol\n\n"
        "class Reader(Protocol):\n"
        "    def read(self) -> None: ...\n"
        "    def measured_files(self) -> list[str]: ...\n\n"
        "def real_fn() -> int:\n"
        "    return 1\n",
        encoding="utf-8",
    )
    symbols = gated_coverage_symbols(path)
    assert "Reader::read" not in symbols
    assert "Reader::measured_files" not in symbols
    assert "real_fn" in symbols


def test_canonical_symbols_for_lines_skips_protocol_ellipsis_stubs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "protocol_mod.py"
    path.write_text(
        "from typing import Protocol\n\n"
        "class Reader(Protocol):\n"
        "    def read(self) -> None: ...\n\n"
        "def real_fn() -> int:\n"
        "    return 1\n",
        encoding="utf-8",
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    real_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    assert canonical_symbols_for_lines(path, {real_fn.lineno}) == {"real_fn"}
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    stub_line = class_node.body[0].lineno
    assert canonical_symbols_for_lines(path, {stub_line}) == {f"Reader::{MODULE_SYMBOL}"}


def test_executable_lines_for_canonical_symbol_returns_function_body(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mod.py"
    path.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    lines = executable_lines_for_canonical_symbol(path, "alpha")
    assert 1 in lines
    assert 2 in lines


# ---------------------------------------------------------------------------
# AST cache
# ---------------------------------------------------------------------------


def test_cache_reuses_parse_result(sample_py_file: Path) -> None:
    spans1 = iter_canonical_definition_spans(sample_py_file)
    spans2 = iter_canonical_definition_spans(sample_py_file)
    assert len(spans1) == len(spans2)


# ---------------------------------------------------------------------------
# Universal decorator mangling
# ---------------------------------------------------------------------------


def test_decorator_mangling_uses_double_quoted_string_literals(tmp_path: Path) -> None:
    path = tmp_path / "quoted.py"
    path.write_text(
        "\n".join(
            [
                "def deco(op):",
                "    def wrap(fn):",
                "        return fn",
                "    return wrap",
                "",
                "@deco('a')",
                "def _():",
                "    return 1",
            ]
        ),
        encoding="utf-8",
    )
    names = {span.qualified_name for span in iter_canonical_definition_spans(path)}
    underscore_names = {name for name in names if name.startswith("_@")}
    assert underscore_names == {'_@deco("a")'}


def test_decorator_line_maps_to_module_symbol(tmp_path: Path) -> None:
    path = tmp_path / "deco_line.py"
    path.write_text(
        "\n".join(
            [
                "def deco(x):",
                "    def wrap(fn):",
                "        return fn",
                "    return wrap",
                "",
                "@deco('tag')",
                "def run():",
                "  return 1",
            ]
        ),
        encoding="utf-8",
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    run_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    deco_line = run_node.decorator_list[0].lineno
    symbols = collect_file_symbols(path)
    assert canonical_symbol_for_line(symbols, deco_line) == MODULE_SYMBOL


def test_class_decorator_line_maps_to_class_body_symbol(tmp_path: Path) -> None:
    path = tmp_path / "class_deco.py"
    path.write_text(
        "\n".join(
            [
                "@dataclass",
                "class Foo:",
                "    x: int = 1",
                "",
                "from dataclasses import dataclass",
            ]
        ),
        encoding="utf-8",
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    deco_line = class_node.decorator_list[0].lineno
    body_line = class_node.body[0].lineno
    symbols = collect_file_symbols(path)
    assert canonical_symbol_for_line(symbols, deco_line) == "Foo::%"
    assert canonical_symbol_for_line(symbols, body_line) == "Foo::%"


def test_class_method_symbol_excludes_class_decorator(tmp_path: Path) -> None:
    path = tmp_path / "method.py"
    path.write_text(
        "\n".join(
            [
                "from dataclasses import dataclass",
                "",
                "@dataclass(frozen=True)",
                "class Foo:",
                "    @staticmethod",
                "    def create() -> 'Foo':",
                "        return Foo()",
            ]
        ),
        encoding="utf-8",
    )
    names = {span.qualified_name for span in iter_canonical_definition_spans(path)}
    assert "Foo@dataclass(frozen=True)" not in names
    assert "Foo::create@staticmethod" in names
    assert "Foo@dataclass(frozen=True)::create" not in names


def test_import_symbol_for_definition_class_method(tmp_path: Path) -> None:
    path = tmp_path / "method_import.py"
    path.write_text(
        "\n".join(
            [
                "class Foo:",
                "    @staticmethod",
                "    def run():",
                "        return 1",
            ]
        ),
        encoding="utf-8",
    )
    assert import_symbol_for_definition(path, "Foo::run@staticmethod") == "Foo::%"


def test_coverage_checks_decorator_only_uses_import_and_proxy(tmp_path: Path) -> None:
    path = tmp_path / "deco_checks.py"
    path.write_text(
        "\n".join(
            [
                "def deco(fn):",
                "    return fn",
                "",
                "@deco",
                "def run():",
                "    return 1",
            ]
        ),
        encoding="utf-8",
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    deco_line = fn.decorator_list[0].lineno
    body_line = fn.body[0].lineno
    checks = coverage_checks_for_definition(path, "run@deco", {deco_line})
    assert checks == CoverageChecks(frozenset({deco_line}), frozenset(), frozenset({body_line}))
    assert import_symbol_for_definition(path, "run@deco") == MODULE_SYMBOL


def test_coverage_checks_def_header_only_uses_proxy(tmp_path: Path) -> None:
    path = tmp_path / "sig_checks.py"
    path.write_text("def run(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    body_line = fn.body[0].lineno
    checks = coverage_checks_for_definition(path, "run", {fn.lineno})
    assert checks == CoverageChecks(frozenset(), frozenset(), frozenset({body_line}))


def test_coverage_checks_body_only_uses_strict(tmp_path: Path) -> None:
    path = tmp_path / "body_checks.py"
    path.write_text("def run():\n    return 1\n", encoding="utf-8")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    body_line = fn.body[0].lineno
    checks = coverage_checks_for_definition(path, "run", {body_line})
    assert checks == CoverageChecks(frozenset(), frozenset({body_line}), frozenset())


def test_coverage_checks_decorator_and_body_skips_proxy(tmp_path: Path) -> None:
    path = tmp_path / "deco_body_checks.py"
    path.write_text(
        "\n".join(
            [
                "def deco(fn):",
                "    return fn",
                "",
                "@deco",
                "def run():",
                "    return 1",
            ]
        ),
        encoding="utf-8",
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    deco_line = fn.decorator_list[0].lineno
    body_line = fn.body[0].lineno
    checks = coverage_checks_for_definition(path, "run@deco", {deco_line, body_line})
    assert checks == CoverageChecks(frozenset({deco_line}), frozenset({body_line}), frozenset())


def test_touched_definition_symbols_includes_mangled_span(tmp_path: Path) -> None:
    path = tmp_path / "touched.py"
    path.write_text(
        "\n".join(
            [
                "def deco(fn):",
                "    return fn",
                "",
                "@deco",
                "def run():",
                "    return 1",
            ]
        ),
        encoding="utf-8",
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    assert touched_definition_symbols(path, {fn.decorator_list[0].lineno}) == frozenset({"run@deco"})


def test_coverage_checks_class_decorator_only(tmp_path: Path) -> None:
    path = tmp_path / "class_deco_checks.py"
    path.write_text(
        "\n".join(
            [
                "from dataclasses import dataclass",
                "",
                "@dataclass",
                "class Foo:",
                "    x: int = 1",
            ]
        ),
        encoding="utf-8",
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    deco_line = class_node.decorator_list[0].lineno
    body_line = class_node.body[0].lineno
    checks = coverage_checks_for_definition(path, "Foo::%", {deco_line})
    assert checks.import_lines == frozenset({deco_line})
    assert checks.strict_lines == frozenset()
    assert body_line in checks.proxy_lines
    assert import_symbol_for_definition(path, "Foo::%") == "Foo::%"


def test_gated_coverage_symbols_excludes_class_decorator_mangled_key(
    tmp_path: Path,
) -> None:
    from pathlib import Path as PathCls

    path = PathCls(__file__).resolve().parents[4] / "scripts/helpers/ci_gate/comments.py"
    required = gated_coverage_symbols(path)
    assert "GitCodeCommentConfig@dataclass(frozen=True, slots=True)" not in required
