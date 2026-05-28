"""AST helpers for mapping tests to source symbols and filtering executable lines."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

_NON_EXECUTABLE_ASSIGN_NAMES: Final = frozenset({"__all__", "__version__"})


@dataclass(frozen=True, slots=True)
class DefinitionSpan:
    qualified_name: str
    start_line: int
    end_line: int


# ---------------------------------------------------------------------------
# AST parse cache — keyed by (path, mtime_ns) to avoid stale cache on change
# ---------------------------------------------------------------------------


@lru_cache(maxsize=128)
def _parse_cached(path_str: str, mtime_ns: int) -> ast.AST:
    return ast.parse(Path(path_str).read_text(encoding="utf-8"), filename=path_str)


def _get_cached_tree(path: Path) -> ast.AST:
    """Parse file to AST with caching by path + mtime."""
    return _parse_cached(str(path), path.stat().st_mtime_ns)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _end_line(node: ast.AST) -> int:
    return getattr(node, "end_lineno", node.lineno)


def top_level_definitions(path: Path) -> list[str]:
    """Return names of top-level functions, async functions, and classes."""
    tree = _get_cached_tree(path)
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or isinstance(node, ast.ClassDef):
            out.append(node.name)
    return out


def iter_qualified_definition_spans(path: Path) -> list[DefinitionSpan]:
    """Return ``DefinitionSpan`` for top-level defs and class methods."""
    try:
        tree = _get_cached_tree(path)
    except (OSError, SyntaxError):
        return []

    spans: list[DefinitionSpan] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append(DefinitionSpan(node.name, node.lineno, _end_line(node)))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    spans.append(
                        DefinitionSpan(
                            f"{node.name}.{item.name}",
                            item.lineno,
                            _end_line(item),
                        )
                    )
    return spans


def symbol_for_line(spans: list[DefinitionSpan], line: int) -> str | None:
    """Pick the smallest enclosing span (innermost definition)."""
    containing = [(s, s.end_line - s.start_line + 1) for s in spans if s.start_line <= line <= s.end_line]
    if not containing:
        return None
    span, _ = min(containing, key=lambda item: item[1])
    return span.qualified_name


# ---------------------------------------------------------------------------
# Executable line filtering (single AST walk)
# ---------------------------------------------------------------------------


def _collect_non_executable_lines(tree: ast.AST) -> set[int]:
    """Single walk: collect docstring lines + non-executable statement lines."""
    lines: set[int] = set()

    for node in ast.walk(tree):
        # Docstring: first statement of function/class/module body that is a string expr
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    start = node.body[0].lineno
                    end = _end_line(node.body[0])
                    lines.update(range(start, end + 1))

        # Type-only annotation (no value): AnnAssign with value=None
        if isinstance(node, ast.AnnAssign) and node.value is None:
            start = node.lineno
            end = _end_line(node)
            lines.update(range(start, end + 1))

        # Non-executable assigns: __all__, __version__
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in _NON_EXECUTABLE_ASSIGN_NAMES:
                    start = node.lineno
                    end = _end_line(node)
                    lines.update(range(start, end + 1))

    return lines


def filter_executable_lines(path: Path, changed_lines: set[int] | frozenset[int]) -> set[int]:
    """Drop comment-only, docstring, type-only, and __all__/__version__ diff lines."""
    if not changed_lines:
        return set()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    source_lines = text.splitlines()
    try:
        tree = _get_cached_tree(path)
    except SyntaxError:
        return {
            line_no
            for line_no in changed_lines
            if 1 <= line_no <= len(source_lines) and _line_text_is_executable(source_lines[line_no - 1])
        }

    skip = _collect_non_executable_lines(tree)
    executable: set[int] = set()
    for line_no in changed_lines:
        if line_no in skip:
            continue
        if line_no < 1 or line_no > len(source_lines):
            continue
        if _line_text_is_executable(source_lines[line_no - 1]):
            executable.add(line_no)
    return executable


def _line_text_is_executable(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def symbols_for_lines(path: Path, line_numbers: set[int]) -> set[str]:
    """Return symbols that enclose the given line numbers."""
    if not line_numbers:
        return set()
    spans = iter_qualified_definition_spans(path)
    symbols: set[str] = set()
    for line_no in line_numbers:
        symbol = symbol_for_line(spans, line_no)
        if symbol is not None:
            symbols.add(symbol)
    return symbols
