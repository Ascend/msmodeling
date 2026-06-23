"""AST helpers for mapping tests to source symbols and filtering executable lines."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

_NON_EXECUTABLE_ASSIGN_NAMES: Final = frozenset({"__all__", "__version__"})
MODULE_SYMBOL: Final = "%"


@dataclass(frozen=True, slots=True)
class DefinitionSpan:
    qualified_name: str
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class FileSymbols:
    definitions: tuple[DefinitionSpan, ...]
    class_spans: tuple[DefinitionSpan, ...]
    decorator_lines: frozenset[int]


@dataclass(frozen=True, slots=True)
class CoverageChecks:
    import_lines: frozenset[int]
    strict_lines: frozenset[int]
    proxy_lines: frozenset[int]


@dataclass(frozen=True, slots=True)
class ShadowWarning:
    file: str
    line: int
    name: str
    shadowed_by_line: int


# ---------------------------------------------------------------------------
# AST parse cache — keyed by (path, mtime_ns) to avoid stale cache on change
# ---------------------------------------------------------------------------


@lru_cache(maxsize=128)
def _parse_cached(path_str: str, mtime_ns: int) -> ast.Module:
    return ast.parse(Path(path_str).read_text(encoding="utf-8"), filename=path_str)


def _get_cached_tree(path: Path) -> ast.Module:
    """Parse file to AST with caching by path + mtime."""
    return _parse_cached(str(path), path.stat().st_mtime_ns)


# ---------------------------------------------------------------------------
# Canonical symbol naming
# ---------------------------------------------------------------------------


def _end_line(node: ast.AST) -> int:
    end_lineno = getattr(node, "end_lineno", None)
    if isinstance(end_lineno, int):
        return end_lineno
    lineno = getattr(node, "lineno", None)
    if isinstance(lineno, int):
        return lineno
    return 0


def _stable_decorator_unparse(node: ast.AST) -> str:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return json.dumps(node.value)
        return ast.unparse(node)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_stable_decorator_unparse(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        func = _stable_decorator_unparse(node.func)
        arg_parts = [_stable_decorator_unparse(arg) for arg in node.args]
        arg_parts.extend(
            f"{keyword.arg}={_stable_decorator_unparse(keyword.value)}"
            for keyword in node.keywords
            if keyword.arg is not None
        )
        arg_parts.extend(
            f"**{_stable_decorator_unparse(keyword.value)}" for keyword in node.keywords if keyword.arg is None
        )
        return f"{func}({', '.join(arg_parts)})"
    if isinstance(node, ast.Subscript):
        return f"{_stable_decorator_unparse(node.value)}[{_stable_decorator_unparse(node.slice)}]"
    if isinstance(node, ast.Tuple):
        inner = ", ".join(_stable_decorator_unparse(elt) for elt in node.elts)
        return f"({inner},)" if len(node.elts) == 1 else f"({inner})"
    if isinstance(node, ast.List):
        inner = ", ".join(_stable_decorator_unparse(elt) for elt in node.elts)
        return f"[{inner}]"
    return ast.unparse(node)


def _mangle_decorator_suffix(decorator_list: list[ast.expr]) -> str:
    return "@".join(_stable_decorator_unparse(dec) for dec in decorator_list)


def _collect_decorator_lines(nodes: list[ast.stmt]) -> set[int]:
    lines: set[int] = set()
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lines.update(_decorator_line_numbers(node))
        if isinstance(node, ast.ClassDef):
            lines.update(_collect_decorator_lines(list(node.body)))
    return lines


def _definition_start_line(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> int:
    if node.decorator_list:
        return min(dec.lineno for dec in node.decorator_list)
    return node.lineno


def _decorator_line_numbers(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> set[int]:
    lines: set[int] = set()
    for dec in node.decorator_list:
        lines.update(range(dec.lineno, _end_line(dec) + 1))
    return lines


def _is_protocol_ellipsis_stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when the function body is a single ``...`` (typing.Protocol stub)."""
    if len(node.body) != 1:
        return False
    stmt = node.body[0]
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis


def _canonical_function_name(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    class_name: str | None,
) -> str:
    name = node.name
    local = f"{name}@{_mangle_decorator_suffix(node.decorator_list)}" if node.decorator_list else name
    if class_name is None:
        return local
    return f"{class_name}::{local}"


def _canonical_class_decorator_symbol(node: ast.ClassDef) -> str | None:
    """Mangled decorator identity for a class; used only for ``_find_definition_node`` lookup.

    Not a ``test_map`` or gate key — class decorator lines gate via ``{name}::%``.
    """
    if not node.decorator_list:
        return None
    return f"{node.name}@{_mangle_decorator_suffix(node.decorator_list)}"


def _effective_definition_spans(
    nodes: list[ast.stmt],
    *,
    class_name: str | None,
) -> tuple[list[DefinitionSpan], list[tuple[int, str, int]]]:
    """Last-wins per mangled qualified name within one module or class body.

    Each definition is keyed by its mangled qualified name (e.g. ``foo``,
    ``_@deco("a")``, ``Bar::run@staticmethod``). Identical mangled collisions
    overwrite earlier spans and emit shadow warnings; different mangled symbols
    for the same bare name (e.g. ``_@deco("a")`` vs ``_@deco("b")``) coexist.
    """
    last_by_qualified: dict[str, tuple[ast.FunctionDef | ast.AsyncFunctionDef, DefinitionSpan]] = {}
    shadows: list[tuple[int, str, int]] = []

    for node in nodes:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _is_protocol_ellipsis_stub(node):
            continue
        qualified = _canonical_function_name(node, class_name=class_name)
        span = DefinitionSpan(
            qualified,
            _definition_start_line(node),
            _end_line(node),
        )
        if qualified in last_by_qualified:
            prev_node, _ = last_by_qualified[qualified]
            shadows.append((prev_node.lineno, qualified, node.lineno))
        last_by_qualified[qualified] = (node, span)

    return [span for _, span in last_by_qualified.values()], shadows


def collect_file_symbols(path: Path) -> FileSymbols:
    """Return canonical definition and class spans for *path*."""
    try:
        tree = _get_cached_tree(path)
    except (OSError, SyntaxError):
        return FileSymbols((), (), frozenset())

    definitions, _ = _effective_definition_spans(list(tree.body), class_name=None)
    class_spans: list[DefinitionSpan] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_spans.append(DefinitionSpan(node.name, _definition_start_line(node), _end_line(node)))
            # Class-level decorators gate via Foo::%; do not add Class@decorator spans.
            method_spans, _ = _effective_definition_spans(list(node.body), class_name=node.name)
            definitions.extend(method_spans)
    decorator_lines = frozenset(_collect_decorator_lines(list(tree.body)))
    return FileSymbols(tuple(definitions), tuple(class_spans), decorator_lines)


def collect_shadow_warnings(path: Path) -> tuple[ShadowWarning, ...]:
    """Return shadow warnings for identical mangled qualified name collisions.

    Intentionally re-walks module/class bodies via ``_effective_definition_spans``
    (AST parse is cached in ``_get_cached_tree``; symbols path is separate).
    """
    try:
        tree = _get_cached_tree(path)
    except (OSError, SyntaxError):
        return ()

    file_str = str(path)
    warnings: list[ShadowWarning] = []
    _, module_shadows = _effective_definition_spans(list(tree.body), class_name=None)
    for line, qualified, shadowed_by_line in module_shadows:
        warnings.append(ShadowWarning(file_str, line, qualified, shadowed_by_line))

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            _, class_shadows = _effective_definition_spans(list(node.body), class_name=node.name)
            for line, qualified, shadowed_by_line in class_shadows:
                warnings.append(ShadowWarning(file_str, line, qualified, shadowed_by_line))
    return tuple(warnings)


def canonical_symbol_for_line(symbols: FileSymbols, line: int) -> str:
    """Map a source line to its canonical symbol, including module/class fallbacks."""
    if line in symbols.decorator_lines:
        for cls in symbols.class_spans:
            if cls.start_line <= line <= cls.end_line:
                return f"{cls.qualified_name}::{MODULE_SYMBOL}"
        return MODULE_SYMBOL
    containing = [
        (span, span.end_line - span.start_line + 1)
        for span in symbols.definitions
        if span.start_line <= line <= span.end_line
    ]
    if containing:
        span, _ = min(containing, key=lambda item: item[1])
        return span.qualified_name
    for cls in symbols.class_spans:
        if cls.start_line <= line <= cls.end_line:
            return f"{cls.qualified_name}::{MODULE_SYMBOL}"
    return MODULE_SYMBOL


def assert_canonical_symbol(symbol: str) -> None:
    """Raise ValueError when *symbol* is empty."""
    if not symbol:
        raise ValueError(f"symbol must be non-empty, got {symbol!r}")


def _definition_executable_lines(
    file_symbols: FileSymbols,
    executable: set[int],
    canonical_symbol: str,
) -> set[int] | None:
    for span in file_symbols.definitions:
        if canonical_symbol != span.qualified_name:
            continue
        span_lines = set(range(span.start_line, span.end_line + 1))
        return span_lines & executable
    return None


def _class_body_executable_lines(
    file_symbols: FileSymbols,
    executable: set[int],
    canonical_symbol: str,
) -> set[int] | None:
    class_suffix = f"::{MODULE_SYMBOL}"
    if not canonical_symbol.endswith(class_suffix):
        return None
    class_name = canonical_symbol[: -len(class_suffix)]
    covered_by_methods: set[int] = set()
    for span in file_symbols.definitions:
        if span.qualified_name.startswith(f"{class_name}::"):
            covered_by_methods |= set(range(span.start_line, span.end_line + 1))
    for cls in file_symbols.class_spans:
        if cls.qualified_name != class_name:
            continue
        class_lines = set(range(cls.start_line, cls.end_line + 1))
        return class_lines & executable - covered_by_methods
    return set()


def _module_executable_lines(file_symbols: FileSymbols, executable: set[int]) -> set[int]:
    covered: set[int] = set()
    for span in file_symbols.definitions:
        covered |= set(range(span.start_line, span.end_line + 1)) & executable
    for cls in file_symbols.class_spans:
        class_lines = set(range(cls.start_line, cls.end_line + 1))
        covered |= class_lines & executable
    return executable - covered


def _read_source_lines(path: Path) -> list[str] | None:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None


def _filter_executable_source_lines(
    path: Path,
    source_lines: list[str],
    changed_lines: set[int] | frozenset[int],
) -> set[int]:
    if not changed_lines:
        return set()
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


def _all_executable_lines(path: Path, source_lines: list[str]) -> set[int]:
    if not source_lines:
        return set()
    return _filter_executable_source_lines(path, source_lines, set(range(1, len(source_lines) + 1)))


def executable_lines_for_canonical_symbol(path: Path, canonical_symbol: str) -> set[int]:
    """Return executable source lines attributed to one canonical gated symbol."""
    source_lines = _read_source_lines(path)
    if not source_lines:
        return set()

    executable = _all_executable_lines(path, source_lines)
    if not executable:
        return set()

    file_symbols = collect_file_symbols(path)
    definition_lines = _definition_executable_lines(file_symbols, executable, canonical_symbol)
    if definition_lines is not None:
        return definition_lines

    class_lines = _class_body_executable_lines(file_symbols, executable, canonical_symbol)
    if class_lines is not None:
        return class_lines

    if canonical_symbol == MODULE_SYMBOL:
        return _module_executable_lines(file_symbols, executable)

    return set()


def _find_class_node(tree: ast.Module, class_name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _find_definition_node(
    path: Path,
    symbol: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | None:
    try:
        tree = _get_cached_tree(path)
    except (OSError, SyntaxError):
        return None

    class_suffix = f"::{MODULE_SYMBOL}"
    if symbol.endswith(class_suffix):
        return _find_class_node(tree, symbol[: -len(class_suffix)])

    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _canonical_function_name(node, class_name=None) == symbol
        ):
            return node
        if isinstance(node, ast.ClassDef):
            deco_symbol = _canonical_class_decorator_symbol(node)
            if deco_symbol == symbol:
                return node
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not _is_protocol_ellipsis_stub(item)
                    and _canonical_function_name(item, class_name=node.name) == symbol
                ):
                    return item
    return None


def _body_executable_lines(path: Path, symbol: str) -> set[int]:
    all_lines = executable_lines_for_canonical_symbol(path, symbol)
    node = _find_definition_node(path, symbol)
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return all_lines
    header = {node.lineno}
    return all_lines - header - _decorator_line_numbers(node)


def _proxy_body_lines(path: Path, symbol: str) -> set[int]:
    node = _find_definition_node(path, symbol)
    if isinstance(node, ast.ClassDef):
        return executable_lines_for_canonical_symbol(path, f"{node.name}::{MODULE_SYMBOL}")
    return _body_executable_lines(path, symbol)


def import_symbol_for_definition(path: Path, symbol: str) -> str:
    """Return ``%`` or ``Class::%`` for decorator import coverage on a mangled definition."""
    node = _find_definition_node(path, symbol)
    if isinstance(node, ast.ClassDef):
        return f"{node.name}::{MODULE_SYMBOL}"
    if "::" in symbol:
        return f"{symbol.split('::', 1)[0]}::{MODULE_SYMBOL}"
    return MODULE_SYMBOL


def touched_definition_symbols(path: Path, changed_lines: set[int] | frozenset[int]) -> frozenset[str]:
    """Mangled definition symbols whose span intersects *changed_lines*."""
    changed = set(changed_lines)
    if not changed:
        return frozenset()
    file_symbols = collect_file_symbols(path)
    touched: set[str] = set()
    for span in file_symbols.definitions:
        span_lines = set(range(span.start_line, span.end_line + 1))
        if span_lines & changed:
            touched.add(span.qualified_name)
    try:
        tree = _get_cached_tree(path)
    except (OSError, SyntaxError):
        return frozenset(touched)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        header_or_deco = {node.lineno} | _decorator_line_numbers(node)
        if changed & header_or_deco:
            touched.add(f"{node.name}::{MODULE_SYMBOL}")
    return frozenset(touched)


def coverage_checks_for_definition(
    path: Path,
    symbol: str,
    changed_lines: set[int] | frozenset[int],
) -> CoverageChecks:
    """Return import / strict / proxy line sets for gate coverage fallback."""
    changed = set(changed_lines)
    if not changed:
        return CoverageChecks(frozenset(), frozenset(), frozenset())

    node = _find_definition_node(path, symbol)
    if node is None:
        return CoverageChecks(frozenset(), frozenset(), frozenset())

    decorator_lines = _decorator_line_numbers(node)
    def_line = {node.lineno}
    if isinstance(node, ast.ClassDef):
        body_executable = _proxy_body_lines(path, symbol) - _decorator_line_numbers(node)
    else:
        body_executable = _body_executable_lines(path, symbol)

    decorator_hit = changed & decorator_lines
    def_hit = changed & def_line
    body_hit = changed & body_executable

    import_lines = frozenset(decorator_hit)
    strict_lines = frozenset(body_hit)
    proxy_lines: frozenset[int] = frozenset()
    if not body_hit and (def_hit or decorator_hit):
        proxy_lines = frozenset(body_executable)
    return CoverageChecks(import_lines, strict_lines, proxy_lines)


def gated_coverage_symbols(path: Path) -> frozenset[str]:
    """Canonical symbols that require test_map coverage for a newly added source file."""
    source_lines = _read_source_lines(path)
    if not source_lines:
        return frozenset()

    executable = _all_executable_lines(path, source_lines)
    if not executable:
        return frozenset()

    file_symbols = collect_file_symbols(path)
    required: set[str] = set()
    covered_lines: set[int] = set()

    for span in file_symbols.definitions:
        span_lines = set(range(span.start_line, span.end_line + 1))
        exec_lines = span_lines & executable
        if exec_lines:
            required.add(span.qualified_name)
            covered_lines |= exec_lines

    class_suffix = f"::{MODULE_SYMBOL}"
    for cls in file_symbols.class_spans:
        class_lines = set(range(cls.start_line, cls.end_line + 1))
        class_body_exec = class_lines & executable - covered_lines
        if class_body_exec:
            required.add(f"{cls.qualified_name}{class_suffix}")
            covered_lines |= class_body_exec

    module_exec = executable - covered_lines
    if module_exec:
        required.add(MODULE_SYMBOL)

    return frozenset(required)


def iter_canonical_definition_spans(path: Path) -> list[DefinitionSpan]:
    """Return canonical ``DefinitionSpan`` values for top-level defs and methods."""
    return list(collect_file_symbols(path).definitions)


# ---------------------------------------------------------------------------
# Definition span helpers
# ---------------------------------------------------------------------------


def top_level_definitions(path: Path) -> list[str]:
    """Return names of top-level functions, async functions, and classes."""
    tree = _get_cached_tree(path)
    return [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


def iter_qualified_definition_spans(path: Path) -> list[DefinitionSpan]:
    """Return canonical definition spans for gate and test_map tooling."""
    return list(iter_canonical_definition_spans(path))


def symbol_for_line(spans: list[DefinitionSpan], line: int) -> str | None:
    """Pick the smallest enclosing canonical span (innermost definition)."""
    containing = [
        (span, span.end_line - span.start_line + 1) for span in spans if span.start_line <= line <= span.end_line
    ]
    if not containing:
        return None
    span, _ = min(containing, key=lambda item: item[1])
    return span.qualified_name


def canonical_symbol_for_path_line(path: Path, line: int) -> str:
    """Resolve a line in *path* to the canonical symbol id."""
    return canonical_symbol_for_line(collect_file_symbols(path), line)


# ---------------------------------------------------------------------------
# Executable line filtering (single AST walk)
# ---------------------------------------------------------------------------


def _collect_non_executable_lines(tree: ast.AST) -> set[int]:
    """Single walk: collect docstring lines + non-executable statement lines."""
    lines: set[int] = set()

    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
            and node.body
            and isinstance(node.body[0], ast.Expr)
        ):
            value = node.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                start = node.body[0].lineno
                end = _end_line(node.body[0])
                lines.update(range(start, end + 1))

        if isinstance(node, ast.AnnAssign) and node.value is None:
            start = node.lineno
            end = _end_line(node)
            lines.update(range(start, end + 1))

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
    source_lines = _read_source_lines(path)
    if source_lines is None:
        return set()
    return _filter_executable_source_lines(path, source_lines, changed_lines)


def _line_text_is_executable(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def symbols_for_lines(path: Path, line_numbers: set[int]) -> set[str]:
    """Return canonical symbols that enclose the given line numbers."""
    return canonical_symbols_for_lines(path, line_numbers)


def canonical_symbols_for_lines(path: Path, line_numbers: set[int]) -> set[str]:
    """Return canonical symbols that enclose the given line numbers."""
    if not line_numbers:
        return set()
    source_lines = _read_source_lines(path)
    if not source_lines:
        return set()
    line_count = len(source_lines)
    symbols = collect_file_symbols(path)
    attributed: set[str] = set()
    for line_no in line_numbers:
        if line_no < 1 or line_no > line_count:
            continue
        attributed.add(canonical_symbol_for_line(symbols, line_no))
    return attributed
