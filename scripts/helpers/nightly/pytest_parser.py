"""Parse pytest stdout into NightlyRunStats."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

_PYTEST_SHORT_ERROR_PREFIX: Final = "E   "
_PYTEST_ERROR_LINE_PREFIX: Final = "ERROR "
_PYTEST_ERRORS_SECTION_TITLE: Final = "ERRORS"
_PYTEST_ERROR_COLLECTING_MARKER: Final = "ERROR collecting"
_LOG_SNIPPET_CONTEXT_LINES: Final = 12
_LOG_SNIPPET_MAX_LINES: Final = 8
_EXCEPTION_MARKERS: Final = ("ValueError:", "ImportError:", "ModuleNotFoundError:")
_NODE_OUTCOME_RE: Final = re.compile(r"^(?P<node>.+?) (FAILED|ERROR)(?:\s|$)")
_XDIST_OUTCOME_RE: Final = re.compile(r"^\[(?:gw\d+|master)\]\s+\[[^\]]+\]\s+(?:FAILED|ERROR)\s+(?P<node>.+?)\s*$")
_TB_LINE_RE: Final = re.compile(r"^(?:FAILED|ERROR)\s+(?P<node>\S+)\s+-\s+(?P<reason>.+)$")
_STDOUT_UNITTEST_NODE_RE: Final = re.compile(
    r"^(?P<module>tests/.+/test_\w+)/(?P<class_name>[A-Z]\w*)\.py::(?P<method>.+)$"
)


def normalize_stdout_node_id(raw: str) -> str:
    """Normalize pytest -vv stdout node ids to canonical ``file.py::Class::method`` form.

    Pytest may print unittest class methods as ``tests/pkg/test_mod/ClassName.py::method``
    instead of ``tests/pkg/test_mod.py::ClassName::method``.
    """
    stripped = raw.strip()
    if not stripped:
        return stripped
    if stripped.count("::") >= 2:
        return stripped
    match = _STDOUT_UNITTEST_NODE_RE.match(stripped)
    if not match:
        return stripped
    return f"{match.group('module')}.py::{match.group('class_name')}::{match.group('method')}"


_SUMMARY_LINE_RE: Final = re.compile(r"^=+\s+(.+?)\s+=+$")
_SUMMARY_COUNT_RE: Final = re.compile(r"(\d+)\s+(failed|error|passed|skipped)")


@dataclass(frozen=True, slots=True)
class NightlyRunStats:
    passed: int
    failed: int
    errors: int
    duration_sec: float
    failed_cases: tuple[str, ...]
    first_error: str
    failure_reasons: dict[str, str]


def _extract_pytest_stdout_snippet(text: str, *, max_lines: int = _LOG_SNIPPET_MAX_LINES) -> str:
    lines = text.splitlines()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(_PYTEST_SHORT_ERROR_PREFIX):
            return stripped.removeprefix(_PYTEST_SHORT_ERROR_PREFIX)
        if stripped.startswith(_PYTEST_ERROR_LINE_PREFIX):
            return stripped.removeprefix(_PYTEST_ERROR_LINE_PREFIX)

    for index, line in enumerate(lines):
        if _PYTEST_ERROR_COLLECTING_MARKER in line or line.strip() == _PYTEST_ERRORS_SECTION_TITLE:
            chunk = [entry.strip() for entry in lines[index : index + _LOG_SNIPPET_CONTEXT_LINES] if entry.strip()]
            return "\n".join(chunk[:max_lines])

    for line in reversed(lines):
        stripped = line.strip()
        if any(marker in stripped for marker in _EXCEPTION_MARKERS):
            return stripped

    return ""


def _parse_summary_counts(summary_body: str) -> tuple[int, int, int, float]:
    passed = 0
    failed = 0
    errors = 0
    for match in _SUMMARY_COUNT_RE.finditer(summary_body):
        count = int(match.group(1))
        kind = match.group(2)
        if kind == "failed":
            failed = count
        elif kind == "error":
            errors = count
        elif kind == "passed":
            passed = count

    duration_sec = -1.0
    duration_match = re.search(r"in\s+([\d.]+)s", summary_body)
    if duration_match:
        duration_sec = float(duration_match.group(1))
    return passed, failed, errors, duration_sec


def parse_pytest_stdout(text: str, *, exit_code: int = 0) -> NightlyRunStats:
    """Parse pytest -vv/--tb=line stdout for FAILED/ERROR node ids, reasons, and counts."""
    failed_cases: list[str] = []
    failure_reasons: dict[str, str] = {}
    passed = 0
    failed = 0
    errors = 0
    duration_sec = -1.0
    first_error = ""

    for line in text.splitlines():
        stripped = line.strip()

        tb_line = _TB_LINE_RE.match(stripped)
        if tb_line:
            node = normalize_stdout_node_id(tb_line.group("node"))
            reason = tb_line.group("reason").strip()
            if node not in failed_cases:
                failed_cases.append(node)
            if reason:
                failure_reasons[node] = reason
            continue

        xdist_outcome = _XDIST_OUTCOME_RE.match(stripped)
        if xdist_outcome:
            node = normalize_stdout_node_id(xdist_outcome.group("node"))
            if node not in failed_cases:
                failed_cases.append(node)
            continue

        outcome = _NODE_OUTCOME_RE.match(stripped)
        if outcome:
            node = normalize_stdout_node_id(outcome.group("node"))
            if node not in failed_cases:
                failed_cases.append(node)
            continue

        summary_match = _SUMMARY_LINE_RE.match(line.strip())
        if summary_match:
            passed, failed, errors, duration_sec = _parse_summary_counts(summary_match.group(1))

    if exit_code != 0 and not failed_cases and failed == 0 and errors == 0:
        first_error = _extract_pytest_stdout_snippet(text)

    return NightlyRunStats(
        passed=passed,
        failed=failed,
        errors=errors,
        duration_sec=duration_sec,
        failed_cases=tuple(failed_cases),
        first_error=first_error,
        failure_reasons=failure_reasons,
    )


def merge_nightly_run_stats(*stats_list: NightlyRunStats) -> NightlyRunStats:
    """Combine stats from parallel or sequential pytest waves (e.g. UT + benchmark)."""
    passed = 0
    failed = 0
    errors = 0
    duration_sec = 0.0
    has_duration = False
    failed_cases: list[str] = []
    failure_reasons: dict[str, str] = {}
    first_error = ""

    for stats in stats_list:
        passed += stats.passed
        failed += stats.failed
        errors += stats.errors
        if stats.duration_sec >= 0:
            duration_sec += stats.duration_sec
            has_duration = True
        for node in stats.failed_cases:
            if node not in failed_cases:
                failed_cases.append(node)
        failure_reasons.update(stats.failure_reasons)
        if not first_error and stats.first_error:
            first_error = stats.first_error

    return NightlyRunStats(
        passed=passed,
        failed=failed,
        errors=errors,
        duration_sec=duration_sec if has_duration else -1.0,
        failed_cases=tuple(failed_cases),
        first_error=first_error,
        failure_reasons=failure_reasons,
    )
