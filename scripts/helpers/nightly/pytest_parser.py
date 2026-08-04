"""Parse pytest JUnit XML into NightlyRunStats."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Sequence

_PYTEST_SHORT_ERROR_PREFIX: Final = "E   "
_PYTEST_ERROR_LINE_PREFIX: Final = "ERROR "
_PYTEST_ERRORS_SECTION_TITLE: Final = "ERRORS"
_PYTEST_ERROR_COLLECTING_MARKER: Final = "ERROR collecting"
_LOG_SNIPPET_CONTEXT_LINES: Final = 12
_LOG_SNIPPET_MAX_LINES: Final = 8
_EXCEPTION_MARKERS: Final = ("ValueError:", "ImportError:", "ModuleNotFoundError:")


@dataclass(frozen=True, slots=True)
class NightlyRunStats:
    passed: int
    failed: int
    errors: int
    duration_sec: float
    failed_cases: tuple[str, ...]
    first_error: str


@dataclass(frozen=True, slots=True)
class JunitPhaseParse:
    stats: NightlyRunStats
    durations: tuple[tuple[str, float], ...]


def _format_testcase_id(classname: str, name: str, file: str = "") -> str:
    """Build a pytest node id from a JUnit ``testcase`` element.

    Prefers the ``file`` attribute (always emitted by pytest) so class-based
    tests keep their class component instead of being folded into the path.
    Falls back to deriving the path from ``classname`` when ``file`` is absent.
    """
    if file:
        module_dotted = file[:-3].replace("/", ".") if file.endswith(".py") else file.replace("/", ".")
        if classname == module_dotted:
            return f"{file}::{name}"
        if classname.startswith(f"{module_dotted}."):
            class_part = classname[len(module_dotted) + 1 :]
            return f"{file}::{class_part}::{name}"
        return f"{file}::{name}"
    if classname:
        module_path = classname.replace(".", "/")
        if not module_path.endswith(".py"):
            module_path = f"{module_path}.py"
        return f"{module_path}::{name}"
    return name


def _element_message(elem: ET.Element) -> str:
    message = (elem.get("message") or "").strip()
    if message:
        return message
    text = (elem.text or "").strip()
    if not text:
        return ""
    return text.splitlines()[0].strip()


def _accumulate_testsuite_direct_errors(
    suite: ET.Element,
    *,
    failed: int,
    errors: int,
    first_error: str,
) -> tuple[int, int, str]:
    updated_failed = failed
    updated_errors = errors
    updated_first_error = first_error
    for child in suite:
        if child.tag == "failure":
            updated_failed += 1
            if not updated_first_error:
                updated_first_error = _element_message(child)
        elif child.tag == "error":
            updated_errors += 1
            if not updated_first_error:
                updated_first_error = _element_message(child)
    return updated_failed, updated_errors, updated_first_error


def _accumulate_testcase(
    testcase: ET.Element,
    *,
    passed: int,
    failed: int,
    errors: int,
    duration_sec: float,
    failed_cases: list[str],
    first_error: str,
) -> tuple[int, int, int, float, list[str], str]:
    time_raw = testcase.get("time")
    updated_duration = duration_sec + float(time_raw) if time_raw is not None else duration_sec

    classname = testcase.get("classname", "")
    name = testcase.get("name", "")
    file = testcase.get("file", "")
    failure = testcase.find("failure")
    error = testcase.find("error")

    updated_passed = passed
    updated_failed = failed
    updated_errors = errors
    updated_failed_cases = failed_cases
    updated_first_error = first_error

    if failure is not None:
        updated_failed += 1
        updated_failed_cases.append(_format_testcase_id(classname, name, file))
        if not updated_first_error:
            updated_first_error = _element_message(failure)
    elif error is not None:
        updated_errors += 1
        if not updated_first_error:
            updated_first_error = _element_message(error)
    elif testcase.find("skipped") is None:
        updated_passed += 1

    return (
        updated_passed,
        updated_failed,
        updated_errors,
        updated_duration,
        updated_failed_cases,
        updated_first_error,
    )


def _parse_junit_root(root: ET.Element) -> JunitPhaseParse:
    passed = 0
    failed = 0
    errors = 0
    duration_sec = 0.0
    failed_cases: list[str] = []
    first_error = ""
    durations: list[tuple[str, float]] = []

    for suite in root.iter("testsuite"):
        failed, errors, first_error = _accumulate_testsuite_direct_errors(
            suite,
            failed=failed,
            errors=errors,
            first_error=first_error,
        )

    for testcase in root.iter("testcase"):
        time_raw = testcase.get("time")
        if time_raw is not None:
            durations.append(
                (
                    _format_testcase_id(
                        testcase.get("classname", ""),
                        testcase.get("name", ""),
                        testcase.get("file", ""),
                    ),
                    float(time_raw),
                )
            )
        passed, failed, errors, duration_sec, failed_cases, first_error = _accumulate_testcase(
            testcase,
            passed=passed,
            failed=failed,
            errors=errors,
            duration_sec=duration_sec,
            failed_cases=failed_cases,
            first_error=first_error,
        )

    return JunitPhaseParse(
        stats=NightlyRunStats(
            passed=passed,
            failed=failed,
            errors=errors,
            duration_sec=duration_sec,
            failed_cases=tuple(failed_cases),
            first_error=first_error,
        ),
        durations=tuple(durations),
    )


def _parse_junit_file(path: pathlib.Path) -> NightlyRunStats:
    return _parse_junit_root(ET.parse(path).getroot()).stats


def _merge_stats(stats_list: Sequence[NightlyRunStats]) -> NightlyRunStats:
    passed = 0
    failed = 0
    errors = 0
    duration_sec = 0.0
    has_duration = False
    failed_cases: list[str] = []
    first_error = ""

    for stats in stats_list:
        passed += stats.passed
        failed += stats.failed
        errors += stats.errors
        if stats.duration_sec >= 0:
            duration_sec += stats.duration_sec
            has_duration = True
        failed_cases.extend(stats.failed_cases)
        if not first_error and stats.first_error:
            first_error = stats.first_error

    return NightlyRunStats(
        passed=passed,
        failed=failed,
        errors=errors,
        duration_sec=duration_sec if has_duration else -1.0,
        failed_cases=tuple(failed_cases),
        first_error=first_error,
    )


def parse_junit_phase(path: pathlib.Path) -> JunitPhaseParse | None:
    """Parse a single JUnit XML file, or None when it is missing."""
    if not path.is_file():
        return None
    return _parse_junit_root(ET.parse(path).getroot())


def aggregate_phase_stats(phases: Sequence[JunitPhaseParse | None]) -> NightlyRunStats:
    """Merge per-phase stats from :func:`parse_junit_phases`."""
    parsed = [phase.stats for phase in phases if phase is not None]
    if not parsed:
        return NightlyRunStats(
            passed=0,
            failed=0,
            errors=0,
            duration_sec=-1.0,
            failed_cases=(),
            first_error="",
        )
    return _merge_stats(parsed)


def parse_junit_phases(
    paths: Sequence[pathlib.Path],
) -> tuple[JunitPhaseParse | None, ...]:
    """Parse each JUnit path once; missing files yield None."""
    return tuple(parse_junit_phase(path) for path in paths)


def parse_junit_xml(paths: Sequence[pathlib.Path]) -> NightlyRunStats:
    """Aggregate test statistics from one or more pytest JUnit XML files."""
    parsed = [phase.stats for phase in parse_junit_phases(paths) if phase is not None]

    if not parsed:
        return NightlyRunStats(
            passed=0,
            failed=0,
            errors=0,
            duration_sec=-1.0,
            failed_cases=(),
            first_error="",
        )

    return _merge_stats(parsed)


def parse_junit_file(path: pathlib.Path) -> NightlyRunStats | None:
    """Parse a single JUnit XML file, or None when it is missing."""
    if not path.is_file():
        return None
    return _parse_junit_file(path)


def extract_pytest_log_snippet(
    log_path: pathlib.Path,
    *,
    max_lines: int = _LOG_SNIPPET_MAX_LINES,
) -> str:
    """Return the first actionable error snippet from captured pytest stdout.

    Used when pytest exits non-zero before populating JUnit XML (e.g. collection
    or conftest import failures).
    """
    if not log_path.is_file():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
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


def slowest_testcase_durations(
    durations: Sequence[tuple[str, float]],
    *,
    top_n: int = 10,
) -> tuple[tuple[str, float], ...]:
    """Return the top-N slowest ``(node_id, seconds)`` from pre-parsed durations."""
    ranked = sorted(durations, key=lambda item: item[1], reverse=True)
    return tuple(ranked[:top_n])


def slowest_testcases(paths: Sequence[pathlib.Path], *, top_n: int = 10) -> tuple[tuple[str, float], ...]:
    """Return the top-N slowest ``(node_id, seconds)`` across the given files."""
    durations: list[tuple[str, float]] = []
    for phase in parse_junit_phases(paths):
        if phase is not None:
            durations.extend(phase.durations)
    return slowest_testcase_durations(durations, top_n=top_n)
