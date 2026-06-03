"""Parse pytest JUnit XML into NightlyRunStats."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class NightlyRunStats:
    passed: int
    failed: int
    errors: int
    duration_sec: float
    failed_cases: tuple[str, ...]
    first_error: str


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


def _parse_junit_file(path: Path) -> NightlyRunStats:
    root = ET.parse(path).getroot()

    passed = 0
    failed = 0
    errors = 0
    duration_sec = 0.0
    failed_cases: list[str] = []
    first_error = ""

    for testcase in root.iter("testcase"):
        time_raw = testcase.get("time")
        if time_raw is not None:
            duration_sec += float(time_raw)

        classname = testcase.get("classname", "")
        name = testcase.get("name", "")
        file = testcase.get("file", "")
        failure = testcase.find("failure")
        error = testcase.find("error")

        if failure is not None:
            failed += 1
            failed_cases.append(_format_testcase_id(classname, name, file))
            if not first_error:
                first_error = _element_message(failure)
        elif error is not None:
            errors += 1
            if not first_error:
                first_error = _element_message(error)
        elif testcase.find("skipped") is None:
            passed += 1

    return NightlyRunStats(
        passed=passed,
        failed=failed,
        errors=errors,
        duration_sec=duration_sec,
        failed_cases=tuple(failed_cases),
        first_error=first_error,
    )


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


def parse_junit_xml(paths: Sequence[Path]) -> NightlyRunStats:
    """Aggregate test statistics from one or more pytest JUnit XML files."""
    parsed: list[NightlyRunStats] = []
    for path in paths:
        if not path.is_file():
            continue
        parsed.append(_parse_junit_file(path))

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


def parse_junit_file(path: Path) -> NightlyRunStats | None:
    """Parse a single JUnit XML file, or None when it is missing."""
    if not path.is_file():
        return None
    return _parse_junit_file(path)


def slowest_testcases(paths: Sequence[Path], *, top_n: int = 10) -> tuple[tuple[str, float], ...]:
    """Return the top-N slowest ``(node_id, seconds)`` across the given files."""
    durations: list[tuple[str, float]] = []
    for path in paths:
        if not path.is_file():
            continue
        root = ET.parse(path).getroot()
        for testcase in root.iter("testcase"):
            time_raw = testcase.get("time")
            if time_raw is None:
                continue
            node_id = _format_testcase_id(
                testcase.get("classname", ""),
                testcase.get("name", ""),
                testcase.get("file", ""),
            )
            durations.append((node_id, float(time_raw)))
    durations.sort(key=lambda item: item[1], reverse=True)
    return tuple(durations[:top_n])
