"""Parse pytest output into NightlyRunStats. Single-pass regex extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

_RE_PASSED = re.compile(r"^(.+?) PASSED", re.MULTILINE)
_RE_FAILED = re.compile(r"^FAILED (.+?) -", re.MULTILINE)
_RE_ERROR = re.compile(r"^ERROR ", re.MULTILINE)
_RE_DURATION = re.compile(r"(\d+(?:\.\d+)?) seconds")
_RE_FIRST_ERROR = re.compile(r"E\s+\w+Error.+")


@dataclass(frozen=True, slots=True)
class NightlyRunStats:
    passed: int
    failed: int
    errors: int
    duration_sec: float
    failed_cases: tuple[str, ...]
    first_error: str


def parse_pytest_output(text: str) -> NightlyRunStats:
    """Extract test statistics from pytest stdout.

    Each pattern applied once — no iterative rescanning.
    """
    passed = len(_RE_PASSED.findall(text))
    failed = len(_RE_FAILED.findall(text))
    errors_count = len(_RE_ERROR.findall(text))

    duration_m = _RE_DURATION.search(text)
    duration = float(duration_m.group(1)) if duration_m else -1.0

    failed_cases = tuple(_RE_FAILED.findall(text))

    first_error_m = _RE_FIRST_ERROR.search(text)
    first_error = first_error_m.group(0).strip() if first_error_m else ""

    return NightlyRunStats(
        passed=passed,
        failed=failed,
        errors=errors_count,
        duration_sec=duration,
        failed_cases=failed_cases,
        first_error=first_error,
    )
