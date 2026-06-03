"""Tests for ci_gate.errors — blocking error formatting."""

from __future__ import annotations

from scripts.helpers.ci_gate.errors import format_blocking_errors
from scripts.helpers.ci_gate.models import GateError


def test_format_blocking_errors_mentions_skipped_incremental_phases() -> None:
    errors = (GateError(category="modified_source", path="cli/main.py", symbol="run"),)
    text = format_blocking_errors(errors)
    assert "Phase 1/2" in text
    assert "Blocking items: 1" in text
    assert "no tests executed" not in text


def test_format_blocking_errors_groups_modified_source() -> None:
    errors = (
        GateError(category="modified_source", path="a.py", symbol="fn1"),
        GateError(category="modified_source", path="a.py", symbol="fn2"),
    )
    text = format_blocking_errors(errors)
    assert "cli/main.py::run" not in text
    assert "a.py::fn1" in text
    assert "a.py::fn2" in text
    assert "test_map entry" in text
