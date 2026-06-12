"""Tests for ci_gate.errors — blocking error formatting."""

from __future__ import annotations

from scripts.helpers.ci_gate.errors import format_blocking_errors, format_pytest_failure_hint
from scripts.helpers.ci_gate.models import GateError


def test_format_blocking_errors_mentions_skipped_pytest() -> None:
    errors = (GateError(category="modified_source", path="cli/main.py", symbol="run"),)
    text = format_blocking_errors(errors)
    assert "pytest was not run" in text
    assert "Blocking items: 1" in text
    assert "Phase" not in text


def test_format_pytest_failure_hint_lists_nodes_and_yaml_template() -> None:
    nodes = (
        "tests/regression/cli/test_a.py::test_one",
        "tests/regression/cli/test_b.py::test_two",
    )
    text = format_pytest_failure_hint(nodes)
    assert "selected test(s) failed" in text
    assert "tests/regression/cli/test_a.py::test_one" in text
    assert "exemptions.tests" in text
    assert "symbols:" in text
    assert "test_id:" not in text


def test_format_blocking_errors_after_pytest_uses_mapping_message() -> None:
    errors = (GateError(category="modified_source", path="cli/main.py", symbol="run"),)
    text = format_blocking_errors(errors, pytest_ran=True)
    assert "coverage mapping policy not satisfied after pytest" in text
    assert "pytest was not run" not in text


def test_format_blocking_errors_groups_modified_source() -> None:
    errors = (
        GateError(category="modified_source", path="a.py", symbol="fn1"),
        GateError(category="modified_source", path="a.py", symbol="fn2"),
    )
    text = format_blocking_errors(errors)
    assert "cli/main.py::run" not in text
    assert "a.py::fn1" in text
    assert "a.py::fn2" in text
    assert "coverage mapping entry" in text
