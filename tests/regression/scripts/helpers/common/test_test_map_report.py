"""Tests for common.test_map_report."""

from __future__ import annotations

from datetime import date

from scripts.helpers.ci_gate.models import GatePolicy, PathPatterns, SourceExemption
from scripts.helpers.common.build_test_map import detect_redundant_cases
from scripts.helpers.common.test_map_report import (
    find_expired_unmapped_in_map,
    iter_unique_symbol_refs,
    summarize_test_map,
)

_TEST_INCLUDE = ("tests/**/test_*.py", "tests/**/*_test.py")
_TEST_EXCLUDE = ("tests/helpers/**", "tests/assets/**")
_CONFIG_INCLUDE = (
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
    "tests/**/conftest.py",
)


def test_summarize_test_map_counts_nodes_and_unique_symbol_refs() -> None:
    summary = summarize_test_map(
        {
            "tests/smoke/test_a.py::test_x": {"a.py": ["fn1"], "b.py": ["fn2"]},
            "tests/smoke/test_b.py::test_y": {"b.py": ["fn2"]},
        }
    )
    assert summary.test_nodes == 2
    assert summary.symbol_refs == 2


def test_iter_unique_symbol_refs_deduplicates() -> None:
    refs = iter_unique_symbol_refs(
        {
            "tests/a.py::test_one": {"cli/main.py": ["run", "run"]},
            "tests/a.py::test_two": {"cli/main.py": ["run"]},
        }
    )
    assert refs == {("cli/main.py", "run")}


def test_detect_redundant_cases_flags_over_covered_symbol() -> None:
    mapping = {
        "tests/smoke/test_x.py::test_x": {
            "cli/main.py": ["run"],
        },
        **{f"tests/smoke/test_{index}.py::test_x": {"cli/main.py": ["run"]} for index in range(5)},
    }
    warnings = detect_redundant_cases(mapping, max_per_symbol=5)
    assert any(warning.get("type") == "over_covered_symbol" for warning in warnings)


def test_find_expired_unmapped_in_map_reports_missing_symbol() -> None:
    policy = GatePolicy(
        sources=PathPatterns(include_patterns=("cli/",), exclude_patterns=()),
        tests=PathPatterns(
            include_patterns=_TEST_INCLUDE,
            exclude_patterns=_TEST_EXCLUDE,
        ),
        configs=PathPatterns(include_patterns=_CONFIG_INCLUDE, exclude_patterns=()),
        source_exemptions=(
            SourceExemption(
                file="cli/main.py",
                symbol="run",
                reason="legacy",
                applicant="dev",
                approver="owner",
                deadline=date(2020, 1, 1),
            ),
        ),
        test_exemptions=(),
        approvers=frozenset({"owner"}),
    )
    reports = find_expired_unmapped_in_map(policy, {}, today=date(2026, 1, 1))
    assert len(reports) == 1
    assert reports[0].symbol_key == "cli/main.py::run"
