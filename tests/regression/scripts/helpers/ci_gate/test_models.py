"""Tests for ci_gate.models — ChangeSet, GateStepResult, Baseline, CiGatePlan."""

from __future__ import annotations

from datetime import date

from scripts.helpers.ci_gate.models import (
    Baseline,
    ChangeSet,
    CiGatePlan,
    CiGatePolicy,
    GateStepResult,
    PathPatterns,
    SourceExemption,
)

_TEST_INCLUDE = ("tests/**/test_*.py", "tests/**/*_test.py")
_TEST_EXCLUDE = ("tests/helpers/**", "tests/assets/**")
_CONFIG_INCLUDE = (
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
    "tests/**/conftest.py",
)


def _sample_exemption(file: str, symbol: str) -> SourceExemption:
    return SourceExemption(
        file=file,
        symbol=symbol,
        reason="test",
        applicant="test",
        approver="fangkai",
        deadline=date(2099, 12, 31),
    )


def _sample_policy(*, exemptions: tuple[SourceExemption, ...] = ()) -> CiGatePolicy:
    return CiGatePolicy(
        sources=PathPatterns(include_patterns=("cli/",), exclude_patterns=()),
        tests=PathPatterns(
            include_patterns=_TEST_INCLUDE,
            exclude_patterns=_TEST_EXCLUDE,
        ),
        configs=PathPatterns(include_patterns=_CONFIG_INCLUDE, exclude_patterns=()),
        source_exemptions=exemptions,
        test_exemptions=(),
        approvers=frozenset({"fangkai"}),
    )


def test_changeset_build_empty_returns_all_empty_tuples() -> None:
    cs = ChangeSet.build()
    assert cs.config == ()
    assert cs.new_test == ()
    assert cs.del_test == ()
    assert cs.new_source == ()
    assert cs.del_source == ()
    assert cs.modified_source == ()


def test_changeset_build_modified_source_stored_as_sorted_tuples() -> None:
    cs = ChangeSet.build(
        modified_source={"b.py": frozenset({3}), "a.py": frozenset({1})},
    )
    assert cs.modified_source[0][0] == "a.py"
    assert cs.modified_source[1][0] == "b.py"


def test_changeset_build_config_as_tuple() -> None:
    cs = ChangeSet.build(config=("pyproject.toml",))
    assert cs.config == ("pyproject.toml",)


def test_gate_step_result_defaults() -> None:
    gs = GateStepResult()
    assert gs.errors == ()
    assert gs.tests == frozenset()


def test_baseline_creation_stores_all_fields() -> None:
    policy = _sample_policy(exemptions=(_sample_exemption("a.py", "fn"),))
    b = Baseline(test_map={"a.py": {}}, policy=policy)
    assert "a.py" in b.test_map
    assert len(b.exemptions) == 1
    assert b.roots == ("cli/",)


def test_ci_gate_plan_all_fields_accessible() -> None:
    plan = CiGatePlan(
        deleted_source_tests=frozenset({"test_a"}),
        changed_test_nodes=frozenset({"test_b"}),
        regression_tests=frozenset({"test_c"}),
        full_suite=False,
    )
    assert plan.deleted_source_tests == frozenset({"test_a"})
    assert plan.changed_test_nodes == frozenset({"test_b"})
    assert plan.regression_tests == frozenset({"test_c"})
    assert plan.full_suite is False
