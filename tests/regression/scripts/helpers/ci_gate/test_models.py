"""Tests for ci_gate.models — ChangeSet, GateStepResult, Baseline, CiGatePlan."""

from __future__ import annotations

from datetime import date

from scripts.helpers.ci_gate.gate_policy import SourceExemption, default_test_discovery
from scripts.helpers.ci_gate.models import Baseline, ChangeSet, CiGatePlan, GateError, GateStepResult


def _sample_exemption(file: str, symbol: str) -> SourceExemption:
    return SourceExemption(
        file=file,
        symbol=symbol,
        reason="test",
        applicant="test",
        approver="fangkai",
        deadline=date(2099, 12, 31),
    )


# ---------------------------------------------------------------------------
# ChangeSet.build
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# GateStepResult
# ---------------------------------------------------------------------------


def test_gate_step_result_defaults() -> None:
    gs = GateStepResult()
    assert gs.errors == ()
    assert gs.tests == frozenset()


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_baseline_creation_stores_all_fields() -> None:
    b = Baseline(
        test_map={"a.py": {}},
        exemptions=(_sample_exemption("a.py", "fn"),),
        discovery=default_test_discovery(),
        test_exemptions=(),
        roots=("cli/",),
    )
    assert "a.py" in b.test_map
    assert len(b.exemptions) == 1
    assert b.roots == ("cli/",)


# ---------------------------------------------------------------------------
# CiGatePlan
# ---------------------------------------------------------------------------


def test_ci_gate_plan_all_fields_accessible() -> None:
    err = GateError(category="new_source", path="a.py")
    plan = CiGatePlan(
        blocking_errors=(err,),
        deleted_source_tests=frozenset({"test_a"}),
        changed_test_nodes=frozenset({"test_b"}),
        regression_tests=frozenset({"test_c"}),
        full_suite=False,
    )
    assert plan.blocking_errors == (err,)
    assert plan.deleted_source_tests == frozenset({"test_a"})
    assert plan.changed_test_nodes == frozenset({"test_b"})
    assert plan.regression_tests == frozenset({"test_c"})
    assert plan.full_suite is False
