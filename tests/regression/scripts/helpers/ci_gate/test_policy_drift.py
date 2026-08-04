"""Tests for ci_gate.policy_drift."""

from __future__ import annotations

from datetime import date

from scripts.helpers.ci_gate.diff import DiffEntry
from scripts.helpers.ci_gate.models import (
    ChangeSet,
    CiGatePolicy,
    PathPatterns,
    SourceExemption,
    TestExemption,
)
from scripts.helpers.ci_gate.policy_drift import gate_exemption_drift, iter_rename_pairs


def _policy(
    *,
    source_exemptions: tuple[SourceExemption, ...] = (),
    test_exemptions: tuple[TestExemption, ...] = (),
) -> CiGatePolicy:
    roots = PathPatterns(("tensor_cast/",), ())
    tests = PathPatterns(("tests/**/test_*.py",), ())
    configs = PathPatterns(("pyproject.toml",), ())
    return CiGatePolicy(
        sources=roots,
        tests=tests,
        configs=configs,
        source_exemptions=source_exemptions,
        test_exemptions=test_exemptions,
        approvers=frozenset({"fangkai"}),
    )


def _source_exemption(file: str, symbol: str) -> SourceExemption:
    return SourceExemption(
        file=file,
        symbol=symbol,
        reason="test",
        applicant="alice",
        approver="fangkai",
        deadline=date(2099, 12, 31),
    )


def test_iter_rename_pairs_extracts_rename_entries() -> None:
    entries = (
        DiffEntry(status="M", old_path="a.py", new_path="a.py"),
        DiffEntry(status="R100", old_path="old.py", new_path="new.py"),
        DiffEntry(status="D", old_path="gone.py", new_path=None),
    )
    assert iter_rename_pairs(entries) == (("old.py", "new.py"),)


def test_gate_exemption_drift_deleted_source() -> None:
    policy = _policy(source_exemptions=(_source_exemption("tensor_cast/ops.py", "add"),))
    changes = ChangeSet.build(del_source=("tensor_cast/ops.py",))
    errors = gate_exemption_drift(policy, changes, ())
    assert len(errors) == 1
    assert errors[0].category == "exemption_drift"
    assert errors[0].path == "tensor_cast/ops.py"
    assert errors[0].symbol == "add"


def test_gate_exemption_drift_renamed_source() -> None:
    policy = _policy(source_exemptions=(_source_exemption("tensor_cast/old.py", "fn"),))
    changes = ChangeSet.build(del_source=("tensor_cast/old.py",), new_source=("tensor_cast/new.py",))
    rename_pairs = (("tensor_cast/old.py", "tensor_cast/new.py"),)
    errors = gate_exemption_drift(policy, changes, rename_pairs)
    assert len(errors) == 1
    assert "renamed source" in errors[0].detail
    assert "tensor_cast/new.py::fn" in errors[0].detail


def test_gate_exemption_drift_deleted_test() -> None:
    policy = _policy(
        test_exemptions=(
            TestExemption(
                test_id="tests/regression/tensor_cast/test_ops.py::test_add",
                reason="x",
                applicant="alice",
                approver="fangkai",
                deadline=date(2099, 12, 31),
            ),
        )
    )
    changes = ChangeSet.build(del_test=("tests/regression/tensor_cast/test_ops.py",))
    errors = gate_exemption_drift(policy, changes, ())
    assert len(errors) == 1
    assert errors[0].category == "exemption_drift"
    assert "deleted test file" in errors[0].detail


def test_gate_exemption_drift_renamed_test() -> None:
    policy = _policy(
        test_exemptions=(
            TestExemption(
                test_id="tests/regression/old/test_ops.py::test_add",
                reason="x",
                applicant="alice",
                approver="fangkai",
                deadline=date(2099, 12, 31),
            ),
        )
    )
    changes = ChangeSet.build(
        del_test=("tests/regression/old/test_ops.py",),
        new_test=("tests/regression/new/test_ops.py",),
    )
    rename_pairs = (("tests/regression/old/test_ops.py", "tests/regression/new/test_ops.py"),)
    errors = gate_exemption_drift(policy, changes, rename_pairs)
    assert len(errors) == 1
    assert "renamed test file" in errors[0].detail
    assert "tests/regression/new/test_ops.py::test_add" in errors[0].detail


def test_gate_exemption_drift_no_errors_when_paths_unchanged() -> None:
    policy = _policy(source_exemptions=(_source_exemption("tensor_cast/ops.py", "add"),))
    changes = ChangeSet.build(modified_source={"tensor_cast/ops.py": frozenset({2})})
    assert gate_exemption_drift(policy, changes, ()) == ()
