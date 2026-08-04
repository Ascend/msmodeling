"""Tests for ci_gate.main — build_ci_gate_plan and compute_execution_plan."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from scripts.helpers.ci_gate.main import (
    build_ci_gate_plan,
    build_coverage_mapping_errors,
    build_hard_blocking_plan,
    collect_product_shadow_warnings,
    compute_execution_plan,
)
from scripts.helpers.ci_gate.models import (
    Baseline,
    ChangeSet,
    CiGatePolicy,
    SourceExemption,
    TestExemption,
)
from tests.regression.scripts.helpers.conftest import default_ci_gate_policy


def _sample_exemption(file: str, symbol: str) -> SourceExemption:
    return SourceExemption(
        file=file,
        symbol=symbol,
        reason="test",
        applicant="test",
        approver="fangkai",
        deadline=date(2099, 12, 31),
    )


@pytest.fixture(scope="module")
def baseline() -> Baseline:
    test_map = {
        "tests/regression/cli/test_run.py::test_run": {
            "cli/main.py": ["run"],
        },
        "tests/regression/tensor_cast/test_ops.py::test_add": {
            "tensor_cast/ops.py": ["add"],
        },
        "tests/regression/cli/test_cross.py::test_cross": {
            "tensor_cast/ops.py": ["add"],
        },
    }
    return Baseline(test_map=test_map, policy=default_ci_gate_policy())


@pytest.fixture(autouse=True)
def _stub_collect_test_node_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_collect(targets: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        return tuple(f"{path}::test_case" for path in targets)

    monkeypatch.setattr(
        "scripts.helpers.ci_gate.rules.collect_all_test_node_ids",
        _fake_collect,
    )


def test_plan_config_change_triggers_full_suite(baseline: Baseline) -> None:
    cs = ChangeSet.build(config=("pyproject.toml",))
    plan = build_ci_gate_plan(Path("/tmp"), cs, baseline)
    assert plan.full_suite is True


def test_plan_build_py_change_does_not_trigger_full_suite(baseline: Baseline) -> None:
    cs = ChangeSet.build(unscoped_python=("build.py",))
    plan = build_ci_gate_plan(Path("/tmp"), cs, baseline)
    assert plan.full_suite is False


def test_plan_gate_policy_change_does_not_trigger_full_suite(
    baseline: Baseline,
) -> None:
    cs = ChangeSet.build()
    plan = build_ci_gate_plan(Path("/tmp"), cs, baseline)
    assert plan.full_suite is False


def test_plan_new_test_selects_collected_nodes(baseline: Baseline) -> None:
    cs = ChangeSet.build(new_test=("tests/smoke/test_new.py",))
    plan = build_ci_gate_plan(Path("/tmp"), cs, baseline)
    assert plan.changed_test_nodes == frozenset({"tests/smoke/test_new.py::test_case"})


def test_plan_all_exempt_changed_test_file_not_scheduled(
    baseline: Baseline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exempt_baseline = Baseline(
        test_map=baseline.test_map,
        policy=baseline.policy.__class__(
            sources=baseline.policy.sources,
            tests=baseline.policy.tests,
            configs=baseline.policy.configs,
            source_exemptions=baseline.policy.source_exemptions,
            test_exemptions=(
                TestExemption(
                    test_id="tests/smoke/test_new.py::test_case",
                    reason="x",
                    applicant="a",
                    approver="fangkai",
                    deadline=date(2099, 12, 31),
                ),
            ),
            approvers=baseline.policy.approvers,
        ),
    )
    cs = ChangeSet.build(new_test=("tests/smoke/test_new.py",))
    plan = build_ci_gate_plan(Path("/tmp"), cs, exempt_baseline)
    assert plan.changed_test_nodes == frozenset()


def test_plan_deleted_test_sole_coverage_blocking(baseline: Baseline) -> None:
    cs = ChangeSet.build(del_test=("tests/regression/cli/test_run.py",))
    errors = build_hard_blocking_plan(cs, baseline.test_map, baseline.policy)
    assert len(errors) == 1
    assert errors[0].category == "deleted_test"


def test_plan_deleted_source_sole_coverage_blocking(baseline: Baseline) -> None:
    cs = ChangeSet.build(del_source=("cli/main.py",))
    errors = build_hard_blocking_plan(cs, baseline.test_map, baseline.policy)
    assert len(errors) == 1
    assert errors[0].category == "deleted_source"


def test_plan_modified_source_selects_regression_tests(baseline: Baseline, tmp_path: Path) -> None:
    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text("def run():\n    x = 1\n", encoding="utf-8")
    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({2})})
    plan = build_ci_gate_plan(tmp_path, cs, baseline)
    assert "tests/regression/cli/test_run.py::test_run" in plan.regression_tests


def test_plan_deleted_source_selects_guard_tests(baseline: Baseline) -> None:
    cs = ChangeSet.build(del_source=("cli/main.py",))
    plan = build_ci_gate_plan(Path("/tmp"), cs, baseline)
    assert "tests/regression/cli/test_run.py::test_run" in plan.deleted_source_tests


def test_plan_blocking_errors_combined(baseline: Baseline, tmp_path: Path) -> None:
    cs = ChangeSet.build(
        del_source=("tensor_cast/unknown.py",),
        del_test=("tests/regression/cli/test_run.py",),
    )
    errors = build_hard_blocking_plan(cs, baseline.test_map, baseline.policy)
    assert len(errors) == 1


def test_plan_exemption_drift_deleted_source_blocking(baseline: Baseline) -> None:
    policy = CiGatePolicy(
        sources=baseline.policy.sources,
        tests=baseline.policy.tests,
        configs=baseline.policy.configs,
        source_exemptions=(_sample_exemption("tensor_cast/unknown.py", "fn"),),
        test_exemptions=baseline.policy.test_exemptions,
        approvers=baseline.policy.approvers,
    )
    cs = ChangeSet.build(del_source=("tensor_cast/unknown.py",))
    errors = build_hard_blocking_plan(cs, baseline.test_map, policy)
    drift = [err for err in errors if err.category == "exemption_drift"]
    assert len(drift) == 1
    assert drift[0].path == "tensor_cast/unknown.py"


def test_collect_product_shadow_warnings_reports_duplicate_defs(tmp_path: Path) -> None:
    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text(
        "def foo():\n    pass\n\ndef foo():\n    return 1\n",
        encoding="utf-8",
    )
    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({4})})
    warnings = collect_product_shadow_warnings(tmp_path, cs, ("cli/",))
    assert len(warnings) == 1
    assert warnings[0].name == "foo"
    assert warnings[0].shadowed_by_line == 4


def test_plan_new_source_with_exemption_no_error(baseline: Baseline, tmp_path: Path) -> None:
    src = tmp_path / "tensor_cast" / "new_mod.py"
    src.parent.mkdir(parents=True)
    src.write_text("def fn():\n    pass\n", encoding="utf-8")
    cs = ChangeSet.build(new_source=("tensor_cast/new_mod.py",))
    exempt_baseline = Baseline(
        test_map=baseline.test_map,
        policy=baseline.policy.__class__(
            sources=baseline.policy.sources,
            tests=baseline.policy.tests,
            configs=baseline.policy.configs,
            source_exemptions=(_sample_exemption("tensor_cast/new_mod.py", "fn"),),
            test_exemptions=(),
            approvers=baseline.policy.approvers,
        ),
    )
    errors = build_coverage_mapping_errors(
        tmp_path,
        cs,
        exempt_baseline.test_map,
        exempt_baseline.exemptions,
        exempt_baseline.roots,
        coverage_path=tmp_path / ".coverage",
    )
    assert len(errors) == 0


def test_compute_execution_plan_full_suite() -> None:
    from scripts.helpers.ci_gate.models import CiGatePlan

    plan = CiGatePlan(
        deleted_source_tests=frozenset(),
        changed_test_nodes=frozenset(),
        regression_tests=frozenset(),
        full_suite=True,
    )
    execution = compute_execution_plan(plan, ())
    assert execution.full_suite is True
    assert execution.waves[0].targets == ("tests",)
    assert execution.waves[0].marker == "not npu and not nightly and not network"


def test_plan_config_change_skips_changed_test_collection(baseline: Baseline) -> None:
    cs = ChangeSet.build(
        config=("pyproject.toml",),
        new_test=("tests/smoke/test_new.py",),
    )
    plan = build_ci_gate_plan(Path("/tmp"), cs, baseline)
    assert plan.full_suite is True
    assert plan.changed_test_nodes == frozenset()


def test_compute_execution_plan_dedupes_changed_and_regression() -> None:
    from scripts.helpers.ci_gate.models import CiGatePlan

    shared = "tests/regression/cli/test_shared.py::test_x"
    plan = CiGatePlan(
        deleted_source_tests=frozenset(),
        changed_test_nodes=frozenset({shared}),
        regression_tests=frozenset({shared, "tests/regression/cli/test_other.py::test_y"}),
        full_suite=False,
    )
    execution = compute_execution_plan(plan, ())
    all_nodes = [node for wave in execution.waves for node in wave.targets]
    assert all_nodes.count(shared) == 1
    assert len(execution.waves) == 2
    assert execution.waves[0].marker is None
    assert execution.waves[1].marker == "not npu and not nightly and not network"
