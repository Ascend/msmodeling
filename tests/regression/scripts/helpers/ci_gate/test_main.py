"""Tests for ci_gate.main — build_ci_gate_plan and compute_execution_plan."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from scripts.helpers.ci_gate.gate_policy import SourceExemption, TestExemption, default_test_discovery
from scripts.helpers.ci_gate.main import (
    build_ci_gate_plan,
    build_coverage_mapping_errors,
    build_hard_blocking_plan,
    compute_execution_plan,
)
from scripts.helpers.ci_gate.models import Baseline, ChangeSet


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
        "cli/main.py": {
            "run": ["tests/regression/cli/test_run.py::test_run"],
        },
        "tensor_cast/ops.py": {
            "add": [
                "tests/regression/tensor_cast/test_ops.py::test_add",
                "tests/regression/cli/test_cross.py::test_cross",
            ],
        },
    }
    return Baseline(
        test_map=test_map,
        exemptions=(),
        test_exemptions=(),
        discovery=default_test_discovery(),
        roots=(
            "cli/",
            "tensor_cast/",
            "serving_cast/",
            "web_ui/",
            "scripts/",
            "tools/",
        ),
    )


@pytest.fixture(autouse=True)
def _stub_collect_test_node_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_collect(targets: tuple[str, ...] | list[str], *, marker: str) -> tuple[str, ...]:
        return tuple(f"{path}::test_case" for path in targets)

    monkeypatch.setattr(
        "scripts.helpers.ci_gate.rules.collect_test_node_ids",
        _fake_collect,
    )


def test_plan_config_change_triggers_full_suite(baseline: Baseline) -> None:
    cs = ChangeSet.build(config=("pyproject.toml",))
    plan = build_ci_gate_plan(Path("/tmp"), cs, baseline)
    assert plan.full_suite is True


def test_plan_gate_policy_change_does_not_trigger_full_suite(baseline: Baseline) -> None:
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
        exemptions=baseline.exemptions,
        test_exemptions=(
            TestExemption(
                test_id="tests/smoke/test_new.py::test_case",
                reason="x",
                applicant="a",
                approver="fangkai",
                deadline=date(2099, 12, 31),
            ),
        ),
        discovery=baseline.discovery,
        roots=baseline.roots,
    )
    cs = ChangeSet.build(new_test=("tests/smoke/test_new.py",))
    plan = build_ci_gate_plan(Path("/tmp"), cs, exempt_baseline)
    assert plan.changed_test_nodes == frozenset()


def test_plan_deleted_test_sole_coverage_blocking(baseline: Baseline) -> None:
    cs = ChangeSet.build(del_test=("tests/regression/cli/test_run.py::test_run",))
    errors = build_hard_blocking_plan(cs, baseline.test_map, baseline.roots)
    assert len(errors) == 1
    assert errors[0].category == "deleted_test"


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
        del_test=("tests/regression/cli/test_run.py::test_run",),
    )
    errors = build_hard_blocking_plan(cs, baseline.test_map, baseline.roots)
    assert len(errors) == 2


def test_plan_new_source_with_exemption_no_error(baseline: Baseline, tmp_path: Path) -> None:
    src = tmp_path / "tensor_cast" / "new_mod.py"
    src.parent.mkdir(parents=True)
    src.write_text("def fn():\n    pass\n", encoding="utf-8")
    cs = ChangeSet.build(new_source=("tensor_cast/new_mod.py",))
    exempt_baseline = Baseline(
        test_map=baseline.test_map,
        exemptions=(_sample_exemption("tensor_cast/new_mod.py", "fn"),),
        test_exemptions=(),
        discovery=baseline.discovery,
        roots=baseline.roots,
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
        blocking_errors=(),
        deleted_source_tests=frozenset(),
        changed_test_nodes=frozenset(),
        regression_tests=frozenset(),
        full_suite=True,
    )
    execution = compute_execution_plan(plan, ())
    assert execution.full_suite is True
    assert execution.waves[0].targets == ("tests",)
    assert execution.waves[0].marker == "not npu"


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
        blocking_errors=(),
        deleted_source_tests=frozenset(),
        changed_test_nodes=frozenset({shared}),
        regression_tests=frozenset({shared, "tests/regression/cli/test_other.py::test_y"}),
        full_suite=False,
    )
    execution = compute_execution_plan(plan, ())
    all_nodes = [node for wave in execution.waves for node in wave.targets]
    assert all_nodes.count(shared) == 1
    assert len(execution.waves) == 2
    assert execution.waves[0].marker == "not npu"
    assert execution.waves[1].marker == "not npu and not nightly and not network"
