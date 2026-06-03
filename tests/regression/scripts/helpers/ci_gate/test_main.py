"""Tests for ci_gate.main — build_ci_gate_plan."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from scripts.helpers.ci_gate.gate_policy import SourceExemption, default_test_discovery
from scripts.helpers.ci_gate.main import build_ci_gate_plan
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


# ---------------------------------------------------------------------------
# Baseline fixture
# ---------------------------------------------------------------------------


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
        discovery=default_test_discovery(),
        product_prefixes=(
            "cli/",
            "tensor_cast/",
            "serving_cast/",
            "web_ui/",
            "scripts/",
        ),
    )


# ---------------------------------------------------------------------------
# build_ci_gate_plan
# ---------------------------------------------------------------------------


def test_plan_config_change_triggers_full_suite(baseline: Baseline) -> None:
    cs = ChangeSet.build(config=("pyproject.toml",))
    plan = build_ci_gate_plan(Path("/tmp"), cs, baseline)
    assert plan.full_suite is True


def test_plan_new_test_selected(baseline: Baseline) -> None:
    cs = ChangeSet.build(new_test=("tests/smoke/test_new.py::test_x",))
    plan = build_ci_gate_plan(Path("/tmp"), cs, baseline)
    assert "tests/smoke/test_new.py::test_x" in plan.incremental_tests


def test_plan_deleted_test_sole_coverage_blocking(baseline: Baseline) -> None:
    cs = ChangeSet.build(del_test=("tests/regression/cli/test_run.py::test_run",))
    plan = build_ci_gate_plan(Path("/tmp"), cs, baseline)
    assert len(plan.blocking_errors) == 1
    assert plan.blocking_errors[0].category == "deleted_test"


def test_plan_modified_source_selects_tests(baseline: Baseline, tmp_path: Path) -> None:
    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text("def run():\n    x = 1\n", encoding="utf-8")
    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({2})})
    plan = build_ci_gate_plan(tmp_path, cs, baseline)
    assert "tests/regression/cli/test_run.py::test_run" in plan.incremental_tests


def test_plan_deleted_source_selects_guard_tests(baseline: Baseline) -> None:
    cs = ChangeSet.build(del_source=("cli/main.py",))
    plan = build_ci_gate_plan(Path("/tmp"), cs, baseline)
    assert "tests/regression/cli/test_run.py::test_run" in plan.deleted_source_tests


def test_plan_blocking_errors_combined(baseline: Baseline, tmp_path: Path) -> None:
    """Deleted source without map entry + deleted test sole coverage."""
    cs = ChangeSet.build(
        del_source=("tensor_cast/unknown.py",),
        del_test=("tests/regression/cli/test_run.py::test_run",),
    )
    plan = build_ci_gate_plan(tmp_path, cs, baseline)
    assert len(plan.blocking_errors) == 2


def test_plan_new_source_with_exemption_no_error(baseline: Baseline, tmp_path: Path) -> None:
    src = tmp_path / "tensor_cast" / "new_mod.py"
    src.parent.mkdir(parents=True)
    src.write_text("def fn():\n    pass\n", encoding="utf-8")
    cs = ChangeSet.build(new_source=("tensor_cast/new_mod.py",))
    exempt_baseline = Baseline(
        test_map=baseline.test_map,
        exemptions=(_sample_exemption("tensor_cast/new_mod.py", "fn"),),
        discovery=baseline.discovery,
        product_prefixes=baseline.product_prefixes,
    )
    plan = build_ci_gate_plan(tmp_path, cs, exempt_baseline)
    assert len(plan.blocking_errors) == 0
