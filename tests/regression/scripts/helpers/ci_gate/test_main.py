"""Tests for ci_gate.main — build_ci_gate_plan, _has_coverage_data."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from scripts.helpers.ci_gate.gate_policy import SourceExemption, default_test_discovery
from scripts.helpers.ci_gate.main import (
    _check_symbol_level_coverage,
    _has_coverage_data,
    build_ci_gate_plan,
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


# ---------------------------------------------------------------------------
# _has_coverage_data
# ---------------------------------------------------------------------------


def test_has_coverage_data_no_file(tmp_path: Path) -> None:
    assert _has_coverage_data(tmp_path / ".coverage") is False


def test_has_coverage_data_empty_file(tmp_path: Path) -> None:
    cov = tmp_path / ".coverage"
    cov.write_text("", encoding="utf-8")
    assert _has_coverage_data(cov) is False  # no measured files


def test_has_coverage_data_with_measured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cov = tmp_path / ".coverage"
    cov.write_text("", encoding="utf-8")

    class _FakeData:
        def read(self) -> None:
            pass

        def measured_files(self) -> list:
            return ["a.py"]

    monkeypatch.setattr("coverage.data.CoverageData", lambda _: _FakeData())
    assert _has_coverage_data(cov) is True


# ---------------------------------------------------------------------------
# _check_symbol_level_coverage
# ---------------------------------------------------------------------------


def test_check_symbol_level_coverage_returns_warnings_for_weak_symbols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.helpers.ci_gate import main as gate_main

    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text("def run():\n    x = 1\n    y = 2\n    z = 3\n", encoding="utf-8")

    monkeypatch.setattr(gate_main, "REPO_ROOT", tmp_path)

    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({1, 2, 3, 4})})

    class _FakeCoverageData:
        def read(self) -> None:
            pass

        def measured_files(self) -> list[str]:
            return ["cli/main.py"]

        def contexts_by_lineno(self, abs_path: str) -> dict:
            if "main.py" in abs_path:
                return {2: ["test_a|run"], 3: ["test_a|run"]}
            return {}

    monkeypatch.setattr("coverage.data.CoverageData", lambda _: _FakeCoverageData())

    cov_path = tmp_path / ".coverage"
    cov_path.write_text("", encoding="utf-8")

    warnings = _check_symbol_level_coverage(cs, cov_path, threshold=0.51)
    assert len(warnings) == 1
    assert "cli/main.py::run" in warnings[0]
    assert "51%" in warnings[0]


def test_check_symbol_level_coverage_returns_empty_when_no_coverage_data(
    tmp_path: Path,
) -> None:
    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({1})})
    warnings = _check_symbol_level_coverage(cs, tmp_path / ".coverage")
    assert warnings == ()


def test_check_symbol_level_coverage_no_warnings_when_well_covered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.helpers.ci_gate import main as gate_main

    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text("def run():\n    x = 1\n    y = 2\n", encoding="utf-8")

    monkeypatch.setattr(gate_main, "REPO_ROOT", tmp_path)

    cs = ChangeSet.build(modified_source={"cli/main.py": frozenset({1, 2})})

    class _FakeCoverageData:
        def read(self) -> None:
            pass

        def measured_files(self) -> list[str]:
            return ["cli/main.py"]

        def contexts_by_lineno(self, abs_path: str) -> dict:
            if "main.py" in abs_path:
                return {1: ["test_a|run"], 2: ["test_a|run"]}
            return {}

    monkeypatch.setattr("coverage.data.CoverageData", lambda _: _FakeCoverageData())

    cov_path = tmp_path / ".coverage"
    cov_path.write_text("", encoding="utf-8")

    warnings = _check_symbol_level_coverage(cs, cov_path, threshold=0.50)
    assert warnings == ()
