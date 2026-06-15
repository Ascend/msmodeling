"""Extra regression tests for scripts.helpers.ci_gate.main."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pytest

from scripts.helpers._config import Config, ConfigError
from scripts.helpers.ci_gate.diff import GitDiffResult
from scripts.helpers.ci_gate.gate_policy import TestExemption, default_test_discovery
from scripts.helpers.ci_gate.main import (
    _log_blocking_errors,
    _log_execution_plan,
    _remap_renamed_sources,
    main,
)
from scripts.helpers.ci_gate.models import Baseline, ChangeSet, CiGatePlan, ExecutionPlan, GateError, TestRunWave


def _empty_diff() -> GitDiffResult:
    return GitDiffResult(line_map={}, entries=())


@pytest.fixture
def gate_cfg() -> Config:
    return Config(
        test_map_path="/tmp/test_map.json",
        base_branch="develop",
        line_threshold=60.0,
        branch_threshold=40.0,
        benchmark_parallel=False,
        feishu_webhook_url="",
        msmodeling_cache=".msmodeling_cache",
        weights_prune=True,
    )


@pytest.fixture
def baseline() -> Baseline:
    return Baseline(
        test_map={
            "cli/old.py": {
                "run": ["tests/regression/cli/test_old.py::test_old"],
            },
        },
        exemptions=(),
        discovery=default_test_discovery(),
        test_exemptions=(),
        roots=(
            "cli/",
            "tensor_cast/",
            "serving_cast/",
            "web_ui/",
            "scripts/",
            "tools/",
        ),
    )


def test_remap_renamed_sources_moves_map_entries() -> None:
    test_map = {"cli/old.py": {"run": ["tests/regression/cli/test_old.py::test_old"]}}

    remapped = _remap_renamed_sources(test_map, (("cli/old.py", "cli/new.py", 100),))

    assert "cli/old.py" not in remapped
    assert remapped == {"cli/new.py": {"run": ["tests/regression/cli/test_old.py::test_old"]}}


def test_remap_renamed_sources_skips_partial_renames() -> None:
    test_map = {"cli/old.py": {"run": ["tests/regression/cli/test_old.py::test_old"]}}

    remapped = _remap_renamed_sources(test_map, (("cli/old.py", "cli/new.py", 95),))

    assert remapped == test_map


def test_log_blocking_errors_emits_category_summary(caplog: pytest.LogCaptureFixture) -> None:
    errors = (
        GateError(category="deleted_source", path="cli/old.py"),
        GateError(category="deleted_source", path="cli/other.py"),
        GateError(category="modified_source", path="cli/main.py"),
    )

    with caplog.at_level(logging.ERROR, logger="ci_gate"):
        _log_blocking_errors(logging.getLogger("ci_gate"), errors)

    assert "deleted_source=2" in caplog.text
    assert "modified_source=1" in caplog.text
    assert "Phase" not in caplog.text


def test_log_execution_plan_logs_reason_counts(caplog: pytest.LogCaptureFixture) -> None:
    execution = ExecutionPlan(
        full_suite=False,
        waves=(
            TestRunWave(
                targets=("tests/regression/cli/test_a.py::test_a",),
                marker="not npu",
            ),
        ),
        reasons={
            "tests/regression/cli/test_a.py::test_a": "new or changed test file",
            "tests/regression/cli/test_b.py::test_b": "changed product file mapped regression",
        },
    )

    with caplog.at_level(logging.INFO, logger="ci_gate"):
        _log_execution_plan(logging.getLogger("ci_gate"), execution)

    assert "new or changed test file" in caplog.text
    assert "changed product file mapped regression" in caplog.text
    assert "Phase" not in caplog.text


def test_main_returns_one_on_resolve_base_ref_error(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
) -> None:
    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.setup_logger", lambda: logging.getLogger("ci_gate"))
    monkeypatch.setattr("scripts.helpers.ci_gate.main.log_env_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.resolve_base_ref", lambda *_args: (_ for _ in ()).throw(ConfigError("base ref"))
    )

    assert main() == 1


def test_main_returns_one_on_load_baseline_error(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
) -> None:
    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.setup_logger", lambda: logging.getLogger("ci_gate"))
    monkeypatch.setattr("scripts.helpers.ci_gate.main.log_env_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda *_args: "abc" * 10)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.validate_gate_policy_if_changed", lambda *_args: None)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.load_baseline", lambda *_args: (_ for _ in ()).throw(ConfigError("baseline"))
    )

    assert main() == 1


def test_main_remaps_before_plan_without_phase0_merge(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
    baseline: Baseline,
) -> None:
    captured: dict[str, Baseline] = {}

    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.setup_logger", lambda: logging.getLogger("ci_gate"))
    monkeypatch.setattr("scripts.helpers.ci_gate.main.log_env_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda *_args: "abc" * 10)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.validate_gate_policy_if_changed", lambda *_args: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.load_baseline", lambda *_args: (baseline, "a" * 40))
    monkeypatch.setattr("scripts.helpers.ci_gate.main.validate_test_map_freshness", lambda *_args: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.gate_policy_changed_in_diff", lambda *_args: False)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.fetch_diff", lambda *_args: _empty_diff())
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.classify_changes",
        lambda *_args: ChangeSet.build(
            new_test=("tests/regression/cli/test_new.py",),
            renames=(("cli/old.py", "cli/new.py", 100),),
        ),
    )

    def _fake_build_plan(
        _repo_root: Path,
        _changes: ChangeSet,
        new_baseline: Baseline,
        **_kwargs: object,
    ) -> CiGatePlan:
        captured["baseline"] = new_baseline
        return CiGatePlan(
            blocking_errors=(),
            deleted_source_tests=frozenset(),
            changed_test_nodes=frozenset(),
            regression_tests=frozenset(),
            full_suite=False,
        )

    monkeypatch.setattr("scripts.helpers.ci_gate.main.build_ci_gate_plan", _fake_build_plan)

    assert main() == 0
    assert "cli/new.py" in captured["baseline"].test_map
    assert captured["baseline"].test_map["cli/new.py"]["run"] == ["tests/regression/cli/test_old.py::test_old"]


def test_main_runs_deleted_source_guards_in_union(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
    baseline: Baseline,
) -> None:
    pytest_calls: list[tuple[list[str], str]] = []

    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.setup_logger", lambda: logging.getLogger("ci_gate"))
    monkeypatch.setattr("scripts.helpers.ci_gate.main.log_env_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda *_args: "abc" * 10)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.validate_gate_policy_if_changed", lambda *_args: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.load_baseline", lambda *_args: (baseline, "a" * 40))
    monkeypatch.setattr("scripts.helpers.ci_gate.main.validate_test_map_freshness", lambda *_args: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.gate_policy_changed_in_diff", lambda *_args: False)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.fetch_diff", lambda *_args: _empty_diff())
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.classify_changes",
        lambda *_args: ChangeSet.build(del_source=("cli/old.py",)),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.build_ci_gate_plan",
        lambda *_args, **_kwargs: CiGatePlan(
            blocking_errors=(),
            deleted_source_tests=frozenset({"tests/regression/cli/test_old.py::test_old"}),
            changed_test_nodes=frozenset(),
            regression_tests=frozenset(),
            full_suite=False,
        ),
    )

    def _fake_run_pytest(
        targets: list[str],
        *,
        marker: str,
        use_cov: bool = False,
        cov_append: bool = False,
    ) -> int:
        pytest_calls.append((targets, marker))
        return 1

    monkeypatch.setattr("scripts.helpers.ci_gate.main._run_pytest", _fake_run_pytest)

    assert main() == 1
    assert pytest_calls == [
        (
            ["tests/regression/cli/test_old.py::test_old"],
            "not npu and not nightly and not network",
        )
    ]


def test_main_uses_full_suite_targets_when_config_changes(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
    baseline: Baseline,
) -> None:
    pytest_calls: list[tuple[list[str], str]] = []

    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.setup_logger", lambda: logging.getLogger("ci_gate"))
    monkeypatch.setattr("scripts.helpers.ci_gate.main.log_env_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda *_args: "abc" * 10)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.validate_gate_policy_if_changed", lambda *_args: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.load_baseline", lambda *_args: (baseline, "a" * 40))
    monkeypatch.setattr("scripts.helpers.ci_gate.main.validate_test_map_freshness", lambda *_args: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.gate_policy_changed_in_diff", lambda *_args: False)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.fetch_diff", lambda *_args: _empty_diff())
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.classify_changes",
        lambda *_args: ChangeSet.build(config=("pyproject.toml",)),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.build_ci_gate_plan",
        lambda *_args, **_kwargs: CiGatePlan(
            blocking_errors=(),
            deleted_source_tests=frozenset(),
            changed_test_nodes=frozenset(),
            regression_tests=frozenset(),
            full_suite=True,
        ),
    )

    def _fake_run_pytest(
        targets: list[str],
        *,
        marker: str,
        use_cov: bool = False,
        cov_append: bool = False,
    ) -> int:
        pytest_calls.append((targets, marker))
        return 0

    monkeypatch.setattr("scripts.helpers.ci_gate.main._run_pytest", _fake_run_pytest)

    assert main() == 0
    assert pytest_calls == [(["tests"], "not npu")]


def test_main_skips_exempt_regression_targets(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
    baseline: Baseline,
) -> None:
    pytest_calls: list[list[str]] = []

    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.setup_logger", lambda: logging.getLogger("ci_gate"))
    monkeypatch.setattr("scripts.helpers.ci_gate.main.log_env_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda *_args: "abc" * 10)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.validate_gate_policy_if_changed", lambda *_args: None)
    exempt_baseline = baseline.__class__(
        test_map=baseline.test_map,
        exemptions=baseline.exemptions,
        test_exemptions=(
            TestExemption(
                test_id="tests/regression/cli/test_new.py::test_new",
                reason="x",
                applicant="a",
                approver="fangkai",
                deadline=date(2099, 12, 31),
            ),
        ),
        discovery=baseline.discovery,
        roots=baseline.roots,
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.load_baseline", lambda *_args: (exempt_baseline, "a" * 40))
    monkeypatch.setattr("scripts.helpers.ci_gate.main.validate_test_map_freshness", lambda *_args: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.gate_policy_changed_in_diff", lambda *_args: False)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.fetch_diff", lambda *_args: _empty_diff())
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.classify_changes",
        lambda *_args: ChangeSet.build(modified_source={"cli/main.py": frozenset({1})}),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.build_ci_gate_plan",
        lambda *_args, **_kwargs: CiGatePlan(
            blocking_errors=(),
            deleted_source_tests=frozenset(),
            changed_test_nodes=frozenset(),
            regression_tests=frozenset({"tests/regression/cli/test_new.py::test_new"}),
            full_suite=False,
        ),
    )

    def _fake_run_pytest(
        targets: list[str],
        *,
        marker: str,
        use_cov: bool = False,
        cov_append: bool = False,
    ) -> int:
        pytest_calls.append(targets)
        return 0

    monkeypatch.setattr("scripts.helpers.ci_gate.main._run_pytest", _fake_run_pytest)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.build_coverage_mapping_errors", lambda *_args, **_kwargs: ())

    assert main() == 0
    assert pytest_calls == []


def test_main_runs_union_targets_when_available(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
    baseline: Baseline,
) -> None:
    pytest_calls: list[tuple[list[str], str]] = []

    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.setup_logger", lambda: logging.getLogger("ci_gate"))
    monkeypatch.setattr("scripts.helpers.ci_gate.main.log_env_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda *_args: "abc" * 10)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.validate_gate_policy_if_changed", lambda *_args: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.load_baseline", lambda *_args: (baseline, "a" * 40))
    monkeypatch.setattr("scripts.helpers.ci_gate.main.validate_test_map_freshness", lambda *_args: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.gate_policy_changed_in_diff", lambda *_args: False)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.fetch_diff", lambda *_args: _empty_diff())
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.classify_changes",
        lambda *_args: ChangeSet.build(modified_source={"cli/main.py": frozenset({1})}),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.build_ci_gate_plan",
        lambda *_args, **_kwargs: CiGatePlan(
            blocking_errors=(),
            deleted_source_tests=frozenset(),
            changed_test_nodes=frozenset(),
            regression_tests=frozenset({"tests/regression/cli/test_new.py::test_new"}),
            full_suite=False,
        ),
    )

    def _fake_run_pytest(
        targets: list[str],
        *,
        marker: str,
        use_cov: bool = False,
        cov_append: bool = False,
    ) -> int:
        pytest_calls.append((targets, marker))
        return 0

    monkeypatch.setattr("scripts.helpers.ci_gate.main._run_pytest", _fake_run_pytest)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.build_coverage_mapping_errors", lambda *_args, **_kwargs: ())

    assert main() == 0
    assert pytest_calls == [
        (
            ["tests/regression/cli/test_new.py::test_new"],
            "not npu and not nightly and not network",
        )
    ]
