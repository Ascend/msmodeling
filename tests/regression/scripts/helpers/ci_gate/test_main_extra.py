"""Extra regression tests for scripts.helpers.ci_gate.main."""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.helpers._config import Config, ConfigError
from scripts.helpers.ci_gate.diff import GitDiffResult
from scripts.helpers.ci_gate.gate_policy import TestExemption
from scripts.helpers.ci_gate.main import (
    _log_blocking_errors,
    _log_execution_plan,
    main,
)
from scripts.helpers.ci_gate.models import (
    Baseline,
    ChangeSet,
    CiGatePlan,
    ExecutionPlan,
    GateError,
    TestRunWave,
)
from scripts.helpers.common.test_map_loader import TestMapFreshness
from tests.regression.scripts.helpers.conftest import default_ci_gate_policy


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
            "tests/regression/cli/test_old.py::test_old": {
                "cli/old.py": ["run"],
            },
            "tests/regression/cli/test_old_guard.py::test_guard": {
                "cli/old.py": ["run"],
            },
        },
        policy=default_ci_gate_policy(),
    )


def test_log_blocking_errors_emits_category_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    errors = (
        GateError(category="deleted_test", path="tests/regression/cli/test_old.py"),
        GateError(category="deleted_test", path="tests/regression/cli/test_other.py"),
        GateError(category="modified_source", path="cli/main.py", symbol="run"),
    )

    with caplog.at_level(logging.ERROR, logger="ci_gate"):
        _log_blocking_errors(logging.getLogger("ci_gate"), errors)

    assert "deleted_test=2" in caplog.text
    assert "modified_source=1" in caplog.text
    assert "Phase" not in caplog.text


def test_log_execution_plan_logs_reason_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.setup_logger",
        lambda: logging.getLogger("ci_gate"),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.log_env_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.resolve_base_ref",
        lambda *_args: (_ for _ in ()).throw(ConfigError("base ref")),
    )

    assert main() == 1


def test_main_returns_one_on_load_baseline_error(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
) -> None:
    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.setup_logger",
        lambda: logging.getLogger("ci_gate"),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.log_env_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda *_args: "abc" * 10)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.validate_gate_policy_if_changed",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.load_baseline",
        lambda *_args: (_ for _ in ()).throw(ConfigError("baseline")),
    )

    assert main() == 1


def test_main_falls_back_to_full_suite_when_test_map_is_stale(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
    baseline: Baseline,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pytest_calls: list[tuple[list[str], str]] = []

    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.setup_logger",
        lambda: logging.getLogger("ci_gate"),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.log_env_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda *_args: "abc" * 10)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.validate_gate_policy_if_changed",
        lambda *_args: None,
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.load_baseline", lambda *_args: (baseline, "a" * 40))
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.assess_test_map_freshness",
        lambda *_args: TestMapFreshness(
            warn_message="test_map: built_from_commit 3b6dbc8b7fb2 is behind merge-base 5cf36c7ad92d"
        ),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.fetch_diff", lambda *_args: _empty_diff())
    monkeypatch.setattr("scripts.helpers.ci_gate.main.classify_changes", lambda *_args: ChangeSet.build())

    def _fake_run_pytest(
        targets: list[str],
        *,
        marker: str | None,
        use_cov: bool = False,
        cov_append: bool = False,
    ) -> int:
        pytest_calls.append((targets, marker))
        return 0

    monkeypatch.setattr("scripts.helpers.ci_gate.main._run_pytest", _fake_run_pytest)

    with caplog.at_level(logging.WARNING, logger="ci_gate"):
        assert main() == 0

    assert "falling back to the full test suite" in caplog.text
    assert pytest_calls == [(["tests"], "not npu and not nightly and not network")]


def test_main_returns_one_on_non_stale_test_map_error(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
    baseline: Baseline,
) -> None:
    pytest_calls: list[list[str]] = []

    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.setup_logger",
        lambda: logging.getLogger("ci_gate"),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.log_env_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda *_args: "abc" * 10)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.validate_gate_policy_if_changed",
        lambda *_args: None,
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.load_baseline", lambda *_args: (baseline, None))
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.assess_test_map_freshness",
        lambda *_args: TestMapFreshness(
            block_message="test_map: built_from_commit is required; rebuild test_map via nightly or build_test_map"
        ),
    )

    def _fake_run_pytest(
        targets: list[str],
        *,
        marker: str | None,
        use_cov: bool = False,
        cov_append: bool = False,
    ) -> int:
        pytest_calls.append(targets)
        return 0

    monkeypatch.setattr("scripts.helpers.ci_gate.main._run_pytest", _fake_run_pytest)

    assert main() == 1
    assert pytest_calls == []


def test_main_passes_baseline_unchanged_to_plan(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
    baseline: Baseline,
) -> None:
    captured: dict[str, Baseline] = {}

    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.setup_logger",
        lambda: logging.getLogger("ci_gate"),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.log_env_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda *_args: "abc" * 10)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.validate_gate_policy_if_changed",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.load_baseline",
        lambda *_args: (baseline, "a" * 40),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.assess_test_map_freshness",
        lambda *_args: TestMapFreshness(),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.fetch_diff", lambda *_args: _empty_diff())
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.classify_changes",
        lambda *_args: ChangeSet.build(
            new_test=("tests/regression/cli/test_new.py",),
            del_source=("cli/old.py",),
            new_source=("cli/new.py",),
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
            deleted_source_tests=frozenset(),
            changed_test_nodes=frozenset(),
            regression_tests=frozenset(),
            full_suite=False,
        )

    monkeypatch.setattr("scripts.helpers.ci_gate.main.build_ci_gate_plan", _fake_build_plan)

    assert main() == 0
    assert captured["baseline"].test_map == baseline.test_map


def test_main_runs_deleted_source_guards_in_union(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
    baseline: Baseline,
) -> None:
    pytest_calls: list[tuple[list[str], str]] = []

    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.setup_logger",
        lambda: logging.getLogger("ci_gate"),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.log_env_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda *_args: "abc" * 10)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.validate_gate_policy_if_changed",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.load_baseline",
        lambda *_args: (baseline, "a" * 40),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.assess_test_map_freshness",
        lambda *_args: TestMapFreshness(),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.fetch_diff", lambda *_args: _empty_diff())
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.classify_changes",
        lambda *_args: ChangeSet.build(del_source=("cli/old.py",)),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.build_ci_gate_plan",
        lambda *_args, **_kwargs: CiGatePlan(
            deleted_source_tests=frozenset(
                {
                    "tests/regression/cli/test_old.py::test_old",
                    "tests/regression/cli/test_old_guard.py::test_guard",
                }
            ),
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
    assert len(pytest_calls) == 1
    targets, marker = pytest_calls[0]
    assert sorted(targets) == [
        "tests/regression/cli/test_old.py::test_old",
        "tests/regression/cli/test_old_guard.py::test_guard",
    ]
    assert marker == "not npu and not nightly and not network"


def test_main_uses_full_suite_targets_when_config_changes(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
    baseline: Baseline,
) -> None:
    pytest_calls: list[tuple[list[str], str]] = []

    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.setup_logger",
        lambda: logging.getLogger("ci_gate"),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.log_env_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda *_args: "abc" * 10)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.validate_gate_policy_if_changed",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.load_baseline",
        lambda *_args: (baseline, "a" * 40),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.assess_test_map_freshness",
        lambda *_args: TestMapFreshness(),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.fetch_diff", lambda *_args: _empty_diff())
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.classify_changes",
        lambda *_args: ChangeSet.build(config=("pyproject.toml",)),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.build_ci_gate_plan",
        lambda *_args, **_kwargs: CiGatePlan(
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
    assert pytest_calls == [(["tests"], "not npu and not nightly and not network")]


def test_main_skips_exempt_regression_targets(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
    baseline: Baseline,
) -> None:
    pytest_calls: list[list[str]] = []

    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.setup_logger",
        lambda: logging.getLogger("ci_gate"),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.log_env_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda *_args: "abc" * 10)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.validate_gate_policy_if_changed",
        lambda *_args: None,
    )
    exempt_baseline = baseline.__class__(
        test_map=baseline.test_map,
        policy=baseline.policy.__class__(
            sources=baseline.policy.sources,
            tests=baseline.policy.tests,
            configs=baseline.policy.configs,
            source_exemptions=baseline.policy.source_exemptions,
            test_exemptions=(
                TestExemption(
                    test_id="tests/regression/cli/test_new.py::test_new",
                    reason="x",
                    applicant="a",
                    approver="fangkai",
                    deadline=date(2099, 12, 31),
                ),
            ),
            approvers=baseline.policy.approvers,
        ),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.load_baseline",
        lambda *_args: (exempt_baseline, "a" * 40),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.assess_test_map_freshness",
        lambda *_args: TestMapFreshness(),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.fetch_diff", lambda *_args: _empty_diff())
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.classify_changes",
        lambda *_args: ChangeSet.build(modified_source={"cli/main.py": frozenset({1})}),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.build_ci_gate_plan",
        lambda *_args, **_kwargs: CiGatePlan(
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
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.build_coverage_mapping_errors",
        lambda *_args, **_kwargs: (),
    )

    assert main() == 0
    assert pytest_calls == []


def test_main_runs_union_targets_when_available(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
    baseline: Baseline,
) -> None:
    pytest_calls: list[tuple[list[str], str]] = []

    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.setup_logger",
        lambda: logging.getLogger("ci_gate"),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.log_env_audit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda *_args: "abc" * 10)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.validate_gate_policy_if_changed",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.load_baseline",
        lambda *_args: (baseline, "a" * 40),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.assess_test_map_freshness",
        lambda *_args: TestMapFreshness(),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.fetch_diff", lambda *_args: _empty_diff())
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.classify_changes",
        lambda *_args: ChangeSet.build(modified_source={"cli/main.py": frozenset({1})}),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.build_ci_gate_plan",
        lambda *_args, **_kwargs: CiGatePlan(
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
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.build_coverage_mapping_errors",
        lambda *_args, **_kwargs: (),
    )

    assert main() == 0
    assert pytest_calls == [
        (
            ["tests/regression/cli/test_new.py::test_new"],
            "not npu and not nightly and not network",
        )
    ]
