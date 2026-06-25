"""Tests for ci_gate.main orchestration — coverage-visible entrypoints."""

from __future__ import annotations

import inspect
import logging
from datetime import date

import pytest

import scripts.helpers.ci_gate.main as ci_gate_main
from scripts.helpers._config import Config
from scripts.helpers.ci_gate.diff import GitDiffResult
from scripts.helpers.ci_gate.main import _run_pytest, main
from scripts.helpers.ci_gate.models import Baseline, ChangeSet, TestExemption
from scripts.helpers.common.test_map_loader import TestMapFreshness
from tests.helpers.fake_subprocess import FakeCompleted
from tests.regression.scripts.helpers.conftest import default_ci_gate_policy


def _empty_diff() -> GitDiffResult:
    return GitDiffResult(line_map={}, entries=())


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def empty_baseline() -> Baseline:
    return Baseline(test_map={}, policy=default_ci_gate_policy())


def test_run_pytest_empty_targets_returns_zero_without_subprocess() -> None:
    assert _run_pytest([], marker="not npu") == 0


def test_run_pytest_invokes_subprocess_for_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> FakeCompleted:
        captured.append(cmd)
        return FakeCompleted(0, "", "")

    monkeypatch.setattr("scripts.helpers.ci_gate.main.subprocess.run", _fake_run)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.count_collected_tests",
        lambda *_args, **_kwargs: 2,
    )
    code = _run_pytest(
        ["tests/regression/scripts/helpers/ci_gate/test_errors.py"],
        marker="not npu and not nightly and not network",
    )
    assert code == 0
    assert captured
    run_cmd = captured[-1]
    assert "-o" in run_cmd
    assert "addopts=" in run_cmd
    assert "-m" in run_cmd
    assert "not npu and not nightly and not network" in run_cmd
    assert "-n" in run_cmd
    assert "pytest" in run_cmd


def test_run_pytest_node_targets_filter_via_collect_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_collect(targets: list[str], *, marker: str) -> tuple[str, ...]:
        captured["collect_targets"] = list(targets)
        captured["collect_marker"] = marker
        return ("tests/regression/cli/test_a.py::test_a",)

    def _fake_build(
        _python: str,
        run_targets: list[str],
        *,
        marker: str,
        collected_count: int,
        **_kwargs: object,
    ) -> list[str]:
        captured["run_targets"] = list(run_targets)
        captured["collected_count"] = collected_count
        captured["build_marker"] = marker
        return ["pytest"]

    monkeypatch.setattr("scripts.helpers.ci_gate.main.filter_collectable_node_ids", _fake_collect)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.build_pytest_cmd", _fake_build)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.subprocess.run",
        lambda *_args, **_kwargs: FakeCompleted(0, "", ""),
    )

    code = _run_pytest(
        [
            "tests/regression/cli/test_a.py::test_a",
            "tests/regression/cli/test_b.py::test_stale",
        ],
        marker="not npu",
    )
    assert code == 0
    assert captured["collect_targets"] == [
        "tests/regression/cli/test_a.py::test_a",
        "tests/regression/cli/test_b.py::test_stale",
    ]
    assert captured["collect_marker"] == "not npu"
    assert captured["run_targets"] == ["tests/regression/cli/test_a.py::test_a"]
    assert captured["collected_count"] == 1
    assert captured["build_marker"] == "not npu"


def test_run_pytest_stale_node_targets_skip_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess_calls: list[list[str]] = []

    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.filter_collectable_node_ids",
        lambda *_args, **_kwargs: (),
    )

    def _fake_run(cmd: list[str], **_kwargs: object) -> FakeCompleted:
        subprocess_calls.append(cmd)
        return FakeCompleted(0, "", "")

    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.subprocess.run",
        _fake_run,
    )

    code = _run_pytest(
        ["tests/regression/cli/test_old.py::test_renamed"],
        marker="not npu and not nightly and not network",
    )
    assert code == 0
    assert subprocess_calls == []


def test_run_pytest_stale_node_targets_log_skipped(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.filter_collectable_node_ids",
        lambda *_args, **_kwargs: ("tests/regression/cli/test_a.py::test_a",),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.subprocess.run",
        lambda *_args, **_kwargs: FakeCompleted(0, "", ""),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.build_pytest_cmd",
        lambda *_args, **_kwargs: ["pytest"],
    )

    with caplog.at_level(logging.INFO, logger="ci_gate"):
        _run_pytest(
            [
                "tests/regression/cli/test_a.py::test_a",
                "tests/regression/cli/test_b.py::test_stale",
            ],
            marker="not npu",
        )

    assert "Skipping non-collectable pytest node(s): tests/regression/cli/test_b.py::test_stale" in caplog.text


def test_run_pytest_adds_cov_args_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, tuple[str, ...]] = {}

    def _fake_build(
        *_args: object,
        extra_args: tuple[str, ...] = (),
        **_kwargs: object,
    ) -> list[str]:
        captured["extra_args"] = extra_args
        return ["pytest"]

    monkeypatch.setattr("scripts.helpers.ci_gate.main.build_pytest_cmd", _fake_build)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.subprocess.run",
        lambda *_args, **_kwargs: FakeCompleted(0, "", ""),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main._collected_count_for_targets",
        lambda *_args, **_kwargs: 1,
    )

    _run_pytest(["tests"], marker="not npu", use_cov=True)
    extra_args = captured["extra_args"]
    assert extra_args
    assert any(str(arg).startswith("--cov=") for arg in extra_args)
    assert "--cov-context=test" in extra_args


def test_run_pytest_cov_append_passes_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, ...]] = []

    def _fake_build(
        *_args: object,
        extra_args: tuple[str, ...] = (),
        **_kwargs: object,
    ) -> list[str]:
        captured.append(extra_args)
        return ["pytest"]

    monkeypatch.setattr("scripts.helpers.ci_gate.main.build_pytest_cmd", _fake_build)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.subprocess.run",
        lambda *_args, **_kwargs: FakeCompleted(0, "", ""),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main._collected_count_for_targets",
        lambda *_args, **_kwargs: 1,
    )

    _run_pytest(["tests"], marker="not npu", use_cov=True, cov_append=True)
    assert "--cov-append" in captured[-1]


def test_run_pytest_full_suite_marker_uses_not_npu_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> FakeCompleted:
        captured.append(cmd)
        return FakeCompleted(0, "", "")

    monkeypatch.setattr("scripts.helpers.ci_gate.main.subprocess.run", _fake_run)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.count_collected_tests",
        lambda *_args, **_kwargs: 1,
    )
    code = _run_pytest(["tests"], marker="not npu")
    assert code == 0
    run_cmd = captured[-1]
    assert run_cmd.count("-m") >= 1
    assert "not npu" in run_cmd
    assert "not nightly and not network" not in run_cmd


def test_main_passes_when_no_gate_work(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
    empty_baseline: Baseline,
) -> None:
    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.resolve_base_ref",
        lambda _root, _branch: "abc" * 10,
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.validate_gate_policy_if_changed",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.assess_test_map_freshness",
        lambda *_args: TestMapFreshness(),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.load_baseline",
        lambda *_args: (empty_baseline, "a" * 40),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.fetch_diff", lambda *_args: _empty_diff())
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.classify_changes",
        lambda *_args: ChangeSet.build(),
    )

    assert main() == 0


def test_main_returns_one_on_unmapped_modified_source(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
    empty_baseline: Baseline,
) -> None:
    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.resolve_base_ref",
        lambda _root, _branch: "abc" * 10,
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.validate_gate_policy_if_changed",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.assess_test_map_freshness",
        lambda *_args: TestMapFreshness(),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.load_baseline",
        lambda *_args: (empty_baseline, "a" * 40),
    )
    monkeypatch.setattr("scripts.helpers.ci_gate.main.fetch_diff", lambda *_args: _empty_diff())
    main_line = inspect.getsourcelines(ci_gate_main.main)[1] + 1
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.classify_changes",
        lambda *_args: ChangeSet.build(
            modified_source={"scripts/helpers/ci_gate/main.py": frozenset({main_line})},
        ),
    )

    assert main() == 1


def test_main_skips_all_exempt_changed_test_file(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.rules.collect_all_test_node_ids",
        lambda targets, **_kwargs: tuple(f"{path}::test_case" for path in targets),
    )
    pytest_calls: list[list[str]] = []
    policy = default_ci_gate_policy()
    baseline = Baseline(
        test_map={},
        policy=policy.__class__(
            sources=policy.sources,
            tests=policy.tests,
            configs=policy.configs,
            source_exemptions=(),
            test_exemptions=(
                TestExemption(
                    test_id="tests/regression/cli/test_omitted.py::test_case",
                    reason="x",
                    applicant="a",
                    approver="fangkai",
                    deadline=date(2099, 12, 31),
                ),
            ),
            approvers=policy.approvers,
        ),
    )

    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.resolve_base_ref",
        lambda _root, _branch: "abc" * 10,
    )
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
            new_test=("tests/regression/cli/test_omitted.py",),
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

    assert main() == 0
    assert pytest_calls == []
