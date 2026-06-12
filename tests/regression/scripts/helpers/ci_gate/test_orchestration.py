"""Tests for ci_gate.main orchestration — coverage-visible entrypoints."""

from __future__ import annotations

import inspect
import logging
from datetime import date
from pathlib import Path

import pytest

import scripts.helpers.ci_gate.main as ci_gate_main
from scripts.helpers._config import Config
from scripts.helpers.ci_gate.gate_policy import TestExemption, default_test_discovery
from scripts.helpers.ci_gate.main import (
    _run_new_tests_and_build_map,
    _run_pytest,
    main,
)
from scripts.helpers.ci_gate.models import Baseline, ChangeSet, CiGatePlan
from tests.helpers.fake_subprocess import FakeCompleted

_TEST_FILE = "tests/regression/scripts/helpers/ci_gate/test_errors.py"
_TEST_NODE = f"{_TEST_FILE}::test_case"


def _node_ids_for(*files: str) -> tuple[str, ...]:
    return tuple(f"{path}::test_case" for path in files)


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
    return Baseline(
        test_map={},
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


def test_run_pytest_empty_targets_returns_zero_without_subprocess() -> None:
    assert _run_pytest([], marker="not npu") == 0


def test_run_pytest_invokes_subprocess_for_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> FakeCompleted:
        captured.append(cmd)
        return FakeCompleted(0, "", "")

    monkeypatch.setattr("scripts.helpers.ci_gate.main.subprocess.run", _fake_run)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.count_collected_tests", lambda *_args, **_kwargs: 2)
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


def test_run_pytest_full_suite_marker_uses_not_npu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> FakeCompleted:
        captured.append(cmd)
        return FakeCompleted(0, "", "")

    monkeypatch.setattr("scripts.helpers.ci_gate.main.subprocess.run", _fake_run)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.count_collected_tests", lambda *_args, **_kwargs: 1)
    code = _run_pytest(["tests"], marker="not npu")
    assert code == 0
    run_cmd = captured[-1]
    assert run_cmd.count("-m") >= 1
    assert "not npu" in run_cmd
    assert "not nightly and not network" not in run_cmd


def test_run_new_tests_and_build_map_returns_collected_map(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_cmds: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> FakeCompleted:
        captured_cmds.append(cmd)
        return FakeCompleted(0, "", "")

    monkeypatch.setattr("scripts.helpers.ci_gate.main.subprocess.run", _fake_run)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.collect_test_node_ids",
        lambda targets, **_kwargs: _node_ids_for(*targets),
    )
    expected = {
        "scripts/helpers/ci_gate/main.py": {
            "main": [
                "tests/regression/scripts/helpers/ci_gate/test_orchestration.py::test_main_passes_when_no_gate_work"
            ]
        }
    }
    captured_map_marker: list[str] = []

    def _fake_collect_test_map(**kwargs: object) -> dict[str, dict[str, list[str]]]:
        captured_map_marker.append(str(kwargs["marker_expr"]))
        return expected

    monkeypatch.setattr("scripts.helpers.ci_gate.main.collect_test_map", _fake_collect_test_map)
    result, ran = _run_new_tests_and_build_map(
        (_TEST_FILE,),
        "not nightly and not network",
        roots=("cli/", "scripts/"),
    )
    assert result == expected
    assert ran is True
    assert captured_map_marker == ["not nightly and not network"]
    assert captured_cmds
    run_cmd = captured_cmds[-1]
    assert "-o" in run_cmd
    assert "addopts=" in run_cmd
    assert "not npu" in run_cmd
    assert "not nightly and not network" not in run_cmd
    assert _TEST_NODE in run_cmd


def test_run_new_tests_and_build_map_skips_pytest_when_all_nodes_exempt(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    exemption = TestExemption(
        test_id=_TEST_NODE,
        reason="x",
        applicant="a",
        approver="fangkai",
        deadline=date(2099, 12, 31),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.collect_test_node_ids",
        lambda targets, **_kwargs: _node_ids_for(*targets),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("pytest should not run"),
    )

    with caplog.at_level(logging.INFO, logger="ci_gate"):
        result, ran = _run_new_tests_and_build_map(
            (_TEST_FILE,),
            "not nightly",
            roots=("cli/",),
            test_exemptions=(exemption,),
        )

    assert result == {}
    assert ran is False
    assert "all 1 node(s) listed in exemptions.tests" in caplog.text
    assert "no runnable nodes" in caplog.text


def test_run_new_tests_and_build_map_exits_on_pytest_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.collect_test_node_ids",
        lambda targets, **_kwargs: _node_ids_for(*targets),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.subprocess.run",
        lambda *_args, **_kwargs: FakeCompleted(1, "", "fail"),
    )
    with pytest.raises(SystemExit) as exc_info:
        _run_new_tests_and_build_map(
            (_TEST_FILE,),
            "not nightly",
            roots=("cli/",),
        )
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "exemptions.tests" in captured.out
    assert _TEST_NODE in captured.out


def test_run_new_tests_and_build_map_executes_nightly_marked_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 0 targets nightly-marked files with ``-m not npu`` only (stub subprocess)."""
    test_file = tmp_path / "test_nightly_gate_phase0.py"
    test_file.write_text(
        "import pytest\n\n@pytest.mark.nightly\ndef test_phase0_runs():\n    assert True\n",
        encoding="utf-8",
    )
    node_id = f"{test_file}::test_phase0_runs"
    captured_cmds: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> FakeCompleted:
        captured_cmds.append(cmd)
        return FakeCompleted(0, "", "")

    monkeypatch.setattr("scripts.helpers.ci_gate.main.subprocess.run", _fake_run)
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.collect_test_node_ids",
        lambda targets, **_kwargs: (node_id,) if str(test_file) in targets else (),
    )
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.collect_test_map",
        lambda **_kwargs: {},
    )
    _run_new_tests_and_build_map(
        (str(test_file),),
        "not nightly and not network",
        roots=("cli/",),
    )
    assert captured_cmds
    run_cmd = captured_cmds[-1]
    assert "-o" in run_cmd
    assert "addopts=" in run_cmd
    assert "not npu" in run_cmd
    assert "not nightly and not network" not in run_cmd
    assert node_id in run_cmd


def test_main_passes_all_test_files_to_phase0(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
) -> None:
    baseline = Baseline(
        test_map={},
        exemptions=(),
        test_exemptions=(
            TestExemption(
                test_id="tests/regression/cli/test_omitted.py::test_omitted",
                reason="x",
                applicant="a",
                approver="fangkai",
                deadline=date(2099, 12, 31),
            ),
        ),
        discovery=default_test_discovery(),
        roots=("cli/", "scripts/"),
    )
    captured: list[tuple[str, ...]] = []

    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda _root, _branch: "abc" * 10)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.validate_gate_policy_if_changed", lambda *_args: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.load_baseline", lambda *_args: baseline)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.fetch_diff_line_map", lambda *_args: {})
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.classify_changes",
        lambda *_args: ChangeSet.build(
            new_test=(
                "tests/regression/cli/test_omitted.py",
                "tests/regression/cli/test_run.py",
            ),
        ),
    )

    def _fake_run_new_tests(
        tests: tuple[str, ...],
        _marker: str,
        *,
        roots: tuple[str, ...] = (),
        test_exemptions: tuple[TestExemption, ...] = (),
    ) -> tuple[dict[str, dict[str, list[str]]], bool]:
        captured.append(tests)
        assert test_exemptions == baseline.test_exemptions
        return {}, False

    monkeypatch.setattr("scripts.helpers.ci_gate.main._run_new_tests_and_build_map", _fake_run_new_tests)

    def _fake_build_plan(*_args, **_kwargs: object) -> CiGatePlan:
        return CiGatePlan(
            blocking_errors=(),
            deleted_source_tests=frozenset(),
            incremental_tests=frozenset(),
            full_suite=False,
        )

    monkeypatch.setattr("scripts.helpers.ci_gate.main.build_ci_gate_plan", _fake_build_plan)

    assert main() == 0
    assert captured == [
        (
            "tests/regression/cli/test_omitted.py",
            "tests/regression/cli/test_run.py",
        )
    ]


def test_main_passes_when_no_gate_work(
    monkeypatch: pytest.MonkeyPatch,
    gate_cfg: Config,
    empty_baseline: Baseline,
) -> None:
    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda _root, _branch: "abc" * 10)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.validate_gate_policy_if_changed", lambda *_args: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.load_baseline", lambda *_args: empty_baseline)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.fetch_diff_line_map", lambda *_args: {})
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
    """Hit main() blocking path when modified product symbols lack test_map entries."""
    monkeypatch.setattr("scripts.helpers.ci_gate.main.Config.from_env", lambda: gate_cfg)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.resolve_base_ref", lambda _root, _branch: "abc" * 10)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.validate_gate_policy_if_changed", lambda *_args: None)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.load_baseline", lambda *_args: empty_baseline)
    monkeypatch.setattr("scripts.helpers.ci_gate.main.fetch_diff_line_map", lambda *_args: {})
    main_line = inspect.getsourcelines(ci_gate_main.main)[1] + 1
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.classify_changes",
        lambda *_args: ChangeSet.build(
            modified_source={"scripts/helpers/ci_gate/main.py": frozenset({main_line})},
        ),
    )

    assert main() == 1
