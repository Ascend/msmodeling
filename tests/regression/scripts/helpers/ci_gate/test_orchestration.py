"""Tests for ci_gate.main orchestration — coverage-visible entrypoints."""

from __future__ import annotations

import pytest
from scripts.helpers._config import Config
from scripts.helpers.ci_gate.gate_policy import default_test_discovery
from scripts.helpers.ci_gate.main import (
    _run_new_tests_and_build_map,
    _run_pytest,
    main,
)
from scripts.helpers.ci_gate.models import Baseline, ChangeSet
from tests.helpers.fake_subprocess import FakeCompleted


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
        discovery=default_test_discovery(),
        product_prefixes=(
            "cli/",
            "tensor_cast/",
            "serving_cast/",
            "web_ui/",
            "scripts/",
        ),
    )


def test_run_pytest_empty_targets_returns_zero_without_subprocess() -> None:
    assert _run_pytest([]) == 0


def test_run_pytest_invokes_subprocess_for_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> FakeCompleted:
        captured.append(cmd)
        return FakeCompleted(0, "", "")

    monkeypatch.setattr("scripts.helpers.ci_gate.main.subprocess.run", _fake_run)
    code = _run_pytest(["tests/regression/scripts/helpers/ci_gate/test_errors.py"])
    assert code == 0
    assert captured
    assert "-m" in captured[0]
    assert "pytest" in captured[0]


def test_run_new_tests_and_build_map_returns_collected_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.subprocess.run",
        lambda *_args, **_kwargs: FakeCompleted(0, "", ""),
    )
    expected = {
        "scripts/helpers/ci_gate/main.py": {
            "main": [
                "tests/regression/scripts/helpers/ci_gate/test_orchestration.py::test_main_passes_when_no_gate_work"
            ]
        }
    }
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.collect_test_map",
        lambda **_kwargs: expected,
    )
    result = _run_new_tests_and_build_map(
        ("tests/regression/scripts/helpers/ci_gate/test_errors.py",),
        "not npu and not nightly and not network",
    )
    assert result == expected


def test_run_new_tests_and_build_map_exits_on_pytest_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.subprocess.run",
        lambda *_args, **_kwargs: FakeCompleted(1, "", "fail"),
    )
    with pytest.raises(SystemExit) as exc_info:
        _run_new_tests_and_build_map(("tests/regression/scripts/helpers/ci_gate/test_errors.py",), "not npu")
    assert exc_info.value.code == 1


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
    monkeypatch.setattr(
        "scripts.helpers.ci_gate.main.classify_changes",
        lambda *_args: ChangeSet.build(
            modified_source={"scripts/helpers/ci_gate/main.py": frozenset({179})},
        ),
    )

    assert main() == 1
