"""Regression tests for scripts.helpers.build.run_test."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.helpers.build import run_test as run_test_mod
from scripts.helpers.build.main import main
from scripts.helpers.build.run_test import run_test
from tests.helpers.cli_runner import run_cli_main
from tests.regression.scripts.helpers.build.conftest import (
    SubprocessRunCapture,
    build_options,
    patch_subprocess_run,
    patch_uv_in_path,
)


def test_run_test_without_test_map_runs_full_pytest(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    subprocess_capture: SubprocessRunCapture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Normal: no MSMODELING_TEST_MAP_PATH → pytest tests (pyproject addopts)."""
    import sys

    monkeypatch.delenv("MSMODELING_TEST_MAP_PATH", raising=False)
    with caplog.at_level("WARNING", logger="build"):
        assert run_test(build_options(is_test=True)) == 0
    assert "falling back to full pytest suite" in caplog.text
    assert len(subprocess_capture.merged_output_calls) == 1
    call = subprocess_capture.merged_output_calls[0]
    assert call["cmd"] == [sys.executable, "-m", "pytest", "tests"]
    log_path = repo_root / "artifacts" / "test-reports" / "full_suite.log"
    assert log_path.is_file()
    summary = json.loads(
        (repo_root / "artifacts" / "test-reports" / "gate-summary.json").read_text(encoding="utf-8"),
    )
    assert summary["exit_code"] == 0
    assert summary["mode"] == "full_suite"
    assert summary["test_map_path"] is None


def test_run_test_whitespace_test_map_env_runs_full_pytest(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    subprocess_capture: SubprocessRunCapture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Edge: blank MSMODELING_TEST_MAP_PATH is treated as unset."""
    import sys

    del repo_root
    monkeypatch.setenv("MSMODELING_TEST_MAP_PATH", "  \t  ")
    with caplog.at_level("WARNING", logger="build"):
        assert run_test(build_options(is_test=True)) == 0
    assert "falling back to full pytest suite" in caplog.text
    assert subprocess_capture.merged_output_calls[0]["cmd"] == [
        sys.executable,
        "-m",
        "pytest",
        "tests",
    ]


def test_run_test_full_suite_propagates_pytest_exit_code(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    """Abnormal: full-suite pytest failure exit code is preserved."""
    monkeypatch.delenv("MSMODELING_TEST_MAP_PATH", raising=False)

    def fail_pytest(_cmd: list[str], **_kwargs: Any) -> int:
        return 5

    subprocess_capture.on_merged_output = fail_pytest
    assert run_test(build_options(is_test=True)) == 5
    summary = json.loads(
        (repo_root / "artifacts" / "test-reports" / "gate-summary.json").read_text(encoding="utf-8"),
    )
    assert summary["exit_code"] == 5
    assert summary["mode"] == "full_suite"


def test_run_test_full_suite_applies_offline_extras(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    """Edge: offline/weights_prune extras still apply in full-suite mode."""
    del repo_root
    monkeypatch.delenv("MSMODELING_TEST_MAP_PATH", raising=False)
    options = build_options(
        is_test=True,
        extras={"offline": "1", "weights_prune": "1"},
    )
    assert run_test(options) == 0
    env = subprocess_capture.merged_output_calls[0]["env"]
    assert env["MSMODELING_OFFLINE"] == "1"
    assert env["MSMODELING_TEST_WEIGHTS_PRUNE"] == "1"


def test_run_test_delegates_env_and_tee(
    repo_root: Path,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    map_file = repo_root / "map.json"
    map_file.write_text("{}", encoding="utf-8")

    options = build_options(
        is_test=True,
        extras={
            "test_map_path": str(map_file),
            "base_branch": "develop",
            "offline": "1",
            "weights_prune": "1",
        },
    )
    assert run_test(options) == 0
    assert len(subprocess_capture.merged_output_calls) == 1
    call = subprocess_capture.merged_output_calls[0]
    assert call["cmd"] == ["bash", str(repo_root / "scripts" / "run_ci_gate.sh")]
    assert call["env"]["MSMODELING_TEST_MAP_PATH"] == str(map_file)
    assert call["env"]["MSMODELING_TEST_BASE_BRANCH"] == "develop"
    assert call["env"]["MSMODELING_OFFLINE"] == "1"
    assert call["env"]["MSMODELING_TEST_WEIGHTS_PRUNE"] == "1"
    log_path = repo_root / "artifacts" / "test-reports" / "ci_gate.log"
    assert log_path.is_file()
    assert log_path.read_text(encoding="utf-8") == "gate output\n"


def test_run_test_uses_env_test_map_path(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    subprocess_capture: SubprocessRunCapture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    map_file = repo_root / "env_map.json"
    map_file.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MSMODELING_TEST_MAP_PATH", str(map_file))

    with caplog.at_level("WARNING", logger="build"):
        assert run_test(build_options(is_test=True)) == 0
    assert "falling back to full pytest suite" not in caplog.text
    call = subprocess_capture.merged_output_calls[0]
    assert call["env"]["MSMODELING_TEST_MAP_PATH"] == str(map_file)


def test_run_test_propagates_subprocess_exit_code(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    map_file = repo_root / "map.json"
    map_file.write_text("{}", encoding="utf-8")

    def fail_gate(_cmd: list[str], **_kwargs: Any) -> int:
        return 17

    subprocess_capture.on_merged_output = fail_gate

    options = build_options(is_test=True, extras={"test_map_path": str(map_file)})
    assert run_test(options) == 17
    summary = json.loads(
        (repo_root / "artifacts" / "test-reports" / "gate-summary.json").read_text(encoding="utf-8"),
    )
    assert summary["exit_code"] == 17
    assert summary["mode"] == "ci_gate"
    assert summary["test_map_path"] == str(map_file)


def test_cli_test_without_test_map_runs_full_suite(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_uv: None,
) -> None:
    """Normal: ``python build.py test`` without map runs full pytest suite."""
    import sys

    del with_uv
    capture = patch_subprocess_run(monkeypatch, SubprocessRunCapture())
    monkeypatch.delenv("MSMODELING_TEST_MAP_PATH", raising=False)
    result = run_cli_main(main, ["test"], prog="build.py")
    assert result.returncode == 0
    assert capture.merged_output_calls[0]["cmd"] == [
        sys.executable,
        "-m",
        "pytest",
        "tests",
    ]
    assert (repo_root / "artifacts" / "test-reports" / "full_suite.log").is_file()


def test_run_test_without_uv_install_failure_returns_1(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.helpers.build import bootstrap as bootstrap_mod

    map_file = repo_root / "x.json"
    map_file.write_text("{}", encoding="utf-8")
    patch_uv_in_path(monkeypatch, uv_path=None)

    def fail_install(*_args: Any, **_kwargs: Any) -> None:
        raise SystemExit(1)

    monkeypatch.setattr(bootstrap_mod, "ensure_uv", fail_install)
    monkeypatch.setattr(run_test_mod, "bootstrap", bootstrap_mod.bootstrap)
    try:
        code = run_test(build_options(is_test=True, extras={"test_map_path": str(map_file)}))
    except SystemExit as exc:
        code = int(exc.code or 1)
    assert code == 1


def test_run_test_syncs_ci_group_not_build(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edge: test mode ensure_deps uses --group ci only."""
    from scripts.helpers.build import bootstrap as bootstrap_mod

    map_file = repo_root / "map.json"
    map_file.write_text("{}", encoding="utf-8")
    patch_uv_in_path(monkeypatch, uv_path="/fake/uv")
    capture = SubprocessRunCapture()
    patch_subprocess_run(monkeypatch, capture)
    monkeypatch.setattr(run_test_mod, "bootstrap", bootstrap_mod.bootstrap)

    assert run_test(build_options(is_test=True, extras={"test_map_path": str(map_file)})) == 0
    assert capture.sync_calls
    sync_cmd = capture.sync_calls[0]
    assert sync_cmd[sync_cmd.index("--group") + 1] == "ci"
    assert "build" not in sync_cmd[sync_cmd.index("--group") :]


def test_run_test_missing_test_map_file_returns_1(
    repo_root: Path,
    with_uv: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    del with_uv
    missing = repo_root / "missing.json"
    with caplog.at_level("ERROR"):
        assert run_test(build_options(is_test=True, extras={"test_map_path": str(missing)})) == 1
    assert "not a file" in caplog.text
