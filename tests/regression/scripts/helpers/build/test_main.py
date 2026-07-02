"""Tests for scripts.helpers.build.main."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from scripts.helpers.build.argv import BuildOptions

from scripts.helpers.build import main as build_main
from scripts.helpers.build.main import _run_shell, main, run_build, run_test
from tests.helpers.cli_runner import run_cli_main
from tests.regression.scripts.helpers.build.conftest import (
    SubprocessRunCapture,
    build_options,
    fake_build_bash,
    patch_subprocess_run,
    patch_uv_in_path,
)


def test_run_test_without_test_map_path_returns_1(
    repo_root: Path,
    with_uv: None,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del repo_root
    monkeypatch.delenv("MSMODELING_TEST_MAP_PATH", raising=False)
    with caplog.at_level("ERROR"):
        assert run_test(build_options(is_test=True)) == 1
    assert "test_map_path" in caplog.text


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
) -> None:
    map_file = repo_root / "env_map.json"
    map_file.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MSMODELING_TEST_MAP_PATH", str(map_file))

    assert run_test(build_options(is_test=True)) == 0
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
    assert summary["test_map_path"] == str(map_file)


def test_run_build_delegates_to_build_sh(
    repo_root: Path,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    subprocess_capture.on_bash = fake_build_bash(
        repo_root,
        wheel_name="msmodeling-3.2.1-py3-none-any.whl",
    )

    assert run_build(build_options(version="3.2.1", version_explicit=True)) == 0
    call = subprocess_capture.shell_calls[0]
    assert call["cmd"] == ["bash", str(repo_root / "scripts" / "build.sh")]
    assert call["kwargs"]["env"]["MSMODELING_WHEEL_OUTPUT_DIR"] == str(repo_root / "artifacts" / "wheels")
    manifest = json.loads((repo_root / "artifacts" / "build-manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "3.2.1"
    assert manifest["pyproject_version"] == "0.2.0"
    assert manifest["version_explicit"] is True
    assert manifest["wheel_path"].endswith("msmodeling-3.2.1-py3-none-any.whl")
    assert subprocess_capture.version_calls == ["3.2.1", "0.2.0"]


def test_run_build_stages_and_restores_pyproject_version(
    repo_root: Path,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    subprocess_capture.on_bash = fake_build_bash(
        repo_root,
        wheel_name="msmodeling-9.9.9-py3-none-any.whl",
    )

    assert run_build(build_options(version="9.9.9", version_explicit=True)) == 0
    assert subprocess_capture.version_calls == ["9.9.9", "0.2.0"]
    wheel_dir = repo_root / "artifacts" / "wheels"
    assert (wheel_dir / "msmodeling-9.9.9-py3-none-any.whl").is_file()
    manifest = json.loads((repo_root / "artifacts" / "build-manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "9.9.9"
    assert manifest["version_explicit"] is True


def test_run_build_restores_version_when_build_fails(
    repo_root: Path,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    def fail_bash(cmd: list[str], _kwargs: dict[str, Any]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 3, "", "")

    subprocess_capture.on_bash = fail_bash

    assert run_build(build_options(version="9.9.9", version_explicit=True)) == 3
    assert subprocess_capture.version_calls == ["9.9.9", "0.2.0"]


def test_run_build_propagates_subprocess_exit_code_without_staging(
    repo_root: Path,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    def fail_bash(cmd: list[str], _kwargs: dict[str, Any]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 5, "", "")

    subprocess_capture.on_bash = fail_bash
    assert run_build(build_options()) == 5


def test_local_token_routes_to_build_not_test(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del repo_root
    calls: list[str] = []

    def fake_build(options: BuildOptions) -> int:
        calls.append("build")
        assert options.is_local is True
        return 0

    def fake_test(options: BuildOptions) -> int:
        calls.append("test")
        return 0

    monkeypatch.setattr(build_main, "run_build", fake_build)
    monkeypatch.setattr(build_main, "run_test", fake_test)
    assert main(["local"]) == 0
    assert calls == ["build"]


def test_main_test_branch_via_cli_runner(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_uv: None,
) -> None:
    del with_uv
    capture = patch_subprocess_run(monkeypatch, SubprocessRunCapture())
    map_file = repo_root / "test_map.json"
    map_file.write_text("{}", encoding="utf-8")

    result = run_cli_main(
        main,
        ["test", "-e", f"test_map_path={map_file}"],
        prog="build.py",
    )
    assert result.returncode == 0
    assert capture.merged_output_calls[0]["cmd"] == ["bash", str(repo_root / "scripts" / "run_ci_gate.sh")]


def test_cli_test_without_test_map_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MSMODELING_TEST_MAP_PATH", raising=False)
    result = run_cli_main(main, ["test"], prog="build.py")
    assert result.returncode == 1


def test_malformed_extra_via_cli_runner() -> None:
    result = run_cli_main(main, ["test", "-e", "not-key-value"], prog="build.py")
    assert result.returncode == 2
    assert "KEY=VALUE" in result.stderr


def test_run_build_without_uv_returns_1(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del repo_root
    patch_uv_in_path(monkeypatch, uv_path=None)
    assert run_build(build_options()) == 1


def test_run_test_without_uv_returns_1(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_uv_in_path(monkeypatch, uv_path=None)
    assert (
        run_test(
            build_options(is_test=True, extras={"test_map_path": str(repo_root / "x.json")}),
        )
        == 1
    )


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


def test_run_build_version_stage_failure_returns_exit_code(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_uv: None,
) -> None:
    del repo_root, with_uv
    capture = patch_subprocess_run(monkeypatch, SubprocessRunCapture())

    def fail_stage(version: str) -> subprocess.CompletedProcess[str]:
        if version == "9.9.9":
            return subprocess.CompletedProcess(["uv", "version", version], 2, "", "")
        return subprocess.CompletedProcess(["uv", "version", version], 0, "", "")

    capture.on_uv_version = fail_stage
    assert run_build(build_options(version="9.9.9", version_explicit=True)) == 2


def test_run_build_restore_failure_returns_1_after_successful_build(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_uv: None,
) -> None:
    del with_uv
    capture = patch_subprocess_run(monkeypatch, SubprocessRunCapture())
    capture.on_bash = fake_build_bash(repo_root, wheel_name="msmodeling-9.9.9-py3-none-any.whl")

    def fail_restore(version: str) -> subprocess.CompletedProcess[str]:
        if version == "0.2.0":
            return subprocess.CompletedProcess(["uv", "version", version], 2, "", "")
        return subprocess.CompletedProcess(["uv", "version", version], 0, "", "")

    capture.on_uv_version = fail_restore

    assert run_build(build_options(version="9.9.9", version_explicit=True)) == 1
    assert capture.version_calls == ["9.9.9", "0.2.0"]
    assert not (repo_root / "artifacts" / "build-manifest.json").exists()


def test_run_shell_without_tee_uses_run_not_popen(monkeypatch: pytest.MonkeyPatch) -> None:
    popen_called = False
    merged_called = False

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def fake_popen(*_args: Any, **_kwargs: Any) -> None:
        nonlocal popen_called
        popen_called = True
        return None

    def fake_merged(*_args: Any, **_kwargs: Any) -> int:
        nonlocal merged_called
        merged_called = True
        return 0

    monkeypatch.setattr("scripts.helpers.build.main.subprocess.run", fake_run)
    monkeypatch.setattr("scripts.helpers.build.main.subprocess.Popen", fake_popen)
    monkeypatch.setattr("scripts.helpers.build.main.run_merged_output", fake_merged)
    _run_shell(["bash", "x.sh"], cwd=Path("."), env={}, timeout=10, tee_path=None)
    assert not popen_called
    assert not merged_called


def test_run_shell_with_tee_merges_stderr_into_stdout(tmp_path: Path) -> None:
    tee = tmp_path / "out.log"
    _run_shell(
        ["bash", "-c", "echo line1; echo line2 >&2"],
        cwd=tmp_path,
        env={},
        timeout=30,
        tee_path=tee,
    )
    assert tee.read_text(encoding="utf-8") == "line1\nline2\n"


def test_run_shell_nonzero_exit_preserves_tee(tmp_path: Path) -> None:
    tee = tmp_path / "fail.log"
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _run_shell(
            ["bash", "-c", "echo before-fail; exit 3"],
            cwd=tmp_path,
            env={},
            timeout=30,
            tee_path=tee,
        )
    assert exc_info.value.returncode == 3
    assert "before-fail" in tee.read_text(encoding="utf-8")


def test_run_shell_timeout_preserves_partial_tee(tmp_path: Path) -> None:
    tee = tmp_path / "timeout.log"
    with pytest.raises(subprocess.TimeoutExpired):
        _run_shell(
            ["bash", "-c", "echo banner; sleep 5"],
            cwd=tmp_path,
            env={},
            timeout=1,
            tee_path=tee,
        )
    assert "banner" in tee.read_text(encoding="utf-8")


def test_e2e_explicit_version_stages_pyproject(
    repo_root: Path,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    subprocess_capture.on_bash = fake_build_bash(
        repo_root,
        wheel_name="msmodeling-26.1.1-py3-none-any.whl",
    )
    wheel_dir = repo_root / "artifacts" / "wheels"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    (wheel_dir / "msmodeling-1.0.0-py3-none-any.whl").write_bytes(b"stale")

    assert run_build(build_options(version="26.1.1", version_explicit=True)) == 0
    assert subprocess_capture.version_calls == ["26.1.1", "0.2.0"]
    wheels = list(wheel_dir.glob("msmodeling-*.whl"))
    assert len(wheels) == 1
    assert wheels[0].name == "msmodeling-26.1.1-py3-none-any.whl"
    manifest = json.loads((repo_root / "artifacts" / "build-manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "26.1.1"
    assert manifest["version_explicit"] is True


def test_e2e_default_version_keeps_pyproject_wheel_name(
    repo_root: Path,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    subprocess_capture.on_bash = fake_build_bash(
        repo_root,
        wheel_name="msmodeling-0.2.0-py3-none-any.whl",
    )
    wheel_dir = repo_root / "artifacts" / "wheels"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    (wheel_dir / "msmodeling-9.9.9-py3-none-any.whl").write_bytes(b"stale")

    assert run_build(build_options()) == 0
    assert subprocess_capture.version_calls == []
    wheels = list(wheel_dir.glob("msmodeling-*.whl"))
    assert len(wheels) == 1
    assert wheels[0].name == "msmodeling-0.2.0-py3-none-any.whl"
    manifest = json.loads((repo_root / "artifacts" / "build-manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "0.2.0"
    assert manifest["version_explicit"] is False


def test_e2e_explicit_version_matching_pyproject_skips_staging(
    repo_root: Path,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    subprocess_capture.on_bash = fake_build_bash(
        repo_root,
        wheel_name="msmodeling-0.2.0-py3-none-any.whl",
    )
    assert run_build(build_options(version="0.2.0", version_explicit=True)) == 0
    assert subprocess_capture.version_calls == []


def test_e2e_cli_short_version_flag(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_uv: None,
) -> None:
    del with_uv
    capture = patch_subprocess_run(monkeypatch, SubprocessRunCapture())
    capture.on_bash = fake_build_bash(repo_root, wheel_name="msmodeling-26.1.1-py3-none-any.whl")
    result = run_cli_main(main, ["-v", "26.1.1"], prog="build.py")
    assert result.returncode == 0
    assert capture.version_calls == ["26.1.1", "0.2.0"]


def test_e2e_build_subprocess_failure_propagates(
    repo_root: Path,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    def fail_bash(cmd: list[str], _kwargs: dict[str, Any]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 9, "", "")

    subprocess_capture.on_bash = fail_bash
    assert run_build(build_options(version="1.2.3", version_explicit=True)) == 9
    assert not (repo_root / "artifacts" / "build-manifest.json").exists()


def test_e2e_missing_build_script_returns_1(repo_root: Path, with_uv: None) -> None:
    del with_uv
    (repo_root / "scripts" / "build.sh").unlink()
    assert run_build(build_options()) == 1


def test_e2e_no_wheel_produced_writes_empty_manifest_path(
    repo_root: Path,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    def empty_build(cmd: list[str], _kwargs: dict[str, Any]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, "ok\n", "")

    subprocess_capture.on_bash = empty_build
    assert run_build(build_options(version="3.0.0", version_explicit=True)) == 0
    manifest = json.loads((repo_root / "artifacts" / "build-manifest.json").read_text(encoding="utf-8"))
    assert manifest["wheel_path"] == ""


def test_main_build_via_cli_runner(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_uv: None,
) -> None:
    del with_uv
    capture = patch_subprocess_run(monkeypatch, SubprocessRunCapture())
    capture.on_bash = fake_build_bash(repo_root, wheel_name="msmodeling-0.2.0-py3-none-any.whl")
    result = run_cli_main(main, [], prog="build.py")
    assert result.returncode == 0
    assert capture.shell_calls[0]["cmd"] == ["bash", str(repo_root / "scripts" / "build.sh")]
