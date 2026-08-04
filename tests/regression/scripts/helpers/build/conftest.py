"""Shared fixtures for scripts.helpers.build regression tests."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from scripts.helpers.build import run_build as run_build_mod
from scripts.helpers.build import run_test as run_test_mod
from scripts.helpers.build.argv import BuildOptions

if TYPE_CHECKING:
    from collections.abc import Callable

_FAKE_UV = "/fake/uv"
_RUN_BUILD = "scripts.helpers.build.run_build"
_RUN_TEST = "scripts.helpers.build.run_test"
_BOOTSTRAP = "scripts.helpers.build.bootstrap"


def _completed(cmd: list[str], returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)


def _raise_if_check(cmd: list[str], completed: subprocess.CompletedProcess[str], **kwargs: Any) -> None:
    if kwargs.get("check") and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            output=completed.stdout,
            stderr=completed.stderr,
        )


@dataclass
class SubprocessRunCapture:
    """Records boundary subprocess invocations from run_build / run_test."""

    version_calls: list[str] = field(default_factory=list)
    shell_calls: list[dict[str, Any]] = field(default_factory=list)
    merged_output_calls: list[dict[str, Any]] = field(default_factory=list)
    sync_calls: list[list[str]] = field(default_factory=list)
    on_uv_version: Callable[[str], subprocess.CompletedProcess[str] | None] | None = None
    on_bash: Callable[[list[str], dict[str, Any]], subprocess.CompletedProcess[str] | None] | None = None
    on_merged_output: Callable[..., int] | None = None
    on_uv_sync: Callable[[list[str]], subprocess.CompletedProcess[str] | None] | None = None

    def _run_uv_version(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        version = cmd[2]
        self.version_calls.append(version)
        if self.on_uv_version is not None:
            custom = self.on_uv_version(version)
            if custom is not None:
                _raise_if_check(cmd, custom, **kwargs)
                return custom
        completed = _completed(cmd, 0)
        _raise_if_check(cmd, completed, **kwargs)
        return completed

    def _run_uv_sync(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.sync_calls.append(list(cmd))
        if self.on_uv_sync is not None:
            custom = self.on_uv_sync(cmd)
            if custom is not None:
                _raise_if_check(cmd, custom, **kwargs)
                return custom
        completed = _completed(cmd, 0)
        _raise_if_check(cmd, completed, **kwargs)
        return completed

    def _run_bash(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        call = {"cmd": cmd, "kwargs": kwargs}
        self.shell_calls.append(call)
        if self.on_bash is not None:
            custom = self.on_bash(cmd, kwargs)
            if custom is not None:
                _raise_if_check(cmd, custom, **kwargs)
                return custom
        completed = _completed(cmd, 0)
        _raise_if_check(cmd, completed, **kwargs)
        return completed

    def run(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if len(cmd) >= 3 and Path(cmd[0]).name == "uv" and cmd[1] == "version":
            return self._run_uv_version(cmd, **kwargs)
        if len(cmd) >= 2 and Path(cmd[0]).name == "uv" and cmd[1] == "sync":
            return self._run_uv_sync(cmd, **kwargs)
        if cmd and cmd[0] == "bash":
            return self._run_bash(cmd, **kwargs)
        msg = f"unexpected subprocess.run command: {cmd!r}"
        raise AssertionError(msg)

    def run_merged_output(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout: float,
        tee_path: Path | None = None,
        **kwargs: Any,
    ) -> int:
        call = {
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
            "timeout": timeout,
            "tee_path": tee_path,
            **kwargs,
        }
        self.merged_output_calls.append(call)
        if self.on_merged_output is not None:
            return self.on_merged_output(
                cmd,
                cwd=cwd,
                env=env,
                timeout=timeout,
                tee_path=tee_path,
                **kwargs,
            )
        if tee_path is not None:
            tee_path.parent.mkdir(parents=True, exist_ok=True)
            tee_path.write_text("gate output\n", encoding="utf-8")
        return 0


def build_options(**overrides: Any) -> BuildOptions:
    defaults: dict[str, Any] = {
        "is_test": False,
        "is_local": False,
        "version": None,
        "version_explicit": False,
        "extras": {},
    }
    defaults.update(overrides)
    return BuildOptions(**defaults)


def patch_uv_in_path(monkeypatch: pytest.MonkeyPatch, *, uv_path: str | None = _FAKE_UV) -> None:
    def which(name: str) -> str | None:
        if name == "uv":
            return uv_path
        return None

    monkeypatch.setattr(f"{_RUN_BUILD}.shutil.which", which)
    monkeypatch.setattr(f"{_BOOTSTRAP}.shutil.which", which)


def patch_bootstrap_success(
    monkeypatch: pytest.MonkeyPatch,
    *,
    uv_path: str = _FAKE_UV,
) -> None:
    """Skip real ensure_uv/ensure_deps; return a fake uv path."""

    def fake_bootstrap(_mode: str) -> str:
        return uv_path

    monkeypatch.setattr(f"{_RUN_BUILD}.bootstrap", fake_bootstrap)
    monkeypatch.setattr(f"{_RUN_TEST}.bootstrap", fake_bootstrap)
    monkeypatch.setattr(f"{_BOOTSTRAP}.bootstrap", fake_bootstrap)


def patch_subprocess_run(
    monkeypatch: pytest.MonkeyPatch,
    capture: SubprocessRunCapture,
) -> SubprocessRunCapture:
    monkeypatch.setattr(f"{_RUN_BUILD}.subprocess.run", capture.run)
    monkeypatch.setattr(f"{_RUN_BUILD}.run_merged_output", capture.run_merged_output)
    monkeypatch.setattr(f"{_RUN_TEST}.run_merged_output", capture.run_merged_output)
    monkeypatch.setattr(f"{_BOOTSTRAP}.subprocess.run", capture.run)
    return capture


def fake_build_bash(
    repo_root: Path,
    *,
    wheel_name: str,
    stdout: str = "",
) -> Callable[[list[str], dict[str, Any]], subprocess.CompletedProcess[str]]:
    def handler(cmd: list[str], _kwargs: dict[str, Any]) -> subprocess.CompletedProcess[str]:
        wheel_dir = repo_root / "artifacts"
        wheel_dir.mkdir(parents=True, exist_ok=True)
        wheel_path = wheel_dir / wheel_name
        wheel_path.write_bytes(b"wheel-bytes")
        message = stdout or f"Successfully built {wheel_path}\n"
        return _completed(cmd, 0, message)

    return handler


@pytest.fixture
def repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "build.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (scripts_dir / "run_ci_gate.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "msmodeling"\nversion = "0.2.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text("# test lock\n", encoding="utf-8")
    monkeypatch.setattr(run_build_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_build_mod, "_BUILD_SCRIPT", scripts_dir / "build.sh")
    monkeypatch.setattr(run_build_mod, "_ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(run_build_mod, "_WHEEL_OUTPUT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(run_test_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_test_mod, "_CI_GATE_SCRIPT", scripts_dir / "run_ci_gate.sh")
    monkeypatch.setattr(run_test_mod, "_TEST_REPORTS_DIR", tmp_path / "artifacts" / "test-reports")
    try:
        from scripts.helpers.build import bootstrap as bootstrap_mod

        monkeypatch.setattr(bootstrap_mod, "REPO_ROOT", tmp_path)
    except ImportError:
        pass
    return tmp_path


@pytest.fixture
def with_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_uv_in_path(monkeypatch)
    patch_bootstrap_success(monkeypatch)


@pytest.fixture
def subprocess_capture(
    monkeypatch: pytest.MonkeyPatch,
    with_uv: None,
) -> SubprocessRunCapture:
    return patch_subprocess_run(monkeypatch, SubprocessRunCapture())
