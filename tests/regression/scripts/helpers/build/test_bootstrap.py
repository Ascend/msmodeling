"""Tests for scripts.helpers.build.bootstrap (ensure_uv / ensure_deps)."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

from scripts.helpers.build import bootstrap as bootstrap_mod

if TYPE_CHECKING:
    from pathlib import Path

_BOOTSTRAP = "scripts.helpers.build.bootstrap"


def test_ensure_uv_returns_existing_path_without_install(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Normal: uv already on PATH — no install, no warning."""
    monkeypatch.setattr(
        f"{_BOOTSTRAP}.shutil.which",
        lambda name: "/usr/bin/uv" if name == "uv" else None,
    )
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(f"{_BOOTSTRAP}.subprocess.run", fake_run)

    with caplog.at_level("WARNING"):
        assert bootstrap_mod.ensure_uv() == "/usr/bin/uv"
    assert calls == []
    assert "uv not found" not in caplog.text.lower()


def test_ensure_uv_warns_and_installs_noninteractively(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Normal: missing uv → WARNING + default-pip install, then which succeeds."""
    which_state: dict[str, str | None] = {"uv": None}

    def fake_which(name: str) -> str | None:
        if name == "uv":
            return which_state["uv"]
        return None

    monkeypatch.setattr(f"{_BOOTSTRAP}.shutil.which", fake_which)
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        assert kwargs.get("input") in (None, "")
        # Non-interactive: no stdin prompt path
        assert kwargs.get("stdin") in (None, subprocess.DEVNULL) or "stdin" not in kwargs
        if cmd[:3] == [sys.executable, "-m", "pip"]:
            which_state["uv"] = "/installed/uv"
            return subprocess.CompletedProcess(cmd, 0, "", "")
        msg = f"unexpected command: {cmd!r}"
        raise AssertionError(msg)

    monkeypatch.setattr(f"{_BOOTSTRAP}.subprocess.run", fake_run)

    with caplog.at_level("WARNING"):
        path = bootstrap_mod.ensure_uv()
    assert path == "/installed/uv"
    assert "WARNING" in caplog.text or "uv not found" in caplog.text.lower()
    assert any("pip" in " ".join(c) and "install" in c and "uv" in c for c in calls)
    pip_cmd = next(c for c in calls if "pip" in c)
    assert pip_cmd == [sys.executable, "-m", "pip", "install", "--upgrade", "uv"]
    assert "-i" not in pip_cmd
    assert not any("ustc" in part for part in pip_cmd)
    assert "PIP_INDEX_URL" in caplog.text
    assert "UV_INDEX_URL" in caplog.text or "UV_DEFAULT_INDEX" in caplog.text


def test_ensure_uv_falls_back_to_interpreter_scripts_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Edge: after install, which misses uv but Scripts-dir candidate exists."""
    scripts_dir = tmp_path / "bin"
    scripts_dir.mkdir(parents=True)
    uv_candidate = scripts_dir / ("uv.exe" if sys.platform == "win32" else "uv")
    uv_candidate.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(scripts_dir / "python"))
    monkeypatch.setattr(f"{_BOOTSTRAP}.shutil.which", lambda name: None)

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == [sys.executable, "-m", "pip"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        msg = f"unexpected command: {cmd!r}"
        raise AssertionError(msg)

    monkeypatch.setattr(f"{_BOOTSTRAP}.subprocess.run", fake_run)
    assert bootstrap_mod.ensure_uv() == str(uv_candidate)


def test_ensure_uv_missing_after_install_exits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Abnormal: install ok but which and Scripts-dir candidate both missing → SystemExit."""
    scripts_dir = tmp_path / "bin"
    scripts_dir.mkdir(parents=True)
    monkeypatch.setattr(sys, "executable", str(scripts_dir / "python"))
    monkeypatch.setattr(f"{_BOOTSTRAP}.shutil.which", lambda name: None)

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(f"{_BOOTSTRAP}.subprocess.run", fake_run)
    with pytest.raises(SystemExit) as exc_info:
        bootstrap_mod.ensure_uv()
    assert exc_info.value.code not in (0, None)


def test_ensure_uv_install_network_failure_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Abnormal: pip install fails → SystemExit non-zero; no second fallback."""
    monkeypatch.setattr(f"{_BOOTSTRAP}.shutil.which", lambda name: None)
    fallback_calls: list[str] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "venv" in cmd or (len(cmd) >= 2 and cmd[1] == "-m" and "build" in cmd):
            fallback_calls.append(" ".join(cmd))
        return subprocess.CompletedProcess(cmd, 1, "", "network error")

    monkeypatch.setattr(f"{_BOOTSTRAP}.subprocess.run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        bootstrap_mod.ensure_uv()
    assert exc_info.value.code not in (0, None)
    assert fallback_calls == []


def test_ensure_deps_build_syncs_build_group_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal / edge: build mode syncs --group build, never --group ci."""
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(f"{_BOOTSTRAP}.subprocess.run", fake_run)
    bootstrap_mod.ensure_deps("build", uv_path="/fake/uv")
    assert len(calls) == 1
    assert calls[0][:3] == ["/fake/uv", "sync", "--frozen"]
    assert "--group" in calls[0]
    group_idx = calls[0].index("--group")
    assert calls[0][group_idx + 1] == "build"
    assert "ci" not in calls[0]


def test_ensure_deps_test_syncs_ci_group_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normal / edge: test mode syncs --group ci, never --group build."""
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(f"{_BOOTSTRAP}.subprocess.run", fake_run)
    bootstrap_mod.ensure_deps("test", uv_path="/fake/uv")
    assert len(calls) == 1
    assert calls[0][:3] == ["/fake/uv", "sync", "--frozen"]
    group_idx = calls[0].index("--group")
    assert calls[0][group_idx + 1] == "ci"
    assert "build" not in calls[0][group_idx:]


def test_ensure_deps_sync_failure_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Abnormal: uv sync network failure → SystemExit; no venv/pip build fallback."""
    fallback_calls: list[str] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        joined = " ".join(cmd)
        if "venv" in joined or "python -m build" in joined:
            fallback_calls.append(joined)
        return subprocess.CompletedProcess(cmd, 2, "", "Could not connect")

    monkeypatch.setattr(f"{_BOOTSTRAP}.subprocess.run", fake_run)
    with pytest.raises(SystemExit) as exc_info:
        bootstrap_mod.ensure_deps("build", uv_path="/fake/uv")
    assert exc_info.value.code not in (0, None)
    assert fallback_calls == []


def test_bootstrap_fail_fast_missing_uv_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Abnormal: missing uv.lock → fail-fast before install/sync."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "build.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
    monkeypatch.setattr(bootstrap_mod, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        bootstrap_mod.bootstrap("build")
    assert exc_info.value.code not in (0, None)


def test_bootstrap_is_noninteractive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Edge: bootstrap never reads stdin / never prompts."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "build.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.1.0"\n', encoding="utf-8")
    (tmp_path / "uv.lock").write_text("# lock\n", encoding="utf-8")
    monkeypatch.setattr(bootstrap_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        f"{_BOOTSTRAP}.shutil.which",
        lambda name: "/fake/uv" if name == "uv" else None,
    )

    seen_kwargs: list[dict[str, Any]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen_kwargs.append(dict(kwargs))
        assert kwargs.get("input") in (None, "")
        stdin = kwargs.get("stdin")
        assert stdin in (None, subprocess.DEVNULL) or "stdin" not in kwargs
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(f"{_BOOTSTRAP}.subprocess.run", fake_run)
    assert bootstrap_mod.bootstrap("build") == "/fake/uv"
    assert seen_kwargs  # sync ran


def test_bootstrap_prepends_uv_dir_to_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Edge: Scripts-only uv is prepended to PATH so shell can resolve ``uv``."""
    repo = tmp_path / "repo"
    scripts_dir = repo / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "build.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.1.0"\n', encoding="utf-8")
    (repo / "uv.lock").write_text("# lock\n", encoding="utf-8")

    pybin = tmp_path / "pybin"
    pybin.mkdir(parents=True)
    uv_candidate = pybin / "uv"
    uv_candidate.write_text("", encoding="utf-8")
    uv_candidate.chmod(0o755)

    monkeypatch.setattr(bootstrap_mod, "REPO_ROOT", repo)
    monkeypatch.setattr(sys, "executable", str(pybin / "python"))
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(f"{_BOOTSTRAP}.shutil.which", lambda name: None)

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == [sys.executable, "-m", "pip"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if len(cmd) >= 2 and cmd[1] == "sync":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        msg = f"unexpected command: {cmd!r}"
        raise AssertionError(msg)

    monkeypatch.setattr(f"{_BOOTSTRAP}.subprocess.run", fake_run)

    path = bootstrap_mod.bootstrap("build")
    assert path == str(uv_candidate)
    path_dirs = os.environ["PATH"].split(os.pathsep)
    assert path_dirs[0] == str(pybin)
    assert (pybin / "uv").is_file()
