"""Tests for scripts.helpers.build.main dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.helpers.build import main as build_main
from scripts.helpers.build.main import main
from tests.helpers.cli_runner import run_cli_main
from tests.regression.scripts.helpers.build.conftest import (
    SubprocessRunCapture,
    fake_build_bash,
    patch_subprocess_run,
)

if TYPE_CHECKING:
    from pathlib import Path

    from scripts.helpers.build.argv import BuildOptions


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
    assert capture.merged_output_calls[0]["cmd"] == [
        "bash",
        str(repo_root / "scripts" / "run_ci_gate.sh"),
    ]


def test_malformed_extra_via_cli_runner() -> None:
    result = run_cli_main(main, ["test", "-e", "not-key-value"], prog="build.py")
    assert result.returncode == 2
    assert "KEY=VALUE" in result.stderr


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
    assert capture.shell_calls[0]["cmd"] == [
        "bash",
        str(repo_root / "scripts" / "build.sh"),
    ]
