"""Tests for nightly.report_builder — fetch_env_info."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from scripts.helpers.nightly.report_builder import fetch_env_info
from tests.helpers.fake_subprocess import FakeCompleted


def test_fetch_env_info_returns_commit_and_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(cmd: list[str], **_kwargs: object) -> FakeCompleted:
        if "rev-parse" in cmd:
            return FakeCompleted(0, "abc1234\n", "")
        if "--show-current" in cmd:
            return FakeCompleted(0, "main\n", "")
        return FakeCompleted(1, "", "")

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/git")
    info = fetch_env_info()
    assert info.commit == "abc1234"
    assert info.branch == "main"
    assert len(info.timestamp) > 0


def test_fetch_env_info_returns_unknown_when_git_stdout_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.nightly.report_builder.git_stdout",
        lambda *_args, **_kwargs: "",
    )
    info = fetch_env_info()
    assert info.commit == "unknown"
    assert info.branch == "unknown"
