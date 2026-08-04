"""Tests for scripts.helpers.common.subprocess_stream."""

from __future__ import annotations

import signal
import subprocess
from typing import TYPE_CHECKING, Any

import pytest

from scripts.helpers.common.subprocess_stream import _terminate_process_tree, run_merged_output
from tests.helpers.fake_subprocess import FakePopen, FakePopenTimeoutOnFirstWait

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def test_run_merged_output_tees_stdout_and_stderr(tmp_path: Path) -> None:
    tee = tmp_path / "merged.log"
    exit_code = run_merged_output(
        ["bash", "-c", "echo a; echo b >&2"],
        cwd=tmp_path,
        env={},
        timeout=30,
        tee_path=tee,
        mirror_stdout=False,
    )
    assert exit_code == 0
    assert tee.read_text(encoding="utf-8") == "a\nb\n"


def test_run_merged_output_mirrors_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_merged_output(
        ["bash", "-c", "echo mirrored"],
        cwd=tmp_path,
        env={},
        timeout=30,
        tee_path=None,
        mirror_stdout=True,
    )
    assert exit_code == 0
    assert "mirrored" in capsys.readouterr().out


def test_run_merged_output_timeout_preserves_partial_tee(tmp_path: Path) -> None:
    tee = tmp_path / "partial.log"
    with pytest.raises(subprocess.TimeoutExpired):
        run_merged_output(
            ["bash", "-c", "echo banner; sleep 5"],
            cwd=tmp_path,
            env={},
            timeout=1,
            tee_path=tee,
            mirror_stdout=False,
            start_new_session=True,
        )
    assert "banner" in tee.read_text(encoding="utf-8")


class _InterruptingStdout:
    closed = False

    def __iter__(self) -> Iterator[str]:
        yield "line1\n"
        raise KeyboardInterrupt

    def close(self) -> None:
        self.closed = True


def test_run_merged_output_keyboard_interrupt_terminates_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_proc = FakePopen(stdout=_InterruptingStdout())
    killpg_calls: list[tuple[int, int]] = []

    def _fake_popen(*_args: Any, **_kwargs: Any) -> FakePopen:
        return fake_proc

    def _fake_killpg(pgid: int, sig: int) -> None:
        killpg_calls.append((pgid, sig))
        fake_proc._returncode = -sig

    monkeypatch.setattr("scripts.helpers.common.subprocess_stream.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("scripts.helpers.common.subprocess_stream.os.getpgid", lambda _pid: fake_proc.pid)
    monkeypatch.setattr("scripts.helpers.common.subprocess_stream.os.killpg", _fake_killpg)

    with pytest.raises(KeyboardInterrupt):
        run_merged_output(
            ["bash", "-c", "true"],
            cwd=tmp_path,
            env={},
            timeout=30,
            mirror_stdout=False,
        )

    assert killpg_calls == [(fake_proc.pid, signal.SIGTERM)]


def test_terminate_process_tree_escalates_to_sigkill_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = FakePopenTimeoutOnFirstWait()
    killpg_calls: list[tuple[int, int]] = []

    def _fake_killpg(pgid: int, sig: int) -> None:
        killpg_calls.append((pgid, sig))
        if sig == signal.SIGKILL:
            fake_proc._returncode = -signal.SIGKILL

    monkeypatch.setattr("scripts.helpers.common.subprocess_stream.os.getpgid", lambda _pid: fake_proc.pid)
    monkeypatch.setattr("scripts.helpers.common.subprocess_stream.os.killpg", _fake_killpg)

    _terminate_process_tree(fake_proc, sigterm_timeout_seconds=0.01)

    assert killpg_calls == [
        (fake_proc.pid, signal.SIGTERM),
        (fake_proc.pid, signal.SIGKILL),
    ]
