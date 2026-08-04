"""Fake subprocess.CompletedProcess for tests that mock subprocess.run."""

from __future__ import annotations

import signal
import subprocess


class FakeCompleted:
    """Minimal fake for subprocess.CompletedProcess.

    Only the attributes used by the code under test are provided:
    returncode, stdout, stderr.
    """

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakePopen:
    """Minimal fake for subprocess.Popen used by nightly pytest streaming."""

    pid = 4242

    def __init__(self, *, stdout: object | None = None, pid: int = 4242) -> None:
        self.pid = pid
        self._returncode: int | None = None
        self.stdout = stdout
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def poll(self) -> int | None:
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self._returncode is None:
            self._returncode = -signal.SIGTERM
        return self._returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._returncode = -signal.SIGTERM

    def kill(self) -> None:
        self.kill_calls += 1
        self._returncode = -signal.SIGKILL


class FakePopenTimeoutOnFirstWait(FakePopen):
    """Fake Popen whose first wait() raises TimeoutExpired, then returns SIGKILL."""

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise subprocess.TimeoutExpired(cmd=["pytest"], timeout=timeout or 0.01)
        self._returncode = -signal.SIGKILL
        return self._returncode
