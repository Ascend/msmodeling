"""Fake subprocess.CompletedProcess for tests that mock subprocess.run."""

from __future__ import annotations


class FakeCompleted:
    """Minimal fake for subprocess.CompletedProcess.

    Only the attributes used by the code under test are provided:
    returncode, stdout, stderr.
    """

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
