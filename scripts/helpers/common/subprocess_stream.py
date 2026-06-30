"""Stream subprocess stdout+stderr with optional tee and wall-clock timeout."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

_PROCESS_TERMINATE_TIMEOUT_SECONDS: float = 5.0


class _TerminableProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


def _terminate_process_tree(
    proc: _TerminableProcess,
    *,
    sigterm_timeout_seconds: float = _PROCESS_TERMINATE_TIMEOUT_SECONDS,
) -> None:
    """SIGTERM the process group, escalate to SIGKILL on timeout."""
    if proc.poll() is not None:
        return

    if hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
    else:
        proc.terminate()

    try:
        proc.wait(timeout=sigterm_timeout_seconds)
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg"):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=sigterm_timeout_seconds)


def run_merged_output(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
    tee_path: Path | None = None,
    mirror_stdout: bool = True,
    start_new_session: bool = False,
) -> int:
    """Run *cmd* merging stderr into stdout; optionally tee and mirror lines.

    Returns exit code. Raises ``TimeoutExpired`` on wall-clock timeout after
    killing the process tree (partial tee output is flushed).
    """
    timed_out = threading.Event()

    proc = subprocess.Popen(
        list(cmd),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=start_new_session,
    )

    def _on_timeout() -> None:
        timed_out.set()
        _terminate_process_tree(cast("_TerminableProcess", proc))

    timer = threading.Timer(timeout, _on_timeout)
    timer.daemon = True
    timer.start()

    try:
        with contextlib.ExitStack() as stack:
            sink = None
            if tee_path is not None:
                tee_path.parent.mkdir(parents=True, exist_ok=True)
                sink = stack.enter_context(tee_path.open("w", encoding="utf-8"))

            if proc.stdout is None:
                msg = "Failed to capture subprocess stdout"
                raise RuntimeError(msg)

            stdout = proc.stdout
            stack.enter_context(contextlib.closing(stdout))

            try:
                for line in stdout:
                    if sink is not None:
                        sink.write(line)
                        sink.flush()
                    if mirror_stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
            except KeyboardInterrupt:
                _terminate_process_tree(cast("_TerminableProcess", proc))
                raise

        exit_code = proc.wait()
        if timed_out.is_set():
            raise subprocess.TimeoutExpired(cmd, timeout)
        return exit_code
    finally:
        timer.cancel()
        if proc.poll() is None:
            _terminate_process_tree(cast("_TerminableProcess", proc))
            proc.wait()
