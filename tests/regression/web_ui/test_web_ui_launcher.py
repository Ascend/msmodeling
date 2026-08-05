"""Unit tests for ``web_ui/main.py`` dev launcher — 100% line+branch coverage.

Covers every platform branch (nt/posix), the SIGTERM → grace → SIGKILL
escalation in ``_stop``, tagged stdout/stderr streaming (``_pipe``), the
first-exit await (``_wait_first``), subprocess spawn args (``_start``), and
the ``main()`` orchestrator's exit/cancel/None-rc paths.

``signal.SIGKILL`` does not exist on Windows (it is only ever referenced under
``if os.name != "nt"`` in the source, so real Windows never evaluates it); the
posix ``_stop`` tests patch ``os.name`` AND supply ``SIGKILL``. Async helpers
run via ``asyncio.run`` — ``pytest-asyncio`` is intentionally not a dependency.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from unittest.mock import MagicMock

import pytest

import web_ui.main as launcher
from web_ui.main import (
    BACKEND_DIR,
    FRONTEND_DIR,
    _frontend_command,
    _get_bind_host,
    _group_kwargs,
    _pipe,
    _should_show,
    _signal_tree,
    _start,
    _stop,
    _tag,
    _wait_first,
    main,
)

# Absent on Windows; posix _stop tests need it as a resolvable attribute.
_SIGKILL = getattr(signal, "SIGKILL", 9)


def _run(coro):
    """Run an async test coroutine on a fresh event loop."""
    return asyncio.run(coro)


class _FakeStream:
    """Async readline stream backing ``_pipe``'s stdout/stderr drains."""

    def __init__(self, lines):
        self._lines = list(lines)

    async def readline(self):
        if self._lines:
            return self._lines.pop(0)
        return b""  # EOF


class _FakeProc:
    """Minimal ``asyncio.subprocess.Process`` stand-in.

    ``wait_behaviors`` is a FIFO of ``("return", rc)`` | ``("raise", exc)`` |
    ``("block",)``; each ``wait()`` call pops one. The default (empty) behavior
    blocks forever so ``wait_for`` times out — exercising the SIGKILL path.
    """

    def __init__(self, *, pid=1, returncode=None, wait_behaviors=None, stdout=None, stderr=None):
        self.pid = pid
        self.returncode = returncode
        self._behaviors = list(wait_behaviors or [])
        self.stdout = stdout
        self.stderr = stderr

    async def wait(self):
        if self._behaviors:
            kind, payload = self._behaviors.pop(0)
        else:
            kind, payload = "block", None
        if kind == "return":
            self.returncode = payload
            return payload
        if kind == "raise":
            raise payload
        await asyncio.Event().wait()  # block forever
        return 0


# ------------------------------------------------------------- _group_kwargs --


class TestGroupKwargs:
    def test_nt_uses_new_process_group(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        # ``CREATE_NEW_PROCESS_GROUP`` is a Windows-only attribute of
        # ``subprocess``; on POSIX hosts we have to inject a placeholder so
        # the lookup inside ``_group_kwargs`` doesn't raise AttributeError.
        fake_flag = 0x200
        monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", fake_flag, raising=False)
        kw = _group_kwargs()
        assert kw["creationflags"] == fake_flag
        assert "start_new_session" not in kw

    def test_posix_uses_new_session(self, monkeypatch):
        monkeypatch.setattr(os, "name", "posix")
        kw = _group_kwargs()
        assert kw["start_new_session"] is True
        assert "creationflags" not in kw


# ---------------------------------------------------------- _frontend_command --


class TestFrontendCommand:
    def test_nt_routes_through_cmd(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        assert _frontend_command() == ["cmd", "/c", "npm", "run", "dev"]

    def test_posix_invokes_npm_directly(self, monkeypatch):
        monkeypatch.setattr(os, "name", "posix")
        assert _frontend_command() == ["npm", "run", "dev"]


# ------------------------------------------------------------ _get_bind_host --


class TestGetBindHost:
    """Covers all three branches of the loopback detection.

    The happy path (IPv4 available) runs on every real machine; the two
    fallback branches need ``socket.socket`` mocked to raise ``OSError``
    on ``bind()`` for specific address families.
    """

    def test_returns_ipv4_when_available(self):
        # Default path: 127.0.0.1 binds successfully.
        assert _get_bind_host() == "127.0.0.1"

    def test_returns_ipv6_when_ipv4_unavailable(self, monkeypatch):
        real_socket = __import__("socket").socket
        real_af_inet = __import__("socket").AF_INET

        def fake_socket(family, stype):
            if family == real_af_inet:
                # Pretend IPv4 stack is disabled — bind() fails.
                s = MagicMock()
                s.bind.side_effect = OSError("IPv4 disabled")
                return s
            # IPv6 socket works normally.
            return real_socket(family, stype)

        monkeypatch.setattr(launcher.socket, "socket", fake_socket)
        assert _get_bind_host() == "::1"

    def test_returns_ipv4_literal_when_both_unavailable(self, monkeypatch):
        # Both families fail — return the literal "127.0.0.1" so the caller
        # fails loudly at bind time with a clear error instead of crashing
        # inside detection.
        def fake_socket(family, stype):
            s = MagicMock()
            s.bind.side_effect = OSError("no loopback")
            return s

        monkeypatch.setattr(launcher.socket, "socket", fake_socket)
        assert _get_bind_host() == "127.0.0.1"


# ----------------------------------------------------------------------- _tag --


class TestTag:
    def test_plain_when_not_a_tty(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        assert _tag("backend") == "[backend]"

    def test_colored_for_known_labels_when_tty(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        assert _tag("backend") == "\033[36m[backend]\033[0m"
        assert _tag("frontend") == "\033[35m[frontend]\033[0m"

    def test_unknown_label_uses_color_default(self, monkeypatch):
        # colors.get(label, "") -> "" for unknown -> no color code, reset present
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        assert _tag("mystery") == "[mystery]\033[0m"


# --------------------------------------------------------------- _should_show --


class TestShouldShow:
    """Covers every branch of the startup-banner filter (main.py ``_should_show``).

    ``_pipe``'s other tests only feed non-banner lines, so the four ``return``
    branches (VERBOSE / blank / vite-banner / uvicorn-banner) and the keep path
    are exercised directly here. Branch coverage also demands the ``and``
    short-circuit on a non-banner ``INFO:`` line and the regression guard for
    the removed ``"ready in"`` fragment (it had matched ``"...already in use"``).
    """

    def test_verbose_short_circuits_and_keeps_banners(self, monkeypatch):
        # Line 198: _VERBOSE True -> return True for EVERYTHING, even banners.
        monkeypatch.setattr(launcher, "_VERBOSE", True)
        assert _should_show("INFO:     Uvicorn running on http://127.0.0.1:8000") is True
        assert _should_show("  VITE v6.4.3  ready in 306 ms") is True
        assert _should_show("") is True

    def test_blank_line_dropped(self):
        # Line 200: empty text -> return False.
        assert _should_show("") is False

    def test_vite_banner_fragments_dropped(self):
        # Line 202: any vite banner fragment -> return False.
        for line in (
            "  VITE v6.4.3  ready in 306 ms",
            "  ➜  Local:   http://localhost:5173/",
            "  ➜  Network: use --host to expose",
            "  ➜  press h + enter to show help",
        ):
            assert _should_show(line) is False, repr(line)

    def test_uvicorn_banner_dropped(self):
        # Line 207: INFO:-prefixed uvicorn startup/shutdown -> return False.
        for line in (
            "INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)",
            "INFO:     Application startup complete.",
            "INFO:     Waiting for application startup.",
            "INFO:     Shutting down",
        ):
            assert _should_show(line) is False, repr(line)

    def test_runtime_lines_kept(self):
        # Line 208 (return True): access logs + every error level pass through.
        assert _should_show('INFO:     127.0.0.1:54005 - "GET /api/jobs HTTP/1.1" 200 OK') is True
        assert _should_show("ERROR:    something broke") is True
        assert _should_show("WARNING:  low memory") is True

    def test_info_line_without_banner_keyword_kept(self):
        # The ``and`` in the uvicorn check: INFO: prefix present but no banner
        # keyword -> must NOT be dropped (a custom app INFO log survives).
        assert _should_show("INFO:     custom application message") is True

    def test_already_in_use_not_falsely_dropped(self):
        # Regression: the removed "ready in" fragment had matched this. The
        # ERROR line must survive so a real bind failure is never hidden.
        assert _should_show("ERROR:    [Errno 98] Address already in use") is True


# ---------------------------------------------------------------- _signal_tree --


class TestSignalTree:
    def test_nt_invokes_taskkill_tree(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        run = MagicMock()
        monkeypatch.setattr(subprocess, "run", run)
        _signal_tree(_FakeProc(pid=4242), signal.SIGTERM)
        assert run.call_args[0][0] == [
            "taskkill",
            "/T",
            "/F",
            "/PID",
            "4242",
        ]

    def test_posix_signals_group(self, monkeypatch):
        monkeypatch.setattr(os, "name", "posix")
        killpg = MagicMock()
        # os.killpg is POSIX-only (absent on Windows) -> raising=False lets us
        # inject it for the posix branch.
        monkeypatch.setattr(os, "killpg", killpg, raising=False)
        _signal_tree(_FakeProc(pid=77), 15)
        killpg.assert_called_once_with(77, 15)

    def test_posix_swallows_oserror(self, monkeypatch):
        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setattr(os, "killpg", MagicMock(side_effect=OSError("gone")), raising=False)
        _signal_tree(_FakeProc(pid=77), 15)  # must not raise


# ----------------------------------------------------------------------- _stop --


class TestStop:
    def test_already_exited_signals_nothing(self, monkeypatch):
        sig = MagicMock()
        monkeypatch.setattr(launcher, "_signal_tree", sig)
        _run(_stop(_FakeProc(returncode=0), grace=0.1))
        sig.assert_not_called()

    def test_nt_exits_within_grace_skips_sigkill(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        sig = MagicMock()
        monkeypatch.setattr(launcher, "_signal_tree", sig)
        proc = _FakeProc(returncode=None, wait_behaviors=[("return", 0)])
        _run(_stop(proc, grace=1))
        assert [c.args[1] for c in sig.call_args_list] == [signal.SIGTERM]

    def test_posix_timeout_escalates_to_sigkill(self, monkeypatch):
        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setattr(signal, "SIGKILL", _SIGKILL, raising=False)
        sig = MagicMock()
        monkeypatch.setattr(launcher, "_signal_tree", sig)
        proc = _FakeProc(
            returncode=None,
            wait_behaviors=[("block", None), ("return", -9)],
        )
        _run(_stop(proc, grace=0.15))
        assert [c.args[1] for c in sig.call_args_list] == [signal.SIGTERM, _SIGKILL]

    def test_posix_sigkill_wait_exception_swallowed(self, monkeypatch):
        # Covers the inner `except (CancelledError, Exception)` after SIGKILL.
        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setattr(signal, "SIGKILL", _SIGKILL, raising=False)
        monkeypatch.setattr(launcher, "_signal_tree", MagicMock())
        proc = _FakeProc(
            returncode=None,
            wait_behaviors=[("block", None), ("raise", RuntimeError("boom"))],
        )
        _run(_stop(proc, grace=0.15))  # inner except swallows -> no raise

    def test_nt_timeout_skips_sigkill_fallthrough(self, monkeypatch):
        # Covers the FALSE branch of `if os.name != "nt":` (skip, fall to
        # implicit end-of-function) — the `87->exit` partial in coverage.
        monkeypatch.setattr(os, "name", "nt")
        sig = MagicMock()
        monkeypatch.setattr(launcher, "_signal_tree", sig)
        proc = _FakeProc(returncode=None, wait_behaviors=[("block", None)])
        _run(_stop(proc, grace=0.15))
        # Only the initial SIGTERM-equivalent taskkill ran; no SIGKILL path.
        assert sig.call_count == 1


# ----------------------------------------------------------------------- _pipe --


class TestPipe:
    def test_drains_stdout_and_stderr_tagged(self, capsys):
        proc = _FakeProc(
            stdout=_FakeStream([b"out-a\n", b"out-b\n"]),
            stderr=_FakeStream([b"err-a\n"]),
        )
        _run(_pipe(proc, "backend"))
        out = capsys.readouterr().out
        assert "[backend] out-a" in out
        assert "[backend] out-b" in out
        assert "[backend] err-a" in out

    def test_none_streams_filtered_from_readers(self, capsys):
        # stderr=None exercises the `if s is not None` False branch.
        proc = _FakeProc(stdout=_FakeStream([b"only-line\n"]), stderr=None)
        _run(_pipe(proc, "frontend"))
        assert "[frontend] only-line" in capsys.readouterr().out

    def test_banner_and_blank_lines_are_skipped(self, capsys):
        # Line 222 (`continue`): startup banners and blank lines return False
        # from _should_show and are never written to stdout — only the real
        # runtime line survives. Exercises the filter end-to-end via _pipe.
        proc = _FakeProc(
            stdout=_FakeStream(
                [
                    b"  VITE v6.4.3  ready in 306 ms\n",  # vite banner -> skip
                    b"\n",  # blank spacer -> skip
                    b"  \xe2\x9e\x9c  Local:   http://localhost:5173/\n",  # ➜ -> skip
                    b"real output\n",  # kept
                    b"INFO:     Uvicorn running on http://127.0.0.1:8000\n",  # skip
                ]
            ),
            stderr=None,
        )
        _run(_pipe(proc, "frontend"))
        out = capsys.readouterr().out
        assert "[frontend] real output" in out
        assert "VITE v6" not in out
        assert "Uvicorn running" not in out
        assert "Local:" not in out
        # No bare spacer line emitted for the blank input.
        assert "[frontend] \n" not in out


# --------------------------------------------------------------- _wait_first --


class TestWaitFirst:
    def test_returns_first_exit_and_cancels_pending(self):
        first = _FakeProc(returncode=None, wait_behaviors=[("return", 1)])
        blocker = _FakeProc(returncode=None)  # blocks forever
        assert _run(_wait_first(first, blocker)) is first

    def test_all_done_leaves_pending_empty(self):
        a = _FakeProc(returncode=None, wait_behaviors=[("return", 0)])
        b = _FakeProc(returncode=None, wait_behaviors=[("return", 0)])
        assert _run(_wait_first(a, b)) in (a, b)


# ---------------------------------------------------------------------- _start --


class TestStart:
    def test_spawns_frontend_then_backend_after_readiness(self, monkeypatch, capsys):
        calls = []

        async def fake_exec(*args, **kwargs):
            calls.append((args, kwargs))
            return _FakeProc(pid=len(calls))

        async def fake_open_conn(host, port):
            # Writer with close()/wait_closed() — mirrors asyncio.StreamWriter.
            writer = MagicMock()

            async def wait_closed():
                pass

            writer.wait_closed = wait_closed
            return MagicMock(), writer

        async def fake_pipe(proc, label):
            return None

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(asyncio, "open_connection", fake_open_conn)
        monkeypatch.setattr(launcher, "_pipe", fake_pipe)

        backend, frontend, frontend_pipe = _run(_start())

        # Frontend spawned FIRST (backend waits for readiness).
        f_args, f_kw = calls[0]
        assert f_args == tuple(_frontend_command())
        assert f_kw["cwd"] == str(FRONTEND_DIR)

        # Backend spawned AFTER readiness probe succeeds.
        b_args, b_kw = calls[1]
        assert b_args[0] == sys.executable
        assert b_args[1] == "main.py"
        assert b_kw["cwd"] == str(BACKEND_DIR)

        assert backend is not frontend
        # Caller owns the frontend pipe task for cancellation on shutdown.
        assert isinstance(frontend_pipe, asyncio.Task)
        assert "frontend ready on :5173" in capsys.readouterr().out

    def test_raises_when_frontend_exits_before_ready(self, monkeypatch):
        """Frontend crashes during readiness wait → RuntimeError, no backend."""

        dead = _FakeProc(pid=9, returncode=None)

        async def fake_exec(*args, **kwargs):
            # After spawn, simulate the frontend dying immediately.
            dead.returncode = 42
            return dead

        async def failing_open_conn(host, port):
            raise OSError("connection refused")

        async def fake_stop(proc, *, grace=5.0):
            return None

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(asyncio, "open_connection", failing_open_conn)
        monkeypatch.setattr(launcher, "_stop", fake_stop)
        # Freeze time so the deadline loop can iterate.
        t = [0.0]
        monkeypatch.setattr(time, "monotonic", lambda: t.__getitem__(0))

        async def fast_sleep(_):
            t[0] += 0.1

        monkeypatch.setattr(asyncio, "sleep", fast_sleep)

        with pytest.raises(RuntimeError, match="frontend exited.*code 42"):
            _run(_start())

    def test_raises_when_readiness_times_out(self, monkeypatch):
        """Frontend never becomes ready → RuntimeError after 60s deadline."""

        alive = _FakeProc(pid=7)  # returncode stays None → appears alive

        async def fake_exec(*args, **kwargs):
            return alive

        async def failing_open_conn(host, port):
            raise OSError("connection refused")

        async def fake_stop(proc, *, grace=5.0):
            return None

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(asyncio, "open_connection", failing_open_conn)
        monkeypatch.setattr(launcher, "_stop", fake_stop)
        # Fast-forward time past the 60s deadline on each monotonic() read.
        t = [0.0]
        monkeypatch.setattr(time, "monotonic", lambda: t.__getitem__(0))

        async def fast_sleep(_):
            t[0] += 10.0

        monkeypatch.setattr(asyncio, "sleep", fast_sleep)

        with pytest.raises(RuntimeError, match="not ready on port 5173 within 60s"):
            _run(_start())

    def test_probe_tries_fallback_hosts_in_order(self, monkeypatch):
        """When ``localhost`` fails, the probe must fall back to 127.0.0.1
        then ``::1``. Verify the sequence and that the FIRST successful host
        short-circuits the rest.
        """
        attempts = []

        async def fake_exec(*args, **kwargs):
            return _FakeProc(pid=1)

        async def selective_open_conn(host, port):
            attempts.append(host)
            if host == "localhost":
                raise OSError("mock DNS failure")
            # 127.0.0.1 succeeds.
            writer = MagicMock()

            async def wait_closed():
                pass

            writer.wait_closed = wait_closed
            return MagicMock(), writer

        async def fake_pipe(proc, label):
            return None

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(asyncio, "open_connection", selective_open_conn)
        monkeypatch.setattr(launcher, "_pipe", fake_pipe)

        _run(_start())
        # localhost failed, 127.0.0.1 succeeded → ::1 was never tried.
        assert attempts == ["localhost", "127.0.0.1"]

    def test_probe_wait_closed_exception_logged_to_stderr(self, monkeypatch, capsys):
        """``writer.wait_closed()`` raising must not break the probe, but
        MUST log a diagnostic to stderr — silent ``pass`` would hide real
        close-handshake failures from operators.
        """

        async def fake_exec(*args, **kwargs):
            return _FakeProc(pid=1)

        async def fake_open_conn(host, port):
            writer = MagicMock()

            async def wait_closed():
                raise RuntimeError("simulated close failure")

            writer.wait_closed = wait_closed
            return MagicMock(), writer

        async def fake_pipe(proc, label):
            return None

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(asyncio, "open_connection", fake_open_conn)
        monkeypatch.setattr(launcher, "_pipe", fake_pipe)

        # Should succeed (not raise) because wait_closed's exception is
        # logged but does not propagate.
        _run(_start())
        captured = capsys.readouterr()
        assert "frontend ready on :5173" in captured.out
        # Diagnostic line must name the host:port and the exception type.
        assert "probe: wait_closed failed on localhost:5173" in captured.err
        assert "RuntimeError" in captured.err
        assert "simulated close failure" in captured.err


# ---------------------------------------------------------------------- main --


class TestMain:
    @pytest.fixture
    def stub_children(self, monkeypatch):
        """Patch _start/_stop/_pipe so main() is tested in isolation."""
        backend = _FakeProc(pid=1, returncode=None)
        frontend = _FakeProc(pid=2, returncode=None)

        async def fake_pipe(proc, label):
            return None

        async def fake_start():
            # _start() now returns (backend, frontend, frontend_pipe);
            # main() puts frontend_pipe into its pipes list for cancellation.
            return backend, frontend, asyncio.create_task(fake_pipe(frontend, "frontend"))

        async def fake_stop(proc, *, grace=5.0):
            return None

        monkeypatch.setattr(launcher, "_start", fake_start)
        monkeypatch.setattr(launcher, "_pipe", fake_pipe)
        monkeypatch.setattr(launcher, "_stop", fake_stop)
        return backend, frontend

    def test_backend_exit_returns_its_code(self, monkeypatch, stub_children, capsys):
        backend, _ = stub_children
        backend.returncode = 1

        async def fake_wait_first(*procs):
            return backend

        monkeypatch.setattr(launcher, "_wait_first", fake_wait_first)
        assert _run(main()) == 1
        assert "backend exited (code 1)" in capsys.readouterr().out

    def test_frontend_exit_labels_frontend(self, monkeypatch, stub_children, capsys):
        _, frontend = stub_children
        frontend.returncode = 2

        async def fake_wait_first(*procs):
            return frontend

        monkeypatch.setattr(launcher, "_wait_first", fake_wait_first)
        assert _run(main()) == 2
        assert "frontend exited (code 2)" in capsys.readouterr().out

    def test_cancelled_returns_zero_and_stops_both(self, monkeypatch, stub_children):
        backend, frontend = stub_children
        stops = []

        async def fake_wait_first(*procs):
            raise asyncio.CancelledError()

        async def fake_stop(proc, *, grace=5.0):
            stops.append(proc)

        monkeypatch.setattr(launcher, "_wait_first", fake_wait_first)
        monkeypatch.setattr(launcher, "_stop", fake_stop)
        assert _run(main()) == 0
        assert set(stops) == {backend, frontend}

    def test_none_returncode_falls_back_to_zero(self, monkeypatch, stub_children):
        backend, _ = stub_children
        backend.returncode = None  # exited-but-unknown -> rc None -> else 0

        async def fake_wait_first(*procs):
            return backend

        monkeypatch.setattr(launcher, "_wait_first", fake_wait_first)
        assert _run(main()) == 0

    def test_cancelled_during_shutdown_covered_at_lines_205(self, monkeypatch, stub_children):
        # Covers the `except asyncio.CancelledError: pass` in finally (lines
        # 205-206). Patch asyncio.shield to raise CancelledError after running
        # the inner gather — simulates external cancellation during shutdown.
        backend, _ = stub_children
        backend.returncode = 1
        stops = []

        async def fake_wait_first(*procs):
            return backend

        async def fake_stop(proc, *, grace=5.0):
            stops.append(proc)
            return None

        async def cancelling_shield(coro):
            try:
                await coro  # let the inner gather run to completion
            except BaseException:
                pass
            raise asyncio.CancelledError()

        monkeypatch.setattr(launcher, "_wait_first", fake_wait_first)
        monkeypatch.setattr(launcher, "_stop", fake_stop)
        monkeypatch.setattr(asyncio, "shield", cancelling_shield)
        rc = _run(main())
        # rc was set to 1 in the try block before finally; the shield's
        # CancelledError is caught inside finally and doesn't change rc.
        assert rc == 1
        assert set(stops) == {backend, stub_children[1]}


class TestDynamicHostBinding:
    """Tests for MSMODELING_BACKEND_HOST and MSMODELING_FRONTEND_HOST handling.

    Recent changes added dynamic host detection and display:
    - _start() detects host via _get_bind_host() and passes it to both children
    - Both MSMODELING_BACKEND_HOST and MSMODELING_FRONTEND_HOST are set
    - main() reads these vars and displays actual addresses (not hardcoded localhost)
    - IPv6 addresses are formatted with brackets in URLs
    """

    @pytest.fixture
    def stub_children(self, monkeypatch):
        """Patch _start/_stop/_pipe so main() is tested in isolation."""
        backend = _FakeProc(pid=1, returncode=None)
        frontend = _FakeProc(pid=2, returncode=None)

        async def fake_pipe(proc, label):
            return None

        async def fake_start():
            # _start() now returns (backend, frontend, frontend_pipe);
            # main() puts frontend_pipe into its pipes list for cancellation.
            return backend, frontend, asyncio.create_task(fake_pipe(frontend, "frontend"))

        async def fake_stop(proc, *, grace=5.0):
            return None

        monkeypatch.setattr(launcher, "_start", fake_start)
        monkeypatch.setattr(launcher, "_stop", fake_stop)
        return backend, frontend

    def test_start_passes_both_host_env_vars_to_frontend(self, monkeypatch):
        """_start() must pass MSMODELING_BACKEND_HOST and MSMODELING_FRONTEND_HOST
        to the frontend subprocess with the detected host value.
        """
        captured_env = {}

        async def fake_exec(*args, **kwargs):
            # Capture the env passed to frontend spawn
            if "env" in kwargs:
                captured_env.update(kwargs["env"])
            return _FakeProc(pid=1)

        async def fake_open_conn(host, port):
            writer = MagicMock()

            async def wait_closed():
                pass

            writer.wait_closed = wait_closed
            return MagicMock(), writer

        async def fake_pipe(proc, label):
            return None

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(asyncio, "open_connection", fake_open_conn)
        monkeypatch.setattr(launcher, "_pipe", fake_pipe)
        # Force _get_bind_host to return a known value
        monkeypatch.setattr(launcher, "_get_bind_host", lambda: "127.0.0.1")

        _run(_start())

        # Both env vars must be set to the detected host
        assert captured_env.get("MSMODELING_BACKEND_HOST") == "127.0.0.1"
        assert captured_env.get("MSMODELING_FRONTEND_HOST") == "127.0.0.1"

    def test_start_passes_both_host_env_vars_to_backend(self, monkeypatch):
        """_start() must pass MSMODELING_BACKEND_HOST to the backend subprocess."""
        captured_calls = []

        async def fake_exec(*args, **kwargs):
            captured_calls.append((args, kwargs))
            return _FakeProc(pid=len(captured_calls))

        async def fake_open_conn(host, port):
            writer = MagicMock()

            async def wait_closed():
                pass

            writer.wait_closed = wait_closed
            return MagicMock(), writer

        async def fake_pipe(proc, label):
            return None

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(asyncio, "open_connection", fake_open_conn)
        monkeypatch.setattr(launcher, "_pipe", fake_pipe)
        monkeypatch.setattr(launcher, "_get_bind_host", lambda: "::1")

        _run(_start())

        # Second call is backend spawn
        assert len(captured_calls) >= 2
        backend_args, backend_kw = captured_calls[1]
        assert backend_kw.get("env", {}).get("MSMODELING_BACKEND_HOST") == "::1"

    def test_start_uses_same_host_for_frontend_and_backend(self, monkeypatch):
        """Both frontend and backend must use the same detected host for consistency."""
        detected_host = "192.168.1.100"  # Simulate custom host detection
        captured_envs = []

        async def fake_exec(*args, **kwargs):
            if "env" in kwargs:
                captured_envs.append(kwargs["env"])
            return _FakeProc(pid=len(captured_envs))

        async def fake_open_conn(host, port):
            writer = MagicMock()

            async def wait_closed():
                pass

            writer.wait_closed = wait_closed
            return MagicMock(), writer

        async def fake_pipe(proc, label):
            return None

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr(asyncio, "open_connection", fake_open_conn)
        monkeypatch.setattr(launcher, "_pipe", fake_pipe)
        monkeypatch.setattr(launcher, "_get_bind_host", lambda: detected_host)

        _run(_start())

        # Frontend env (first call)
        frontend_env = captured_envs[0]
        assert frontend_env.get("MSMODELING_BACKEND_HOST") == detected_host
        assert frontend_env.get("MSMODELING_FRONTEND_HOST") == detected_host

    def test_main_displays_detected_backend_host(self, monkeypatch, stub_children, capsys):
        """main() must display the detected backend host, not hardcoded localhost."""
        backend, _ = stub_children
        backend.returncode = 0

        async def fake_wait_first(*procs):
            return backend

        monkeypatch.setattr(launcher, "_wait_first", fake_wait_first)
        # Set the env var that _start() would have set
        monkeypatch.setenv("MSMODELING_BACKEND_HOST", "10.0.0.1")
        monkeypatch.setenv("MSMODELING_FRONTEND_HOST", "10.0.0.1")

        _run(main())
        output = capsys.readouterr().out

        # Must show the actual host, not "localhost"
        assert "http://10.0.0.1:8000" in output
        assert "localhost" not in output or "http://10.0.0.1:5173" in output

    def test_main_formats_ipv6_with_brackets(self, monkeypatch, stub_children, capsys):
        """IPv6 addresses must be formatted with brackets in URLs."""
        backend, _ = stub_children
        backend.returncode = 0

        async def fake_wait_first(*procs):
            return backend

        monkeypatch.setattr(launcher, "_wait_first", fake_wait_first)
        monkeypatch.setenv("MSMODELING_BACKEND_HOST", "::1")
        monkeypatch.setenv("MSMODELING_FRONTEND_HOST", "::1")

        _run(main())
        output = capsys.readouterr().out

        # IPv6 must be wrapped in brackets
        assert "http://[::1]:8000" in output
        assert "http://[::1]:5173" in output

    def test_main_formats_ipv4_without_brackets(self, monkeypatch, stub_children, capsys):
        """IPv4 addresses must NOT have brackets."""
        backend, _ = stub_children
        backend.returncode = 0

        async def fake_wait_first(*procs):
            return backend

        monkeypatch.setattr(launcher, "_wait_first", fake_wait_first)
        monkeypatch.setenv("MSMODELING_BACKEND_HOST", "127.0.0.1")
        monkeypatch.setenv("MSMODELING_FRONTEND_HOST", "127.0.0.1")

        _run(main())
        output = capsys.readouterr().out

        # IPv4 without brackets
        assert "http://127.0.0.1:8000" in output
        assert "http://127.0.0.1:5173" in output
        # Must not have brackets around IPv4
        assert "http://[127.0.0.1]" not in output

    def test_main_defaults_to_localhost_when_env_not_set(self, monkeypatch, stub_children, capsys):
        """When env vars are not set, main() should default to 'localhost'."""
        backend, _ = stub_children
        backend.returncode = 0

        async def fake_wait_first(*procs):
            return backend

        monkeypatch.setattr(launcher, "_wait_first", fake_wait_first)
        # Ensure env vars are not set
        monkeypatch.delenv("MSMODELING_BACKEND_HOST", raising=False)
        monkeypatch.delenv("MSMODELING_FRONTEND_HOST", raising=False)

        _run(main())
        output = capsys.readouterr().out

        # Should fall back to localhost
        assert "localhost" in output
