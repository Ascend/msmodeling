"""web_ui dev launcher — start frontend + backend with a single command.

Startup order is sequential, not concurrent:
  1. frontend: ``npm run dev``  (vite, http://localhost:5173)
  2. wait for vite to be ready (TCP probe on port 5173)
  3. backend:  ``python backend/main.py``  (uvicorn, http://localhost:8000)

The frontend MUST come up first because its dev step generates form-schema
JSON files that the backend imports on startup — starting the backend first
would fail with missing-schema errors.

Output from both is interleaved with ``[backend]`` / ``[frontend]`` prefixes.
One-shot startup banners (vite's launch banner, uvicorn's "running on"/startup
chatter) are filtered out by default for a quieter launch; set
``MSMODELING_DEV_VERBOSE=1`` to stream everything raw.
Ctrl+C (or either child exiting) stops BOTH process trees via escalation:
POSIX sends SIGTERM (lets uvicorn run lifespan shutdown → drain in-flight
jobs), waits a grace window, then SIGKILLs stragglers; Windows uses
``taskkill /T /F``. A plain ``terminate()`` would only hit the immediate child,
orphaning the backend's uvicorn worker and the frontend's vite (the
grandchildren holding ports 8000/5173) — signaling the whole group
(``start_new_session``) via the SAVED leader id is what reaches them even after
the direct child has exited.

Run with the repo venv (it owns the backend deps):
    .venv/Scripts/python.exe web_ui/main.py        # Windows
    .venv/bin/python web_ui/main.py                # POSIX

Backend port via ``MSMODELING_PORT`` (default 8000); frontend is vite's 5173.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

_WEB_UI_DIR = Path(__file__).resolve().parent
BACKEND_DIR = _WEB_UI_DIR / "backend"
FRONTEND_DIR = _WEB_UI_DIR / "frontend"


def _get_bind_host() -> str:
    """Detect which loopback address to bind/connect to.

    Mirrors ``web_ui.backend.main.get_bind_address()`` so launcher and
    backend agree on the same host without either importing the other.
    Prefers IPv4 (``127.0.0.1``) — virtually all developer machines have
    IPv4 — and falls back to IPv6 (``::1``) only on the rare host where
    the IPv4 stack is disabled.

    Returns a bare IP literal (no URL brackets). The caller is responsible
    for adding ``[...]`` when embedding in a URL.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        s.close()
        return "127.0.0.1"
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        s.bind(("::1", 0))
        s.close()
        return "::1"
    except OSError:
        return "127.0.0.1"  # let the caller fail with a clear bind error


def _signal_tree(proc: asyncio.subprocess.Process, posix_sig: int) -> None:
    """Signal the child's WHOLE tree, not just the immediate process.

    POSIX: signal the process group led by ``proc.pid``. Because the child was
    started with ``start_new_session=True``, ``proc.pid`` IS that group's id —
    so this still reaches grandchildren (npm→vite, uvicorn) even AFTER the
    direct child has exited (the pgid outlives the leader). This avoids the
    ``os.getpgid(pid)``-at-kill-time race that raises ESRCH once the direct
    child is gone and would skip the kill, orphaning the grandchildren.

    Windows: ``taskkill /T /F`` force-kills the parent-child tree (console
    procs have no SIGTERM analog, so graceful/force collapse to one operation).
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],  # nosec B607
            capture_output=True,
        )
    else:
        try:
            os.killpg(proc.pid, posix_sig)  # pylint: disable=no-member
        except OSError:
            pass  # group empty / already gone — best effort


async def _stop(proc: asyncio.subprocess.Process, *, grace: float = 5.0) -> None:
    """Stop a child and its tree: graceful first, force as a backstop.

    POSIX escalation lets uvicorn run its lifespan shutdown on SIGTERM — that
    drains in-flight jobs (``manager.shutdown(wait=True, worker_timeout=30)`` in
    ``backend/main.py``); a straight SIGKILL would skip it and orphan running
    job workers (each in its OWN session via ``run_module_subprocess``). After
    ``grace`` seconds, any survivor is SIGKILL'd so a child that ignores/hangs
    on SIGTERM can't linger. Windows goes straight to ``taskkill /T /F``.

    The signal sends are synchronous (uninterruptible); only the grace wait can
    be raced by a shutdown cancellation, in which case the force-kill still runs
    on the next line.
    """
    if proc.returncode is not None:
        return  # already exited
    _signal_tree(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace)
        return  # exited gracefully within the grace window
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass  # still alive, or shutdown raced the wait — force below
    if os.name != "nt":  # Windows already force-killed in _signal_tree above
        _signal_tree(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
        try:
            await proc.wait()
        except (asyncio.CancelledError, Exception):  # pylint: disable=broad-except
            pass


def _group_kwargs() -> dict:
    """Flags putting each child in its OWN process group / session.

    On Windows, ``CREATE_NEW_PROCESS_GROUP`` also stops the console's Ctrl+C
    (a CTRL_C_EVENT) from being delivered to the children, so shutdown stays
    under our control via :func:`_tree_kill` rather than racing the OS.
    """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _frontend_command() -> list[str]:
    """``npm`` is ``npm.cmd`` on Windows — route through ``cmd /c`` so we avoid
    ``shell=True`` (keeping the process-group flags clean). The ``cmd`` process
    becomes the tree root that ``taskkill /T`` reaps down to vite.
    """
    if os.name == "nt":
        return ["cmd", "/c", "npm", "run", "dev"]
    return ["npm", "run", "dev"]


def _tag(label: str) -> str:
    """Colored ``[label]`` prefix when stdout is a tty, plain otherwise."""
    if not sys.stdout.isatty():
        return f"[{label}]"
    colors = {"backend": "\033[36m", "frontend": "\033[35m"}  # cyan / magenta
    return f"{colors.get(label, '')}[{label}]\033[0m"


# Suppress one-shot child startup banners for a quieter launch. Off only when
# MSMODELING_DEV_VERBOSE is set truthy ("1"/"true"/"yes") — then everything
# streams through raw, useful when debugging a child that fails to come up.
_VERBOSE = os.environ.get("MSMODELING_DEV_VERBOSE", "").lower() in ("1", "true", "yes")

# Narrow, anchored fragments — vite's launch banner (no level prefix) and
# uvicorn's startup/shutdown chitchat (always ``INFO:``-prefixed). Runtime
# access logs (``INFO: 127.0.0.1 - "GET /api" 200``) and ALL error-level
# output pass through untouched; only these specific banners are dropped.
_VITE_BANNER_FRAGMENTS = (
    "VITE v",  # "  VITE v6.4.3  ready in 306 ms" (covers the ready line)
    "press h + enter",  # vite hint line
    "use --host",  # vite "Network: use --host to expose"
    "➜",  # vite ➜-prefixed Local/Network/Hints lines
)
_UVICORN_BANNER_FRAGMENTS = (
    "uvicorn running on",
    "started reloader process",
    "started server process",
    "finished server process",
    "waiting for application startup",
    "application startup complete",
    "waiting for application shutdown",
    "application shutdown complete",
    "shutting down",
)


def _should_show(text: str) -> bool:
    """Pass-through filter for a child stdout/stderr line.

    Drops blank lines (the ``[frontend]``-only spacer lines are pure noise)
    and the one-shot startup banners, so the terminal isn't littered with
    launch progress. Anything runtime — access logs, warnings, tracebacks —
    is kept verbatim. ``MSMODELING_DEV_VERBOSE`` disables all filtering.
    """
    if _VERBOSE:
        return True
    if not text:
        return False
    if any(frag in text for frag in _VITE_BANNER_FRAGMENTS):
        return False
    stripped = text.lstrip()
    if stripped.startswith("INFO:") and any(frag in stripped.lower() for frag in _UVICORN_BANNER_FRAGMENTS):
        return False
    return True


async def _pipe(proc: asyncio.subprocess.Process, label: str) -> None:
    """Forward a child's stdout+stderr to the console, line by line, tagged."""

    async def _drain(stream: asyncio.StreamReader) -> None:
        tag = _tag(label)
        while True:
            line = await stream.readline()
            if not line:
                return
            text = line.decode("utf-8", "replace").rstrip()
            if not _should_show(text):
                continue
            sys.stdout.write(f"{tag} {text}\n")
            sys.stdout.flush()

    readers = [s for s in (proc.stdout, proc.stderr) if s is not None]
    await asyncio.gather(*(_drain(s) for s in readers))


async def _wait_first(
    *procs: asyncio.subprocess.Process,
) -> asyncio.subprocess.Process:
    """Return the first process that exits (cancels the other wait tasks)."""
    waits = {asyncio.create_task(p.wait()): p for p in procs}
    done, pending = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    return waits[next(iter(done))]


async def _start() -> tuple[
    asyncio.subprocess.Process,
    asyncio.subprocess.Process,
    asyncio.Task,
]:
    """Start frontend first, wait for vite to be ready, then start backend.

    Frontend is started first because the backend depends on form-schema files
    generated by the frontend's dev step. We wait for vite to accept TCP
    connections on port 5173 (which implies the dev server and its pre-route
    generation hooks have finished) before launching the backend, so the
    backend can read those generated files on import.

    The frontend's stdout/stderr is piped in a background task while we wait,
    so vite's progress output (including any schema-generation errors) is
    visible during the readiness wait. The caller is responsible for including
    this task in its cancellation list so the pipe drains cleanly on shutdown.

    Returns ``(backend, frontend, frontend_pipe)``.
    """
    # Tell vite which loopback the backend will bind to, so its /api proxy
    # target matches. Without this, vite uses ``localhost`` which normally
    # works via Node's DNS but fails on machines where ``localhost`` is
    # unresolvable (broken /etc/hosts, minimal container, etc.). Backend
    # runs the same detection independently in ``get_bind_address()`` —
    # the two must agree, so they share the same algorithm.
    backend_host = _get_bind_host()
    # Use the same host for frontend binding for consistency
    frontend_host = backend_host
    frontend_env = {
        **os.environ,
        "MSMODELING_BACKEND_HOST": backend_host,
        "MSMODELING_FRONTEND_HOST": frontend_host,
    }

    # Store detected hosts for main() to display
    os.environ["MSMODELING_BACKEND_HOST"] = backend_host
    os.environ["MSMODELING_FRONTEND_HOST"] = frontend_host

    frontend = await asyncio.create_subprocess_exec(
        *_frontend_command(),
        cwd=str(FRONTEND_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=frontend_env,
        **_group_kwargs(),
    )
    frontend_pipe = asyncio.create_task(_pipe(frontend, "frontend"))

    # Wait for vite to accept connections before starting backend.
    # Try ``localhost`` first (matches vite's default binding and follows
    # the system's IPv4/IPv6 preference via getaddrinfo), then fall back
    # to explicit loopback literals so a broken DNS/``/etc/hosts`` can't
    # deadlock the probe: on a machine where ``localhost`` fails to
    # resolve, the fallbacks still hit whichever stack vite actually
    # bound to. Each iteration of the outer loop retries all three.
    port = 5173
    deadline = time.monotonic() + 60.0
    ready = False
    while time.monotonic() < deadline:
        for host in ("localhost", "127.0.0.1", "::1"):
            try:
                reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=2.0)
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception as exc:  # pylint: disable=broad-except
                    # Benign — the TCP probe itself succeeded, only the
                    # close handshake failed (e.g. server reset). Log to
                    # stderr so operators have a breadcrumb if the probe
                    # ever misbehaves; don't let it break the readiness
                    # check, which already has its signal.
                    print(
                        f"  probe: wait_closed failed on {host}:{port}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                ready = True
                break
            except (OSError, asyncio.TimeoutError):
                continue  # try next host in the fallback list
        if ready:
            break
        if frontend.returncode is not None:
            raise RuntimeError(f"frontend exited (code {frontend.returncode}) before becoming ready on port {port}")
        await asyncio.sleep(1.0)

    if not ready:
        await _stop(frontend)
        raise RuntimeError(f"frontend not ready on port {port} within 60s (tried localhost, 127.0.0.1, ::1)")

    print(f"  frontend ready on :{port}, starting backend...", flush=True)

    # Pass detected backend_host to backend process
    backend_env = {**os.environ, "MSMODELING_BACKEND_HOST": backend_host}

    backend = await asyncio.create_subprocess_exec(
        sys.executable,
        "main.py",
        cwd=str(BACKEND_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=backend_env,
        **_group_kwargs(),
    )
    return backend, frontend, frontend_pipe


async def main() -> int:
    """Run both servers until Ctrl+C or one exits, then tree-kill both."""
    backend, frontend, frontend_pipe = await _start()
    backend_port = os.environ.get("MSMODELING_PORT", "8000")
    backend_host = os.environ.get("MSMODELING_BACKEND_HOST", "localhost")
    frontend_host = os.environ.get("MSMODELING_FRONTEND_HOST", "localhost")

    # Format addresses for display (handle IPv6)
    backend_addr = f"[{backend_host}]" if ":" in backend_host else backend_host
    frontend_addr = f"[{frontend_host}]" if ":" in frontend_host else frontend_host

    print(
        "\n  msmodeling web_ui dev\n"
        f"    backend  -> http://{backend_addr}:{backend_port}\n"
        f"    frontend -> http://{frontend_addr}:5173\n"
        "  (Ctrl+C stops both)\n"
        "\n"
        "  WARNING: no authentication - loopback only blocks remote\n"
        "    access. Any local user on this machine can call all APIs.\n"
        "    Single-user / single-machine use ONLY. Do NOT run on shared hosts.\n",
        flush=True,
    )
    pipes = [
        asyncio.create_task(_pipe(backend, "backend")),
        frontend_pipe,
    ]
    rc: int | None = 0
    try:
        exited = await _wait_first(backend, frontend)
        label = "backend" if exited is backend else "frontend"
        rc = exited.returncode
        print(
            f"\n{label} exited (code {rc}); stopping the other...",
            flush=True,
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        rc = 0  # clean Ctrl+C shutdown
    finally:
        # Graceful-then-force stop for both trees (POSIX: SIGTERM → grace →
        # SIGKILL; Windows: taskkill /T /F). Shielded so a second Ctrl+C racing
        # into shutdown can't skip the force-kill backstop.
        try:
            await asyncio.shield(asyncio.gather(_stop(backend), _stop(frontend), return_exceptions=True))
        except asyncio.CancelledError:
            pass
        for task in pipes:
            task.cancel()
    return rc if rc is not None else 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(0)
