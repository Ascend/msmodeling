"""Non-interactive bootstrap for ``python build.py`` / ``python build.py test``.

Strongly depends on ``uv``. Missing uv → WARNING + non-interactive pip install.
Network/permission failures exit without a venv/pip build fallback.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from typing import Final, Literal, NoReturn

from scripts.helpers._paths import REPO_ROOT

Mode = Literal["build", "test"]

_MIN_PYTHON: Final = (3, 10)
_SYNC_TIMEOUT_SECONDS: Final = 3600
_PIP_INSTALL_TIMEOUT_SECONDS: Final = 600

logger: Final = logging.getLogger("build.bootstrap")


def _noninteractive_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base if base is not None else os.environ)
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PIP_NO_INPUT", "1")
    env.setdefault("UV_NO_PROGRESS", "1")
    # Avoid prompts from tools that honor CI.
    env.setdefault("CI", "1")
    return env


def _fail(message: str, *, code: int = 1) -> NoReturn:
    logger.error("%s", message)
    raise SystemExit(code)


def _uv_beside_interpreter() -> str | None:
    scripts_dir = os.path.dirname(os.path.abspath(sys.executable))
    candidate = os.path.join(scripts_dir, "uv.exe" if os.name == "nt" else "uv")
    if os.path.isfile(candidate):
        return candidate
    return None


def fail_fast(*, mode: Mode) -> None:
    """Exit early on missing prerequisites (before install/sync)."""
    if sys.version_info < _MIN_PYTHON:
        _fail(
            f"Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ required; "
            f"got {sys.version_info.major}.{sys.version_info.minor}",
        )

    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        _fail(f"missing required file: {pyproject}")

    lock = REPO_ROOT / "uv.lock"
    if not lock.is_file():
        _fail(f"missing required file: {lock}")

    script_name = "build.sh" if mode == "build" else "run_ci_gate.sh"
    script = REPO_ROOT / "scripts" / script_name
    if not script.is_file():
        _fail(f"missing required script: {script}")


def ensure_uv() -> str:
    """Return path to ``uv``. If missing, WARNING then non-interactive install."""
    existing = shutil.which("uv")
    if existing is not None:
        return existing

    logger.warning(
        "uv not found in PATH; installing non-interactively via pip (default index). "
        "If download is slow, configure indexes yourself "
        "(e.g. PIP_INDEX_URL for pip; UV_INDEX_URL / UV_DEFAULT_INDEX for uv sync). "
        "build.py requires uv and will not fall back to venv/pip build.",
    )
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "uv",
    ]
    env = _noninteractive_env()
    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=_PIP_INSTALL_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        _fail(f"failed to install uv (OS error): {exc}")
    except subprocess.TimeoutExpired:
        _fail("failed to install uv: pip install timed out")

    if completed.returncode != 0:
        _fail(
            f"failed to install uv (exit {completed.returncode}); "
            "fix network/permissions and retry. No secondary fallback.",
            code=completed.returncode or 1,
        )

    installed = shutil.which("uv")
    if installed is not None:
        return installed
    beside = _uv_beside_interpreter()
    if beside is not None:
        return beside
    _fail(
        "uv executable not found after installation. Ensure the Python Scripts directory is on PATH.",
    )


def ensure_deps(mode: Mode, *, uv_path: str) -> None:
    """``uv sync --frozen --group build|ci`` for the requested mode."""
    group = "build" if mode == "build" else "ci"
    cmd = [uv_path, "sync", "--frozen", "--group", group]
    env = _noninteractive_env()
    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=_SYNC_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        _fail(f"uv sync failed (OS error): {exc}")
    except subprocess.TimeoutExpired:
        _fail(f"uv sync --group {group} timed out")

    if completed.returncode != 0:
        _fail(
            f"uv sync --frozen --group {group} failed (exit {completed.returncode}); "
            "fix network/lockfile and retry. No secondary fallback.",
            code=completed.returncode or 1,
        )


def bootstrap(mode: Mode) -> str:
    """Fail-fast, ensure uv, sync the mode's dependency group. Returns uv path."""
    fail_fast(mode=mode)
    uv_path = ensure_uv()
    uv_dir = os.path.dirname(os.path.abspath(uv_path))
    current_path = os.environ.get("PATH", "")
    if uv_dir not in current_path.split(os.pathsep):
        os.environ["PATH"] = uv_dir + os.pathsep + current_path
    ensure_deps(mode, uv_path=uv_path)
    return uv_path
