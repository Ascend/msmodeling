"""Root build.py entry — build wheel or run CI gate."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.build.argv import BuildOptions, parse_argv
from scripts.helpers.common._logging import setup_logger
from scripts.helpers.common.subprocess_stream import run_merged_output

if TYPE_CHECKING:
    from collections.abc import Sequence

_BUILD_SCRIPT: Final = REPO_ROOT / "scripts" / "build.sh"
_CI_GATE_SCRIPT: Final = REPO_ROOT / "scripts" / "run_ci_gate.sh"
_ARTIFACTS_DIR: Final = REPO_ROOT / "artifacts"
_WHEELS_DIR: Final = _ARTIFACTS_DIR / "wheels"
_TEST_REPORTS_DIR: Final = _ARTIFACTS_DIR / "test-reports"
_SHELL_TIMEOUT_SECONDS: Final = 36000
_VERSION_CMD_TIMEOUT_SECONDS: Final = 120

logger: Final = setup_logger("build")


def _uv_executable() -> str:
    uv = shutil.which("uv")
    if uv is None:
        msg = "uv not found in PATH"
        raise RuntimeError(msg)
    return uv


def _require_uv() -> bool:
    if shutil.which("uv") is not None:
        return True
    logger.error("uv not found in PATH")
    return False


def _read_pyproject_version() -> str:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    version = data.get("project", {}).get("version")
    if not isinstance(version, str):
        msg = "pyproject.toml project.version must be a string"
        raise TypeError(msg)
    return version


def _clear_wheel_output_dir(wheel_dir: Path) -> None:
    wheel_dir.mkdir(parents=True, exist_ok=True)
    for wheel_path in wheel_dir.glob("msmodeling-*.whl"):
        wheel_path.unlink()


def _newest_wheel(wheel_dir: Path) -> Path | None:
    wheels = list(wheel_dir.glob("msmodeling-*.whl"))
    if not wheels:
        return None
    return max(wheels, key=lambda path: path.stat().st_mtime)


def _set_pyproject_version(version: str) -> None:
    """Write ``project.version`` via uv without re-locking the lockfile."""
    logger.info("setting project version to %s", version)
    subprocess.run(
        [_uv_executable(), "version", version, "--frozen"],
        cwd=REPO_ROOT,
        check=True,
        timeout=_VERSION_CMD_TIMEOUT_SECONDS,
    )


def _run_build_script(env: dict[str, str]) -> int:
    try:
        _run_shell(
            ["bash", str(_BUILD_SCRIPT)],
            cwd=REPO_ROOT,
            env=env,
            timeout=_SHELL_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("command failed (exit %d): %s", exc.returncode, exc.cmd)
        return exc.returncode
    except Exception:
        logger.exception("unexpected error running build")
        return 1
    return 0


def _run_with_version_staging(
    *,
    pyproject_version: str,
    target_version: str,
    should_stage: bool,
    env: dict[str, str],
) -> tuple[int, bool]:
    exit_code = 0
    restore_failed = False
    staging_applied = False
    try:
        if should_stage:
            try:
                _set_pyproject_version(target_version)
                staging_applied = True
            except subprocess.CalledProcessError as exc:
                logger.error(
                    "failed to set project version to %s (exit %d)",
                    target_version,
                    exc.returncode,
                )
                return exc.returncode, False
        exit_code = _run_build_script(env)
    finally:
        if staging_applied:
            try:
                logger.info("restoring project version to %s", pyproject_version)
                _set_pyproject_version(pyproject_version)
            except subprocess.CalledProcessError:
                logger.exception(
                    "build finished but failed to restore pyproject version to %s",
                    pyproject_version,
                )
                restore_failed = True
    return exit_code, restore_failed


def _run_shell(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    tee_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a shell command.

    Without *tee_path*, stdout/stderr inherit the parent terminal.
    With *tee_path*, merged output is streamed to the log file and mirrored live.
    """
    logger.info("running: %s", " ".join(cmd))
    if tee_path is None:
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            timeout=timeout,
            check=True,
            text=True,
        )
    exit_code = run_merged_output(
        cmd,
        cwd=cwd,
        env=env,
        timeout=timeout,
        tee_path=tee_path,
        mirror_stdout=True,
        start_new_session=True,
    )
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, cmd)
    return subprocess.CompletedProcess(cmd, exit_code, stdout="", stderr=None)


def run_build(options: BuildOptions) -> int:
    """Build msmodeling wheel via scripts/build.sh."""
    if not _BUILD_SCRIPT.is_file():
        logger.error("missing script: %s", _BUILD_SCRIPT)
        return 1
    if not _require_uv():
        return 1

    try:
        pyproject_version = _read_pyproject_version()
    except (OSError, TypeError):
        logger.exception("failed to read version from pyproject.toml")
        return 1

    if options.version_explicit:
        if options.version is None:
            logger.error("internal error: version_explicit without version value")
            return 1
        target_version = options.version
    else:
        target_version = pyproject_version

    # ``local`` is parsed for department spec compatibility; no behavioral branch.
    should_stage_version = options.version_explicit and target_version != pyproject_version
    _clear_wheel_output_dir(_WHEELS_DIR)
    env = os.environ.copy()
    env["MSMODELING_WHEEL_OUTPUT_DIR"] = str(_WHEELS_DIR)

    exit_code, restore_failed = _run_with_version_staging(
        pyproject_version=pyproject_version,
        target_version=target_version,
        should_stage=should_stage_version,
        env=env,
    )

    if restore_failed:
        return 1
    if exit_code != 0:
        return exit_code

    wheel_path = _newest_wheel(_WHEELS_DIR)
    manifest = {
        "wheel_path": str(wheel_path) if wheel_path is not None else "",
        "version": target_version,
        "pyproject_version": pyproject_version,
        "version_explicit": options.version_explicit,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (_ARTIFACTS_DIR / "build-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def run_test(options: BuildOptions) -> int:
    """Run CI gate via scripts/run_ci_gate.sh."""
    if not _CI_GATE_SCRIPT.is_file():
        logger.error("missing script: %s", _CI_GATE_SCRIPT)
        return 1
    if not _require_uv():
        return 1

    test_map_path = options.extras.get("test_map_path") or os.environ.get("MSMODELING_TEST_MAP_PATH")
    if not test_map_path:
        logger.error(
            "test requires test_map_path via --extra test_map_path=... or MSMODELING_TEST_MAP_PATH",
        )
        return 1
    if not Path(test_map_path).is_file():
        logger.error("test_map_path is not a file: %s", test_map_path)
        return 1

    env = os.environ.copy()
    env["MSMODELING_TEST_MAP_PATH"] = test_map_path
    env["MSMODELING_TEST_BASE_BRANCH"] = options.extras.get(
        "base_branch",
        os.environ.get("MSMODELING_TEST_BASE_BRANCH", "master"),
    )
    if "offline" in options.extras:
        env["MSMODELING_OFFLINE"] = options.extras["offline"]
    if "weights_prune" in options.extras:
        env["MSMODELING_TEST_WEIGHTS_PRUNE"] = options.extras["weights_prune"]

    _TEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _TEST_REPORTS_DIR / "ci_gate.log"
    started = time.monotonic()

    try:
        _run_shell(
            ["bash", str(_CI_GATE_SCRIPT)],
            cwd=REPO_ROOT,
            env=env,
            timeout=_SHELL_TIMEOUT_SECONDS,
            tee_path=log_path,
        )
        exit_code = 0
    except subprocess.CalledProcessError as exc:
        logger.error("command failed (exit %d): %s", exc.returncode, exc.cmd)
        exit_code = exc.returncode
    except Exception:
        logger.exception("unexpected error running ci gate")
        exit_code = 1

    summary = {
        "exit_code": exit_code,
        "test_map_path": test_map_path,
        "duration_seconds": time.monotonic() - started,
    }
    (_TEST_REPORTS_DIR / "gate-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """Parse argv and run build or test."""
    options = parse_argv(argv)
    if options.is_test:
        return run_test(options)
    return run_build(options)


if __name__ == "__main__":
    raise SystemExit(main())
