"""Run tests for ``python build.py test`` — CI gate, full suite, or named suites."""

from __future__ import annotations

import json
import subprocess
import time
from typing import TYPE_CHECKING, Final

from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.build.argv import BuildSuite
from scripts.helpers.build.bootstrap import bootstrap, fail_fast
from scripts.helpers.build.runtime_env import apply_test_defaults
from scripts.helpers.build.test_map_fetch import MapFetchError, resolve_test_map_path
from scripts.helpers.common._logging import setup_logger
from scripts.helpers.common.subprocess_stream import run_merged_output
from scripts.helpers.defaults import DEFAULT_BASE_BRANCH, PYTEST_XDIST_ARGS

if TYPE_CHECKING:
    from pathlib import Path

    from scripts.helpers.build.argv import BuildOptions

_SUITE_SCRIPTS: Final = {
    BuildSuite.CI_GATE: REPO_ROOT / "scripts" / "run_ci_gate.sh",
    BuildSuite.SMOKE: REPO_ROOT / "scripts" / "run_smoke.sh",
    BuildSuite.REGRESSION: REPO_ROOT / "scripts" / "run_regression.sh",
    BuildSuite.BENCHMARK: REPO_ROOT / "scripts" / "run_benchmark.sh",
}
_TEST_REPORTS_DIR: Final = REPO_ROOT / "artifacts" / "test-reports"
_SHELL_TIMEOUT_SECONDS: Final = 36000

logger: Final = setup_logger("build")


def run_test(options: BuildOptions) -> int:
    """Dispatch ``python build.py test --suite ...``."""
    if options.suite == BuildSuite.CI_GATE:
        return _run_ci_gate(options)
    if options.suite == BuildSuite.FULL:
        return _run_full_suite(options)
    return _run_named_suite(options)


def _run_teed(cmd: list[str], *, env: dict[str, str], log_path: Path) -> int:
    _TEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("running: %s", " ".join(cmd))
    try:
        exit_code = run_merged_output(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            timeout=_SHELL_TIMEOUT_SECONDS,
            tee_path=log_path,
            mirror_stdout=True,
            start_new_session=True,
        )
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, cmd)
        return 0
    except subprocess.CalledProcessError as exc:
        logger.error("command failed (exit %d): %s", exc.returncode, exc.cmd)
        return exc.returncode
    except Exception:
        logger.exception("unexpected error running tests")
        return 1


def _apply_extras(env: dict[str, str], options: BuildOptions) -> dict[str, str]:
    if "offline" in options.extras:
        env["MSMODELING_OFFLINE"] = options.extras["offline"]
    if "weights_prune" in options.extras:
        env["MSMODELING_TEST_WEIGHTS_PRUNE"] = options.extras["weights_prune"]
    if "base_branch" in options.extras:
        env["MSMODELING_TEST_BASE_BRANCH"] = options.extras["base_branch"]
    return env


def _write_summary(*, exit_code: int, mode: str, test_map_path: str | None, started: float) -> None:
    _TEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (_TEST_REPORTS_DIR / "gate-summary.json").write_text(
        json.dumps(
            {
                "exit_code": exit_code,
                "mode": mode,
                "test_map_path": test_map_path,
                "duration_seconds": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_full_suite(options: BuildOptions) -> int:
    """Run ``pytest tests`` with pyproject markers plus xdist worksteal."""
    uv_path = bootstrap("test")
    env = _apply_extras(apply_test_defaults(), options)
    log_path = _TEST_REPORTS_DIR / "full_suite.log"
    started = time.monotonic()
    cmd = [uv_path, "run", "pytest", "tests", *PYTEST_XDIST_ARGS]
    exit_code = _run_teed(cmd, env=env, log_path=log_path)
    _write_summary(exit_code=exit_code, mode="full", test_map_path=None, started=started)
    return exit_code


def _run_named_suite(options: BuildOptions) -> int:
    # Before bootstrap/sync: suite script is not checked in fail_fast (suite-dependent).
    script = _SUITE_SCRIPTS[options.suite]
    if not script.is_file():
        logger.error(
            "Missing suite script for --suite %s: %s. "
            "Re-clone the repository or restore scripts/ from the default branch.",
            options.suite.value,
            script,
        )
        return 1

    bootstrap("test")
    env = _apply_extras(apply_test_defaults(), options)
    log_path = _TEST_REPORTS_DIR / f"{options.suite.value}.log"
    started = time.monotonic()
    exit_code = _run_teed(["bash", str(script)], env=env, log_path=log_path)
    _write_summary(
        exit_code=exit_code,
        mode=options.suite.value,
        test_map_path=None,
        started=started,
    )
    return exit_code


def _run_ci_gate(options: BuildOptions) -> int:
    """Resolve test_map (download if needed) then run scripts/run_ci_gate.sh."""
    # Before bootstrap/sync: ci_gate script is not checked in fail_fast (suite-dependent).
    script = _SUITE_SCRIPTS[BuildSuite.CI_GATE]
    if not script.is_file():
        logger.error(
            "Missing CI gate script: %s. Re-clone the repository or restore scripts/run_ci_gate.sh.",
            script,
        )
        return 1

    # Cheap prerequisites before network I/O (test_map download) or uv sync.
    fail_fast(mode="test")

    env = _apply_extras(apply_test_defaults(), options)
    configured = options.extras.get("test_map_path") or env.get("MSMODELING_TEST_MAP_PATH")
    base_branch = env.get("MSMODELING_TEST_BASE_BRANCH", DEFAULT_BASE_BRANCH)
    cache_dir = env.get("MSMODELING_CACHE")
    logger.info(
        "suite=ci_gate base_branch=%s; resolving test_map (download only if needed)",
        base_branch,
    )
    try:
        test_map_path = resolve_test_map_path(
            configured=configured,
            base_branch=base_branch,
            cache_dir=cache_dir,
        )
    except MapFetchError as exc:
        logger.error("%s", exc)
        logger.error(
            "Cannot start CI gate without a test_map. "
            "Fix network/OBS access, set MSMODELING_TEST_MAP_PATH to an existing file, "
            "or pass -e test_map_path=/path/to/test_map.json. "
            "Wrong MSMODELING_TEST_BASE_BRANCH also yields 404 (URL uses that branch name). "
            "For a full local run without test_map: python build.py test --suite full"
        )
        return 1

    bootstrap("test")
    env["MSMODELING_TEST_MAP_PATH"] = str(test_map_path)
    env["MSMODELING_TEST_BASE_BRANCH"] = base_branch

    log_path = _TEST_REPORTS_DIR / "ci_gate.log"
    started = time.monotonic()
    exit_code = _run_teed(["bash", str(script)], env=env, log_path=log_path)
    _write_summary(
        exit_code=exit_code,
        mode="ci_gate",
        test_map_path=str(test_map_path),
        started=started,
    )
    return exit_code
