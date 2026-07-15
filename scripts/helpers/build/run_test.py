"""Run tests for ``python build.py test`` — full suite or CI gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.build.bootstrap import bootstrap
from scripts.helpers.common._logging import setup_logger
from scripts.helpers.common.subprocess_stream import run_merged_output

if TYPE_CHECKING:
    from scripts.helpers.build.argv import BuildOptions

_CI_GATE_SCRIPT: Final = REPO_ROOT / "scripts" / "run_ci_gate.sh"
_TEST_REPORTS_DIR: Final = REPO_ROOT / "artifacts" / "test-reports"
_SHELL_TIMEOUT_SECONDS: Final = 36000

logger: Final = setup_logger("build")


def run_test(options: BuildOptions) -> int:
    """Run full ``pytest tests``, or CI gate when test_map is provided."""
    raw = options.extras.get("test_map_path") or os.environ.get("MSMODELING_TEST_MAP_PATH")
    if not raw or not raw.strip():
        logger.warning(
            "MSMODELING_TEST_MAP_PATH not set; falling back to full pytest suite. Set it explicitly to use CI gate.",
        )
        return _run_full_suite(options)
    return _run_ci_gate(options, raw.strip())


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


def _run_full_suite(options: BuildOptions) -> int:
    """Run ``pytest tests`` using markers from ``pyproject.toml`` addopts."""
    bootstrap("test")

    env = os.environ.copy()
    if "offline" in options.extras:
        env["MSMODELING_OFFLINE"] = options.extras["offline"]
    if "weights_prune" in options.extras:
        env["MSMODELING_TEST_WEIGHTS_PRUNE"] = options.extras["weights_prune"]

    log_path = _TEST_REPORTS_DIR / "full_suite.log"
    started = time.monotonic()
    exit_code = _run_teed(
        [sys.executable, "-m", "pytest", "tests"],
        env=env,
        log_path=log_path,
    )
    (_TEST_REPORTS_DIR / "gate-summary.json").write_text(
        json.dumps(
            {
                "exit_code": exit_code,
                "mode": "full_suite",
                "test_map_path": None,
                "duration_seconds": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return exit_code


def _run_ci_gate(options: BuildOptions, test_map_path: str) -> int:
    """Run CI gate via scripts/run_ci_gate.sh."""
    if not _CI_GATE_SCRIPT.is_file():
        logger.error("missing script: %s", _CI_GATE_SCRIPT)
        return 1
    if not Path(test_map_path).is_file():
        logger.error("test_map_path is not a file: %s", test_map_path)
        return 1

    bootstrap("test")

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

    log_path = _TEST_REPORTS_DIR / "ci_gate.log"
    started = time.monotonic()
    exit_code = _run_teed(
        ["bash", str(_CI_GATE_SCRIPT)],
        env=env,
        log_path=log_path,
    )
    (_TEST_REPORTS_DIR / "gate-summary.json").write_text(
        json.dumps(
            {
                "exit_code": exit_code,
                "mode": "ci_gate",
                "test_map_path": test_map_path,
                "duration_seconds": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return exit_code
