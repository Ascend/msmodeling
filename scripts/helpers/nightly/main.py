#!/usr/bin/env python3
"""Nightly: two-phase UT, refresh test_map, nightly + benchmark, report.

CLI entry point for run_nightly.sh. Phase 1: smoke/regression ``not npu and
not nightly`` with coverage → test_map on success. Phase 2: smoke/regression
``not npu and nightly`` + benchmark (remaining full suite) → report.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

from scripts.helpers._config import Config, ConfigError
from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.ci_gate.gate_policy import (
    find_expired_unmapped,
    format_expired_exemptions_section,
    load_gate_policy,
)
from scripts.helpers.common._logging import log_env_audit, setup_logger
from scripts.helpers.common.build_test_map import build_test_map, detect_redundant_cases
from scripts.helpers.common.coverage_config import cov_pytest_args, pytest_xdist_args
from scripts.helpers.common.coverage_gate import GateConfig, check_thresholds, load_totals
from scripts.helpers.common.test_map_config import resolve_test_map_path
from scripts.helpers.nightly.feishu_notifier import build_feishu_payload, push_feishu
from scripts.helpers.nightly.pytest_parser import NightlyRunStats, parse_junit_xml
from scripts.helpers.nightly.report_builder import compute_weak_coverage_symbols, fetch_env_info, load_test_map_summary
from scripts.helpers.nightly.report_models import CoverageSummary

_TEST_MAP_MARKER = "not npu and not nightly"
_NIGHTLY_MARKER = "not npu and nightly"
_PROCESS_TERMINATE_TIMEOUT_SECONDS: Final[float] = 5.0


# ---------------------------------------------------------------------------
# Pytest command builders
# ---------------------------------------------------------------------------


def _build_test_map_pytest_cmd(python_exe: str, cfg: Config, *, junit_xml: Path) -> list[str]:
    """Phase 1: incremental-scope smoke/regression for coverage + test_map."""
    return [
        python_exe,
        "-m",
        "pytest",
        "tests/smoke/",
        "tests/regression/",
        "-m",
        _TEST_MAP_MARKER,
        *pytest_xdist_args(),
        *cov_pytest_args(cov_context=True),
        "-q",
        "--no-header",
        "--tb=short",
        "--disable-warnings",
        "--durations=20",
        f"--junit-xml={junit_xml}",
    ]


def _build_nightly_pytest_cmd(python_exe: str, *, junit_xml: Path) -> list[str]:
    """Phase 2a: nightly-marked smoke/regression (excludes npu)."""
    return [
        python_exe,
        "-m",
        "pytest",
        "tests/smoke/",
        "tests/regression/",
        "-m",
        _NIGHTLY_MARKER,
        "-n",
        "auto",
        "-q",
        "--no-header",
        "--tb=short",
        "--disable-warnings",
        "--durations=20",
        f"--junit-xml={junit_xml}",
    ]


def _benchmark_models_enabled() -> bool:
    return os.environ.get("MSMODELING_BENCHMARK_MODELS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _build_benchmark_pytest_cmd(python_exe: str, cfg: Config, *, junit_xml: Path) -> list[str]:
    benchmark_target = (
        str(REPO_ROOT / "tests" / "benchmark")
        if _benchmark_models_enabled()
        else str(REPO_ROOT / "tests" / "benchmark" / "ops")
    )
    cmd = [
        python_exe,
        "-m",
        "pytest",
        benchmark_target,
        "-m",
        "not npu",
        "-q",
        "--no-header",
        "--tb=short",
        "--disable-warnings",
        "--durations=20",
        f"--junit-xml={junit_xml}",
    ]
    if cfg.benchmark_parallel:
        cmd.extend(["-n", "auto"])
    return cmd


# ---------------------------------------------------------------------------
# Pytest runner (streaming)
# ---------------------------------------------------------------------------


def _terminate_process_tree(
    proc: subprocess.Popen[str],
    *,
    sigterm_timeout_seconds: float = _PROCESS_TERMINATE_TIMEOUT_SECONDS,
) -> None:
    """SIGTERM the pytest process group, escalate to SIGKILL on timeout."""
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
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        proc.wait(timeout=sigterm_timeout_seconds)


def _stream_pytest(cmd: list[str], cwd: Path) -> int:
    """Run pytest, stream stdout line by line, return exit code."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    try:
        if proc.stdout is None:
            raise RuntimeError("Failed to capture pytest stdout")
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
        except KeyboardInterrupt:
            _terminate_process_tree(proc)
            raise
        return proc.wait()
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.poll() is None:
            _terminate_process_tree(proc)
            proc.wait()


# ---------------------------------------------------------------------------
# Coverage summary
# ---------------------------------------------------------------------------


def _coverage_summary(cfg: Config) -> CoverageSummary | None:
    """Build CoverageSummary from .coverage data.

    Loads totals once, checks thresholds directly — no duplicate subprocess.
    """
    gate_cfg = GateConfig.from_config(cfg)
    try:
        totals = load_totals(REPO_ROOT / ".coverage")
    except (FileNotFoundError, RuntimeError):
        return None

    failures = check_thresholds(totals.line_percent, totals.branch_percent, gate_cfg)
    passed = len(failures) == 0
    message = (
        f"Coverage gate passed: line={totals.line_percent:.1f}% branch={totals.branch_percent:.1f}%"
        if passed
        else "Coverage gate failed: " + "; ".join(failures)
    )
    return CoverageSummary(
        line_percent=totals.line_percent,
        branch_percent=totals.branch_percent,
        line_threshold=gate_cfg.line_threshold,
        branch_threshold=gate_cfg.branch_threshold,
        gate_passed=passed,
        message=message,
    )


# ---------------------------------------------------------------------------
# Report emission
# ---------------------------------------------------------------------------


def emit_report(
    junit_xml_paths: tuple[Path, ...],
    *,
    coverage: CoverageSummary | None,
    test_map_written: bool,
    test_map_path: Path | None,
    webhook_url: str | None,
    weak_coverage_symbols: tuple[str, ...] = (),
    redundancy_warnings: tuple[dict[str, object], ...] = (),
    expired_exemption_section: str = "",
) -> NightlyRunStats:
    """Parse pytest JUnit XML, push report to Feishu when webhook is set.

    Returns the parsed run stats so the caller can build a final summary
    without re-parsing the XML.
    """
    logger = logging.getLogger("nightly")
    stats = parse_junit_xml(junit_xml_paths)
    env = fetch_env_info()
    test_map = load_test_map_summary(test_map_path if test_map_written else None)

    if not webhook_url:
        logger.warning("FEISHU_WEBHOOK_URL not set — skipping Feishu push")
        return stats

    payload = build_feishu_payload(
        timestamp=env.timestamp,
        branch=env.branch,
        commit=env.commit,
        passed=stats.passed,
        failed=stats.failed,
        duration_sec=stats.duration_sec,
        coverage_line_percent=coverage.line_percent if coverage else None,
        coverage_branch_percent=coverage.branch_percent if coverage else None,
        coverage_line_threshold=coverage.line_threshold if coverage else None,
        coverage_branch_threshold=coverage.branch_threshold if coverage else None,
        coverage_gate_passed=coverage.gate_passed if coverage else None,
        test_map_source_files=test_map.source_files,
        test_map_symbols=test_map.symbols,
        test_map_written=test_map_written,
        failed_cases=stats.failed_cases,
        first_error=stats.first_error,
        weak_coverage_symbols=weak_coverage_symbols,
        redundancy_warnings=redundancy_warnings,
        expired_exemption_section=expired_exemption_section,
    )
    push_feishu(webhook_url, payload)
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    logger = setup_logger("nightly")
    cfg = Config.from_env()
    log_env_audit(cfg, logger)

    try:
        test_map_path = resolve_test_map_path(cfg, must_exist=False)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1

    with tempfile.TemporaryDirectory(prefix="nightly-junit-") as junit_dir:
        junit_root = Path(junit_dir)
        map_junit = junit_root / "phase1_test_map.xml"
        nightly_junit = junit_root / "phase2a_nightly.xml"
        bench_junit = junit_root / "phase2b_benchmark.xml"

        try:
            logger.info("Phase 1: test_map — smoke/regression (not npu and not nightly)")
            map_cmd = _build_test_map_pytest_cmd(sys.executable, cfg, junit_xml=map_junit)
            logger.info("Running pytest: %s", shlex.join(map_cmd))
            map_exit = _stream_pytest(map_cmd, cwd=REPO_ROOT)
            logger.info("Phase 1: pytest exit=%d", map_exit)

            coverage = _coverage_summary(cfg)
            if coverage:
                logger.info("%s", coverage.message)
                if not coverage.gate_passed:
                    logger.warning(
                        "Nightly coverage below threshold (non-blocking): line=%.1f%% branch=%.1f%%",
                        coverage.line_percent,
                        coverage.branch_percent,
                    )

            map_written = False
            weak_symbols: tuple[str, ...] = ()
            redundancy: tuple[dict[str, object], ...] = ()
            expired_section = ""
            if map_exit == 0:
                logger.info("Building test_map ...")
                build_test_map(test_map_path, marker_expr=_TEST_MAP_MARKER)
                map_written = True

                from scripts.helpers.common.test_map_loader import (
                    load_test_map as _load_tm,
                )

                baseline_map = _load_tm(cfg)
                redundancy = tuple(detect_redundant_cases(baseline_map))
                weak_symbols = compute_weak_coverage_symbols(test_map_path, REPO_ROOT / ".coverage")
                try:
                    gate_policy = load_gate_policy(REPO_ROOT)
                    expired = find_expired_unmapped(gate_policy, baseline_map)
                    expired_section = format_expired_exemptions_section(expired)
                    if expired:
                        logger.warning(
                            "Found %d expired exemption(s) still unmapped in test_map",
                            len(expired),
                        )
                except ConfigError as exc:
                    logger.warning("Skipping expired exemption audit: %s", exc)
                logger.info("test_map written: %s", test_map_path)
            else:
                logger.warning("Skipping test_map write: smoke/regression (not npu and not nightly) failed")

            logger.info("Phase 2a: nightly — smoke/regression (not npu and nightly)")
            nightly_cmd = _build_nightly_pytest_cmd(sys.executable, junit_xml=nightly_junit)
            logger.info("Running pytest: %s", shlex.join(nightly_cmd))
            nightly_exit = _stream_pytest(nightly_cmd, cwd=REPO_ROOT)
            logger.info("Phase 2a: pytest exit=%d", nightly_exit)

            logger.info(
                "Phase 2b: benchmark (%s)",
                "models+ops" if _benchmark_models_enabled() else "ops only",
            )
            bench_cmd = _build_benchmark_pytest_cmd(sys.executable, cfg, junit_xml=bench_junit)
            logger.info("Running pytest: %s", shlex.join(bench_cmd))
            bench_exit = _stream_pytest(bench_cmd, cwd=REPO_ROOT)
            logger.info("Phase 2b: pytest exit=%d", bench_exit)

            overall_exit = map_exit or nightly_exit or bench_exit

            logger.info("Building report ...")
            stats = emit_report(
                (map_junit, nightly_junit, bench_junit),
                coverage=coverage,
                test_map_written=map_written,
                test_map_path=test_map_path if map_written else None,
                webhook_url=cfg.feishu_webhook_url or None,
                weak_coverage_symbols=weak_symbols,
                redundancy_warnings=redundancy,
                expired_exemption_section=expired_section,
            )

            summary_lines = [
                (
                    f"Nightly exit={overall_exit}: passed={stats.passed} "
                    f"failed={stats.failed} errors={stats.errors} "
                    f"duration={stats.duration_sec:.0f}s"
                ),
            ]
            if coverage:
                summary_lines.append(coverage.message)
            print("\n".join(summary_lines))
            return overall_exit
        except KeyboardInterrupt:
            logger.warning("Interrupted — stopping nightly run")
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
