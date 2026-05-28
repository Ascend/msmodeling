#!/usr/bin/env python3
"""Nightly: two-phase UT, refresh test_map, nightly + benchmark, report.

CLI entry point for run_nightly.sh. Phase 1: smoke/regression ``not npu and
not nightly`` with coverage → test_map on success. Phase 2: smoke/regression
``not npu and nightly`` + benchmark (remaining full suite) → report.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Final

from scripts.helpers._config import Config, ConfigError
from scripts.helpers.common.build_test_map import build_test_map, detect_redundant_cases
from scripts.helpers.common.coverage_config import cov_pytest_args
from scripts.helpers.common.coverage_gate import GateConfig, check_thresholds, load_totals
from scripts.helpers.common.test_map_config import resolve_test_map_path
from scripts.helpers.nightly.feishu_notifier import build_feishu_payload, push_feishu
from scripts.helpers.nightly.pytest_parser import parse_pytest_output
from scripts.helpers.nightly.report_builder import (
    build_report,
    compute_weak_coverage_symbols,
    fetch_env_info,
    load_test_map_summary,
)
from scripts.helpers.nightly.report_models import CoverageSummary

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

_NIGHTLY_MARKER = "not npu and nightly"
_PROCESS_TERMINATE_TIMEOUT_SECONDS: Final[float] = 5.0


# ---------------------------------------------------------------------------
# Pytest command builders
# ---------------------------------------------------------------------------


def _build_test_map_pytest_cmd(python_exe: str, cfg: Config) -> list[str]:
    """Phase 1: incremental-scope smoke/regression for coverage + test_map."""
    return [
        python_exe,
        "-m",
        "pytest",
        "tests/smoke/",
        "tests/regression/",
        "-m",
        cfg.test_map_marker,
        "-n0",
        *cov_pytest_args(cov_context=True),
        "-q",
        "--no-header",
        "-vv",
        "--tb=short",
        "--durations=20",
    ]


def _build_nightly_pytest_cmd(python_exe: str) -> list[str]:
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
        "-vv",
        "--tb=short",
        "--durations=20",
    ]


def _build_benchmark_pytest_cmd(python_exe: str, cfg: Config) -> list[str]:
    cmd = [
        python_exe,
        "-m",
        "pytest",
        "tests/benchmark/",
        "-m",
        "not npu",
        "-q",
        "--no-header",
        "-vv",
        "--tb=short",
        "--durations=20",
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


def _stream_pytest(cmd: list[str], cwd: Path) -> tuple[int, str]:
    """Run pytest, stream stdout line by line, return (exit_code, full_output)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)

    chunks: list[str] = []
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
                chunks.append(line)
        except KeyboardInterrupt:
            _terminate_process_tree(proc)
            raise
        return proc.wait(), "".join(chunks)
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
    pytest_output: str,
    pytest_exit_code: int,
    *,
    coverage: CoverageSummary | None,
    test_map_written: bool,
    test_map_path: Path | None,
    webhook_url: str | None,
    weak_coverage_symbols: tuple[str, ...] = (),
    redundancy_warnings: tuple[dict[str, object], ...] = (),
) -> None:
    """Parse pytest output, build report JSON, optionally push to Feishu."""
    import json

    stats = parse_pytest_output(pytest_output)
    env = fetch_env_info()
    test_map = load_test_map_summary(test_map_path if test_map_written else None)
    report = build_report(
        stats,
        env,
        test_map,
        pytest_exit_code,
        coverage=coverage,
        test_map_written=test_map_written,
        test_map_path=test_map_path,
        weak_coverage_symbols=weak_coverage_symbols,
        redundancy_warnings=redundancy_warnings,
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))

    if not webhook_url:
        print("FEISHU_WEBHOOK_URL not set — skipping Feishu push", file=sys.stderr)
        return

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
    )
    push_feishu(webhook_url, payload)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    cfg = Config.from_env()

    try:
        test_map_path = resolve_test_map_path(cfg, must_exist=False)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        map_cmd = _build_test_map_pytest_cmd(sys.executable, cfg)
        map_exit, map_output = _stream_pytest(map_cmd, cwd=REPO_ROOT)

        coverage = _coverage_summary(cfg)
        if coverage:
            print(coverage.message)
            if not coverage.gate_passed:
                print(
                    "Nightly coverage below threshold (non-blocking): "
                    f"line={coverage.line_percent:.1f}% branch={coverage.branch_percent:.1f}%",
                    file=sys.stderr,
                )

        map_written = False
        weak_symbols: tuple[str, ...] = ()
        redundancy: tuple[dict[str, object], ...] = ()
        if map_exit == 0:
            build_test_map(test_map_path, marker_expr=cfg.test_map_marker)
            map_written = True

            from scripts.helpers.common.test_map_loader import load_test_map as _load_tm

            baseline_map = _load_tm(cfg)
            redundancy = tuple(detect_redundant_cases(baseline_map))
            weak_symbols = compute_weak_coverage_symbols(test_map_path, REPO_ROOT / ".coverage")
        else:
            print(
                "Skipping test_map write: smoke/regression (not npu and not nightly) failed",
                file=sys.stderr,
            )

        nightly_cmd = _build_nightly_pytest_cmd(sys.executable)
        nightly_exit, nightly_output = _stream_pytest(nightly_cmd, cwd=REPO_ROOT)

        bench_cmd = _build_benchmark_pytest_cmd(sys.executable, cfg)
        bench_exit, bench_output = _stream_pytest(bench_cmd, cwd=REPO_ROOT)

        combined_output = map_output + "\n" + nightly_output + "\n" + bench_output
        overall_exit = map_exit or nightly_exit or bench_exit

        emit_report(
            combined_output,
            overall_exit,
            coverage=coverage,
            test_map_written=map_written,
            test_map_path=test_map_path if map_written else None,
            webhook_url=cfg.feishu_webhook_url or None,
            weak_coverage_symbols=weak_symbols,
            redundancy_warnings=redundancy,
        )
        return overall_exit
    except KeyboardInterrupt:
        print("\nInterrupted — stopping nightly run", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
