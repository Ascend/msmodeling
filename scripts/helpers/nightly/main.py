#!/usr/bin/env python3
"""Nightly: full UT with coverage and report.

CLI entry for run_nightly.sh. Intended for CI only — local runs are discouraged.
Two pytest waves over ``tests/`` run in parallel: non-benchmark/non-network (xdist + coverage) and
benchmark-or-network (serial). A non-blocking config drift check compares vendored remote configs
against the live Hub → report.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from pathlib import Path

from scripts.helpers._config import Config
from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.common._logging import log_env_audit, setup_logger
from scripts.helpers.common.coverage_config import cov_pytest_args, pytest_xdist_args
from scripts.helpers.common.coverage_gate import (
    GateConfig,
    check_thresholds,
    load_totals,
)
from scripts.helpers.nightly.failure_attribution import (
    ATTRIBUTION_BUDGET_SKIP_LABEL,
    attribute_failures,
    default_max_workers,
    results_to_failure_blames,
)
from scripts.helpers.nightly.feishu_notifier import (
    FEISHU_DRIFT_LIMIT,
    display_status,
    push_feishu_report,
)
from scripts.helpers.nightly.pytest_parser import (
    NightlyRunStats,
    merge_nightly_run_stats,
    parse_pytest_stdout,
)
from scripts.helpers.nightly.report_builder import fetch_env_info
from scripts.helpers.nightly.report_models import (
    AttributionConclusion,
    CoverageSummary,
    FailureBlame,
    FeishuReportInput,
)
from tensor_cast.core.model_source_security import warn_remote_code_risk

_PYTEST_MARKER_WAVE_A = "not npu and not benchmark and not network"
_PYTEST_MARKER_WAVE_B = "not npu and (benchmark or network)"
_PROCESS_TERMINATE_TIMEOUT_SECONDS: Final[float] = 5.0
_PYTEST_CAPTURE_MAX_CHARS: Final[int] = 10 * 1024 * 1024
_HAS_PROCESS_GROUPS: Final[bool] = sys.platform != "win32"
_DEFAULT_NIGHTLY_TIMEOUT_SECONDS: Final[float] = 50 * 60
PYTEST_TIMEOUT_EXIT_CODE: Final[int] = 124
ATTRIBUTION_HARD_FAIL_EXIT_CODE: Final[int] = 3
_TIMEOUT_ENV: Final = "MSMODELING_NIGHTLY_TIMEOUT_SECONDS"
_COVERAGE_MERGED_FILE: Final = ".coverage"
_COVERAGE_NON_BENCHMARK_FILE: Final = ".coverage.non_benchmark"
_COVERAGE_BENCHMARK_FILE: Final = ".coverage.benchmark"

# Vendored remote configs whose live Hub counterpart we watch for drift.
_DRIFT_FIXTURE_MAP: Final[dict[str, str]] = {
    "deepseek-ai/DeepSeek-V3.1": "deepseekv3.1_remote",
    "MiniMaxAI/MiniMax-M2": "minimax_m2",
}
_DRIFT_COMPARE_KEYS: Final[tuple[str, ...]] = (
    "model_type",
    "architectures",
    "num_hidden_layers",
    "hidden_size",
    "vocab_size",
)


def resolve_nightly_timeout_seconds() -> float:
    """Self-timeout for the nightly process; pipeline may override via env."""
    raw = (os.environ.get(_TIMEOUT_ENV) or "").strip()
    if not raw:
        return _DEFAULT_NIGHTLY_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logging.getLogger("nightly").warning(
            "Invalid %s=%r; using default %.0fs",
            _TIMEOUT_ENV,
            raw,
            _DEFAULT_NIGHTLY_TIMEOUT_SECONDS,
        )
        return _DEFAULT_NIGHTLY_TIMEOUT_SECONDS
    if value <= 0:
        return _DEFAULT_NIGHTLY_TIMEOUT_SECONDS
    return value


def _build_pytest_cmd_wave_a(python_exe: str) -> list[str]:
    """Non-benchmark, non-network tests/ UT with xdist and coverage."""
    return [
        python_exe,
        "-m",
        "pytest",
        "tests/",
        "-m",
        _PYTEST_MARKER_WAVE_A,
        *pytest_xdist_args(),
        *cov_pytest_args(),
        "-vv",
        "--tb=line",
        "--disable-warnings",
    ]


def _build_pytest_cmd_wave_b(python_exe: str) -> list[str]:
    """Benchmark or network tests serially (no xdist); Hub cache-safe; separate coverage."""
    return [
        python_exe,
        "-m",
        "pytest",
        "tests/",
        "-m",
        _PYTEST_MARKER_WAVE_B,
        *cov_pytest_args(),
        "-vv",
        "--tb=line",
        "--disable-warnings",
    ]


def _combine_pytest_exits(*exits: int) -> int:
    if any(code == PYTEST_TIMEOUT_EXIT_CODE for code in exits):
        return PYTEST_TIMEOUT_EXIT_CODE
    for code in exits:
        if code != 0:
            return code
    return 0


def _load_vendored_config(fixture_dir: str) -> dict[str, object] | None:
    config_path = REPO_ROOT / "tests" / "assets" / "model_config" / fixture_dir / "config.json"
    if not config_path.is_file():
        return None
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def _fetch_hub_config(model_id: str) -> dict[str, object]:
    from transformers import AutoConfig

    warn_remote_code_risk(model_id, "huggingface")
    try:
        hf_config = AutoConfig.from_pretrained(model_id)
    except (OSError, ValueError, KeyError):
        hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    hub = hf_config.to_dict()
    if not isinstance(hub, dict):
        msg = f"AutoConfig.to_dict() returned {type(hub).__name__}, expected dict"
        raise TypeError(msg)
    return hub


def _diff_config(
    model_id: str,
    fixture_dir: str,
    vendored: dict[str, object],
    hub: dict[str, object],
) -> list[str]:
    drifts: list[str] = []
    for key in _DRIFT_COMPARE_KEYS:
        old = vendored.get(key)
        new = hub.get(key)
        if old != new:
            drifts.append(f"{model_id} [{fixture_dir}] {key}: vendored={old!r} hub={new!r}")
    return drifts


def _run_config_drift_check() -> tuple[str, ...]:
    """Compare vendored remote configs against the live Hub. Never raises."""
    warnings: list[str] = []
    for model_id, fixture_dir in _DRIFT_FIXTURE_MAP.items():
        vendored = _load_vendored_config(fixture_dir)
        if vendored is None:
            warnings.append(f"{model_id}: missing config.json at fixture '{fixture_dir}' (drift baseline absent)")
            continue
        try:
            hub = _fetch_hub_config(model_id)
        except Exception as exc:
            warnings.append(f"{model_id}: Hub config fetch failed ({exc}); cannot check drift")
            continue
        warnings.extend(_diff_config(model_id, fixture_dir, vendored, hub))
    return tuple(warnings)


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
    """SIGTERM the pytest process group, escalate to SIGKILL on timeout."""
    if proc.poll() is not None:
        return

    if _HAS_PROCESS_GROUPS:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
    else:
        proc.terminate()

    try:
        proc.wait(timeout=sigterm_timeout_seconds)
    except subprocess.TimeoutExpired:
        if _HAS_PROCESS_GROUPS:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=sigterm_timeout_seconds)


def _remaining_timeout_seconds(*, deadline: float | None, timeout_seconds: float | None) -> float:
    if deadline is not None:
        return max(0.01, deadline - time.monotonic())
    if timeout_seconds is not None:
        return timeout_seconds
    return _DEFAULT_NIGHTLY_TIMEOUT_SECONDS


def _cleanup_coverage_artifacts() -> None:
    for name in (_COVERAGE_MERGED_FILE, _COVERAGE_NON_BENCHMARK_FILE, _COVERAGE_BENCHMARK_FILE):
        path = REPO_ROOT / name
        if path.is_file():
            path.unlink()
    for path in REPO_ROOT.glob(".coverage.*"):
        if path.is_file():
            path.unlink()


def _combine_wave_coverage(logger: logging.Logger) -> None:
    wave_paths = [REPO_ROOT / _COVERAGE_NON_BENCHMARK_FILE, REPO_ROOT / _COVERAGE_BENCHMARK_FILE]
    existing = [path for path in wave_paths if path.is_file()]
    merged = REPO_ROOT / _COVERAGE_MERGED_FILE
    if not existing:
        return
    if merged.is_file():
        merged.unlink()
    if len(existing) == 1:
        existing[0].replace(merged)
        return
    cmd = [
        sys.executable,
        "-m",
        "coverage",
        "combine",
        f"--data-file={merged}",
        *[str(path) for path in existing],
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "coverage combine failed"
        logger.warning("coverage combine failed (exit %d): %s", proc.returncode, detail)


def _stream_pytest(
    cmd: list[str],
    cwd: Path,
    *,
    timeout_seconds: float | None = None,
    deadline: float | None = None,
    env_extra: dict[str, str] | None = None,
    log_prefix: str | None = None,
) -> tuple[int, str]:
    """Run pytest with stdout tee'd to the console and a captured buffer."""
    logger = logging.getLogger("nightly")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    if env_extra:
        env.update(env_extra)
    capture = io.StringIO()
    effective_timeout = _remaining_timeout_seconds(deadline=deadline, timeout_seconds=timeout_seconds)

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    capture_truncated = False
    timed_out = False

    def _on_timeout() -> None:
        nonlocal timed_out
        timed_out = True
        logger.error("Nightly pytest exceeded %.0fs timeout", effective_timeout)
        _terminate_process_tree(proc)

    timer = threading.Timer(effective_timeout, _on_timeout)
    timer.start()
    try:
        if proc.stdout is None:
            raise RuntimeError("Failed to capture pytest stdout")
        stdout = proc.stdout
        captured_len = 0
        with contextlib.closing(stdout):
            for line in stdout:
                if timed_out:
                    break
                if log_prefix:
                    sys.stdout.write(f"[{log_prefix}] {line}")
                else:
                    sys.stdout.write(line)
                sys.stdout.flush()
                if captured_len < _PYTEST_CAPTURE_MAX_CHARS:
                    remaining = _PYTEST_CAPTURE_MAX_CHARS - captured_len
                    if len(line) <= remaining:
                        capture.write(line)
                        captured_len += len(line)
                    else:
                        capture.write(line[:remaining])
                        captured_len = _PYTEST_CAPTURE_MAX_CHARS
                        capture_truncated = True
                elif not capture_truncated:
                    capture_truncated = True
            if capture_truncated:
                capture.write("\n... [pytest capture truncated]\n")
            exit_code = PYTEST_TIMEOUT_EXIT_CODE if timed_out else proc.wait()
            if timed_out:
                capture.write(f"\n... [pytest killed after {int(effective_timeout)}s timeout]\n")
            return exit_code, capture.getvalue()
    finally:
        timer.cancel()
        if proc.poll() is None:
            _terminate_process_tree(proc)
            proc.wait()


def _run_pytest_waves(
    logger: logging.Logger,
    *,
    python_exe: str,
    deadline: float,
) -> tuple[int, str, int, str]:
    """Run non-benchmark and benchmark waves in parallel under one shared deadline."""
    _cleanup_coverage_artifacts()
    wave_a_cmd = _build_pytest_cmd_wave_a(python_exe)
    wave_b_cmd = _build_pytest_cmd_wave_b(python_exe)

    def _run_wave(label: str, cmd: list[str], coverage_file: str) -> tuple[int, str]:
        logger.info("Running pytest %s: %s", label, shlex.join(cmd))
        exit_code, stdout = _stream_pytest(
            cmd,
            cwd=REPO_ROOT,
            deadline=deadline,
            env_extra={"COVERAGE_FILE": str(REPO_ROOT / coverage_file)},
            log_prefix=label,
        )
        logger.info("Pytest %s finished with exit=%d", label, exit_code)
        return exit_code, stdout

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="nightly-pytest") as pool:
        future_a = pool.submit(_run_wave, "non-benchmark", wave_a_cmd, _COVERAGE_NON_BENCHMARK_FILE)
        future_b = pool.submit(_run_wave, "benchmark", wave_b_cmd, _COVERAGE_BENCHMARK_FILE)
        wave_a_exit, wave_a_stdout = future_a.result()
        wave_b_exit, wave_b_stdout = future_b.result()

    _combine_wave_coverage(logger)
    return wave_a_exit, wave_a_stdout, wave_b_exit, wave_b_stdout


def _coverage_summary(cfg: Config) -> CoverageSummary | None:
    """Build CoverageSummary from .coverage data."""
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


def _resolve_exit_code(pytest_exit: int, failure_blames: tuple[FailureBlame, ...], *, timed_out: bool) -> int:
    if timed_out:
        return PYTEST_TIMEOUT_EXIT_CODE
    if any(blame.needs_human for blame in failure_blames):
        return ATTRIBUTION_HARD_FAIL_EXIT_CODE
    return pytest_exit


def emit_report(
    pytest_stdout: str,
    *,
    overall_exit: int,
    stats: NightlyRunStats | None = None,
    coverage: CoverageSummary | None,
    webhook_url: str | None,
    pipeline_log_url: str = "",
    drift_warnings: tuple[str, ...] = (),
    failure_blames: tuple[FailureBlame, ...] | None = None,
    timed_out: bool = False,
    status_note: str = "",
) -> NightlyRunStats:
    """Parse pytest stdout, push report to Feishu when webhook is set."""
    logger = logging.getLogger("nightly")
    resolved_stats = stats if stats is not None else parse_pytest_stdout(pytest_stdout, exit_code=overall_exit)
    env = fetch_env_info()

    if not webhook_url:
        logger.warning("FEISHU_WEBHOOK_URL not set — skipping Feishu push")
        return resolved_stats

    resolved_blames = failure_blames if failure_blames is not None else ()
    infra_message = resolved_stats.first_error if overall_exit != 0 and not resolved_stats.failed_cases else ""

    report = FeishuReportInput(
        timestamp=env.timestamp,
        branch=env.branch,
        commit=env.commit,
        passed=resolved_stats.passed,
        failed=resolved_stats.failed,
        errors=resolved_stats.errors,
        duration_sec=resolved_stats.duration_sec,
        overall_exit=overall_exit,
        coverage_line_percent=coverage.line_percent if coverage else None,
        coverage_branch_percent=coverage.branch_percent if coverage else None,
        coverage_line_threshold=coverage.line_threshold if coverage else None,
        coverage_branch_threshold=coverage.branch_threshold if coverage else None,
        coverage_gate_passed=coverage.gate_passed if coverage else None,
        failure_blames=resolved_blames,
        drift_warnings=drift_warnings,
        pipeline_log_url=pipeline_log_url,
        infra_message=infra_message,
        timed_out=timed_out,
        status_note=status_note,
    )
    push_feishu_report(webhook_url, report)
    return resolved_stats


def emit_init_failure_report(
    *,
    webhook_url: str | None,
    pipeline_log_url: str,
    error: BaseException,
) -> None:
    """Feishu + console on init/early failure (before pytest completes)."""
    logger = logging.getLogger("nightly")
    note = f"Nightly init failed: {error}"
    logger.error("%s", note)
    if pipeline_log_url:
        logger.error("Pipeline log: %s", pipeline_log_url)
    if not webhook_url:
        return
    env = fetch_env_info()
    report = FeishuReportInput(
        timestamp=env.timestamp,
        branch=env.branch,
        commit=env.commit,
        passed=0,
        failed=0,
        errors=0,
        duration_sec=-1.0,
        overall_exit=1,
        coverage_line_percent=None,
        coverage_branch_percent=None,
        coverage_line_threshold=None,
        coverage_branch_threshold=None,
        coverage_gate_passed=None,
        pipeline_log_url=pipeline_log_url,
        status_note=note,
    )
    push_feishu_report(webhook_url, report)


def _build_terminal_summary(
    *,
    pytest_exit: int,
    stats: NightlyRunStats,
    failure_blames: tuple[FailureBlame, ...],
    coverage: CoverageSummary | None,
    drift_warnings: tuple[str, ...],
    timed_out: bool = False,
    status_note: str = "",
) -> list[str]:
    """Expert/agent console lines — node id, status, last error, attribution."""
    lines = [
        (
            f"Nightly exit={pytest_exit}: passed={stats.passed} "
            f"failed={stats.failed} errors={stats.errors} "
            f"duration={stats.duration_sec:.0f}s"
            if stats.duration_sec >= 0
            else (
                f"Nightly exit={pytest_exit}: passed={stats.passed} "
                f"failed={stats.failed} errors={stats.errors} duration=n/a"
            )
        ),
    ]
    if timed_out:
        lines.append("Timed out: partial results below")
    if status_note:
        lines.append(status_note)
    if coverage is not None:
        lines.append(coverage.message)
    for blame in failure_blames:
        parts = [f"Failed: {blame.node_id}", f"status={display_status(blame)}"]
        if blame.attributed and blame.commit_id != "unknown":
            parts.append(f"commit={blame.commit_id}")
            parts.append(f"author={blame.author}")
            if blame.subject:
                parts.append(f"subject={blame.subject}")
        if blame.last_reason:
            # Last error line only — no fix advice.
            error_line = blame.last_reason.strip().splitlines()[-1].strip()
            parts.append(f"error={error_line}")
        lines.append(" | ".join(parts))
    if stats.first_error and not stats.failed_cases:
        lines.append(f"Infra: {stats.first_error}")
    if drift_warnings:
        lines.append(f"Config drift: {drift_warnings[0]}")
        if len(drift_warnings) > 1:
            lines.append(f"  ... and {len(drift_warnings) - 1} more (Feishu lists up to {FEISHU_DRIFT_LIMIT})")
    follow_up = sum(1 for blame in failure_blames if blame.needs_human)
    if follow_up:
        lines.append(f"{follow_up} failure(s) need follow-up (e.g. no good commit within lookback)")
    return lines


def _run_nightly_pipeline(
    logger: logging.Logger,
    cfg: Config,
    feishu_url: str | None,
    pipeline_log_url: str,
    *,
    timeout_seconds: float,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    logger.info("Starting nightly UT waves in parallel: %s and %s", _PYTEST_MARKER_WAVE_A, _PYTEST_MARKER_WAVE_B)
    wave_a_exit, wave_a_stdout, wave_b_exit, wave_b_stdout = _run_pytest_waves(
        logger,
        python_exe=sys.executable,
        deadline=deadline,
    )

    pytest_exit = _combine_pytest_exits(wave_a_exit, wave_b_exit)
    pytest_stdout = wave_a_stdout + ("\n" if wave_a_stdout and not wave_a_stdout.endswith("\n") else "") + wave_b_stdout
    timed_out = pytest_exit == PYTEST_TIMEOUT_EXIT_CODE
    logger.info("Nightly pytest combined exit=%d", pytest_exit)

    coverage = _coverage_summary(cfg)
    if coverage:
        logger.info("%s", coverage.message)
        if not coverage.gate_passed:
            logger.warning(
                "Nightly coverage below threshold (non-blocking): line=%.1f%% branch=%.1f%%",
                coverage.line_percent,
                coverage.branch_percent,
            )

    drift_warnings: tuple[str, ...] = ()
    if timed_out or time.monotonic() >= deadline:
        logger.info("Skipping Hub drift check after nightly timeout/budget exhausted")
    else:
        logger.info("Drift check: vendored remote configs vs live Hub (non-blocking)")
        drift_warnings = _run_config_drift_check()
        if drift_warnings:
            logger.warning(
                "Config drift / baseline warnings (non-blocking, %d):",
                len(drift_warnings),
            )
            for warning in drift_warnings:
                logger.warning("  - %s", warning)

    logger.info("Building report ...")
    wave_a_stats = parse_pytest_stdout(wave_a_stdout, exit_code=wave_a_exit)
    wave_b_stats = parse_pytest_stdout(wave_b_stdout, exit_code=wave_b_exit)
    stats = merge_nightly_run_stats(wave_a_stats, wave_b_stats)
    failure_blames: tuple[FailureBlame, ...] = ()
    status_note = "Timed out; partial results below" if timed_out else ""

    if stats.failed_cases and not timed_out:
        results = attribute_failures(
            REPO_ROOT,
            stats.failed_cases,
            python_exe=sys.executable,
            max_workers=default_max_workers(len(stats.failed_cases)),
            deadline=deadline,
        )
        failure_blames = results_to_failure_blames(results, failure_reasons=stats.failure_reasons)
        if any(blame.subject == ATTRIBUTION_BUDGET_SKIP_LABEL for blame in failure_blames):
            timed_out = True
            status_note = "Timed out; partial results below"
    elif stats.failed_cases and timed_out:
        failure_blames = tuple(
            FailureBlame(
                node_id=node,
                commit_id="unknown",
                author="unknown",
                subject=ATTRIBUTION_BUDGET_SKIP_LABEL,
                conclusion=AttributionConclusion.NEED_HUMAN,
                last_reason=stats.failure_reasons.get(node, ""),
            )
            for node in stats.failed_cases
        )

    exit_code = _resolve_exit_code(pytest_exit, failure_blames, timed_out=timed_out)

    emit_report(
        pytest_stdout,
        overall_exit=exit_code,
        stats=stats,
        coverage=coverage,
        webhook_url=feishu_url,
        pipeline_log_url=pipeline_log_url,
        drift_warnings=drift_warnings,
        failure_blames=failure_blames,
        timed_out=timed_out,
        status_note=status_note,
    )

    summary_lines = _build_terminal_summary(
        pytest_exit=exit_code,
        stats=stats,
        failure_blames=failure_blames,
        coverage=coverage,
        drift_warnings=drift_warnings,
        timed_out=timed_out,
        status_note=status_note,
    )
    for line in summary_lines:
        logger.info("%s", line)
    return exit_code


def main() -> int:
    logger = setup_logger("nightly")
    feishu_url: str | None = (os.environ.get("FEISHU_WEBHOOK_URL") or "").strip() or None
    pipeline_log_url = (os.environ.get("MSMODELING_PIPELINE_LOG_URL") or "").strip()
    try:
        cfg = Config.from_env()
        log_env_audit(cfg, logger)
        feishu_url = cfg.feishu_webhook_url or None
        timeout_seconds = resolve_nightly_timeout_seconds()
        logger.info("Nightly self-timeout: %.0fs (%s)", timeout_seconds, _TIMEOUT_ENV)
        return _run_nightly_pipeline(
            logger,
            cfg,
            feishu_url,
            pipeline_log_url,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        emit_init_failure_report(
            webhook_url=feishu_url,
            pipeline_log_url=pipeline_log_url,
            error=exc,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
