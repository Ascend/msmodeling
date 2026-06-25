#!/usr/bin/env python3
"""Nightly: multi-phase UT, refresh test_map, nightly + benchmark + network, report.

CLI entry point for run_nightly.sh. Phase 1: smoke/regression ``not npu and
not nightly and not network`` with coverage → test_map on success. Phase 2a:
smoke/regression ``not npu and nightly and not network``. Phase 2b: benchmark.
Phase 2c: ``not npu and network`` real model Hub cases. A non-blocking config
drift check then compares vendored remote configs against the live Hub → report.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final, Protocol

from scripts.helpers._config import Config, ConfigError
from scripts.helpers._paths import REPO_ROOT
from scripts.helpers.ci_gate.gate_policy import (
    find_expired_test_exemptions,
    format_expired_exemptions_section,
    format_expired_test_exemptions_section,
    load_gate_policy,
)
from scripts.helpers.common._logging import log_env_audit, setup_logger
from scripts.helpers.common.build_test_map import (
    collect_allowed_node_ids,
    collect_test_map,
    detect_redundant_cases,
    write_test_map,
)
from scripts.helpers.common.coverage_config import cov_pytest_args, pytest_xdist_args
from scripts.helpers.common.coverage_gate import (
    GateConfig,
    check_thresholds,
    load_totals,
)
from scripts.helpers.common.test_map_config import (
    TEST_MAP_COLLECTION_MARKER,
    TEST_MAP_EXECUTION_MARKER,
    resolve_test_map_path,
)
from scripts.helpers.common.test_map_report import find_expired_unmapped_in_map
from scripts.helpers.nightly.feishu_notifier import build_feishu_payload, push_feishu
from scripts.helpers.nightly.pytest_parser import (
    NightlyRunStats,
    aggregate_phase_stats,
    parse_junit_phases,
    slowest_testcase_durations,
)
from scripts.helpers.nightly.report_builder import (
    build_phase_breakdown,
    compute_weak_coverage_symbols,
    fetch_env_info,
    load_test_map_summary,
    resolve_first_error,
)
from scripts.helpers.nightly.report_models import CoverageSummary, FeishuReportInput
from tensor_cast.core.model_source_security import warn_remote_code_risk

_NIGHTLY_MARKER = "not npu and nightly and not network"
_NETWORK_MARKER = "not npu and network"
_PROCESS_TERMINATE_TIMEOUT_SECONDS: Final[float] = 5.0
_PHASE_LABELS: Final[tuple[str, ...]] = (
    "smoke UT (coverage mapping)",
    "long-running tests",
    "benchmark",
    "network Hub tests",
)
_SLOWEST_TESTS_TOP_N: Final[int] = 10

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


# ---------------------------------------------------------------------------
# Pytest command builders
# ---------------------------------------------------------------------------


def _build_test_map_pytest_cmd(python_exe: str, *, junit_xml: Path) -> list[str]:
    """Phase 1: incremental-scope smoke/regression for coverage + test_map."""
    return [
        python_exe,
        "-m",
        "pytest",
        "tests/smoke/",
        "tests/regression/",
        "-m",
        TEST_MAP_EXECUTION_MARKER,
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
        "--dist",
        "worksteal",
        "-q",
        "--no-header",
        "--tb=short",
        "--disable-warnings",
        "--durations=20",
        f"--junit-xml={junit_xml}",
    ]


def _build_benchmark_pytest_cmd(python_exe: str, cfg: Config, *, junit_xml: Path) -> list[str]:
    cmd = [
        python_exe,
        "-m",
        "pytest",
        str(REPO_ROOT / "tests" / "benchmark"),
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
        cmd.extend(["-n", "auto", "--dist", "worksteal"])
    return cmd


def _build_network_pytest_cmd(python_exe: str, *, junit_xml: Path) -> list[str]:
    """Phase 2c: network-marked cases validated against the live model Hub.

    Runs serially: the cases share one Hub cache dir, so concurrent workers
    could race on the same config-only snapshot download.
    """
    return [
        python_exe,
        "-m",
        "pytest",
        "tests/",
        "-m",
        _NETWORK_MARKER,
        "-q",
        "--no-header",
        "--tb=short",
        "--disable-warnings",
        "--durations=20",
        f"--junit-xml={junit_xml}",
    ]


# ---------------------------------------------------------------------------
# Config drift check (non-blocking)
# ---------------------------------------------------------------------------


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
    except Exception:
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
    """Compare vendored remote configs against the live Hub. Never raises.

    Config-only ``AutoConfig`` fetches (no weight shards) for mapped fixtures.
    """
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


# ---------------------------------------------------------------------------
# Pytest runner (streaming)
# ---------------------------------------------------------------------------


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


def _stream_pytest(cmd: list[str], cwd: Path, *, log_file: Path | None = None) -> int:
    """Run pytest, return exit code.

    With ``log_file`` unset the output is streamed to the console line by line.
    When ``log_file`` is provided the output is captured there instead, keeping
    the console quiet (used when a Feishu webhook carries the detailed report).
    """
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
        with contextlib.ExitStack() as stack:
            sink = stack.enter_context(log_file.open("w", encoding="utf-8")) if log_file is not None else None
            if proc.stdout is None:
                raise RuntimeError("Failed to capture pytest stdout")
            stdout = proc.stdout
            stack.enter_context(contextlib.closing(stdout))
            try:
                for line in stdout:
                    if sink is not None:
                        sink.write(line)
                    else:
                        sys.stdout.write(line)
                        sys.stdout.flush()
            except KeyboardInterrupt:
                _terminate_process_tree(proc)
                raise
            return proc.wait()
    finally:
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
    overall_exit: int = 0,
    phase_exits: tuple[int, ...] = (),
    phase_log_paths: tuple[Path | None, ...] = (),
    weak_coverage_symbols: tuple[str, ...] = (),
    redundancy_warnings: tuple[dict[str, object], ...] = (),
    expired_exemption_section: str = "",
    drift_warnings: tuple[str, ...] = (),
) -> tuple[NightlyRunStats, tuple[NightlyRunStats | None, ...]]:
    """Parse pytest JUnit XML, push report to Feishu when webhook is set.

    Returns aggregate stats and per-phase stats so callers avoid re-parsing XML.
    """
    logger = logging.getLogger("nightly")
    phase_parsed = parse_junit_phases(junit_xml_paths)
    phase_stats = tuple(phase.stats if phase is not None else None for phase in phase_parsed)
    stats = aggregate_phase_stats(phase_parsed)
    all_durations = [duration for phase in phase_parsed if phase is not None for duration in phase.durations]
    env = fetch_env_info()
    test_map = load_test_map_summary(test_map_path if test_map_written else None)

    if not webhook_url:
        logger.warning("FEISHU_WEBHOOK_URL not set — skipping Feishu push")
        return stats, phase_stats

    phase_exit_list = phase_exits or tuple(0 for _ in junit_xml_paths)
    phase_log_list = phase_log_paths or tuple(None for _ in junit_xml_paths)
    phase_labels = _PHASE_LABELS[: len(junit_xml_paths)]
    phase_breakdown = build_phase_breakdown(
        phase_labels,
        junit_xml_paths,
        phase_exit_list,
        phase_stats=phase_stats,
    )
    slowest_tests = slowest_testcase_durations(all_durations, top_n=_SLOWEST_TESTS_TOP_N)
    first_error = resolve_first_error(stats, phase_exit_list, phase_log_list)

    report = FeishuReportInput(
        timestamp=env.timestamp,
        branch=env.branch,
        commit=env.commit,
        passed=stats.passed,
        failed=stats.failed,
        errors=stats.errors,
        duration_sec=stats.duration_sec,
        overall_exit=overall_exit,
        coverage_line_percent=coverage.line_percent if coverage else None,
        coverage_branch_percent=coverage.branch_percent if coverage else None,
        coverage_line_threshold=coverage.line_threshold if coverage else None,
        coverage_branch_threshold=coverage.branch_threshold if coverage else None,
        coverage_gate_passed=coverage.gate_passed if coverage else None,
        test_map_test_nodes=test_map.test_nodes,
        test_map_symbol_refs=test_map.symbol_refs,
        test_map_written=test_map_written,
        failed_cases=stats.failed_cases,
        first_error=first_error,
        weak_coverage_symbols=weak_coverage_symbols,
        redundancy_warnings=redundancy_warnings,
        expired_exemption_section=expired_exemption_section,
        phase_breakdown=phase_breakdown,
        slowest_tests=slowest_tests,
        drift_warnings=drift_warnings,
    )
    payload = build_feishu_payload(report)
    push_feishu(webhook_url, payload)
    return stats, phase_stats


# ---------------------------------------------------------------------------
# Terminal summary (stdout)
# ---------------------------------------------------------------------------


def _build_terminal_summary(
    *,
    overall_exit: int,
    stats: NightlyRunStats,
    phase_labels: tuple[str, ...],
    junit_paths: tuple[Path, ...],
    phase_exits: tuple[int, ...],
    coverage: CoverageSummary | None,
    drift_warnings: tuple[str, ...],
    phase_log_paths: tuple[Path | None, ...],
    include_phase_logs: bool,
    phase_stats: tuple[NightlyRunStats | None, ...] | None = None,
) -> list[str]:
    """Build human-readable nightly stdout summary lines."""
    if phase_stats is None:
        phase_stats = tuple(parse_junit_phases(junit_paths))
        phase_stats = tuple(phase.stats if phase is not None else None for phase in phase_stats)
    lines: list[str] = []
    for label, parsed_stats, exit_code in zip(phase_labels, phase_stats, phase_exits, strict=True):
        if parsed_stats is None:
            passed = 0
            failed = 0
            duration = "n/a"
        else:
            passed = parsed_stats.passed
            failed = parsed_stats.failed + parsed_stats.errors
            duration = f"{parsed_stats.duration_sec:.0f}s" if parsed_stats.duration_sec >= 0 else "n/a"
        lines.append(f"{label}: exit={exit_code} passed={passed} failed={failed} duration={duration}")

    lines.append(
        f"Nightly exit={overall_exit}: passed={stats.passed} "
        f"failed={stats.failed} errors={stats.errors} "
        f"duration={stats.duration_sec:.0f}s"
    )
    if coverage is not None:
        lines.append(coverage.message)
    if drift_warnings:
        lines.append(f"Config drift: {drift_warnings[0]}")
    if include_phase_logs:
        for label, log_path in zip(phase_labels, phase_log_paths, strict=True):
            if log_path is not None:
                lines.append(f"{label} log: {log_path}")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _write_test_map_artifacts(
    logger: logging.Logger,
    cfg: Config,
    test_map_path: Path,
    map_exit: int,
    *,
    allowed_node_ids: frozenset[str] | None = None,
) -> tuple[bool, tuple[str, ...], tuple[dict[str, object], ...], str]:
    if map_exit != 0:
        logger.warning("Skipping coverage mapping write: smoke UT failed")
        return False, (), (), ""

    logger.info("Building coverage mapping from collected coverage data ...")
    gate_policy = load_gate_policy(REPO_ROOT)
    fresh_map = collect_test_map(
        marker_expr=TEST_MAP_COLLECTION_MARKER,
        roots=gate_policy.roots,
        allowed_node_ids=allowed_node_ids,
    )
    write_test_map(test_map_path, fresh_map)
    redundancy = tuple(detect_redundant_cases(fresh_map))
    weak_symbols = compute_weak_coverage_symbols(
        test_map_path,
        REPO_ROOT / ".coverage",
        mapping=fresh_map,
    )
    expired = find_expired_unmapped_in_map(gate_policy, fresh_map)
    expired_tests = find_expired_test_exemptions(gate_policy)
    expired_section = format_expired_exemptions_section(expired) + format_expired_test_exemptions_section(expired_tests)
    if expired:
        logger.warning(
            "Found %d expired exemption(s) still without coverage mapping",
            len(expired),
        )
    if expired_tests:
        logger.warning("Found %d expired test exemption(s)", len(expired_tests))
    logger.info("Coverage mapping written: %s", test_map_path)
    return True, weak_symbols, redundancy, expired_section


def _run_nightly_pipeline(
    logger: logging.Logger,
    cfg: Config,
    test_map_path: Path,
    feishu_url: str | None,
    junit_root: Path,
) -> int:
    map_junit = junit_root / "phase1_test_map.xml"
    nightly_junit = junit_root / "phase2a_nightly.xml"
    bench_junit = junit_root / "phase2b_benchmark.xml"
    network_junit = junit_root / "phase2c_network.xml"

    def _phase_log(name: str) -> Path | None:
        return junit_root / f"{name}.log" if feishu_url else None

    phase_log_paths = (
        _phase_log("phase1_test_map"),
        _phase_log("phase2a_nightly"),
        _phase_log("phase2b_benchmark"),
        _phase_log("phase2c_network"),
    )

    logger.info(
        "Starting smoke UT (coverage mapping): tests/smoke + tests/regression, marker=%s",
        TEST_MAP_EXECUTION_MARKER,
    )
    map_cmd = _build_test_map_pytest_cmd(sys.executable, junit_xml=map_junit)
    logger.info("Running pytest: %s", shlex.join(map_cmd))
    map_exit = _stream_pytest(map_cmd, cwd=REPO_ROOT, log_file=_phase_log("phase1_test_map"))
    logger.info("Smoke UT finished with exit=%d", map_exit)

    allowed_node_ids: frozenset[str] | None = None
    if map_exit == 0:
        allowed_node_ids = collect_allowed_node_ids(TEST_MAP_COLLECTION_MARKER)

    coverage = _coverage_summary(cfg)
    if coverage:
        logger.info("%s", coverage.message)
        if not coverage.gate_passed:
            logger.warning(
                "Nightly coverage below threshold (non-blocking): line=%.1f%% branch=%.1f%%",
                coverage.line_percent,
                coverage.branch_percent,
            )

    map_written, weak_symbols, redundancy, expired_section = _write_test_map_artifacts(
        logger,
        cfg,
        test_map_path,
        map_exit,
        allowed_node_ids=allowed_node_ids,
    )

    logger.info(
        "Starting long-running tests: tests/smoke + tests/regression, marker=%s",
        _NIGHTLY_MARKER,
    )
    nightly_cmd = _build_nightly_pytest_cmd(sys.executable, junit_xml=nightly_junit)
    logger.info("Running pytest: %s", shlex.join(nightly_cmd))
    nightly_exit = _stream_pytest(nightly_cmd, cwd=REPO_ROOT, log_file=_phase_log("phase2a_nightly"))
    logger.info("Long-running tests finished with exit=%d", nightly_exit)

    logger.info("Starting benchmark: tests/benchmark")
    bench_cmd = _build_benchmark_pytest_cmd(sys.executable, cfg, junit_xml=bench_junit)
    logger.info("Running pytest: %s", shlex.join(bench_cmd))
    bench_exit = _stream_pytest(bench_cmd, cwd=REPO_ROOT, log_file=_phase_log("phase2b_benchmark"))
    logger.info("Benchmark finished with exit=%d", bench_exit)

    logger.info("Starting network Hub tests: tests/, marker=%s", _NETWORK_MARKER)
    network_cmd = _build_network_pytest_cmd(sys.executable, junit_xml=network_junit)
    logger.info("Running pytest: %s", shlex.join(network_cmd))
    network_exit = _stream_pytest(network_cmd, cwd=REPO_ROOT, log_file=_phase_log("phase2c_network"))
    logger.info("Network Hub tests finished with exit=%d", network_exit)

    logger.info("Drift check: vendored remote configs vs live Hub (non-blocking)")
    drift_warnings = _run_config_drift_check()
    if drift_warnings:
        logger.warning(
            "Config drift / baseline warnings (non-blocking, %d):",
            len(drift_warnings),
        )
        for warning in drift_warnings:
            logger.warning("  - %s", warning)

    overall_exit = map_exit or nightly_exit or bench_exit or network_exit

    logger.info("Building report ...")
    stats, phase_stats = emit_report(
        (map_junit, nightly_junit, bench_junit, network_junit),
        coverage=coverage,
        test_map_written=map_written,
        test_map_path=test_map_path if map_written else None,
        webhook_url=feishu_url,
        overall_exit=overall_exit,
        phase_exits=(map_exit, nightly_exit, bench_exit, network_exit),
        phase_log_paths=phase_log_paths,
        weak_coverage_symbols=weak_symbols,
        redundancy_warnings=redundancy,
        expired_exemption_section=expired_section,
        drift_warnings=drift_warnings,
    )

    summary_lines = _build_terminal_summary(
        overall_exit=overall_exit,
        stats=stats,
        phase_labels=_PHASE_LABELS,
        junit_paths=(map_junit, nightly_junit, bench_junit, network_junit),
        phase_exits=(map_exit, nightly_exit, bench_exit, network_exit),
        coverage=coverage,
        drift_warnings=drift_warnings,
        phase_log_paths=phase_log_paths,
        include_phase_logs=feishu_url is not None,
        phase_stats=phase_stats,
    )
    print("\n".join(summary_lines))
    return overall_exit


def main() -> int:
    logger = setup_logger("nightly")
    cfg = Config.from_env()
    log_env_audit(cfg, logger)

    try:
        test_map_path = resolve_test_map_path(cfg, must_exist=False)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1

    feishu_url = cfg.feishu_webhook_url or None

    with tempfile.TemporaryDirectory(prefix="nightly-junit-") as junit_dir:
        try:
            return _run_nightly_pipeline(logger, cfg, test_map_path, feishu_url, Path(junit_dir))
        except KeyboardInterrupt:
            logger.warning("Interrupted — stopping nightly run")
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
