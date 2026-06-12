"""Feishu webhook push for nightly report notifications."""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from scripts.helpers.nightly.report_models import FeishuReportInput, PhaseBreakdownEntry

FEISHU_TIMEOUT_SEC: Final = 10
_TRUNCATED_LIST_LIMIT: Final = 10
_FAILED_CASES_LIMIT: Final = 20

logger = logging.getLogger(__name__)


def _format_duration(duration_sec: float) -> str:
    return f"{duration_sec:.0f}s" if duration_sec >= 0 else "n/a"


def _build_nightly_status(report: FeishuReportInput) -> str:
    total_failures = report.failed + report.errors
    if report.overall_exit == 0 and total_failures == 0:
        return "All passed"
    if total_failures > 0:
        return f"{total_failures} failed"
    return f"FAILED (pytest exit {report.overall_exit}, no JUnit testcase data)"


def _render_summary_header(report: FeishuReportInput, status: str) -> list[str]:
    lines = [
        f"Nightly Report — {report.timestamp[:10]}",
        f"Branch: {report.branch} | Commit: {report.commit}",
        f"Result: {status}",
        (
            f"Passed: {report.passed} | Failed: {report.failed} | Errors: {report.errors} "
            f"| Duration: {_format_duration(report.duration_sec)}"
        ),
    ]
    if report.overall_exit != 0:
        lines.append(f"Overall exit code: {report.overall_exit}")
    return lines


def _render_phase_line(phase: PhaseBreakdownEntry) -> str:
    line = f"- {phase.label}: passed {phase.passed} / failed {phase.failed} / {_format_duration(phase.duration_sec)}"
    if phase.exit_code != 0:
        line += f" (exit {phase.exit_code})"
    if phase.infra_failure:
        line += " — no JUnit details"
    return line


def _render_phase_breakdown(phases: tuple[PhaseBreakdownEntry, ...]) -> list[str]:
    if not phases:
        return []
    lines = ["\nPer-phase:"]
    lines.extend(_render_phase_line(phase) for phase in phases)
    return lines


def _render_slowest_tests(slowest_tests: tuple[tuple[str, float], ...]) -> list[str]:
    if not slowest_tests:
        return []
    lines = [f"\nSlowest tests (top {len(slowest_tests)}):"]
    lines.extend(f"- {seconds:.1f}s {node_id}" for node_id, seconds in slowest_tests)
    return lines


def _render_coverage_section(report: FeishuReportInput) -> list[str]:
    if report.coverage_line_percent is None or report.coverage_branch_percent is None:
        return []
    cov_status = "PASS" if report.coverage_gate_passed else "BELOW THRESHOLD"
    return [
        (
            f"Coverage ({cov_status}): line {report.coverage_line_percent:.1f}% "
            f"(>={report.coverage_line_threshold:.0f}%) | branch {report.coverage_branch_percent:.1f}% "
            f"(>={report.coverage_branch_threshold:.0f}%)"
        )
    ]


def _render_test_map_line(report: FeishuReportInput) -> list[str]:
    if report.test_map_written:
        return [f"Test map: {report.test_map_source_files} files / {report.test_map_symbols} symbols (updated)"]
    return ["Test map: not updated (UT phase failed)"]


def _render_truncated_bullets(title: str, items: tuple[str, ...], *, limit: int) -> list[str]:
    if not items:
        return []
    lines = [title]
    lines.extend(f"- {item}" for item in items[:limit])
    remaining = len(items) - limit
    if remaining > 0:
        lines.append(f"- ... and {remaining} more")
    return lines


def _render_weak_coverage(symbols: tuple[str, ...]) -> list[str]:
    if not symbols:
        return []
    return _render_truncated_bullets(
        f"\nWeak coverage symbols ({len(symbols)}):",
        symbols,
        limit=_TRUNCATED_LIST_LIMIT,
    )


def _render_redundancy_warnings(warnings: tuple[dict[str, object], ...]) -> list[str]:
    if not warnings:
        return []
    lines: list[str] = []
    over_covered = [warning for warning in warnings if warning.get("type") == "over_covered_symbol"]
    redundant_pairs = [warning for warning in warnings if warning.get("type") == "redundant_pair"]

    if over_covered:
        lines.append(f"\nOver-covered symbols ({len(over_covered)}):")
        lines.extend(
            f"- {warning['symbol']} ({warning['test_count']} tests, threshold {warning['threshold']})"
            for warning in over_covered[:_TRUNCATED_LIST_LIMIT]
        )
        remaining = len(over_covered) - _TRUNCATED_LIST_LIMIT
        if remaining > 0:
            lines.append(f"- ... and {remaining} more")

    if redundant_pairs:
        lines.append(f"\nRedundant test pairs ({len(redundant_pairs)}):")
        lines.extend(
            f"- {warning['test_a']} / {warning['test_b']} (Jaccard={warning['jaccard']:.2f})"
            for warning in redundant_pairs[:_TRUNCATED_LIST_LIMIT]
        )
        remaining = len(redundant_pairs) - _TRUNCATED_LIST_LIMIT
        if remaining > 0:
            lines.append(f"- ... and {remaining} more")
    return lines


def _render_drift_warnings(warnings: tuple[str, ...]) -> list[str]:
    if not warnings:
        return []
    return _render_truncated_bullets(f"\nConfig drift ({len(warnings)}):", warnings, limit=_TRUNCATED_LIST_LIMIT)


def _render_failed_cases(failed_cases: tuple[str, ...]) -> list[str]:
    if not failed_cases:
        return []
    lines = ["\nFailed cases:"]
    lines.extend(f"- {case}" for case in failed_cases[:_FAILED_CASES_LIMIT])
    remaining = len(failed_cases) - _FAILED_CASES_LIMIT
    if remaining > 0:
        lines.append(f"- ... and {remaining} more")
    return lines


def _render_first_error(first_error: str) -> list[str]:
    if not first_error:
        return []
    return [f"\nFirst error: {first_error}"]


def build_feishu_payload(report: FeishuReportInput) -> dict[str, Any]:
    """Build Feishu text message payload dict. Does not send."""
    status = _build_nightly_status(report)
    lines = _render_summary_header(report, status)
    lines.extend(_render_phase_breakdown(report.phase_breakdown))
    lines.extend(_render_slowest_tests(report.slowest_tests))
    lines.extend(_render_coverage_section(report))
    lines.extend(_render_test_map_line(report))
    lines.extend(_render_weak_coverage(report.weak_coverage_symbols))
    lines.extend(_render_redundancy_warnings(report.redundancy_warnings))
    if report.expired_exemption_section:
        lines.append(report.expired_exemption_section)
    lines.extend(_render_drift_warnings(report.drift_warnings))
    lines.extend(_render_failed_cases(report.failed_cases))
    lines.extend(_render_first_error(report.first_error))

    return {
        "msg_type": "text",
        "content": {"text": "\n".join(lines)},
    }


def _parse_feishu_response(body: str) -> None:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        logger.info("Feishu HTTP response (non-JSON): %s", body)
        return

    code = parsed.get("code")
    msg = parsed.get("msg", "")
    if code is not None and code != 0:
        logger.warning("Feishu push rejected: code=%s msg=%s", code, msg)
        return

    logger.info("Feishu push accepted: code=%s msg=%s", code, msg)


def push_feishu(webhook_url: str, payload: dict[str, Any]) -> None:
    """Send payload to Feishu webhook. Non-blocking on failure."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=FEISHU_TIMEOUT_SEC) as resp:
            _parse_feishu_response(resp.read().decode())
    except OSError as exc:
        logger.warning("Feishu push failed (non-blocking): %s", exc)
