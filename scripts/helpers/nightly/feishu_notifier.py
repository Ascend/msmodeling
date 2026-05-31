"""Feishu webhook push for nightly report notifications."""

from __future__ import annotations

import json
import logging
import urllib.request

FEISHU_TIMEOUT_SEC = 10

logger = logging.getLogger(__name__)


def build_feishu_payload(
    *,
    timestamp: str,
    branch: str,
    commit: str,
    passed: int,
    failed: int,
    duration_sec: float,
    coverage_line_percent: float | None,
    coverage_branch_percent: float | None,
    coverage_line_threshold: float | None,
    coverage_branch_threshold: float | None,
    coverage_gate_passed: bool | None,
    test_map_source_files: int,
    test_map_symbols: int,
    test_map_written: bool,
    failed_cases: tuple[str, ...],
    first_error: str,
    weak_coverage_symbols: tuple[str, ...] = (),
    redundancy_warnings: tuple[dict[str, object], ...] = (),
    expired_exemption_section: str = "",
) -> dict:
    """Build Feishu text message payload dict. Does not send."""
    status = "All passed" if failed == 0 else f"{failed} failed"
    lines = [
        f"Nightly Report — {timestamp[:10]}",
        f"Branch: {branch} | Commit: {commit}",
        f"Result: {status}",
        f"Passed: {passed} | Failed: {failed} | Duration: {duration_sec:.0f}s",
    ]
    if coverage_line_percent is not None and coverage_branch_percent is not None:
        cov_status = "PASS" if coverage_gate_passed else "BELOW THRESHOLD"
        lines.append(
            f"Coverage ({cov_status}): line {coverage_line_percent:.1f}% "
            f"(>={coverage_line_threshold:.0f}%) | branch {coverage_branch_percent:.1f}% "
            f"(>={coverage_branch_threshold:.0f}%)"
        )
    if test_map_written:
        lines.append(f"Test map: {test_map_source_files} files / {test_map_symbols} symbols (updated)")
    else:
        lines.append("Test map: not updated (UT phase failed)")
    if weak_coverage_symbols:
        lines.append(f"\nWeak coverage symbols ({len(weak_coverage_symbols)}):")
        for sym in weak_coverage_symbols[:10]:
            lines.append(f"- {sym}")
        if len(weak_coverage_symbols) > 10:
            lines.append(f"- ... and {len(weak_coverage_symbols) - 10} more")
    if redundancy_warnings:
        over_covered = [w for w in redundancy_warnings if w.get("type") == "over_covered_symbol"]
        redundant_pairs = [w for w in redundancy_warnings if w.get("type") == "redundant_pair"]
        if over_covered:
            lines.append(f"\nOver-covered symbols ({len(over_covered)}):")
            for w in over_covered[:10]:
                lines.append(f"- {w['symbol']} ({w['test_count']} tests, threshold {w['threshold']})")
            if len(over_covered) > 10:
                lines.append(f"- ... and {len(over_covered) - 10} more")
        if redundant_pairs:
            lines.append(f"\nRedundant test pairs ({len(redundant_pairs)}):")
            for w in redundant_pairs[:10]:
                lines.append(f"- {w['test_a']} / {w['test_b']} (Jaccard={w['jaccard']:.2f})")
            if len(redundant_pairs) > 10:
                lines.append(f"- ... and {len(redundant_pairs) - 10} more")
    if expired_exemption_section:
        lines.append(expired_exemption_section)
    if failed_cases:
        lines.append("\nFailed cases:")
        lines.extend(f"- {case}" for case in failed_cases[:20])
        if len(failed_cases) > 20:
            lines.append(f"- ... and {len(failed_cases) - 20} more")
    if first_error:
        lines.append(f"\nFirst error: {first_error}")

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


def push_feishu(webhook_url: str, payload: dict) -> None:
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
