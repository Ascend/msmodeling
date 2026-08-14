"""Feishu webhook push for nightly report notifications (card schema 2.0)."""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Final

from scripts.helpers.nightly.report_models import AttributionConclusion

if TYPE_CHECKING:
    from scripts.helpers.nightly.report_models import FailureBlame, FeishuReportInput

FEISHU_TIMEOUT_SEC: Final = 10
FEISHU_DRIFT_LIMIT: Final = 5
# Custom-bot webhook hard limit is 20 KB; keep headroom for JSON overhead.
_FEISHU_MESSAGE_BYTE_BUDGET: Final = 18_000

# Plain-English Feishu labels (internal enums stay first_bad / need_human / …).
STATUS_NEEDS_FOLLOW_UP: Final = "Needs follow-up"
STATUS_ROOT_CAUSE_FOUND: Final = "Root cause found"
STATUS_NOT_REPRODUCED: Final = "Not reproduced"

_STATUS_ORDER: Final = (
    STATUS_NEEDS_FOLLOW_UP,
    STATUS_ROOT_CAUSE_FOUND,
    STATUS_NOT_REPRODUCED,
)

_STATUS_BORDER: Final = {
    STATUS_NEEDS_FOLLOW_UP: "orange",
    STATUS_ROOT_CAUSE_FOUND: "red",
    STATUS_NOT_REPRODUCED: "grey",
}

_STATUS_LEGEND: Final = (
    f"- **{STATUS_NEEDS_FOLLOW_UP}** — could not find introducing commit / needs human follow-up",
    f"- **{STATUS_ROOT_CAUSE_FOUND}** — attribution succeeded; shows introducing commit and author",
    f"- **{STATUS_NOT_REPRODUCED}** — failed in suite but could not reproduce at HEAD",
)

logger = logging.getLogger(__name__)


def _format_duration(duration_sec: float) -> str:
    if duration_sec < 0:
        return "n/a"
    if duration_sec < 60:
        return f"{duration_sec:.0f}s"
    minutes, seconds = divmod(int(duration_sec), 60)
    return f"{minutes}m {seconds}s"


def display_status(blame: FailureBlame) -> str:
    """Map attribution conclusion to the Feishu / console status label."""
    conclusion = blame.conclusion
    if conclusion == AttributionConclusion.FIRST_BAD:
        return STATUS_ROOT_CAUSE_FOUND
    if conclusion == AttributionConclusion.CANNOT_REPRODUCE:
        return STATUS_NOT_REPRODUCED
    # NEED_HUMAN, UNCOLLECTIBLE, and unknown → follow-up.
    return STATUS_NEEDS_FOLLOW_UP


def _summary_counts(report: FeishuReportInput) -> tuple[int, int, int, int]:
    """Return (failed_listed, root_cause_found, needs_follow_up, not_reproduced)."""
    blames = report.failure_blames
    root_cause = sum(1 for b in blames if display_status(b) == STATUS_ROOT_CAUSE_FOUND)
    follow_up = sum(1 for b in blames if display_status(b) == STATUS_NEEDS_FOLLOW_UP)
    not_repro = sum(1 for b in blames if display_status(b) == STATUS_NOT_REPRODUCED)
    return len(blames), root_cause, follow_up, not_repro


def _payload_byte_size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _render_failure_line(blame: FailureBlame) -> str:
    """One failure: full node id; root-cause items also show introduced-by."""
    lines = [f"`{blame.node_id}`"]
    if display_status(blame) == STATUS_ROOT_CAUSE_FOUND and blame.commit_id not in {"", "unknown"}:
        author = blame.author if blame.author and blame.author != "unknown" else "unknown"
        lines.append(f"Introduced by: `{blame.commit_id}` ({author})")
    return "\n".join(lines)


def _group_blames_by_status(
    blames: tuple[FailureBlame, ...],
) -> list[tuple[str, list[FailureBlame]]]:
    by_status: dict[str, list[FailureBlame]] = defaultdict(list)
    for blame in blames:
        by_status[display_status(blame)].append(blame)
    return [(label, by_status[label]) for label in _STATUS_ORDER if label in by_status]


def _element_id(prefix: str, index: int) -> str:
    """Feishu element_id: letters/digits/underscore, start with letter, max 20 chars."""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", prefix)[:12].lstrip("0123456789_") or "grp"
    if not safe[0].isalpha():
        safe = f"g{safe}"
    return f"{safe}_{index}"[:20]


def _coverage_gate_word(passed: bool | None) -> str:
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    return "n/a"


def _metric_pass(value: float | None, threshold: float | None) -> bool | None:
    if value is None or threshold is None:
        return None
    return value >= threshold


def _render_coverage_markdown(report: FeishuReportInput) -> str | None:
    if report.coverage_line_percent is None and report.coverage_branch_percent is None:
        return None

    lines = ["**Coverage:**"]
    if report.coverage_line_percent is not None:
        line_pass = _metric_pass(report.coverage_line_percent, report.coverage_line_threshold)
        threshold = f" (>={report.coverage_line_threshold:.0f}%)" if report.coverage_line_threshold is not None else ""
        lines.append(f"- Line: {report.coverage_line_percent:.1f}%{threshold} **{_coverage_gate_word(line_pass)}**")
    if report.coverage_branch_percent is not None:
        branch_pass = _metric_pass(report.coverage_branch_percent, report.coverage_branch_threshold)
        threshold = (
            f" (>={report.coverage_branch_threshold:.0f}%)" if report.coverage_branch_threshold is not None else ""
        )
        lines.append(
            f"- Branch: {report.coverage_branch_percent:.1f}%{threshold} **{_coverage_gate_word(branch_pass)}**"
        )
    if report.coverage_gate_passed is not None and (
        report.coverage_line_percent is not None and report.coverage_branch_percent is not None
    ):
        lines.append(f"- Gate: **{_coverage_gate_word(report.coverage_gate_passed)}**")
    return "\n".join(lines)


def _build_nightly_status(report: FeishuReportInput) -> str:
    if report.timed_out:
        return "Timed out (partial results sent)"
    listed, root_cause, follow_up, not_repro = _summary_counts(report)
    total_failures = report.failed + report.errors
    if report.overall_exit == 0 and total_failures == 0 and listed == 0:
        return "All passed"
    if listed > 0:
        return (
            f"Failed {listed} | {STATUS_ROOT_CAUSE_FOUND} {root_cause} | "
            f"{STATUS_NEEDS_FOLLOW_UP} {follow_up} | {STATUS_NOT_REPRODUCED} {not_repro}"
        )
    if report.infra_message:
        return f"Failed (infra: {report.infra_message})"
    if report.status_note:
        return report.status_note
    return f"Failed (pytest exit {report.overall_exit})"


def _status_card_template(report: FeishuReportInput) -> str:
    if report.timed_out or report.overall_exit != 0 or report.failure_blames:
        total_failures = report.failed + report.errors
        if report.overall_exit == 0 and total_failures == 0 and not report.failure_blames:
            return "green"
        return "red"
    return "green"


def _header_title(report: FeishuReportInput) -> str:
    listed, _, _, _ = _summary_counts(report)
    total_failures = max(listed, report.failed + report.errors)
    if report.timed_out:
        return f"Nightly TIMED OUT — {total_failures} tests"
    if _status_card_template(report) == "green":
        return "Nightly PASSED"
    if total_failures > 0:
        return f"Nightly FAILED — {total_failures} tests"
    return "Nightly FAILED"


def _render_exit_code_markdown(report: FeishuReportInput) -> str | None:
    """Standalone Exit code field (never glued to Legend / list tails)."""
    if report.overall_exit == 0:
        return None
    return f"**Exit code:** `{report.overall_exit}`"


def _render_summary_markdown(
    report: FeishuReportInput,
    *,
    include_pipeline_link: bool = False,
    include_exit_code: bool = True,
) -> str:
    """Build summary markdown. Pipeline URL is omitted by default (use button once)."""
    listed, root_cause, follow_up, not_repro = _summary_counts(report)
    lines = [
        f"**Branch:** `{report.branch}`",
        f"**Commit:** `{report.commit}`",
        f"**Result:** {_build_nightly_status(report)}",
        (
            f"**Counts:** Passed **{report.passed}** · Failed **{report.failed}** · "
            f"Errors **{report.errors}** · Duration **{_format_duration(report.duration_sec)}**"
        ),
    ]
    if listed > 0:
        lines.append(
            f"**Triage:** {STATUS_ROOT_CAUSE_FOUND} **{root_cause}** · "
            f"{STATUS_NEEDS_FOLLOW_UP} **{follow_up}** · {STATUS_NOT_REPRODUCED} **{not_repro}**"
        )
        lines.append("")
        lines.append("**Legend:**")
        lines.extend(_STATUS_LEGEND)

    if include_exit_code:
        exit_md = _render_exit_code_markdown(report)
        if exit_md:
            # Blank line so Feishu never glues Legend / list tails to Exit code.
            lines.append("")
            lines.append(exit_md)

    coverage = _render_coverage_markdown(report)
    if coverage:
        lines.append("")
        lines.append(coverage)

    if report.drift_warnings:
        lines.append("")
        lines.append(f"**Config drift ({len(report.drift_warnings)}):**")
        lines.extend(f"- {warning}" for warning in report.drift_warnings[:FEISHU_DRIFT_LIMIT])
        remaining = len(report.drift_warnings) - FEISHU_DRIFT_LIMIT
        if remaining > 0:
            lines.append(f"- ... and {remaining} more")

    if report.status_note and report.status_note != _build_nightly_status(report):
        lines.append("")
        lines.append(report.status_note)

    if include_pipeline_link and report.pipeline_log_url:
        lines.append("")
        lines.append(f"[Pipeline log]({report.pipeline_log_url})")
    return "\n".join(lines)


def _pipeline_button(url: str) -> dict[str, Any]:
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": "Open pipeline log"},
        "type": "primary",
        "width": "fill",
        "size": "large",
        "behaviors": [
            {
                "type": "open_url",
                "default_url": url,
                "pc_url": url,
                "ios_url": url,
                "android_url": url,
            }
        ],
    }


def _pipeline_action_elements(url: str) -> list[dict[str, Any]]:
    """One emphasized Pipeline log control with helper text (Nielsen visibility)."""
    return [
        {"tag": "hr"},
        {
            "tag": "markdown",
            "content": "**Next step:** Open the pipeline log for error details",
        },
        _pipeline_button(url),
    ]


def _collapsible_status_panel(
    status: str,
    blames: list[FailureBlame],
    *,
    index: int,
    expanded: bool = False,
) -> dict[str, Any]:
    body = "\n\n".join(_render_failure_line(b) for b in blames)
    border_color = _STATUS_BORDER.get(status, "grey")
    return {
        "tag": "collapsible_panel",
        "element_id": _element_id(status.replace(" ", "_").replace("-", "_"), index),
        "expanded": expanded,
        "header": {
            "title": {
                "tag": "markdown",
                "content": f"**{status}** — {len(blames)} failure{'s' if len(blames) != 1 else ''}",
            },
            "vertical_align": "center",
            "icon": {
                "tag": "standard_icon",
                "token": "down-small-ccm_outlined",  # nosec B105 — Feishu icon id, not a credential
                "size": "16px 16px",
            },
            "icon_position": "right",
            "icon_expanded_angle": -180,
        },
        "border": {"color": border_color, "corner_radius": "5px"},
        "vertical_spacing": "8px",
        "padding": "8px 8px 8px 8px",
        "elements": [{"tag": "markdown", "content": body}],
    }


def _card_shell(
    report: FeishuReportInput,
    elements: list[dict[str, Any]],
    *,
    title: str | None = None,
    subtitle: str | None = None,
) -> dict[str, Any]:
    header: dict[str, Any] = {
        "template": _status_card_template(report),
        "title": {"tag": "plain_text", "content": title or _header_title(report)},
    }
    if subtitle:
        header["subtitle"] = {"tag": "plain_text", "content": subtitle}
    return {
        "msg_type": "interactive",
        "card": {
            "schema": "2.0",
            "config": {
                "enable_forward": True,
                "update_multi": True,
                "width_mode": "fill",
                "summary": {"content": title or _header_title(report)},
            },
            "header": header,
            "body": {
                "direction": "vertical",
                "padding": "12px 12px 12px 12px",
                "vertical_spacing": "8px",
                "elements": elements,
            },
        },
    }


def _body_elements_for_status_groups(
    report: FeishuReportInput,
    status_groups: list[tuple[str, list[FailureBlame]]],
    *,
    expand_first: bool = True,
    part: int | None = None,
    total_parts: int | None = None,
) -> list[dict[str, Any]]:
    # Exit code is its own markdown element so Feishu never glues it to Legend.
    summary = _render_summary_markdown(report, include_pipeline_link=False, include_exit_code=False)
    if part is not None and total_parts is not None and total_parts > 1:
        summary = f"**Part {part}/{total_parts}**\n\n{summary}"
    elements: list[dict[str, Any]] = [{"tag": "markdown", "content": summary}]
    exit_md = _render_exit_code_markdown(report)
    if exit_md:
        elements.append({"tag": "markdown", "content": exit_md})
    if status_groups:
        elements.append({"tag": "hr"})
        for index, (status, blames) in enumerate(status_groups):
            elements.append(
                _collapsible_status_panel(
                    status,
                    blames,
                    index=index if part is None else part * 100 + index,
                    expanded=expand_first and index == 0,
                )
            )
    # Exactly one Pipeline log control: helper text + primary button (no duplicate link).
    if report.pipeline_log_url:
        elements.extend(_pipeline_action_elements(report.pipeline_log_url))
    return elements


def build_feishu_text_payload(report: FeishuReportInput) -> dict[str, Any]:
    """Build plain-text Feishu webhook payload (single message). Does not send."""
    lines = [_header_title(report), _render_summary_markdown(report, include_pipeline_link=True)]
    groups = _group_blames_by_status(report.failure_blames)
    if groups:
        lines.append("")
        lines.append(f"Failures ({sum(len(b) for _, b in groups)}):")
        for status, blames in groups:
            lines.append(f"[{status}]")
            for blame in blames:
                lines.append(_render_failure_line(blame).replace("**", "").replace("`", ""))
                lines.append("")
    return {
        "msg_type": "text",
        "content": {"text": "\n".join(lines).rstrip()},
    }


def build_feishu_card_payload(
    report: FeishuReportInput,
    *,
    part: int | None = None,
    total_parts: int | None = None,
    status_groups: list[tuple[str, list[FailureBlame]]] | None = None,
    subtitle: str | None = None,
) -> dict[str, Any]:
    """Build Feishu interactive card payload (schema 2.0). Does not send."""
    groups = status_groups
    if groups is None:
        groups = _group_blames_by_status(report.failure_blames)
    title = _header_title(report)
    if part is not None and total_parts is not None and total_parts > 1:
        title = f"{title} ({part}/{total_parts})"
    elements = _body_elements_for_status_groups(
        report,
        groups,
        expand_first=True,
        part=part,
        total_parts=total_parts,
    )
    return _card_shell(report, elements, title=title, subtitle=subtitle)


def _split_oversized_group(
    report: FeishuReportInput,
    status: str,
    blames: list[FailureBlame],
) -> list[tuple[str, list[FailureBlame]]]:
    """Split one status group into blame chunks that each fit the byte budget."""
    probe = build_feishu_card_payload(report, status_groups=[(status, blames)], part=1, total_parts=2)
    if _payload_byte_size(probe) <= _FEISHU_MESSAGE_BYTE_BUDGET or len(blames) <= 1:
        return [(status, blames)]

    chunks: list[tuple[str, list[FailureBlame]]] = []
    bucket: list[FailureBlame] = []
    for blame in blames:
        candidate = [*bucket, blame]
        probe = build_feishu_card_payload(
            report,
            status_groups=[(status, candidate)],
            part=1,
            total_parts=2,
        )
        if bucket and _payload_byte_size(probe) > _FEISHU_MESSAGE_BYTE_BUDGET:
            chunks.append((status, bucket))
            bucket = [blame]
        else:
            bucket = candidate
    if bucket:
        chunks.append((status, bucket))
    return chunks


def _pack_status_groups(
    report: FeishuReportInput,
    groups: list[tuple[str, list[FailureBlame]]],
) -> list[list[tuple[str, list[FailureBlame]]]]:
    """Greedy pack of status groups so each card stays under the byte budget."""
    if not groups:
        return [[]]
    flat: list[tuple[str, list[FailureBlame]]] = []
    for status, blames in groups:
        flat.extend(_split_oversized_group(report, status, blames))

    packed: list[list[tuple[str, list[FailureBlame]]]] = []
    bucket: list[tuple[str, list[FailureBlame]]] = []
    for status, blames in flat:
        candidate = [*bucket, (status, blames)]
        probe = build_feishu_card_payload(
            report,
            status_groups=candidate,
            part=1,
            total_parts=2,
        )
        if bucket and _payload_byte_size(probe) > _FEISHU_MESSAGE_BYTE_BUDGET:
            packed.append(bucket)
            bucket = [(status, blames)]
        else:
            bucket = candidate
    if bucket:
        packed.append(bucket)
    return packed


def build_feishu_payloads(
    report: FeishuReportInput,
    *,
    subtitle: str | None = None,
) -> list[dict[str, Any]]:
    """Build one or more schema-2.0 cards; split by status group only past ~18 KB."""
    groups = _group_blames_by_status(report.failure_blames)
    single = build_feishu_card_payload(report, status_groups=groups, subtitle=subtitle)
    if _payload_byte_size(single) <= _FEISHU_MESSAGE_BYTE_BUDGET:
        return [single]

    packed = _pack_status_groups(report, groups)
    total = len(packed)
    return [
        build_feishu_card_payload(
            report,
            status_groups=chunk,
            part=index if total > 1 else None,
            total_parts=total if total > 1 else None,
            subtitle=subtitle,
        )
        for index, chunk in enumerate(packed, start=1)
    ]


def build_feishu_payload(report: FeishuReportInput, *, subtitle: str | None = None) -> dict[str, Any]:
    """Build preferred Feishu payload (first card if split). Does not send."""
    return build_feishu_payloads(report, subtitle=subtitle)[0]


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
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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


def push_feishu_report(
    webhook_url: str,
    report: FeishuReportInput,
    *,
    subtitle: str | None = None,
) -> None:
    """Send all Feishu payloads for a report (split when needed)."""
    for payload in build_feishu_payloads(report, subtitle=subtitle):
        push_feishu(webhook_url, payload)
