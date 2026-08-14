"""Tests for nightly.feishu_notifier — build_feishu_payload, push_feishu."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

from scripts.helpers.nightly.feishu_notifier import (
    _FEISHU_MESSAGE_BYTE_BUDGET,
    STATUS_NEEDS_FOLLOW_UP,
    STATUS_NOT_REPRODUCED,
    STATUS_ROOT_CAUSE_FOUND,
    _payload_byte_size,
    build_feishu_card_payload,
    build_feishu_payload,
    build_feishu_payloads,
    build_feishu_text_payload,
    push_feishu,
)
from scripts.helpers.nightly.report_models import AttributionConclusion, FailureBlame, FeishuReportInput


def _report(**overrides: object) -> FeishuReportInput:
    report = FeishuReportInput(
        timestamp="2026-01-15T08:30:00Z",
        branch="main",
        commit="abc1234",
        passed=42,
        failed=0,
        errors=0,
        duration_sec=180.0,
        overall_exit=0,
        coverage_line_percent=85.0,
        coverage_branch_percent=70.0,
        coverage_line_threshold=70.0,
        coverage_branch_threshold=50.0,
        coverage_gate_passed=True,
        pipeline_log_url="https://ci.example/log/1",
    )
    if not overrides:
        return report
    fields = {field.name: getattr(report, field.name) for field in report.__dataclass_fields__.values()}
    fields.update(overrides)
    return FeishuReportInput(**fields)


def _card_markdown(payload: dict[str, Any]) -> str:
    elements = payload["card"]["body"]["elements"]
    parts: list[str] = []
    for element in elements:
        if element.get("tag") == "markdown":
            parts.append(element.get("content", ""))
        elif element.get("tag") == "collapsible_panel":
            parts.extend(
                child.get("content", "") for child in element.get("elements", []) if child.get("tag") == "markdown"
            )
            header = element.get("header", {}).get("title", {})
            parts.append(header.get("content", ""))
    return "\n".join(parts)


def _pipeline_button_count(payload: dict[str, Any]) -> int:
    count = 0
    for element in payload["card"]["body"]["elements"]:
        if element.get("tag") != "button":
            continue
        label = element.get("text", {}).get("content", "")
        if label in {"Pipeline log", "Open pipeline log"}:
            count += 1
    return count


def _markdown_element_contents(payload: dict[str, Any]) -> list[str]:
    return [
        element.get("content", "")
        for element in payload["card"]["body"]["elements"]
        if element.get("tag") == "markdown"
    ]


def _status_panel_borders(payload: dict[str, Any]) -> dict[str, str]:
    borders: dict[str, str] = {}
    for element in payload["card"]["body"]["elements"]:
        if element.get("tag") != "collapsible_panel":
            continue
        title = element.get("header", {}).get("title", {}).get("content", "")
        color = element.get("border", {}).get("color", "")
        borders[title] = color
    return borders


def test_card_payload_all_passed_shows_green_header() -> None:
    payload = build_feishu_card_payload(_report())
    assert payload["msg_type"] == "interactive"
    assert payload["card"]["schema"] == "2.0"
    assert payload["card"]["header"]["template"] == "green"
    assert "All passed" in _card_markdown(payload)


def test_card_payload_includes_line_and_branch_coverage() -> None:
    payload = build_feishu_card_payload(_report())
    text = _card_markdown(payload)
    assert "Line: 85.0% (>=70%)" in text
    assert "Branch: 70.0% (>=50%)" in text
    assert "**PASS**" in text


def test_card_payload_pipeline_log_once_as_button() -> None:
    payload = build_feishu_card_payload(_report())
    text = _card_markdown(payload)
    assert _pipeline_button_count(payload) == 1
    assert "[Pipeline log]" not in text
    assert "Open the pipeline log for error details" in text
    assert text.count("https://ci.example/log/1") == 0
    assert "pull/" not in text
    assert "PR" not in text
    button = next(element for element in payload["card"]["body"]["elements"] if element.get("tag") == "button")
    assert button["type"] == "primary"
    assert button["text"]["content"] == "Open pipeline log"


def test_card_payload_exit_code_not_glued_to_legend() -> None:
    blames = (
        FailureBlame(
            node_id="tests/smoke/test_a.py::test_x",
            commit_id="deadbeef",
            author="alice",
            subject="add test",
            conclusion=AttributionConclusion.FIRST_BAD,
            last_reason="AssertionError: boom",
        ),
    )
    payload = build_feishu_card_payload(_report(failed=1, errors=0, overall_exit=3, failure_blames=blames))
    text = _card_markdown(payload)
    assert "at HEADExit code" not in text.replace("\n", "")
    assert "**Exit code:** `3`" in text
    # Exit code is its own markdown element (not the same block as Legend).
    blocks = _markdown_element_contents(payload)
    legend_blocks = [block for block in blocks if "Legend:" in block]
    exit_blocks = [block for block in blocks if block.startswith("**Exit code:**")]
    assert legend_blocks
    assert exit_blocks
    assert all("**Exit code:**" not in block for block in legend_blocks)
    # Text payload also keeps a blank line before Exit code.
    text_payload = build_feishu_text_payload(_report(failed=1, errors=0, overall_exit=3, failure_blames=blames))
    text_body = text_payload["content"]["text"]
    assert "at HEAD\n\n**Exit code:**" in text_body or "\n\n**Exit code:** `3`" in text_body
    assert "HEADExit code" not in text_body.replace("\n", "")


def test_card_payload_groups_by_status_with_node_ids() -> None:
    blames = (
        FailureBlame(
            node_id="tests/smoke/test_a.py::test_x",
            commit_id="deadbeef",
            author="alice",
            subject="add test",
            conclusion=AttributionConclusion.FIRST_BAD,
            last_reason="AssertionError: boom",
        ),
        FailureBlame(
            node_id="tests/smoke/test_b.py::test_y",
            commit_id="unknown",
            author="unknown",
            subject="flaky",
            conclusion=AttributionConclusion.CANNOT_REPRODUCE,
            last_reason="",
        ),
        FailureBlame(
            node_id="tests/regression/test_c.py::test_z",
            commit_id="unknown",
            author="unknown",
            subject="stale failure",
            conclusion=AttributionConclusion.NEED_HUMAN,
            last_reason="ValueError: x",
        ),
    )
    payload = build_feishu_card_payload(
        _report(
            failed=3,
            errors=0,
            overall_exit=3,
            failure_blames=blames,
        )
    )
    assert payload["card"]["header"]["template"] == "red"
    text = _card_markdown(payload)
    # Group by status, not directory.
    assert f"**{STATUS_ROOT_CAUSE_FOUND}**" in text
    assert f"**{STATUS_NEEDS_FOLLOW_UP}**" in text
    assert f"**{STATUS_NOT_REPRODUCED}**" in text
    assert "**tests/smoke**" not in text
    assert "**tests/regression**" not in text
    # Full node ids; no Error / jargon primary labels.
    assert "`tests/smoke/test_a.py::test_x`" in text
    assert "`tests/smoke/test_b.py::test_y`" in text
    assert "`tests/regression/test_c.py::test_z`" in text
    assert "Introduced by: `deadbeef` (alice)" in text
    assert "**Error:**" not in text
    assert "first-bad" not in text
    assert "need-human" not in text
    assert "cannot-reproduce" not in text
    assert "Legend:" in text
    assert STATUS_NEEDS_FOLLOW_UP in text
    borders = _status_panel_borders(payload)
    assert any(color == "orange" for title, color in borders.items() if STATUS_NEEDS_FOLLOW_UP in title)
    assert any(color == "grey" for title, color in borders.items() if STATUS_NOT_REPRODUCED in title)
    assert "ATTRIBUTION HARD FAIL" not in text
    assert "pull/" not in text
    # English only.
    assert "失败" not in text
    assert "全部通过" not in text
    assert "覆盖率" not in text


def test_card_payload_lists_all_failures_no_20_cap() -> None:
    blames = tuple(
        FailureBlame(
            node_id=f"tests/a.py::test_{index}",
            commit_id="abc",
            author="a",
            subject="s",
            conclusion=AttributionConclusion.FIRST_BAD,
            last_reason=f"err-{index}",
        )
        for index in range(25)
    )
    payloads = build_feishu_payloads(_report(failure_blames=blames, failed=25, overall_exit=1))
    joined = "\n".join(_card_markdown(p) for p in payloads)
    assert "`tests/a.py::test_0`" in joined
    assert "`tests/a.py::test_24`" in joined
    assert "... and" not in joined


def test_text_payload_coverage_below_threshold() -> None:
    payload = build_feishu_text_payload(
        _report(
            coverage_line_percent=60.0,
            coverage_gate_passed=False,
        )
    )
    text = payload["content"]["text"]
    assert "Line: 60.0%" in text
    assert "FAIL" in text


def test_text_payload_no_coverage_omits_coverage_section() -> None:
    payload = build_feishu_text_payload(
        _report(
            coverage_line_percent=None,
            coverage_branch_percent=None,
            coverage_line_threshold=None,
            coverage_branch_threshold=None,
            coverage_gate_passed=None,
        )
    )
    assert "Coverage:" not in payload["content"]["text"]


def test_text_payload_includes_config_drift_section() -> None:
    warnings = ("deepseek-ai/DeepSeek-V3.1 [deepseekv3.1_remote] model_type: vendored='a' hub='b'",)
    payload = build_feishu_text_payload(_report(drift_warnings=warnings))
    text = payload["content"]["text"]
    assert "Config drift (1):" in text
    assert warnings[0] in text


def test_build_feishu_payload_defaults_to_card() -> None:
    payload = build_feishu_payload(_report())
    assert payload["msg_type"] == "interactive"
    assert payload["card"]["schema"] == "2.0"


def test_design_preview_subtitle() -> None:
    payload = build_feishu_card_payload(_report(), subtitle="DESIGN PREVIEW")
    assert payload["card"]["header"]["subtitle"]["content"] == "DESIGN PREVIEW"


def test_payloads_stay_under_byte_budget_when_split() -> None:
    # Many long reasons under one status force a multi-card split under 18 KB.
    blames = tuple(
        FailureBlame(
            node_id=f"tests/dir_{index // 3}/test_{index}.py::test_case_{index}",
            commit_id=f"c{index:08x}",
            author=f"author_{index}",
            subject="subject",
            conclusion=(
                AttributionConclusion.FIRST_BAD
                if index % 3 == 0
                else AttributionConclusion.NEED_HUMAN
                if index % 3 == 1
                else AttributionConclusion.CANNOT_REPRODUCE
            ),
            last_reason=("timeout waiting for metrics export " * 8) + f"#{index}",
        )
        for index in range(60)
    )
    payloads = build_feishu_payloads(
        _report(failure_blames=blames, failed=60, overall_exit=1, pipeline_log_url="https://ci.example/log/big")
    )
    assert len(payloads) >= 1
    for payload in payloads:
        assert _payload_byte_size(payload) <= _FEISHU_MESSAGE_BYTE_BUDGET
        assert _pipeline_button_count(payload) == 1


def test_push_feishu_posts_json(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Resp:
        def read(self) -> bytes:
            return b'{"code":0,"msg":"ok"}'

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    captured: dict[str, Any] = {}

    def _urlopen(req: object, timeout: float = 0) -> _Resp:
        captured["timeout"] = timeout
        captured["data"] = json.loads(req.data.decode())  # type: ignore[attr-defined]
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    with caplog.at_level(logging.INFO):
        push_feishu("https://example/hook", {"msg_type": "text", "content": {"text": "hi"}})
    assert captured["timeout"] == 10
    assert captured["data"]["msg_type"] == "text"


def test_push_feishu_swallows_oserror(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    with caplog.at_level(logging.WARNING):
        push_feishu("https://example/hook", {"msg_type": "text", "content": {"text": "hi"}})
    assert "Feishu push failed" in caplog.text
