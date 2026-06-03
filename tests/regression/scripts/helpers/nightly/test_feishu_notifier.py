"""Tests for nightly.feishu_notifier — build_feishu_payload, push_feishu."""

from __future__ import annotations

import io
import json
import logging

import pytest
from scripts.helpers.nightly.feishu_notifier import build_feishu_payload, push_feishu

# ---------------------------------------------------------------------------
# Payload builder kwargs
# ---------------------------------------------------------------------------


def _kwargs(**overrides):
    defaults = {
        "timestamp": "2026-01-15T08:30:00Z",
        "branch": "main",
        "commit": "abc1234",
        "passed": 42,
        "failed": 0,
        "duration_sec": 180.0,
        "coverage_line_percent": 85.0,
        "coverage_branch_percent": 70.0,
        "coverage_line_threshold": 70.0,
        "coverage_branch_threshold": 50.0,
        "coverage_gate_passed": True,
        "test_map_source_files": 12,
        "test_map_symbols": 56,
        "test_map_written": True,
        "failed_cases": (),
        "first_error": "",
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# build_feishu_payload — happy path
# ---------------------------------------------------------------------------


def test_payload_all_passed_shows_success_status() -> None:
    payload = build_feishu_payload(**_kwargs())
    assert payload["msg_type"] == "text"
    assert "All passed" in payload["content"]["text"]


def test_payload_includes_coverage_section_when_present() -> None:
    payload = build_feishu_payload(**_kwargs())
    assert "Coverage (PASS)" in payload["content"]["text"]


def test_payload_includes_test_map_updated_when_written() -> None:
    payload = build_feishu_payload(**_kwargs())
    assert "12 files / 56 symbols (updated)" in payload["content"]["text"]


# ---------------------------------------------------------------------------
# build_feishu_payload — failures
# ---------------------------------------------------------------------------


def test_payload_with_failures_shows_failed_count_and_cases() -> None:
    payload = build_feishu_payload(
        **_kwargs(
            failed=3,
            failed_cases=("test_a", "test_b", "test_c"),
            first_error="AssertionError: x != y",
        )
    )
    text = payload["content"]["text"]
    assert "3 failed" in text
    assert "Failed cases:" in text
    assert "test_a" in text
    assert "AssertionError" in text


def test_payload_coverage_below_threshold_shows_below_status() -> None:
    payload = build_feishu_payload(
        **_kwargs(
            coverage_line_percent=60.0,
            coverage_gate_passed=False,
        )
    )
    assert "BELOW THRESHOLD" in payload["content"]["text"]


def test_payload_no_coverage_omits_coverage_section() -> None:
    payload = build_feishu_payload(
        **_kwargs(
            coverage_line_percent=None,
            coverage_branch_percent=None,
            coverage_line_threshold=None,
            coverage_branch_threshold=None,
            coverage_gate_passed=None,
        )
    )
    assert "Coverage" not in payload["content"]["text"]


def test_payload_test_map_not_written_shows_not_updated() -> None:
    payload = build_feishu_payload(**_kwargs(test_map_written=False))
    assert "not updated" in payload["content"]["text"]


# ---------------------------------------------------------------------------
# build_feishu_payload — weak coverage symbols
# ---------------------------------------------------------------------------


def test_payload_includes_weak_coverage_symbols() -> None:
    payload = build_feishu_payload(**_kwargs(weak_coverage_symbols=("cli/main.py::run", "tensor_cast/ops.py::add")))
    text = payload["content"]["text"]
    assert "Weak coverage symbols" in text
    assert "cli/main.py::run" in text


def test_payload_omits_weak_coverage_when_empty() -> None:
    payload = build_feishu_payload(**_kwargs(weak_coverage_symbols=()))
    text = payload["content"]["text"]
    assert "Weak coverage symbols" not in text


def test_payload_truncates_weak_coverage_at_10() -> None:
    symbols = tuple(f"cli/main.py::fn_{i}" for i in range(15))
    payload = build_feishu_payload(**_kwargs(weak_coverage_symbols=symbols))
    text = payload["content"]["text"]
    assert "fn_0" in text
    assert "fn_9" in text
    assert "fn_10" not in text
    assert "and 5 more" in text


# ---------------------------------------------------------------------------
# build_feishu_payload — redundancy warnings
# ---------------------------------------------------------------------------


def test_payload_includes_over_covered_symbols() -> None:
    warning = {
        "type": "over_covered_symbol",
        "symbol": "cli/main.py::run",
        "test_count": 8,
        "threshold": 5,
    }
    payload = build_feishu_payload(**_kwargs(redundancy_warnings=(warning,)))
    text = payload["content"]["text"]
    assert "Over-covered symbols" in text
    assert "cli/main.py::run" in text
    assert "8 tests" in text


def test_payload_includes_redundant_pairs() -> None:
    warning = {
        "type": "redundant_pair",
        "test_a": "tests/a.py::test_1",
        "test_b": "tests/a.py::test_2",
        "jaccard": 0.92,
    }
    payload = build_feishu_payload(**_kwargs(redundancy_warnings=(warning,)))
    text = payload["content"]["text"]
    assert "Redundant test pairs" in text
    assert "tests/a.py::test_1" in text
    assert "0.92" in text


def test_payload_omits_redundancy_section_when_empty() -> None:
    payload = build_feishu_payload(**_kwargs(redundancy_warnings=()))
    text = payload["content"]["text"]
    assert "Over-covered" not in text
    assert "Redundant test pairs" not in text


# ---------------------------------------------------------------------------
# build_feishu_payload — phase breakdown
# ---------------------------------------------------------------------------


def test_payload_includes_phase_breakdown_section() -> None:
    phases = (
        {"label": "phase1 (test_map UT)", "passed": 40, "failed": 0, "duration_sec": 120.0},
        {"label": "phase2c (network)", "passed": 5, "failed": 2, "duration_sec": 30.0},
    )
    payload = build_feishu_payload(**_kwargs(phase_breakdown=phases))
    text = payload["content"]["text"]
    assert "Per-phase:" in text
    assert "phase1 (test_map UT): passed 40 / failed 0 / 120s" in text
    assert "phase2c (network): passed 5 / failed 2 / 30s" in text


def test_payload_omits_phase_breakdown_when_empty() -> None:
    payload = build_feishu_payload(**_kwargs(phase_breakdown=()))
    assert "Per-phase:" not in payload["content"]["text"]


# ---------------------------------------------------------------------------
# build_feishu_payload — slowest tests
# ---------------------------------------------------------------------------


def test_payload_includes_slowest_tests_section() -> None:
    slowest = (("tests/a.py::test_slow", 9.0), ("tests/b.py::test_mid", 3.0))
    payload = build_feishu_payload(**_kwargs(slowest_tests=slowest))
    text = payload["content"]["text"]
    assert "Slowest tests (top 2):" in text
    assert "9.0s tests/a.py::test_slow" in text


def test_payload_omits_slowest_tests_when_empty() -> None:
    payload = build_feishu_payload(**_kwargs(slowest_tests=()))
    assert "Slowest tests" not in payload["content"]["text"]


# ---------------------------------------------------------------------------
# build_feishu_payload — config drift warnings
# ---------------------------------------------------------------------------


def test_payload_includes_config_drift_section() -> None:
    warnings = ("deepseek-ai/DeepSeek-V3.1 [deepseekv3.1_remote] model_type: vendored='a' hub='b'",)
    payload = build_feishu_payload(**_kwargs(drift_warnings=warnings))
    text = payload["content"]["text"]
    assert "Config drift (1):" in text
    assert warnings[0] in text


def test_payload_omits_config_drift_when_empty() -> None:
    payload = build_feishu_payload(**_kwargs(drift_warnings=()))
    assert "Config drift" not in payload["content"]["text"]


def test_payload_truncates_config_drift_at_10() -> None:
    warnings = tuple(f"model-{i} drift" for i in range(15))
    payload = build_feishu_payload(**_kwargs(drift_warnings=warnings))
    text = payload["content"]["text"]
    assert "model-0 drift" in text
    assert "model-9 drift" in text
    assert "model-10 drift" not in text
    assert "and 5 more" in text


def test_payload_default_kwargs_omit_new_optional_sections() -> None:
    """Default (empty) phase_breakdown / slowest_tests / drift_warnings leave output unchanged."""
    baseline = build_feishu_payload(**_kwargs())
    explicit = build_feishu_payload(**_kwargs(phase_breakdown=(), slowest_tests=(), drift_warnings=()))
    assert baseline["content"]["text"] == explicit["content"]["text"]


# ---------------------------------------------------------------------------
# build_feishu_payload — truncation
# ---------------------------------------------------------------------------


def test_payload_truncates_failed_cases_at_20() -> None:
    cases = tuple(f"test_{i}" for i in range(25))
    payload = build_feishu_payload(**_kwargs(failed=25, failed_cases=cases))
    text = payload["content"]["text"]
    assert "test_0" in text
    assert "test_19" in text
    assert "test_20" not in text
    assert "and 5 more" in text


def test_payload_exactly_20_failed_cases_no_truncation_message() -> None:
    cases = tuple(f"test_{i}" for i in range(20))
    payload = build_feishu_payload(**_kwargs(failed=20, failed_cases=cases))
    text = payload["content"]["text"]
    assert "and more" not in text


# ---------------------------------------------------------------------------
# push_feishu
# ---------------------------------------------------------------------------


def test_push_feishu_sends_json_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_url = None
    captured_data = None

    def _fake_urlopen(req, timeout):
        nonlocal captured_url, captured_data
        captured_url = req.full_url
        captured_data = req.data
        return io.BytesIO(b'{"ok":true}')

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    payload = {"msg_type": "text", "content": {"text": "hello"}}
    push_feishu("https://example.com/hook", payload)

    assert captured_url == "https://example.com/hook"
    assert json.loads(captured_data) == payload


def test_push_feishu_logs_failure_on_os_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout: (_ for _ in ()).throw(OSError("connection refused")),
    )
    with caplog.at_level(logging.WARNING):
        push_feishu("https://example.com/hook", {"text": "x"})
    assert "Feishu push failed" in caplog.text


def test_push_feishu_warns_on_nonzero_code(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout: io.BytesIO(b'{"code":9499,"msg":"bad webhook"}'),
    )
    with caplog.at_level(logging.WARNING):
        push_feishu("https://example.com/hook", {"msg_type": "text", "content": {"text": "x"}})
    assert "Feishu push rejected" in caplog.text
    assert "9499" in caplog.text
