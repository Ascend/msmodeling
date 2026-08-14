"""Tests for nightly.main — command builders, emit_report, exit 3, timeout."""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

import pytest

from scripts.helpers._config import Config, ConfigError
from scripts.helpers.nightly.failure_attribution import ATTRIBUTION_BUDGET_SKIP_LABEL, FirstBadResult
from scripts.helpers.nightly.main import (
    ATTRIBUTION_HARD_FAIL_EXIT_CODE,
    PYTEST_TIMEOUT_EXIT_CODE,
    _build_pytest_cmd_wave_a,
    _build_pytest_cmd_wave_b,
    _build_terminal_summary,
    _combine_pytest_exits,
    _coverage_summary,
    _fetch_hub_config,
    _load_vendored_config,
    _resolve_exit_code,
    _run_config_drift_check,
    _run_nightly_pipeline,
    _run_pytest_waves,
    _stream_pytest,
    _terminate_process_tree,
    emit_init_failure_report,
    emit_report,
    resolve_nightly_timeout_seconds,
)
from scripts.helpers.nightly.pytest_parser import NightlyRunStats, merge_nightly_run_stats, parse_pytest_stdout
from scripts.helpers.nightly.report_models import AttributionConclusion, CoverageSummary, FailureBlame
from tests.helpers.fake_subprocess import FakeCompleted, FakePopen, FakePopenTimeoutOnFirstWait

_BASE_CFG = Config(
    base_branch="master",
    line_threshold=80.0,
    branch_threshold=60.0,
    benchmark_parallel=False,
    feishu_webhook_url="",
    msmodeling_cache=".msmodeling_cache",
    weights_prune=True,
)

_SAMPLE_STDOUT = """\
tests/smoke/test_a.py::test_ok PASSED                                          [ 50%]
tests/smoke/test_b.py::test_fail FAILED                                        [100%]
FAILED tests/smoke/test_b.py::test_fail - AssertionError: boom

========================= 1 failed, 1 passed in 2.50s =========================
"""


def test_pytest_cmd_wave_a_targets_non_benchmark_non_network_with_xdist_coverage_and_tb_line() -> None:
    cmd = _build_pytest_cmd_wave_a("python3")
    assert cmd[0] == "python3"
    assert "tests/" in cmd
    marker = " ".join(cmd)
    assert "not npu" in marker
    assert "not benchmark" in marker
    assert "not network" in marker
    assert "-n" in cmd
    assert "--cov-branch" in cmd
    assert "--cov-append" not in cmd
    assert "-vv" in cmd
    assert "--tb=line" in cmd
    assert "--junit-xml" not in marker


def test_pytest_cmd_wave_b_runs_benchmark_or_network_serial_without_xdist() -> None:
    cmd = _build_pytest_cmd_wave_b("python3")
    marker = " ".join(cmd)
    assert "not npu" in marker
    assert "benchmark" in marker
    assert "network" in marker
    assert "not benchmark" not in marker
    assert "not network" not in marker
    assert "-n" not in cmd
    assert "--cov-append" not in cmd
    assert "-vv" in cmd
    assert "--tb=line" in cmd


def test_merge_nightly_run_stats_sums_counts_and_dedupes_failures() -> None:
    wave_a = parse_pytest_stdout(
        "tests/a.py::test_ok PASSED\n========================= 1 passed in 1.00s =========================\n",
        exit_code=0,
    )
    wave_b = parse_pytest_stdout(_SAMPLE_STDOUT, exit_code=1)
    merged = merge_nightly_run_stats(wave_a, wave_b)
    assert merged.passed == 2
    assert merged.failed == 1
    assert merged.failed_cases == ("tests/smoke/test_b.py::test_fail",)
    assert merged.duration_sec == pytest.approx(3.5)


def test_run_pytest_waves_runs_both_markers_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.helpers.nightly import main as nightly_main

    monkeypatch.setattr(nightly_main, "REPO_ROOT", tmp_path)
    calls: list[tuple[str, str]] = []

    def _fake_stream(
        cmd: list[str],
        cwd: object,
        *,
        deadline: float | None = None,
        timeout_seconds: float | None = None,
        env_extra: dict[str, str] | None = None,
        log_prefix: str | None = None,
    ) -> tuple[int, str]:
        del cwd, deadline, timeout_seconds
        marker = " ".join(cmd)
        calls.append((log_prefix or "", marker))
        stdout = "========================= 1 passed in 1.00s =========================\n"
        return 0, stdout

    monkeypatch.setattr(nightly_main, "_stream_pytest", _fake_stream)
    monkeypatch.setattr(nightly_main, "_combine_wave_coverage", lambda _logger: None)
    exit_a, out_a, exit_b, out_b = _run_pytest_waves(
        logging.getLogger("nightly-test"),
        python_exe="python3",
        deadline=time.monotonic() + 60,
    )
    assert exit_a == 0
    assert exit_b == 0
    assert out_a
    assert out_b
    assert len(calls) == 2
    markers = [marker for _label, marker in calls]
    assert any("not benchmark" in marker and "not network" in marker for marker in markers)
    assert any("benchmark" in marker and "network" in marker and "not benchmark" not in marker for marker in markers)
    assert {label for label, _marker in calls} == {"non-benchmark", "benchmark"}


def test_combine_pytest_exits_prefers_timeout_then_first_failure() -> None:
    from scripts.helpers.nightly.main import PYTEST_TIMEOUT_EXIT_CODE

    assert _combine_pytest_exits(0, 0) == 0
    assert _combine_pytest_exits(1, 0) == 1
    assert _combine_pytest_exits(0, 1) == 1
    assert _combine_pytest_exits(1, 2) == 1
    assert _combine_pytest_exits(0, PYTEST_TIMEOUT_EXIT_CODE) == PYTEST_TIMEOUT_EXIT_CODE


def test_resolve_nightly_timeout_seconds_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MSMODELING_NIGHTLY_TIMEOUT_SECONDS", raising=False)
    assert resolve_nightly_timeout_seconds() == 3000.0
    monkeypatch.setenv("MSMODELING_NIGHTLY_TIMEOUT_SECONDS", "120")
    assert resolve_nightly_timeout_seconds() == 120.0


def test_resolve_exit_code_need_human_is_3_cannot_reproduce_keeps_pytest() -> None:
    need_human = (
        FailureBlame(
            node_id="tests/a.py::test_x",
            commit_id="unknown",
            author="unknown",
            subject="Still failing after 7-day lookback; needs human follow-up",
            conclusion=AttributionConclusion.NEED_HUMAN,
        ),
    )
    cannot = (
        FailureBlame(
            node_id="tests/a.py::test_x",
            commit_id="abc",
            author="a",
            subject="需关注/可能偶发",
            conclusion=AttributionConclusion.CANNOT_REPRODUCE,
        ),
    )
    attributed = (
        FailureBlame(
            node_id="tests/a.py::test_x",
            commit_id="abc",
            author="a",
            subject="s",
            conclusion=AttributionConclusion.FIRST_BAD,
        ),
    )
    assert _resolve_exit_code(1, need_human, timed_out=False) == ATTRIBUTION_HARD_FAIL_EXIT_CODE
    assert _resolve_exit_code(1, cannot, timed_out=False) == 1
    assert _resolve_exit_code(1, attributed + cannot, timed_out=False) == 1
    assert _resolve_exit_code(1, attributed, timed_out=True) == PYTEST_TIMEOUT_EXIT_CODE


def test_load_vendored_config_reads_repo_fixture() -> None:
    config = _load_vendored_config("deepseekv3.1_remote")
    assert config is not None
    assert isinstance(config, dict)
    assert "model_type" in config


def test_load_vendored_config_missing_fixture_returns_none() -> None:
    assert _load_vendored_config("__no_such_fixture__") is None


def test_fetch_hub_config_returns_config_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeConfig:
        def to_dict(self) -> dict[str, str]:
            return {"model_type": "fake"}

    def _from_pretrained(_model_id: str, **_kwargs: object) -> _FakeConfig:
        return _FakeConfig()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=_from_pretrained)),
    )

    hub = _fetch_hub_config("org/model")
    assert hub == {"model_type": "fake"}


def test_fetch_hub_config_retries_with_trust_remote_code_on_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeConfig:
        def to_dict(self) -> dict[str, str]:
            return {"model_type": "remote"}

    calls: list[dict[str, object]] = []

    def _from_pretrained(_model_id: str, **kwargs: object) -> _FakeConfig:
        calls.append(dict(kwargs))
        if "trust_remote_code" not in kwargs:
            raise ValueError("requires trust_remote_code")
        return _FakeConfig()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=_from_pretrained)),
    )

    hub = _fetch_hub_config("org/remote-model")
    assert hub == {"model_type": "remote"}
    assert len(calls) == 2
    assert calls[0] == {}
    assert calls[1] == {"trust_remote_code": True}


def test_coverage_summary_no_data_file_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts.helpers.nightly import main as nightly_main

    monkeypatch.setattr(nightly_main, "REPO_ROOT", tmp_path)
    result = _coverage_summary(_BASE_CFG)
    assert result is None


def test_coverage_summary_above_threshold_marks_passed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts.helpers.nightly import main as nightly_main

    (tmp_path / ".coverage").write_text("", encoding="utf-8")
    monkeypatch.setattr(nightly_main, "REPO_ROOT", tmp_path)

    totals_json = json.dumps(
        {
            "totals": {
                "percent_covered_display": "85.0%",
                "num_branches": 10,
                "covered_branches": 8,
            },
        }
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, totals_json, ""))
    result = _coverage_summary(_BASE_CFG)
    assert result is not None
    assert result.line_percent == 85.0
    assert result.branch_percent == 80.0
    assert result.gate_passed is True


def test_emit_report_skips_feishu_without_webhook(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="nightly"):
        emit_report(
            _SAMPLE_STDOUT,
            overall_exit=1,
            coverage=None,
            webhook_url=None,
        )
    assert "FEISHU_WEBHOOK_URL not set" in caplog.text


def test_emit_report_feishu_payload_includes_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    pushed: list[tuple[str, dict[str, Any]]] = []

    def _fake_push(url: str, payload: dict[str, Any]) -> None:
        pushed.append((url, payload))

    monkeypatch.setattr(
        "scripts.helpers.nightly.main.push_feishu_report", lambda url, report: pushed.append((url, {"report": report}))
    )
    # Also stub the lower-level path used inside push_feishu_report in case of import style
    monkeypatch.setattr("scripts.helpers.nightly.feishu_notifier.push_feishu", _fake_push)
    cov = CoverageSummary(
        line_percent=80.0,
        branch_percent=60.0,
        line_threshold=80.0,
        branch_threshold=60.0,
        gate_passed=True,
        message="passed",
    )
    emit_report(
        _SAMPLE_STDOUT,
        overall_exit=0,
        coverage=cov,
        webhook_url="https://example.com/hook",
        pipeline_log_url="https://ci.example/log/1",
    )
    assert len(pushed) == 1


def test_emit_report_includes_failure_blame_in_feishu(monkeypatch: pytest.MonkeyPatch) -> None:
    reports: list[Any] = []
    monkeypatch.setattr(
        "scripts.helpers.nightly.main.push_feishu_report",
        lambda url, report: reports.append(report),
    )
    blames = (
        FailureBlame(
            node_id="tests/a.py::test_x",
            commit_id="abc1234",
            author="alice",
            subject="add test",
            conclusion=AttributionConclusion.FIRST_BAD,
            last_reason="AssertionError: x",
        ),
    )
    emit_report(
        _SAMPLE_STDOUT,
        overall_exit=1,
        coverage=None,
        webhook_url="https://example.com/hook",
        failure_blames=blames,
    )
    assert reports[0].failure_blames[0].last_reason == "AssertionError: x"


def test_emit_init_failure_report_sends_feishu(monkeypatch: pytest.MonkeyPatch) -> None:
    reports: list[Any] = []
    monkeypatch.setattr(
        "scripts.helpers.nightly.main.push_feishu_report",
        lambda url, report: reports.append(report),
    )
    emit_init_failure_report(
        webhook_url="https://example.com/hook",
        pipeline_log_url="https://ci.example/log/1",
        error=RuntimeError("boom"),
    )
    assert "init failed" in reports[0].status_note
    assert reports[0].pipeline_log_url == "https://ci.example/log/1"


def test_drift_check_reports_key_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.helpers.nightly import main as nightly_main

    monkeypatch.setattr(nightly_main, "_DRIFT_FIXTURE_MAP", {"some/Model": "some_fixture"})
    monkeypatch.setattr(
        nightly_main,
        "_load_vendored_config",
        lambda _fixture: {"model_type": "deepseek_v3"},
    )
    monkeypatch.setattr(
        nightly_main,
        "_fetch_hub_config",
        lambda _model_id: {"model_type": "deepseek_v4"},
    )

    warnings = _run_config_drift_check()
    assert any("model_type: vendored='deepseek_v3' hub='deepseek_v4'" in w for w in warnings)


def test_stream_pytest_returns_captured_stdout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Stdout:
        def __iter__(self) -> Iterator[str]:
            yield "line\n"

        def close(self) -> None:
            return None

    fake_proc = FakePopen(stdout=_Stdout())
    fake_proc._returncode = 0
    monkeypatch.setattr("scripts.helpers.nightly.main.subprocess.Popen", lambda *_a, **_k: fake_proc)
    exit_code, captured = _stream_pytest(["python3", "-m", "pytest"], cwd=tmp_path, timeout_seconds=30)
    assert exit_code == 0
    assert captured == "line\n"


def test_stream_pytest_timeout_kills_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _BlockingStdout:
        def __iter__(self) -> Iterator[str]:
            while True:
                yield "still running\n"

        def close(self) -> None:
            return None

    fake_proc = FakePopen(stdout=_BlockingStdout())
    terminated: list[int] = []

    def _fake_terminate(proc: object) -> None:
        terminated.append(getattr(proc, "pid", -1))
        fake_proc._returncode = -15

    monkeypatch.setattr("scripts.helpers.nightly.main.subprocess.Popen", lambda *_a, **_k: fake_proc)
    monkeypatch.setattr("scripts.helpers.nightly.main._terminate_process_tree", _fake_terminate)

    exit_code, captured = _stream_pytest(["python3", "-m", "pytest"], cwd=tmp_path, timeout_seconds=0.05)
    assert exit_code == PYTEST_TIMEOUT_EXIT_CODE
    assert terminated == [fake_proc.pid]
    assert "timeout" in captured


def test_terminate_process_tree_escalates_to_sigkill_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = FakePopenTimeoutOnFirstWait()
    killpg_calls: list[tuple[int, int]] = []

    def _fake_killpg(pgid: int, sig: int) -> None:
        killpg_calls.append((pgid, sig))
        if sig == signal.SIGKILL:
            fake_proc._returncode = -signal.SIGKILL

    monkeypatch.setattr("scripts.helpers.nightly.main.os.getpgid", lambda _pid: fake_proc.pid)
    monkeypatch.setattr("scripts.helpers.nightly.main.os.killpg", _fake_killpg)

    _terminate_process_tree(fake_proc, sigterm_timeout_seconds=0.01)

    assert killpg_calls == [
        (fake_proc.pid, signal.SIGTERM),
        (fake_proc.pid, signal.SIGKILL),
    ]


def test_build_terminal_summary_matches_feishu_fields() -> None:
    coverage = CoverageSummary(
        line_percent=80.0,
        branch_percent=60.0,
        line_threshold=80.0,
        branch_threshold=60.0,
        gate_passed=True,
        message="Coverage gate passed: line=80.0% branch=60.0%",
    )
    stats = NightlyRunStats(
        passed=3,
        failed=1,
        errors=0,
        duration_sec=30.0,
        failed_cases=("tests/a.py::test_x",),
        first_error="",
        failure_reasons={"tests/a.py::test_x": "AssertionError: x"},
    )
    blames = (
        FailureBlame(
            node_id="tests/a.py::test_x",
            commit_id="deadbeef",
            author="bob",
            subject="fix test",
            conclusion=AttributionConclusion.FIRST_BAD,
            last_reason="AssertionError: x",
        ),
    )
    lines = _build_terminal_summary(
        pytest_exit=1,
        stats=stats,
        failure_blames=blames,
        coverage=coverage,
        drift_warnings=("some/Model [fixture] model_type: vendored='a' hub='b'",),
    )

    assert "Nightly exit=1: passed=3 failed=1 errors=0 duration=30s" in lines
    assert coverage.message in lines
    assert "status=Root cause found" in lines[2]
    assert "error=AssertionError: x" in lines[2]
    assert "Config drift: some/Model" in lines[-1]


def test_pipeline_returns_3_on_need_human_and_sends_feishu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.helpers.nightly import main as nightly_main

    monkeypatch.setattr(nightly_main, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        nightly_main,
        "_run_pytest_waves",
        lambda *a, **k: (1, _SAMPLE_STDOUT, 0, ""),
    )
    monkeypatch.setattr(nightly_main, "_coverage_summary", lambda _cfg: None)
    monkeypatch.setattr(nightly_main, "_run_config_drift_check", lambda: ())

    reports: list[Any] = []
    monkeypatch.setattr(nightly_main, "push_feishu_report", lambda url, report: reports.append(report))

    def _attrib(*_a: object, **_k: object) -> tuple[FirstBadResult, ...]:
        return (
            FirstBadResult(
                node_id="tests/smoke/test_b.py::test_fail",
                commit_id="unknown",
                author="unknown",
                subject="Still failing after 7-day lookback; needs human follow-up",
                conclusion=AttributionConclusion.NEED_HUMAN,
                detail="Still failing after 7-day lookback; needs human follow-up",
            ),
        )

    monkeypatch.setattr(nightly_main, "attribute_failures", _attrib)

    cfg = Config(
        base_branch="master",
        line_threshold=80.0,
        branch_threshold=60.0,
        feishu_webhook_url="https://example.com/hook",
    )
    logger = logging.getLogger("nightly-test")
    exit_code = _run_nightly_pipeline(
        logger,
        cfg,
        "https://example.com/hook",
        "",
        timeout_seconds=3000,
    )
    assert exit_code == ATTRIBUTION_HARD_FAIL_EXIT_CODE
    assert len(reports) == 1
    assert reports[0].failure_blames[0].needs_human


def test_pipeline_cannot_reproduce_does_not_force_exit_3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.helpers.nightly import main as nightly_main

    monkeypatch.setattr(nightly_main, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        nightly_main,
        "_run_pytest_waves",
        lambda *a, **k: (1, _SAMPLE_STDOUT, 0, ""),
    )
    monkeypatch.setattr(nightly_main, "_coverage_summary", lambda _cfg: None)
    monkeypatch.setattr(nightly_main, "_run_config_drift_check", lambda: ())
    monkeypatch.setattr(nightly_main, "push_feishu_report", lambda *_a, **_k: None)
    monkeypatch.setattr(
        nightly_main,
        "attribute_failures",
        lambda *_a, **_k: (
            FirstBadResult(
                node_id="tests/smoke/test_b.py::test_fail",
                commit_id="abc",
                author="a",
                subject="Flaky / not reproduced at HEAD",
                conclusion=AttributionConclusion.CANNOT_REPRODUCE,
            ),
        ),
    )
    cfg = Config(base_branch="master", line_threshold=80.0, branch_threshold=60.0)
    exit_code = _run_nightly_pipeline(
        logging.getLogger("nightly-test"),
        cfg,
        None,
        "",
        timeout_seconds=3000,
    )
    assert exit_code == 1


def test_pipeline_attribution_budget_skip_times_out_and_sends_feishu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.helpers.nightly import main as nightly_main

    monkeypatch.setattr(nightly_main, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        nightly_main,
        "_run_pytest_waves",
        lambda *a, **k: (1, _SAMPLE_STDOUT, 0, ""),
    )
    monkeypatch.setattr(nightly_main, "_coverage_summary", lambda _cfg: None)
    monkeypatch.setattr(nightly_main, "_run_config_drift_check", lambda: ())

    reports: list[Any] = []
    monkeypatch.setattr(nightly_main, "push_feishu_report", lambda url, report: reports.append(report))

    def _attrib(*_a: object, **kwargs: object) -> tuple[FirstBadResult, ...]:
        assert "deadline" in kwargs
        return (
            FirstBadResult(
                node_id="tests/smoke/test_b.py::test_fail",
                commit_id="unknown",
                author="unknown",
                subject=ATTRIBUTION_BUDGET_SKIP_LABEL,
                conclusion=AttributionConclusion.NEED_HUMAN,
                detail=ATTRIBUTION_BUDGET_SKIP_LABEL,
            ),
        )

    monkeypatch.setattr(nightly_main, "attribute_failures", _attrib)
    cfg = Config(
        base_branch="master",
        line_threshold=80.0,
        branch_threshold=60.0,
        feishu_webhook_url="https://example.com/hook",
    )
    exit_code = _run_nightly_pipeline(
        logging.getLogger("nightly-test"),
        cfg,
        "https://example.com/hook",
        "",
        timeout_seconds=3000,
    )
    assert exit_code == PYTEST_TIMEOUT_EXIT_CODE
    assert len(reports) == 1
    assert reports[0].timed_out is True
    assert reports[0].failure_blames[0].subject == ATTRIBUTION_BUDGET_SKIP_LABEL


def test_main_config_error_still_sends_feishu_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.helpers.nightly import main as nightly_main

    monkeypatch.setenv("FEISHU_WEBHOOK_URL", "https://example.com/hook")
    monkeypatch.setattr(
        nightly_main.Config,
        "from_env",
        classmethod(lambda cls: (_ for _ in ()).throw(ConfigError("bad threshold"))),
    )
    reports: list[Any] = []
    monkeypatch.setattr(
        nightly_main,
        "push_feishu_report",
        lambda url, report: reports.append((url, report)),
    )

    assert nightly_main.main() == 1
    assert len(reports) == 1
    assert reports[0][0] == "https://example.com/hook"
    assert "init failed" in reports[0][1].status_note


def test_pipeline_skips_drift_check_after_pytest_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from scripts.helpers.nightly import main as nightly_main

    monkeypatch.setattr(nightly_main, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        nightly_main,
        "_run_pytest_waves",
        lambda *a, **k: (PYTEST_TIMEOUT_EXIT_CODE, _SAMPLE_STDOUT, 0, ""),
    )
    monkeypatch.setattr(nightly_main, "_coverage_summary", lambda _cfg: None)

    drift_calls = {"n": 0}

    def _drift() -> tuple[str, ...]:
        drift_calls["n"] += 1
        return ("should-not-run",)

    monkeypatch.setattr(nightly_main, "_run_config_drift_check", _drift)
    reports: list[Any] = []
    monkeypatch.setattr(nightly_main, "push_feishu_report", lambda url, report: reports.append(report))

    cfg = Config(
        base_branch="master",
        line_threshold=80.0,
        branch_threshold=60.0,
        feishu_webhook_url="https://example.com/hook",
    )
    with caplog.at_level(logging.INFO):
        exit_code = _run_nightly_pipeline(
            logging.getLogger("nightly-test"),
            cfg,
            "https://example.com/hook",
            "",
            timeout_seconds=3000,
        )
    assert exit_code == PYTEST_TIMEOUT_EXIT_CODE
    assert drift_calls["n"] == 0
    assert "Skipping Hub drift check" in caplog.text
    assert len(reports) == 1
    assert reports[0].timed_out is True


def test_parse_pytest_stdout_extracts_failed_nodes_and_tb_line_reason() -> None:
    stats = parse_pytest_stdout(_SAMPLE_STDOUT, exit_code=1)
    assert stats.failed_cases == ("tests/smoke/test_b.py::test_fail",)
    assert stats.passed == 1
    assert stats.failed == 1
    assert stats.failure_reasons["tests/smoke/test_b.py::test_fail"] == "AssertionError: boom"
