"""Tests for nightly.main — command builders, _coverage_summary, emit_report."""

from __future__ import annotations

import json
import logging
import signal
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from scripts.helpers._config import Config
from scripts.helpers.nightly.main import (
    _build_benchmark_pytest_cmd,
    _build_nightly_pytest_cmd,
    _build_test_map_pytest_cmd,
    _coverage_summary,
    _stream_pytest,
    _terminate_process_tree,
    emit_report,
    main,
)
from tests.helpers.fake_subprocess import FakeCompleted

# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_cfg() -> Config:
    return Config(
        test_map_path="",
        base_branch="master",
        line_threshold=70.0,
        branch_threshold=50.0,
        benchmark_parallel=False,
        feishu_webhook_url="",
        msmodeling_cache=".msmodeling_cache",
        weights_prune=True,
    )


@pytest.fixture(scope="module")
def parallel_cfg() -> Config:
    return Config(
        test_map_path="",
        base_branch="master",
        line_threshold=70.0,
        branch_threshold=50.0,
        benchmark_parallel=True,
        feishu_webhook_url="",
        msmodeling_cache=".msmodeling_cache",
        weights_prune=True,
    )


# ---------------------------------------------------------------------------
# _build_test_map_pytest_cmd
# ---------------------------------------------------------------------------


def test_test_map_cmd_contains_smoke_and_regression_and_coverage(base_cfg: Config, tmp_path: Path) -> None:
    junit = tmp_path / "phase1.xml"
    cmd = _build_test_map_pytest_cmd("python3", base_cfg, junit_xml=junit)
    assert "tests/smoke/" in cmd
    assert "tests/regression/" in cmd
    assert "not npu and not nightly" in cmd
    assert "-n" in cmd and "auto" in cmd
    assert "--cov-context=test" in cmd
    assert "--cov-branch" in cmd
    assert f"--junit-xml={junit}" in cmd


# ---------------------------------------------------------------------------
# _build_nightly_pytest_cmd
# ---------------------------------------------------------------------------


def test_nightly_cmd_contains_smoke_and_regression_with_nightly_marker(tmp_path: Path) -> None:
    junit = tmp_path / "phase2a.xml"
    cmd = _build_nightly_pytest_cmd("python3", junit_xml=junit)
    assert "tests/smoke/" in cmd
    assert "tests/regression/" in cmd
    assert "not npu and nightly" in cmd
    assert "-n" in cmd
    assert "auto" in cmd
    assert "--cov-context=test" not in cmd
    assert f"--junit-xml={junit}" in cmd


# ---------------------------------------------------------------------------
# _build_benchmark_pytest_cmd
# ---------------------------------------------------------------------------


# def test_benchmark_cmd_default_no_parallel_flag(base_cfg: Config, tmp_path: Path) -> None:
#     junit = tmp_path / "bench.xml"
#     cmd = _build_benchmark_pytest_cmd("python3", base_cfg, junit_xml=junit)
#     assert "tests/benchmark/" in cmd
#     assert "-n" not in cmd
#     assert f"--junit-xml={junit}" in cmd


def test_benchmark_cmd_parallel_adds_auto_flag(parallel_cfg: Config, tmp_path: Path) -> None:
    junit = tmp_path / "bench.xml"
    cmd = _build_benchmark_pytest_cmd("python3", parallel_cfg, junit_xml=junit)
    assert "-n" in cmd
    assert "auto" in cmd
    assert f"--junit-xml={junit}" in cmd


# ---------------------------------------------------------------------------
# _coverage_summary
# ---------------------------------------------------------------------------


def test_coverage_summary_no_data_file_returns_none(
    base_cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.helpers.nightly import main

    monkeypatch.setattr(main, "REPO_ROOT", tmp_path)
    result = _coverage_summary(base_cfg)
    assert result is None


def test_coverage_summary_above_threshold_marks_passed(
    base_cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.helpers.nightly import main

    cov_path = tmp_path / ".coverage"
    cov_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(main, "REPO_ROOT", tmp_path)

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
    result = _coverage_summary(base_cfg)
    assert result is not None
    assert result.line_percent == 85.0
    assert result.branch_percent == 80.0
    assert result.gate_passed is True


def test_coverage_summary_below_threshold_marks_failed(
    base_cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.helpers.nightly import main

    cov_path = tmp_path / ".coverage"
    cov_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(main, "REPO_ROOT", tmp_path)

    totals_json = json.dumps(
        {
            "totals": {
                "percent_covered_display": "60.0%",
                "num_branches": 10,
                "covered_branches": 4,
            },
        }
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, totals_json, ""))
    result = _coverage_summary(base_cfg)
    assert result is not None
    assert result.line_percent == 60.0
    assert result.gate_passed is False


# ---------------------------------------------------------------------------
# emit_report
# ---------------------------------------------------------------------------


def _write_junit(path: Path, *, passed: int = 0, failed: int = 0, duration: float = 0.0) -> None:
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<testsuites>", "<testsuite>"]
    for index in range(passed):
        lines.append(f'<testcase classname="tests.smoke.test_a" name="test_pass_{index}" time="{duration}"/>')
    for index in range(failed):
        lines.append(
            f'<testcase classname="tests.smoke.test_fail" name="test_fail_{index}" time="{duration}">'
            '<failure message="AssertionError: bad">E   AssertionError: bad</failure>'
            "</testcase>"
        )
    lines.extend(["</testsuite>", "</testsuites>"])
    path.write_text("\n".join(lines), encoding="utf-8")


def test_emit_report_skips_feishu_without_webhook(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    junit = tmp_path / "report.xml"
    _write_junit(junit, passed=1, failed=1, duration=10.0)
    with caplog.at_level(logging.WARNING, logger="nightly"):
        emit_report(
            junit_xml_paths=(junit,),
            coverage=None,
            test_map_written=False,
            test_map_path=None,
            webhook_url=None,
        )
    assert "FEISHU_WEBHOOK_URL not set" in caplog.text


def test_emit_report_feishu_payload_includes_coverage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts.helpers.nightly.report_models import CoverageSummary

    pushed: list[tuple[str, dict]] = []

    def _fake_push(url: str, payload: dict) -> None:
        pushed.append((url, payload))

    monkeypatch.setattr("scripts.helpers.nightly.main.push_feishu", _fake_push)
    cov = CoverageSummary(
        line_percent=80.0,
        branch_percent=60.0,
        line_threshold=70.0,
        branch_threshold=50.0,
        gate_passed=True,
        message="passed",
    )
    junit = tmp_path / "report.xml"
    _write_junit(junit, passed=1, duration=5.0)
    emit_report(
        junit_xml_paths=(junit,),
        coverage=cov,
        test_map_written=True,
        test_map_path=Path("/tmp/map.json"),
        webhook_url="https://example.com/hook",
    )
    assert len(pushed) == 1
    text = pushed[0][1]["content"]["text"]
    assert "Coverage (PASS): line 80.0%" in text
    assert "Test map: " in text


def test_emit_report_pushes_to_feishu_when_url_provided(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    pushed = []

    def _fake_push(url: str, payload: dict) -> None:
        pushed.append((url, payload))

    monkeypatch.setattr("scripts.helpers.nightly.main.push_feishu", _fake_push)
    junit = tmp_path / "report.xml"
    _write_junit(junit, passed=1, duration=1.0)
    emit_report(
        junit_xml_paths=(junit,),
        coverage=None,
        test_map_written=False,
        test_map_path=None,
        webhook_url="https://example.com/hook",
    )
    assert len(pushed) == 1
    assert pushed[0][0] == "https://example.com/hook"
    assert pushed[0][1]["msg_type"] == "text"


def _write_phase_junit(
    path: Path,
    *,
    file_path: str,
    passed: int,
    failed: int = 0,
    duration: float,
) -> None:
    """Write a minimal JUnit XML file for one nightly pipeline phase."""
    module = file_path.replace("/", ".").removesuffix(".py")
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<testsuites>", "<testsuite>"]
    for index in range(passed):
        lines.append(f'<testcase classname="{module}" name="test_pass_{index}" file="{file_path}" time="{duration}"/>')
    for index in range(failed):
        lines.append(
            f'<testcase classname="{module}" name="test_fail_{index}" '
            f'file="{file_path}" time="{duration}">'
            '<failure message="AssertionError: bad">E   AssertionError: bad</failure>'
            "</testcase>"
        )
    lines.extend(["</testsuite>", "</testsuites>"])
    path.write_text("\n".join(lines), encoding="utf-8")


def test_emit_report_merges_nightly_pipeline_phase_junit_xml(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Regression: aggregate phase1 smoke/regression UT, phase2a nightly, phase2b benchmark JUnit."""
    phase1 = tmp_path / "phase1_test_map.xml"
    phase2a = tmp_path / "phase2a_nightly.xml"
    phase2b = tmp_path / "phase2b_benchmark.xml"
    _write_phase_junit(
        phase1,
        file_path="tests/regression/cli/test_run.py",
        passed=2,
        duration=100.0,
    )
    _write_phase_junit(
        phase2a,
        file_path="tests/regression/tensor_cast/test_compile.py",
        passed=1,
        failed=1,
        duration=200.0,
    )
    _write_phase_junit(
        phase2b,
        file_path="tests/benchmark/models/test_model_regression.py",
        passed=3,
        duration=50.0,
    )

    with caplog.at_level(logging.WARNING, logger="nightly"):
        stats = emit_report(
            junit_xml_paths=(phase1, phase2a, phase2b),
            coverage=None,
            test_map_written=False,
            test_map_path=None,
            webhook_url=None,
        )

    assert stats.passed == 6
    assert stats.failed == 1
    assert stats.errors == 0
    assert stats.duration_sec == pytest.approx(750.0)
    assert stats.failed_cases == ("tests/regression/tensor_cast/test_compile.py::test_fail_0",)
    assert "FEISHU_WEBHOOK_URL not set" in caplog.text


# ---------------------------------------------------------------------------
# _stream_pytest / _terminate_process_tree
# ---------------------------------------------------------------------------


class _InterruptingStdout:
    closed = False

    def __iter__(self) -> Iterator[str]:
        yield "line1\n"
        raise KeyboardInterrupt

    def close(self) -> None:
        self.closed = True


class _FakePytestProc:
    pid = 4242

    def __init__(self) -> None:
        self._returncode: int | None = None
        self.stdout = _InterruptingStdout()
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def poll(self) -> int | None:
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self._returncode is None:
            self._returncode = -signal.SIGTERM
        return self._returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._returncode = -signal.SIGTERM

    def kill(self) -> None:
        self.kill_calls += 1
        self._returncode = -signal.SIGKILL


def test_stream_pytest_keyboard_interrupt_terminates_process_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_proc = _FakePytestProc()
    killpg_calls: list[tuple[int, int]] = []

    def _fake_popen(*_args, **_kwargs):
        return fake_proc

    def _fake_killpg(pgid: int, sig: int) -> None:
        killpg_calls.append((pgid, sig))
        fake_proc._returncode = -sig

    monkeypatch.setattr("scripts.helpers.nightly.main.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("scripts.helpers.nightly.main.os.getpgid", lambda _pid: fake_proc.pid)
    monkeypatch.setattr("scripts.helpers.nightly.main.os.killpg", _fake_killpg)

    with pytest.raises(KeyboardInterrupt):
        _stream_pytest(["python3", "-m", "pytest"], cwd=tmp_path)

    assert killpg_calls == [(fake_proc.pid, signal.SIGTERM)]


def test_terminate_process_tree_escalates_to_sigkill_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = _FakePytestProc()
    killpg_calls: list[tuple[int, int]] = []

    def _wait(timeout: float | None = None) -> int:
        fake_proc.wait_calls += 1
        if fake_proc.wait_calls == 1:
            raise subprocess.TimeoutExpired(cmd=["pytest"], timeout=timeout or 0.01)
        fake_proc._returncode = -signal.SIGKILL
        return fake_proc._returncode

    def _fake_killpg(pgid: int, sig: int) -> None:
        killpg_calls.append((pgid, sig))
        if sig == signal.SIGKILL:
            fake_proc._returncode = -signal.SIGKILL

    fake_proc.wait = _wait  # type: ignore[method-assign]
    monkeypatch.setattr("scripts.helpers.nightly.main.os.getpgid", lambda _pid: fake_proc.pid)
    monkeypatch.setattr("scripts.helpers.nightly.main.os.killpg", _fake_killpg)

    _terminate_process_tree(fake_proc, sigterm_timeout_seconds=0.01)

    assert killpg_calls == [
        (fake_proc.pid, signal.SIGTERM),
        (fake_proc.pid, signal.SIGKILL),
    ]


def test_main_returns_130_on_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch, base_cfg: Config) -> None:
    monkeypatch.setattr("scripts.helpers.nightly.main.Config.from_env", lambda: base_cfg)
    monkeypatch.setattr(
        "scripts.helpers.nightly.main.resolve_test_map_path",
        lambda _cfg, must_exist=False: Path("/tmp/map.json"),
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.main._stream_pytest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    assert main() == 130
