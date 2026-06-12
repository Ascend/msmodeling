"""Tests for nightly.main — command builders, _coverage_summary, emit_report."""

from __future__ import annotations

import json
import logging
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

import pytest

from scripts.helpers._config import Config
from scripts.helpers.ci_gate.gate_policy import GatePolicy, default_test_discovery
from scripts.helpers.nightly.main import (
    _PHASE_LABELS,
    _TEST_MAP_MARKER,
    _build_benchmark_pytest_cmd,
    _build_network_pytest_cmd,
    _build_nightly_pytest_cmd,
    _build_terminal_summary,
    _build_test_map_pytest_cmd,
    _coverage_summary,
    _fetch_hub_config,
    _load_vendored_config,
    _run_config_drift_check,
    _stream_pytest,
    _terminate_process_tree,
    _write_test_map_artifacts,
    emit_report,
    main,
)
from scripts.helpers.nightly.pytest_parser import NightlyRunStats
from scripts.helpers.nightly.report_models import CoverageSummary
from tests.helpers.fake_subprocess import FakeCompleted
from tests.helpers.junit_xml import write_junit_xml, write_phase_junit

# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

_BASE_CFG = Config(
    test_map_path="",
    base_branch="master",
    line_threshold=70.0,
    branch_threshold=50.0,
    benchmark_parallel=False,
    feishu_webhook_url="",
    msmodeling_cache=".msmodeling_cache",
    weights_prune=True,
)

_PARALLEL_CFG = Config(
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


def test_test_map_cmd_contains_smoke_and_regression_and_coverage(tmp_path: Path) -> None:
    junit = tmp_path / "phase1.xml"
    cmd = _build_test_map_pytest_cmd("python3", junit_xml=junit)
    assert "tests/smoke/" in cmd
    assert "tests/regression/" in cmd
    assert "not npu and not nightly and not network" in cmd
    assert "-n" in cmd
    assert "--dist" in cmd
    assert "worksteal" in cmd
    assert "--cov-context=test" in cmd
    assert "--cov-branch" in cmd
    assert f"--junit-xml={junit}" in cmd


def test_test_map_write_marker_keeps_npu_tests_collectible() -> None:
    assert _TEST_MAP_MARKER == "not nightly and not network"


# ---------------------------------------------------------------------------
# _build_nightly_pytest_cmd
# ---------------------------------------------------------------------------


def test_nightly_cmd_contains_smoke_and_regression_with_nightly_marker(
    tmp_path: Path,
) -> None:
    junit = tmp_path / "phase2a.xml"
    cmd = _build_nightly_pytest_cmd("python3", junit_xml=junit)
    assert "tests/smoke/" in cmd
    assert "tests/regression/" in cmd
    assert "not npu and nightly and not network" in cmd
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


def test_benchmark_cmd_parallel_adds_auto_flag(tmp_path: Path) -> None:
    junit = tmp_path / "bench.xml"
    cmd = _build_benchmark_pytest_cmd("python3", _PARALLEL_CFG, junit_xml=junit)
    assert "-n" in cmd
    assert "auto" in cmd
    assert f"--junit-xml={junit}" in cmd


# ---------------------------------------------------------------------------
# _build_network_pytest_cmd
# ---------------------------------------------------------------------------


def test_network_cmd_targets_network_marker_and_serial_run(tmp_path: Path) -> None:
    junit = tmp_path / "network.xml"
    cmd = _build_network_pytest_cmd("python3", junit_xml=junit)
    assert cmd[0] == "python3"
    assert "tests/" in cmd
    assert "not npu and network" in cmd
    assert "-n" not in cmd
    assert f"--junit-xml={junit}" in cmd


# ---------------------------------------------------------------------------
# _load_vendored_config / _fetch_hub_config
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _coverage_summary
# ---------------------------------------------------------------------------


def test_coverage_summary_no_data_file_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts.helpers.nightly import main

    monkeypatch.setattr(main, "REPO_ROOT", tmp_path)
    result = _coverage_summary(_BASE_CFG)
    assert result is None


def test_coverage_summary_above_threshold_marks_passed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    result = _coverage_summary(_BASE_CFG)
    assert result is not None
    assert result.line_percent == 85.0
    assert result.branch_percent == 80.0
    assert result.gate_passed is True


def test_coverage_summary_below_threshold_marks_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    result = _coverage_summary(_BASE_CFG)
    assert result is not None
    assert result.line_percent == 60.0
    assert result.gate_passed is False


# ---------------------------------------------------------------------------
# emit_report
# ---------------------------------------------------------------------------


def test_emit_report_skips_feishu_without_webhook(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    junit = tmp_path / "report.xml"
    write_junit_xml(junit, passed=1, failed=1, duration=10.0)
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

    pushed: list[tuple[str, dict[str, Any]]] = []

    def _fake_push(url: str, payload: dict[str, Any]) -> None:
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
    map_path = tmp_path / "map.json"
    write_junit_xml(junit, passed=1, duration=5.0)
    emit_report(
        junit_xml_paths=(junit,),
        coverage=cov,
        test_map_written=True,
        test_map_path=map_path,
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

    def _fake_push(url: str, payload: dict[str, Any]) -> None:
        pushed.append((url, payload))

    monkeypatch.setattr("scripts.helpers.nightly.main.push_feishu", _fake_push)
    junit = tmp_path / "report.xml"
    write_junit_xml(junit, passed=1, duration=1.0)
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


def test_emit_report_merges_nightly_pipeline_phase_junit_xml(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Regression: aggregate phase1 smoke/regression UT, phase2a nightly, phase2b benchmark JUnit."""
    phase1 = tmp_path / "phase1_test_map.xml"
    phase2a = tmp_path / "phase2a_nightly.xml"
    phase2b = tmp_path / "phase2b_benchmark.xml"
    write_phase_junit(
        phase1,
        file_path="tests/regression/cli/test_run.py",
        passed=2,
        duration=100.0,
    )
    write_phase_junit(
        phase2a,
        file_path="tests/regression/tensor_cast/test_compile.py",
        passed=1,
        failed=1,
        duration=200.0,
    )
    write_phase_junit(
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


def test_emit_report_uses_phase_log_when_junit_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pushed: list[tuple[str, dict[str, Any]]] = []

    def _fake_push(url: str, payload: dict[str, Any]) -> None:
        pushed.append((url, payload))

    monkeypatch.setattr("scripts.helpers.nightly.main.push_feishu", _fake_push)
    phase_log = tmp_path / "phase1_test_map.log"
    phase_log.write_text(
        "collecting ...\nE   ValueError: 'deepseek_v4' is already used\n",
        encoding="utf-8",
    )
    missing_junit = tmp_path / "phase1_test_map.xml"

    emit_report(
        junit_xml_paths=(missing_junit,),
        coverage=None,
        test_map_written=False,
        test_map_path=None,
        webhook_url="https://example.com/hook",
        overall_exit=1,
        phase_exits=(1,),
        phase_log_paths=(phase_log,),
    )

    text = pushed[0][1]["content"]["text"]
    assert "FAILED (pytest exit 1, no JUnit testcase data)" in text
    assert "deepseek_v4" in text


def test_emit_report_four_phase_all_infra_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pushed: list[tuple[str, dict[str, Any]]] = []

    def _fake_push(url: str, payload: dict[str, Any]) -> None:
        pushed.append((url, payload))

    monkeypatch.setattr("scripts.helpers.nightly.main.push_feishu", _fake_push)

    phase_names = (
        "phase1_test_map",
        "phase2a_nightly",
        "phase2b_benchmark",
        "phase2c_network",
    )
    phase_logs: list[Path] = []
    missing_junits: list[Path] = []
    for name in phase_names:
        log_path = tmp_path / f"{name}.log"
        log_path.write_text(
            f"collecting ...\nE   ImportError: {name} collection failed\n",
            encoding="utf-8",
        )
        phase_logs.append(log_path)
        missing_junits.append(tmp_path / f"{name}.xml")

    emit_report(
        junit_xml_paths=tuple(missing_junits),
        coverage=None,
        test_map_written=False,
        test_map_path=None,
        webhook_url="https://example.com/hook",
        overall_exit=1,
        phase_exits=(1, 1, 1, 1),
        phase_log_paths=tuple(phase_logs),
    )

    text = pushed[0][1]["content"]["text"]
    assert "FAILED (pytest exit 1, no JUnit testcase data)" in text
    assert "Per-phase:" in text
    assert text.count("no JUnit details") == 4
    assert "smoke UT (coverage mapping)" in text
    assert "network Hub tests" in text
    assert "phase1_test_map collection failed" in text


# ---------------------------------------------------------------------------
# _run_config_drift_check (non-blocking, mocked Hub)
# ---------------------------------------------------------------------------


def test_drift_check_does_not_raise_when_hub_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.helpers.nightly import main as nightly_main

    monkeypatch.setattr(
        nightly_main,
        "_load_vendored_config",
        lambda _fixture: {"model_type": "deepseek_v3"},
    )

    def _boom(_model_id: str) -> dict[str, object]:
        raise RuntimeError("network down")

    monkeypatch.setattr(nightly_main, "_fetch_hub_config", _boom)

    warnings = _run_config_drift_check()
    assert isinstance(warnings, tuple)
    assert any("Hub config fetch failed" in w for w in warnings)


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


def test_drift_check_warns_on_missing_vendored_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.helpers.nightly import main as nightly_main

    monkeypatch.setattr(nightly_main, "_DRIFT_FIXTURE_MAP", {"some/Model": "missing_fixture"})
    monkeypatch.setattr(nightly_main, "_load_vendored_config", lambda _fixture: None)

    warnings = _run_config_drift_check()
    assert any("drift baseline absent" in w for w in warnings)


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

    def _fake_popen(*_args: Any, **_kwargs: Any) -> _FakePytestProc:
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

    _terminate_process_tree(cast("subprocess.Popen[str]", fake_proc), sigterm_timeout_seconds=0.01)

    assert killpg_calls == [
        (fake_proc.pid, signal.SIGTERM),
        (fake_proc.pid, signal.SIGKILL),
    ]


def _cfg_with_map_path(map_path: Path) -> Config:
    return Config(
        test_map_path=str(map_path),
        base_branch="master",
        line_threshold=70.0,
        branch_threshold=50.0,
        benchmark_parallel=False,
        feishu_webhook_url="",
        msmodeling_cache=".msmodeling_cache",
        weights_prune=True,
    )


def _minimal_gate_policy() -> GatePolicy:
    return GatePolicy(
        discovery=default_test_discovery(),
        roots=("cli/", "tensor_cast/", "serving_cast/", "web_ui/", "scripts/", "tools/"),
        source_exemptions=(),
        test_exemptions=(),
        approvers=frozenset({"fangkai"}),
    )


# ---------------------------------------------------------------------------
# _write_test_map_artifacts
# ---------------------------------------------------------------------------


def test_write_test_map_artifacts_skips_when_map_exit_nonzero(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    logger = logging.getLogger("nightly.test")
    map_path = tmp_path / "map.json"
    with caplog.at_level(logging.WARNING, logger="nightly.test"):
        written, weak, redundancy, expired = _write_test_map_artifacts(
            logger,
            _BASE_CFG,
            map_path,
            map_exit=1,
        )
    assert written is False
    assert weak == ()
    assert redundancy == ()
    assert expired == ""
    assert not map_path.exists()
    assert "Skipping coverage mapping write" in caplog.text


def test_write_test_map_artifacts_writes_map_and_returns_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.helpers.nightly import main as nightly_main

    map_path = tmp_path / "map.json"
    fresh_map = {"cli/main.py": {"run": ["tests/regression/cli/test_run.py::test_run"]}}
    collect_calls: list[tuple[str, tuple[str, ...]]] = []
    write_calls: list[tuple[Path, dict[str, dict[str, list[str]]]]] = []

    allowed = frozenset({"tests/regression/cli/test_run.py::test_run"})

    def _fake_collect(
        *,
        marker_expr: str,
        roots: tuple[str, ...],
        allowed_node_ids: frozenset[str] | None = None,
    ) -> dict[str, dict[str, list[str]]]:
        collect_calls.append((marker_expr, roots, allowed_node_ids))
        return fresh_map

    def _fake_write(path: Path, mapping: dict[str, dict[str, list[str]]]) -> None:
        write_calls.append((path, mapping))
        path.write_text('{"schema_version": 1, "map": {}}', encoding="utf-8")

    gate_policy = _minimal_gate_policy()
    monkeypatch.setattr(nightly_main, "collect_test_map", _fake_collect)
    monkeypatch.setattr(nightly_main, "write_test_map", _fake_write)
    monkeypatch.setattr(nightly_main, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        nightly_main, "detect_redundant_cases", lambda mapping: [{"case": "x"}] if mapping == fresh_map else []
    )
    weak_calls: list[dict[str, object]] = []

    def _fake_weak(
        _path: object,
        _cov: object,
        *,
        mapping: dict[str, dict[str, list[str]]] | None = None,
        **_kwargs: object,
    ) -> tuple[str, ...]:
        weak_calls.append({"mapping": mapping})
        return ("cli/main.py::run",)

    monkeypatch.setattr(nightly_main, "compute_weak_coverage_symbols", _fake_weak)
    monkeypatch.setattr(nightly_main, "load_gate_policy", lambda _root: gate_policy)
    monkeypatch.setattr(nightly_main, "find_expired_unmapped", lambda _p, _m: ())
    monkeypatch.setattr(nightly_main, "find_expired_test_exemptions", lambda _p: ())
    monkeypatch.setattr(nightly_main, "format_expired_exemptions_section", lambda _e: "")
    monkeypatch.setattr(nightly_main, "format_expired_test_exemptions_section", lambda _e: "")

    cfg = _cfg_with_map_path(map_path)
    logger = logging.getLogger("nightly.test")
    written, weak, redundancy, expired = _write_test_map_artifacts(
        logger,
        cfg,
        map_path,
        map_exit=0,
        allowed_node_ids=allowed,
    )
    assert written is True
    assert collect_calls == [(_TEST_MAP_MARKER, gate_policy.roots, allowed)]
    assert write_calls == [(map_path, fresh_map)]
    assert weak_calls == [{"mapping": fresh_map}]
    assert weak == ("cli/main.py::run",)
    assert redundancy == ({"case": "x"},)
    assert expired == ""


def test_write_test_map_artifacts_uses_fresh_collect_mapping_not_cfg_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.helpers.nightly import main as nightly_main

    map_path = tmp_path / "map.json"
    fresh_map = {"cli/main.py": {"run": ["tests/regression/cli/test_run.py::test_fresh"]}}
    stale_map = {"cli/main.py": {"run": ["tests/regression/cli/test_run.py::test_stale"]}}
    expired_maps: list[dict[str, dict[str, list[str]]]] = []
    redundant_maps: list[dict[str, dict[str, list[str]]]] = []

    monkeypatch.setattr(nightly_main, "collect_test_map", lambda **_kwargs: fresh_map)
    monkeypatch.setattr(nightly_main, "write_test_map", lambda path, mapping: path.write_text("{}", encoding="utf-8"))
    monkeypatch.setattr(nightly_main, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "scripts.helpers.common.test_map_loader.load_test_map",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("must not load cfg baseline")),
    )

    def _track_redundant(mapping: dict[str, dict[str, list[str]]]) -> tuple[()]:
        redundant_maps.append(mapping)
        return ()

    def _track_expired(_policy: object, mapping: dict[str, dict[str, list[str]]]) -> tuple[()]:
        expired_maps.append(mapping)
        return ()

    monkeypatch.setattr(nightly_main, "detect_redundant_cases", _track_redundant)
    monkeypatch.setattr(nightly_main, "compute_weak_coverage_symbols", lambda _p, _c, **_kw: ())
    monkeypatch.setattr(nightly_main, "load_gate_policy", lambda _root: _minimal_gate_policy())
    monkeypatch.setattr(nightly_main, "find_expired_unmapped", _track_expired)
    monkeypatch.setattr(nightly_main, "find_expired_test_exemptions", lambda _p: ())
    monkeypatch.setattr(nightly_main, "format_expired_exemptions_section", lambda _e: "")
    monkeypatch.setattr(nightly_main, "format_expired_test_exemptions_section", lambda _e: "")

    _write_test_map_artifacts(
        logging.getLogger("nightly.test"),
        _cfg_with_map_path(map_path),
        map_path,
        map_exit=0,
    )

    assert expired_maps == [fresh_map]
    assert redundant_maps == [fresh_map]
    assert stale_map != fresh_map


def test_write_test_map_artifacts_includes_expired_test_section(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    from datetime import date

    from scripts.helpers.ci_gate.gate_policy import ExpiredExemptionReport
    from scripts.helpers.nightly import main as nightly_main

    map_path = tmp_path / "map.json"
    expired_report = ExpiredExemptionReport(
        symbol_key="tests/regression/cli/test_old.py",
        deadline=date(2020, 1, 1),
        reason="legacy",
        applicant="alice",
        approver="fangkai",
        ticket=None,
    )

    gate_policy = _minimal_gate_policy()
    monkeypatch.setattr(nightly_main, "collect_test_map", lambda **_kwargs: {})
    monkeypatch.setattr(
        nightly_main,
        "write_test_map",
        lambda path, _mapping: path.write_text('{"schema_version": 1, "map": {}}', encoding="utf-8"),
    )
    monkeypatch.setattr(nightly_main, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(nightly_main, "detect_redundant_cases", lambda _m: ())
    monkeypatch.setattr(nightly_main, "compute_weak_coverage_symbols", lambda _p, _c, **_kw: ())
    monkeypatch.setattr(nightly_main, "load_gate_policy", lambda _root: gate_policy)
    monkeypatch.setattr(nightly_main, "find_expired_unmapped", lambda _p, _m: ())
    monkeypatch.setattr(nightly_main, "find_expired_test_exemptions", lambda _p: (expired_report,))
    monkeypatch.setattr(nightly_main, "format_expired_exemptions_section", lambda _e: "")
    monkeypatch.setattr(
        nightly_main,
        "format_expired_test_exemptions_section",
        lambda reports: f"expired-tests:{len(reports)}",
    )

    cfg = _cfg_with_map_path(map_path)
    logger = logging.getLogger("nightly.test")
    with caplog.at_level(logging.WARNING, logger="nightly.test"):
        written, _, _, expired = _write_test_map_artifacts(logger, cfg, map_path, map_exit=0)
    assert written is True
    assert expired == "expired-tests:1"
    assert "Found 1 expired test exemption(s)" in caplog.text


def test_build_terminal_summary_includes_phase_coverage_drift_and_logs(
    tmp_path: Path,
) -> None:
    phase1 = tmp_path / "phase1.xml"
    phase2a = tmp_path / "phase2a.xml"
    write_phase_junit(phase1, file_path="tests/regression/cli/test_run.py", passed=2, duration=10.0)
    write_phase_junit(
        phase2a,
        file_path="tests/regression/tensor_cast/test_compile.py",
        passed=1,
        failed=1,
        duration=20.0,
    )
    phase_logs = (
        tmp_path / "phase1.log",
        tmp_path / "phase2a.log",
        None,
        None,
    )
    for log_path in phase_logs:
        if log_path is not None:
            log_path.write_text("pytest output", encoding="utf-8")

    coverage = CoverageSummary(
        line_percent=80.0,
        branch_percent=60.0,
        line_threshold=70.0,
        branch_threshold=50.0,
        gate_passed=True,
        message="Coverage gate passed: line=80.0% branch=60.0%",
    )
    stats = NightlyRunStats(
        passed=3,
        failed=1,
        errors=0,
        duration_sec=30.0,
        failed_cases=(),
        first_error="",
    )
    lines = _build_terminal_summary(
        overall_exit=1,
        stats=stats,
        phase_labels=_PHASE_LABELS,
        junit_paths=(phase1, phase2a, tmp_path / "missing.xml", tmp_path / "missing2.xml"),
        phase_exits=(0, 1, 0, 0),
        coverage=coverage,
        drift_warnings=("some/Model [fixture] model_type: vendored='a' hub='b'",),
        phase_log_paths=phase_logs,
        include_phase_logs=True,
    )

    assert lines[0] == "smoke UT (coverage mapping): exit=0 passed=2 failed=0 duration=20s"
    assert lines[1] == "long-running tests: exit=1 passed=1 failed=1 duration=40s"
    assert "Nightly exit=1: passed=3 failed=1 errors=0 duration=30s" in lines
    assert coverage.message in lines
    assert "Config drift: some/Model [fixture] model_type: vendored='a' hub='b'" in lines
    assert f"smoke UT (coverage mapping) log: {phase_logs[0]}" in lines
    assert f"long-running tests log: {phase_logs[1]}" in lines


def test_build_terminal_summary_omits_phase_logs_without_webhook(tmp_path: Path) -> None:
    junit = tmp_path / "phase1.xml"
    write_phase_junit(junit, file_path="tests/regression/cli/test_run.py", passed=1, duration=1.0)
    log_path = tmp_path / "phase1.log"
    log_path.write_text("pytest output", encoding="utf-8")

    lines = _build_terminal_summary(
        overall_exit=0,
        stats=NightlyRunStats(1, 0, 0, 1.0, (), ""),
        phase_labels=_PHASE_LABELS[:1],
        junit_paths=(junit,),
        phase_exits=(0,),
        coverage=None,
        drift_warnings=(),
        phase_log_paths=(log_path,),
        include_phase_logs=False,
    )

    assert all(" log: " not in line for line in lines)


def test_main_returns_130_on_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.helpers.nightly.main.Config.from_env", lambda: _BASE_CFG)
    monkeypatch.setattr(
        "scripts.helpers.nightly.main.resolve_test_map_path",
        lambda _cfg, must_exist=False: Path("/tmp/map.json"),
    )
    monkeypatch.setattr(
        "scripts.helpers.nightly.main._stream_pytest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    assert main() == 130
