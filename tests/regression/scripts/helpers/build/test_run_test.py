"""Regression tests for scripts.helpers.build.run_test."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.helpers.build import run_test as run_test_mod
from scripts.helpers.build.argv import BuildSuite
from scripts.helpers.build.main import main
from scripts.helpers.build.run_test import run_test
from scripts.helpers.defaults import PYTEST_XDIST_ARGS
from tests.helpers.cli_runner import run_cli_main
from tests.regression.scripts.helpers.build.conftest import (
    SubprocessRunCapture,
    build_options,
    patch_subprocess_run,
    patch_uv_in_path,
)


def test_run_test_ci_gate_downloads_when_map_unset(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    subprocess_capture: SubprocessRunCapture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Normal: unset map → download then CI gate."""
    monkeypatch.delenv("MSMODELING_TEST_MAP_PATH", raising=False)
    dest = repo_root / ".msmodeling_cache" / "test_map" / "master" / "test_map.json"

    def fake_resolve(**_kwargs: Any) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("{}", encoding="utf-8")
        return dest

    monkeypatch.setattr(run_test_mod, "resolve_test_map_path", fake_resolve)
    with caplog.at_level("WARNING", logger="build"):
        assert run_test(build_options(is_test=True, suite=BuildSuite.CI_GATE)) == 0
    call = subprocess_capture.merged_output_calls[0]
    assert call["cmd"] == ["bash", str(repo_root / "scripts" / "run_ci_gate.sh")]
    assert call["env"]["MSMODELING_TEST_MAP_PATH"] == str(dest)
    summary = json.loads(
        (repo_root / "artifacts" / "test-reports" / "gate-summary.json").read_text(encoding="utf-8"),
    )
    assert summary["mode"] == "ci_gate"
    assert summary["test_map_path"] == str(dest)


def test_run_test_full_suite_uses_xdist(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    monkeypatch.delenv("MSMODELING_TEST_MAP_PATH", raising=False)
    assert run_test(build_options(is_test=True, suite=BuildSuite.FULL)) == 0
    assert len(subprocess_capture.merged_output_calls) == 1
    call = subprocess_capture.merged_output_calls[0]
    assert call["cmd"] == ["/fake/uv", "run", "pytest", "tests", *PYTEST_XDIST_ARGS]
    summary = json.loads(
        (repo_root / "artifacts" / "test-reports" / "gate-summary.json").read_text(encoding="utf-8"),
    )
    assert summary["mode"] == "full"
    assert summary["test_map_path"] is None


def test_run_test_full_suite_propagates_pytest_exit_code(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    monkeypatch.delenv("MSMODELING_TEST_MAP_PATH", raising=False)

    def fail_pytest(_cmd: list[str], **_kwargs: Any) -> int:
        return 5

    subprocess_capture.on_merged_output = fail_pytest
    assert run_test(build_options(is_test=True, suite=BuildSuite.FULL)) == 5
    summary = json.loads(
        (repo_root / "artifacts" / "test-reports" / "gate-summary.json").read_text(encoding="utf-8"),
    )
    assert summary["exit_code"] == 5
    assert summary["mode"] == "full"


def test_run_test_full_suite_applies_offline_extras(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    del repo_root
    monkeypatch.delenv("MSMODELING_TEST_MAP_PATH", raising=False)
    options = build_options(
        is_test=True,
        suite=BuildSuite.FULL,
        extras={"offline": "1", "weights_prune": "1"},
    )
    assert run_test(options) == 0
    env = subprocess_capture.merged_output_calls[0]["env"]
    assert env["MSMODELING_OFFLINE"] == "1"
    assert env["MSMODELING_TEST_WEIGHTS_PRUNE"] == "1"


@pytest.mark.parametrize(
    ("suite", "script_name"),
    [
        (BuildSuite.SMOKE, "run_smoke.sh"),
        (BuildSuite.REGRESSION, "run_regression.sh"),
        (BuildSuite.BENCHMARK, "run_benchmark.sh"),
    ],
)
def test_run_test_named_suite_delegates_to_script(
    repo_root: Path,
    subprocess_capture: SubprocessRunCapture,
    suite: BuildSuite,
    script_name: str,
) -> None:
    assert run_test(build_options(is_test=True, suite=suite)) == 0
    call = subprocess_capture.merged_output_calls[0]
    assert call["cmd"] == ["bash", str(repo_root / "scripts" / script_name)]
    summary = json.loads(
        (repo_root / "artifacts" / "test-reports" / "gate-summary.json").read_text(encoding="utf-8"),
    )
    assert summary["mode"] == suite.value


def test_run_test_delegates_env_and_tee(
    repo_root: Path,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    map_file = repo_root / "map.json"
    map_file.write_text("{}", encoding="utf-8")

    options = build_options(
        is_test=True,
        suite=BuildSuite.CI_GATE,
        extras={
            "test_map_path": str(map_file),
            "base_branch": "develop",
            "offline": "1",
            "weights_prune": "1",
        },
    )
    assert run_test(options) == 0
    assert len(subprocess_capture.merged_output_calls) == 1
    call = subprocess_capture.merged_output_calls[0]
    assert call["cmd"] == ["bash", str(repo_root / "scripts" / "run_ci_gate.sh")]
    assert call["env"]["MSMODELING_TEST_MAP_PATH"] == str(map_file)
    assert call["env"]["MSMODELING_TEST_BASE_BRANCH"] == "develop"
    assert call["env"]["MSMODELING_OFFLINE"] == "1"
    assert call["env"]["MSMODELING_TEST_WEIGHTS_PRUNE"] == "1"
    log_path = repo_root / "artifacts" / "test-reports" / "ci_gate.log"
    assert log_path.is_file()
    assert log_path.read_text(encoding="utf-8") == "gate output\n"


def test_run_test_uses_env_test_map_path(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    map_file = repo_root / "env_map.json"
    map_file.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MSMODELING_TEST_MAP_PATH", str(map_file))

    assert run_test(build_options(is_test=True, suite=BuildSuite.CI_GATE)) == 0
    call = subprocess_capture.merged_output_calls[0]
    assert call["env"]["MSMODELING_TEST_MAP_PATH"] == str(map_file)


def test_run_test_propagates_subprocess_exit_code(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    map_file = repo_root / "map.json"
    map_file.write_text("{}", encoding="utf-8")

    def fail_gate(_cmd: list[str], **_kwargs: Any) -> int:
        return 17

    subprocess_capture.on_merged_output = fail_gate

    options = build_options(
        is_test=True,
        suite=BuildSuite.CI_GATE,
        extras={"test_map_path": str(map_file)},
    )
    assert run_test(options) == 17
    summary = json.loads(
        (repo_root / "artifacts" / "test-reports" / "gate-summary.json").read_text(encoding="utf-8"),
    )
    assert summary["exit_code"] == 17
    assert summary["mode"] == "ci_gate"
    assert summary["test_map_path"] == str(map_file)


def test_cli_test_default_suite_is_full(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_uv: None,
) -> None:
    del with_uv
    del repo_root
    capture = patch_subprocess_run(monkeypatch, SubprocessRunCapture())
    monkeypatch.delenv("MSMODELING_TEST_MAP_PATH", raising=False)
    result = run_cli_main(main, ["test"], prog="build.py")
    assert result.returncode == 0
    assert capture.merged_output_calls[0]["cmd"][:4] == ["/fake/uv", "run", "pytest", "tests"]


def test_run_test_ci_gate_fail_fast_before_test_map_download(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python/prereq fail-fast must run before any test_map download."""
    del repo_root
    order: list[str] = []

    def record_fail_fast(*, mode: str) -> None:
        order.append(f"fail_fast:{mode}")
        raise SystemExit(42)

    def record_resolve(**_kwargs: Any) -> Path:
        order.append("resolve")
        return Path("/tmp/should-not-be-used.json")

    monkeypatch.setattr(run_test_mod, "fail_fast", record_fail_fast)
    monkeypatch.setattr(run_test_mod, "resolve_test_map_path", record_resolve)
    monkeypatch.delenv("MSMODELING_TEST_MAP_PATH", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        run_test(build_options(is_test=True, suite=BuildSuite.CI_GATE))
    assert exc_info.value.code == 42
    assert order == ["fail_fast:test"]


def test_run_test_ci_gate_orders_fail_fast_then_download_then_bootstrap(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    order: list[str] = []
    dest = repo_root / "map.json"
    dest.write_text("{}", encoding="utf-8")

    def record_fail_fast(*, mode: str) -> None:
        order.append(f"fail_fast:{mode}")

    def record_resolve(**_kwargs: Any) -> Path:
        order.append("resolve")
        return dest

    def record_bootstrap(mode: str) -> str:
        order.append(f"bootstrap:{mode}")
        return "/fake/uv"

    monkeypatch.setattr(run_test_mod, "fail_fast", record_fail_fast)
    monkeypatch.setattr(run_test_mod, "resolve_test_map_path", record_resolve)
    monkeypatch.setattr(run_test_mod, "bootstrap", record_bootstrap)
    monkeypatch.delenv("MSMODELING_TEST_MAP_PATH", raising=False)
    assert run_test(build_options(is_test=True, suite=BuildSuite.CI_GATE)) == 0
    assert order[:3] == ["fail_fast:test", "resolve", "bootstrap:test"]
    assert subprocess_capture.merged_output_calls


def test_run_test_download_failure_stops(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_uv: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    del with_uv
    del repo_root
    from scripts.helpers.build.test_map_fetch import MapFetchError

    monkeypatch.delenv("MSMODELING_TEST_MAP_PATH", raising=False)

    def fail_resolve(**_kwargs: Any) -> Path:
        raise MapFetchError("HTTP 404")

    monkeypatch.setattr(run_test_mod, "resolve_test_map_path", fail_resolve)
    with caplog.at_level("ERROR", logger="build"):
        assert run_test(build_options(is_test=True, suite=BuildSuite.CI_GATE)) == 1
    assert "Cannot start CI gate without a test_map" in caplog.text


def test_run_test_without_uv_install_failure_returns_1(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.helpers.build import bootstrap as bootstrap_mod

    map_file = repo_root / "x.json"
    map_file.write_text("{}", encoding="utf-8")
    patch_uv_in_path(monkeypatch, uv_path=None)

    def fail_install(*_args: Any, **_kwargs: Any) -> None:
        raise SystemExit(1)

    monkeypatch.setattr(bootstrap_mod, "ensure_uv", fail_install)
    monkeypatch.setattr(run_test_mod, "bootstrap", bootstrap_mod.bootstrap)
    try:
        code = run_test(
            build_options(
                is_test=True,
                suite=BuildSuite.CI_GATE,
                extras={"test_map_path": str(map_file)},
            )
        )
    except SystemExit as exc:
        code = int(exc.code or 1)
    assert code == 1


def test_run_test_syncs_ci_group_not_build(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.helpers.build import bootstrap as bootstrap_mod

    map_file = repo_root / "map.json"
    map_file.write_text("{}", encoding="utf-8")
    patch_uv_in_path(monkeypatch, uv_path="/fake/uv")
    capture = SubprocessRunCapture()
    patch_subprocess_run(monkeypatch, capture)
    monkeypatch.setattr(run_test_mod, "bootstrap", bootstrap_mod.bootstrap)

    assert (
        run_test(
            build_options(
                is_test=True,
                suite=BuildSuite.CI_GATE,
                extras={"test_map_path": str(map_file)},
            )
        )
        == 0
    )
    assert capture.sync_calls
    sync_cmd = capture.sync_calls[0]
    assert sync_cmd[sync_cmd.index("--group") + 1] == "ci"
    assert "build" not in sync_cmd[sync_cmd.index("--group") :]


def test_run_test_applies_uv_and_hf_defaults(
    monkeypatch: pytest.MonkeyPatch,
    subprocess_capture: SubprocessRunCapture,
) -> None:
    monkeypatch.delenv("UV_INDEX_URL", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert run_test(build_options(is_test=True, suite=BuildSuite.FULL)) == 0
    env = subprocess_capture.merged_output_calls[0]["env"]
    assert env["UV_INDEX_URL"] == "https://repo.huaweicloud.com/repository/pypi/simple"
    assert env["HF_ENDPOINT"] == "https://hf-mirror.com"
