"""Tests for common.pytest_runner."""

from __future__ import annotations

import os

import pytest

from scripts.helpers.common.pytest_runner import (
    PYTEST_IGNORE_ADDOPTS,
    _parse_not_found_node_ids,
    _run_collect_only,
    build_pytest_cmd,
    collect_all_test_node_ids,
    collect_test_node_ids,
    count_collected_tests,
    filter_collectable_node_ids,
    xdist_worker_args,
)
from tests.helpers.fake_subprocess import FakeCompleted

# ---------------------------------------------------------------------------
# PYTEST_IGNORE_ADDOPTS
# ---------------------------------------------------------------------------


def test_pytest_ignore_addopts_value() -> None:
    assert PYTEST_IGNORE_ADDOPTS == ["-o", "addopts="]


# ---------------------------------------------------------------------------
# xdist_worker_args
# ---------------------------------------------------------------------------


def test_xdist_worker_args_zero_returns_empty() -> None:
    assert xdist_worker_args(0) == []


def test_xdist_worker_args_positive_caps_at_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert xdist_worker_args(3) == ["-n", "3", "--dist", "worksteal"]


def test_xdist_worker_args_limited_by_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert xdist_worker_args(100) == ["-n", "2", "--dist", "worksteal"]


def test_xdist_worker_args_cpu_count_none_uses_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert xdist_worker_args(5) == ["-n", "1", "--dist", "worksteal"]


# ---------------------------------------------------------------------------
# collect_test_node_ids
# ---------------------------------------------------------------------------


def test_collect_test_node_ids_empty_targets_returns_empty_tuple() -> None:
    assert collect_test_node_ids([], marker="not npu") == ()


def test_collect_all_test_node_ids_skips_marker_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_stdout = "tests/smoke/test_a.py::test_foo\n"
    captured: list[list[str]] = []

    def _fake_run(cmd: list[str], **_kw: object) -> FakeCompleted:
        captured.append(cmd)
        return FakeCompleted(0, fake_stdout, "")

    monkeypatch.setattr("scripts.helpers.common.pytest_runner.subprocess.run", _fake_run)
    result = collect_all_test_node_ids(["tests/smoke"])
    assert result == ("tests/smoke/test_a.py::test_foo",)
    pytest_idx = captured[0].index("pytest")
    assert "not npu" not in captured[0][pytest_idx:]


def test_run_collect_only_returns_completed_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_stdout = "tests/smoke/test_a.py::test_foo\n"

    def _fake_run(cmd: list[str], **_kw: object) -> FakeCompleted:
        del cmd
        return FakeCompleted(0, fake_stdout, "")

    monkeypatch.setattr("scripts.helpers.common.pytest_runner.subprocess.run", _fake_run)
    proc = _run_collect_only(["tests/smoke/test_a.py::test_foo"], marker="not npu")
    assert proc.returncode == 0
    assert "tests/smoke/test_a.py::test_foo" in proc.stdout


def test_parse_not_found_node_ids_multiline_stderr() -> None:
    stderr = "ERROR: not found: tests/a.py::test_one\nsome other line\nERROR: not found: tests/b.py::test_two\n"
    assert _parse_not_found_node_ids(stderr) == frozenset({"tests/a.py::test_one", "tests/b.py::test_two"})


def test_parse_not_found_node_ids_normalizes_absolute_paths() -> None:
    stderr = "ERROR: not found: /build/msmodeling/tests/a.py::test_one\nERROR: not found: tests/b.py::test_two \n"
    assert _parse_not_found_node_ids(stderr) == frozenset({"tests/a.py::test_one", "tests/b.py::test_two"})


def test_filter_collectable_node_ids_exit_four_partial_stale_returns_valid_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = "tests/regression/cli/test_b.py::test_stale"
    valid = "tests/regression/cli/test_a.py::test_a"

    def _fake_run(cmd: list[str], **_kw: object) -> FakeCompleted:
        del cmd
        return FakeCompleted(
            4,
            "",
            f"ERROR: not found: /abs/repo/{stale}\n(no match in any of [<Module test_b.py>])",
        )

    monkeypatch.setattr("scripts.helpers.common.pytest_runner._run_collect_only", _fake_run)
    result = filter_collectable_node_ids((valid, stale), marker="not npu")
    assert result == (valid,)


def test_filter_collectable_node_ids_all_stale_skips_per_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collect_calls: list[tuple[str, ...]] = []
    stale = "tests/regression/cli/test_old.py::test_renamed"

    def _fake_run(cmd: list[str], **_kw: object) -> FakeCompleted:
        collect_calls.append(tuple(arg for arg in cmd if "::" in arg))
        return FakeCompleted(
            4,
            "",
            f"ERROR: not found: {stale}\n(no match in any of [<Module test_old.py>])",
        )

    monkeypatch.setattr("scripts.helpers.common.pytest_runner._run_collect_only", _fake_run)
    assert filter_collectable_node_ids((stale,), marker="not npu") == ()
    assert collect_calls == [(stale,)]


def test_filter_collectable_node_ids_exit_zero_summary_stdout_returns_all_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_a = "tests/regression/cli/test_a.py::test_a"
    valid_b = "tests/regression/cli/test_b.py::test_b"

    def _fake_run(cmd: list[str], **_kw: object) -> FakeCompleted:
        del cmd
        return FakeCompleted(0, "========================= 2 tests collected in 0.01s =========================\n", "")

    monkeypatch.setattr("scripts.helpers.common.pytest_runner._run_collect_only", _fake_run)
    result = filter_collectable_node_ids((valid_a, valid_b), marker="not npu")
    assert result == (valid_a, valid_b)


def test_filter_collectable_node_ids_exit_four_unparseable_stderr_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.helpers._config import ConfigError

    def _fake_run(cmd: list[str], **_kw: object) -> FakeCompleted:
        del cmd
        return FakeCompleted(4, "", "internal pytest error without not-found lines")

    monkeypatch.setattr("scripts.helpers.common.pytest_runner._run_collect_only", _fake_run)
    with pytest.raises(ConfigError, match="unparseable stderr"):
        filter_collectable_node_ids(("tests/regression/cli/test_a.py::test_a",), marker="not npu")


def test_filter_collectable_node_ids_exit_five_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(cmd: list[str], **_kw: object) -> FakeCompleted:
        del cmd
        return FakeCompleted(5, "", "")

    monkeypatch.setattr("scripts.helpers.common.pytest_runner._run_collect_only", _fake_run)
    assert filter_collectable_node_ids(("tests/regression/cli/test_a.py::test_a",), marker="not npu") == ()


def test_collect_test_node_ids_parses_node_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_stdout = "tests/smoke/test_a.py::test_foo\ntests/regression/test_b.py::test_bar\nother noise\n"
    monkeypatch.setattr(
        "scripts.helpers.common.pytest_runner.subprocess.run",
        lambda *a, **kw: FakeCompleted(0, fake_stdout, ""),
    )
    assert collect_test_node_ids(["tests/smoke", "tests/regression"], marker="not npu") == (
        "tests/smoke/test_a.py::test_foo",
        "tests/regression/test_b.py::test_bar",
    )


def test_collect_test_node_ids_empty_stdout_returns_empty_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.common.pytest_runner.subprocess.run",
        lambda *a, **kw: FakeCompleted(0, "", ""),
    )
    assert collect_test_node_ids(["tests/smoke"], marker="not npu") == ()


def test_collect_test_node_ids_exit_code_five_treated_as_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.common.pytest_runner.subprocess.run",
        lambda *a, **kw: FakeCompleted(5, "", ""),
    )
    assert collect_test_node_ids(["tests/smoke"], marker="not npu") == ()


def test_collect_test_node_ids_nonzero_exit_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.helpers._config import ConfigError

    monkeypatch.setattr(
        "scripts.helpers.common.pytest_runner.subprocess.run",
        lambda *a, **kw: FakeCompleted(1, "", "error"),
    )
    with pytest.raises(ConfigError, match="collect-only failed"):
        collect_test_node_ids(["tests/smoke"], marker="not npu")


def test_collect_test_node_ids_includes_ignore_addopts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def _fake_run(cmd: list[str], **_kw: object) -> FakeCompleted:
        captured.append(cmd)
        return FakeCompleted(0, "", "")

    monkeypatch.setattr("scripts.helpers.common.pytest_runner.subprocess.run", _fake_run)
    collect_test_node_ids(["tests/smoke"], marker="not npu")
    assert PYTEST_IGNORE_ADDOPTS[0] in captured[0]
    assert "-m" in captured[0]
    assert "not npu" in captured[0]


# ---------------------------------------------------------------------------
# count_collected_tests
# ---------------------------------------------------------------------------


def test_count_collected_tests_empty_targets_returns_zero() -> None:
    assert count_collected_tests([], marker="not npu") == 0


def test_count_collected_tests_parses_node_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_stdout = "tests/smoke/test_a.py::test_foo\ntests/regression/test_b.py::test_bar\nother noise\n"
    monkeypatch.setattr(
        "scripts.helpers.common.pytest_runner.subprocess.run",
        lambda *a, **kw: FakeCompleted(0, fake_stdout, ""),
    )
    assert count_collected_tests(["tests/smoke", "tests/regression"], marker="not npu") == 2


def test_count_collected_tests_empty_stdout_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.common.pytest_runner.subprocess.run",
        lambda *a, **kw: FakeCompleted(0, "", ""),
    )
    assert count_collected_tests(["tests/smoke"], marker="not npu") == 0


def test_count_collected_tests_exit_code_five_treated_as_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.common.pytest_runner.subprocess.run",
        lambda *a, **kw: FakeCompleted(5, "", ""),
    )
    assert count_collected_tests(["tests/smoke"], marker="not npu") == 0


def test_count_collected_tests_nonzero_exit_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.helpers._config import ConfigError

    monkeypatch.setattr(
        "scripts.helpers.common.pytest_runner.subprocess.run",
        lambda *a, **kw: FakeCompleted(1, "", "error"),
    )
    with pytest.raises(ConfigError, match="collect-only failed"):
        count_collected_tests(["tests/smoke"], marker="not npu")


def test_count_collected_tests_includes_ignore_addopts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def _fake_run(cmd: list[str], **_kw: object) -> FakeCompleted:
        captured.append(cmd)
        return FakeCompleted(0, "", "")

    monkeypatch.setattr("scripts.helpers.common.pytest_runner.subprocess.run", _fake_run)
    count_collected_tests(["tests/smoke"], marker="not npu")
    assert PYTEST_IGNORE_ADDOPTS[0] in captured[0]
    assert "-m" in captured[0]
    assert "not npu" in captured[0]


# ---------------------------------------------------------------------------
# build_pytest_cmd
# ---------------------------------------------------------------------------


def test_build_pytest_cmd_includes_ignore_addopts_and_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    cmd = build_pytest_cmd(
        "python",
        ["tests/smoke/"],
        marker="not npu",
        collected_count=1,
        extra_args=[],
    )
    assert cmd[0] == "python"
    assert "-m" in cmd
    assert "pytest" in cmd
    assert PYTEST_IGNORE_ADDOPTS[0] in cmd
    assert "not npu" in cmd
    assert "-vv" in cmd


def test_build_pytest_cmd_assembles_expected_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    cmd = build_pytest_cmd(
        "/usr/bin/python",
        ["tests/smoke", "tests/regression"],
        marker="not npu and not nightly",
        collected_count=2,
        extra_args=["--junit-xml=out.xml"],
    )
    assert cmd[:4] == ["/usr/bin/python", "-m", "pytest", "tests/smoke"]
    assert "tests/regression" in cmd
    assert PYTEST_IGNORE_ADDOPTS[0] in cmd
    assert "-m" in cmd
    assert "not npu and not nightly" in cmd
    assert "-n" in cmd
    assert "2" in cmd
    assert "--dist" in cmd
    assert "worksteal" in cmd
    assert "-vv" in cmd
    assert "--tb=short" in cmd
    assert "--durations=20" in cmd
    assert "--disable-warnings" in cmd
    assert "--junit-xml=out.xml" in cmd


def test_build_pytest_cmd_zero_collected_omits_xdist() -> None:
    cmd = build_pytest_cmd(
        "python",
        ["tests/smoke"],
        marker="not npu",
        collected_count=0,
        extra_args=[],
    )
    assert "-n" not in cmd
    assert "--dist" not in cmd
