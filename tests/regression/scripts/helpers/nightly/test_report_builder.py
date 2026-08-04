"""Tests for nightly.report_builder — fetch_env_info, load_test_map_summary, compute_weak_coverage_symbols."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.helpers.nightly.pytest_parser import NightlyRunStats
from scripts.helpers.nightly.report_builder import (
    _weak_symbols_for_file,
    build_phase_breakdown,
    compute_weak_coverage_symbols,
    fetch_env_info,
    load_test_map_summary,
    resolve_first_error,
)
from scripts.helpers.nightly.report_models import PhaseBreakdownEntry
from tests.helpers.fake_subprocess import FakeCompleted

# ---------------------------------------------------------------------------
# fetch_env_info
# ---------------------------------------------------------------------------


def test_fetch_env_info_returns_commit_and_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(cmd: list[str], **_kwargs: object) -> FakeCompleted:
        if "rev-parse" in cmd:
            return FakeCompleted(0, "abc1234\n", "")
        if "--show-current" in cmd:
            return FakeCompleted(0, "main\n", "")
        return FakeCompleted(1, "", "")

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/git")
    info = fetch_env_info()
    assert info.commit == "abc1234"
    assert info.branch == "main"
    assert len(info.timestamp) > 0


def test_fetch_env_info_returns_unknown_when_git_stdout_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scripts.helpers.nightly.report_builder.git_stdout",
        lambda *_args, **_kwargs: "",
    )
    info = fetch_env_info()
    assert info.commit == "unknown"
    assert info.branch == "unknown"


# ---------------------------------------------------------------------------
# load_test_map_summary
# ---------------------------------------------------------------------------


def test_load_test_map_summary_valid_file_counts_nodes_and_symbol_refs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "map.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "map": {
                    "tests/smoke/test_a.py::test_one": {
                        "cli/a.py": ["fn1"],
                        "cli/b.py": ["fn2", "fn3"],
                    },
                    "tests/smoke/test_b.py::test_two": {"cli/b.py": ["fn3"]},
                },
            }
        ),
        encoding="utf-8",
    )
    summary = load_test_map_summary(path)
    assert summary.test_nodes == 2
    assert summary.symbol_refs == 3


def test_load_test_map_summary_rejects_source_oriented_map(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "map": {
                    "a.py": {"fn1": ["tests/smoke/test_a.py::test_one"]},
                    "b.py": {"fn3": ["tests/smoke/test_b.py::test_two"]},
                },
            }
        ),
        encoding="utf-8",
    )
    summary = load_test_map_summary(path)
    assert summary.test_nodes == 0
    assert summary.symbol_refs == 0


def test_load_test_map_summary_none_path_returns_zeroes() -> None:
    summary = load_test_map_summary(None)
    assert summary.test_nodes == 0
    assert summary.symbol_refs == 0


def test_load_test_map_summary_missing_file_returns_zeroes(tmp_path: Path) -> None:
    summary = load_test_map_summary(tmp_path / "nonexistent.json")
    assert summary.test_nodes == 0
    assert summary.symbol_refs == 0


def test_load_test_map_summary_invalid_json_returns_zeroes(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text("{bad", encoding="utf-8")
    summary = load_test_map_summary(path)
    assert summary.test_nodes == 0


def test_load_test_map_summary_map_not_dict_returns_zeroes(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"schema_version": 1, "map": []}), encoding="utf-8")
    summary = load_test_map_summary(path)
    assert summary.test_nodes == 0


# ---------------------------------------------------------------------------
# _weak_symbols_for_file
# ---------------------------------------------------------------------------


class _FakeCoverageData:
    def __init__(self, ctxmap: dict[int, list[str]]) -> None:
        self._ctxmap = ctxmap

    def contexts_by_lineno(self, path: str) -> dict[int, list[str]]:
        del path
        return self._ctxmap


def test_weak_symbols_for_file_flags_symbols_below_threshold(tmp_path: Path) -> None:
    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text("def run():\n    x = 1\n    y = 2\n", encoding="utf-8")
    coverage_data = _FakeCoverageData({2: ["tests/foo.py::test_bar"]})
    weak = _weak_symbols_for_file(
        "cli/main.py",
        src.resolve(),
        coverage_data,
        symbols={"run"},
        threshold=0.5,
    )
    assert weak == ["cli/main.py::run"]


def test_weak_symbols_for_file_skips_symbols_above_threshold(tmp_path: Path) -> None:
    src = tmp_path / "cli" / "main.py"
    src.parent.mkdir(parents=True)
    src.write_text("def run():\n    x = 1\n", encoding="utf-8")
    coverage_data = _FakeCoverageData({1: ["t"], 2: ["t"]})
    weak = _weak_symbols_for_file(
        "cli/main.py",
        src.resolve(),
        coverage_data,
        symbols={"run"},
        threshold=0.5,
    )
    assert weak == []


# ---------------------------------------------------------------------------
# compute_weak_coverage_symbols
# ---------------------------------------------------------------------------


def test_compute_weak_coverage_symbols_returns_empty_when_no_test_map(
    tmp_path: Path,
) -> None:
    result = compute_weak_coverage_symbols(None, tmp_path / ".coverage")
    assert result == ()


def test_compute_weak_coverage_symbols_returns_empty_when_test_map_missing(
    tmp_path: Path,
) -> None:
    result = compute_weak_coverage_symbols(tmp_path / "nonexistent.json", tmp_path / ".coverage")
    assert result == ()


def test_compute_weak_coverage_symbols_uses_in_memory_mapping_without_file_read(
    tmp_path: Path,
) -> None:
    from scripts.helpers.nightly import report_builder

    in_memory = {"tests/regression/cli/test_run.py::test_run": {"cli/main.py": ["run"]}}
    monkeypatch_local = pytest.MonkeyPatch()
    monkeypatch_local.setattr(report_builder, "REPO_ROOT", tmp_path)
    (tmp_path / "cli").mkdir()
    (tmp_path / "cli" / "main.py").write_text("def run():\n    x = 1\n", encoding="utf-8")
    try:
        result = compute_weak_coverage_symbols(
            tmp_path / "missing-map.json",
            tmp_path / ".coverage",
            mapping=in_memory,
        )
        assert result == ()
    finally:
        monkeypatch_local.undo()


def test_compute_weak_coverage_symbols_returns_empty_when_no_coverage_data(
    tmp_path: Path,
) -> None:
    from scripts.helpers.nightly import report_builder

    map_path = tmp_path / "map.json"
    map_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "map": {"tests/foo.py::test_a": {"cli/main.py": ["run"]}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "cli").mkdir()
    (tmp_path / "cli" / "main.py").write_text("def run():\n    x = 1\n", encoding="utf-8")

    monkeypatch_local = pytest.MonkeyPatch()
    monkeypatch_local.setattr(report_builder, "REPO_ROOT", tmp_path)
    try:
        result = compute_weak_coverage_symbols(map_path, tmp_path / ".coverage")
        assert result == ()
    finally:
        monkeypatch_local.undo()


# ---------------------------------------------------------------------------
# build_phase_breakdown / resolve_first_error
# ---------------------------------------------------------------------------


def test_build_phase_breakdown_marks_missing_junit_with_infra_failure(
    tmp_path: Path,
) -> None:
    entries = build_phase_breakdown(
        ("smoke UT (coverage mapping)",),
        (tmp_path / "missing.xml",),
        (1,),
    )
    assert entries == (
        PhaseBreakdownEntry(
            label="smoke UT (coverage mapping)",
            passed=0,
            failed=0,
            duration_sec=-1.0,
            exit_code=1,
            infra_failure=True,
        ),
    )


def test_resolve_first_error_falls_back_to_phase_log(tmp_path: Path) -> None:
    log_path = tmp_path / "phase1.log"
    log_path.write_text("collecting ...\nE   ValueError: duplicate config name\n", encoding="utf-8")
    stats = NightlyRunStats(
        passed=0,
        failed=0,
        errors=0,
        duration_sec=-1.0,
        failed_cases=(),
        first_error="",
    )
    assert resolve_first_error(stats, (1,), (log_path,)) == "ValueError: duplicate config name"
