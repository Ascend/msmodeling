"""Tests for nightly.report_builder — fetch_env_info, load_test_map_summary, compute_weak_coverage_symbols."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.helpers.nightly.pytest_parser import NightlyRunStats
from scripts.helpers.nightly.report_builder import (
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


def test_fetch_env_info_git_not_found_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="git not found"):
        fetch_env_info()


# ---------------------------------------------------------------------------
# load_test_map_summary
# ---------------------------------------------------------------------------


def test_load_test_map_summary_valid_file_counts_files_and_symbols(
    tmp_path: Path,
) -> None:
    path = tmp_path / "map.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "map": {
                    "a.py": {"fn1": ["t1"], "fn2": ["t2", "t3"]},
                    "b.py": {"fn3": ["t4"]},
                },
            }
        ),
        encoding="utf-8",
    )
    summary = load_test_map_summary(path)
    assert summary.source_files == 2
    assert summary.symbols == 3


def test_load_test_map_summary_none_path_returns_zeroes() -> None:
    summary = load_test_map_summary(None)
    assert summary.source_files == 0
    assert summary.symbols == 0


def test_load_test_map_summary_missing_file_returns_zeroes(tmp_path: Path) -> None:
    summary = load_test_map_summary(tmp_path / "nonexistent.json")
    assert summary.source_files == 0
    assert summary.symbols == 0


def test_load_test_map_summary_invalid_json_returns_zeroes(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text("{bad", encoding="utf-8")
    summary = load_test_map_summary(path)
    assert summary.source_files == 0


def test_load_test_map_summary_map_not_dict_returns_zeroes(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"schema_version": 1, "map": []}), encoding="utf-8")
    summary = load_test_map_summary(path)
    assert summary.source_files == 0


# ---------------------------------------------------------------------------
# compute_weak_coverage_symbols
# ---------------------------------------------------------------------------


def test_compute_weak_coverage_symbols_returns_empty_when_no_test_map(tmp_path: Path) -> None:
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

    in_memory = {"cli/main.py": {"run": ["test_a"]}}
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
        json.dumps({"schema_version": 1, "map": {"cli/main.py": {"run": ["test_a"]}}}),
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
