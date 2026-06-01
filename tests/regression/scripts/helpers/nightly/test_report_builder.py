"""Tests for nightly.report_builder — fetch_env_info, load_test_map_summary, compute_weak_coverage_symbols."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.helpers.nightly.report_builder import (
    compute_weak_coverage_symbols,
    fetch_env_info,
    load_test_map_summary,
)
from tests.helpers.fake_subprocess import FakeCompleted

# ---------------------------------------------------------------------------
# fetch_env_info
# ---------------------------------------------------------------------------


def test_fetch_env_info_returns_commit_and_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(cmd, **_kwargs):
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


def test_compute_weak_coverage_symbols_returns_empty_when_no_test_map() -> None:
    result = compute_weak_coverage_symbols(None, Path("/tmp/.coverage"))
    assert result == ()


def test_compute_weak_coverage_symbols_returns_empty_when_test_map_missing(
    tmp_path: Path,
) -> None:
    result = compute_weak_coverage_symbols(tmp_path / "nonexistent.json", tmp_path / ".coverage")
    assert result == ()


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
