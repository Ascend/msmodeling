"""Tests for common.coverage_symbol_check."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.helpers.common.coverage_symbol_check import symbol_lines_covered_in_data


class _FakeCoverageData:
    def __init__(
        self,
        _path: str,
        *,
        measured: str,
        ctxmap: dict[int, list[str]],
        executed_lines: list[int] | None = None,
    ) -> None:
        self._measured = measured
        self._ctxmap = ctxmap
        self._executed_lines = executed_lines or []

    def read(self) -> None:
        return None

    def measured_files(self) -> list[str]:
        return [self._measured]

    def contexts_by_lineno(self, _path: str) -> dict[int, list[str]]:
        return self._ctxmap

    def lines(self, _path: str) -> list[int]:
        return self._executed_lines


def test_symbol_lines_covered_true_for_empty_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    source = repo / "cli" / "main.py"
    source.parent.mkdir(parents=True)
    source.write_text("def run():\n    return 1\n", encoding="utf-8")
    coverage_path = repo / ".coverage"
    coverage_path.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "coverage.data.CoverageData",
        lambda _path: _FakeCoverageData(
            _path,
            measured=str(source.resolve()),
            ctxmap={2: [""]},
        ),
    )

    assert symbol_lines_covered_in_data(repo, "cli/main.py", "run", {2}, coverage_path) is True


def test_symbol_lines_covered_false_when_line_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    source = repo / "cli" / "main.py"
    source.parent.mkdir(parents=True)
    source.write_text("def run():\n    return 1\n", encoding="utf-8")
    coverage_path = repo / ".coverage"
    coverage_path.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "coverage.data.CoverageData",
        lambda _path: _FakeCoverageData(
            _path,
            measured=str(source.resolve()),
            ctxmap={3: ["tests/regression/cli/test_a.py::test_x"]},
        ),
    )

    assert symbol_lines_covered_in_data(repo, "cli/main.py", "run", {2}, coverage_path) is False


def test_symbol_lines_covered_false_when_coverage_missing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    assert symbol_lines_covered_in_data(repo, "cli/main.py", "run", {1}, repo / ".coverage") is False


def test_symbol_lines_covered_true_when_executed_lines_without_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "cli" / "main.py"
    source.parent.mkdir(parents=True)
    source.write_text("def run():\n    return 1\n", encoding="utf-8")
    coverage_path = repo / ".coverage"
    coverage_path.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "coverage.data.CoverageData",
        lambda _path: _FakeCoverageData(
            _path,
            measured=str(source.resolve()),
            ctxmap={},
            executed_lines=[2],
        ),
    )

    assert symbol_lines_covered_in_data(repo, "cli/main.py", "run", {2}, coverage_path) is True
