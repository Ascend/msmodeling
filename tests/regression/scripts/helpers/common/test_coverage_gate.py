"""Tests for common.coverage_gate — GateConfig, check_thresholds, load_totals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.helpers.common.coverage_gate import GateConfig, check_thresholds, load_totals
from tests.helpers.fake_subprocess import FakeCompleted

# ---------------------------------------------------------------------------
# check_thresholds — pass
# ---------------------------------------------------------------------------


def test_check_thresholds_all_met_returns_empty(gate_config: GateConfig) -> None:
    failures = check_thresholds(80.0, 60.0, gate_config)
    assert failures == []


def test_check_thresholds_line_equals_threshold_passes(gate_config: GateConfig) -> None:
    failures = check_thresholds(70.0, 60.0, gate_config)
    assert failures == []


# ---------------------------------------------------------------------------
# check_thresholds — fail
# ---------------------------------------------------------------------------


def test_check_thresholds_line_below_reports_failure(gate_config: GateConfig) -> None:
    failures = check_thresholds(65.0, 60.0, gate_config)
    assert len(failures) == 1
    assert "line coverage 65.0% < 70.0%" in failures[0]


def test_check_thresholds_branch_below_reports_failure(gate_config: GateConfig) -> None:
    failures = check_thresholds(80.0, 45.0, gate_config)
    assert len(failures) == 1
    assert "branch coverage 45.0% < 50.0%" in failures[0]


def test_check_thresholds_both_below_reports_two_failures(
    gate_config: GateConfig,
) -> None:
    failures = check_thresholds(60.0, 40.0, gate_config)
    assert len(failures) == 2


# ---------------------------------------------------------------------------
# load_totals
# ---------------------------------------------------------------------------


_COVERAGE_JSON_STDOUT = json.dumps(
    {
        "totals": {
            "percent_covered_display": "85.0%",
            "num_branches": 10,
            "covered_branches": 8,
        },
    }
)


def test_load_totals_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / ".coverage"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_totals(missing)


@pytest.fixture(scope="module")
def coverage_data_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("cov") / ".coverage"
    path.write_text("", encoding="utf-8")
    return path


def test_load_totals_parses_subprocess_json(monkeypatch: pytest.MonkeyPatch, coverage_data_file: Path) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, _COVERAGE_JSON_STDOUT, ""))
    totals = load_totals(coverage_data_file)
    assert totals.line_percent == 85.0
    assert totals.branch_percent == 80.0


def test_load_totals_subprocess_failure_raises(monkeypatch: pytest.MonkeyPatch, coverage_data_file: Path) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(1, "", "coverage error"))
    with pytest.raises(RuntimeError, match="coverage json failed"):
        load_totals(coverage_data_file)
