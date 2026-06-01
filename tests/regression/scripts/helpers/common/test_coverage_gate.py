"""Tests for common.coverage_gate — GateConfig, check_thresholds, check_ut_gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.helpers.common.coverage_gate import GateConfig, check_thresholds, check_ut_gate
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
# check_ut_gate — no data file
# ---------------------------------------------------------------------------


def test_check_ut_gate_missing_file_returns_false(tmp_path: Path, gate_config: GateConfig) -> None:
    missing = tmp_path / ".coverage"
    passed, message = check_ut_gate(missing, config=gate_config)
    assert passed is False
    assert f"not found: {missing}" in message


# ---------------------------------------------------------------------------
# check_ut_gate — with mocked subprocess
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


@pytest.fixture(scope="module")
def coverage_data_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("cov") / ".coverage"
    path.write_text("", encoding="utf-8")
    return path


def test_check_ut_gate_passes_when_thresholds_met(
    monkeypatch: pytest.MonkeyPatch, coverage_data_file: Path, gate_config: GateConfig
) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, _COVERAGE_JSON_STDOUT, ""))
    passed, message = check_ut_gate(coverage_data_file, config=gate_config)
    assert passed is True
    assert "passed" in message


def test_check_ut_gate_fails_when_below_threshold(
    monkeypatch: pytest.MonkeyPatch, coverage_data_file: Path, gate_config: GateConfig
) -> None:
    low = json.dumps(
        {
            "totals": {
                "percent_covered_display": "60.0%",
                "num_branches": 10,
                "covered_branches": 4,
            },
        }
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(0, low, ""))
    passed, message = check_ut_gate(coverage_data_file, config=gate_config)
    assert passed is False
    assert "line coverage 60.0% < 70.0%" in message


def test_check_ut_gate_subprocess_failure_returns_false(
    monkeypatch: pytest.MonkeyPatch, coverage_data_file: Path, gate_config: GateConfig
) -> None:
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeCompleted(1, "", "coverage error"))
    passed, _ = check_ut_gate(coverage_data_file, config=gate_config)
    assert passed is False
