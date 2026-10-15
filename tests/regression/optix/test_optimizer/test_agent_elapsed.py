# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Round progress retains trial time across failures, retries and process restarts."""

import contextlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from optix.config.config import PerformanceIndex
from optix.optimizer.agentic import strategies
from optix.optimizer.agentic.strategies import AgentCandidateStrategy


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(strategies.time, "monotonic", lambda: now[0])
    return now


def run_attempt(run_dir, clock, seconds=7, *, fail=False, round_index=1, count=1):
    candidates = [{"candidate_id": f"candidate-{i}", "params": {}} for i in range(count)]
    (run_dir / f"candidates.round-{round_index}.json").write_text(
        json.dumps({"round": round_index, "candidates": candidates}), encoding="utf-8"
    )
    scheduler = Mock(error_info="", last_outcome=None, performance_index=None)
    scheduler.disable_early_exit.return_value = contextlib.nullcontext()

    def execute(*args):
        clock[0] += seconds
        if fail:
            raise RuntimeError("trial failed")
        return PerformanceIndex(generate_speed=100, success_rate=1)

    scheduler.run.side_effect = execute
    result = AgentCandidateStrategy(run_dir, round_index=round_index).run(
        scheduler, [], {"search_space": {"parameters": [], "constants": []}}
    )
    progress = json.loads((run_dir / f"progress.round-{round_index}.json").read_text(encoding="utf-8"))
    return result, progress


def test_failed_attempt_and_restart_retain_37_seconds(tmp_path, clock, monkeypatch):
    result, progress = run_attempt(tmp_path, clock, 30, fail=True)
    assert result.trials[0].error == "trial failed"
    assert progress["cumulative_elapsed_s"] == 30
    assert "elapsed_complete" not in progress
    snapshots = []
    write_progress = strategies._write_progress

    def capture(path, data):
        snapshots.append(dict(data))
        write_progress(path, data)

    monkeypatch.setattr(strategies, "_write_progress", capture)
    clock[0] = 0  # A new process has its own monotonic origin.
    result, progress = run_attempt(tmp_path, clock, 7)
    assert result.trials[0].error is None
    assert [p["cumulative_elapsed_s"] for p in snapshots] == [30, 30, 37, 37]
    assert all("elapsed_complete" not in p for p in snapshots)
    assert progress["trials"][0]["elapsed_s"] == 7
    assert progress["cumulative_elapsed_s"] == 37
    assert "elapsed_complete" not in progress
    _, progress = run_attempt(tmp_path, clock, 5)
    assert progress["cumulative_elapsed_s"] == 42


def test_trials_are_counted_once_and_rounds_are_independent(tmp_path, clock):
    _, first = run_attempt(tmp_path, clock, 3, count=2)
    _, second = run_attempt(tmp_path, clock, 4, round_index=2)
    assert first["cumulative_elapsed_s"] == 6
    assert second["cumulative_elapsed_s"] == 4
    assert sum(p["cumulative_elapsed_s"] for p in (first, second)) == 10
    _, empty = run_attempt(tmp_path, clock, count=0)
    assert empty["cumulative_elapsed_s"] == 6


def test_interrupted_trial_preserves_previously_recorded_time(tmp_path, clock, monkeypatch):
    run_attempt(tmp_path, clock, 30, fail=True)
    with monkeypatch.context() as patch:
        patch.setattr(AgentCandidateStrategy, "_run_trial", Mock(side_effect=KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            run_attempt(tmp_path, clock)
    path = tmp_path / "progress.round-1.json"
    interrupted = json.loads(path.read_text(encoding="utf-8"))
    assert interrupted["cumulative_elapsed_s"] == 30
    assert "elapsed_complete" not in interrupted
    _, retried = run_attempt(tmp_path, clock, 7)
    assert retried["cumulative_elapsed_s"] == 37
    assert "elapsed_complete" not in retried


def test_legacy_snapshot_preserves_recorded_time(tmp_path, clock):
    path = tmp_path / "progress.round-1.json"
    path.write_text(
        json.dumps({"round": 1, "stage": "round_done", "trials": [{"status": "done", "elapsed_s": 30}]}),
        encoding="utf-8",
    )
    _, progress = run_attempt(tmp_path, clock, 7)
    assert progress["cumulative_elapsed_s"] == 37
    assert "elapsed_complete" not in progress


@pytest.mark.parametrize("value", [-1, "30", True, float("inf"), float("nan")])
def test_invalid_history_is_not_overwritten(tmp_path, clock, value):
    path = tmp_path / "progress.round-1.json"
    path.write_text(json.dumps({"round": 1, "trials": [], "cumulative_elapsed_s": value}), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="elapsed time"):
        run_attempt(tmp_path, clock)
    assert path.read_bytes() == before
    assert clock[0] == 0


def test_missing_progress_with_existing_results_is_not_reset(tmp_path, clock):
    (tmp_path / "results.round-1.json").write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError, match="Missing elapsed-time history"):
        run_attempt(tmp_path, clock)
    assert not (tmp_path / "progress.round-1.json").exists()
    assert clock[0] == 0


def test_failed_atomic_write_preserves_budget_and_prevents_execution(tmp_path, clock, monkeypatch):
    run_attempt(tmp_path, clock, 30, fail=True)
    path = tmp_path / "progress.round-1.json"
    before = path.read_bytes()
    files_before = set(tmp_path.iterdir())
    monkeypatch.setattr(Path, "replace", Mock(side_effect=OSError("write failed")))
    with pytest.raises(OSError, match="write failed"):
        run_attempt(tmp_path, clock, 7)
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == files_before
    assert clock[0] == 30


def test_failure_to_persist_finished_trial_preserves_previous_total(tmp_path, clock, monkeypatch):
    run_attempt(tmp_path, clock, 30, fail=True)
    write_progress = strategies._write_progress

    def fail_after_trial(path, data):
        if data["stage"] == "trial_completed":
            raise OSError("cannot save trial time")
        write_progress(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(strategies, "_write_progress", fail_after_trial)
        with pytest.raises(OSError, match="cannot save trial time"):
            run_attempt(tmp_path, clock, 7)
    progress = json.loads((tmp_path / "progress.round-1.json").read_text(encoding="utf-8"))
    assert progress["cumulative_elapsed_s"] == 30
    assert "elapsed_complete" not in progress
    _, retry = run_attempt(tmp_path, clock, 5)
    assert retry["cumulative_elapsed_s"] == 35  # Known time only; the lost 7 seconds remain a gap.
    assert "elapsed_complete" not in retry


@pytest.mark.parametrize("content", ['{', '[]', '{"round":2,"trials":[]}'])
def test_corrupt_progress_stops_before_trial_and_is_preserved(tmp_path, clock, content):
    path = tmp_path / "progress.round-1.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        run_attempt(tmp_path, clock)
    assert path.read_text(encoding="utf-8") == content
    assert clock[0] == 0
