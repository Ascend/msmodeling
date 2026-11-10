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

"""concurrency_sweep candidate type: validation, level selection, SLO rules and sweep execution."""

import contextlib
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from optix.config.config import PerformanceIndex
from optix.optimizer.agentic.candidates import Candidate
from optix.optimizer.agentic.strategies import (
    AgentCandidateStrategy,
    default_concurrency_levels,
    select_best_sweep_level,
)
from optix.optimizer.agentic.validation import validate_candidate
from optix.optimizer.plugins.benchmark import AisBench
from optix.optimizer.scheduler import ConcurrencySweepError, Scheduler


def entry(name, default, dtype="int", minimum=0, maximum=128, **kw):
    return dict(name=name, default=default, dtype=dtype, min=minimum, max=maximum, config_position="run") | kw


def context(*fields):
    return {"engine": "vllm", "search_space": {"parameters": list(fields), "constants": []}, "constraints": {}}


def sweep_context(minimum=1, maximum=16):
    return context(
        entry("MAX_NUM_SEQS", 16),
        entry("CONCURRENCY", 8, minimum=minimum, maximum=maximum),
        entry("REQUESTRATE", 0, "float", minimum=0.0, maximum=100.0),
    )


def sweep_candidate(levels="default", **params):
    c = {"candidate_id": "r1-sweep", "type": "concurrency_sweep", "params": params}
    return {**c, "concurrency_levels": levels} if levels != "default" else c


def test_sweep_candidate_rejects_managed_params_in_params():
    for managed in ({"CONCURRENCY": 8}, {"REQUESTRATE": 10}):
        result = validate_candidate(sweep_context(), sweep_candidate([2, 4, 8], **managed))
        assert not result["valid"]
        assert any("concurrency_levels" in error for error in result["errors"])


def test_sweep_candidate_levels_optional_and_service_params_validated():
    result = validate_candidate(sweep_context(), sweep_candidate(MAX_NUM_SEQS=32))
    assert result["valid"], result["errors"]  # levels omitted -> executor generates defaults
    assert result["params"]["MAX_NUM_SEQS"] == 32
    result = validate_candidate(sweep_context(), sweep_candidate([2, 4, 8, 16], MAX_NUM_SEQS=32))
    assert result["valid"], result["errors"]
    result = validate_candidate(sweep_context(), sweep_candidate([2, 4, 8], unknown_param=1))
    assert any("Unknown parameter" in error for error in result["errors"])
    result = validate_candidate(sweep_context(), sweep_candidate([2, 4, 8], MAX_NUM_SEQS=4096))
    assert any("outside range" in error for error in result["errors"])


@pytest.mark.parametrize(
    "levels, message",
    [
        ([8], "2~8 levels"),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], "2~8 levels"),
        ([4, 2, 8], "strictly ascending"),
        ([2, 2, 4], "strictly ascending"),
        ([2, 32], "outside CONCURRENCY range"),
        (["2", "4"], "list of integers"),
        ([2.5, 4], "list of integers"),
        ([True, False], "list of integers"),
        ("not-a-list", "list of integers"),
    ],
)
def test_sweep_candidate_level_list_checks(levels, message):
    result = validate_candidate(sweep_context(), sweep_candidate(levels, MAX_NUM_SEQS=32))
    assert not result["valid"]
    assert any(message in error for error in result["errors"])


def test_sweep_candidate_requires_sweepable_concurrency_range():
    result = validate_candidate(context(entry("MAX_NUM_SEQS", 16)), sweep_candidate([2, 4, 8]))
    assert any("requires CONCURRENCY" in error for error in result["errors"])
    result = validate_candidate(sweep_context(minimum=8, maximum=8), sweep_candidate([2, 4, 8]))
    assert any("min == max" in error for error in result["errors"])
    # regular (non-sweep) candidates are unaffected by sweep rules
    result = validate_candidate(sweep_context(), {"candidate_id": "r1-a", "params": {"CONCURRENCY": 8}})
    assert result["valid"], result["errors"]


def test_candidate_sweep_fields_round_trip():
    data = dict(
        candidate_id="r1-sweep",
        direction_id="d1-sweep",
        type="concurrency_sweep",
        params={"MAX_NUM_SEQS": 32},
        concurrency_levels=[2, 4, 8],
    )
    candidate = Candidate.from_dict(data)
    assert candidate.candidate_type == "concurrency_sweep"
    assert candidate.trial_type == "optimization"
    emitted = candidate.to_dict()
    assert emitted["type"] == "concurrency_sweep"
    assert "trial_type" not in emitted
    assert emitted["direction_id"] == "d1-sweep"
    assert emitted["concurrency_levels"] == [2, 4, 8]
    legacy = Candidate.from_dict({"candidate_id": "r1-a", "params": {}})
    assert legacy.candidate_type == legacy.trial_type == "optimization"
    assert legacy.direction_id is None and legacy.concurrency_levels is None
    assert "type" not in legacy.to_dict() and "trial_type" not in legacy.to_dict()
    restored = Candidate.from_dict({**data, "trial_type": "diagnosis"})
    assert restored.candidate_type == "concurrency_sweep"
    assert restored.trial_type == "diagnosis"


def test_default_concurrency_levels():
    assert default_concurrency_levels(sweep_context(minimum=1, maximum=100)) == [1, 3, 10, 32, 100]
    assert default_concurrency_levels(sweep_context(minimum=2, maximum=4)) == [2, 3, 4]
    with pytest.raises(ValueError, match="requires CONCURRENCY"):
        default_concurrency_levels(context(entry("MAX_NUM_SEQS", 16)))
    with pytest.raises(ValueError, match="no room"):
        default_concurrency_levels(sweep_context(minimum=8, maximum=8))


def sweep_result(level, speed, tpot=None):
    perf = PerformanceIndex(generate_speed=speed, success_rate=1)
    if tpot is not None:
        perf.time_per_output_token = tpot
    return {"concurrency": level, "performance": perf}


def slo_evaluator(tpot_slo=None, ttft_slo=None):
    evaluator = Mock(tpot_slo=tpot_slo, ttft_slo=ttft_slo)
    evaluator.minimum_algorithm.side_effect = lambda perf: perf.generate_speed
    return evaluator


@pytest.mark.parametrize(
    "results, mode, expected",
    [
        ([sweep_result(2, 100.0), sweep_result(8, 250.0), sweep_result(16, 200.0)], "none", 8),
        ([sweep_result(8, 200.0), sweep_result(16, 203.0)], "none", 8),  # plateau -> lower level
        ([sweep_result(4, 196.0), sweep_result(8, 200.0), sweep_result(16, 201.0)], "none", 4),
        ([sweep_result(8, 200.0), sweep_result(16, 220.0)], "none", 16),  # clear gain -> higher level
        ([sweep_result(8, 200.0, 0.018), sweep_result(16, 260.0, 0.026)], "slo", 8),  # compliant > faster violator
        ([sweep_result(8, 200.0, 0.030), sweep_result(16, 260.0, 0.045)], "slo", 8),  # all violated -> lowest TPOT
        ([sweep_result(8, 200.0, 0.019), sweep_result(16, 203.0, 0.025)], "slo", 8),  # SLO filter precedes plateau
        ([sweep_result(8, 200.0, 0.018), sweep_result(16, 260.0, 0.026)], "no_slo", 16),  # no SLO -> throughput rule
    ],
)
def test_select_best_sweep_level(results, mode, expected):
    evaluator = {"none": None, "slo": slo_evaluator(tpot_slo=0.02), "no_slo": slo_evaluator()}[mode]
    assert select_best_sweep_level(results, fitness_evaluator=evaluator)["concurrency"] == expected


def test_select_best_sweep_level_skips_failed_levels():
    results = [sweep_result(2, 100.0), {"concurrency": 8, "error": "timeout"}, {"concurrency": 16, "performance": None}]
    assert select_best_sweep_level(results)["concurrency"] == 2
    assert select_best_sweep_level([{"concurrency": 8, "error": "boom"}]) is None
    assert select_best_sweep_level([]) is None


# --- Sweep trial execution ---


def sweep_validation(levels=(2, 4, 8)):
    candidate = {"candidate_id": "r1-sweep", "type": "concurrency_sweep", "concurrency_levels": list(levels)}
    return {"candidate_id": "r1-sweep", "params": {}, "candidate": candidate}


def sweep_scheduler(results_or_error):
    scheduler = Mock(error_info="", last_outcome=None, performance_index=None)
    scheduler.disable_early_exit.return_value = contextlib.nullcontext()
    setattr(
        scheduler.run_concurrency_sweep,
        "side_effect" if isinstance(results_or_error, Exception) else "return_value",
        results_or_error,
    )
    return scheduler


def test_run_sweep_trial_success_paths(tmp_path):
    scheduler = sweep_scheduler([sweep_result(2, 150.0), sweep_result(4, 220.0), sweep_result(8, 210.0)])
    trial = AgentCandidateStrategy(tmp_path)._run_sweep_trial(sweep_validation(), scheduler, [], sweep_context())
    assert trial.error is None and trial.fitness == 220.0
    assert trial.params["CONCURRENCY"] == 4 and trial.performance.generate_speed == 220.0
    assert trial.metadata["best_concurrency"] == 4
    assert trial.metadata["trial_kind"] == "concurrency_sweep" and trial.metadata["best_metric"] == "generate_speed"
    assert [e["concurrency"] for e in trial.metadata["sweep"]] == [2, 4, 8]
    scheduler.save_result.assert_called_once()
    results = [sweep_result(2, 150.0), {"concurrency": 4, "error": "benchmark timeout"}, sweep_result(8, 210.0)]
    scheduler = sweep_scheduler(results)
    trial = AgentCandidateStrategy(tmp_path)._run_sweep_trial(sweep_validation(), scheduler, [], sweep_context())
    assert trial.error is None and trial.params["CONCURRENCY"] == 8
    assert trial.metadata["sweep"][1]["error"] == "benchmark timeout"
    assert "performance" not in trial.metadata["sweep"][1]


def test_run_sweep_trial_failure_paths(tmp_path):
    results = [sweep_result(2, 150.0), {"concurrency": 4, "error": "boom"}, {"concurrency": 8, "error": "boom"}]
    scheduler = sweep_scheduler(results)
    trial = AgentCandidateStrategy(tmp_path)._run_sweep_trial(sweep_validation(), scheduler, [], sweep_context())
    assert "1 of 3" in trial.error and trial.fitness == float("inf")
    scheduler = sweep_scheduler(ConcurrencySweepError("simulator process exited", [sweep_result(2, 150.0)]))
    trial = AgentCandidateStrategy(tmp_path)._run_sweep_trial(sweep_validation(), scheduler, [], sweep_context())
    assert trial.error == "simulator process exited" and trial.fitness == float("inf")
    assert trial.failure.category == "service"
    assert [e["concurrency"] for e in trial.metadata["sweep"]] == [2]
    assert trial.metadata["sweep"][0]["performance"]["generate_speed"] == 150.0
    scheduler.save_result.assert_called_once()


def archive_scheduler(tmp_path, monkeypatch, *, with_backup):
    """Real Scheduler with mocked simulator/benchmark and settings.output at tmp_path."""
    monkeypatch.setattr("optix.optimizer.scheduler.get_settings", lambda: Mock(output=tmp_path, wait_start_time=1))
    output_dir = tmp_path / "ais_bench"
    (output_dir / "run1").mkdir(parents=True)
    (output_dir / "run1" / "result.csv").write_text("col\n1\n", encoding="utf-8")
    benchmark = Mock()
    benchmark.config.output_path = output_dir
    scheduler = Scheduler(Mock(), benchmark, Mock(), bak_path=(tmp_path / "back_up") if with_backup else None)
    if with_backup:
        scheduler.current_back_path = tmp_path / "back_up" / "default_001" / "1"
    return scheduler, output_dir


@pytest.mark.parametrize("with_backup", [False, True])
def test_archive_sweep_level_targets(tmp_path, monkeypatch, with_backup):
    scheduler, output_dir = archive_scheduler(tmp_path, monkeypatch, with_backup=with_backup)
    scheduler.set_backup_phase("default", 1)
    scheduler._archive_sweep_level("r1-sweep", 4)
    scheduler._archive_sweep_level("r1-sweep", 8)
    base = tmp_path / "back_up" / "default_001" / "1"
    for level in (4, 8):
        assert (base / "sweep" / "r1-sweep" / f"c{level}" / output_dir.name / "run1" / "result.csv").exists()
    assert not (tmp_path / "back_up" / "default_001" / "2").exists()
    assert (scheduler._sweep_fallback_back_path is not None) != with_backup


def test_archive_sweep_level_skips_when_output_missing(tmp_path, monkeypatch):
    scheduler, _ = archive_scheduler(tmp_path, monkeypatch, with_backup=False)
    scheduler.benchmark.config.output_path = tmp_path / "nonexistent"
    scheduler._archive_sweep_level("r1-sweep", 4)
    assert not (tmp_path / "back_up").exists()


def test_run_concurrency_sweep_executes_real_scheduler_loop(tmp_path, monkeypatch):
    scheduler, _ = archive_scheduler(tmp_path, monkeypatch, with_backup=False)
    fields = [SimpleNamespace(name="CONCURRENCY", value=0), SimpleNamespace(name="REQUESTRATE", value=1, min=1, max=50)]
    monkeypatch.setattr("optix.optimizer.scheduler.map_param_with_value", lambda *_: fields)
    monkeypatch.setattr("optix.optimizer.scheduler.time.sleep", lambda _: None)
    scheduler.simulator = SimpleNamespace(process=SimpleNamespace(poll=lambda: None))
    for name in [
        "set_back_up_path",
        "update_data_field",
        "run_target_server",
        "monitoring_status",
        "_configure_early_exit_controller",
        "_reset_early_exit_window",
        "_archive_sweep_level",
        "_attach_warmup_summary",
    ]:
        setattr(scheduler, name, Mock())
    scheduler._new_case_id = Mock(return_value="case-1")
    scheduler._attach_metrics_window_reference = Mock(side_effect=lambda perf: perf)
    scheduler._merge_report_only_early_exit = Mock(side_effect=lambda perf: perf)
    scheduler.benchmark.get_performance_index.side_effect = [
        PerformanceIndex(generate_speed=100),
        ValueError("benchmark timeout"),
        PerformanceIndex(generate_speed=200),
    ]
    scheduler.benchmark.get_measured_concurrency.side_effect = [2, 8]

    results = scheduler.run_concurrency_sweep(Mock(), Mock(), [2, 4, 8], candidate_id="r1-sweep")

    assert [result["concurrency"] for result in results] == [2, 4, 8]
    assert results[1]["error"] == "benchmark timeout"
    assert results[2]["measured_concurrency"] == 8
    assert (fields[0].value, fields[1].value) == (8, 50)
    assert (scheduler.benchmark.stop.call_count, scheduler.benchmark.run.call_count) == (2, 2)


def test_ais_bench_reads_measured_concurrency(tmp_path):
    result_dir = tmp_path / "run" / "performances" / "summary"
    result_dir.mkdir(parents=True)
    (result_dir / "result.csv").write_text("metric,value\n", encoding="utf-8")
    (result_dir / "result.json").write_text(json.dumps({"Concurrency": {"total": 12}}), encoding="utf-8")
    benchmark = object.__new__(AisBench)
    benchmark.config = SimpleNamespace(output_path=str(tmp_path))
    assert benchmark.get_measured_concurrency() == 12.0


def test_sweep_hit_slo_cliff():
    hit = Scheduler._sweep_hit_slo_cliff
    assert hit(PerformanceIndex(time_per_output_token=0.025), 0.02, None, 1.2)
    assert not hit(PerformanceIndex(time_per_output_token=0.023), 0.02, None, 1.2)
    assert hit(PerformanceIndex(time_to_first_token=1.3), None, 1.0, 1.2)
    assert not hit(None, 0.02, None, 1.2)
    assert not hit(PerformanceIndex(generate_speed=100), 0.02, None, 1.2)
    assert not hit(PerformanceIndex(time_per_output_token=0.05), None, None, 1.2)


def test_run_sweep_trial_forwards_early_stop_thresholds(tmp_path):
    scheduler = sweep_scheduler([sweep_result(2, 150.0), sweep_result(4, 220.0)])
    evaluator = slo_evaluator(tpot_slo=0.02, ttft_slo=1.0)
    strategy = AgentCandidateStrategy(tmp_path, fitness_evaluator=evaluator)
    trial = strategy._run_sweep_trial(sweep_validation(levels=(2, 4)), scheduler, [], sweep_context())
    assert trial.error is None
    kwargs = scheduler.run_concurrency_sweep.call_args.kwargs
    assert kwargs["early_stop_tpot"] == 0.02
    assert kwargs["early_stop_ttft"] == 1.0


def test_run_sweep_trial_marks_skipped_levels_and_excludes_them_from_min_check(tmp_path):
    results = [
        sweep_result(2, 150.0),
        sweep_result(4, 220.0),
        {"concurrency": 8, "skipped": "slo_cliff"},
        {"concurrency": 16, "skipped": "slo_cliff"},
    ]
    scheduler = sweep_scheduler(results)
    trial = AgentCandidateStrategy(tmp_path)._run_sweep_trial(
        sweep_validation(levels=(2, 4, 8, 16)), scheduler, [], sweep_context()
    )
    assert trial.error is None
    assert trial.metadata["sweep"][2]["skipped"] == "slo_cliff"
    assert "performance" not in trial.metadata["sweep"][2]
    assert trial.params["CONCURRENCY"] == 4


def test_backfill_cliff_constant_and_startup_failure_fixes(tmp_path, monkeypatch):
    results = [sweep_result(2, 100.0), sweep_result(4, 200.0), sweep_result(8, 120.0)]
    scheduler = sweep_scheduler(results)
    order = []
    scheduler.backfill_sweep_best = Mock(side_effect=lambda *a: order.append("b"))
    scheduler.save_result = Mock(side_effect=lambda **k: order.append("s"))
    assert (
        AgentCandidateStrategy(tmp_path)._run_sweep_trial(sweep_validation(), scheduler, [], sweep_context()).fitness
        == 200.0
    )
    scheduler.backfill_sweep_best.assert_called_once_with(results[1]["performance"], 4)
    assert order == ["b", "s"]
    r = [
        sweep_result(2, 100.0, tpot=0.030),
        {"concurrency": 4, "skipped": "slo_cliff"},
        {"concurrency": 8, "skipped": "slo_cliff"},
    ]
    t = AgentCandidateStrategy(tmp_path, fitness_evaluator=slo_evaluator(tpot_slo=0.02))._run_sweep_trial(
        sweep_validation(levels=(2, 4, 8)), sweep_scheduler(r), [], sweep_context()
    )
    assert t.error is None and t.fitness == 100.0 and t.slo_violated is True
    ctx = context(entry("MAX_NUM_SEQS", 16), entry("CONCURRENCY", 8, minimum=1, maximum=64, constant=8))
    assert not validate_candidate(ctx, sweep_candidate([2, 4, 8, 16, 32]))["valid"]
    scheduler, _ = archive_scheduler(tmp_path, monkeypatch, with_backup=False)
    monkeypatch.setattr(
        "optix.optimizer.scheduler.map_param_with_value", lambda *_: [SimpleNamespace(name="CONCURRENCY", value=0)]
    )
    monkeypatch.setattr("optix.optimizer.scheduler.time.sleep", lambda _: None)
    scheduler.simulator = SimpleNamespace(process=SimpleNamespace(poll=lambda: None))
    for n in ["set_back_up_path", "update_data_field", "_configure_early_exit_controller"]:
        setattr(scheduler, n, Mock())
    scheduler._new_case_id = Mock(return_value="case-1")
    scheduler.run_target_server = Mock(side_effect=ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        scheduler.run_concurrency_sweep(Mock(), Mock(), [2, 4, 8])
    assert scheduler.del_log is False and scheduler.last_outcome.status.value == "failed"


CLI_IMPORT_ORDER_SCRIPT = (
    "import optix.optimizer.scheduler as scheduler; "
    "import optix.optimizer.agentic.strategies; "
    "from optix.optimizer.errors import ConcurrencySweepError; "
    "assert scheduler.ConcurrencySweepError is ConcurrencySweepError"
)


def test_cli_import_order_scheduler_first():
    result = subprocess.run(
        [sys.executable, "-c", CLI_IMPORT_ORDER_SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_backfill_sweep_best_persists_performance_and_concurrency(tmp_path, monkeypatch):
    scheduler, _ = archive_scheduler(tmp_path, monkeypatch, with_backup=False)
    perf = PerformanceIndex(generate_speed=300.0, success_rate=1)
    field = SimpleNamespace(name="CONCURRENCY", value=0)
    scheduler.simulate_run_info = [field]
    scheduler.backfill_sweep_best(perf, 8)
    assert scheduler.performance_index is perf
    assert field.value == 8
