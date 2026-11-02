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
"""Single-token Prefill results may omit TPOT when TPOT scoring is disabled."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from optix.config.config import (
    AisBenchConfig,
    OptimizerConfigField,
    PerformanceIndex,
    field_to_param,
    map_param_with_value,
)
from optix.optimizer.experience_fine_tunning import FineTune, StopFineTune
from optix.optimizer.optimizer import PSOOptimizer
from optix.optimizer.plugins.benchmark import AisBench


@pytest.fixture
def aisbench_result(tmp_path):
    result_dir = tmp_path / "run" / "performances" / "vllm-api-stream-chat"
    result_dir.mkdir(parents=True)
    csv_path = result_dir / "gsm8k.csv"
    csv_path.write_text(
        "Performance Parameters,Stage,Average\nTTFT,total,150.0 ms\nOutputTokens,total,1.0\n", encoding="utf-8"
    )
    csv_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "Total Requests": {"total": 18},
                "Success Requests": {"total": 18},
                "Request Throughput": {"total": "4.0 req/s"},
                "Output Token Throughput": {"total": "4.0 token/s"},
            }
        ),
        encoding="utf-8",
    )
    bench = object.__new__(AisBench)
    bench.config = AisBenchConfig(output_path=tmp_path)
    bench.mindie_benchmark_perf_columns = ["average"]
    return bench, csv_path


@pytest.mark.parametrize("penalty", [0.0, 1.0])
def test_missing_tpot_is_allowed_only_when_unscored(aisbench_result, penalty):
    bench, _ = aisbench_result
    with patch("optix.optimizer.plugins.benchmark.get_settings", return_value=SimpleNamespace(tpot_penalty=penalty)):
        if penalty:
            with pytest.raises(ValueError, match="metric_name tpot"):
                bench.get_performance_index()
        else:
            result = bench.get_performance_index()
            assert result.time_to_first_token == pytest.approx(0.15)
            assert result.time_per_output_token is None
            assert result.success_rate == 1.0
            assert result.throughput == 4.0
            assert result.generate_speed == 4.0


@pytest.mark.parametrize("penalty", [0.0, 1.0])
@pytest.mark.parametrize("tpot_ms", [0.0, 20.0])
def test_measured_tpot_is_preserved_even_when_unscored(aisbench_result, penalty, tpot_ms):
    bench, csv_path = aisbench_result
    with csv_path.open("a", encoding="utf-8") as stream:
        stream.write(f"TPOT,total,{tpot_ms} ms\n")
    with patch("optix.optimizer.plugins.benchmark.get_settings", return_value=SimpleNamespace(tpot_penalty=penalty)):
        assert bench.get_performance_index().time_per_output_token == pytest.approx(tpot_ms / 1000)


def test_disabling_tpot_does_not_allow_missing_ttft(aisbench_result):
    bench, csv_path = aisbench_result
    csv_path.write_text("Performance Parameters,Stage,Average\nOutputTokens,total,1.0\n", encoding="utf-8")
    with patch("optix.optimizer.plugins.benchmark.get_settings", return_value=SimpleNamespace(tpot_penalty=0.0)):
        with pytest.raises(ValueError, match="metric_name ttft"):
            bench.get_performance_index()


def _prefill_optimizer():
    scheduler = MagicMock()
    scheduler.error_info = None
    return PSOOptimizer(
        scheduler=scheduler,
        target_field=(OptimizerConfigField(name="CONCURRENCY", min=1, max=8, value=4, dtype="int"),),
        fine_tune=SimpleNamespace(ttft_upper_bound=1.0),
        ttft_penalty=1.0,
        tpot_penalty=0.0,
        ttft_slo=1.0,
        use_request_rate_calibration=False,
        manage_simulator_lifecycle=False,
    )


def test_prefill_selection_preserves_missing_tpot():
    optimizer = _prefill_optimizer()
    results = [
        PerformanceIndex(generate_speed=2.0, time_to_first_token=0.2, success_rate=1.0),
        PerformanceIndex(generate_speed=4.0, time_to_first_token=0.3, success_rate=1.0),
    ]
    fitness = [optimizer.minimum_algorithm(result) for result in results]
    _, params, result = optimizer.best_params(fitness, [[2], [4]], results)
    assert params == [4]
    assert result is results[1]
    assert all(item.time_per_output_token is None for item in results)
    assert result.model_dump(mode="json")["time_per_output_token"] is None


@pytest.mark.parametrize("penalty", [0.0, 1.0])
@pytest.mark.parametrize("direct", [False, True], ids=["best_params", "pd_selector"])
@pytest.mark.parametrize(
    "tpot_values,threshold,expected",
    [
        ((None, 0.04, 0.03), 0.05, 2),
        ((0.08, None, 0.07), 0.05, 2),
        ((None, None, None), 0.05, 0),
        ((None, 0.0, 0.07), 0.05, 1),
        ((None, 0.04, 0.03), 0.0, 0),
    ],
    ids=["within_slo", "over_slo", "all_missing", "measured_zero", "zero_threshold"],
)
def test_pd_disaggregation_selection_handles_missing_tpot(tpot_values, threshold, expected, direct, penalty):
    optimizer = _prefill_optimizer()
    optimizer.tpot_penalty = penalty
    optimizer.fine_tune = SimpleNamespace(fine_tune_mode="pd_disaggregation", tpot_upper_bound=threshold)
    results = [
        PerformanceIndex(
            generate_speed=speed,
            throughput=speed,
            time_to_first_token=None,
            time_per_output_token=tpot,
            success_rate=1.0,
        )
        for speed, tpot in zip((100.0, 3.0, 4.0), tpot_values)
    ]
    fitness = [optimizer.minimum_algorithm(result) for result in results]
    selector = optimizer.best_pd_disaggregation_params if direct else optimizer.best_params
    selected_fitness, params, result = selector(fitness, [[0], [1], [2]], results)
    assert selected_fitness == fitness[expected]
    assert params == [expected]
    assert result is results[expected]
    assert all(item.time_to_first_token is None for item in results)
    assert tuple(item.time_per_output_token for item in results) == tpot_values
    if tpot_values[expected] is None:
        assert result.model_dump(mode="json")["time_per_output_token"] is None


@pytest.mark.parametrize("penalty", [0.0, 1.0])
def test_baseline_warns_about_missing_tpot_only_when_required(penalty):
    optimizer = _prefill_optimizer()
    optimizer.tpot_penalty = penalty
    optimizer._baseline_from_checkpoint = True
    optimizer.default_res = PerformanceIndex(generate_speed=4.0, time_to_first_token=0.15, success_rate=1.0)
    optimizer.scheduler.simulator.data_field = optimizer.target_field
    optimizer.scheduler.benchmark.data_field = ()
    with patch("optix.optimizer.optimizer.logger") as log:
        optimizer.prepare_plugin()
    assert log.warning.call_count == int(bool(penalty))


def _fine_tune_fields(fixed_rate=False):
    return (
        OptimizerConfigField(name="CONCURRENCY", min=1, max=16, value=8, dtype="int"),
        OptimizerConfigField(
            name="REQUESTRATE",
            min=0.0 if fixed_rate else 0.1,
            max=0.0 if fixed_rate else 16.0,
            value=0.0 if fixed_rate else 2.0,
            dtype="float",
        ),
    )


@pytest.mark.parametrize("mode", ["pd_mixed", "pd_disaggregation"])
@pytest.mark.parametrize("penalty", [0.0, 1.0])
def test_fine_tune_requires_tpot_only_when_scored(mode, penalty):
    fine_tune = FineTune(ttft_penalty=1.0, tpot_penalty=penalty, fine_tune_mode=mode)
    result = PerformanceIndex(time_to_first_token=0.15, time_per_output_token=None)
    if penalty:
        with pytest.raises(ValueError, match="Missing performance data for TPOT"):
            fine_tune.check_config_and_performance(result)
    else:
        fine_tune.check_config_and_performance(result)


def test_missing_tpot_clears_previous_direction_flags():
    fine_tune = FineTune(ttft_penalty=1.0, tpot_penalty=0.0, ttft_slo=0.1)
    fine_tune.direction_of_field_update(PerformanceIndex(time_to_first_token=0.15, time_per_output_token=0.1))
    assert fine_tune.tpot_over_slo
    fine_tune.direction_of_field_update(PerformanceIndex(time_to_first_token=0.15, time_per_output_token=None))
    assert not fine_tune.tpot_over_slo
    assert not fine_tune.tpot_under_lower_bound
    assert fine_tune.ttft_over_slo


@pytest.mark.parametrize("fixed_rate", [False, True], ids=["request_rate", "concurrency_fallback"])
def test_single_token_prefill_adjusts_load_without_tpot(aisbench_result, fixed_rate):
    bench, _ = aisbench_result
    with patch("optix.optimizer.plugins.benchmark.get_settings", return_value=SimpleNamespace(tpot_penalty=0.0)):
        result = bench.get_performance_index()
    fields = _fine_tune_fields(fixed_rate)
    fine_tune = FineTune(ttft_penalty=1.0, tpot_penalty=0.0, ttft_slo=0.1, target_field=fields)
    updated = fine_tune.fine_tune_with_concurrency_and_request_rate(field_to_param(fields), result)
    if fixed_rate:
        assert updated[0].value < fields[0].value
        assert updated[1].value == 0.0
    else:
        assert updated[0].value == fields[0].value
        assert updated[1].value < fields[1].value
    assert result.time_per_output_token is None


def test_pd_request_rate_fine_tune_continues_without_tpot(aisbench_result):
    bench, _ = aisbench_result
    with patch("optix.optimizer.plugins.benchmark.get_settings", return_value=SimpleNamespace(tpot_penalty=0.0)):
        result = bench.get_performance_index()
    fields = _fine_tune_fields()
    fine_tune = FineTune(
        ttft_penalty=1.0, tpot_penalty=0.0, ttft_slo=0.1, target_field=fields, fine_tune_mode="pd_disaggregation"
    )
    probe = fine_tune.prepare_pd_disaggregation_probe(field_to_param(fields))
    assert fine_tune.init_pd_disaggregation_request_rate(result) == 4.0
    initialized = fine_tune.fine_tune_pd_disaggregation(field_to_param(probe), result)
    assert [field.value for field in initialized] == [8, 4.0]
    reduced = fine_tune.fine_tune_pd_disaggregation(field_to_param(initialized), result)
    assert [field.value for field in reduced] == [8, 2.0]
    low_ttft = result.model_copy(update={"time_to_first_token": 0.05})
    bisected = fine_tune.fine_tune_pd_disaggregation(field_to_param(reduced), low_ttft)
    assert [field.value for field in bisected] == [8, 3.0]
    in_band = result.model_copy(update={"time_to_first_token": 0.09})
    with pytest.raises(StopFineTune, match="TTFT is within"):
        fine_tune.fine_tune_pd_disaggregation(field_to_param(bisected), in_band)
    assert result.time_per_output_token is None


def test_pd_concurrency_still_requires_tpot_when_scored():
    fields = _fine_tune_fields()
    fine_tune = FineTune(ttft_penalty=1.0, tpot_penalty=1.0, target_field=fields, fine_tune_mode="pd_disaggregation")
    result = PerformanceIndex(time_to_first_token=0.15, throughput=4.0, time_per_output_token=None)
    fine_tune.init_pd_disaggregation_request_rate(result)
    with pytest.raises(ValueError, match="Missing performance data for TPOT"):
        fine_tune.fine_tune_pd_disaggregation(field_to_param(fields), result)


@pytest.mark.parametrize("mode", ["pd_mixed", "pd_disaggregation"])
def test_candidate_refinement_runs_measurement_without_tpot(aisbench_result, mode):
    bench, _ = aisbench_result
    with patch("optix.optimizer.plugins.benchmark.get_settings", return_value=SimpleNamespace(tpot_penalty=0.0)):
        result = bench.get_performance_index()
    optimizer = _prefill_optimizer()
    optimizer.target_field = _fine_tune_fields()
    optimizer.fine_tune = FineTune(
        ttft_penalty=1.0,
        tpot_penalty=0.0,
        ttft_slo=0.1,
        target_field=optimizer.target_field,
        fine_tune_mode=mode,
    )
    optimizer.max_fine_tune = 2
    optimizer.default_run_param = field_to_param(optimizer.target_field)
    optimizer.default_res = result
    optimizer.default_fitness = optimizer.minimum_algorithm(result)
    optimizer.scheduler.last_outcome = None
    optimizer.scheduler.data_storage.config.pso_top_k = 3
    optimizer.scheduler.rerun_benchmark_only.return_value = result
    optimizer.scheduler.run_benchmark_only.return_value = result
    candidate_fields = _fine_tune_fields()
    candidate_fields[1].value = 3.0
    optimizer.get_target_field_from_case_data = MagicMock(return_value=candidate_fields)
    with patch("optix.optimizer.optimizer.logger") as log:
        _, record_params, record_results = optimizer.refine_optimization_candidates(pd.DataFrame([{"case_id": 1}]))
    log.error.assert_not_called()
    assert optimizer.scheduler.run_benchmark_only.call_count >= 1
    assert len(record_results) > 2
    assert all(item.time_per_output_token is None for item in record_results)
    measured = map_param_with_value(optimizer.scheduler.run_benchmark_only.call_args.args[0], optimizer.target_field)
    assert measured[0].value == 8
    assert measured[1].value < 6.0
    assert len(record_params) == len(record_results)
