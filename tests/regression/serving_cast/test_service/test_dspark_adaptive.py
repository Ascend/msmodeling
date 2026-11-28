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

"""DSpark adaptive plans must change both physical work and serving progress."""

from unittest.mock import patch

import pytest

from serving_cast.service.agg_throughput_optimizer import AggThroughputOptimizer
from serving_cast.service.disagg_throughput_optimizer import DisaggThroughputOptimizer
from serving_cast.service.aggregation_timeline import RequestShape, TimelineRequest
from serving_cast.service.optimizer_summary import EARLY_STOP_PREFILL_OOM, EARLY_STOP_TPOT_LIMIT, OptimizerSummary
from serving_cast.service.utils import OptimizerData
from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.core.user_config import UserInputConfig


@pytest.fixture
def adaptive_context():
    user = UserInputConfig(
        model_id="tests/assets/model_config/qwen3_dense_0_6b",
        device="TEST_DEVICE",
        num_queries=2,
        query_len=4,
        context_length=16,
        decode=True,
        num_hidden_layers_override=1,
        speculative_method="dspark",
        num_speculative_tokens=3,
        num_draft_layers=1,
        dspark_adaptive_verification=True,
        dspark_acceptance_rates=[0.0, 0.0, 0.0],
    )
    runner = ModelRunner(user)
    data = OptimizerData(
        input_length=8,
        output_length=5,
        batch_size=2,
        max_batched_tokens=64,
        num_devices=1,
        serving_cost=0,
        tpot_limits=100000,
        dspark_block_size=4,
        dspark_acceptance_length=3,
        dspark_markov_rank=256,
        speculative_method="dspark",
        dspark_adaptive_verification=True,
    )
    return runner, data


def test_disagg_prefill_keeps_normal_batch_search(adaptive_context):
    runner, data = adaptive_context
    data.ttft_limits = 100000
    data.tpot_limits = None
    optimizer = DisaggThroughputOptimizer()
    optimizer.initialize(runner)
    with patch.object(optimizer.adaptive_verification, "run_range") as run_range:
        summary = optimizer.run(data, [1, 2])
    run_range.assert_not_called()
    frame = summary.get_summary_df()
    assert not frame.empty
    assert all("verify=adaptive" not in label for label in frame["parallel"])


@pytest.mark.parametrize("oom", [False, True])
def test_adaptive_scan_returns_no_result_if_all_batches_fail(adaptive_context, oom):
    runner, data = adaptive_context
    optimizer = AggThroughputOptimizer()
    optimizer.initialize(runner)
    summary = OptimizerSummary(data)
    summary.set_early_stop_flag(-1 if oom else 1, data.tpot_limits + 1, None, reason=EARLY_STOP_PREFILL_OOM)
    with patch.object(optimizer, "get_inference_info", return_value=summary) as evaluate:
        assert optimizer.run(data, [1, 2]) is None
    assert evaluate.call_count == 1
    assert optimizer._last_run_early_stop_reason == (EARLY_STOP_PREFILL_OOM if oom else EARLY_STOP_TPOT_LIMIT)


def test_adaptive_scan_breaks_on_early_stop_and_preserves_reason_with_partial_rows(adaptive_context):
    import pandas as pd

    runner, data = adaptive_context
    optimizer = DisaggThroughputOptimizer()
    optimizer.initialize(runner)
    ok = OptimizerSummary(data)
    ok.set_summary_df(pd.DataFrame([{"batch_size": 1, "token/s": 100.0}]))
    stopped = OptimizerSummary(data)
    stopped.set_early_stop_flag(1.0, data.tpot_limits + 1, None)
    with patch.object(optimizer, "get_inference_info", side_effect=[ok, stopped, ok]) as evaluate:
        result = optimizer.run(data, [1, 3])
    assert evaluate.call_count == 2
    assert len(result.get_summary_df()) == 1
    assert optimizer._last_run_early_stop_reason == EARLY_STOP_TPOT_LIMIT


def test_select_step_rejects_prefill_only_batch_and_enforces_agg_tpot_limit(adaptive_context):
    from serving_cast.service.latency_table import ForwardLatencyRecord
    from tensor_cast.core.speculative_verification import AcceptanceProfile

    runner, data = adaptive_context
    optimizer = AggThroughputOptimizer()
    optimizer.initialize(runner)
    optimizer.adaptive_verification.profile = AcceptanceProfile((0.9, 0.8, 0.7))

    with pytest.raises(ValueError, match="no decode request"):
        optimizer.adaptive_verification.select_step((RequestShape(False, 4, 4),), data, lambda _: None)

    # With remaining_outputs and tpot_limits both present, planner must choose the shorter TPOT-feasible prefix.
    data.tpot_limits = 10.0

    def cost(actual):
        q = actual[0].query_len
        # q=4 gives higher raw throughput (3.124/35 = 0.089) but TPOT=11.2ms > 10ms;
        # q=2 meets TPOT (18/1.9 = 9.47ms <= 10ms).
        latency = {1: 15.0, 2: 18.0, 3: 32.0, 4: 35.0}[q]
        return ForwardLatencyRecord(latency, 10.0, "")

    selection, _, _ = optimizer.adaptive_verification.select_step(
        (RequestShape(True, 4, 20),), data, cost, remaining_outputs=(5.0,)
    )
    assert selection.feasible
    assert selection.plan.prefixes == (1,)

    # When no candidate meets tpot_limits on a pure decode step with remaining>=1, step is infeasible and not logged.
    optimizer.adaptive_verification.steps.clear()
    data.tpot_limits = 1.0
    infeasible, _, _ = optimizer.adaptive_verification.select_step(
        (RequestShape(True, 4, 20),), data, cost, remaining_outputs=(5.0,)
    )
    assert not infeasible.feasible
    assert optimizer.adaptive_verification.steps == []


def test_disagg_handles_zero_latency_and_infeasible_tpot_without_crashing(adaptive_context):
    from serving_cast.service.latency_table import ForwardLatencyRecord

    runner, data = adaptive_context
    optimizer = DisaggThroughputOptimizer()
    optimizer.initialize(runner)
    data.tpot_limits = 5.0
    zero_record = ForwardLatencyRecord(0.0, 10.0, "")
    with patch.object(optimizer.adaptive_verification, "forward_record", return_value=zero_record):
        summary = optimizer.get_inference_info(data)
    row = summary.get_summary_df().iloc[0]
    assert row["tpot"] == 0.0
    assert row["token/s"] == 0.0
    assert summary.check_early_stop_flag()
    assert summary.get_early_stop_reason() == EARLY_STOP_TPOT_LIMIT
    assert optimizer.adaptive_verification.steps == []

    slow_record = ForwardLatencyRecord(50.0, 10.0, "")
    with patch.object(optimizer.adaptive_verification, "forward_record", return_value=slow_record):
        assert optimizer.run(data, [2, 3]) is None
    assert optimizer._last_run_early_stop_reason == EARLY_STOP_TPOT_LIMIT


def test_mixed_step_counts_only_decode_cache_and_reuses_selected_record(adaptive_context):
    runner, data = adaptive_context
    data.max_batched_tokens = 8
    data.ttft_limits = 100000
    optimizer = AggThroughputOptimizer()
    optimizer.initialize(runner)
    requests = [TimelineRequest(n, [RequestShape(False, n, n)]) for n in (5, 4)]
    with (
        patch("serving_cast.service.aggregation_timeline.build_requests", return_value=requests),
        patch.object(
            optimizer.adaptive_verification, "forward_record", wraps=optimizer.adaptive_verification.forward_record
        ) as forward,
    ):
        row = optimizer.run(data, [2, 2]).get_summary_df().iloc[0]
    assert any(
        {s.is_decode for s in shapes} == {False, True} for shapes, _ in optimizer._last_aggregation_timeline.steps
    )
    prefill_keys = [key for key in optimizer._forward_record_cache if not key.is_decode]
    decode_keys = [key for key in optimizer._forward_record_cache if key.is_decode]
    assert len(prefill_keys) == 2
    assert row["additional_performance_evaluations"] == len(decode_keys)
    assert forward.call_count == row["verification_candidates"]


def test_pd_metrics_use_selected_output_not_fixed_acceptance(adaptive_context):
    runner, data = adaptive_context
    optimizer = DisaggThroughputOptimizer()
    optimizer.initialize(runner)
    summary = optimizer.run(data, [2, 2])
    selected = optimizer.adaptive_verification.steps[-1][0]
    row = summary.get_summary_df().iloc[0]
    assert selected.plan.expected_outputs == (1.0, 1.0)
    assert row["tpot"] == pytest.approx(selected.cost.latency_ms, abs=0.001)
    assert row["token/s"] == pytest.approx(2000 / selected.cost.latency_ms, abs=0.001)
    # The fixed scalar deliberately says full acceptance and must have no effect.
    assert data.dspark_acceptance_length == 3
    repeated = optimizer.run(data, [2, 2]).get_summary_df().iloc[0]
    assert row["additional_performance_evaluations"] > 0
    assert repeated["additional_performance_evaluations"] == 0
    assert repeated["forward_cache_hits"] == repeated["verification_candidates"]


def test_cache_distinguishes_request_organization_and_reuses_it(adaptive_context):
    runner, data = adaptive_context
    optimizer = DisaggThroughputOptimizer()
    optimizer.initialize(runner)
    a = (RequestShape(True, 4, 20), RequestShape(True, 1, 17))
    b = (RequestShape(True, 3, 19), RequestShape(True, 2, 18))
    with patch.object(runner, "run_inference", wraps=runner.run_inference) as forward:
        first = optimizer.adaptive_verification.forward_record(a, data)
        optimizer.adaptive_verification.forward_record(b, data)
        optimizer.adaptive_verification.forward_record((RequestShape(True, 4, 21), a[1]), data)
        repeated = optimizer.adaptive_verification.forward_record(a, data)
    assert repeated is first
    assert forward.call_count == 3
    assert [r.query_len for r in forward.call_args_list[0].args[0]] == [4, 1]
    assert [r.query_len for r in forward.call_args_list[1].args[0]] == [3, 2]


@pytest.mark.parametrize("output_length", [1, 5])
def test_aggregation_advances_each_request_and_records_only_selected_work(adaptive_context, output_length):
    runner, data = adaptive_context
    data.output_length = output_length
    data.ttft_limits = 100000
    optimizer = AggThroughputOptimizer()
    optimizer.initialize(runner)
    summary = optimizer.run(data, [2, 2])
    timeline = optimizer._last_aggregation_timeline
    assert not timeline.oom
    assert all(r.remaining_output == 0 for r in timeline.requests)
    decode = [shapes for shapes, _ in timeline.steps if any(s.is_decode for s in shapes)]
    assert len(decode) == data.output_length - 1
    for shapes, step in zip(decode, optimizer.adaptive_verification.steps):
        assert tuple(s.query_len for s in shapes if s.is_decode) == step[0].plan.query_lens
        assert step[0].plan.expected_outputs == (1.0, 1.0)
    assert len(optimizer.adaptive_verification.steps) == len(decode)
    if output_length == 1:
        assert "verification_candidates" not in summary.get_summary_df()
        assert summary.get_summary_df().iloc[0]["max_request_tpot_ms"] == 0
    else:
        assert summary.get_summary_df().iloc[0]["max_request_tpot_ms"] > 0


def test_unequal_prefix_metrics_and_request_slo(adaptive_context):
    from serving_cast.service.latency_table import ForwardLatencyRecord
    from tensor_cast.core.speculative_verification import AcceptanceProfile

    runner, data = adaptive_context
    optimizer = DisaggThroughputOptimizer()
    optimizer.initialize(runner)
    optimizer.dp = 2
    optimizer.adaptive_verification.profile = AcceptanceProfile((0.9, 0.8, 0.7))

    def cost(shapes, _):
        q = tuple(s.query_len for s in shapes)
        return ForwardLatencyRecord(1 if q == (2, 1) else 100, 10, "")

    with patch.object(optimizer.adaptive_verification, "forward_record", side_effect=cost):
        summary = optimizer.get_inference_info(data)
    row = summary.get_summary_df().iloc[0]
    assert row["token/s"] == pytest.approx(1000 * 2 * 2.9, abs=0.001)
    assert row["tpot"] == pytest.approx(2 / 2.9, abs=0.001)
    assert row["max_request_tpot_ms"] == 1
    assert "verify=adaptive" in row["parallel"]
    assert row["concurrency"] == 4
