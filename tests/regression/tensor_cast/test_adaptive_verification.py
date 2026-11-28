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

"""Adaptive DSpark probability, budget and physical-layout regression tests."""

import pytest
import torch

from tensor_cast.core.speculative_verification import (
    AcceptanceProfile,
    InfeasibleVerificationPlanError,
    PlanCost,
    select_verification_plan,
)
from tensor_cast.layers.sampler import Sampler, SamplingMetadata, SpecDecodeMetadata


def test_conditional_probability_product():
    profile = AcceptanceProfile.from_rates([0.9, 0.8, 0.1, 0.1], 4)
    assert profile.survival == pytest.approx([0.9, 0.72, 0.072, 0.0072])
    assert 1 + sum(profile.survival) == pytest.approx(2.6992)


@pytest.mark.parametrize("values", [[-0.1], [1.1], [float("nan")], [True]])
def test_invalid_rates_rejected(values):
    with pytest.raises(ValueError, match="finite numbers"):
        AcceptanceProfile.from_rates(values, 1)


def test_rates_window_mismatch():
    with pytest.raises(ValueError, match="count"):
        AcceptanceProfile.from_rates([0.9, 0.8], 3)
    for bad_n, rates in ((0, []), (-1, []), (True, [0.5])):
        with pytest.raises(ValueError, match="positive integer"):
            AcceptanceProfile.from_rates(rates, bad_n)


def test_planner_rejects_empty_jagged_zero_cost_and_infeasible_plans():
    for bad in ((), ((),), ((0.9, 0.8), (0.9,))):
        with pytest.raises(ValueError):
            select_verification_plan(bad, lambda p: PlanCost(1.0, 1.0))

    # Zero latency and zero remaining output must score as infeasible without ZeroDivisionError.
    with pytest.raises(InfeasibleVerificationPlanError):
        select_verification_plan(((0.9, 0.8),), lambda p: PlanCost(0.0, 1.0), tpot_limit_ms=10.0)
    with pytest.raises(InfeasibleVerificationPlanError):
        select_verification_plan(
            ((0.9, 0.8),), lambda p: PlanCost(1.0, 1.0), remaining_outputs=(0.0,), tpot_limit_ms=10.0
        )
    with pytest.raises(InfeasibleVerificationPlanError) as exc_info:
        select_verification_plan(((0.9, 0.8),), lambda p: PlanCost(100.0, 1.0), tpot_limit_ms=5.0)
    assert not exc_info.value.selection.feasible
    with pytest.raises(InfeasibleVerificationPlanError) as oom_info:
        select_verification_plan(((0.9, 0.8),), lambda p: PlanCost(1.0, -1.0))
    assert oom_info.value.selection.cost.memory_left_gb < 0

    # A zero-latency candidate is skipped in favor of a positive-latency feasible candidate.
    selected = select_verification_plan(
        ((0.9, 0.8),),
        lambda p: PlanCost(0.0 if p.draft_tokens == 0 else 2.0, 1.0),
        tpot_limit_ms=10.0,
    )
    assert selected.feasible and selected.plan.prefixes == (2,)


def test_user_config_validates_adaptive_rates_and_window_directly():
    from tensor_cast.core.user_config import UserInputConfig

    base = dict(
        model_id="tests/assets/model_config/qwen3_dense_0_6b",
        device="TEST_DEVICE",
        speculative_method="dspark",
        dspark_adaptive_verification=True,
    )
    for bad in (
        {"num_speculative_tokens": None, "dspark_acceptance_rates": [0.9]},
        {"num_speculative_tokens": 0, "dspark_acceptance_rates": []},
        {"num_speculative_tokens": True, "dspark_acceptance_rates": [0.9]},
        {"num_speculative_tokens": 3, "dspark_acceptance_rates": [0.9, 0.8]},
        {"num_speculative_tokens": 2, "dspark_acceptance_rates": [0.9, 1.2]},
        {"num_speculative_tokens": 2, "dspark_acceptance_rates": [0.9, float("nan")]},
        {"num_speculative_tokens": 2, "dspark_acceptance_rates": [0.9, True]},
    ):
        with pytest.raises(ValueError):
            UserInputConfig(**base, **bad)


def test_packed_rope_resets_positions_per_request_and_caps_sliding_window_blocks():
    from types import MethodType, SimpleNamespace
    from tensor_cast.layers.dflash import DflashDraftModel

    captured_pos = []

    class DummyDraft:
        head_dim = 4
        kv_dim = 8
        num_draft_layers = 2
        context_length = 512
        layer_types = ["sliding_attention", "full_attention"]
        sliding_window = 128
        layers = [SimpleNamespace(layer_idx=0), SimpleNamespace(layer_idx=1)]
        _draft_k_norms = [lambda x: x, lambda x: x]
        _draft_attn_layer_indices = [0, 1]

        def hidden_norm(self, x):
            return x

        def fc(self, x):
            return x

        def _split_context_kv(self, target_hidden, packed_lengths=None):
            k = torch.empty(1, target_hidden.shape[1], 8, device=target_hidden.device)
            return [(target_hidden, (k, k)), (target_hidden, (k, k))]

        def rotary_emb(self, _ref, fused_pos):
            captured_pos.append(fused_pos.clone())
            cos = torch.ones(1, fused_pos.shape[1], 4, device=fused_pos.device)
            return cos, cos

        def _apply_context_rope(self, key, _cos, _sin):
            return key.transpose(1, 2)

        def _get_layer_attn(self, _layer):
            raise RuntimeError("stub")

        def max_l_ctx(self):
            return 512

    dummy = DummyDraft()
    dummy._fuse_context_rope_and_cache = MethodType(DflashDraftModel._fuse_context_rope_and_cache, dummy)
    dummy.prepare_context_kv = MethodType(DflashDraftModel.prepare_context_kv, dummy)

    with pytest.raises(ValueError, match="positive length"):
        dummy.prepare_context_kv(torch.empty(1, 1, 8), batch=1, block=9, packed_lengths=[])
    with pytest.raises(ValueError, match="positive length"):
        dummy.prepare_context_kv(torch.empty(1, 3, 8), batch=2, block=9, packed_lengths=[3, 0])

    # Heterogeneous packed_lengths=(3, 5): RoPE positions must reset per request [0,1,2,0,1,2,3,4].
    dummy.prepare_context_kv(torch.empty(1, 8, 8), batch=2, block=9, packed_lengths=(3, 5))
    assert captured_pos[-1].flatten().tolist() == [0, 1, 2, 0, 1, 2, 3, 4] * 2

    # Sliding window layer (128) with long requests (512, 512) must cap block allocation by 128, not 512.
    prepared = dummy.prepare_context_kv(
        torch.empty(1, 1024, 8), batch=2, block=9, attn_use_configured_context=False, packed_lengths=(512, 512)
    )
    kv_caches = prepared["kv_cache_by_layers"]
    # Layer 0 (sliding_window=128): ceil((128+9)/128) * 2 = 4 blocks; Layer 1 (full=512): ceil((512+9)/128) * 2 = 10.
    assert kv_caches[0].shape[1] == 4
    assert kv_caches[1].shape[1] == 10
    assert prepared["packed_layer_ctx_lens"] == [128, 512]


@pytest.mark.parametrize("survival, expected", [((1.0,) * 4, (4,)), ((0.0,) * 4, (0,))])
def test_acceptance_extremes(survival, expected):
    result = select_verification_plan((survival,), lambda p: PlanCost(1 + 0.1 * p.draft_tokens, 10))
    assert result.plan.prefixes == expected


def test_tail_drop_and_non_monotonic_cost():
    row = (0.9, 0.72, 0.072, 0.0072)
    result = select_verification_plan((row,), lambda p: PlanCost(1 + 0.1 * p.draft_tokens, 10))
    assert result.plan.prefixes == (2,)
    assert result.plan.expected_outputs == pytest.approx((2.62,))
    # An expensive valley is followed by a cheaper full-window execution bucket.
    costs = [1, 2, 100, 1]
    result = select_verification_plan(((0.9, 0.8, 0.7),), lambda p: PlanCost(costs[p.draft_tokens], 10))
    assert result.plan.prefixes == (3,)


def test_budget_bound_and_deterministic_ties():
    def run():
        seen = []
        result = select_verification_plan(
            ((1.0,) * 8,) * 10,
            lambda p: (seen.append(p.prefixes) or PlanCost(1 + p.draft_tokens, 1)),
            max_evaluations=8,
        )
        return result, seen

    result, seen = run()
    assert result.evaluated_candidates <= 8
    assert len(seen) == len(set(seen))
    assert (0,) * 10 in seen and (8,) * 10 in seen
    assert run()[1] == seen


def test_packed_draft_inputs_preserve_request_order():
    from types import SimpleNamespace
    from tensor_cast.layers.dflash import DflashWrapper

    lengths, expected = (3, 1, 3), [2, 3, 6]
    total = sum(lengths)
    spec = SpecDecodeMetadata(torch.arange(total), len(lengths), 3, lengths, adaptive_verification=True)
    wrapper = SimpleNamespace(
        sampler=Sampler(),
        _draft_query_length=lambda: 4,
        _run_target_collect_aux=lambda *args, **kwargs: (
            torch.eye(total),
            [torch.arange(total).reshape(1, total, 1)],
        ),
        _run_draft_block=lambda target, aux, sampled, *args, **kwargs: (
            aux[0],
            sampled[:, -1:],
            kwargs["packed_lengths"],
        ),
    )
    result = DflashWrapper._forward_adaptive(
        wrapper, None, None, None, sampling_metadata=SamplingMetadata(spec_decode_metadata=spec)
    )
    aux, anchors, packed_lengths = result
    assert aux.flatten().tolist() == list(range(total))
    assert anchors.flatten().tolist() == expected
    assert packed_lengths == lengths


def test_frontier_prefers_requests_with_remaining_gain():
    evaluated = []

    def cost(plan):
        evaluated.append(plan.prefixes)
        return PlanCost(1 + 0.1 * plan.draft_tokens, 1)

    selected = select_verification_plan(((0.99, 0.9), (0.7, 0.3)), cost, remaining_outputs=(1.0, 3.0))
    assert evaluated[:3] == [(0, 0), (0, 1), (0, 2)]
    assert selected.plan.prefixes == (0, 2)


def test_sampled_homogeneous_batch_covers_all_uniform_prefixes():
    evaluated = set()

    def cost(plan):
        evaluated.add(plan.prefixes)
        return PlanCost(1 if plan.prefixes == (15,) * 32 else 100, 1)

    selected = select_verification_plan(((1.0,) * 16,) * 32, cost)
    assert {(k,) * 32 for k in range(17)} <= evaluated
    assert selected.plan.prefixes == (15,) * 32
    assert selected.evaluated_candidates == len(evaluated) == 32


@pytest.mark.parametrize("compile_model", [False, True])
def test_dspark_target_is_varlen_while_draft_stays_full(compile_model):
    from dataclasses import replace
    from tensor_cast.core.input_generator import RequestInfo, generate_inputs_varlen
    from tensor_cast.core.model_runner import ModelRunner
    from tensor_cast.core.user_config import UserInputConfig

    config = UserInputConfig(
        model_id="tests/assets/model_config/qwen3_dense_0_6b",
        device="TEST_DEVICE",
        num_queries=2,
        query_len=4,
        context_length=16,
        decode=True,
        do_compile=compile_model,
        num_hidden_layers_override=1,
        speculative_method="dspark",
        num_speculative_tokens=3,
        num_draft_layers=1,
        dspark_adaptive_verification=True,
        dspark_acceptance_rates=[0.9, 0.8, 0.1],
    )
    runner = ModelRunner(config)
    requests = [RequestInfo(3, 19), RequestInfo(1, 17)]
    inputs = generate_inputs_varlen(runner.model, requests, config.block_size)
    spec = inputs["sampling_metadata"].spec_decode_metadata
    assert spec.verification_lens == (3, 1)
    assert spec.logits_indices.numel() == 4
    assert inputs["attention_meta"].query_lens_values == [3, 1]
    events = []
    result = runner.run_inference(
        requests,
        generate_inputs_func=generate_inputs_varlen,
        runtime_observer=lambda rt: events.extend(e.op_invoke_info for e in rt.event_list),
    )
    confidence = [e for e in events if str(e.func) == "aten.mm.default" and e.args[1].shape[-1] == 1]
    assert len(confidence) == 4  # One full generic draft window for the batch.
    attention = [e for e in events if str(e.func) == "tensor_cast.attention.default"]
    assert len(attention) == 2  # One target pass and one draft pass.
    draft = runner.model._inner.draft
    projection_dims = (draft.context_kv_proj.in_features, draft.context_kv_proj.out_features)
    context_projections = [
        e for e in events if str(e.func) == "aten.mm.default" and tuple(e.args[1].shape) == projection_dims
    ]
    assert [e.args[0].shape[0] for e in context_projections] == [sum(spec.verification_lens)]
    assert result.execution_time_s["analytic"] > 0
    assert result.peak_memory_usage_gb >= result.model_weight_size_gb
    if not compile_model:
        base = runner.run_inference([RequestInfo(1, 17)] * 2, generate_inputs_func=generate_inputs_varlen)
        one_extra = runner.run_inference(
            [RequestInfo(2, 18), RequestInfo(1, 17)], generate_inputs_func=generate_inputs_varlen
        )
        assert base.execution_time_s["analytic"] <= one_extra.execution_time_s["analytic"]

    full_requests = [RequestInfo(4, 20), RequestInfo(4, 20)]
    fixed = ModelRunner(replace(config, dspark_adaptive_verification=False, dspark_acceptance_rates=None))
    fixed_metrics = fixed.run_inference(full_requests, generate_inputs_func=generate_inputs_varlen)
    full_metrics = runner.run_inference(full_requests, generate_inputs_func=generate_inputs_varlen)
    assert full_metrics.execution_time_s == pytest.approx(fixed_metrics.execution_time_s)
    assert full_metrics.peak_memory_usage_gb == pytest.approx(fixed_metrics.peak_memory_usage_gb)
    if not compile_model:
        from collections import Counter

        histograms = []
        for batch_requests in (requests, requests * 5):
            step_events = []
            runner.run_inference(
                batch_requests,
                generate_inputs_func=generate_inputs_varlen,
                runtime_observer=lambda rt, sink=step_events: sink.extend(e.op_invoke_info for e in rt.event_list),
            )
            histograms.append(Counter(str(e.func) for e in step_events))
        assert histograms[0] == histograms[1]
        for op_name in ("aten.slice.Tensor", "aten.cat.default", "aten.arange.default"):
            assert histograms[0][op_name] == histograms[1][op_name]
