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

"""Behavior matrix for the forward pipeline and stage-memory estimators."""

from __future__ import annotations

import pytest

from serving_cast.service.pipeline_schedule import (
    estimate_forward_pipeline,
    estimate_mixed_pd_stage_memory,
    estimate_mixed_pd_stage_memory_across_profiles,
    estimate_pipeline_stage_memory,
    estimate_pipeline_stage_memory_across_profiles,
    estimate_repeated_pipeline,
    split_batch_size,
)
from tensor_cast.pipeline_parallel import (
    PipelineProfile,
    PipelineStageProfile,
    PipelineTransferProfile,
)


def _stage_values(value, pp_size):
    if isinstance(value, (tuple, list)):
        assert len(value) == pp_size
        return tuple(value)
    return (value,) * pp_size


def _profile(
    pp_size: int,
    *,
    compute=1.0,
    comm=0.0,
    weight=0,
    runtime_peak=0,
    kv_cache=0,
    indexer_cache=0,
    payload_bytes=0,
    model="analytic",
) -> PipelineProfile:
    compute_values = _stage_values(compute, pp_size)
    weight_values = _stage_values(weight, pp_size)
    runtime_values = _stage_values(runtime_peak, pp_size)
    kv_values = _stage_values(kv_cache, pp_size)
    indexer_values = _stage_values(indexer_cache, pp_size)
    if isinstance(comm, (tuple, list)):
        assert len(comm) == max(0, pp_size - 1)
        comm_values = tuple(comm)
    else:
        comm_values = (comm,) * max(0, pp_size - 1)

    stages = tuple(
        PipelineStageProfile(
            stage_id=stage_id,
            layer_start=stage_id,
            layer_end=stage_id + 1,
            compute_time_s_by_model={model: compute_values[stage_id]},
            outgoing_comm_time_s_by_model={model: comm_values[stage_id] if stage_id < pp_size - 1 else 0.0},
            weight_bytes=weight_values[stage_id],
            activation_bytes=0,
            kv_cache_bytes=kv_values[stage_id],
            kv_cache_per_token_bytes=0.0,
            indexer_cache_bytes=indexer_values[stage_id],
            indexer_cache_per_token_bytes=0.0,
            runtime_peak_bytes=runtime_values[stage_id],
            outgoing_payload_bytes=(payload_bytes if stage_id < pp_size - 1 else 0),
        )
        for stage_id in range(pp_size)
    )
    transfers = tuple(
        PipelineTransferProfile(
            source_stage_id=stage_id,
            target_stage_id=stage_id + 1,
            payload_bytes=payload_bytes,
            time_s_by_model={model: comm_values[stage_id]},
            bandwidth_bytes_ps=1.0e9,
            latency_s=0.0,
        )
        for stage_id in range(pp_size - 1)
    )
    return PipelineProfile(
        pp_size=pp_size,
        layer_partition=(1,) * pp_size,
        stages=stages,
        transfers=transfers,
    )


@pytest.mark.parametrize(
    ("batch_size", "microbatch_size", "expected"),
    [
        (8, 4, (4, 4)),
        (9, 4, (4, 4, 1)),
        (3, 8, (3,)),
    ],
)
def test_split_batch_size_matrix(batch_size, microbatch_size, expected):
    assert split_batch_size(batch_size, microbatch_size) == expected


@pytest.mark.parametrize("batch_size,microbatch_size", [(0, 1), (1, 0), (-1, 1)])
def test_split_batch_size_rejects_non_positive(batch_size, microbatch_size):
    with pytest.raises(ValueError, match="positive"):
        split_batch_size(batch_size, microbatch_size)


@pytest.mark.parametrize(
    (
        "profile",
        "num_microbatches",
        "expected_completion",
        "expected_busy",
        "expected_makespan",
    ),
    [
        pytest.param(
            _profile(1, compute=2.0),
            3,
            (2.0, 4.0, 6.0),
            (6.0,),
            6.0,
            id="pp1-serial",
        ),
        pytest.param(
            _profile(2, compute=1.0, comm=0.5),
            2,
            (2.5, 4.0),
            (3.0, 3.0),
            4.0,
            id="pp2-blocking-communication",
        ),
        pytest.param(
            _profile(4, compute=1.0),
            2,
            (4.0, 5.0),
            (2.0, 2.0, 2.0, 2.0),
            5.0,
            id="pp4-n-le-pp",
        ),
    ],
)
def test_forward_schedule_matrix(
    profile,
    num_microbatches,
    expected_completion,
    expected_busy,
    expected_makespan,
):
    estimate = estimate_forward_pipeline(
        (profile,) * num_microbatches,
        "analytic",
    )

    assert estimate.microbatch_completion_s == pytest.approx(expected_completion)
    assert estimate.stage_busy_s == pytest.approx(expected_busy)
    assert estimate.makespan_s == pytest.approx(expected_makespan)
    assert estimate.first_completion_s == pytest.approx(expected_completion[0])
    assert estimate.warmup_s + estimate.steady_s + estimate.cooldown_s == pytest.approx(estimate.makespan_s)
    expected_bubble = profile.pp_size * expected_makespan - sum(expected_busy)
    assert estimate.aggregate_bubble_s == pytest.approx(expected_bubble)
    assert estimate.bottleneck_stage_id == 0


def test_forward_schedule_serializes_stage_intervals_and_uses_max_payload():
    profile = _profile(3, compute=1.0, comm=0.5, payload_bytes=128)
    estimate = estimate_forward_pipeline((profile, profile), "analytic")

    intervals = sorted(
        estimate.stage_compute_intervals_s[1]
        + estimate.stage_incoming_transfer_intervals_s[1]
        + estimate.stage_outgoing_transfer_intervals_s[1]
    )
    assert all(finish <= next_start for (_, finish), (next_start, _) in zip(intervals, intervals[1:]))
    assert estimate.stage_incoming_payload_bytes_s == (0, 128, 128)
    assert estimate.stage_outgoing_payload_bytes_s == (128, 128, 0)
    assert estimate.stage_communication_buffer_bytes_s == (128, 128, 128)


def test_forward_schedule_rejects_empty_inconsistent_and_missing_model():
    with pytest.raises(ValueError, match="at least one"):
        estimate_forward_pipeline((), "analytic")
    with pytest.raises(ValueError, match="inconsistent"):
        estimate_forward_pipeline((_profile(1), _profile(2)), "analytic")
    with pytest.raises(ValueError, match="missing"):
        estimate_forward_pipeline((_profile(2),), "missing")


@pytest.mark.parametrize(
    ("pp_size", "expected_repeated_makespan", "expected_period", "expected_steady_completions"),
    # With 2 microbatches (K=2): PP=1 and PP=2 are filled (K>=P), period=2.0;
    # PP=4 is under-filled (K<P), so the period is the single-microbatch full
    # traversal (4 stages x 1.0 = 4.0), not the filled-pipeline cycle mean.
    # The steady wave (wave 2) starts when stage 0 finishes wave 1 (2.0) and
    # its completions continue from there.
    [(1, 4.0, 2.0, (3.0, 4.0)), (2, 5.0, 2.0, (4.0, 5.0)), (4, 7.0, 4.0, (6.0, 7.0))],
)
def test_repeated_wave_matrix(pp_size, expected_repeated_makespan, expected_period, expected_steady_completions):
    profile = _profile(pp_size, compute=1.0)
    estimate = estimate_repeated_pipeline((profile, profile), "analytic")

    assert estimate.same_slot_period_s == pytest.approx((expected_period, expected_period))
    assert estimate.worst_tpot_s == pytest.approx(expected_period)
    assert estimate.measured_interval_s == pytest.approx(expected_period)
    assert estimate.repeated_makespan_s == pytest.approx(expected_repeated_makespan)
    assert estimate.completed_tokens == 2
    assert estimate.steady_wave_start_s == pytest.approx(2.0)
    assert estimate.steady_wave_completions_s == pytest.approx(expected_steady_completions)


def test_repeated_wave_steady_offsets_reproduce_wave1_completion_profile():
    # Scheduler invariant for uniform profiles: the saturated closed-batch
    # orbit's per-slot offsets (wave-2 completions relative to the wave's own
    # stage-0 start) equal wave 1's completion profile. Skewed stage
    # distributions do NOT have this equality (a permanent orbit gap remains),
    # which is why the optimizer anchors prefill request-level TTFT on wave 1
    # and uses the orbit only for the steady rate (measured_interval_s). This
    # test pins the uniform-profile orbit property of the exposed diagnostic
    # fields, not a prefill TTFT dependency.
    profile = _profile(4, compute=1.0)
    estimate = estimate_repeated_pipeline((profile, profile, profile), "analytic")

    offsets = tuple(completion - estimate.steady_wave_start_s for completion in estimate.steady_wave_completions_s)
    assert offsets == pytest.approx(estimate.first_wave.microbatch_completion_s)


def test_repeated_wave_under_filled_single_microbatch_respects_full_traversal():
    # K=1 microbatch on a P=2 pipeline: the single microbatch must finish the
    # last stage before re-entering stage 0, so TPOT equals the full traversal
    # (compute0 + comm0 + compute1 = 2.1), not the filled-pipeline cycle mean
    # (1.1). Guards the per-request feedback dependency (zhenyu_zhang P1).
    profile = _profile(2, compute=1.0, comm=0.1)
    estimate = estimate_repeated_pipeline((profile,), "analytic")

    assert estimate.worst_tpot_s == pytest.approx(2.1)
    assert estimate.measured_interval_s == pytest.approx(2.1)


def test_repeated_wave_uses_asymptotic_period_after_nonuniform_warmup():
    profiles = (
        _profile(3, compute=(1.0, 9.0, 14.0), comm=(0.0, 5.0)),
        _profile(3, compute=(15.0, 9.0, 1.0), comm=(1.0, 0.0)),
    )

    four_waves = estimate_forward_pipeline(profiles * 4, "analytic")
    last_slot_completion = four_waves.microbatch_completion_s[1::2]
    last_slot_periods = tuple(
        current - previous for previous, current in zip(last_slot_completion, last_slot_completion[1:])
    )
    estimate = estimate_repeated_pipeline(profiles, "analytic")

    # The forward 8-microbatch schedule converges to the filled-pipeline cycle
    # mean (24.0). The repeated estimate, however, models only 2 microbatches
    # (K=2 < P=3, under-filled): a microbatch cannot re-enter stage 0 until it
    # finishes the last stage, so the period is the single-microbatch full
    # traversal (1+9+5comm+14 = 29.0), not the filled cycle mean (24.0).
    assert last_slot_periods == pytest.approx((25.0, 24.0, 24.0))
    assert estimate.first_wave.microbatch_completion_s == pytest.approx((29.0, 30.0))
    assert estimate.repeated_makespan_s == pytest.approx(55.0)
    assert estimate.same_slot_period_s == pytest.approx((29.0, 29.0))
    assert estimate.worst_tpot_s == pytest.approx(29.0)
    assert estimate.measured_interval_s == pytest.approx(29.0)


PP1_MEMORY = _profile(1, weight=1000, runtime_peak=2000, kv_cache=200)
PP2_UNIFORM = _profile(
    2,
    compute=2.0,
    comm=0.5,
    weight=1000,
    runtime_peak=2000,
    kv_cache=1000,
)
PP2_REMAINDER = _profile(
    2,
    compute=2.0,
    comm=0.5,
    weight=1000,
    runtime_peak=5000,
    kv_cache=250,
)
PP4_LIGHT = _profile(
    4,
    compute=2.0,
    comm=0.5,
    runtime_peak=2000,
    kv_cache=(0, 0, 0, 200),
)
PP4_HEAVY = _profile(
    4,
    compute=2.0,
    comm=0.5,
    runtime_peak=8000,
    kv_cache=(0, 0, 0, 200),
)


@pytest.mark.parametrize(
    (
        "profiles",
        "num_microbatches",
        "resident_microbatches",
        "perf_model_name",
        "resident_policy",
        "device_memory_bytes",
        "expected_compute",
        "expected_exceeds",
    ),
    [
        pytest.param(
            (PP1_MEMORY,),
            3,
            2,
            "analytic",
            "full",
            10_000,
            (2200,),
            False,
            id="pp1-full",
        ),
        pytest.param(
            (PP1_MEMORY,),
            3,
            2,
            "analytic",
            "inflight",
            10_000,
            (2200,),
            False,
            id="pp1-inflight-gate-disabled",
        ),
        pytest.param(
            (PP2_UNIFORM,),
            3,
            2,
            "analytic",
            "inflight",
            10_000,
            (3000, 3000),
            False,
            id="pp2-uniform-n-gt-pp",
        ),
        pytest.param(
            (PP2_UNIFORM, PP2_UNIFORM, PP2_REMAINDER),
            3,
            2,
            "analytic",
            "full",
            10_000,
            (6000, 6000),
            False,
            id="pp2-nonuniform-full",
        ),
        pytest.param(
            (PP2_UNIFORM, PP2_UNIFORM, PP2_REMAINDER),
            3,
            2,
            "analytic",
            "inflight",
            10_000,
            (6000, 5000),
            False,
            id="pp2-nonuniform-inflight",
        ),
        pytest.param(
            (PP2_UNIFORM, PP2_UNIFORM, PP2_REMAINDER),
            3,
            2,
            None,
            "inflight",
            10_000,
            (6000, 6000),
            False,
            id="missing-model-disables-reachability",
        ),
        pytest.param(
            (PP4_LIGHT, PP4_HEAVY),
            2,
            None,
            "analytic",
            "full",
            8100,
            (8000, 8000, 8000, 8200),
            True,
            id="pp4-n-le-pp-full-oom",
        ),
        pytest.param(
            (PP4_LIGHT, PP4_HEAVY),
            2,
            None,
            "analytic",
            "inflight",
            8100,
            (8000, 8000, 8000, 8000),
            False,
            id="pp4-n-le-pp-inflight-reachable",
        ),
    ],
)
def test_stage_memory_behavior_matrix(
    profiles,
    num_microbatches,
    resident_microbatches,
    perf_model_name,
    resident_policy,
    device_memory_bytes,
    expected_compute,
    expected_exceeds,
):
    estimate = estimate_pipeline_stage_memory_across_profiles(
        profiles,
        num_microbatches=num_microbatches,
        resident_microbatches=resident_microbatches,
        device_memory_bytes=device_memory_bytes,
        reserved_memory_bytes=0,
        perf_model_name=perf_model_name,
        resident_policy=resident_policy,
    )

    assert estimate.compute_peak_bytes_s == expected_compute
    assert estimate.exceeds_budget is expected_exceeds


def test_stage_memory_uses_max_not_sum_for_communication_buffer():
    profile = _profile(
        3,
        weight=1000,
        runtime_peak=2000,
        payload_bytes=400,
    )
    estimate = estimate_pipeline_stage_memory(
        profile,
        num_microbatches=1,
        resident_microbatches=1,
        device_memory_bytes=10_000,
    )

    assert estimate.incoming_payload_bytes_s == (0, 400, 400)
    assert estimate.outgoing_payload_bytes_s == (400, 400, 0)
    assert estimate.communication_buffer_bytes_s == (400, 400, 400)
    assert estimate.communication_peak_bytes_s == (1400, 1400, 1400)


@pytest.mark.parametrize(
    ("pp_size", "num_microbatches"),
    [(1, 3), (2, 3), (4, 2)],
)
def test_uniform_profile_matches_expanded_slots(pp_size, num_microbatches):
    profile = _profile(
        pp_size,
        compute=2.0,
        comm=0.5,
        weight=1000,
        runtime_peak=2000,
        kv_cache=500,
    )
    kwargs = dict(
        num_microbatches=num_microbatches,
        resident_microbatches=min(pp_size, num_microbatches),
        device_memory_bytes=10_000,
        perf_model_name="analytic",
        resident_policy="inflight",
    )

    compact = estimate_pipeline_stage_memory_across_profiles((profile,), **kwargs)
    expanded = estimate_pipeline_stage_memory_across_profiles(
        (profile,) * num_microbatches,
        **kwargs,
    )
    assert compact == expanded


def test_single_profile_public_wrapper_matches_sequence_wrapper():
    kwargs = dict(
        num_microbatches=3,
        resident_microbatches=2,
        device_memory_bytes=10_000,
        perf_model_name="analytic",
        resident_policy="inflight",
    )
    assert estimate_pipeline_stage_memory(PP2_UNIFORM, **kwargs) == (
        estimate_pipeline_stage_memory_across_profiles((PP2_UNIFORM,), **kwargs)
    )


MIXED_PREFILL_PP1 = _profile(
    1,
    weight=1000,
    runtime_peak=4000,
    kv_cache=500,
)
MIXED_DECODE_PP1 = _profile(
    1,
    weight=1000,
    runtime_peak=2000,
    kv_cache=1000,
)


@pytest.mark.parametrize(
    (
        "prefill_profiles",
        "decode_profiles",
        "num_microbatches",
        "resident_microbatches",
        "resident_policy",
        "expected_persistent",
        "expected_compute",
        "expected_stage_peak",
    ),
    [
        pytest.param(
            (MIXED_PREFILL_PP1,),
            (MIXED_DECODE_PP1,),
            2,
            2,
            "full",
            (3000,),
            (4500,),
            (6000,),
            id="mixed-pp1-full",
        ),
        pytest.param(
            (MIXED_PREFILL_PP1,),
            (MIXED_DECODE_PP1,),
            2,
            2,
            "inflight",
            (3000,),
            (4500,),
            (6000,),
            id="mixed-pp1-inflight-gate-disabled",
        ),
        pytest.param(
            (PP2_UNIFORM, PP2_UNIFORM, PP2_REMAINDER),
            (PP2_UNIFORM, PP2_UNIFORM, PP2_REMAINDER),
            3,
            2,
            "full",
            (3000, 3000),
            (6000, 6000),
            (7000, 7000),
            id="mixed-pp2-nonuniform-full",
        ),
        pytest.param(
            (PP2_UNIFORM, PP2_UNIFORM, PP2_REMAINDER),
            (PP2_UNIFORM, PP2_UNIFORM, PP2_REMAINDER),
            3,
            2,
            "inflight",
            (3000, 3000),
            (6000, 6000),
            (7000, 7000),
            id="mixed-internal-phases-remain-full",
        ),
    ],
)
def test_mixed_pd_behavior_matrix(
    prefill_profiles,
    decode_profiles,
    num_microbatches,
    resident_microbatches,
    resident_policy,
    expected_persistent,
    expected_compute,
    expected_stage_peak,
):
    estimate = estimate_mixed_pd_stage_memory_across_profiles(
        prefill_profiles,
        decode_profiles,
        num_microbatches=num_microbatches,
        resident_microbatches=resident_microbatches,
        device_memory_bytes=100_000,
        perf_model_name="analytic",
        resident_policy=resident_policy,
    )

    assert estimate.persistent_bytes_s == expected_persistent
    assert estimate.compute_peak_bytes_s == expected_compute
    assert estimate.stage_peak_bytes_s == expected_stage_peak


def test_mixed_public_wrapper_matches_sequence_wrapper():
    kwargs = dict(
        num_microbatches=2,
        resident_microbatches=2,
        device_memory_bytes=100_000,
        perf_model_name="analytic",
        resident_policy="full",
    )
    assert estimate_mixed_pd_stage_memory(
        MIXED_PREFILL_PP1,
        MIXED_DECODE_PP1,
        **kwargs,
    ) == estimate_mixed_pd_stage_memory_across_profiles(
        (MIXED_PREFILL_PP1,),
        (MIXED_DECODE_PP1,),
        **kwargs,
    )


def test_missing_model_is_required_only_when_a_timeline_is_built():
    full = estimate_pipeline_stage_memory_across_profiles(
        (PP2_UNIFORM,),
        num_microbatches=3,
        resident_microbatches=2,
        device_memory_bytes=10_000,
        perf_model_name="missing",
        resident_policy="full",
    )
    assert full.compute_peak_bytes_s == (3000, 3000)

    with pytest.raises(ValueError, match="missing"):
        estimate_pipeline_stage_memory_across_profiles(
            (PP2_UNIFORM,),
            num_microbatches=3,
            resident_microbatches=2,
            device_memory_bytes=10_000,
            perf_model_name="missing",
            resident_policy="inflight",
        )
    with pytest.raises(ValueError, match="missing"):
        estimate_mixed_pd_stage_memory_across_profiles(
            (PP2_UNIFORM,),
            (PP2_UNIFORM,),
            num_microbatches=3,
            resident_microbatches=2,
            device_memory_bytes=10_000,
            perf_model_name="missing",
            resident_policy="inflight",
        )
    with pytest.raises(ValueError, match="missing"):
        estimate_repeated_pipeline((PP2_UNIFORM,), "missing")


@pytest.mark.parametrize("resident_microbatches", [0, -1, 4])
def test_stage_memory_rejects_invalid_resident_window(resident_microbatches):
    with pytest.raises(ValueError, match="resident_microbatches"):
        estimate_pipeline_stage_memory_across_profiles(
            (PP2_UNIFORM,),
            num_microbatches=3,
            resident_microbatches=resident_microbatches,
            device_memory_bytes=10_000,
        )


def test_stage_memory_rejects_invalid_policy_and_ambiguous_profiles():
    with pytest.raises(ValueError, match="resident_policy"):
        estimate_pipeline_stage_memory_across_profiles(
            (PP2_UNIFORM,),
            num_microbatches=3,
            device_memory_bytes=10_000,
            resident_policy="in-flight",
        )

    error = "either one repeated profile or exactly one profile per microbatch slot"
    with pytest.raises(ValueError, match=error):
        estimate_pipeline_stage_memory_across_profiles(
            (PP2_UNIFORM, PP2_REMAINDER),
            num_microbatches=3,
            device_memory_bytes=10_000,
        )
    with pytest.raises(ValueError, match=error):
        estimate_mixed_pd_stage_memory_across_profiles(
            (PP2_UNIFORM, PP2_REMAINDER),
            (PP2_UNIFORM, PP2_REMAINDER),
            num_microbatches=3,
            device_memory_bytes=10_000,
        )
