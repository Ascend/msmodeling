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

"""Forward-only pipeline schedule estimator.

Models a blocking, no-overlap forward pipeline: each stage owns one serialized
execution resource, and a stage's outgoing communication is part of its service
time (the source stage cannot start its next microbatch until its outgoing
transfer completes). This licenses ``communication_buffer = max(incoming,
outgoing)`` rather than their sum, because a stage never holds incoming and
outgoing payloads simultaneously.

The estimator consumes one ``PipelineProfile`` per microbatch shape and a
performance model name, and produces a ``PipelineScheduleEstimate`` with an
exact timeline, phase decomposition (warmup/steady/cooldown around the
bottleneck stage), aggregate bubble, and per-stage communication buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

from tensor_cast.pipeline_parallel import PipelineProfile


@dataclass(frozen=True)
class PipelineScheduleEstimate:
    """Result of a forward-only blocking pipeline schedule.

    All time fields are in seconds. Per-stage tuples are indexed by ``stage_id``
    (stage 0 is the first stage).

    - ``makespan_s``: wall-clock time from the first microbatch entering stage 0
      to the last microbatch leaving the final stage.
    - ``microbatch_completion_s``: finish time of each microbatch on the final
      stage, one entry per input microbatch.
    - ``bottleneck_stage_id``: the stage with the maximum total busy time. Ties
      are broken toward the lowest ``stage_id``.
    - ``warmup_s`` / ``steady_s`` / ``cooldown_s``: phases defined around the
      bottleneck stage so they are additive
      (``warmup + steady + cooldown == makespan``). The phases span the
      bottleneck stage's full occupancy (compute + incoming transfer + outgoing
      transfer, not compute alone): ``warmup`` is the earliest start of any
      bottleneck interval; ``steady`` is the latest finish minus warmup;
      ``cooldown`` is makespan minus that finish. A stage bottlenecked by
      communication is still described correctly.
    - ``aggregate_bubble_s`` / ``bubble_ratio``: aggregate stage idle capacity
      (``pp_size * makespan - sum(stage_busy)``), NOT wall-clock idle time. The
      ratio is ``aggregate_bubble / (pp_size * makespan)``.
    - ``stage_busy_s`` / ``stage_idle_s``: per-stage busy time and idle time
      (``makespan - stage_busy``). ``stage_busy`` counts compute plus every
      transfer the stage participates in (outgoing as source AND incoming as
      target), because a transfer occupies both ends under the blocking
      contract.
    - ``stage_compute_intervals_s``: per-stage list of ``(start, finish)``
      compute intervals, one per microbatch.
    - ``stage_outgoing_transfer_intervals_s`` / ``stage_incoming_transfer_intervals_s``:
      per-stage transfer intervals. A boundary transfer appears once on its
      source (outgoing) and once on its target (incoming) with the same
      ``(start, finish)``. Under the rendezvous blocking contract, on any given
      stage the compute, outgoing-transfer, and incoming-transfer intervals are
      pairwise disjoint (serialized resource), which licenses
      ``communication_buffer = max(incoming, outgoing)``.
    - ``stage_incoming_payload_bytes_s`` / ``stage_outgoing_payload_bytes_s``:
      per-stage payload bytes of the incoming (upstream) and outgoing
      (downstream) boundary transfers, aggregated as the max across microbatch
      profiles.
    - ``stage_communication_buffer_bytes_s``: per-stage communication buffer,
      ``max(incoming, outgoing)`` under the no-overlap blocking contract.
    """

    makespan_s: float
    first_completion_s: float
    warmup_s: float
    steady_s: float
    cooldown_s: float
    aggregate_bubble_s: float
    bubble_ratio: float
    bottleneck_stage_id: int
    stage_busy_s: tuple[float, ...]
    stage_idle_s: tuple[float, ...]
    microbatch_completion_s: tuple[float, ...]
    stage_compute_intervals_s: tuple[tuple[tuple[float, float], ...], ...]
    stage_outgoing_transfer_intervals_s: tuple[tuple[tuple[float, float], ...], ...]
    stage_incoming_transfer_intervals_s: tuple[tuple[tuple[float, float], ...], ...]
    stage_incoming_payload_bytes_s: tuple[int, ...]
    stage_outgoing_payload_bytes_s: tuple[int, ...]
    stage_communication_buffer_bytes_s: tuple[int, ...]


def _stage_service_time(stage, model_name: str) -> tuple[float, float]:
    """Return (compute_s, outgoing_comm_s) for a stage under the given model.

    Raises ``ValueError`` if the model key is missing from either the compute
    or the outgoing-comm dict; the scheduler must not fall back to another
    model's time or silently treat a missing key as zero.
    """
    if model_name not in stage.compute_time_s_by_model:
        raise ValueError(
            f"pipeline stage {stage.stage_id} has no compute time for performance model "
            f"{model_name!r}; cannot schedule forward pipeline"
        )
    if model_name not in stage.outgoing_comm_time_s_by_model:
        raise ValueError(
            f"pipeline stage {stage.stage_id} has no outgoing comm time for performance model "
            f"{model_name!r}; cannot schedule forward pipeline"
        )
    compute_s = float(stage.compute_time_s_by_model[model_name])
    outgoing_comm_s = float(stage.outgoing_comm_time_s_by_model[model_name])
    return compute_s, outgoing_comm_s


def estimate_forward_pipeline(
    microbatch_profiles: tuple[PipelineProfile, ...] | list[PipelineProfile],
    perf_model_name: str,
) -> PipelineScheduleEstimate:
    """Estimate a forward-only, no-overlap pipeline schedule.

    Each entry in ``microbatch_profiles`` is one microbatch's ``PipelineProfile``.
    All profiles must share the same ``pp_size`` and stage count. The
    performance model name selects the per-stage compute/comm times; a missing
    key is a hard error (no fallback to another model).
    """
    if not microbatch_profiles:
        raise ValueError("at least one microbatch profile is required for a forward pipeline schedule")
    if perf_model_name is None:
        raise ValueError("a performance model name is required for a forward pipeline schedule")

    pp_size = microbatch_profiles[0].pp_size
    num_stages = len(microbatch_profiles[0].stages)
    if pp_size != num_stages:
        raise ValueError(f"pp_size ({pp_size}) must equal the number of stages ({num_stages})")
    expected_stage_ids = list(range(num_stages))
    for index, profile in enumerate(microbatch_profiles):
        if profile.pp_size != pp_size or len(profile.stages) != num_stages:
            raise ValueError(
                f"inconsistent pp_size or stage count across microbatch profiles: "
                f"profile {index} has pp_size={profile.pp_size}, "
                f"{len(profile.stages)} stages; expected pp_size={pp_size}, {num_stages} stages"
            )
        actual_stage_ids = [stage.stage_id for stage in profile.stages]
        if actual_stage_ids != expected_stage_ids:
            raise ValueError(
                f"microbatch profile {index} has non-contiguous stage ids {actual_stage_ids}; "
                f"expected {expected_stage_ids}"
            )

    # Per-stage incoming/outgoing payload bytes, aggregated across ALL
    # microbatch profiles (different shapes may carry different payloads). Take
    # the max per stage so the buffer covers the largest microbatch shape.
    incoming_payload: list[int] = [0] * num_stages
    outgoing_payload: list[int] = [0] * num_stages
    for profile in microbatch_profiles:
        for transfer in profile.transfers:
            if 0 <= transfer.target_stage_id < num_stages:
                incoming_payload[transfer.target_stage_id] = max(
                    incoming_payload[transfer.target_stage_id],
                    int(transfer.payload_bytes),
                )
            if 0 <= transfer.source_stage_id < num_stages:
                outgoing_payload[transfer.source_stage_id] = max(
                    outgoing_payload[transfer.source_stage_id],
                    int(transfer.payload_bytes),
                )
        for stage in profile.stages:
            if 0 <= stage.stage_id < num_stages:
                outgoing_payload[stage.stage_id] = max(
                    outgoing_payload[stage.stage_id], int(stage.outgoing_payload_bytes)
                )

    communication_buffer = [max(incoming_payload[s], outgoing_payload[s]) for s in range(num_stages)]

    wave = _schedule_wave(
        microbatch_profiles,
        perf_model_name,
        num_stages,
        initial_free=[0.0] * num_stages,
    )

    makespan = wave.final_free[-1] if num_stages > 0 else 0.0
    if wave.microbatch_completion:
        makespan = max(makespan, max(wave.microbatch_completion))

    return _build_schedule_estimate(
        wave=wave,
        pp_size=pp_size,
        makespan=makespan,
        incoming_payload=incoming_payload,
        outgoing_payload=outgoing_payload,
        communication_buffer=communication_buffer,
    )


@dataclass(frozen=True)
class _WaveSchedule:
    """Per-wave forward timeline state (internal, reused for repeated waves)."""

    microbatch_completion: tuple[float, ...]
    final_free: tuple[float, ...]
    stage_busy: tuple[float, ...]
    compute_intervals: tuple[tuple[tuple[float, float], ...], ...]
    outgoing_transfer_intervals: tuple[tuple[tuple[float, float], ...], ...]
    incoming_transfer_intervals: tuple[tuple[tuple[float, float], ...], ...]


def _schedule_wave(
    microbatch_profiles,
    perf_model_name: str,
    num_stages: int,
    *,
    initial_free: list[float],
) -> _WaveSchedule:
    """Schedule one wave of microbatches starting from ``initial_free``.

    Rendezvous blocking: each stage owns one serialized resource. A transfer
    occupies both its source (send) and target (recv) for its full duration,
    and does not start until the target is free to receive (so a payload never
    queues at the target). ``free[s]`` is the next time stage s can begin
    compute; a transfer from s to s+1 sets ``free[s] = transfer_finish``
    (source blocked until transfer done) and ``free[s+1] = transfer_finish``
    (target's compute starts right after the incoming transfer).

    The returned ``final_free`` lets a caller schedule a follow-on wave without
    resetting stage availability (used by repeated-wave decode estimation).
    """
    free = list(initial_free)
    compute_intervals: list[list[tuple[float, float]]] = [[] for _ in range(num_stages)]
    outgoing_transfer_intervals: list[list[tuple[float, float]]] = [[] for _ in range(num_stages)]
    incoming_transfer_intervals: list[list[tuple[float, float]]] = [[] for _ in range(num_stages)]
    stage_busy = [0.0] * num_stages
    microbatch_completion: list[float] = []

    for profile in microbatch_profiles:
        for stage_id, stage in enumerate(profile.stages):
            compute_s, outgoing_comm_s = _stage_service_time(stage, perf_model_name)
            compute_start = free[stage_id]
            compute_finish = compute_start + compute_s
            compute_intervals[stage_id].append((compute_start, compute_finish))
            stage_busy[stage_id] += compute_s
            free[stage_id] = compute_finish

            is_last_stage = stage_id == num_stages - 1
            if not is_last_stage and outgoing_comm_s > 0.0:
                transfer_start = max(compute_finish, free[stage_id + 1])
                transfer_finish = transfer_start + outgoing_comm_s
                outgoing_transfer_intervals[stage_id].append((transfer_start, transfer_finish))
                incoming_transfer_intervals[stage_id + 1].append((transfer_start, transfer_finish))
                stage_busy[stage_id] += outgoing_comm_s
                stage_busy[stage_id + 1] += outgoing_comm_s
                free[stage_id] = transfer_finish
                free[stage_id + 1] = transfer_finish
            elif not is_last_stage:
                # Zero-duration transfer: still must respect target readiness,
                # but no time elapses. Advance target readiness to compute_finish
                # if it was earlier (payload arrives with no comm cost).
                free[stage_id + 1] = max(free[stage_id + 1], compute_finish)

            if is_last_stage:
                microbatch_completion.append(compute_finish)

    return _WaveSchedule(
        microbatch_completion=tuple(microbatch_completion),
        final_free=tuple(free),
        stage_busy=tuple(stage_busy),
        compute_intervals=tuple(tuple(ivs) for ivs in compute_intervals),
        outgoing_transfer_intervals=tuple(tuple(ivs) for ivs in outgoing_transfer_intervals),
        incoming_transfer_intervals=tuple(tuple(ivs) for ivs in incoming_transfer_intervals),
    )


def _build_schedule_estimate(
    *,
    wave: _WaveSchedule,
    pp_size: int,
    makespan: float,
    incoming_payload: list[int],
    outgoing_payload: list[int],
    communication_buffer: list[int],
) -> PipelineScheduleEstimate:
    """Assemble a PipelineScheduleEstimate from one scheduled wave."""
    num_stages = pp_size
    stage_busy = list(wave.stage_busy)

    bottleneck_stage_id = 0
    max_busy = stage_busy[0] if num_stages > 0 else 0.0
    for s in range(1, num_stages):
        if stage_busy[s] > max_busy:
            max_busy = stage_busy[s]
            bottleneck_stage_id = s

    if num_stages > 0:
        bottleneck_intervals = (
            list(wave.compute_intervals[bottleneck_stage_id])
            + list(wave.incoming_transfer_intervals[bottleneck_stage_id])
            + list(wave.outgoing_transfer_intervals[bottleneck_stage_id])
        )
    else:
        bottleneck_intervals = []
    if bottleneck_intervals:
        bottleneck_first_start = min(iv[0] for iv in bottleneck_intervals)
        bottleneck_last_finish = max(iv[1] for iv in bottleneck_intervals)
    else:
        bottleneck_first_start = 0.0
        bottleneck_last_finish = 0.0
    warmup = bottleneck_first_start
    steady = bottleneck_last_finish - warmup
    cooldown = makespan - bottleneck_last_finish

    aggregate_capacity = pp_size * makespan
    aggregate_busy = sum(stage_busy)
    aggregate_bubble = max(0.0, aggregate_capacity - aggregate_busy)
    bubble_ratio = aggregate_bubble / aggregate_capacity if aggregate_capacity > 0 else 0.0

    stage_idle = [max(0.0, makespan - busy) for busy in stage_busy]
    first_completion = wave.microbatch_completion[0] if wave.microbatch_completion else 0.0

    return PipelineScheduleEstimate(
        makespan_s=makespan,
        first_completion_s=first_completion,
        warmup_s=warmup,
        steady_s=steady,
        cooldown_s=cooldown,
        aggregate_bubble_s=aggregate_bubble,
        bubble_ratio=bubble_ratio,
        bottleneck_stage_id=bottleneck_stage_id,
        stage_busy_s=tuple(stage_busy),
        stage_idle_s=tuple(stage_idle),
        microbatch_completion_s=wave.microbatch_completion,
        stage_compute_intervals_s=wave.compute_intervals,
        stage_outgoing_transfer_intervals_s=wave.outgoing_transfer_intervals,
        stage_incoming_transfer_intervals_s=wave.incoming_transfer_intervals,
        stage_incoming_payload_bytes_s=tuple(incoming_payload),
        stage_outgoing_payload_bytes_s=tuple(outgoing_payload),
        stage_communication_buffer_bytes_s=tuple(communication_buffer),
    )


@dataclass(frozen=True)
class RepeatedPipelineEstimate:
    """Repeated-wave steady-state estimate for a forward pipeline.

    Two identical waves are scheduled without resetting stage availability, so
    wave 2 queues behind wave 1's tail. The steady-state wave period is
    computed from the max-plus free-state transition rather than assuming the
    wave-1/wave-2 interval has already converged.
    ``same_slot_period_s`` contains that steady-state period for every
    microbatch slot, and ``worst_tpot_s`` is its maximum (the SLO-relevant
    TPOT).

    Decode consumes the period fields. The steady-wave completion profile
    (``steady_wave_completions_s`` relative to ``steady_wave_start_s``) is a
    diagnostic of the saturated closed-batch orbit (identical for every wave
    from wave 2 on): it is NOT the data source for prefill request-level
    TTFT — the throughput optimizer anchors prefill TTFT on wave 1's
    completions (the no-queue anchor matching production's continuous
    injection) and takes the steady rate from ``measured_interval_s``. See
    ``BaseThroughputOptimizer._evaluate_pp_wave`` for the full rationale.

    - ``completed_tokens``: number of decode STEPS completed in one wave
      (= number of microbatches in the wave). Each step produces one token per
      request in its microbatch, so the raw decoded-token count is
      ``sum(microbatch_sizes)``; callers that know the microbatch sizes (the
      optimizer does, via ``batch_size``) should use that for token throughput
      rather than this field, which counts steps for shape-agnostic consumers.
    - ``measured_interval_s``: steady-state wave period = the last microbatch
      slot's asymptotic same-slot completion period. This pairs exactly one
      wave's decoded tokens with one steady-state wave period, so
      ``batch_size * dp / measured_interval_s`` is the steady-state decode
      throughput and excludes wave-1 warmup.
    - ``repeated_makespan_s``: full two-wave wall-clock span from t=0 to wave
      2's last completion (diagnostic; NOT the throughput interval).
    - ``steady_wave_completions_s``: finish time of each microbatch on the
      final stage in wave 2, one entry per input microbatch. Wave 2 continues
      from wave 1's ``final_free``, so these absolute timestamps include
      queueing behind wave 1's tail. Diagnostic only (saturated closed-batch
      orbit); not consumed as the prefill TTFT data source.
    - ``steady_wave_start_s``: nominal start of wave 2 = wave 1's
      ``final_free[0]``, the earliest time the first wave-2 microbatch can
      begin stage-0 compute. ``completion - steady_wave_start_s`` is one
      microbatch's latency within its own steady wave (wave-1 warmup
      excluded). Diagnostic only; for uniform profiles these offsets equal
      wave 1's completion profile, while skewed stage distributions leave a
      permanent orbit gap that the prefill TTFT anchor deliberately excludes.
    """

    first_wave: PipelineScheduleEstimate
    repeated_makespan_s: float
    same_slot_period_s: tuple[float, ...]
    worst_tpot_s: float
    completed_tokens: int
    measured_interval_s: float
    steady_wave_completions_s: tuple[float, ...]
    steady_wave_start_s: float


def split_batch_size(batch_size: int, microbatch_size: int) -> tuple[int, ...]:
    """Split ``batch_size`` into uniform microbatches of ``microbatch_size``.

    Returns a tuple of microbatch sizes: ``full`` copies of ``microbatch_size``
    plus one remainder microbatch when ``batch_size`` is not evenly divisible.
    Both arguments must be positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if microbatch_size <= 0:
        raise ValueError(f"microbatch_size must be positive, got {microbatch_size}")
    full, remainder = divmod(batch_size, microbatch_size)
    result = (microbatch_size,) * full
    return result + ((remainder,) if remainder else ())


def estimate_repeated_pipeline(
    microbatch_profiles: tuple[PipelineProfile, ...] | list[PipelineProfile],
    perf_model_name: str,
) -> RepeatedPipelineEstimate:
    """Estimate steady-state quantities for indefinitely repeated identical waves.

    Wave 1 schedules from idle stages (``free = [0, ...]``). Wave 2 schedules
    the same microbatch sequence continuing from wave 1's ``final_free``, so
    each wave-2 microbatch queues behind the wave-1 tail. Their combined
    makespan is retained as a warmup diagnostic. The steady-state period is the
    maximum cycle mean of the max-plus transition for one whole wave, avoiding
    the invalid assumption that the first repeated interval is already steady.
    The wave-2 completion profile relative to wave 1's ``final_free[0]`` gives
    each microbatch's latency within the saturated closed-batch orbit; it is
    exposed for diagnostics and regression checks, not as the prefill TTFT
    data source (the optimizer anchors prefill TTFT on wave 1 and takes the
    steady rate from ``measured_interval_s``).

    Requires the same consistency as ``estimate_forward_pipeline`` (uniform
    ``pp_size``/stage count, contiguous stage ids, present perf-model keys).
    """
    if not microbatch_profiles:
        raise ValueError("at least one microbatch profile is required for a repeated pipeline schedule")
    if perf_model_name is None:
        raise ValueError("a performance model name is required for a repeated pipeline schedule")
    pp_size = microbatch_profiles[0].pp_size
    num_stages = len(microbatch_profiles[0].stages)
    if pp_size != num_stages:
        raise ValueError(f"pp_size ({pp_size}) must equal the number of stages ({num_stages})")
    expected_stage_ids = list(range(num_stages))
    for index, profile in enumerate(microbatch_profiles):
        if profile.pp_size != pp_size or len(profile.stages) != num_stages:
            raise ValueError(
                f"inconsistent pp_size or stage count across microbatch profiles: "
                f"profile {index} has pp_size={profile.pp_size}, "
                f"{len(profile.stages)} stages; expected pp_size={pp_size}, {num_stages} stages"
            )
        actual_stage_ids = [stage.stage_id for stage in profile.stages]
        if actual_stage_ids != expected_stage_ids:
            raise ValueError(
                f"microbatch profile {index} has non-contiguous stage ids {actual_stage_ids}; "
                f"expected {expected_stage_ids}"
            )

    wave1 = _schedule_wave(
        microbatch_profiles,
        perf_model_name,
        num_stages,
        initial_free=[0.0] * num_stages,
    )
    wave2 = _schedule_wave(
        microbatch_profiles,
        perf_model_name,
        num_stages,
        initial_free=list(wave1.final_free),
    )

    steady_wave_period = _estimate_steady_wave_period(
        microbatch_profiles,
        perf_model_name,
        num_stages,
    )
    # Under-filled pipeline (fewer microbatches than stages): a microbatch
    # cannot re-enter stage 0 until it has completed the last stage of its
    # previous pass, so the steady-state period is bounded below by one
    # microbatch's full pipeline traversal. The max-plus cycle mean assumes a
    # filled pipeline and under-reports TPOT when K < P (e.g. one microbatch on
    # a 2-stage pipeline reports the stage cycle mean instead of the full
    # traversal). The per-microbatch wave-2 re-entry feedback (next-round
    # first-stage readiness per microbatch) is a separate refinement tracked
    # for the repeated_makespan diagnostic; it requires per-microbatch lower
    # bounds in _schedule_wave.
    if len(microbatch_profiles) < num_stages:
        single_mb_latency = 0.0
        for s_id, stage in enumerate(microbatch_profiles[0].stages):
            c_s, comm_s = _stage_service_time(stage, perf_model_name)
            single_mb_latency += c_s
            if s_id != num_stages - 1:
                single_mb_latency += comm_s
        steady_wave_period = max(steady_wave_period, single_mb_latency)
    same_slot_period = (steady_wave_period,) * len(wave1.microbatch_completion)
    worst_tpot = max(same_slot_period) if same_slot_period else 0.0
    # This pairs one wave's decoded tokens with the asymptotic wave period.
    # Keep the full two-wave span separately as a warmup diagnostic.
    measured_interval = steady_wave_period if wave1.microbatch_completion else 0.0
    repeated_makespan = wave2.microbatch_completion[-1] if wave2.microbatch_completion else 0.0

    # Build the public first-wave estimate directly from the already-computed
    # wave1, avoiding a redundant _schedule_wave call that estimate_forward_pipeline
    # would perform internally (it uses the same initial_free=[0.0]*num_stages).
    incoming_payload: list[int] = [0] * num_stages
    outgoing_payload: list[int] = [0] * num_stages
    for profile in microbatch_profiles:
        for transfer in profile.transfers:
            if 0 <= transfer.target_stage_id < num_stages:
                incoming_payload[transfer.target_stage_id] = max(
                    incoming_payload[transfer.target_stage_id],
                    int(transfer.payload_bytes),
                )
            if 0 <= transfer.source_stage_id < num_stages:
                outgoing_payload[transfer.source_stage_id] = max(
                    outgoing_payload[transfer.source_stage_id],
                    int(transfer.payload_bytes),
                )
        for stage in profile.stages:
            if 0 <= stage.stage_id < num_stages:
                outgoing_payload[stage.stage_id] = max(
                    outgoing_payload[stage.stage_id], int(stage.outgoing_payload_bytes)
                )
    communication_buffer = [max(incoming_payload[s], outgoing_payload[s]) for s in range(num_stages)]
    makespan = wave1.final_free[-1] if num_stages > 0 else 0.0
    if wave1.microbatch_completion:
        makespan = max(makespan, max(wave1.microbatch_completion))
    first_wave = _build_schedule_estimate(
        wave=wave1,
        pp_size=microbatch_profiles[0].pp_size,
        makespan=makespan,
        incoming_payload=incoming_payload,
        outgoing_payload=outgoing_payload,
        communication_buffer=communication_buffer,
    )

    return RepeatedPipelineEstimate(
        first_wave=first_wave,
        repeated_makespan_s=repeated_makespan,
        same_slot_period_s=same_slot_period,
        worst_tpot_s=worst_tpot,
        completed_tokens=len(microbatch_profiles),
        measured_interval_s=measured_interval,
        steady_wave_completions_s=wave2.microbatch_completion,
        steady_wave_start_s=wave1.final_free[0],
    )


def _estimate_steady_wave_period(
    microbatch_profiles: tuple[PipelineProfile, ...] | list[PipelineProfile],
    perf_model_name: str,
    num_stages: int,
) -> float:
    """Return the exact asymptotic period of the repeated-wave schedule.

    A complete wave maps the stage availability vector ``x`` to ``A ⊗ x``:
    additions in the scheduler become max-plus edge weights and readiness
    joins become max operations. The asymptotic growth per wave is therefore
    the maximum cycle mean of ``A``. Karp's dynamic-programming formula
    computes that value without relying on a finite warmup heuristic.
    """
    if num_stages == 0:
        return 0.0

    negative_infinity = float("-inf")
    free_state = [
        [0.0 if source_stage == target_stage else negative_infinity for source_stage in range(num_stages)]
        for target_stage in range(num_stages)
    ]

    for profile in microbatch_profiles:
        for stage_id, stage in enumerate(profile.stages):
            compute_s, outgoing_comm_s = _stage_service_time(stage, perf_model_name)
            compute_finish = [value + compute_s for value in free_state[stage_id]]
            free_state[stage_id] = compute_finish

            is_last_stage = stage_id == num_stages - 1
            if is_last_stage:
                continue

            transfer_start = [
                max(compute_finish[source_stage], free_state[stage_id + 1][source_stage])
                for source_stage in range(num_stages)
            ]
            if outgoing_comm_s > 0.0:
                transfer_finish = [value + outgoing_comm_s for value in transfer_start]
                free_state[stage_id] = transfer_finish
                free_state[stage_id + 1] = transfer_finish
            else:
                free_state[stage_id + 1] = transfer_start

    # best_path_weight[k][v] is the maximum weight of any k-edge path
    # ending at v. Initializing every vertex to zero is equivalent to adding
    # a zero-weight super-source and covers reducible transition graphs.
    best_path_weight = [[0.0] * num_stages]
    for _ in range(num_stages):
        previous = best_path_weight[-1]
        current = [
            max(previous[source_stage] + free_state[target_stage][source_stage] for source_stage in range(num_stages))
            for target_stage in range(num_stages)
        ]
        best_path_weight.append(current)

    cycle_mean_by_target = []
    for target_stage in range(num_stages):
        final_weight = best_path_weight[num_stages][target_stage]
        lower_bounds = [
            (final_weight - best_path_weight[path_length][target_stage]) / (num_stages - path_length)
            for path_length in range(num_stages)
            if best_path_weight[path_length][target_stage] != negative_infinity
        ]
        if lower_bounds:
            cycle_mean_by_target.append(min(lower_bounds))

    return max(cycle_mean_by_target, default=0.0)


@dataclass(frozen=True)
class PipelineStageMemoryEstimate:
    """Rank-aware per-stage memory estimate under the blocking contract.

    Each per-stage tuple is indexed by ``stage_id``. ``stage_peak_bytes_s`` is
    ``max(compute_peak, communication_peak)`` per stage; ``exceeds_budget`` is
    True when any stage peak exceeds ``device_memory_bytes - reserved_memory_bytes``.
    ``bottleneck_stage_id`` is the stage with the least remaining memory (ties
    broken toward the lowest ``stage_id``), which is also the stage reported on
    eviction.

    The communication buffer is ``max(incoming, outgoing)`` — never their sum
    and never the transfer Runtime's combined ``2 * payload`` peak — because
    under the blocking contract a stage never holds incoming and outgoing
    payloads simultaneously.
    """

    persistent_bytes_s: tuple[int, ...]
    compute_peak_bytes_s: tuple[int, ...]
    communication_buffer_bytes_s: tuple[int, ...]
    communication_peak_bytes_s: tuple[int, ...]
    stage_peak_bytes_s: tuple[int, ...]
    incoming_payload_bytes_s: tuple[int, ...]
    outgoing_payload_bytes_s: tuple[int, ...]
    remaining_bytes_s: tuple[int, ...]
    bottleneck_stage_id: int
    exceeds_budget: bool


def _materialize_slot_profiles(
    profiles: tuple[PipelineProfile, ...] | list[PipelineProfile],
    num_microbatches: int,
) -> tuple[PipelineProfile, ...]:
    """Validate a profile sequence and return one profile per logical slot."""
    if not profiles:
        raise ValueError("at least one microbatch profile is required for stage memory estimation")
    if num_microbatches <= 0:
        raise ValueError(f"num_microbatches must be positive, got {num_microbatches}")
    if len(profiles) not in (1, num_microbatches):
        raise ValueError(
            "profiles must contain either one repeated profile or exactly one "
            f"profile per microbatch slot; got {len(profiles)} profiles for "
            f"{num_microbatches} microbatches. Expand compressed distinct-shape "
            "profiles to their logical slot order."
        )
    if len(profiles) == 1:
        return (profiles[0],) * num_microbatches
    return profiles if isinstance(profiles, tuple) else tuple(profiles)


def _build_reachability_timeline(
    slot_profiles: tuple[PipelineProfile, ...],
    perf_model_name: str | None,
    resident_policy: str,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]] | None:
    """Build the timeline used to remove completed in-flight caches."""
    if perf_model_name is None or slot_profiles[0].pp_size < 2 or resident_policy != "inflight":
        return None

    schedule = estimate_forward_pipeline(slot_profiles, perf_model_name)
    return (
        tuple(
            tuple(interval[0] for interval in stage_intervals) for stage_intervals in schedule.stage_compute_intervals_s
        ),
        schedule.microbatch_completion_s,
    )


def _sum_stage_cache(
    slot_profiles: tuple[PipelineProfile, ...],
    slots: range | list[int],
    stage_id: int,
) -> tuple[int, int]:
    """Return summed KV and indexer cache bytes for one stage and slot set."""
    kv_cache = 0
    indexer_cache = 0
    for slot in slots:
        stage = slot_profiles[slot].stages[stage_id]
        kv_cache += int(stage.kv_cache_bytes)
        indexer_cache += int(stage.indexer_cache_bytes)
    return kv_cache, indexer_cache


def _aggregate_payloads(
    profiles: tuple[PipelineProfile, ...],
    num_stages: int,
) -> tuple[list[int], list[int]]:
    """Return the largest incoming and outgoing payload for every stage."""
    incoming = [0] * num_stages
    outgoing = [0] * num_stages
    for profile in profiles:
        for transfer in profile.transfers:
            if 0 <= transfer.target_stage_id < num_stages:
                incoming[transfer.target_stage_id] = max(
                    incoming[transfer.target_stage_id],
                    int(transfer.payload_bytes),
                )
        for stage in profile.stages:
            if 0 <= stage.stage_id < num_stages:
                outgoing[stage.stage_id] = max(
                    outgoing[stage.stage_id],
                    int(stage.outgoing_payload_bytes),
                )
    return incoming, outgoing


def _evaluate_stage_memory(
    profiles: tuple[PipelineProfile, ...] | list[PipelineProfile],
    *,
    num_microbatches: int,
    device_memory_bytes: int,
    reserved_memory_bytes: int,
    resident_microbatches: int | None,
    perf_model_name: str | None,
    resident_policy: str | None,
    prefill_profiles: tuple[PipelineProfile, ...] | list[PipelineProfile] | None = None,
) -> PipelineStageMemoryEstimate:
    """Evaluate single-phase or mixed-PD stage memory through one phase loop.

    A regular estimate has one phase whose resident and runtime sequences are
    identical. Mixed-PD has prefill-full, decode-full, and mixed-overlap phases;
    only the overlap phase pairs decode-resident profiles with prefill runtime
    profiles and honors the caller's resident policy.
    """
    is_mixed = prefill_profiles is not None
    if not is_mixed:
        slot_profiles = _materialize_slot_profiles(profiles, num_microbatches)
    if resident_policy is None:
        resident_policy = "full"
    if resident_policy not in ("inflight", "full"):
        raise ValueError(f"resident_policy must be 'inflight' or 'full', got {resident_policy!r}")
    if is_mixed:
        prefill_slot_profiles = _materialize_slot_profiles(
            prefill_profiles,
            num_microbatches,
        )
        slot_profiles = _materialize_slot_profiles(profiles, num_microbatches)
    else:
        prefill_slot_profiles = None

    if resident_microbatches is not None:
        if resident_microbatches <= 0:
            raise ValueError(f"resident_microbatches must be positive, got {resident_microbatches}")
        if resident_microbatches > num_microbatches:
            raise ValueError(
                f"resident_microbatches ({resident_microbatches}) cannot exceed num_microbatches ({num_microbatches})"
            )

    source_profiles = profiles if isinstance(profiles, tuple) else tuple(profiles)
    if prefill_slot_profiles is None:
        phase_inputs = (
            (
                source_profiles,
                slot_profiles,
                slot_profiles,
                resident_policy,
                False,
            ),
        )
    else:
        prefill_source_profiles = prefill_profiles if isinstance(prefill_profiles, tuple) else tuple(prefill_profiles)
        phase_inputs = (
            (
                prefill_source_profiles,
                prefill_slot_profiles,
                prefill_slot_profiles,
                "full",
                False,
            ),
            (
                source_profiles,
                slot_profiles,
                slot_profiles,
                "full",
                False,
            ),
            (
                prefill_source_profiles + source_profiles,
                slot_profiles,
                prefill_slot_profiles,
                resident_policy,
                True,
            ),
        )

    phase_persistent: list[list[int]] = []
    phase_compute_peak: list[list[int]] = []
    phase_comm_buffer: list[list[int]] = []
    phase_comm_peak: list[list[int]] = []
    phase_stage_peak: list[list[int]] = []
    phase_incoming: list[list[int]] = []
    phase_outgoing: list[list[int]] = []

    for (
        phase_profiles,
        resident_profiles,
        runtime_profiles,
        phase_policy,
        mixed_overlap,
    ) in phase_inputs:
        num_stages = len(resident_profiles[0].stages)
        resident_count = (
            resident_microbatches
            if resident_microbatches is not None
            else min(resident_profiles[0].pp_size, num_microbatches)
        )
        incoming, outgoing = _aggregate_payloads(phase_profiles, num_stages)
        communication_buffer = [max(incoming[stage_id], outgoing[stage_id]) for stage_id in range(num_stages)]
        timeline = _build_reachability_timeline(
            resident_profiles,
            perf_model_name,
            phase_policy,
        )
        weight_profile = (
            resident_profiles[0]
            if mixed_overlap
            else max(
                phase_profiles,
                key=lambda profile: int(profile.stages[0].runtime_peak_bytes),
            )
        )

        persistent = [0] * num_stages
        compute_peak = [0] * num_stages
        communication_peak = [0] * num_stages
        stage_peak = [0] * num_stages
        window_starts = num_microbatches - resident_count + 1

        for stage_id in range(num_stages):
            weight = int(weight_profile.stages[stage_id].weight_bytes)
            mixed_runtime_peak = (
                max(int(profile.stages[stage_id].runtime_peak_bytes) for profile in runtime_profiles)
                if mixed_overlap
                else 0
            )
            worst_persistent = 0
            worst_compute_peak = 0

            for start in range(window_starts):
                window_slots = range(start, start + resident_count)
                if timeline is None:
                    transient_slot = (
                        None
                        if mixed_overlap
                        else max(
                            window_slots,
                            key=lambda slot: (
                                int(runtime_profiles[slot].stages[stage_id].runtime_peak_bytes)
                                - int(runtime_profiles[slot].stages[stage_id].kv_cache_bytes)
                                - int(runtime_profiles[slot].stages[stage_id].indexer_cache_bytes)
                            ),
                        )
                    )
                    scenarios = ((window_slots, transient_slot),)
                else:
                    compute_starts, completion = timeline
                    scenarios = tuple(
                        (
                            [slot for slot in window_slots if completion[slot] >= compute_starts[stage_id][microbatch]],
                            microbatch,
                        )
                        for microbatch in window_slots
                    )

                for resident_slots, transient_slot in scenarios:
                    if not resident_slots:
                        continue
                    resident_kv, resident_indexer = _sum_stage_cache(
                        resident_profiles,
                        resident_slots,
                        stage_id,
                    )
                    persistent_w = weight + resident_kv + resident_indexer
                    if mixed_overlap:
                        compute_peak_w = mixed_runtime_peak + resident_kv + resident_indexer
                    else:
                        transient_stage = runtime_profiles[transient_slot].stages[stage_id]
                        compute_peak_w = (
                            int(transient_stage.runtime_peak_bytes)
                            + resident_kv
                            - int(transient_stage.kv_cache_bytes)
                            + resident_indexer
                            - int(transient_stage.indexer_cache_bytes)
                        )
                    worst_persistent = max(worst_persistent, persistent_w)
                    worst_compute_peak = max(worst_compute_peak, compute_peak_w)

            persistent[stage_id] = worst_persistent
            compute_peak[stage_id] = max(0, worst_compute_peak)
            communication_peak[stage_id] = worst_persistent + communication_buffer[stage_id]
            stage_peak[stage_id] = max(
                compute_peak[stage_id],
                communication_peak[stage_id],
            )

        phase_persistent.append(persistent)
        phase_compute_peak.append(compute_peak)
        phase_comm_buffer.append(communication_buffer)
        phase_comm_peak.append(communication_peak)
        phase_stage_peak.append(stage_peak)
        phase_incoming.append(incoming)
        phase_outgoing.append(outgoing)

    num_stages = len(slot_profiles[0].stages)
    if prefill_slot_profiles is None:
        persistent = phase_persistent[0]
        compute_peak = phase_compute_peak[0]
        communication_buffer = phase_comm_buffer[0]
        communication_peak = phase_comm_peak[0]
        stage_peak = phase_stage_peak[0]
        incoming = phase_incoming[0]
        outgoing = phase_outgoing[0]
    else:
        persistent = [max(phase[stage_id] for phase in phase_persistent) for stage_id in range(num_stages)]
        # The mixed compute footprint contributes to stage_peak, not this
        # single-phase diagnostic field.
        compute_peak = [
            max(phase_compute_peak[0][stage_id], phase_compute_peak[1][stage_id]) for stage_id in range(num_stages)
        ]
        communication_buffer = phase_comm_buffer[-1]
        communication_peak = [max(phase[stage_id] for phase in phase_comm_peak) for stage_id in range(num_stages)]
        stage_peak = [max(phase[stage_id] for phase in phase_stage_peak) for stage_id in range(num_stages)]
        incoming = phase_incoming[-1]
        outgoing = phase_outgoing[-1]

    budget = device_memory_bytes - reserved_memory_bytes
    remaining = [budget - peak for peak in stage_peak]
    bottleneck_stage_id = 0
    min_remaining = remaining[0] if num_stages > 0 else 0
    for stage_id in range(1, num_stages):
        if remaining[stage_id] < min_remaining:
            min_remaining = remaining[stage_id]
            bottleneck_stage_id = stage_id

    return PipelineStageMemoryEstimate(
        persistent_bytes_s=tuple(persistent),
        compute_peak_bytes_s=tuple(compute_peak),
        communication_buffer_bytes_s=tuple(communication_buffer),
        communication_peak_bytes_s=tuple(communication_peak),
        stage_peak_bytes_s=tuple(stage_peak),
        incoming_payload_bytes_s=tuple(incoming),
        outgoing_payload_bytes_s=tuple(outgoing),
        remaining_bytes_s=tuple(remaining),
        bottleneck_stage_id=bottleneck_stage_id,
        exceeds_budget=any(value < 0 for value in remaining),
    )


def estimate_pipeline_stage_memory(
    profile: PipelineProfile,
    *,
    num_microbatches: int,
    device_memory_bytes: int,
    reserved_memory_bytes: int = 0,
    resident_microbatches: int | None = None,
    perf_model_name: str | None = None,
    resident_policy: str | None = "full",
) -> PipelineStageMemoryEstimate:
    """Estimate stage memory for one uniform profile."""
    return _evaluate_stage_memory(
        (profile,),
        num_microbatches=num_microbatches,
        device_memory_bytes=device_memory_bytes,
        reserved_memory_bytes=reserved_memory_bytes,
        resident_microbatches=resident_microbatches,
        perf_model_name=perf_model_name,
        resident_policy=resident_policy,
    )


def estimate_pipeline_stage_memory_across_profiles(
    profiles: tuple[PipelineProfile, ...] | list[PipelineProfile],
    *,
    num_microbatches: int,
    device_memory_bytes: int,
    reserved_memory_bytes: int = 0,
    resident_microbatches: int | None = None,
    perf_model_name: str | None = None,
    resident_policy: str | None = "full",
) -> PipelineStageMemoryEstimate:
    """Estimate stage memory for uniform or slot-expanded profiles."""
    return _evaluate_stage_memory(
        profiles,
        num_microbatches=num_microbatches,
        device_memory_bytes=device_memory_bytes,
        reserved_memory_bytes=reserved_memory_bytes,
        resident_microbatches=resident_microbatches,
        perf_model_name=perf_model_name,
        resident_policy=resident_policy,
    )


def estimate_mixed_pd_stage_memory(
    prefill_profile: PipelineProfile,
    decode_profile: PipelineProfile,
    *,
    num_microbatches: int,
    resident_microbatches: int | None = None,
    device_memory_bytes: int,
    reserved_memory_bytes: int = 0,
    perf_model_name: str | None = None,
    resident_policy: str | None = "full",
) -> PipelineStageMemoryEstimate:
    """Estimate mixed-PD stage memory for uniform profiles."""
    return _evaluate_stage_memory(
        (decode_profile,),
        prefill_profiles=(prefill_profile,),
        num_microbatches=num_microbatches,
        resident_microbatches=resident_microbatches,
        device_memory_bytes=device_memory_bytes,
        reserved_memory_bytes=reserved_memory_bytes,
        perf_model_name=perf_model_name,
        resident_policy=resident_policy,
    )


def estimate_mixed_pd_stage_memory_across_profiles(
    prefill_profiles: tuple[PipelineProfile, ...] | list[PipelineProfile],
    decode_profiles: tuple[PipelineProfile, ...] | list[PipelineProfile],
    *,
    num_microbatches: int,
    resident_microbatches: int | None = None,
    device_memory_bytes: int,
    reserved_memory_bytes: int = 0,
    perf_model_name: str | None = None,
    resident_policy: str | None = "full",
) -> PipelineStageMemoryEstimate:
    """Estimate mixed-PD stage memory for uniform or expanded profiles."""
    return _evaluate_stage_memory(
        decode_profiles,
        prefill_profiles=prefill_profiles,
        num_microbatches=num_microbatches,
        resident_microbatches=resident_microbatches,
        device_memory_bytes=device_memory_bytes,
        reserved_memory_bytes=reserved_memory_bytes,
        perf_model_name=perf_model_name,
        resident_policy=resident_policy,
    )
