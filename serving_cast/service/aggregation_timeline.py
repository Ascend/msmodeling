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

"""Finite-batch aggregation: all requests arrive at zero and drain to completion.

Time is in milliseconds. Speculative progress is an expectation, whereas every
verification iteration pays its full physical forward cost (including the tail).
Decode context uses the optimizer's representative mid-output context, not a
claim to replay an actual serving engine or an arrival-rate workload.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import islice
from math import ceil
from typing import Callable, Iterable

from tensor_cast.pipeline_parallel import PipelineProfile

from .latency_table import ForwardLatencyRecord
from .pipeline_schedule import PipelineScheduleEstimate, _WaveSchedule, _build_schedule_estimate, _schedule_wave
from .scheduler import Scheduler, SchedulerState
from .utils import OptimizerData


@dataclass(frozen=True, order=True)
class RequestShape:
    is_decode: bool
    query_len: int
    seq_len: int


@dataclass
class TimelineRequest:
    prompt_length: int
    chunks: list[RequestShape]
    chunk_index: int = 0
    remaining_output: float = 0.0
    first_token_ms: float | None = None
    finish_ms: float | None = None


@dataclass
class TimelineResult:
    requests: list[TimelineRequest]
    elapsed_ms: float = 0.0
    # (shape tuple, raw physical latency); repeated decode shapes are cached upstream.
    steps: list[tuple[tuple[RequestShape, ...], float]] = field(default_factory=list)
    oom: bool = False
    pipeline_schedule_estimate: PipelineScheduleEstimate | None = None

    @property
    def ttft(self) -> float:
        if self.oom or not self.requests or any(r.first_token_ms is None for r in self.requests):
            return float("inf")
        return sum(r.first_token_ms for r in self.requests) / len(self.requests)

    @property
    def e2e(self) -> float:
        if self.oom or not self.requests or any(r.finish_ms is None for r in self.requests):
            return float("inf")
        return sum(r.finish_ms for r in self.requests) / len(self.requests)


def build_requests(data: OptimizerData, concurrency: int) -> list[TimelineRequest]:
    """Build fixed chunks once; scheduler admission never resplits them."""
    if data.length_distribution is None:
        rows = [
            {
                "samples": concurrency,
                "num_input_tokens": data.input_length,
                "query_len": data.get_effective_input_length(),
            }
        ]
    else:
        rows = data.build_concurrency_samples(concurrency)
    requests = []
    for row in rows:
        prompt = row["num_input_tokens"]
        query = row["query_len"]
        if query <= 0:
            raise ValueError("Aggregation requests require a positive prefill query length")
        cached = prompt - query
        chunks = []
        consumed = 0
        while consumed < query:
            q = min(data.max_batched_tokens, query - consumed)
            consumed += q
            chunks.append(RequestShape(False, q, cached + consumed))
        requests.extend(TimelineRequest(prompt, chunks.copy()) for _ in range(row["samples"]))
    return requests


def simulate(
    requests: list[TimelineRequest],
    data: OptimizerData,
    scheduler: Scheduler,
    decode_query_len: int,
    accepted_tokens: float,
    evaluate: Callable[[tuple[RequestShape, ...], tuple[RequestShape, ...]], ForwardLatencyRecord],
) -> TimelineResult:
    """Evaluate each selected batch once, then update request timestamps.

    evaluate(shapes, resident_shapes) returns a latency record. resident_shapes
    includes inactive live KV; completed requests release their KV immediately.
    """
    if not requests or data.output_length < 1 or not 0 < accepted_tokens <= decode_query_len:
        raise ValueError("Positive concurrency/output and valid speculative progress are required")
    if any(not request.chunks for request in requests):
        raise ValueError("Each request must have at least one prefill chunk")
    result = TimelineResult(requests)
    pending: deque[TimelineRequest] = deque(requests)
    ready: deque[TimelineRequest] = deque()
    partial_pending: dict[int, TimelineRequest] = {}
    decode_shapes = {
        id(r): RequestShape(True, decode_query_len, r.prompt_length + data.output_length // 2 + decode_query_len)
        for r in requests
    }
    resident_decode_shapes = {
        id(r): RequestShape(True, decode_query_len, r.prompt_length + data.output_length + decode_query_len)
        for r in requests
    }
    while pending or ready:
        front = pending[0].chunks[pending[0].chunk_index] if pending else None
        # Only a contiguous run of the same chunk shape is admitted together.
        count = 0
        for request in pending:
            if request.chunks[request.chunk_index] != front:
                break
            count += 1
        decision = scheduler.decide(
            SchedulerState(
                len(ready),
                count,
                front.query_len if front else 0,
                data.max_batched_tokens,
                decode_query_len,
            )
        )
        prefill = list(islice(pending, decision.p_step))
        decode = list(islice(ready, decision.d_step))
        if not prefill and not decode:
            raise RuntimeError("Aggregation schedule made no progress within the token budget")
        shapes = tuple(sorted([decode_shapes[id(r)] for r in decode] + [r.chunks[r.chunk_index] for r in prefill]))
        # Reserve a conservative end-of-output KV envelope for all ready requests.
        # The latency still uses the representative context, independently of KV residency.
        resident = [resident_decode_shapes[id(r)] for r in ready]
        selected_ids = {id(r) for r in prefill}
        resident.extend(r.chunks[r.chunk_index] for r in prefill)
        resident.extend(r.chunks[r.chunk_index - 1] for key, r in partial_pending.items() if key not in selected_ids)
        record = evaluate(shapes, tuple(resident))
        result.steps.append((shapes, record.latency_ms))
        result.elapsed_ms += record.latency_ms
        if record.memory_left_gb < 0:
            result.oom = True
            return result
        # Snapshot selections precede completion: a newly prefilling request cannot
        # also decode in the same physical forward.
        for r in prefill:
            pending.popleft()
            partial_pending.pop(id(r), None)
        for _ in decode:
            ready.popleft()
        for r in decode:
            r.remaining_output = max(0.0, r.remaining_output - accepted_tokens)
            if r.remaining_output < 1e-9:
                r.finish_ms = result.elapsed_ms
            else:
                ready.append(r)
        for r in prefill:
            r.chunk_index += 1
            if r.chunk_index < len(r.chunks):
                pending.append(r)
                partial_pending[id(r)] = r
            else:
                r.first_token_ms = result.elapsed_ms
                r.remaining_output = data.output_length - 1
                if r.remaining_output:
                    ready.append(r)
                else:
                    r.finish_ms = result.elapsed_ms
    return result


def simulate_pipeline(
    requests: list[TimelineRequest],
    data: OptimizerData,
    scheduler: Scheduler,
    decode_query_len: int,
    accepted_tokens: float,
    microbatch_size: int,
    evaluate: Callable[[tuple[RequestShape, ...], tuple[RequestShape, ...]], tuple[PipelineProfile, float]],
    perf_model: str,
) -> TimelineResult:
    """Run a finite batch through PP stages, admitting work when stage 0 is free.

    Each admitted microbatch contains the selected P/D requests. Completion on
    the final stage makes a request eligible for its next chunk or decode turn.
    ``evaluate`` returns a stage profile and its resident-memory margin.
    """
    if not requests or data.output_length < 1 or not 0 < accepted_tokens <= decode_query_len:
        raise ValueError("Positive concurrency/output and valid speculative progress are required")
    if microbatch_size < 1 or any(not request.chunks for request in requests):
        raise ValueError("A positive microbatch size and prefill chunks are required")
    result = TimelineResult(requests)
    pending: deque[TimelineRequest] = deque(requests)
    ready: deque[TimelineRequest] = deque()
    inflight: set[int] = set()
    events: list[tuple[float, int, tuple[TimelineRequest, ...], tuple[TimelineRequest, ...]]] = []
    event_number = 0
    stage_free: list[float] | None = None
    stage_busy = []
    compute_intervals = []
    outgoing_intervals = []
    incoming_intervals = []
    incoming_payload = []
    outgoing_payload = []
    completions = []
    now = 0.0
    decode_shapes = {
        id(r): RequestShape(True, decode_query_len, r.prompt_length + data.output_length // 2 + decode_query_len)
        for r in requests
    }
    resident_decode_shapes = {
        id(r): RequestShape(True, decode_query_len, r.prompt_length + data.output_length + decode_query_len)
        for r in requests
    }

    def complete_until(time_ms: float) -> None:
        while events and events[0][0] <= time_ms:
            completion, _, prefill, decode = heappop(events)
            for request in decode:
                inflight.remove(id(request))
                request.remaining_output = max(0.0, request.remaining_output - accepted_tokens)
                if request.remaining_output < 1e-9:
                    request.finish_ms = completion
                else:
                    ready.append(request)
            for request in prefill:
                inflight.remove(id(request))
                request.chunk_index += 1
                if request.chunk_index < len(request.chunks):
                    pending.append(request)
                else:
                    request.first_token_ms = completion
                    request.remaining_output = data.output_length - 1
                    if request.remaining_output:
                        ready.append(request)
                    else:
                        request.finish_ms = completion

    while pending or ready or events:
        now = max(now, stage_free[0] * 1000 if stage_free is not None else 0.0)
        complete_until(now)
        if not pending and not ready:
            if events:
                now = max(now, events[0][0])
                complete_until(now)
            continue
        front = pending[0].chunks[pending[0].chunk_index] if pending else None
        same_shape = 0
        for request in pending:
            if request.chunks[request.chunk_index] != front:
                break
            same_shape += 1
        decision = scheduler.decide(
            SchedulerState(
                min(len(ready), microbatch_size),
                same_shape,
                front.query_len if front else 0,
                data.max_batched_tokens,
                decode_query_len,
            )
        )
        d_count = decision.d_step
        p_count = min(decision.p_step, microbatch_size - d_count)
        if not p_count and not d_count:
            if events:
                now = max(now, events[0][0])
                complete_until(now)
                continue
            raise RuntimeError("Pipeline aggregation schedule made no progress within the token budget")
        prefill = tuple(islice(pending, p_count))
        decode = tuple(islice(ready, d_count))
        shapes = tuple(sorted([r.chunks[r.chunk_index] for r in prefill] + [decode_shapes[id(r)] for r in decode]))
        resident = []
        selected_ids = {id(r) for r in prefill}
        for request in requests:
            if request.finish_ms is not None:
                continue
            key = id(request)
            if request.first_token_ms is not None:
                resident.append(resident_decode_shapes[key])
            elif key in selected_ids or key in inflight:
                resident.append(request.chunks[request.chunk_index])
            elif request.chunk_index:
                resident.append(request.chunks[request.chunk_index - 1])
        profile, memory_left_gb = evaluate(shapes, tuple(resident))
        if memory_left_gb < 0:
            result.oom = True
            return result
        if stage_free is None:
            stage_free = [0.0] * profile.pp_size
            stage_busy = [0.0] * profile.pp_size
            compute_intervals = [[] for _ in range(profile.pp_size)]
            outgoing_intervals = [[] for _ in range(profile.pp_size)]
            incoming_intervals = [[] for _ in range(profile.pp_size)]
            incoming_payload = [0] * profile.pp_size
            outgoing_payload = [0] * profile.pp_size
        stage_free[0] = max(stage_free[0], now / 1000)
        wave = _schedule_wave((profile,), perf_model, profile.pp_size, initial_free=stage_free)
        stage_free = list(wave.final_free)
        for stage_id, busy in enumerate(wave.stage_busy):
            stage_busy[stage_id] += busy
            compute_intervals[stage_id].extend(wave.compute_intervals[stage_id])
            outgoing_intervals[stage_id].extend(wave.outgoing_transfer_intervals[stage_id])
            incoming_intervals[stage_id].extend(wave.incoming_transfer_intervals[stage_id])
            outgoing_payload[stage_id] = max(
                outgoing_payload[stage_id], profile.stages[stage_id].outgoing_payload_bytes
            )
        for transfer in profile.transfers:
            outgoing_payload[transfer.source_stage_id] = max(
                outgoing_payload[transfer.source_stage_id], transfer.payload_bytes
            )
            incoming_payload[transfer.target_stage_id] = max(
                incoming_payload[transfer.target_stage_id], transfer.payload_bytes
            )
        completion = wave.microbatch_completion[0] * 1000
        completions.append(wave.microbatch_completion[0])
        result.elapsed_ms = max(result.elapsed_ms, completion)
        result.steps.append((shapes, completion - now))
        for _ in prefill:
            pending.popleft()
        for _ in decode:
            ready.popleft()
        inflight.update(id(request) for request in prefill + decode)
        heappush(events, (completion, event_number, prefill, decode))
        event_number += 1
    wave = _WaveSchedule(
        microbatch_completion=tuple(completions),
        final_free=tuple(stage_free),
        stage_busy=tuple(stage_busy),
        compute_intervals=tuple(tuple(intervals) for intervals in compute_intervals),
        outgoing_transfer_intervals=tuple(tuple(intervals) for intervals in outgoing_intervals),
        incoming_transfer_intervals=tuple(tuple(intervals) for intervals in incoming_intervals),
    )
    communication_buffer = [max(incoming, outgoing) for incoming, outgoing in zip(incoming_payload, outgoing_payload)]
    result.pipeline_schedule_estimate = _build_schedule_estimate(
        wave=wave,
        pp_size=len(stage_busy),
        makespan=result.elapsed_ms / 1000,
        incoming_payload=incoming_payload,
        outgoing_payload=outgoing_payload,
        communication_buffer=communication_buffer,
    )
    return result


def page_tokens(shapes: Iterable[RequestShape], block_size: int, speculative_tail: int = 1) -> int:
    """Round each request separately, including speculative tail headroom."""
    return sum(ceil((s.seq_len + speculative_tail) / block_size) * block_size for s in shapes)
