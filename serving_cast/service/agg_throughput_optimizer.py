# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections import deque
from dataclasses import dataclass

import pandas as pd

from tensor_cast.core.model_runner import ModelRunner
from .base_throughput_optimizer import BaseThroughputOptimizer
from .latency_table import ForwardLatencyTable, ForwardShapeKey
from .optimizer_summary import OptimizerSummary
from .scheduler import DecodeFirstWithSlack, Scheduler, SchedulerState
from .utils import AGG_COLUMNS, format_parallel_label, OptimizerData


logger = logging.getLogger(__name__)


@dataclass
class _PrefillGroup:
    count: int
    chunk_index: int


@dataclass
class _DecodeGroup:
    count: int
    remaining_decode_tokens: int
    first_token_time: float


@dataclass(frozen=True)
class _ScheduleStep:
    prefill_key: ForwardShapeKey | None
    decode_key: ForwardShapeKey | None
    p_step: int
    d_step: int


@dataclass
class _ChunkedAggMetrics:
    ttft: float
    tpot: float
    output_throughput: float
    memory_left_gb: float
    prefill_latency: float
    prefill_last_latency: float
    prefill_memory_left_gb: float
    decode_latency: float
    prefill_breakdowns: str
    decode_breakdowns: str


class AggThroughputOptimizer(BaseThroughputOptimizer):
    name = "aggregation"

    def initialize(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.num_mtp_tokens = (
            self.model_runner.model.model_config.mtp_config.num_mtp_layers
            if self.model_runner.model.model_config.mtp_config is not None
            else 0
        )
        self.dp = self.model_runner.model.model_config.parallel_config.data_parallel_size
        self.tp = self.model_runner.model.model_config.parallel_config.tensor_parallel_size
        self.pp = self.model_runner.model.model_config.parallel_config.pipeline_parallel_size
        self.ep = self.model_runner.model.model_config.parallel_config.expert_parallel_size
        self.moe_tp = self.model_runner.model.model_config.parallel_config.moe_tensor_parallel_size
        self.moe_dp = self.model_runner.model.model_config.parallel_config.moe_data_parallel_size
        self.is_moe_model = self.model_runner.model.model_config.moe_config is not None
        self._forward_record_cache.clear()
        self.scheduler = DecodeFirstWithSlack()

    def get_inference_info(self, optimizer_data: OptimizerData) -> OptimizerSummary:
        max_batched_tokens = optimizer_data.max_batched_tokens
        batch_size = optimizer_data.batch_size
        input_length = optimizer_data.input_length
        effective_input_length = optimizer_data.get_effective_input_length()
        output_length = optimizer_data.output_length
        concurrency = batch_size * self.dp * self.pp
        chunk_plan = optimizer_data.get_prefill_chunk_plan()

        # Single-chunk prompts keep the historical formula so existing short-prompt results stay stable.
        if len(chunk_plan) == 1:
            metrics = self._get_full_prefill_metrics(optimizer_data, concurrency)
        else:
            metrics = self._simulate_chunked_prefill(optimizer_data, chunk_plan, concurrency, self.scheduler)

        memory_left = metrics.memory_left_gb
        token_s_device = metrics.output_throughput / self.dp / self.pp / self.tp
        parallel = format_parallel_label(
            self.model_runner.model.model_config.parallel_config,
            self.is_moe_model,
            optimizer_data.num_mtp_tokens,
        )

        logger.info(
            "Prefill Wave Latency: %.4f ms, "
            "Prefill Last Wave Latency: %.4f ms, "
            "Decode Latency: %.4f ms, "
            "TTFT: %.4f ms, TPOT: %.4f ms, "
            "Output Throughput: %.2f token/s, "
            "Concurrency: %d, "
            "parallel: %s, "
            "Memory Left: %.2f GB, "
            "Prefill Wave Memory Left: %.2f GB",
            metrics.prefill_latency,
            metrics.prefill_last_latency,
            metrics.decode_latency,
            metrics.ttft,
            metrics.tpot,
            metrics.output_throughput,
            concurrency,
            parallel,
            memory_left,
            metrics.prefill_memory_left_gb,
        )
        summary = OptimizerSummary(optimizer_data)
        result_df = pd.DataFrame(
            columns=AGG_COLUMNS,
            data=[
                [
                    self.model_runner.user_input.device,
                    optimizer_data.num_devices,
                    self.model_runner.user_input.model_id,
                    self.model_runner.user_input.quantize_linear_action,
                    self.model_runner.user_input.quantize_attention_action,
                    input_length,
                    output_length,
                    effective_input_length,
                    max_batched_tokens,
                    len(chunk_plan),
                    concurrency,
                    metrics.ttft,
                    metrics.tpot,
                    metrics.output_throughput,
                    token_s_device,
                    parallel,
                    batch_size,
                    metrics.prefill_breakdowns,
                    metrics.decode_breakdowns,
                ]
            ],
        ).round(3)
        summary.set_summary_df(result_df)
        summary.set_early_stop_flag(memory_left, metrics.tpot, metrics.ttft)

        self._maybe_set_search_info(optimizer_data, memory_left, batch_size, metrics.ttft, metrics.tpot, summary)

        return summary

    def _get_full_prefill_metrics(self, optimizer_data: OptimizerData, concurrency: int) -> _ChunkedAggMetrics:
        """Compute aggregation metrics for prompts that fit in one prefill chunk.

        This keeps the original wave-based TTFT/TPOT formula for short prompts while also
        checking memory across both the full prefill wave and any remainder wave.
        """
        max_batched_tokens = optimizer_data.max_batched_tokens
        effective_input_length = optimizer_data.get_effective_input_length()
        output_length = optimizer_data.output_length
        batch_size = optimizer_data.batch_size

        # Preserve the existing short-prompt formula when one request fits in a single prefill chunk.
        prefill_batch_size = max_batched_tokens // effective_input_length
        calc_nums_for_ttft = concurrency // prefill_batch_size
        left_calc_num = concurrency % prefill_batch_size

        prefill_latency = 0
        prefill_last_latency = 0
        prefill_min_memory_left_gb = float("inf")
        prefill_breakdowns = ""
        if calc_nums_for_ttft > 0:
            prefill_latency, prefill_memory_left_gb, prefill_breakdowns = self._get_or_compute_latency(
                prefill_batch_size, optimizer_data, is_decode=False
            )
            prefill_last_latency = prefill_latency
            prefill_min_memory_left_gb = prefill_memory_left_gb
            if prefill_memory_left_gb < 0:
                return _ChunkedAggMetrics(
                    ttft=float("inf"),
                    tpot=float("inf"),
                    output_throughput=0,
                    memory_left_gb=prefill_memory_left_gb,
                    prefill_latency=prefill_latency,
                    prefill_last_latency=prefill_last_latency,
                    prefill_memory_left_gb=prefill_memory_left_gb,
                    decode_latency=0,
                    prefill_breakdowns=prefill_breakdowns,
                    decode_breakdowns="",
                )

        left_latency = 0
        if left_calc_num != 0:
            left_latency, left_memory_left_gb, left_breakdowns = self._get_or_compute_latency(
                left_calc_num,
                optimizer_data,
                is_decode=False,
            )
            prefill_last_latency = left_latency
            if calc_nums_for_ttft == 0:
                prefill_latency = left_latency
                prefill_breakdowns = left_breakdowns
            if calc_nums_for_ttft > 0:
                prefill_min_memory_left_gb = min(prefill_memory_left_gb, left_memory_left_gb)
            else:
                prefill_min_memory_left_gb = left_memory_left_gb

        left_batch_time = (calc_nums_for_ttft * prefill_latency + left_latency) * left_calc_num
        sum_for_ttft = (prefill_batch_size * prefill_latency) * (
            1 + calc_nums_for_ttft
        ) * calc_nums_for_ttft / 2 + left_batch_time
        ttft = sum_for_ttft / concurrency

        if prefill_min_memory_left_gb < 0:
            return _ChunkedAggMetrics(
                ttft=float("inf"),
                tpot=float("inf"),
                output_throughput=0,
                memory_left_gb=prefill_min_memory_left_gb,
                prefill_latency=prefill_latency,
                prefill_last_latency=prefill_last_latency,
                prefill_memory_left_gb=prefill_min_memory_left_gb,
                decode_latency=0,
                prefill_breakdowns=prefill_breakdowns,
                decode_breakdowns="",
            )

        decode_latency, decode_memory_left_gb, decode_breakdowns = self._get_or_compute_latency(
            batch_size, optimizer_data, is_decode=True
        )
        tpot = (ttft + decode_latency * output_length) / output_length
        output_throughput = 1000 * (output_length * concurrency) / (ttft + tpot * output_length)

        return _ChunkedAggMetrics(
            ttft=ttft,
            tpot=tpot,
            output_throughput=output_throughput,
            memory_left_gb=min(prefill_min_memory_left_gb, decode_memory_left_gb),
            prefill_latency=prefill_latency,
            prefill_last_latency=prefill_last_latency,
            prefill_memory_left_gb=prefill_min_memory_left_gb,
            decode_latency=decode_latency,
            prefill_breakdowns=prefill_breakdowns,
            decode_breakdowns=decode_breakdowns,
        )

    def _simulate_chunked_prefill(
        self,
        optimizer_data: OptimizerData,
        chunk_plan: list,
        concurrency: int,
        scheduler: Scheduler,
    ) -> _ChunkedAggMetrics:
        """Simulate aggregation scheduling when prefill is split into multiple chunks.

        Requests move from pending prefill to ready decode after their final prefill chunk.
        Each simulated step lets the scheduler choose prefill and decode concurrency under
        the mixed-step token budget, then accumulates TTFT, TPOT, throughput, and memory.
        The scheduler is injected by the caller so upper layers can select a scheduling
        policy without changing the simulation loop.
        """
        schedule = self._build_chunked_prefill_schedule(optimizer_data, chunk_plan, concurrency, scheduler)
        latency_table = ForwardLatencyTable(
            self,
            optimizer_data,
        )
        latency_table.prefetch(self._collect_schedule_keys(schedule))
        return self._replay_chunked_prefill_schedule(
            optimizer_data,
            chunk_plan,
            concurrency,
            scheduler,
            schedule,
            latency_table,
        )

    def _build_chunked_prefill_schedule(
        self,
        optimizer_data: OptimizerData,
        chunk_plan: list,
        concurrency: int,
        scheduler: Scheduler,
    ) -> list[_ScheduleStep]:
        """Build a latency-free schedule plan for chunked prefill."""
        # pending_prefill keeps requests that have not emitted the first visible token yet.
        pending_prefill = deque([_PrefillGroup(count=concurrency, chunk_index=0)])
        # ready_decode keeps requests whose final prefill chunk has completed and can decode immediately.
        ready_decode = deque()

        # The last prefill chunk produces the first token, so decode only needs output_length - 1 steps.
        remaining_decode_tokens = max(optimizer_data.output_length - 1, 0)
        finished = 0
        schedule = []

        while finished < concurrency:
            chunk = chunk_plan[pending_prefill[0].chunk_index] if pending_prefill else None
            pending_count = self._count_front_prefill_group(pending_prefill)
            ready_decode_count = sum(group.count for group in ready_decode)
            state = SchedulerState(
                ready_decode=ready_decode_count,
                pending_prefill=pending_count,
                chunk_query_len=chunk.query_len if chunk is not None else optimizer_data.max_batched_tokens,
                max_batched_tokens=optimizer_data.max_batched_tokens,
            )
            decision = scheduler.decide(state)
            p_step = decision.p_step
            d_step = decision.d_step

            if p_step == 0 and d_step == 0:
                raise RuntimeError("Chunked prefill simulation made no progress.")

            prefill_key = None
            if p_step > 0:
                prefill_key = self._make_forward_shape_key(
                    p_step,
                    optimizer_data,
                    is_decode=False,
                    query_len=chunk.query_len,
                    seq_len=chunk.seq_len,
                )

            decode_key = None
            if d_step > 0:
                decode_key = self._make_forward_shape_key(d_step, optimizer_data, is_decode=True)

            schedule.append(
                _ScheduleStep(
                    prefill_key=prefill_key,
                    decode_key=decode_key,
                    p_step=p_step,
                    d_step=d_step,
                )
            )

            if p_step > 0:
                _, finished, _ = self._advance_prefill_groups(
                    pending_prefill,
                    ready_decode,
                    chunk_plan,
                    p_step,
                    current_time=0.0,
                    remaining_decode_tokens=remaining_decode_tokens,
                    first_token_time_sum=0.0,
                    finished=finished,
                    max_finish_time=0.0,
                )

            if d_step > 0:
                _, finished, _ = self._advance_decode_groups(
                    ready_decode,
                    d_step,
                    current_time=0.0,
                    initial_decode_tokens=remaining_decode_tokens,
                    tpot_sum=0.0,
                    finished=finished,
                    max_finish_time=0.0,
                )

        return schedule

    @staticmethod
    def _collect_schedule_keys(schedule: list[_ScheduleStep]) -> list[ForwardShapeKey]:
        keys = []
        for step in schedule:
            if step.prefill_key is not None:
                keys.append(step.prefill_key)
            if step.decode_key is not None:
                keys.append(step.decode_key)
        return keys

    def _replay_chunked_prefill_schedule(
        self,
        optimizer_data: OptimizerData,
        chunk_plan: list,
        concurrency: int,
        scheduler: Scheduler,
        schedule: list[_ScheduleStep],
        latency_table: ForwardLatencyTable,
    ) -> _ChunkedAggMetrics:
        # pending_prefill keeps requests that have not emitted the first visible token yet.
        pending_prefill = deque([_PrefillGroup(count=concurrency, chunk_index=0)])
        # ready_decode keeps requests whose final prefill chunk has completed and can decode immediately.
        ready_decode = deque()

        # The last prefill chunk produces the first token, so decode only needs output_length - 1 steps.
        remaining_decode_tokens = max(optimizer_data.output_length - 1, 0)
        finished = 0
        current_time = 0.0
        max_finish_time = 0.0
        first_token_time_sum = 0.0
        tpot_sum = 0.0
        memory_left_gb = float("inf")
        prefill_memory_left_gb = float("inf")
        prefill_breakdowns = ""
        decode_breakdowns = ""
        last_prefill_latency = 0.0
        last_decode_latency = 0.0

        for step in schedule:
            prefill_step_latency = 0.0
            if step.prefill_key is not None:
                prefill_record = latency_table.get(step.prefill_key)
                prefill_step_latency = self._get_forward_latency_ms(step.prefill_key, prefill_record, optimizer_data)
                memory_left_gb = min(memory_left_gb, prefill_record.memory_left_gb)
                prefill_memory_left_gb = min(prefill_memory_left_gb, prefill_record.memory_left_gb)
                prefill_breakdowns = prefill_breakdowns or prefill_record.breakdowns
                last_prefill_latency = prefill_step_latency
                if prefill_record.memory_left_gb < 0:
                    return _ChunkedAggMetrics(
                        ttft=current_time + prefill_step_latency,
                        tpot=float("inf"),
                        output_throughput=0,
                        memory_left_gb=memory_left_gb,
                        prefill_latency=last_prefill_latency,
                        prefill_last_latency=last_prefill_latency,
                        prefill_memory_left_gb=prefill_memory_left_gb,
                        decode_latency=last_decode_latency,
                        prefill_breakdowns=prefill_breakdowns,
                        decode_breakdowns=decode_breakdowns,
                    )

            decode_step_latency = 0.0
            if step.decode_key is not None:
                decode_record = latency_table.get(step.decode_key)
                decode_step_latency = self._get_forward_latency_ms(step.decode_key, decode_record, optimizer_data)
                memory_left_gb = min(memory_left_gb, decode_record.memory_left_gb)
                decode_breakdowns = decode_breakdowns or decode_record.breakdowns
                last_decode_latency = decode_step_latency
                if decode_record.memory_left_gb < 0:
                    return _ChunkedAggMetrics(
                        ttft=current_time + decode_step_latency,
                        tpot=float("inf"),
                        output_throughput=0,
                        memory_left_gb=memory_left_gb,
                        prefill_latency=last_prefill_latency,
                        prefill_last_latency=last_prefill_latency,
                        prefill_memory_left_gb=prefill_memory_left_gb,
                        decode_latency=last_decode_latency,
                        prefill_breakdowns=prefill_breakdowns,
                        decode_breakdowns=decode_breakdowns,
                    )

            # The default mixed scheduler models prefill and decode as overlapping in the same step.
            step_latency = scheduler.step_latency(prefill_step_latency, decode_step_latency)
            current_time += step_latency

            if step.p_step > 0:
                first_token_time_sum, finished, max_finish_time = self._advance_prefill_groups(
                    pending_prefill,
                    ready_decode,
                    chunk_plan,
                    step.p_step,
                    current_time,
                    remaining_decode_tokens,
                    first_token_time_sum,
                    finished,
                    max_finish_time,
                )

            if step.d_step > 0:
                tpot_sum, finished, max_finish_time = self._advance_decode_groups(
                    ready_decode,
                    step.d_step,
                    current_time,
                    remaining_decode_tokens,
                    tpot_sum,
                    finished,
                    max_finish_time,
                )

        if finished < concurrency:
            raise RuntimeError("Chunked prefill schedule replay ended before all requests finished.")

        ttft = first_token_time_sum / concurrency
        tpot = 0 if remaining_decode_tokens == 0 else tpot_sum / concurrency
        output_throughput = (
            1000 * optimizer_data.output_length * concurrency / max_finish_time if max_finish_time > 0 else 0
        )

        return _ChunkedAggMetrics(
            ttft=ttft,
            tpot=tpot,
            output_throughput=output_throughput,
            memory_left_gb=memory_left_gb,
            prefill_latency=last_prefill_latency,
            prefill_last_latency=last_prefill_latency,
            prefill_memory_left_gb=prefill_memory_left_gb,
            decode_latency=last_decode_latency,
            prefill_breakdowns=prefill_breakdowns,
            decode_breakdowns=decode_breakdowns,
        )

    @staticmethod
    def _count_front_prefill_group(pending_prefill: deque[_PrefillGroup]) -> int:
        """Count pending prefill requests that share the same next chunk shape."""
        if not pending_prefill:
            return 0
        chunk_index = pending_prefill[0].chunk_index
        total = 0
        # Stop at the first different chunk index because query_len/seq_len would differ.
        for group in pending_prefill:
            if group.chunk_index != chunk_index:
                break
            total += group.count
        return total

    @staticmethod
    def _advance_prefill_groups(
        pending_prefill: deque[_PrefillGroup],
        ready_decode: deque[_DecodeGroup],
        chunk_plan: list,
        p_step: int,
        current_time: float,
        remaining_decode_tokens: int,
        first_token_time_sum: float,
        finished: int,
        max_finish_time: float,
    ) -> tuple[float, int, float]:
        """Advance selected prefill requests by one chunk and update request queues.

        Non-final chunks are requeued for their next chunk. Final chunks emit the first
        visible token, which contributes to TTFT and either enters decode or finishes the
        request when output length is one.

        Args:
            pending_prefill: Queue of requests waiting for their next prefill chunk.
            ready_decode: Queue of requests whose first token is available and can decode.
            chunk_plan: Ordered chunk shapes for one request's prefill phase.
            p_step: Number of prefill requests selected by the scheduler.
            current_time: Simulated timestamp after the current scheduling step.
            remaining_decode_tokens: Decode tokens left after the first visible token.
            first_token_time_sum: Accumulated first-token timestamps across requests.
            finished: Number of requests that have completed all output tokens.
            max_finish_time: Latest finish timestamp among completed requests.

        Returns:
            Updated first-token sum, finished request count, and latest finish time.
        """
        selected = p_step
        while selected > 0:
            group = pending_prefill[0]
            take = min(selected, group.count)
            if take == group.count:
                pending_prefill.popleft()
            else:
                group.count -= take

            next_chunk_index = group.chunk_index + 1
            if next_chunk_index < len(chunk_plan):
                # Non-final chunks remain in prefill and will be scheduled with their next chunk shape.
                pending_prefill.append(_PrefillGroup(count=take, chunk_index=next_chunk_index))
            else:
                # Final prefill chunk emits the first visible token and starts TTFT accounting.
                first_token_time_sum += take * current_time
                if remaining_decode_tokens > 0:
                    ready_decode.append(
                        _DecodeGroup(
                            count=take,
                            remaining_decode_tokens=remaining_decode_tokens,
                            first_token_time=current_time,
                        )
                    )
                else:
                    # output_length == 1: the request finishes when the first token is produced.
                    finished += take
                    max_finish_time = max(max_finish_time, current_time)
            selected -= take

        return first_token_time_sum, finished, max_finish_time

    @staticmethod
    def _advance_decode_groups(
        ready_decode: deque[_DecodeGroup],
        d_step: int,
        current_time: float,
        initial_decode_tokens: int,
        tpot_sum: float,
        finished: int,
        max_finish_time: float,
    ) -> tuple[float, int, float]:
        """Advance selected decode requests by one token and update TPOT accounting.

        Args:
            ready_decode: Queue of requests that can produce decode tokens.
            d_step: Number of decode requests selected by the scheduler.
            current_time: Simulated timestamp after the current scheduling step.
            initial_decode_tokens: Decode token count used to average per-request TPOT.
            tpot_sum: Accumulated per-request TPOT values weighted by request count.
            finished: Number of requests that have completed all output tokens.
            max_finish_time: Latest finish timestamp among completed requests.

        Returns:
            Updated TPOT sum, finished request count, and latest finish time.
        """
        selected = d_step
        while selected > 0:
            group = ready_decode[0]
            take = min(selected, group.count)
            if take == group.count:
                ready_decode.popleft()
            else:
                group.count -= take

            remaining_decode_tokens = group.remaining_decode_tokens - 1
            if remaining_decode_tokens == 0:
                # The per-request TPOT is averaged from first-token time to finish time.
                finished += take
                max_finish_time = max(max_finish_time, current_time)
                tpot_sum += take * ((current_time - group.first_token_time) / initial_decode_tokens)
            else:
                # Unfinished decode requests re-enter the queue for the next decode token.
                ready_decode.append(
                    _DecodeGroup(
                        count=take,
                        remaining_decode_tokens=remaining_decode_tokens,
                        first_token_time=group.first_token_time,
                    )
                )
            selected -= take

        return tpot_sum, finished, max_finish_time
