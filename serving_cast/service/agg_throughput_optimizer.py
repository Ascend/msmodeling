# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
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

import logging
from collections import Counter
from dataclasses import dataclass, replace

import pandas as pd

from tensor_cast.core.input_generator import RequestInfo, generate_inputs_varlen, kv_cache_excluded_layer_indices
from tensor_cast.core.model_runner import ModelRunner
from tensor_cast.performance_model.utils import bytes_of_tensor
from .base_throughput_optimizer import BaseThroughputOptimizer
from .optimizer_summary import (
    EARLY_STOP_DECODE_OOM,
    EARLY_STOP_PREFILL_OOM,
    OptimizerSummary,
)
from .scheduler import DecodeFirstWithSlack
from .utils import (
    AGG_COLUMNS,
    format_parallel_label,
    MemoryInfo,
    OptimizerData,
    select_tightest_memory_info,
)


logger = logging.getLogger(__name__)


def _resident_kv_cache_size_gb(runner: ModelRunner, requests) -> float:
    """Size live KV from the physical cache layout, including indexer caches."""
    inputs = generate_inputs_varlen(runner.model, requests, runner.user_input.block_size)
    excluded = kv_cache_excluded_layer_indices(runner.model)
    main = sum(bytes_of_tensor(cache) for layer, cache in inputs["kv_cache_by_layers"].items() if layer not in excluded)
    indexer = sum(bytes_of_tensor(cache) for cache in inputs.get("indexer_cache_by_layers", {}).values())
    return (main + indexer) / 1024**3


def _reserved_kv_cache_size_gb(runner: ModelRunner, shapes) -> float:
    """Reserve complete KV pages per request, including waiting requests."""
    sizer = getattr(runner, "get_resident_kv_cache_size_gb", None)
    return sum(
        count
        * (
            sizer([RequestInfo(shape.query_len, shape.seq_len, is_decode=True)])
            if sizer is not None
            else _resident_kv_cache_size_gb(runner, [RequestInfo(shape.query_len, shape.seq_len, is_decode=True)])
        )
        for shape, count in Counter(shapes).items()
    )


def _resident_pp_kv_bytes(runner: ModelRunner, shapes, profile, block_size: int) -> tuple[float, ...]:
    """Size resident KV on each PP stage using that stage's layer partition."""
    batch = [RequestInfo(s.query_len, s.seq_len, is_decode=s.is_decode) for s in shapes]
    inputs = generate_inputs_varlen(runner.model, batch, block_size)
    excluded = kv_cache_excluded_layer_indices(runner.model)
    caches = inputs["kv_cache_by_layers"]
    indexers = inputs.get("indexer_cache_by_layers", {})
    return tuple(
        sum(
            bytes_of_tensor(cache)
            for layer, cache in caches.items()
            if stage.layer_start <= layer < stage.layer_end and layer not in excluded
        )
        + sum(
            bytes_of_tensor(cache) for layer, cache in indexers.items() if stage.layer_start <= layer < stage.layer_end
        )
        for stage in profile.stages
    )


def _reserved_pp_kv_bytes(runner: ModelRunner, shapes, profile, block_size: int) -> tuple[float, ...]:
    """Sum stage-local KV pages for every request in the concurrency batch."""
    totals = [0.0] * len(profile.stages)
    for shape, count in Counter(shapes).items():
        for stage_id, size in enumerate(_resident_pp_kv_bytes(runner, (shape,), profile, block_size)):
            totals[stage_id] += count * size
    return tuple(totals)


def _overlap_pipeline_profiles(prefill, decode):
    """Approximate a mixed PP microbatch with concurrent P/D stage costs."""
    if prefill.pp_size != decode.pp_size or prefill.layer_partition != decode.layer_partition:
        raise ValueError("Prefill and decode pipeline partitions differ")
    stages = []
    for p, d in zip(prefill.stages, decode.stages):
        stages.append(
            replace(
                p,
                compute_time_s_by_model={
                    key: max(p.compute_time_s_by_model.get(key, 0), d.compute_time_s_by_model.get(key, 0))
                    for key in p.compute_time_s_by_model.keys() | d.compute_time_s_by_model.keys()
                },
                outgoing_comm_time_s_by_model={
                    key: max(p.outgoing_comm_time_s_by_model.get(key, 0), d.outgoing_comm_time_s_by_model.get(key, 0))
                    for key in p.outgoing_comm_time_s_by_model.keys() | d.outgoing_comm_time_s_by_model.keys()
                },
                activation_bytes=p.activation_bytes + d.activation_bytes,
                kv_cache_bytes=max(p.kv_cache_bytes, d.kv_cache_bytes),
                indexer_cache_bytes=max(p.indexer_cache_bytes, d.indexer_cache_bytes),
                runtime_peak_bytes=max(
                    p.runtime_peak_bytes + d.activation_bytes,
                    d.runtime_peak_bytes + p.activation_bytes,
                ),
                outgoing_payload_bytes=max(p.outgoing_payload_bytes, d.outgoing_payload_bytes),
            )
        )
    transfers = tuple(
        replace(
            p,
            payload_bytes=max(p.payload_bytes, d.payload_bytes),
            time_s_by_model={
                key: max(p.time_s_by_model.get(key, 0), d.time_s_by_model.get(key, 0))
                for key in p.time_s_by_model.keys() | d.time_s_by_model.keys()
            },
        )
        for p, d in zip(prefill.transfers, decode.transfers)
    )
    if len(transfers) != len(prefill.transfers) or len(transfers) != len(decode.transfers):
        raise ValueError("Prefill and decode pipeline transfers differ")
    return replace(prefill, stages=tuple(stages), transfers=transfers)


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
    decode_memory_left_gb: float = 0.0
    memory_info: MemoryInfo | None = None
    pp_mixed_pd_overlap_approx: bool = False
    e2e: float = float("nan")
    makespan: float = float("nan")
    mixed_latency_source: str = "pp_approximation"
    mixed_steps: int = 0
    decode_latency_source: str = "exact"


class AggThroughputOptimizer(BaseThroughputOptimizer):
    name = "aggregation"

    def initialize(self, model_runner: ModelRunner):
        super().initialize(model_runner)
        self.scheduler = DecodeFirstWithSlack()
        self._mixed_record_cache = {}
        self._pp_step_cache = {}
        self._pp_resident_kv_cache = {}
        enable_static_capture = getattr(type(model_runner), "enable_static_meta_capture", None)
        if enable_static_capture is not None:
            enable_static_capture(model_runner)

    def get_inference_info(self, optimizer_data: OptimizerData) -> OptimizerSummary:
        max_batched_tokens = optimizer_data.max_batched_tokens
        batch_size = optimizer_data.batch_size
        input_length = optimizer_data.input_length
        effective_input_length = optimizer_data.get_effective_input_length()
        output_length = optimizer_data.output_length
        # global_concurrency = batch_size * DP (PP does not add replicas).
        concurrency = batch_size * self.dp
        variable_input_mode = optimizer_data.length_distribution is not None
        chunk_plan = optimizer_data.get_prefill_chunk_plan(
            (concurrency + self.dp - 1) // self.dp if variable_input_mode else None
        )
        prefill_num_chunks = optimizer_data.get_prefill_num_chunks(chunk_plan)

        composition_rows = []
        if self.pp > 1:
            metrics = self._get_pp_timeline_metrics(optimizer_data, concurrency=concurrency)
        else:
            metrics = self._get_timeline_metrics(optimizer_data, concurrency)
        if variable_input_mode:
            prefill_num_chunks = max(len(request.chunks) for request in self._last_aggregation_timeline.requests)
            composition_rows = optimizer_data.build_concurrency_samples(batch_size)
            composition_rows = [dict(row, samples=row["samples"] * self.dp) for row in composition_rows]

        memory_left = metrics.memory_left_gb
        token_s_device = metrics.output_throughput / self.dp / self.pp / self.tp
        parallel = format_parallel_label(
            self.model_runner.model.model_config.parallel_config,
            self.is_moe_model,
            optimizer_data.num_mtp_tokens,
            dflash_block_size=optimizer_data.dflash_block_size,
            dflash_acceptance_length=optimizer_data.dflash_acceptance_length,
            dspark_block_size=optimizer_data.dspark_block_size,
            dspark_acceptance_length=optimizer_data.dspark_acceptance_length,
            dspark_markov_rank=optimizer_data.dspark_markov_rank,
            mtp_acceptance_length=(
                optimizer_data.acceptance_length
                if getattr(optimizer_data, "speculative_method", None) == "mtp"
                else None
            ),
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
        memory_info = getattr(metrics, "memory_info", None)
        if memory_info:
            summary.set_memory_info(memory_info)
        if getattr(metrics, "pp_mixed_pd_overlap_approx", False):
            summary.set_pp_mixed_pd_overlap_approx(True)
        columns = AGG_COLUMNS.copy()
        data = [
            self.model_runner.user_input.device,
            optimizer_data.num_devices,
            self.model_runner.user_input.model_id,
            self.model_runner.user_input.quantize_linear_action,
            self.model_runner.user_input.quantize_attention_action,
            input_length,
            output_length,
            effective_input_length,
            max_batched_tokens,
            prefill_num_chunks,
            concurrency,
            metrics.ttft,
            metrics.tpot,
            metrics.output_throughput,
            token_s_device,
            parallel,
            batch_size,
            metrics.prefill_breakdowns,
            metrics.decode_breakdowns,
            memory_info["model_weight_size_gb"] if memory_info else float("nan"),
            memory_info["kv_cache_size_gb"] if memory_info else float("nan"),
            memory_info["model_activation_size_gb"] if memory_info else float("nan"),
            memory_info["device_memory_available_gb"] if memory_info else float("nan"),
        ]
        columns.append("qps")
        data.append(metrics.output_throughput / output_length)
        rows = [data]
        if variable_input_mode:
            columns.insert(columns.index("output_length"), "num_input_tokens")
            data.insert(columns.index("output_length") - 1, "all")
            columns.insert(columns.index("concurrency"), "request_ratio")
            data.insert(columns.index("concurrency") - 1, 1.0)
            columns.insert(columns.index("concurrency"), "samples")
            data.insert(columns.index("concurrency") - 1, concurrency)
            for composition_row in composition_rows:
                detail_row = data.copy()
                detail_row[columns.index("num_input_tokens")] = composition_row["num_input_tokens"]
                detail_row[columns.index("request_ratio")] = composition_row["request_ratio"]
                detail_row[columns.index("samples")] = composition_row["samples"]
                detail_row[columns.index("qps")] = None
                detail_row[columns.index("ttft")] = None
                detail_row[columns.index("tpot")] = None
                detail_row[columns.index("token/s")] = None
                detail_row[columns.index("token/s/device")] = None
                detail_row[columns.index("percentage_breakdowns(p)")] = None
                detail_row[columns.index("percentage_breakdowns(d)")] = None
                rows.append(detail_row)

        result_df = pd.DataFrame(columns=columns, data=rows).round(3)
        summary.set_summary_df(result_df)
        early_stop_reason = None
        if metrics.prefill_memory_left_gb < 0:
            early_stop_reason = EARLY_STOP_PREFILL_OOM
        elif memory_left < 0:
            early_stop_reason = EARLY_STOP_DECODE_OOM
        summary.set_early_stop_flag(memory_left, metrics.tpot, metrics.ttft, reason=early_stop_reason)

        self._maybe_set_search_info(optimizer_data, memory_left, batch_size, metrics.ttft, metrics.tpot, summary)

        return summary

    def _get_pp_timeline_metrics(self, data: OptimizerData, *, concurrency: int) -> _ChunkedAggMetrics:
        """Schedule variable or fixed requests through PP stages to completion."""
        from .aggregation_timeline import RequestShape, build_requests, page_tokens, simulate_pipeline

        runner = self.model_runner
        local_concurrency = concurrency // self.dp
        requests = build_requests(data, local_concurrency)
        decode_query_len, _ = self._resolve_forward_shape(data, True)
        decode_query_len = max(
            decode_query_len,
            (data.num_mtp_tokens or 0) + 1,
            data.dflash_block_size or 0,
            data.dspark_block_size or 0,
        )
        reserved_shapes = tuple(
            RequestShape(True, decode_query_len, request.prompt_length + data.output_length + decode_query_len)
            for request in requests
        )
        block_size = runner.user_input.block_size
        if not isinstance(block_size, int) or block_size <= 0:
            block_size = 128
        reserved_tokens = page_tokens(reserved_shapes, block_size, decode_query_len)
        accepted = 1.0 / self._fold_decode_latency_ms(1.0, data)
        perf_model = self._pp_perf_model_name()
        memory_records = []
        prefill_memory_records = []
        mixed_steps = 0
        self._pp_step_cache = getattr(self, "_pp_step_cache", {})
        self._pp_resident_kv_cache = getattr(self, "_pp_resident_kv_cache", {})
        scheduler = getattr(self, "scheduler", DecodeFirstWithSlack())

        def evaluate(shapes, resident):
            nonlocal mixed_steps
            shapes = tuple(sorted(shapes))
            mixed = any(s.is_decode for s in shapes) and any(not s.is_decode for s in shapes)
            if mixed and ((data.dflash_block_size or 0) >= 2 or (data.dspark_block_size or 0) >= 2):
                raise ValueError("Mixed-step draft speculative decoding is not supported yet")

            def phase_profile(phase_shapes):
                if phase_shapes not in self._pp_step_cache:
                    batch = [RequestInfo(s.query_len, s.seq_len, is_decode=s.is_decode) for s in phase_shapes]
                    if all(shape == phase_shapes[0] for shape in phase_shapes):
                        shape = phase_shapes[0]
                        metrics = self._get_forward_info(
                            len(batch) * self.dp,
                            data,
                            shape.is_decode,
                            query_len=shape.query_len,
                            seq_len=shape.seq_len,
                        )
                    else:
                        metrics = runner.run_inference(batch, generate_inputs_func=generate_inputs_varlen)
                    profile = getattr(metrics, "pipeline_profile", None)
                    if profile is None:
                        raise RuntimeError("PP aggregation requires a pipeline profile for every selected microbatch")
                    self._pp_step_cache[phase_shapes] = profile
                return self._pp_step_cache[phase_shapes]

            if mixed:
                prefill_shapes = tuple(s for s in shapes if not s.is_decode)
                decode_shapes = tuple(s for s in shapes if s.is_decode)
                representative = RequestShape(
                    True, max(s.query_len for s in decode_shapes), max(s.seq_len for s in decode_shapes)
                )
                count = len(decode_shapes)
                upper = min(
                    1 << (count - 1).bit_length(),
                    max(count, data.max_batched_tokens // representative.query_len),
                )
                profile = _overlap_pipeline_profiles(
                    phase_profile(prefill_shapes), phase_profile((representative,) * upper)
                )
            else:
                profile = phase_profile(shapes)
            if isinstance(runner, ModelRunner):
                reserved_key = ("reserved", reserved_shapes)
                if reserved_key not in self._pp_resident_kv_cache:
                    self._pp_resident_kv_cache[reserved_key] = _reserved_pp_kv_bytes(
                        runner, reserved_shapes, profile, block_size
                    )
                reserved_bytes = self._pp_resident_kv_cache[reserved_key]
            else:
                scheduled_tokens = page_tokens(shapes, block_size, decode_query_len)
                reserved_bytes = tuple(
                    (stage.kv_cache_bytes + stage.indexer_cache_bytes) * reserved_tokens / scheduled_tokens
                    for stage in profile.stages
                )
            # Every concurrent request has a reserved KV slot, including requests
            # still waiting for their first prefill. The profile already includes
            # KV for the selected microbatch, so add only the remaining reserve.
            stage_left = []
            incoming_payload = {transfer.target_stage_id: transfer.payload_bytes for transfer in profile.transfers}
            for stage, reserved_kv in zip(profile.stages, reserved_bytes):
                scheduled_kv = stage.kv_cache_bytes + stage.indexer_cache_bytes
                compute_peak = stage.runtime_peak_bytes + max(0, reserved_kv - scheduled_kv)
                communication_buffer = max(incoming_payload.get(stage.stage_id, 0), stage.outgoing_payload_bytes)
                communication_peak = stage.weight_bytes + reserved_kv + communication_buffer
                peak = max(compute_peak, communication_peak)
                stage_left.append(
                    runner.total_device_memory_gb
                    - (peak / 1024**3)
                    - float(runner.user_input.reserved_memory_gb or 0.0)
                )
            memory_left = min(stage_left)
            memory_records.append(memory_left)
            if any(not shape.is_decode for shape in shapes):
                prefill_memory_records.append(memory_left)
            if mixed:
                mixed_steps += 1
            return profile, memory_left

        timeline = simulate_pipeline(
            requests,
            data,
            scheduler,
            decode_query_len,
            accepted,
            self._pp_microbatch_size(),
            evaluate,
            perf_model,
        )
        self._last_aggregation_timeline = timeline
        data.pipeline_schedule_estimate = timeline.pipeline_schedule_estimate
        memory_left = min(memory_records, default=float("inf"))
        prefill_left = min(prefill_memory_records, default=memory_left)
        if timeline.oom:
            ttft = tpot = e2e = float("inf")
            throughput = 0.0
        else:
            ttft, e2e = timeline.ttft, timeline.e2e
            tpot = (e2e - ttft) / (data.output_length - 1) if data.output_length > 1 else 0.0
            throughput = 1000 * concurrency * data.output_length / timeline.elapsed_ms
        memory_info = {
            "total_device_memory_gb": runner.total_device_memory_gb,
            "model_weight_size_gb": float("nan"),
            "kv_cache_size_gb": float("nan"),
            "model_activation_size_gb": float("nan"),
            "reserved_memory_gb": float(runner.user_input.reserved_memory_gb or 0.0),
            "device_memory_available_gb": memory_left,
        }
        prefill_latencies = [latency for shapes, latency in timeline.steps if all(not s.is_decode for s in shapes)]
        decode_latencies = [latency for shapes, latency in timeline.steps if all(s.is_decode for s in shapes)]
        return _ChunkedAggMetrics(
            ttft=ttft,
            tpot=tpot,
            output_throughput=throughput,
            memory_left_gb=memory_left,
            prefill_latency=prefill_latencies[0] if prefill_latencies else 0.0,
            prefill_last_latency=prefill_latencies[-1] if prefill_latencies else 0.0,
            prefill_memory_left_gb=prefill_left,
            decode_latency=max(decode_latencies, default=0.0),
            prefill_breakdowns="",
            decode_breakdowns="",
            memory_info=memory_info,
            e2e=e2e,
            makespan=timeline.elapsed_ms,
            mixed_latency_source="phase_max_approx" if mixed_steps else "no_mixed_steps",
            mixed_steps=mixed_steps,
        )

    def _get_timeline_metrics(self, data, concurrency):
        """Run one replica's finite batch and scale completion counts by DP."""
        from .aggregation_timeline import RequestShape, build_requests, page_tokens, simulate

        runner = self.model_runner
        offload_enabled = any(
            getattr(runner.user_input, name, False)
            for name in (
                "enable_prefill_kv_offload",
                "enable_decode_kv_offload",
            )
        )
        local_concurrency = concurrency // self.dp
        requests = build_requests(data, local_concurrency)
        query_len, _ = self._resolve_forward_shape(data, True)
        reserved_shapes = tuple(
            RequestShape(True, query_len, request.prompt_length + data.output_length + query_len)
            for request in requests
        )
        block = runner.user_input.block_size or 128
        reserved_tokens = page_tokens(reserved_shapes, block, query_len)
        resident_sizer = getattr(runner, "get_resident_kv_cache_size_gb", None)
        if offload_enabled:
            reserved_kv = None
        elif resident_sizer is not None or isinstance(runner, ModelRunner):
            reserved_kv = _reserved_kv_cache_size_gb(runner, reserved_shapes)
        else:
            reserved_kv = None
        resident_kv_cache = {}
        accepted = 1.0 / self._fold_decode_latency_ms(1.0, data)
        records = []
        phase_records = {False: [], True: []}
        mixed_count = 0
        interpolated_decode_count = 0

        def forward_record(shapes):
            """Evaluate a homogeneous P or D phase with the existing caches."""
            if all(s == shapes[0] for s in shapes):
                shape = shapes[0]
                key = self._make_forward_shape_key(
                    len(shapes) * self.dp,
                    data,
                    shape.is_decode,
                    query_len=shape.query_len,
                    seq_len=shape.seq_len,
                )
                return self._compute_forward_latency_record(key, data)
            if shapes not in self._mixed_record_cache:
                batch = [RequestInfo(s.query_len, s.seq_len, is_decode=s.is_decode) for s in shapes]
                metrics = runner.run_inference(batch, generate_inputs_func=generate_inputs_varlen)
                self._mixed_record_cache[shapes] = self._build_forward_latency_record(metrics)
                logger.info(
                    "Heterogeneous Forward Latency: %.4f ms, batch=%d, prefill=%d, decode=%d, query_tokens=%d",
                    self._mixed_record_cache[shapes].latency_ms,
                    len(shapes),
                    sum(not s.is_decode for s in shapes),
                    sum(s.is_decode for s in shapes),
                    sum(s.query_len for s in shapes),
                )
            return self._mixed_record_cache[shapes]

        def pure_decode_record(shape, count):
            """Interpolate latency between measured powers of two; bound memory above."""
            budget_cap = max(count, data.max_batched_tokens // shape.query_len)
            upper = min(1 << (count - 1).bit_length(), budget_cap)
            upper_record = forward_record((shape,) * upper)
            if upper == count:
                return upper_record, False
            lower = 1 << (count.bit_length() - 1)
            lower_record = forward_record((shape,) * lower)
            weight = (count - lower) / (upper - lower)
            latency = lower_record.latency_ms + weight * (upper_record.latency_ms - lower_record.latency_ms)
            return replace(upper_record, latency_ms=latency), True

        def evaluate(shapes, resident):
            nonlocal mixed_count, interpolated_decode_count
            shapes = tuple(sorted(shapes))
            homogeneous = all(s == shapes[0] for s in shapes)
            mixed = any(s.is_decode for s in shapes) and any(not s.is_decode for s in shapes)
            decode_interpolated = False
            if mixed and ((data.dflash_block_size or 0) >= 2 or (data.dspark_block_size or 0) >= 2):
                raise ValueError("Mixed-step draft speculative decoding is not supported yet")
            if not homogeneous and offload_enabled:
                raise ValueError("Heterogeneous aggregation steps with KV offload are not supported")
            if mixed:
                prefill_shapes = tuple(s for s in shapes if not s.is_decode)
                decode_shapes = tuple(s for s in shapes if s.is_decode)
                p_record = forward_record(prefill_shapes)
                # One upper power-of-two D batch covers the current bucket. Use
                # its memory and latency directly; pure-D steps interpolate.
                representative = RequestShape(
                    True, max(s.query_len for s in decode_shapes), max(s.seq_len for s in decode_shapes)
                )
                count = len(decode_shapes)
                upper = min(
                    1 << (count - 1).bit_length(),
                    max(count, data.max_batched_tokens // representative.query_len),
                )
                d_record = forward_record((representative,) * upper)
                record = replace(p_record, latency_ms=max(p_record.latency_ms, d_record.latency_ms))
                if (
                    p_record.memory_info
                    and d_record.memory_info
                    and "model_activation_size_gb" in p_record.memory_info
                    and "model_activation_size_gb" in d_record.memory_info
                ):
                    p_activation = p_record.memory_info["model_activation_size_gb"]
                    d_activation = d_record.memory_info["model_activation_size_gb"]
                    # Keep one phase's weights/workspace, then reserve both
                    # phases' activations for the mixed batch.
                    base_record, other_activation = min(
                        ((p_record, d_activation), (d_record, p_activation)),
                        key=lambda pair: pair[0].memory_left_gb - pair[1],
                    )
                    info = dict(base_record.memory_info)
                    activation = p_activation + d_activation
                    info["model_activation_size_gb"] = activation
                    info["device_memory_available_gb"] -= other_activation
                    record = replace(
                        record, memory_left_gb=base_record.memory_left_gb - other_activation, memory_info=info
                    )
                else:
                    record = replace(record, memory_left_gb=min(p_record.memory_left_gb, d_record.memory_left_gb))
            else:
                if homogeneous and shapes[0].is_decode:
                    record, decode_interpolated = pure_decode_record(shapes[0], len(shapes))
                else:
                    record = forward_record(shapes)
            if mixed:
                mixed_count += 1
            # Reserve KV for the entire requested concurrency, including requests
            # waiting for prefill.
            scheduled_tokens = page_tokens(shapes, block, query_len)
            resident_tokens = page_tokens(resident, block, query_len) if offload_enabled else 0
            if offload_enabled and resident_tokens > scheduled_tokens:
                raise ValueError(
                    "Inactive resident KV accounting with offload is not supported by aggregation timeline"
                )

            def with_resident_kv(phase_record):
                info = phase_record.memory_info
                if not info or not scheduled_tokens:
                    return phase_record
                if offload_enabled:
                    if resident_sizer is not None or isinstance(runner, ModelRunner):
                        resident_key = tuple(sorted(resident))
                        if resident_key not in resident_kv_cache:
                            resident_requests = [
                                RequestInfo(shape.query_len, shape.seq_len, is_decode=shape.is_decode)
                                for shape in resident_key
                            ]
                            resident_kv_cache[resident_key] = (
                                resident_sizer(resident_requests)
                                if resident_sizer is not None
                                else _resident_kv_cache_size_gb(runner, resident_requests)
                            )
                        full_kv = resident_kv_cache[resident_key]
                    else:
                        full_kv = info["kv_cache_size_gb"] * resident_tokens / scheduled_tokens
                elif reserved_kv is None:
                    # Lightweight evaluator implementations may expose only a
                    # latency record, with no model/input generator attached.
                    full_kv = info["kv_cache_size_gb"] * reserved_tokens / scheduled_tokens
                else:
                    full_kv = reserved_kv
                extra_kv = max(0.0, full_kv - info["kv_cache_size_gb"])
                info = dict(info)
                info["kv_cache_size_gb"] += extra_kv
                info["device_memory_available_gb"] -= extra_kv
                return replace(phase_record, memory_left_gb=phase_record.memory_left_gb - extra_kv, memory_info=info)

            record = with_resident_kv(record)
            if decode_interpolated:
                interpolated_decode_count += 1
            if not mixed:
                phase_records[shapes[0].is_decode].append(record)
            records.append(record)
            return record

        timeline = simulate(requests, data, self.scheduler, query_len, accepted, evaluate)
        self._last_aggregation_timeline = timeline
        info = select_tightest_memory_info(r.memory_info for r in records)
        memory = min(r.memory_left_gb for r in records)
        prefill = phase_records[False]
        decode = phase_records[True]
        p_latency = prefill[0].latency_ms if prefill else 0.0
        p_last_latency = prefill[-1].latency_ms if prefill else 0.0
        d_latency = max((r.latency_ms for r in decode), default=0.0)
        if timeline.oom:
            ttft, tpot, e2e, throughput = float("inf"), float("inf"), float("inf"), 0.0
            if any(not shape.is_decode for shape in timeline.steps[-1][0]):
                prefill_memory = memory
            else:
                prefill_memory = min((r.memory_left_gb for r in prefill), default=float("inf"))
        else:
            ttft, e2e = timeline.ttft, timeline.e2e
            tpot = (e2e - ttft) / (data.output_length - 1) if data.output_length > 1 else 0.0
            throughput = 1000 * concurrency * data.output_length / timeline.elapsed_ms
            prefill_memory = min((r.memory_left_gb for r in prefill), default=memory)
        source = "phase_max_approx" if mixed_count else "no_mixed_steps"
        logger.info(
            "Aggregation timeline v2: workload=finite_batch, E2E mean=%.4f ms, "
            "Makespan=%.4f ms, QPS=%.6f, Mixed steps=%d, source=%s",
            e2e,
            timeline.elapsed_ms,
            throughput / data.output_length,
            mixed_count,
            source,
        )
        return _ChunkedAggMetrics(
            ttft=ttft,
            tpot=tpot,
            output_throughput=throughput,
            memory_left_gb=memory,
            prefill_latency=p_latency,
            prefill_last_latency=p_last_latency,
            prefill_memory_left_gb=prefill_memory,
            decode_latency=d_latency,
            prefill_breakdowns=prefill[0].breakdowns if prefill else "",
            decode_breakdowns=decode[0].breakdowns if decode else "",
            memory_info=info,
            e2e=e2e,
            makespan=float("inf") if timeline.oom else timeline.elapsed_ms,
            mixed_latency_source=source,
            mixed_steps=mixed_count,
            decode_latency_source="exact" if not interpolated_decode_count else "interpolated",
        )
