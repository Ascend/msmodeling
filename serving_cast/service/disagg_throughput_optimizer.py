# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.

import logging

import pandas as pd

from tensor_cast.core.model_runner import ModelRunner

from .base_throughput_optimizer import BaseThroughputOptimizer
from .latency_table import ForwardLatencyTable
from .optimizer_summary import (
    EARLY_STOP_DECODE_OOM,
    EARLY_STOP_PREFILL_OOM,
    OptimizerSummary,
)
from .utils import (
    DISAGG_COLUMNS,
    build_memory_info,
    format_breakdowns,
    format_parallel_label,
    OptimizerData,
    select_tightest_memory_info,
    UnsupportedPPConfigurationError,
)


logger = logging.getLogger(__name__)


class DisaggThroughputOptimizer(BaseThroughputOptimizer):
    name = "disaggregation"

    @staticmethod
    def _is_unbounded_prefill(optimizer_data: OptimizerData) -> bool:
        """Whether Prefill should model one full-concurrency forward."""
        return (
            optimizer_data.ttft_limits is not None
            and optimizer_data.length_distribution is None
            and optimizer_data.max_batched_tokens is None
        )

    def run(self, optimizer_data: OptimizerData, batch_range: list[int]) -> OptimizerSummary | None:
        """Run an unbounded Prefill when no serving token budget was requested.

        Disaggregated Prefill results, including the Prefill side of PD ratio,
        are convertible to one ``text_generate`` forward. An omitted
        ``max_batched_tokens`` therefore must not acquire the base optimizer's
        automatic serving budget: that budget would turn one reported
        concurrency into several Prefill waves and understate the corresponding
        single-forward peak memory. Explicit budgets retain the existing
        chunk/wave model through the base implementation. Length-distribution
        Prefill also retains the base path because it needs an effective token
        budget to construct its representative chunk plan.

        An unbounded run can still search for a smaller valid concurrency after
        an OOM, but it never retries the same concurrency with a smaller token
        budget: each reported row continues to represent one full forward.
        """
        if self._is_unbounded_prefill(optimizer_data):
            return self._run_once(optimizer_data, batch_range)
        return super().run(optimizer_data, batch_range)

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

    def get_inference_info(self, optimizer_data: OptimizerData) -> OptimizerSummary:
        # check prefill or decode
        decode_flag = optimizer_data.ttft_limits is None
        variable_input_mode = optimizer_data.length_distribution is not None
        composition_rows = []

        batch_size = optimizer_data.batch_size
        input_length = optimizer_data.input_length
        effective_input_length = optimizer_data.get_effective_input_length()
        # Pipeline parallel only splits a single request's model execution
        # across stages — it does not create request replicas.  Global request
        # concurrency is batch_size * dp, matching the aggregation optimizer.
        concurrency = batch_size * self.dp
        unbounded_prefill = self._is_unbounded_prefill(optimizer_data)
        if decode_flag or unbounded_prefill:
            chunk_plan = []
            global_batched_token_limit = None
        else:
            chunk_plan = optimizer_data.get_prefill_chunk_plan(
                (concurrency + self.dp - 1) // self.dp if variable_input_mode else None
            )
            global_batched_token_limit = self._get_global_batched_token_limit(optimizer_data)

        prefill_num_chunks = 1 if unbounded_prefill else optimizer_data.get_prefill_num_chunks(chunk_plan)
        output_length = optimizer_data.output_length
        prefill_ttft_sum_ms = None
        prefill_ttft_request_count = 0
        single_prefill_fits_budget = (
            not decode_flag
            and not variable_input_mode
            and len(chunk_plan) == 1
            and concurrency * chunk_plan[0].query_len <= global_batched_token_limit
        )

        if self.pp > 1:
            return self._get_pp_inference_info(
                optimizer_data,
                concurrency=concurrency,
                decode_flag=decode_flag,
                chunk_plan=chunk_plan,
            )

        if decode_flag or variable_input_mode or unbounded_prefill or single_prefill_fits_budget:
            if variable_input_mode:
                chunk_results, composition_rows = self._get_batched_forward_info(
                    concurrency,
                    optimizer_data,
                    chunk_plan,
                )
                latency_ms = optimizer_data.serving_cost
                device_memory_available_gb = float("inf")
                memory_info = None
                breakdown_sums = {}
                breakdown_counts = {}
                prefill_ttft_sum_ms = 0.0
                for batch_result, completed_requests in chunk_results:
                    chunk_latency_ms = self._select_latency_s(batch_result.execution_time_s) * 1000
                    latency_ms += chunk_latency_ms
                    device_memory_available_gb = min(
                        device_memory_available_gb,
                        batch_result.device_memory_available_gb,
                    )
                    memory_info = select_tightest_memory_info((memory_info, build_memory_info(batch_result)))
                    if batch_result.device_memory_available_gb < 0:
                        break

                    # Chunks execute sequentially, so requests completed by this chunk
                    # observe the cumulative latency of this and all preceding chunks.
                    prefill_ttft_sum_ms += completed_requests * latency_ms
                    prefill_ttft_request_count += completed_requests

                    for breakdown_name, breakdown in batch_result.breakdowns.items():
                        total = sum(breakdown.values())
                        if total == 0:
                            continue
                        normalized_breakdown = {}
                        for category, value in breakdown.items():
                            if isinstance(value, float):
                                normalized_breakdown[category] = value / total
                        if normalized_breakdown:
                            accumulated = breakdown_sums.setdefault(breakdown_name, {})
                            for category, value in normalized_breakdown.items():
                                accumulated[category] = accumulated.get(category, 0.0) + value
                            breakdown_counts[breakdown_name] = breakdown_counts.get(breakdown_name, 0) + 1

                breakdowns = ""
                if breakdown_sums:
                    average_breakdowns = {
                        breakdown_name: {
                            category: value / breakdown_counts[breakdown_name] for category, value in breakdown.items()
                        }
                        for breakdown_name, breakdown in breakdown_sums.items()
                    }
                    breakdowns = format_breakdowns(average_breakdowns)
            else:
                batch_result = self._get_forward_info(concurrency, optimizer_data, decode_flag)
                latency_ms = self._select_latency_s(batch_result.execution_time_s) * 1000 + optimizer_data.serving_cost
                device_memory_available_gb = batch_result.device_memory_available_gb
                breakdowns = format_breakdowns(batch_result.breakdowns)
                memory_info = build_memory_info(batch_result)
        else:
            latency_ms = optimizer_data.serving_cost
            device_memory_available_gb = float("inf")
            breakdowns = ""
            memory_info = None
            breakdown_sums = {}
            breakdown_counts = {}
            wave_keys = []
            wave_specs = []
            prefill_ttft_sum_ms = 0.0
            # Keep disaggregated prefill modeling simple and deterministic: each wave contains
            # only one chunk shape and is capped by max_batched_tokens. We do not aggregate
            # different chunk positions across queries into one wave, so this may be conservative
            # compared with engines that do cross-query chunk packing.
            # serving_cost is treated as one fixed phase overhead, while breakdowns are averaged
            # across all modeled waves to include every chunk shape. latency_ms is the phase
            # makespan; final-chunk wave completion timestamps are used for request-level TTFT.
            for chunk_index, chunk in enumerate(chunk_plan):
                wave_size = max(global_batched_token_limit // chunk.query_len, 1)
                remaining = concurrency
                while remaining > 0:
                    wave_concurrency = min(wave_size, remaining)
                    wave_key = self._make_forward_shape_key(
                        wave_concurrency,
                        optimizer_data,
                        decode_flag,
                        query_len=chunk.query_len,
                        seq_len=chunk.seq_len,
                    )
                    wave_keys.append(wave_key)
                    wave_specs.append(
                        (
                            wave_key,
                            wave_concurrency,
                            chunk_index == len(chunk_plan) - 1,
                        )
                    )
                    remaining -= wave_concurrency

            latency_table = ForwardLatencyTable(
                self,
                optimizer_data,
            )
            latency_table.prefetch(wave_keys)

            for key, wave_concurrency, is_final_chunk in wave_specs:
                record = latency_table.get(key)
                latency_ms += record.latency_ms
                device_memory_available_gb = min(
                    device_memory_available_gb,
                    record.memory_left_gb,
                )
                memory_info = select_tightest_memory_info((memory_info, record.memory_info))
                if record.memory_left_gb < 0:
                    break
                if is_final_chunk:
                    prefill_ttft_sum_ms += wave_concurrency * latency_ms
                    prefill_ttft_request_count += wave_concurrency
                # Preserve the historical per-wave weighting: each wave contributes one normalized
                # breakdown distribution, even when multiple waves reuse the same latency table record.
                for breakdown_name, breakdown in record.raw_breakdowns.items():
                    total = sum(breakdown.values())
                    if total == 0:
                        continue
                    normalized_breakdown = {}
                    for category, value in breakdown.items():
                        if isinstance(value, float):
                            normalized_breakdown[category] = value / total
                    if normalized_breakdown:
                        accumulated = breakdown_sums.setdefault(breakdown_name, {})
                        for category, value in normalized_breakdown.items():
                            accumulated[category] = accumulated.get(category, 0.0) + value
                        breakdown_counts[breakdown_name] = breakdown_counts.get(breakdown_name, 0) + 1

            if breakdown_sums:
                average_breakdowns = {
                    breakdown_name: {
                        category: value / breakdown_counts[breakdown_name] for category, value in breakdown.items()
                    }
                    for breakdown_name, breakdown in breakdown_sums.items()
                }
                breakdowns = format_breakdowns(average_breakdowns)

        ttft = tpot = None
        if decode_flag:
            latency_ms = self._fold_decode_latency_ms(latency_ms, optimizer_data)
            tpot = latency_ms
            output_throughput = concurrency / tpot * 1000 if tpot > 0 else 0
        else:
            total_input_tokens = 0
            if variable_input_mode:
                for composition_row in composition_rows:
                    total_input_tokens += composition_row["num_input_tokens"] * composition_row["samples"]
                total_input_tokens *= self.dp
            else:
                total_input_tokens = concurrency * input_length
            expected_request_count = (
                sum(composition_row["samples"] for composition_row in composition_rows)
                if variable_input_mode
                else concurrency
            )
            if prefill_ttft_sum_ms is not None and prefill_ttft_request_count == expected_request_count:
                ttft = prefill_ttft_sum_ms / expected_request_count
            else:
                # OOM/partial replay has no complete request-level TTFT average. Keep the
                # phase latency for diagnostics; memory-based early stop takes precedence.
                ttft = latency_ms
            output_throughput = total_input_tokens / latency_ms * 1000 if latency_ms > 0 else 0

        token_s_device = output_throughput / self.dp / self.pp / self.tp
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
            "TTFT: %r ms, TPOT: %r ms, "
            "Output Throughput: %.2f token/s, "
            "Concurrency: %d, "
            "parallel: %s, "
            "Memory Left: %.2f GB",
            ttft,
            tpot,
            output_throughput,
            concurrency,
            parallel,
            device_memory_available_gb,
        )

        summary = OptimizerSummary(optimizer_data)
        if memory_info:
            summary.set_memory_info(memory_info)
        columns = DISAGG_COLUMNS.copy()
        data = [
            self.model_runner.user_input.device,
            optimizer_data.num_devices,
            self.model_runner.user_input.model_id,
            self.model_runner.user_input.quantize_linear_action,
            self.model_runner.user_input.quantize_attention_action,
            input_length,
            output_length,
            effective_input_length,
            optimizer_data.max_batched_tokens,
            prefill_num_chunks,
            concurrency,
            ttft,
            tpot,
            output_throughput,
            token_s_device,
            parallel,
            batch_size,
            breakdowns,
            memory_info["model_weight_size_gb"] if memory_info else float("nan"),
            memory_info["kv_cache_size_gb"] if memory_info else float("nan"),
            memory_info["model_activation_size_gb"] if memory_info else float("nan"),
            memory_info["device_memory_available_gb"] if memory_info else float("nan"),
        ]
        rows = [data]
        if variable_input_mode and not decode_flag:
            columns.insert(columns.index("output_length"), "num_input_tokens")
            data.insert(columns.index("output_length") - 1, "all")
            columns.insert(columns.index("concurrency"), "request_ratio")
            data.insert(columns.index("concurrency") - 1, 1.0)
            columns.insert(columns.index("concurrency"), "samples")
            data.insert(columns.index("concurrency") - 1, concurrency)
            #
            for composition_row in composition_rows:
                detail_row = data.copy()
                detail_row[columns.index("num_input_tokens")] = composition_row["num_input_tokens"]
                detail_row[columns.index("request_ratio")] = composition_row["request_ratio"]
                detail_row[columns.index("samples")] = composition_row["samples"]
                detail_row[columns.index("ttft")] = None
                detail_row[columns.index("tpot")] = None
                detail_row[columns.index("token/s")] = None
                detail_row[columns.index("token/s/device")] = None
                detail_row[columns.index("percentage_breakdowns")] = None
                rows.append(detail_row)

        result_df = pd.DataFrame(columns=columns, data=rows).round(3)
        summary.set_summary_df(result_df)
        early_stop_reason = None
        if device_memory_available_gb < 0:
            early_stop_reason = EARLY_STOP_DECODE_OOM if decode_flag else EARLY_STOP_PREFILL_OOM
        summary.set_early_stop_flag(device_memory_available_gb, tpot, ttft, reason=early_stop_reason)

        self._maybe_set_search_info(optimizer_data, device_memory_available_gb, batch_size, ttft, tpot, summary)

        return summary

    def _get_pp_inference_info(
        self,
        optimizer_data: OptimizerData,
        *,
        concurrency: int,
        decode_flag: bool,
        chunk_plan: list,
    ) -> OptimizerSummary:
        """Evaluate one disaggregated PP phase with phase-specific formulas."""
        batch_size = optimizer_data.batch_size
        input_length = optimizer_data.input_length
        effective_input_length = optimizer_data.get_effective_input_length()
        max_batched_tokens = optimizer_data.max_batched_tokens
        output_length = optimizer_data.output_length
        serving_cost_ms = optimizer_data.serving_cost or 0.0

        if decode_flag:
            if optimizer_data.length_distribution is not None:
                raise UnsupportedPPConfigurationError(
                    "PP>1 decode does not support variable-length input distribution; "
                    "use a fixed input_length or disable PP."
                )
            wave = self._evaluate_pp_wave(
                batch_size,
                optimizer_data,
                is_decode=True,
                repeat=True,
                resident_policy="full",
            )
            repeated = wave.repeated
            assert repeated is not None
            # Apply speculative-decode fold (MTP/DFlash/DSpark) to the scheduler's
            # steady-state TPOT and wave period. serving_cost_ms is a per-step
            # overhead and must stay outside the fold (it is not compute latency).
            # When no speculative method is configured the fold is a no-op.
            tpot = self._fold_decode_latency_ms(repeated.worst_tpot_s * 1000.0, optimizer_data) + serving_cost_ms
            ttft = None
            throughput_interval_s = (
                self._fold_decode_latency_ms(repeated.measured_interval_s * 1000.0, optimizer_data) / 1000.0
                + serving_cost_ms / 1000.0
            )
            output_throughput = batch_size * self.dp / throughput_interval_s if throughput_interval_s > 0 else 0.0
        else:
            early_stop_reason = self._validate_pp_prefill_wave(optimizer_data, chunk_plan)
            if early_stop_reason is not None:
                summary = OptimizerSummary(optimizer_data)
                summary.set_early_stop_flag(-1.0, None, None, reason=early_stop_reason)
                return summary
            wave = self._evaluate_pp_wave(
                batch_size,
                optimizer_data,
                is_decode=False,
                query_len=effective_input_length,
                seq_len=effective_input_length,
                resident_policy="inflight",
                chunk_shapes=[(c.query_len, c.seq_len) for c in chunk_plan] if len(chunk_plan) > 1 else None,
            )
            ttft = wave.schedule.makespan_s * 1000.0 + serving_cost_ms
            tpot = None
            completed_tokens = batch_size * effective_input_length
            output_throughput = completed_tokens * self.dp * 1000.0 / ttft if ttft > 0 else 0.0

        device_memory_available_gb = wave.memory_left_gb
        token_s_device = output_throughput / self.dp / self.pp / self.tp
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
            "PP>1 %s: TTFT=%r ms, TPOT=%r ms, Throughput=%.2f token/s, "
            "Concurrency=%d, makespan=%.4f s, bottleneck_stage=%d, Memory Left=%.2f GB",
            "decode" if decode_flag else "prefill",
            ttft,
            tpot,
            output_throughput,
            concurrency,
            wave.schedule.makespan_s,
            wave.bottleneck_stage_id,
            device_memory_available_gb,
        )

        memory_info = {
            "total_device_memory_gb": self.model_runner.total_device_memory_gb,
            "model_weight_size_gb": float("nan"),
            "kv_cache_size_gb": float("nan"),
            "model_activation_size_gb": float("nan"),
            "reserved_memory_gb": float(self.model_runner.user_input.reserved_memory_gb or 0.0),
            "device_memory_available_gb": device_memory_available_gb,
        }
        summary = OptimizerSummary(optimizer_data)
        summary.set_memory_info(memory_info)
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
            len(chunk_plan),
            concurrency,
            ttft,
            tpot,
            output_throughput,
            token_s_device,
            parallel,
            batch_size,
            "",
            memory_info["model_weight_size_gb"],
            memory_info["kv_cache_size_gb"],
            memory_info["model_activation_size_gb"],
            memory_info["device_memory_available_gb"],
        ]
        result_df = pd.DataFrame(columns=DISAGG_COLUMNS, data=[data]).round(3)
        summary.set_summary_df(result_df)

        early_stop_reason = None
        if wave.memory_exceeded:
            early_stop_reason = EARLY_STOP_DECODE_OOM if decode_flag else EARLY_STOP_PREFILL_OOM
        summary.set_early_stop_flag(device_memory_available_gb, tpot, ttft, reason=early_stop_reason)
        self._maybe_set_search_info(
            optimizer_data,
            device_memory_available_gb,
            batch_size,
            ttft,
            tpot,
            summary,
        )
        return summary
