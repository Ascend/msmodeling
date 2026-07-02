# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.

import logging

import pandas as pd

from tensor_cast.core.model_runner import ModelRunner
from .base_throughput_optimizer import BaseThroughputOptimizer
from .latency_table import ForwardLatencyTable
from .optimizer_summary import OptimizerSummary
from .utils import (
    DISAGG_COLUMNS,
    build_memory_info,
    format_breakdowns,
    format_parallel_label,
    OptimizerData,
    select_tightest_memory_info,
)


logger = logging.getLogger(__name__)


class DisaggThroughputOptimizer(BaseThroughputOptimizer):
    name = "disaggregation"

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
        max_batched_tokens = optimizer_data.max_batched_tokens
        if decode_flag:
            chunk_plan = []
        else:
            chunk_plan = optimizer_data.get_prefill_chunk_plan()
        output_length = optimizer_data.output_length
        concurrency = batch_size * self.dp * self.pp

        if decode_flag or len(chunk_plan) == 1:
            if variable_input_mode:
                batch_result, composition_rows = self._get_batched_forward_info(concurrency, optimizer_data)
            else:
                batch_result = self._get_forward_info(concurrency, optimizer_data, decode_flag)
            latency_ms = batch_result.execution_time_s.get("analytic") * 1000 + optimizer_data.serving_cost
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
            # Keep disaggregated prefill modeling simple and deterministic: each wave contains
            # only one chunk shape and is capped by max_batched_tokens. We do not aggregate
            # different chunk positions across queries into one wave, so this may be conservative
            # compared with engines that do cross-query chunk packing.
            # serving_cost is treated as one fixed phase overhead, while breakdowns are averaged
            # across all modeled waves to include every chunk shape.
            for chunk in chunk_plan:
                wave_size = max(max_batched_tokens // chunk.query_len, 1)
                remaining = concurrency
                while remaining > 0:
                    wave_concurrency = min(wave_size, remaining)
                    wave_keys.append(
                        self._make_forward_shape_key(
                            wave_concurrency,
                            optimizer_data,
                            decode_flag,
                            query_len=chunk.query_len,
                            seq_len=chunk.seq_len,
                        )
                    )
                    remaining -= wave_concurrency

            latency_table = ForwardLatencyTable(
                self,
                optimizer_data,
            )
            latency_table.prefetch(wave_keys)

            for key in wave_keys:
                record = latency_table.get(key)
                latency_ms += record.latency_ms
                device_memory_available_gb = min(
                    device_memory_available_gb,
                    record.memory_left_gb,
                )
                memory_info = select_tightest_memory_info((memory_info, record.memory_info))
                if record.memory_left_gb < 0:
                    break
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
            average_tokens = sum(optimizer_data.mtp_acceptance_rate[: optimizer_data.num_mtp_tokens]) + 1
            latency_ms /= average_tokens
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
            ttft = latency_ms
            output_throughput = total_input_tokens / latency_ms * 1000 if latency_ms > 0 else 0

        token_s_device = output_throughput / self.dp / self.pp / self.tp
        parallel = format_parallel_label(
            self.model_runner.model.model_config.parallel_config,
            self.is_moe_model,
            optimizer_data.num_mtp_tokens,
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
            max_batched_tokens,
            len(chunk_plan),
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
        summary.set_early_stop_flag(device_memory_available_gb, tpot, ttft)

        self._maybe_set_search_info(optimizer_data, device_memory_available_gb, batch_size, ttft, tpot, summary)

        return summary
