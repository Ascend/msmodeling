# Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.

import logging

import pandas as pd

from tensor_cast.core.model_runner import ModelRunner
from .base_throughput_optimizer import BaseThroughputOptimizer
from .optimizer_summary import OptimizerSummary
from .utils import (
    DISAGG_COLUMNS,
    format_breakdowns,
    format_parallel_label,
    OptimizerData,
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

    def get_inference_info(self, optimizer_data: OptimizerData) -> OptimizerSummary:
        # check prefill or decode
        decode_flag = optimizer_data.ttft_limits is None

        batch_size = optimizer_data.batch_size
        input_length = optimizer_data.input_length
        output_length = optimizer_data.output_length
        concurrency = batch_size * self.dp * self.pp

        batch_result = self._get_forward_info(concurrency, optimizer_data, decode_flag)
        latency_ms = batch_result.execution_time_s.get("analytic") * 1000 + optimizer_data.serving_cost
        device_memory_available_gb = batch_result.device_memory_available_gb
        breakdowns = format_breakdowns(batch_result.breakdowns)

        ttft = tpot = None
        if decode_flag:
            average_tokens = sum(optimizer_data.mtp_acceptance_rate[: optimizer_data.num_mtp_tokens]) + 1
            latency_ms /= average_tokens
            tpot = latency_ms
            output_throughput = concurrency / tpot * 1000 if tpot > 0 else 0
        else:
            ttft = latency_ms
            output_throughput = concurrency / latency_ms * 1000 * input_length if latency_ms > 0 else 0

        token_s_device = output_throughput / self.dp / self.pp / self.tp
        parallel = format_parallel_label(
            self.model_runner.model.model_config.parallel_config,
            self.is_moe_model,
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
        result_df = pd.DataFrame(
            columns=DISAGG_COLUMNS,
            data=[
                [
                    self.model_runner.user_input.device,
                    optimizer_data.num_devices,
                    self.model_runner.user_input.model_id,
                    self.model_runner.user_input.quantize_linear_action,
                    self.model_runner.user_input.quantize_attention_action,
                    input_length,
                    output_length,
                    concurrency,
                    ttft,
                    tpot,
                    output_throughput,
                    token_s_device,
                    parallel,
                    batch_size,
                    breakdowns,
                ]
            ],
        ).round(3)
        summary.set_summary_df(result_df)
        summary.set_early_stop_flag(device_memory_available_gb, tpot, ttft)

        self._maybe_set_search_info(optimizer_data, device_memory_available_gb, batch_size, ttft, tpot, summary)

        return summary
