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
from collections import defaultdict

import pandas as pd

from tensor_cast.core.model_runner import ModelRunner
from .base_throughput_optimizer import BaseThroughputOptimizer
from .optimizer_summary import OptimizerSummary
from .utils import AGG_COLUMNS, format_breakdowns, format_parallel_label, OptimizerData


logger = logging.getLogger(__name__)


class AggThroughputOptimizer(BaseThroughputOptimizer):
    name = "aggregation"

    def initialize(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.num_mtp_tokens = (
            self.model_runner.model.model_config.mtp_config.num_mtp_layers
            if self.model_runner.model.model_config.mtp_config is not None
            else 0
        )
        self.dp = (
            self.model_runner.model.model_config.parallel_config.data_parallel_size
        )
        self.tp = (
            self.model_runner.model.model_config.parallel_config.tensor_parallel_size
        )
        self.pp = (
            self.model_runner.model.model_config.parallel_config.pipeline_parallel_size
        )
        self.ep = (
            self.model_runner.model.model_config.parallel_config.expert_parallel_size
        )
        self.moe_tp = self.model_runner.model.model_config.parallel_config.moe_tensor_parallel_size
        self.moe_dp = (
            self.model_runner.model.model_config.parallel_config.moe_data_parallel_size
        )
        self.is_moe_model = self.model_runner.model.model_config.moe_config is not None
        self._prefill_cache = defaultdict(lambda: None)
        self._decode_cache = defaultdict(lambda: None)

    def get_inference_info(self, optimizer_data: OptimizerData) -> OptimizerSummary:
        max_prefill_tokens = optimizer_data.max_prefill_tokens
        batch_size = optimizer_data.batch_size
        input_length = optimizer_data.input_length
        effective_input_length = optimizer_data.get_effective_input_length()
        output_length = optimizer_data.output_length
        concurrency = batch_size * self.dp * self.pp

        # calculate TTFT, we get average ttft = sum_for_ttft / concurrency, for sum_for_ttft,
        # we assume one prefill wave can admit
        # `max_prefill_tokens // effective_input_length` requests after prefix-cache reduction.
        # This does not change the user-visible search batch_size. It only changes how many
        # requests fit into one prefill scheduling wave under the prefill token budget.
        # For example, if we have 12 requests, max_prefill_tokens is 8192, and
        # effective_input_length is 2048, then prefill_batch_size is 4.
        # And 12 requests are processed in 3 waves.
        # so sum_for_ttft = (prefill_latency * prefill_batch_size(4 in this case) )) *
        #                       (1 + (calc_nums_for_ttft(3 in this case) )) * (calc_nums_for_ttft) / 2
        # ttft = sum_for_ttft / concurrency (12 in this case)
        # LEFT: the number of tokens that cannot be calculated in one prefill batch
        prefill_batch_size = max_prefill_tokens // effective_input_length
        calc_nums_for_ttft = concurrency // prefill_batch_size
        left_calc_num = concurrency % prefill_batch_size

        prefill_latency, prefill_memory_left_gb, prefill_breakdowns = (
            self._get_or_compute_latency(
                prefill_batch_size, optimizer_data, is_decode=False
            )
        )
        left_latency = 0
        if left_calc_num != 0:
            left_latency, _, _ = self._get_or_compute_latency(
                left_calc_num, optimizer_data, is_decode=False
            )

        left_batch_time = (
            calc_nums_for_ttft * prefill_latency + left_latency
        ) * left_calc_num
        sum_for_ttft = (prefill_batch_size * prefill_latency) * (
            1 + calc_nums_for_ttft
        ) * calc_nums_for_ttft / 2 + left_batch_time
        ttft = sum_for_ttft / concurrency

        # calculate TPOT
        decode_latency, decode_memory_left_gb, decode_breakdowns = (
            self._get_or_compute_latency(batch_size, optimizer_data, is_decode=True)
        )
        # LEFT: we don't consider the bubble time
        tpot = (ttft + decode_latency * output_length) / output_length
        # calculate output throughput: we assume e2e latency is ttft + tpot * output_length
        output_throughput = (
            1000 * (output_length * concurrency) / (ttft + tpot * output_length)
        )

        memory_left = min(prefill_memory_left_gb, decode_memory_left_gb)
        token_s_device = output_throughput / self.dp / self.pp / self.tp
        parallel = format_parallel_label(
            self.model_runner.model.model_config.parallel_config,
            self.is_moe_model,
        )

        logger.info(
            "Prefill Latency: %.4f ms, "
            "Decode Latency: %.4f ms, "
            "TTFT: %.4f ms, TPOT: %.4f ms, "
            "Output Throughput: %.2f token/s, "
            "Concurrency: %d, "
            "parallel: %s, "
            "Memory Left: %.2f GB",
            prefill_latency,
            decode_latency,
            ttft,
            tpot,
            output_throughput,
            concurrency,
            parallel,
            memory_left,
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
                    concurrency,
                    ttft,
                    tpot,
                    output_throughput,
                    token_s_device,
                    parallel,
                    batch_size,
                    prefill_breakdowns,
                    decode_breakdowns,
                ]
            ],
        ).round(3)
        summary.set_summary_df(result_df)
        summary.set_early_stop_flag(memory_left, tpot, ttft)

        return summary

    def _get_or_compute_latency(
        self, batch_size: int, optimizer_data: OptimizerData, is_decode=False
    ):
        """
        Unified method for computing prefill or decode latency with caching.

        Args:
            batch_size: The batch size for processing
            optimizer_data: OptimizerData
            is_decode: Whether this is a decode operation (affects latency calculation)

        Returns:
            Tuple of (latency_ms, memory_left_gb, breakdowns)
        """
        # Select appropriate cache based on operation type
        cache = self._decode_cache if is_decode else self._prefill_cache
        cache_key = batch_size
        model_concurrency = batch_size * self.dp * self.pp if is_decode else batch_size

        # Check if result already exists in cache
        batch_flag = cache.get(cache_key)

        if batch_flag:
            (latency, memory_left_gb, breakdowns) = cache[cache_key]
        else:
            # Compute result
            batch_result = self._get_forward_info(
                model_concurrency, optimizer_data, is_decode
            )

            # Convert execution time to milliseconds
            latency = batch_result.execution_time_s.get("analytic") * 1000
            memory_left_gb = batch_result.device_memory_available_gb
            breakdowns = format_breakdowns(batch_result.breakdowns)

            # Apply decode-specific adjustments
            if is_decode:
                average_tokens = (
                    sum(
                        optimizer_data.mtp_acceptance_rate[
                            : optimizer_data.num_mtp_tokens
                        ]
                    )
                    + 1
                )
                # average_tokens is always greater than 0
                latency /= average_tokens

            # Cache result
            if memory_left_gb > 0:
                cache[batch_size] = (latency, memory_left_gb, breakdowns)

        return latency, memory_left_gb, breakdowns
