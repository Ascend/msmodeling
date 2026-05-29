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

from abc import ABC, abstractmethod

from typing import Optional

import pandas as pd

from tensor_cast.core.input_generator import generate_inputs, RequestInfo
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from .optimizer_summary import OptimizerSummary
from .utils import AGG_COLUMNS, MAX_ITER_NUMS, OptimizerData


class BaseThroughputOptimizer(ABC):
    """
    Abstract base class for throughput optimization strategies.
    This class provides a framework for optimizing model inference throughput by
    finding the optimal batch size through binary search. Subclasses must implement
    the initialize and get_inference_info methods to support specific optimization
    strategies.
    Attributes:
        name: Identifier for the optimizer strategy, defaults to "base".
    """

    name = "base"

    def __init__(self) -> None:
        self.model_runner: Optional[ModelRunner] = None
        self.num_mtp_tokens: int = 0

    @abstractmethod
    def initialize(self, model_runner: ModelRunner):
        """
        Initialize the optimizer with a model runner instance.
        Args:
            model_runner: The ModelRunner instance used for model inference.
        Note:
            This method should be implemented to set up any required resources
            or configurations for the optimization process.
        """

    @abstractmethod
    def get_inference_info(self, optimizer_data: OptimizerData) -> OptimizerSummary:
        """
        Execute inference and return optimization summary.
        Args:
            optimizer_data: Contains optimization parameters including batch size,
                input length, output length, etc.
        Returns:
            OptimizerSummary containing inference metrics and results.
        Note:
            This method should be implemented to perform model inference with
            the specified batch size and return performance metrics.
        """

    def run(self, optimizer_data: OptimizerData, batch_range: list[int]) -> OptimizerSummary:
        left, right = 1, 512
        result = []
        result_df = pd.DataFrame(columns=AGG_COLUMNS)

        if batch_range:
            if len(batch_range) == 2:
                left, right = batch_range
            elif len(batch_range) == 1:
                right = batch_range[0]

        # early_stop
        optimizer_data.batch_size = left
        summary = self.get_inference_info(optimizer_data)
        if summary.check_early_stop_flag():
            return None

        if not batch_range:
            if optimizer_data.concurrency_search_strategy == 'exponential':
                left, right = self._exponential_search(optimizer_data, left, right, summary)
            elif optimizer_data.concurrency_search_strategy == 'linear_exponential':
                left, right = self._exponential_search(optimizer_data, left, right, summary, True)

        while left <= right:
            mid = (left + right) // 2
            optimizer_data.batch_size = mid
            summary = self.get_inference_info(optimizer_data)
            if summary.check_early_stop_flag():
                right = mid - 1
            else:
                left = mid + 1
                result.append(summary.get_summary_df())

        if result:
            result_df = pd.concat(result, axis=0, ignore_index=True)

        sorted_df = result_df.sort_values(by=["token/s"], ascending=[True]).round(3)

        ret_summary = OptimizerSummary(optimizer_data)
        ret_summary.set_summary_df(sorted_df)

        return ret_summary

    def _exponential_search(self, optimizer_data, left, right, summary_left, linear_acc_search=False):
        estimated_right = float("inf")

        if linear_acc_search:
            search_info_left = summary_left.get_search_info() or {}
            total_info_left = {"batch_size": left, **search_info_left}

        for _ in range(MAX_ITER_NUMS):
            optimizer_data.batch_size = right
            summary = self.get_inference_info(optimizer_data)
            if linear_acc_search:
                search_info_right = summary.get_search_info() or {}
                total_info_right = {"batch_size": right, **search_info_right}
                estimated_right = self._estimate_right_boundary(total_info_left, total_info_right, optimizer_data)

            if summary.check_early_stop_flag():
                if linear_acc_search:
                    right = min(estimated_right, right)
                break

            if estimated_right <= right * 2:
                right = round(estimated_right)
                break

            left, right = right, right * 2

        return left, right

    def _estimate_by_latency(self, bs_left, bs_right, lat_left, lat_right, lat_limit, relax_factor, estimated_right):
        bs_diff = bs_right - bs_left
        if (
            lat_limit is not None
            and lat_left is not None
            and lat_right is not None
            and lat_right > lat_left
            and bs_diff > 0
        ):
            slope = (lat_right - lat_left) / bs_diff
            max_batch = max(
                1,
                round((bs_left + (lat_limit - lat_left) / slope) * relax_factor) + 1,
            )
            return min(estimated_right, max_batch)

        return estimated_right

    def _estimate_right_boundary(self, total_info_left, total_info_right, optimizer_data):
        """
        Estimate the upper boundary for batch size based on hardware memory
        constraints and SLO (Service Level Objective) latency limits using
        linear extrapolation.

        Args:
            total_info_left: Dictionary containing inference metrics (batch_size,
                tpot, ttft, memory info) from the left (smaller) batch size probe.
            total_info_right: Dictionary containing inference metrics from the
                right (larger) batch size probe.

        Returns:
            int: The estimated maximum safe batch size. Returns _DEFAULT_MAX_BATCH_ESTIMATE
            as a conservative fallback if no valid estimation can be made.

        Note:
            Tradeoff: Using linear extrapolation for fast estimation, assuming
            metrics scale linearly. In real LLM inference, memory and latency
            often scale non-linearly, which risks underestimating the boundary.
            Relaxation factors mitigate this by intentionally magnifying the
            estimate, Sacrificing algorithm time to obtain accurate simulation
            results.
        """
        # Unable to estimate the maximum value of returned batch size.
        # Default is 512 * 2 ** MAX_ITER_NUMS -1
        _DEFAULT_MAX_BATCH_ESTIMATE = 2 ** (MAX_ITER_NUMS + 9) - 1
        estimated_right = float("inf")
        slo_relax_factor = 1.5
        mem_relax_factor = 1.0

        bs_left = total_info_left.get("batch_size")
        bs_right = total_info_right.get("batch_size")

        per_req = total_info_right.get("per_request_memory_gb", 0)
        available = total_info_right.get("device_memory_available_gb", 0)
        if per_req > 0:
            max_batch_by_memory = max(
                1,
                round((bs_right + available / per_req) * mem_relax_factor) + 1,
            )
            estimated_right = min(estimated_right, max_batch_by_memory)

        tpot_left = total_info_left.get("tpot")
        tpot_right = total_info_right.get("tpot")
        estimated_right = self._estimate_by_latency(
            bs_left, bs_right, tpot_left, tpot_right, optimizer_data.tpot_limits, slo_relax_factor, estimated_right
        )

        ttft_left = total_info_left.get("ttft")
        ttft_right = total_info_right.get("ttft")
        estimated_right = self._estimate_by_latency(
            bs_left, bs_right, ttft_left, ttft_right, optimizer_data.ttft_limits, slo_relax_factor, estimated_right
        )

        if estimated_right == float("inf"):
            estimated_right = _DEFAULT_MAX_BATCH_ESTIMATE

        return estimated_right

    def _compute_per_request_memory_gb(
        self,
        total_device_memory_gb,
        model_weight_size_gb,
        reserved_memory_gb,
        memory_left_gb,
        batch_size,
    ):
        if batch_size <= 0:
            return 0
        return (total_device_memory_gb - model_weight_size_gb - reserved_memory_gb - memory_left_gb) / batch_size

    def _maybe_set_search_info(self, optimizer_data, memory_left_gb, batch_size, ttft, tpot, summary):
        if optimizer_data.concurrency_search_strategy == 'linear_exponential':
            per_request_memory_gb = self._compute_per_request_memory_gb(
                self.model_runner.total_device_memory_gb,
                self.model_runner.model_weight_size_gb,
                self.model_runner.user_input.reserved_memory_gb,
                memory_left_gb,
                batch_size,
            )

            summary.set_search_info(
                {
                    "per_request_memory_gb": per_request_memory_gb,
                    "device_memory_available_gb": memory_left_gb,
                    "ttft": ttft,
                    "tpot": tpot,
                }
            )

    def _get_forward_info(
        self,
        concurrency: int,
        optimizer_data: OptimizerData,
        is_decode: bool,
    ) -> ModelRunnerMetrics:
        if is_decode:
            query_len = self.num_mtp_tokens + 1
            seq_len = optimizer_data.output_length // 2 + optimizer_data.input_length + query_len
        else:
            seq_len = query_len = optimizer_data.get_effective_input_length()

        # avoid print duplicate image input log
        _image_batch_size = None
        if optimizer_data.image_height is not None:
            _image_batch_size = (
                optimizer_data.image_batch_size
                if optimizer_data.image_batch_size is not None
                else optimizer_data.batch_size
            )
        requests = [
            RequestInfo(
                query_len=query_len,
                seq_len=seq_len,
                image_batch_size=_image_batch_size,
                image_height=optimizer_data.image_height,
                image_width=optimizer_data.image_width,
                concurrency=concurrency,
                is_decode=is_decode,
            )
        ]

        runner = self.model_runner
        assert runner is not None, "initialize() must set model_runner"
        metrics = runner.run_inference(requests, generate_inputs_func=generate_inputs)

        return metrics
