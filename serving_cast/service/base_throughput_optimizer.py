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

from tensor_cast.core.input_generator import (
    generate_inputs,
    generate_inputs_varlen,
    RequestInfo,
)
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from .latency_table import ForwardLatencyRecord, ForwardShapeKey
from .optimizer_summary import OptimizerSummary
from .utils import AGG_COLUMNS, build_memory_info, format_breakdowns, MAX_ITER_NUMS, OptimizerData


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
        self.dp: int = 1
        self.tp: int = 1
        self.pp: int = 1
        self.ep: int = 1
        self.moe_tp: int = 1
        self.moe_dp: int = 1
        self.is_moe_model: bool = False
        self._forward_record_cache: dict[ForwardShapeKey, ForwardLatencyRecord] = {}

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
        last_valid_summary = None

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
                last_valid_summary = summary

        if result:
            result_df = pd.concat(result, axis=0, ignore_index=True)

        sorted_df = result_df.sort_values(by=["token/s"], ascending=[True]).round(3)

        ret_summary = OptimizerSummary(optimizer_data)
        ret_summary.set_summary_df(sorted_df)
        if last_valid_summary is not None:
            memory_info = last_valid_summary.get_memory_info()
            if memory_info:
                ret_summary.set_memory_info(memory_info)

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

    def _make_forward_shape_key(
        self,
        concurrency: int,
        optimizer_data: OptimizerData,
        is_decode: bool,
        *,
        query_len: int = None,
        seq_len: int = None,
    ) -> ForwardShapeKey:
        query_len, seq_len = self._resolve_forward_shape(
            optimizer_data,
            is_decode,
            query_len=query_len,
            seq_len=seq_len,
        )

        return ForwardShapeKey(
            is_decode=is_decode,
            model_concurrency=concurrency,
            query_len=query_len,
            seq_len=seq_len,
            image_batch_size=self._resolve_image_batch_size(optimizer_data),
            image_height=optimizer_data.image_height,
            image_width=optimizer_data.image_width,
        )

    def _compute_forward_latency_record(
        self,
        key: ForwardShapeKey,
        optimizer_data: OptimizerData,
    ) -> ForwardLatencyRecord:
        cached_record = self._get_cached_forward_latency_record(key)
        if cached_record is not None:
            return cached_record

        batch_result = self._get_forward_info(
            key.model_concurrency,
            optimizer_data,
            key.is_decode,
            query_len=key.query_len,
            seq_len=key.seq_len,
        )

        record = ForwardLatencyRecord(
            latency_ms=self._select_latency_s(batch_result.execution_time_s) * 1000,
            memory_left_gb=batch_result.device_memory_available_gb,
            breakdowns=format_breakdowns(batch_result.breakdowns),
            memory_info=build_memory_info(batch_result),
            raw_breakdowns=batch_result.breakdowns,
        )
        self._cache_forward_latency_record(key, record)
        return record

    @staticmethod
    def _select_latency_s(execution_time_s: dict) -> float:
        """Prefer the empirical (profiling) latency when present, else analytic.

        Uses an explicit ``is not None`` check so a measured ``0.0`` is not
        treated as a missing value (which ``or`` would do).
        """
        empirical = execution_time_s.get("empirical")
        return empirical if empirical is not None else execution_time_s.get("analytic")

    def _get_cached_forward_latency_record(self, key: ForwardShapeKey) -> ForwardLatencyRecord | None:
        return self._forward_record_cache.get(key)

    def _cache_forward_latency_record(self, key: ForwardShapeKey, record: ForwardLatencyRecord) -> None:
        self._forward_record_cache[key] = record

    def _get_forward_latency_ms(
        self,
        key: ForwardShapeKey,
        record: ForwardLatencyRecord,
        optimizer_data: OptimizerData,
    ) -> float:
        if not key.is_decode:
            return record.latency_ms
        num_mtp_tokens = optimizer_data.num_mtp_tokens or 0
        mtp_acceptance_rate = optimizer_data.mtp_acceptance_rate or []
        average_tokens = sum(mtp_acceptance_rate[:num_mtp_tokens]) + 1
        return record.latency_ms / average_tokens

    def _get_or_compute_latency(
        self,
        batch_size: int,
        optimizer_data: OptimizerData,
        is_decode=False,
        *,
        query_len: int = None,
        seq_len: int = None,
        concurrency_is_model: bool = False,
    ):
        """
        Unified method for computing prefill or decode latency with caching.

        Args:
            batch_size: The batch size for processing.
            optimizer_data: OptimizerData.
            is_decode: Whether this is a decode operation.

        Returns:
            Tuple of (latency_ms, memory_left_gb, breakdowns, memory_info).

        Optional query_len/seq_len override the default request shape for chunked prefill.
        When concurrency_is_model is true, batch_size is already model-level concurrency
        and should not be multiplied by DP/PP.
        """
        model_concurrency = (
            batch_size if concurrency_is_model else batch_size * self.dp * self.pp if is_decode else batch_size
        )
        query_len, seq_len = self._resolve_forward_shape(
            optimizer_data,
            is_decode,
            query_len=query_len,
            seq_len=seq_len,
        )

        key = self._make_forward_shape_key(
            model_concurrency,
            optimizer_data,
            is_decode,
            query_len=query_len,
            seq_len=seq_len,
        )
        record = self._compute_forward_latency_record(key, optimizer_data)
        latency = self._get_forward_latency_ms(key, record, optimizer_data)
        memory_left_gb = record.memory_left_gb
        breakdowns = record.breakdowns

        return latency, memory_left_gb, breakdowns, record.memory_info

    def _get_forward_info(
        self,
        concurrency: int,
        optimizer_data: OptimizerData,
        is_decode: bool,
        *,
        query_len: int = None,
        seq_len: int = None,
    ) -> ModelRunnerMetrics:
        query_len, seq_len = self._resolve_forward_shape(
            optimizer_data,
            is_decode,
            query_len=query_len,
            seq_len=seq_len,
        )

        requests = [
            RequestInfo(
                query_len=query_len,
                seq_len=seq_len,
                image_batch_size=self._resolve_image_batch_size(optimizer_data),
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

    @staticmethod
    def _resolve_image_batch_size(optimizer_data: OptimizerData) -> int | None:
        if optimizer_data.image_height is None:
            return None
        if optimizer_data.image_batch_size is not None:
            return optimizer_data.image_batch_size
        return optimizer_data.batch_size

    def _resolve_forward_shape(
        self,
        optimizer_data: OptimizerData,
        is_decode: bool,
        *,
        query_len: int = None,
        seq_len: int = None,
    ) -> tuple[int, int]:
        """Resolve the RequestInfo shape, allowing chunked prefill callers to override it.

        Without overrides, prefill uses the effective input length after prefix-cache reduction,
        while decode keeps the original prompt length and only computes the next decode/MTP tokens.
        Chunked prefill passes explicit query_len/seq_len so each chunk can be modeled with its
        own newly computed token count and accumulated context length.

        Prefix-cache hit rate is intentionally represented through the resolved shape instead of
        being added to ForwardShapeKey separately: prefill changes query_len/seq_len or the chunk
        plan, while decode continues to use the original prompt length.
        """
        if is_decode:
            # Decode computes one normal token plus any MTP speculative tokens.
            resolved_query_len = query_len or self.num_mtp_tokens + 1
            resolved_seq_len = seq_len or (
                optimizer_data.output_length // 2 + optimizer_data.input_length + resolved_query_len
            )
        else:
            # Full prefill defaults to the effective prompt; chunked prefill provides explicit shapes.
            effective_input_length = optimizer_data.get_effective_input_length()
            resolved_query_len = query_len or effective_input_length
            resolved_seq_len = seq_len or resolved_query_len

        return resolved_query_len, resolved_seq_len

    def _get_batched_forward_info(
        self, concurrency: int, optimizer_data: OptimizerData
    ) -> tuple[ModelRunnerMetrics, list[dict]]:
        dp_size = self.model_runner.model.model_config.parallel_config.data_parallel_size
        concurrency = (concurrency + dp_size - 1) // dp_size
        composition_rows = optimizer_data.build_concurrency_samples(concurrency)

        requests = []
        for row in composition_rows:
            # repeat samples for same input length
            for _ in range(row["samples"]):
                requests.append(
                    RequestInfo(
                        query_len=row["query_len"],
                        seq_len=row["query_len"],
                        is_decode=False,
                        num_input_tokens=row["num_input_tokens"],
                        num_output_tokens=optimizer_data.output_length,
                    )
                )

        metrics = self.model_runner.run_inference(requests, generate_inputs_func=generate_inputs_varlen)

        return metrics, composition_rows
