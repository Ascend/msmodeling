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
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from itertools import groupby
from typing import Optional

import pandas as pd

from tensor_cast.core.input_generator import (
    generate_inputs,
    generate_inputs_varlen,
    RequestInfo,
)
from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.pipeline_parallel import PipelineProfile, PipelineStageProfile
from .latency_table import ForwardLatencyRecord, ForwardShapeKey
from .optimizer_summary import EARLY_STOP_PREFILL_OOM, OptimizerSummary
from .pipeline_schedule import (
    estimate_forward_pipeline,
    estimate_mixed_pd_stage_memory_across_profiles,
    estimate_pipeline_stage_memory_across_profiles,
    estimate_repeated_pipeline,
    PipelineScheduleEstimate,
    RepeatedPipelineEstimate,
    split_batch_size,
)
from .utils import (
    AGG_COLUMNS,
    HISTORICAL_DEFAULT_MAX_BATCHED_TOKENS,
    BYTES_TO_GB,
    build_memory_info,
    format_breakdowns,
    MAX_ITER_NUMS,
    OptimizerData,
    UnsupportedPPConfigurationError,
    PrefillChunk,
)


logger = logging.getLogger(__name__)

DEFAULT_MAX_BATCH_SIZE = 512


@dataclass(frozen=True)
class _PPWaveEvaluation:
    profiles: tuple[PipelineProfile, ...]
    schedule: PipelineScheduleEstimate
    repeated: RepeatedPipelineEstimate | None
    num_microbatches: int
    memory_left_gb: float
    memory_exceeded: bool
    bottleneck_stage_id: int


class BaseThroughputOptimizer(ABC):
    """
    Abstract base class for throughput optimization strategies.
    This class provides a framework for optimizing model inference throughput by
    finding the optimal batch size through binary search. Subclasses may extend
    initialize for strategy-specific lifecycle state and must implement
    get_inference_info.
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
        # PP>1 microbatch PipelineProfile cache, keyed by
        # (is_decode, query_len, seq_len, microbatch_size). Distinct shapes are
        # profiled once through run_inference and expanded to the full wave.
        # Per-instance (cleared on initialize): each parallel candidate gets an
        # isolated strategy instance, so candidates never share this cache.
        self._pp_profile_cache: dict[tuple, PipelineProfile] = {}
        self._last_run_early_stop_reason: str | None = None

    def initialize(self, model_runner: ModelRunner):
        """Initialize state shared by all throughput optimization strategies."""
        self.model_runner = model_runner
        model_config = model_runner.model.model_config
        parallel_config = model_config.parallel_config
        self.num_mtp_tokens = model_config.mtp_config.num_mtp_layers if model_config.mtp_config is not None else 0
        self.dp = parallel_config.data_parallel_size
        self.tp = parallel_config.tensor_parallel_size
        self.pp = parallel_config.pipeline_parallel_size
        self.ep = parallel_config.expert_parallel_size
        self.moe_tp = parallel_config.moe_tensor_parallel_size
        self.moe_dp = parallel_config.moe_data_parallel_size
        self.is_moe_model = model_config.moe_config is not None
        self._forward_record_cache.clear()
        self._pp_profile_cache.clear()

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

    def run(self, optimizer_data: OptimizerData, batch_range: list[int]) -> OptimizerSummary | None:
        if optimizer_data.max_batched_tokens is None:
            return self._run_with_auto_max_batched_tokens(optimizer_data, batch_range)
        return self._run_once(optimizer_data, batch_range)

    def _get_global_batched_token_limit(self, optimizer_data: OptimizerData) -> int:
        """Return the total scheduler budget across all data-parallel replicas.

        ``max_batched_tokens`` is a per-DP-replica serving-engine limit.  The
        optimizer schedules global request concurrency and TensorCast divides
        that concurrency across data-parallel replicas before modeling a
        forward pass, so scheduling must use the sum of those identical per-DP
        budgets.
        """
        max_batched_tokens = optimizer_data.max_batched_tokens
        if max_batched_tokens is None or max_batched_tokens <= 0:
            raise ValueError(f"max_batched_tokens must be a positive integer, got {max_batched_tokens!r}.")
        return max_batched_tokens * self.dp

    def _run_with_auto_max_batched_tokens(
        self,
        optimizer_data: OptimizerData,
        batch_range: list[int],
    ) -> OptimizerSummary | None:
        original_max_batched_tokens = optimizer_data.max_batched_tokens
        try:
            candidates = optimizer_data.get_auto_max_batched_tokens_candidates()
            if not candidates:
                logger.warning(
                    "Cannot determine auto max_batched_tokens because input_length and length_distribution are both "
                    "unset; falling back to the historical default of %d.",
                    HISTORICAL_DEFAULT_MAX_BATCHED_TOKENS,
                )
                optimizer_data.max_batched_tokens = HISTORICAL_DEFAULT_MAX_BATCHED_TOKENS
                return self._run_once(optimizer_data, batch_range)

            for candidate in candidates:
                optimizer_data.max_batched_tokens = candidate
                logger.info("Auto max_batched_tokens candidate: %d.", candidate)
                summary = self._run_once(optimizer_data, batch_range)
                if summary is not None or self._last_run_early_stop_reason != EARLY_STOP_PREFILL_OOM:
                    return summary

                logger.info(
                    "Prefill OOM with auto max_batched_tokens=%d; retrying with a smaller token budget.",
                    candidate,
                )

            logger.warning("Prefill still OOM with the minimum auto max_batched_tokens; no valid result found.")
            return None
        finally:
            optimizer_data.max_batched_tokens = original_max_batched_tokens

    def _run_once(self, optimizer_data: OptimizerData, batch_range: list[int]) -> OptimizerSummary | None:
        self._last_run_early_stop_reason = None
        left, right = 1, DEFAULT_MAX_BATCH_SIZE
        result = []
        result_df = pd.DataFrame(columns=AGG_COLUMNS)
        last_valid_summary = None
        pp_mixed_pd_overlap_approx = False

        if batch_range:
            if len(batch_range) == 2:
                left, right = batch_range
            elif len(batch_range) == 1:
                right = batch_range[0]

        # early_stop
        optimizer_data.batch_size = left
        summary = self.get_inference_info(optimizer_data)
        if summary.check_early_stop_flag():
            self._last_run_early_stop_reason = summary.get_early_stop_reason()
            return None

        if not batch_range:
            # PP>1 must not use the two-point linear TTFT/TPOT extrapolation
            # across microbatch-count discontinuities (plan Task-4 Step 7):
            # schedule makespan/TPOT is not linear in batch size. Force the
            # plain exponential search so every returned boundary is evaluated
            # exactly via get_inference_info.
            strategy = optimizer_data.concurrency_search_strategy
            if strategy == "linear_exponential" and self.pp > 1:
                logger.info(
                    "PP>1 disables linear SLO extrapolation; using exponential concurrency search "
                    "instead of linear_exponential for this candidate."
                )
                strategy = "exponential"
            if strategy == "exponential":
                left, right = self._exponential_search(optimizer_data, left, right, summary)
            elif strategy == "linear_exponential":
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
                # Propagate the PP aggregation overlap-approx flag: if any
                # valid per-batch summary was approximated, the aggregated
                # result is too. (Only aggregation PP>1 sets this flag.)
                if summary.get_pp_mixed_pd_overlap_approx():
                    pp_mixed_pd_overlap_approx = True
                last_valid_summary = summary

        if result:
            result_df = pd.concat(result, axis=0, ignore_index=True)

        sorted_df = result_df.sort_values(by=["token/s"], ascending=[True]).round(3)

        ret_summary = OptimizerSummary(optimizer_data)
        ret_summary.set_summary_df(sorted_df)
        if pp_mixed_pd_overlap_approx:
            ret_summary.set_pp_mixed_pd_overlap_approx(True)
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

    def _estimate_by_latency(
        self,
        bs_left,
        bs_right,
        lat_left,
        lat_right,
        lat_limit,
        relax_factor,
        estimated_right,
    ):
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
            bs_left,
            bs_right,
            tpot_left,
            tpot_right,
            optimizer_data.tpot_limits,
            slo_relax_factor,
            estimated_right,
        )

        ttft_left = total_info_left.get("ttft")
        ttft_right = total_info_right.get("ttft")
        estimated_right = self._estimate_by_latency(
            bs_left,
            bs_right,
            ttft_left,
            ttft_right,
            optimizer_data.ttft_limits,
            slo_relax_factor,
            estimated_right,
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
        if optimizer_data.concurrency_search_strategy == "linear_exponential":
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

    def get_compile_calibration_probe(
        self,
        optimizer_data: OptimizerData,
        batch_range: list[int],
        *,
        is_decode: bool,
    ) -> tuple[ForwardShapeKey, RequestInfo]:
        """Return the high-concurrency request used to calibrate compile mode.

        The optimizer's regular search starts with the upper batch candidate
        before binary search.  Reusing that candidate keeps calibration aligned
        with the existing search contract instead of introducing a second
        concurrency heuristic.
        """
        probe_batch_size = max(batch_range) if batch_range else DEFAULT_MAX_BATCH_SIZE
        concurrency = probe_batch_size * self.dp
        key = self._make_forward_shape_key(concurrency, optimizer_data, is_decode)
        request = RequestInfo(
            query_len=key.query_len,
            seq_len=key.seq_len,
            image_batch_size=key.image_batch_size,
            image_height=key.image_height,
            image_width=key.image_width,
            concurrency=key.model_concurrency,
            is_decode=is_decode,
        )
        return key, request

    def cache_compile_calibration_metrics(
        self,
        key: ForwardShapeKey,
        metrics: ModelRunnerMetrics,
    ) -> None:
        """Seed the selected runner's latency cache with its calibration result."""
        self._cache_forward_latency_record(key, self._build_forward_latency_record(metrics))

    def _build_forward_latency_record(self, metrics: ModelRunnerMetrics) -> ForwardLatencyRecord:
        return ForwardLatencyRecord(
            latency_ms=self._select_latency_s(metrics.execution_time_s) * 1000,
            memory_left_gb=metrics.device_memory_available_gb,
            breakdowns=format_breakdowns(metrics.breakdowns),
            memory_info=build_memory_info(metrics),
            raw_breakdowns=metrics.breakdowns,
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

        record = self._build_forward_latency_record(batch_result)
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
        return self._fold_decode_latency_ms(record.latency_ms, optimizer_data)

    @staticmethod
    def _fold_decode_latency_ms(latency_ms: float, optimizer_data: OptimizerData) -> float:
        """Apply speculative decode fold (mutually exclusive; never combine).

        - DSpark / Dflash: unified clamp to n (= block-1), ``latency / (accept + 1)``.
        - New MTP (``speculative_method == 'mtp'``): ``latency / (clamp(accept, 0, n) + 1)``.
        - Legacy MTP (no speculative_method): ``latency / (sum(rates[:N]) + 1)``.
        """
        # DSpark: unified clamp to n (block-1)
        dspark_block_size = optimizer_data.dspark_block_size or 0
        if dspark_block_size >= 2:
            accept = optimizer_data.dspark_acceptance_length
            if accept is None:
                accept = 5.0
            n = dspark_block_size - 1
            accept = min(max(float(accept), 0.0), float(n))
            return latency_ms / (accept + 1.0)
        # DFlash: clamp to n (block-1)
        dflash_block_size = optimizer_data.dflash_block_size or 0
        if dflash_block_size >= 2:
            accept = optimizer_data.dflash_acceptance_length
            if accept is None:
                accept = 5.0
            n = dflash_block_size - 1
            accept = min(max(float(accept), 0.0), float(n))
            return latency_ms / (accept + 1.0)
        # New MTP entry (--speculative-method mtp): accept+1 fold
        if getattr(optimizer_data, "speculative_method", None) == "mtp":
            n = optimizer_data.num_mtp_tokens or 0
            if n <= 0:
                return latency_ms
            accept = optimizer_data.acceptance_length
            if accept is None:
                accept = 5.0
            accept = min(max(float(accept), 0.0), float(n))
            return latency_ms / (accept + 1.0)
        # Legacy MTP path (C1: sum(rates[:N])+1, must not change)
        num_mtp_tokens = optimizer_data.num_mtp_tokens or 0
        mtp_acceptance_rate = optimizer_data.mtp_acceptance_rate or []
        average_tokens = sum(mtp_acceptance_rate[:num_mtp_tokens]) + 1
        return latency_ms / average_tokens

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
        model_concurrency = batch_size if concurrency_is_model else batch_size * self.dp if is_decode else batch_size
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
            # Decode: DSpark/Dflash use full block as query_len; otherwise one token + MTP.
            dspark_block = optimizer_data.dspark_block_size or 0
            dflash_block = optimizer_data.dflash_block_size or 0
            if dspark_block >= 2:
                resolved_query_len = query_len or dspark_block
            elif dflash_block >= 2:
                resolved_query_len = query_len or dflash_block
            else:
                # Existing MTP / baseline path — keep expression unchanged for G1.
                resolved_query_len = query_len or self.num_mtp_tokens + 1
            resolved_seq_len = seq_len or (
                optimizer_data.output_length // 2 + optimizer_data.get_decode_context_length() + resolved_query_len
            )
        else:
            # Full prefill defaults to the effective prompt; chunked prefill provides explicit shapes.
            effective_input_length = optimizer_data.get_effective_input_length()
            resolved_query_len = query_len or effective_input_length
            resolved_seq_len = seq_len or resolved_query_len

        return resolved_query_len, resolved_seq_len

    def _get_batched_forward_info(
        self,
        concurrency: int,
        optimizer_data: OptimizerData,
        chunk_plan: Optional[list[PrefillChunk]] = None,
    ) -> tuple[list[tuple[ModelRunnerMetrics, int]], list[dict]]:
        dp_size = self.model_runner.model.model_config.parallel_config.data_parallel_size
        concurrency = (concurrency + dp_size - 1) // dp_size
        composition_rows = optimizer_data.build_concurrency_samples(concurrency)

        if chunk_plan is not None:
            chunk_indices = [chunk.index for chunk in chunk_plan]
            if chunk_indices:
                expected_indices = list(range(max(chunk_indices) + 1))
                actual_indices = list(dict.fromkeys(chunk_indices))
                if chunk_indices != sorted(chunk_indices) or actual_indices != expected_indices:
                    raise ValueError(f"chunk_plan indices must be contiguous and non-decreasing, got {chunk_indices}")

            chunk_results = []
            for _, chunks in groupby(chunk_plan, key=lambda chunk: chunk.index):
                chunks = list(chunks)
                requests = [
                    RequestInfo(
                        query_len=chunk.query_len,
                        seq_len=chunk.seq_len,
                        is_decode=False,
                        num_output_tokens=optimizer_data.output_length,
                    )
                    for chunk in chunks
                ]
                metrics = self.model_runner.run_inference(
                    requests,
                    generate_inputs_func=generate_inputs_varlen,
                )
                completed_requests = sum(chunk.is_last_chunk for chunk in chunks)
                chunk_results.append((metrics, completed_requests))
                if metrics.device_memory_available_gb < 0:
                    break
            return chunk_results, composition_rows

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
        completed_requests = sum(row["samples"] for row in composition_rows)

        return [(metrics, completed_requests)], composition_rows

    # ------------------------------------------------------------------
    # Pipeline-parallel (PP>1) evaluation helpers.
    #
    # PP=1 never calls any of these: the subclasses branch on ``self.pp > 1``.
    # The scheduler consumes one PipelineProfile per microbatch shape; each
    # profile comes from a single ``run_inference`` call with
    # ``concurrency == microbatch_size``. Distinct shapes are profiled once and
    # cached per strategy instance (candidates are isolated, so no cross-
    # candidate sharing).
    # ------------------------------------------------------------------

    def _validate_pp_prefill_wave(
        self,
        optimizer_data: OptimizerData,
        chunk_plan: list,
        *,
        phase_label: str = "prefill",
    ) -> str | None:
        """Validate the PP prefill contract.

        Returns ``EARLY_STOP_PREFILL_OOM`` when the *current* batch exceeds the
        token budget but smaller batches may still be valid — the binary
        search will shrink its upper bound instead of discarding the entire
        candidate.  Raises ``UnsupportedPPConfigurationError`` only for
        candidate-level incompatibilities that no batch size can fix
        (variable-length input, or prompt too long for even batch=1).

        Chunked prefill (``len(chunk_plan) > 1``) is supported: each chunk
        becomes a separate microbatch in the pipeline schedule, mirroring
        vLLM's 1F1B scheduling where chunks pipeline across stages.
        """
        effective_input_length = optimizer_data.get_effective_input_length()
        max_batched_tokens = optimizer_data.max_batched_tokens
        if optimizer_data.length_distribution is not None:
            raise UnsupportedPPConfigurationError(
                f"PP>1 {phase_label} does not support variable-length input "
                "distribution; use a fixed input_length or disable PP."
            )

        if len(chunk_plan) > 1:
            # Chunked prefill: each chunk already respects max_batched_tokens
            # (the chunk plan splits by it).  The pipeline schedule handles
            # multiple microbatches, so no batch-too-large check is needed.
            return None

        prefill_query_len = chunk_plan[0].query_len if chunk_plan else effective_input_length
        batch_tokens = optimizer_data.batch_size * prefill_query_len
        if max_batched_tokens is None:
            # Unbounded prefill (auto mode, no explicit --max-batched-tokens):
            # the whole prompt is one forward with no per-wave token budget, so
            # the batch-budget gate does not apply. PP>1 schedules it as a single
            # microbatch through the pipeline.
            return None
        if batch_tokens > max_batched_tokens:
            max_batch_by_tokens = max_batched_tokens // prefill_query_len
            if max_batch_by_tokens < 1:
                raise UnsupportedPPConfigurationError(
                    f"PP>1 {phase_label} prompt length {prefill_query_len} exceeds "
                    f"max_batched_tokens {max_batched_tokens} per DP replica; even "
                    "batch=1 cannot fit. Increase --max-batched-tokens or disable PP."
                )
            logger.info(
                "PP>1 %s batch_size %d * query_len %d = %d exceeds max_batched_tokens "
                "%d; shrinking search upper bound to %d.",
                phase_label,
                optimizer_data.batch_size,
                prefill_query_len,
                batch_tokens,
                max_batched_tokens,
                max_batch_by_tokens,
            )
            return EARLY_STOP_PREFILL_OOM
        return None

    def _pp_microbatch_size(self) -> int:
        """The candidate's PP microbatch size (1 for PP=1, but this is PP>1 only)."""
        return int(getattr(self.model_runner.user_input, "microbatch_size", 1) or 1)

    def _pp_perf_model_name(self) -> str:
        """Performance model name used to select per-stage compute/comm times."""
        runner = self.model_runner
        perf_models = getattr(runner, "perf_models", None) or []
        if perf_models:
            return perf_models[0].name
        return "analytic"

    def _profile_pp_microbatch(
        self,
        microbatch_size: int,
        optimizer_data: OptimizerData,
        is_decode: bool,
        *,
        query_len: int = None,
        seq_len: int = None,
    ) -> PipelineProfile:
        """Profile one microbatch shape and return its PipelineProfile.

        Caches per (is_decode, query_len, seq_len, microbatch_size). Raises if
        the run did not produce a pipeline_profile (PP>1 must always produce
        one — never silently fall back to a mismatched shape).

        ``concurrency`` passed to ``run_inference`` is ``microbatch_size * dp``:
        ``generate_inputs`` divides concurrency by DP to get the per-rank
        batch, so this yields a stage-local batch of exactly ``microbatch_size``
        requests per DP rank (not ``microbatch_size / dp``, which would
        under-size latency, payload, KV cache and peak memory by DP×).
        """
        resolved_query_len, resolved_seq_len = self._resolve_forward_shape(
            optimizer_data, is_decode, query_len=query_len, seq_len=seq_len
        )
        cache_key = (is_decode, resolved_query_len, resolved_seq_len, microbatch_size)
        cached = self._pp_profile_cache.get(cache_key)
        if cached is not None:
            return cached

        model_concurrency = microbatch_size * self.dp
        metrics = self._get_forward_info(
            model_concurrency,
            optimizer_data,
            is_decode,
            query_len=resolved_query_len,
            seq_len=resolved_seq_len,
        )
        profile = getattr(metrics, "pipeline_profile", None)
        if profile is None:
            raise UnsupportedPPConfigurationError(
                "PP>1 evaluation expected a pipeline_profile from run_inference but got None; "
                "the model is not a PipelineModel or profiling did not produce a profile."
            )
        self._pp_profile_cache[cache_key] = profile
        return profile

    @staticmethod
    def _consolidate_prefill_profiles(
        profiles: tuple[PipelineProfile, ...],
    ) -> PipelineProfile:
        """Consolidate per-chunk prefill profiles into one representative profile.

        Each stage keeps the per-stage conservative upper bound of every memory
        field across chunks, so a chunk whose peak lands on a different stage is
        not dropped. Structural fields (stage_id, layer range, per-model
        timings) are identical across chunks for the same model, so only the
        memory fields (weight/activation/kv/indexer caches/runtime peak/payload)
        are merged by max. Picking a single chunk by its global max peak can
        miss a higher per-stage peak on another chunk and under-estimate the
        mixed P/D memory, falsely passing an OOM candidate.
        """
        base = profiles[0]
        consolidated_stages: list[PipelineStageProfile] = []
        for stage_id, base_stage in enumerate(base.stages):
            candidates = [p.stages[stage_id] for p in profiles]
            consolidated_stages.append(
                replace(
                    base_stage,
                    weight_bytes=max(int(s.weight_bytes) for s in candidates),
                    activation_bytes=max(int(s.activation_bytes) for s in candidates),
                    kv_cache_bytes=max(int(s.kv_cache_bytes) for s in candidates),
                    kv_cache_per_token_bytes=max(float(s.kv_cache_per_token_bytes) for s in candidates),
                    indexer_cache_bytes=max(int(s.indexer_cache_bytes) for s in candidates),
                    indexer_cache_per_token_bytes=max(float(s.indexer_cache_per_token_bytes) for s in candidates),
                    runtime_peak_bytes=max(int(s.runtime_peak_bytes) for s in candidates),
                    outgoing_payload_bytes=max(int(s.outgoing_payload_bytes) for s in candidates),
                )
            )
        return replace(base, stages=tuple(consolidated_stages))

    def _evaluate_pp_wave(
        self,
        batch_size: int,
        optimizer_data: OptimizerData,
        *,
        is_decode: bool,
        repeat: bool = False,
        query_len: int = None,
        seq_len: int = None,
        resident_policy: str = "full",
        mixed_prefill: _PPWaveEvaluation | None = None,
        publish_schedule: bool = True,
        chunk_shapes: list[tuple[int, int]] | None = None,
    ) -> _PPWaveEvaluation:
        """Profile, schedule, and memory-gate one PP microbatch wave.

        ``resident_policy`` owns the cache lifecycle: ``full`` keeps the entire
        wave resident, while ``inflight`` delegates the window size to the
        scheduler estimator (``min(pp, num_microbatches)``). When
        ``mixed_prefill`` is supplied, the mixed P/D estimator covers prefill,
        decode, and their overlap in one gate.

        When ``chunk_shapes`` is provided, each ``(query_len, seq_len)`` pair
        becomes a group of microbatches in the pipeline schedule, mirroring
        vLLM's chunked-prefill + PP 1F1B scheduling.  This is used for
        chunked prefill where the effective prompt exceeds
        ``max_batched_tokens`` and is split into multiple chunks.
        """
        microbatch_sizes = split_batch_size(batch_size, self._pp_microbatch_size())
        if chunk_shapes is not None:
            # Chunked prefill: each chunk is a group of microbatches with its
            # own (query_len, seq_len).  The combined sequence is ordered by
            # chunk index then microbatch index within each chunk, so the
            # pipeline schedule models 1F1B chunk pipelining across stages.
            profiles = tuple(
                self._profile_pp_microbatch(
                    mb_size,
                    optimizer_data,
                    is_decode,
                    query_len=c_query_len,
                    seq_len=c_seq_len,
                )
                for c_query_len, c_seq_len in chunk_shapes
                for mb_size in microbatch_sizes
            )
        else:
            profiles = tuple(
                self._profile_pp_microbatch(
                    microbatch_size,
                    optimizer_data,
                    is_decode,
                    query_len=query_len,
                    seq_len=seq_len,
                )
                for microbatch_size in microbatch_sizes
            )
        perf_model_name = self._pp_perf_model_name()
        repeated = estimate_repeated_pipeline(profiles, perf_model_name) if repeat else None
        schedule = repeated.first_wave if repeated is not None else estimate_forward_pipeline(profiles, perf_model_name)
        if publish_schedule:
            optimizer_data.pipeline_schedule_estimate = schedule

        num_microbatches = len(profiles)
        resident_microbatches = num_microbatches if resident_policy == "full" else None
        device_memory_bytes = int(round(self.model_runner.total_device_memory_gb * BYTES_TO_GB))
        reserved_memory_bytes = int(round(float(self.model_runner.user_input.reserved_memory_gb or 0.0) * BYTES_TO_GB))
        if mixed_prefill is None:
            memory_estimate = estimate_pipeline_stage_memory_across_profiles(
                profiles,
                num_microbatches=num_microbatches,
                device_memory_bytes=device_memory_bytes,
                reserved_memory_bytes=reserved_memory_bytes,
                resident_microbatches=resident_microbatches,
                perf_model_name=perf_model_name,
                resident_policy=resident_policy,
            )
        else:
            # When chunked prefill produces more profiles than decode (e.g.
            # multi-chunk prefill + single decode wave), consolidate prefill
            # profiles into one representative profile. Each stage takes the
            # conservative per-stage upper bound of all memory fields across
            # chunks, so a chunk that peaks on a different stage is not lost
            # (which would under-estimate memory and falsely pass an OOM
            # candidate). Structural fields are identical across chunks for
            # the same model, so only the memory fields are merged by max.
            prefill_profiles = mixed_prefill.profiles
            if len(prefill_profiles) not in (1, num_microbatches):
                prefill_profiles = (self._consolidate_prefill_profiles(prefill_profiles),)
            memory_estimate = estimate_mixed_pd_stage_memory_across_profiles(
                prefill_profiles,
                profiles,
                num_microbatches=num_microbatches,
                resident_microbatches=resident_microbatches,
                device_memory_bytes=device_memory_bytes,
                reserved_memory_bytes=reserved_memory_bytes,
                perf_model_name=perf_model_name,
                resident_policy=resident_policy,
            )

        min_remaining_bytes = min(memory_estimate.remaining_bytes_s) if memory_estimate.remaining_bytes_s else 0
        return _PPWaveEvaluation(
            profiles=profiles,
            schedule=schedule,
            repeated=repeated,
            num_microbatches=num_microbatches,
            memory_left_gb=min_remaining_bytes / BYTES_TO_GB,
            memory_exceeded=memory_estimate.exceeds_budget,
            bottleneck_stage_id=memory_estimate.bottleneck_stage_id,
        )
