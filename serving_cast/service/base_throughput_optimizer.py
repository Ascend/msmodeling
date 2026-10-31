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
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import groupby
from typing import Optional

import pandas as pd

from tensor_cast.core.input_generator import (
    generate_inputs,
    generate_inputs_varlen,
    RequestInfo,
)
from tensor_cast.core.kv_offload import HostKVCapacityExceeded, fold_offload_transfer_time_ms
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

# Warn once globally when image_batch_size falls back to the default: search
# loops resolve every candidate and would otherwise flood the log.
_IMAGE_BATCH_DEFAULT_WARNED = False

DEFAULT_MAX_BATCH_SIZE = 512


def _host_capacity_rejected_metrics(exc: HostKVCapacityExceeded) -> ModelRunnerMetrics:
    """Synthesize an OOM-flagged metrics row for a host-capacity-rejected candidate.

    ``device_memory_available_gb < 0`` routes the candidate into the existing OOM
    early-stop path (the search shrinks its upper bound and keeps feasible rows);
    the offload fields carry the rejected pool size and phase for diagnostics.
    """
    return ModelRunnerMetrics(
        total_device_memory_gb=0.0,
        model_weight_size_gb=0.0,
        peak_memory_usage_gb=0.0,
        kv_cache_size_gb=0.0,
        kv_cache_per_token_gb=0.0,
        model_activation_size_gb=0.0,
        reserved_memory_gb=0.0,
        execution_time_s={},
        tps_per_model={},
        run_time_s=0.0,
        batch_size=0,
        device_memory_available_gb=-1.0,
        host_kv_pool_gb=exc.host_kv_pool_gb,
        offload_mode=exc.phase,
    )


@dataclass(frozen=True)
class _PPWaveEvaluation:
    profiles: tuple[PipelineProfile, ...]
    schedule: PipelineScheduleEstimate
    repeated: RepeatedPipelineEstimate | None
    num_microbatches: int
    memory_left_gb: float
    memory_exceeded: bool
    bottleneck_stage_id: int
    profiling_source_times_s: dict[str, float]
    profiling_miss_reasons: dict[str, int]
    # Request-level prefill TTFT in seconds: the request-weighted mean of
    # wave-1 final-chunk microbatch completion times (the no-queue anchor;
    # see _evaluate_pp_wave). None for decode waves and non-repeated
    # estimates.
    prefill_request_ttft_s: float | None = None


@dataclass(frozen=True)
class _PPMicrobatchProfile:
    """A cached PP profile together with the empirical lookup evidence it used."""

    profile: PipelineProfile
    profiling_source_times_s: dict[str, float]
    profiling_miss_reasons: dict[str, int]


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
        self._pp_profile_cache: dict[tuple, _PPMicrobatchProfile] = {}
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

        if optimizer_data.max_concurrency is not None:
            local_batch_limit = optimizer_data.max_concurrency // self.dp
            if not batch_range:
                right = local_batch_limit
            else:
                right = min(right, local_batch_limit)
            if left > right:
                logger.warning(
                    "Skipping candidate: max_concurrency=%d with attention DP=%d leaves no batch size in range [%d, %d].",
                    optimizer_data.max_concurrency,
                    self.dp,
                    left,
                    right,
                )
                return None

        # early_stop
        optimizer_data.batch_size = left
        summary = self.get_inference_info(optimizer_data)
        if summary.check_early_stop_flag():
            self._last_run_early_stop_reason = summary.get_early_stop_reason()
            return None

        if not batch_range and optimizer_data.max_concurrency is None:
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
        image_tokens_included: bool = False,
        carry_image_input: bool = True,
    ) -> ForwardShapeKey:
        """Build the forward-shape cache key.

        ``image_tokens_included`` mirrors ``RequestInfo.image_tokens_included``:
        chunk-plan callers pass True because their ``query_len``/``seq_len``
        already contain the image token budget (see
        ``OptimizerData.get_prefill_chunk_plan``).

        ``carry_image_input`` controls whether the key (and therefore the
        request it describes) carries the image forward inputs. Chunk-plan
        callers pass it only for the request's first chunk so the vision
        tower is evaluated exactly once per request; text-only chunks get
        ``image_*=None`` keys.
        """
        query_len, seq_len = self._resolve_forward_shape(
            optimizer_data,
            is_decode,
            query_len=query_len,
            seq_len=seq_len,
        )

        image_batch_size = self._resolve_image_batch_size(optimizer_data) if carry_image_input else None
        return ForwardShapeKey(
            is_decode=is_decode,
            model_concurrency=concurrency,
            query_len=query_len,
            seq_len=seq_len,
            image_batch_size=image_batch_size,
            image_height=(optimizer_data.image_height if carry_image_input else None),
            image_width=(optimizer_data.image_width if carry_image_input else None),
            image_tokens_included=image_tokens_included,
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
        if optimizer_data.max_concurrency is None:
            probe_batch_size = max(batch_range) if batch_range else DEFAULT_MAX_BATCH_SIZE
        else:
            local_batch_limit = optimizer_data.max_concurrency // self.dp
            if local_batch_limit < 1 or (batch_range and len(batch_range) == 2 and batch_range[0] > local_batch_limit):
                raise ValueError("max_concurrency and batch_range leave no batch size for compile calibration.")
            probe_batch_size = min(max(batch_range) if batch_range else local_batch_limit, local_batch_limit)
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
        # offload fields are ModelRunnerMetrics fields (Wiki RFC ``rfc_kv_cache_offloading_en.md``).
        # fold_offload_transfer_time_ms is the shared phase-aware fold-in
        # (prefill: transfer overlaps compute -> max; decode: serial -> "+");
        # its getattr/isinstance contract keeps this robust to lightweight
        # metrics mocks (SimpleNamespace) that predate the offload substrate.
        base_latency_ms = self._select_latency_s(metrics.execution_time_s) * 1000
        latency_ms = fold_offload_transfer_time_ms(
            base_latency_ms,
            getattr(metrics, "offload_mode", "none"),
            getattr(metrics, "offload_transfer_time_ms", 0.0),
        )
        return ForwardLatencyRecord(
            latency_ms=latency_ms,
            memory_left_gb=metrics.device_memory_available_gb,
            breakdowns=format_breakdowns(metrics.breakdowns),
            memory_info=build_memory_info(metrics),
            raw_breakdowns=metrics.breakdowns,
            profiling_source_times_s=getattr(metrics, "profiling_source_times_s", {}),
            profiling_miss_reasons=getattr(metrics, "profiling_miss_reasons", {}),
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
            forward_image=(key.image_batch_size, key.image_height, key.image_width),
            image_tokens_included=key.image_tokens_included,
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
        forward_image: tuple[int | None, int | None, int | None] | None = None,
        image_tokens_included: bool = False,
    ) -> ModelRunnerMetrics:
        """Run one uniform forward and return its metrics.

        ``forward_image`` is an explicit ``(image_batch_size, image_height,
        image_width)`` override for the request's image inputs. ``None`` (the
        default) resolves them from ``optimizer_data`` — the historical
        behavior. Chunk-plan callers pass the first-chunk geometry or an
        all-``None`` triple for text-only chunks, so the vision tower runs
        exactly once per request instead of once per chunk.

        ``image_tokens_included`` mirrors ``RequestInfo.image_tokens_included``:
        chunk-plan callers pass True (their lengths already contain the image
        token budget) so the input generator does not add it a second time.
        """
        query_len, seq_len = self._resolve_forward_shape(
            optimizer_data,
            is_decode,
            query_len=query_len,
            seq_len=seq_len,
        )

        if forward_image is None:
            image_batch_size = self._resolve_image_batch_size(optimizer_data)
            image_height = optimizer_data.image_height
            image_width = optimizer_data.image_width
        else:
            image_batch_size, image_height, image_width = forward_image

        requests = [
            RequestInfo(
                query_len=query_len,
                seq_len=seq_len,
                image_batch_size=image_batch_size,
                image_height=image_height,
                image_width=image_width,
                concurrency=concurrency,
                is_decode=is_decode,
                image_tokens_included=image_tokens_included,
            )
        ]

        runner = self.model_runner
        assert runner is not None, "initialize() must set model_runner"
        try:
            metrics = runner.run_inference(requests, generate_inputs_func=generate_inputs)
        except HostKVCapacityExceeded as exc:
            # Host-pool capacity is a search-domain signal (review 10-09): probe
            # batches may legitimately outgrow --host-memory-available-gb. Fold
            # the candidate into the existing OOM early-stop path (available < 0)
            # so the search shrinks its upper bound and keeps feasible results
            # instead of aborting the whole run.
            logger.warning("Candidate rejected: %s", exc)
            metrics = _host_capacity_rejected_metrics(exc)

        return metrics

    def _resolve_image_batch_size(self, optimizer_data: OptimizerData) -> int | None:
        """Resolve the per-request image count.

        ``image_batch_size`` is the number of images per request and must not
        be coupled to the concurrency search's ``batch_size`` (review:
        Hudingyi/zhangzhenyu — the historical fixed-length fallback silently
        multiplied the per-request image count with the search concurrency,
        inflating vision input, H2D bytes, placeholder tokens and latency
        ~batch_size times). Both the length-distribution and the fixed-length
        paths default to one image per request (issue #586: the surviving
        fixed-length fallback also broke the fail-fast image-budget guard once
        the chunk planner started rejecting undersized ``max_batched_tokens``).
        Pass --image-batch-size explicitly to control the per-request image
        count. Without an explicit value the fallback warns once per process.
        Whether the model actually consumes image inputs is decided by
        ``_image_tokens_per_request`` / the input generators (stray image
        parameters on a text model keep the historical "ignored + warning"
        behaviour there).
        """
        global _IMAGE_BATCH_DEFAULT_WARNED
        if optimizer_data.image_height is None:
            return None
        if optimizer_data.image_batch_size is not None:
            return optimizer_data.image_batch_size
        if not _IMAGE_BATCH_DEFAULT_WARNED:
            logger.warning(
                "image_batch_size defaults to 1 image per request; pass "
                "--image-batch-size explicitly to control the per-request "
                "image count."
            )
            _IMAGE_BATCH_DEFAULT_WARNED = True
        return 1

    def _image_tokens_per_request(self, optimizer_data: OptimizerData) -> int:
        """Language-side image token budget per request for the chunk planner.

        The chunk plan must count the multimodal tokens toward each
        request's prefill length, otherwise the first chunk can exceed
        ``max_batched_tokens`` once the image tokens are appended by the
        varlen input generator. Text-only configurations return 0.
        """
        image_batch_size = self._resolve_image_batch_size(optimizer_data)
        if image_batch_size is None or optimizer_data.image_width is None:
            return 0
        if not getattr(getattr(self.model_runner, "model", None), "is_vl_model", False):
            # Non-VL models never contribute an image token budget: stray
            # image parameters on a text model are configuration noise and
            # keep the historical "ignored" behaviour in the generators
            # (PR-1 review round 6 — the fail-closed raise below only
            # guards VL models whose own budget formula fails).
            return 0
        try:
            from tensor_cast.core.input_generator import resolve_multimodal_token_budget

            return resolve_multimodal_token_budget(
                self.model_runner.model,
                image_batch_size,
                optimizer_data.image_height,
                optimizer_data.image_width,
            )
        except (AttributeError, ValueError) as exc:
            # Fail closed (review: zhangzhenyu round 6): a fully configured
            # image geometry whose budget cannot be resolved must not build a
            # chunk plan that under-counts the image tokens while still
            # declaring image_tokens_included=True — that silently produces
            # wrong shapes. Text-only configurations never reach here (no
            # image geometry resolves to 0 before the try).
            raise ValueError(
                "Image token budget resolution failed for a configured image geometry; "
                "refusing to build a chunk plan that under-counts image tokens "
                f"(image_batch_size={image_batch_size}, "
                f"image_height={optimizer_data.image_height}, "
                f"image_width={optimizer_data.image_width}): {exc}"
            ) from exc

    def _resolve_forward_shape(
        self,
        optimizer_data: OptimizerData,
        is_decode: bool,
        *,
        query_len: int = None,
        seq_len: int = None,
    ) -> tuple[int, int]:
        """Resolve the RequestInfo shape, allowing chunked prefill callers to override it.

        Without overrides, prefill computes only the effective input tokens after prefix-cache
        reduction while retaining the cached prefix in the total KV sequence length. Decode keeps
        the original prompt length and only computes the next decode/MTP tokens. Chunked prefill
        passes explicit query_len/seq_len so each chunk can be modeled with its own newly computed
        token count and accumulated context length, including the cached prefix.

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
            # Full prefill computes only cache-miss tokens, but those queries still attend to the
            # cached prefix. Chunked prefill provides both lengths explicitly.
            effective_input_length = optimizer_data.get_effective_input_length()
            cached_prefix_length = optimizer_data.get_prefill_cached_prefix_length() or 0
            resolved_query_len = query_len or effective_input_length
            resolved_seq_len = seq_len or cached_prefix_length + resolved_query_len

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
                        # The chunk that consumes the request's image
                        # placeholders carries the image forward inputs;
                        # later chunks of the same request are text-only
                        # (image_*=None), so the image is generated and
                        # counted exactly once per request.
                        image_batch_size=(
                            self._resolve_image_batch_size(optimizer_data) if chunk.is_first_chunk else None
                        ),
                        image_height=(optimizer_data.image_height if chunk.is_first_chunk else None),
                        image_width=(optimizer_data.image_width if chunk.is_first_chunk else None),
                        is_decode=False,
                        num_output_tokens=optimizer_data.output_length,
                        # Length contract: the chunk plan already folded the
                        # image token budget into query_len/seq_len (see
                        # OptimizerData.get_prefill_chunk_plan), so the varlen
                        # generator must not add num_image_tokens again.
                        image_tokens_included=True,
                    )
                    for chunk in chunks
                ]
                try:
                    metrics = self.model_runner.run_inference(
                        requests,
                        generate_inputs_func=generate_inputs_varlen,
                    )
                except HostKVCapacityExceeded as exc:
                    # Search-domain signal (see _get_forward_info): fold the
                    # chunk into the OOM early-stop path; the loop below breaks
                    # on a negative ``device_memory_available_gb``.
                    logger.warning("Candidate chunk rejected: %s", exc)
                    metrics = _host_capacity_rejected_metrics(exc)
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
                        seq_len=row["num_input_tokens"],
                        image_batch_size=self._resolve_image_batch_size(optimizer_data),
                        image_height=optimizer_data.image_height,
                        image_width=optimizer_data.image_width,
                        is_decode=False,
                        num_input_tokens=row["num_input_tokens"],
                        num_output_tokens=optimizer_data.output_length,
                    )
                )

        try:
            metrics = self.model_runner.run_inference(requests, generate_inputs_func=generate_inputs_varlen)
        except HostKVCapacityExceeded as exc:
            # Search-domain signal (see _get_forward_info): fold the
            # candidate into the OOM early-stop path instead of aborting.
            logger.warning("Candidate rejected: %s", exc)
            metrics = _host_capacity_rejected_metrics(exc)
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

        The token budget limits each scheduled microbatch, not the candidate's
        total concurrency. Raises ``UnsupportedPPConfigurationError`` only for
        candidate-level incompatibilities that no batch size can fix.

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

        if chunk_plan:
            # Chunked prefill token-budget contract: the pipeline schedule
            # sends one microbatch per stage at a time (1F1B), so the binding
            # per-step constraint is one microbatch's tokens, not the whole
            # wave's batch_size * query_len. get_prefill_chunk_plan caps every
            # chunk's query_len at max_batched_tokens; guard it explicitly so
            # the contract stays executable if the chunk plan changes. The
            # same rule applies to single- and multi-chunk prompts.
            if max_batched_tokens is not None:
                oversized = [chunk.query_len for chunk in chunk_plan if chunk.query_len > max_batched_tokens]
                if oversized:
                    raise UnsupportedPPConfigurationError(
                        f"PP>1 {phase_label} chunk query_len {max(oversized)} exceeds "
                        f"max_batched_tokens {max_batched_tokens}; the chunk plan must split "
                        "prompts to fit the per-microbatch token budget."
                    )
        elif max_batched_tokens is not None and effective_input_length > max_batched_tokens:
            raise UnsupportedPPConfigurationError(
                f"PP>1 {phase_label} query length {effective_input_length} exceeds "
                f"max_batched_tokens {max_batched_tokens}; split the prompt into chunks."
            )
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
        forward_image: tuple[int | None, int | None, int | None] | None = None,
        image_tokens_included: bool = False,
    ) -> _PPMicrobatchProfile:
        """Profile one microbatch shape and return its profile plus lookup evidence.

        Caches per (is_decode, query_len, seq_len, microbatch_size,
        forward_image, image_tokens_included) — chunked-prefill microbatches
        of the same (query_len, seq_len) but different image state (vision
        tower on the first chunk vs text-only follow-ups) are distinct
        shapes. Raises if the run did not produce a pipeline_profile (PP>1
        must always produce one — never silently fall back to a mismatched
        shape).

        ``concurrency`` passed to ``run_inference`` is ``microbatch_size * dp``:
        ``generate_inputs`` divides concurrency by DP to get the per-rank
        batch, so this yields a stage-local batch of exactly ``microbatch_size``
        requests per DP rank (not ``microbatch_size / dp``, which would
        under-size latency, payload, KV cache and peak memory by DP×).

        ``forward_image``/``image_tokens_included`` follow the
        ``_get_forward_info`` contract for chunk-plan rows.
        """
        resolved_query_len, resolved_seq_len = self._resolve_forward_shape(
            optimizer_data, is_decode, query_len=query_len, seq_len=seq_len
        )
        cache_key = (
            is_decode,
            resolved_query_len,
            resolved_seq_len,
            microbatch_size,
            forward_image,
            image_tokens_included,
        )
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
            forward_image=forward_image,
            image_tokens_included=image_tokens_included,
        )
        profile = getattr(metrics, "pipeline_profile", None)
        if profile is None:
            raise UnsupportedPPConfigurationError(
                "PP>1 evaluation expected a pipeline_profile from run_inference but got None; "
                "the model is not a PipelineModel or profiling did not produce a profile."
            )
        profile_record = _PPMicrobatchProfile(
            profile=profile,
            profiling_source_times_s=self._profiling_metric_values(metrics, "profiling_source_times_s"),
            profiling_miss_reasons=self._profiling_metric_values(metrics, "profiling_miss_reasons"),
        )
        self._pp_profile_cache[cache_key] = profile_record
        return profile_record

    @staticmethod
    def _profiling_metric_values(metrics: ModelRunnerMetrics, attribute: str) -> dict:
        values = getattr(metrics, attribute, None)
        return dict(values) if isinstance(values, Mapping) else {}

    @staticmethod
    def _aggregate_pp_profiling_metrics(
        profile_records: tuple[_PPMicrobatchProfile, ...],
    ) -> tuple[dict[str, float], dict[str, int]]:
        source_times_s: dict[str, float] = {}
        miss_reasons: dict[str, int] = {}
        for record in profile_records:
            for source, latency_s in record.profiling_source_times_s.items():
                source_times_s[source] = source_times_s.get(source, 0.0) + latency_s
            for reason, count in record.profiling_miss_reasons.items():
                miss_reasons[reason] = miss_reasons.get(reason, 0) + count
        return source_times_s, miss_reasons

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
        chunk_shapes: list[PrefillChunk] | None = None,
        prefill_token_budget: int | None = None,
    ) -> _PPWaveEvaluation:
        """Profile, schedule, and memory-gate one PP microbatch wave.

        ``resident_policy`` owns the cache lifecycle: ``full`` keeps the entire
        wave resident, while ``inflight`` delegates the window size to the
        scheduler estimator (``min(pp, num_microbatches)``). When
        ``mixed_prefill`` is supplied, the mixed P/D estimator covers prefill,
        decode, and their overlap in one gate.

        When ``chunk_shapes`` is provided (a list of ``PrefillChunk``), each
        chunk becomes a group of microbatches in the pipeline schedule,
        mirroring vLLM's chunked-prefill + PP 1F1B scheduling. This is used
        for chunked prefill where the effective prompt exceeds
        ``max_batched_tokens`` and is split into multiple chunks. Chunk
        lengths already include the image token budget
        (``image_tokens_included=True``); only each request's first chunk
        carries image forward inputs, so the vision tower is evaluated once
        per request across the chunk pipeline. An explicit
        ``prefill_token_budget`` caps each microbatch by its chunk's query
        length; it does not cap the number of requests in the candidate.
        """
        microbatch_sizes = split_batch_size(batch_size, self._pp_microbatch_size())
        if chunk_shapes is not None:
            # Chunked prefill: each chunk is a group of microbatches with its
            # own (query_len, seq_len).  The combined sequence is ordered by
            # chunk index then microbatch index within each chunk, so the
            # pipeline schedule models 1F1B chunk pipelining across stages.
            chunk_microbatch_sizes = [
                split_batch_size(batch_size, min(self._pp_microbatch_size(), prefill_token_budget // chunk.query_len))
                if prefill_token_budget is not None
                else microbatch_sizes
                for chunk in chunk_shapes
            ]
            profile_records = tuple(
                self._profile_pp_microbatch(
                    mb_size,
                    optimizer_data,
                    is_decode,
                    query_len=chunk.query_len,
                    seq_len=chunk.seq_len,
                    forward_image=(
                        (
                            self._resolve_image_batch_size(optimizer_data),
                            optimizer_data.image_height,
                            optimizer_data.image_width,
                        )
                        if chunk.is_first_chunk
                        else (None, None, None)
                    ),
                    image_tokens_included=True,
                )
                for chunk, sizes in zip(chunk_shapes, chunk_microbatch_sizes)
                for mb_size in sizes
            )
        else:
            if prefill_token_budget is not None:
                microbatch_sizes = split_batch_size(
                    batch_size, min(self._pp_microbatch_size(), prefill_token_budget // query_len)
                )
            profile_records = tuple(
                self._profile_pp_microbatch(
                    microbatch_size,
                    optimizer_data,
                    is_decode,
                    query_len=query_len,
                    seq_len=seq_len,
                )
                for microbatch_size in microbatch_sizes
            )
        profiles = tuple(record.profile for record in profile_records)
        profiling_source_times_s, profiling_miss_reasons = self._aggregate_pp_profiling_metrics(profile_records)
        perf_model_name = self._pp_perf_model_name()
        repeated = estimate_repeated_pipeline(profiles, perf_model_name) if repeat else None
        schedule = repeated.first_wave if repeated is not None else estimate_forward_pipeline(profiles, perf_model_name)
        if publish_schedule:
            optimizer_data.pipeline_schedule_estimate = schedule

        # Request-level TTFT definition for PP>1 prefill (uniform across the
        # optimizer): ttft = sum(size_i * completion_i) / sum(size_i), where
        # completion_i is microbatch i's finish time on the final stage in
        # wave 1 and size_i is its per-DP-rank request count:
        # - profiles are chunk-major (chunk index outer, microbatch inner), so
        #   the final chunk owns the last len(microbatch_sizes) completion
        #   slots — earlier chunks produce no first token;
        # - weights are request counts because split_batch_size may produce
        #   uneven microbatches (a naive per-microbatch average would give the
        #   remainder microbatch an equal share);
        # - the anchor is wave 1, not the steady-wave (wave-2) offsets.
        #   Production engines inject batches continuously into a pipeline
        #   that never drains and has no inter-batch barrier, so at
        #   sub-saturation operating points a batch's requests queue behind
        #   in-flight work only — not behind a whole previous batch. The
        #   wave-2 offsets model a saturated closed batch (verified: chaining
        #   waves converges exactly at wave 2, so the wave-1/wave-2 gap is a
        #   permanent orbit difference, not a dissolving transition) and
        #   over-state TTFT against production. The steady-state rate stays
        #   carried by measured_interval_s (the wave-2 orbit), so TTFT
        #   (no-queue anchor) and throughput (saturated anchor) intentionally
        #   model different physics. Wave-1's makespan/cooldown remain
        #   available as warmup diagnostics (pp_makespan_ms column).
        prefill_request_ttft_s = None
        if repeated is not None and not is_decode:
            final_chunk_sizes = chunk_microbatch_sizes[-1] if chunk_shapes is not None else microbatch_sizes
            wave1_final_chunk = repeated.first_wave.microbatch_completion_s[-len(final_chunk_sizes) :]
            prefill_request_ttft_s = sum(
                size * completion for size, completion in zip(final_chunk_sizes, wave1_final_chunk)
            ) / sum(final_chunk_sizes)

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
            profiling_source_times_s=profiling_source_times_s,
            profiling_miss_reasons=profiling_miss_reasons,
            prefill_request_ttft_s=prefill_request_ttft_s,
        )
