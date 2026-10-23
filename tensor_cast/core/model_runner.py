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
# _*_coding:utf-8_*_
"""
ModelRuner
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

import torch

from ..device import DeviceProfile
from ..layers.sampler import Sampler
from ..performance_model.analytic import AnalyticPerformanceModel
from ..performance_model.calibrated_analytic import CalibratedAnalyticPerformanceModel
from ..performance_model.calibration import ProfileCalibrationDataSource
from ..performance_model.empirical import (
    EmpiricalOpRecord,
    EmpiricalPerformanceModel,
    PROFILING_SOURCE_SCOPE,
    summarize_empirical_records,
)
from ..performance_model.memory_tracker import MemoryTracker
from ..performance_model.profiling_database import InterpolatingDataSource, ProfilingDataSource
from ..performance_model.utils import bytes_of_tensor
from ..pipeline_parallel import PipelineModel, PipelineProfile, PipelineRunResult, PipelineRunner
from ..runtime import Runtime
from ..transformers.custom_model_registry import get_visual
from .input_generator import dcp_kv_token_capacity_factor
from .input_generator import kv_cache_excluded_layer_indices
from .input_generator import (
    generate_inputs_varlen,
    get_inputs_num_bytes,
    get_kv_cache_info,
    RequestInfo,
)
from .model_builder import build_model

if TYPE_CHECKING:
    from ..performance_model import PerformanceModel
    from .user_config import UserInputConfig


logger = logging.getLogger(__name__)


def _primary_model_output(output: object) -> object:
    """Unwrap ``(primary, side_outputs...)`` used to keep compile residency edges.

    Dflash Prefill returns ``(target_logits, draft_logits)`` so the draft subgraph
    remains a formal graph output under ``torch.compile``. Dflash/DSpark Decode may
    return ``(draft_tokens, next_tokens[, confidence])`` so verify / ConfidenceHead
    paths are not DCE'd. Callers only consume the primary tensor — Decode primary is
    token ids, not sampler logits (see ``_spec_decode_skips_runner_sampler``).
    """
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


def _spec_decode_skips_runner_sampler(model: object, input_kwargs: dict) -> bool:
    """True when DFlash/DSpark/MTP Decode already sampled inside the wrapper.

    Decode primary is ``draft_tokens`` (DFlash) or ``token_ids`` (MTP), not vocab
    logits — do not feed it to the runner sampler.
    """
    from ..layers.dflash import DflashWrapper
    from ..layers.mtp import MtpWrapper

    # For PipelineModel, the wrapper sits on the last stage's model, not on
    # the top-level PipelineModel (which has no ``_inner``). Delegate to it.
    if hasattr(model, "stages"):
        stages = list(getattr(model, "stages", []))
        if stages:
            last_stage_model = getattr(stages[-1], "model", None)
            if last_stage_model is not None:
                model = last_stage_model

    # Start at ``model`` itself: wrappers may be the top node or nested under
    # ``TransformerModel._inner``. Skipping straight to ``_inner`` would miss a
    # bare wrapper and incorrectly keep the runner sampler on Decode.
    node = model
    seen: set[int] = set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, DflashWrapper):
            return DflashWrapper._is_decode_step(input_kwargs)
        if isinstance(node, MtpWrapper):
            # MTP always samples internally (self.sampler = Sampler() in __init__).
            # Its forward returns [batch, num_mtp_layers+1] token IDs, not logits.
            return True
        node = getattr(node, "_inner", None)
    return False


class ModelRunner:
    """
    corresponding to one data-parallel partition ('dp_rank')
    """

    def __init__(self, user_input: UserInputConfig):
        self.user_input = user_input

        # ---------- 1. init device profile and performance_model ----------
        if user_input.device not in DeviceProfile.all_device_profiles:
            logger.error(
                "Unsupported device: %s. Available devices: %s",
                user_input.device,
                list(DeviceProfile.all_device_profiles.keys()),
            )
            raise ValueError(f"Device '{user_input.device}' not recognized.")

        logger.info("Loading device profile")
        self.device_profile = DeviceProfile.all_device_profiles[user_input.device]
        logger.debug("Device profile loaded: %s", self.device_profile)

        logger.info("Initializing performance model")
        self.perf_models = self.create_performance_models(user_input, self.device_profile)
        logger.debug("Performance models initialized: %s", self.perf_models)

        #  ---------- 2. generate default request from user config----------
        logger.info("Generating request information")
        if user_input.num_queries != 0:
            self.request_info_default = [user_input.get_request_info()]
            logger.debug("Request configured: %s", self.request_info_default)
        else:
            logger.debug("No default requests configured (num_queries = 0)")
            self.request_info_default = None

        # ---------- 3. build model ----------
        logger.info("Building model architecture")
        self.model = build_model(user_input).eval()
        logger.debug("Model built:_%s", self.model)

        # ---------- 4. static_memory ----------
        self.total_device_memory_gb = self.device_profile.memory_size_bytes / 1024**3
        self.model_weight_size_gb = self.model.weight_size / 1024**3
        text_config = getattr(self.model, "text_config", None)
        engram_rows = tuple(getattr(text_config, "engram_num_embeddings", ()))
        engram_head_dim = int(getattr(text_config, "engram_head_dim", 0) or 0)
        engram_row_bytes = engram_head_dim + (engram_head_dim // 32 if engram_head_dim else 0)
        self.engram_host_resident_size_gb = sum(engram_rows) * engram_row_bytes / 1024**3
        group_manager = getattr(self.model, "parallel_group_manager", None)
        all_rank_group = getattr(group_manager, "all_rank_group", None)
        engram_shard_size = max(int(getattr(all_rank_group, "world_size", 1)), 1)
        self.engram_host_resident_per_rank_gb = self.engram_host_resident_size_gb / engram_shard_size

        logger.info("Initializing Sampler")
        self.sampler = Sampler()
        self._static_meta_forward = None
        logger.debug("Sampler initialized: %s", self.sampler)

    def enable_static_meta_capture(self) -> bool:
        """Enable the validated static V4 serving path without changing model config.

        Other models, dynamic/partial graphs and scheduling-sensitive compiler
        modes retain the Dynamo path. The compiler captures each exact workload;
        this does not interpolate shapes or share device performance results.
        """
        from .. import config

        native = getattr(self.model, "_orig_mod", None)
        if (
            not self.user_input.do_compile
            or self.user_input.dynamic_shapes
            or self.user_input.allow_graph_break
            or native is None
            or getattr(native, "is_vl_model", False)
            or getattr(getattr(native, "hf_config", None), "model_type", None) != "deepseek_v4"
            or native.model_config.parallel_config.pipeline_parallel_size != 1
            or self.user_input.dflash
            or self.user_input.dspark
            or getattr(self.user_input, "enable_prefill_kv_offload", False)
            or getattr(self.user_input, "enable_decode_kv_offload", False)
            or not config.compilation.enable_freezing
            or config.compilation.multistream.enable
            or config.compilation.passes.enable_sequence_parallel
        ):
            self._static_meta_forward = None
            return False
        from ..compilation import get_backend
        from ..compilation.static_meta import StaticMetaForward

        self._static_meta_forward = StaticMetaForward(native, get_backend(device_name=self.user_input.device))
        logger.info("Aggregation uses exact static FX capture for DeepSeek V4")
        return True

    @staticmethod
    def create_performance_models(
        user_input: "UserInputConfig",
        device_profile: DeviceProfile,
    ) -> List["PerformanceModel"]:
        """Create performance models without building a model graph.

        Workload-reuse callers use this factory to bind an already captured runtime
        workload to another device profile. Keeping the construction here ensures
        the replay path uses exactly the same analytic/profiling setup as
        ``ModelRunner``'s direct path.
        """
        perf_model_types: List[str] = getattr(user_input, "performance_model", ["analytic"])
        if isinstance(perf_model_types, str):
            perf_model_types = [perf_model_types]
        profiling_database = getattr(user_input, "profiling_database", None)
        perf_models: List[PerformanceModel] = []
        for perf_model_type in perf_model_types:
            if perf_model_type == "profiling":
                if not profiling_database:
                    raise ValueError("--profiling-database must be specified when using --performance-model profiling")
                data_source = ProfilingDataSource(
                    profiling_database,
                    device_profile,
                    parallel_config=user_input.get_parallel_config(),
                )
                if not getattr(user_input, "disable_profiling_interpolation", False):
                    data_source = InterpolatingDataSource(data_source)
                perf_models.append(
                    EmpiricalPerformanceModel(
                        device_profile,
                        data_source=data_source,
                        fallback_model=AnalyticPerformanceModel(device_profile),
                    )
                )
            elif perf_model_type == "analytic":
                analytic_model = AnalyticPerformanceModel(device_profile)
                perf_models.append(analytic_model)
            elif perf_model_type == "calibrated":
                calibration_profile = getattr(user_input, "analytic_calibration_profile", None)
                if not calibration_profile:
                    raise ValueError(
                        "--analytic-calibration-profile must be specified when using --performance-model calibrated"
                    )
                perf_models.append(
                    CalibratedAnalyticPerformanceModel(
                        AnalyticPerformanceModel(device_profile),
                        ProfileCalibrationDataSource(
                            calibration_profile,
                            device_name=device_profile.name,
                            software_stack=getattr(user_input, "analytic_calibration_stack", None),
                            device_compute_efficiency=device_profile.compute_efficiency,
                        ),
                    )
                )
            else:
                raise ValueError(f"Unsupported performance model: {perf_model_type}")
        return perf_models

    def _check_peak_memory_usage_gb(
        self,
        peak_memory_usage_gb: float,
        kv_cache_size_gb: float,
        requests: Optional[List[RequestInfo]],
    ) -> float:
        """
        Check peak memory usage against device limits.

        Args:
            peak_memory_usage_gb: Current peak memory usage in GB
            kv_cache_size_gb: Size of KV cache in GB
            requests: List of RequestInfo objects

        Returns:
            float: Adjusted peak memory usage in GB
        """
        model_activation_size_gb = peak_memory_usage_gb - kv_cache_size_gb - self.model_weight_size_gb
        device_memory_available_gb = (
            self.total_device_memory_gb - peak_memory_usage_gb - self.user_input.reserved_memory_gb
        )
        query_lens = [request.query_len for request in requests if not request.is_decode]

        if device_memory_available_gb < 0 and len(set(query_lens)) > 1:
            if model_activation_size_gb <= 0:
                return self.model_weight_size_gb + kv_cache_size_gb
            # Empirical calibration for heterogeneous multi-sequence Prefill batches.
            peak_memory_usage_gb = 0.1 * model_activation_size_gb + self.model_weight_size_gb + kv_cache_size_gb
            logger.warning("Adjusting peak memory usage")

        return peak_memory_usage_gb

    # Prefill layerwise offload shape defaults (RFC §2.4). B=2 shared physical
    # buffers is the upstream-measured anchor (2x top-k hot buffer reaches an
    # 80-90 percent hit rate); I=0 assumes every layer is cache-bearing --
    # deriving I for hybrid-attention models is an RFC open item. Deliberately
    # not user-facing: revisit in code only when a hybrid model needs it.
    _PREFILL_OFFLOAD_SHARED_BUFFERS = 2
    _PREFILL_OFFLOAD_INDEPENDENT_LAYERS = 0

    def _validate_kv_offload_model(self) -> None:
        """CSA2 ownership and compressed scales need dedicated offload accounting."""
        if not (
            getattr(self.user_input, "enable_decode_kv_offload", False)
            or getattr(self.user_input, "enable_prefill_kv_offload", False)
        ):
            return
        model_config = getattr(getattr(self, "model", None), "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        text_config = getattr(hf_config, "text_config", None) or getattr(hf_config, "hf_text_config", None)
        if any(
            getattr(cfg, "model_type", None) in {"deepseek_v41", "deepseek_v41_text"}
            for cfg in (hf_config, text_config)
        ):
            raise ValueError(
                "DeepSeek V4.1 KV offload is not supported yet: CSA2 shared ownership, "
                "compressed KV payloads and scales require dedicated offload accounting. "
                "Disable enable_decode_kv_offload and enable_prefill_kv_offload."
            )

    @staticmethod
    def _resolve_model_sparse_topk(model) -> int:
        """Resolve the DSA sparse attention window (top-k) from the model config.

        The window is a property of the model checkpoint, not a deployment
        choice, so it is read from the config instead of a CLI flag. Mirrors
        the DSA predicate in ``config_resolver`` (composite configs keep the
        fields on ``text_config`` / ``hf_text_config``) and the adapter
        normalization in the MLA layer (``index_topk``, or ``topk_limit``
        together with ``index_n_heads``). Returns 0 when the config declares
        no usable window; callers fail fast on 0.
        """
        model_config = getattr(model, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        text_config = getattr(hf_config, "text_config", None) or getattr(hf_config, "hf_text_config", None)
        config = text_config or hf_config
        topk = getattr(config, "index_topk", None)
        if topk is None:
            topk_limit = getattr(config, "topk_limit", None)
            if topk_limit is not None and hasattr(config, "index_n_heads"):
                topk = topk_limit
        if isinstance(topk, bool) or not isinstance(topk, (int, float)) or topk <= 0:
            return 0
        return int(topk)

    def _apply_decode_kv_offload_accounting(
        self,
        peak_memory_usage_gb: float,
        kv_cache_bytes: int,
        indexer_cache_bytes: int,
        main_kv_per_token: int,
        batch_size: int,
        is_decode: bool,
    ) -> tuple[float, float, str, float, float]:
        """Peak/KV accounting for decode sparse KV offload.

        (Wiki RFC ``rfc_kv_cache_offloading_en.md``). When decode offload is enabled
        and this is a decode step, the main KV lives on host: drop it from the
        device peak, add the top-k hot buffer, and compute the per-step miss-row
        H2D transfer time. Otherwise return the inputs unchanged with
        ``offload_mode="none"`` so non-offload runs are byte-identical.

        Returns ``(peak_gb, kv_cache_size_gb, offload_mode, host_kv_pool_gb,
        offload_transfer_time_ms)``.
        """
        kv_cache_size_gb = (kv_cache_bytes + indexer_cache_bytes) / 1024**3
        offload_mode = "none"
        host_kv_pool_gb = 0.0
        offload_transfer_time_ms = 0.0
        if not (getattr(self.user_input, "enable_decode_kv_offload", False) and is_decode):
            return (
                peak_memory_usage_gb,
                kv_cache_size_gb,
                offload_mode,
                host_kv_pool_gb,
                offload_transfer_time_ms,
            )

        from .kv_offload import (
            HostKVCapacityExceeded,
            decode_hot_buffer_bytes,
            decode_offload_miss_transfer_time_ms,
        )

        user_input = self.user_input
        topk = ModelRunner._resolve_model_sparse_topk(getattr(self, "model", None))
        if topk <= 0:
            # A zero/unknown top-k would free the whole main KV while fetching
            # nothing: a physically impossible free benefit. The window is a
            # property of the model checkpoint config, not a deployment choice.
            raise ValueError(
                "decode KV offload requires the model checkpoint config to "
                "declare the sparse attention window (index_topk, or topk_limit "
                "with index_n_heads); refusing to remove the main KV without a "
                "sparse top-k window (model config resolves none)"
            )
        if indexer_cache_bytes <= 0:
            # Only sparse-attention models with an indexer can restore a decode
            # step from a top-k hot buffer; dense/GQA models read the full KV.
            raise ValueError(
                "decode KV offload requires a sparse-attention model with an "
                "indexer cache (dense/GQA attention is unsupported)"
            )
        host_bw = self.device_profile.host_bandwidth_bytes_ps
        # Fail fast on the effective bandwidth (bw x efficiency), matching the
        # prefill accounting and the kv_offload_transfer estimator: a mis-set
        # efficiency of 0 would otherwise keep the benefit (main KV freed) while
        # the pure-function transfer math silently returns swap-only cost.
        if host_bw * self.device_profile.host_transfer_efficiency <= 0:
            raise ValueError(
                f"device profile {self.device_profile.name!r} has no usable host "
                "bandwidth; configure host_bandwidth_bytes_ps and "
                "host_transfer_efficiency before enabling decode KV offload"
            )
        if user_input.decode_offload_buffer_multiplier < 1:
            # A sub-1x hot buffer cannot hold even one full top-k window, so the
            # configured hit rate has no physical carrier: the device would keep
            # (almost) no resident KV while still on-loading miss rows from host.
            raise ValueError(
                "decode offload hot buffer multiplier must be >= 1: the device "
                "hot buffer must hold at least one full top-k window for a "
                "positive hit rate to be physically deployable"
            )
        hot_bytes = decode_hot_buffer_bytes(
            main_kv_per_token,
            topk=topk,
            buffer_multiplier=user_input.decode_offload_buffer_multiplier,
            num_seqs=batch_size,
        )
        # Main KV is offloaded to host: remove it from the device peak and add
        # the hot buffer; the device-resident KV becomes indexer + hot buffer.
        peak_memory_usage_gb = peak_memory_usage_gb - kv_cache_bytes / 1024**3 + hot_bytes / 1024**3
        kv_cache_size_gb = (indexer_cache_bytes + hot_bytes) / 1024**3
        host_kv_pool_gb = kv_cache_bytes / 1024**3
        host_limit_gb = getattr(user_input, "host_memory_available_gb", 0.0)
        if host_limit_gb > 0 and host_kv_pool_gb > host_limit_gb:
            raise HostKVCapacityExceeded(
                f"host KV pool for decode offload requires {host_kv_pool_gb:.2f} GB, "
                f"exceeding the configured host memory budget of {host_limit_gb:.2f} GB "
                "(--host-memory-available-gb); reduce batch/context or raise the budget",
                host_kv_pool_gb=host_kv_pool_gb,
                phase="decode",
            )
        offload_transfer_time_ms = decode_offload_miss_transfer_time_ms(
            main_kv_per_token,
            topk=topk,
            hit_rate=user_input.decode_offload_hit_rate,
            num_seqs=batch_size,
            host_bandwidth_bytes_ps=host_bw,
            host_latency_s=self.device_profile.host_latency_s,
            host_efficiency=self.device_profile.host_transfer_efficiency,
        )
        offload_mode = "decode"
        return (
            peak_memory_usage_gb,
            kv_cache_size_gb,
            offload_mode,
            host_kv_pool_gb,
            offload_transfer_time_ms,
        )

    def _apply_prefill_kv_offload_accounting(
        self,
        peak_memory_usage_gb: float,
        kv_cache_bytes: int,
        indexer_cache_bytes: int,
        main_kv_per_token: int,
        batch_size: int,
        is_decode: bool,
        num_layers: int,
    ) -> tuple[float, float, str, float, float]:
        """Peak/KV/transfer accounting for prefill layerwise KV offload.

        (Wiki RFC ``rfc_kv_cache_offloading_en.md``). On a prefill step with offload
        ON, only ``ratio = (I + min(B, N-I)) / N`` of the main KV stays on device
        (B shared physical buffers round-robin across N layers); the rest is on
        host. Drop the offloaded share from device peak so the optimizer's OOM
        search finds a larger prefill batch (the benefit). The transfer is a
        d2h+h2d round trip of the full main KV plus one host-link latency term;
        the first-version cost model is the aggregate ``max(compute, transfer)``
        folded into TTFT -- transfers hide when prefill is compute-bound and
        dominate when the host link is the bottleneck. This aggregate model
        matches the per-layer ``Σ max`` for uniform layers but omits pipeline
        fill/drain overlap, so it is a first-order approximation rather than a
        pessimistic upper bound (per-layer makespan refinement tracked as RFC
        open enhancements). OFF or decode -> unchanged, "none".

        Returns ``(peak_gb, kv_cache_size_gb, offload_mode, host_kv_pool_gb,
        offload_transfer_time_ms)``.
        """
        kv_cache_size_gb = (kv_cache_bytes + indexer_cache_bytes) / 1024**3
        offload_mode = "none"
        host_kv_pool_gb = 0.0
        offload_transfer_time_ms = 0.0
        if not (getattr(self.user_input, "enable_prefill_kv_offload", False) and not is_decode):
            return (
                peak_memory_usage_gb,
                kv_cache_size_gb,
                offload_mode,
                host_kv_pool_gb,
                offload_transfer_time_ms,
            )

        from .kv_offload import HostKVCapacityExceeded, prefill_layerwise_footprint_ratio

        ratio = prefill_layerwise_footprint_ratio(
            num_layers,
            ModelRunner._PREFILL_OFFLOAD_INDEPENDENT_LAYERS,
            ModelRunner._PREFILL_OFFLOAD_SHARED_BUFFERS,
        )
        kv_gb = kv_cache_bytes / 1024**3
        indexer_gb = indexer_cache_bytes / 1024**3
        # Device holds only `ratio` of the main KV (B shared buffers); the
        # remaining (1 - ratio) is offloaded to host. Only the offloaded share
        # counts toward the host pool and the round-trip transfer -- a ratio of
        # 1 (nothing to offload) must be a no-op on both.
        offloaded_gb = kv_gb * (1 - ratio)
        peak_memory_usage_gb = peak_memory_usage_gb - offloaded_gb
        kv_cache_size_gb = ratio * kv_gb + indexer_gb
        host_kv_pool_gb = offloaded_gb
        host_limit_gb = getattr(self.user_input, "host_memory_available_gb", 0.0)
        if host_limit_gb > 0 and host_kv_pool_gb > host_limit_gb:
            raise HostKVCapacityExceeded(
                f"host KV pool for prefill offload requires {host_kv_pool_gb:.2f} GB, "
                f"exceeding the configured host memory budget of {host_limit_gb:.2f} GB "
                "(--host-memory-available-gb); reduce batch/context or raise the budget",
                host_kv_pool_gb=host_kv_pool_gb,
                phase="prefill",
            )

        host_bw_raw = self.device_profile.host_bandwidth_bytes_ps
        bandwidth = host_bw_raw * self.device_profile.host_transfer_efficiency
        if bandwidth <= 0:
            raise ValueError(
                f"device profile {self.device_profile.name!r} has no usable host "
                "bandwidth; configure host_bandwidth_bytes_ps before enabling "
                "prefill KV offload"
            )
        if ratio < 1:
            # d2h (save) + h2d (onload) round trip of the offloaded share only,
            # plus one host-link latency term for the round trip.
            offload_transfer_time_ms = (
                self.device_profile.host_latency_s + (2 * kv_cache_bytes * (1 - ratio)) / bandwidth
            ) * 1000.0
        offload_mode = "prefill"
        return (
            peak_memory_usage_gb,
            kv_cache_size_gb,
            offload_mode,
            host_kv_pool_gb,
            offload_transfer_time_ms,
        )

    def _apply_kv_offload_accounting(
        self,
        peak_memory_usage_gb: float,
        kv_cache_bytes: int,
        indexer_cache_bytes: int,
        main_kv_per_token: int,
        batch_size: int,
        is_decode: bool,
        num_layers: int,
    ) -> tuple[float, float, str, float, float]:
        """Dispatch decode/prefill KV offload accounting by phase.

        Only the phase matching the run is applied; the other path is a no-op,
        so a single run_inference call accounts for at most one offload mode
        (this avoids a no-op recomputing kv_cache_size_gb and undoing the
        active phase's adjustment).
        """
        ModelRunner._validate_kv_offload_model(self)
        if is_decode:
            return self._apply_decode_kv_offload_accounting(
                peak_memory_usage_gb,
                kv_cache_bytes,
                indexer_cache_bytes,
                main_kv_per_token,
                batch_size,
                is_decode,
            )
        return self._apply_prefill_kv_offload_accounting(
            peak_memory_usage_gb,
            kv_cache_bytes,
            indexer_cache_bytes,
            main_kv_per_token,
            batch_size,
            is_decode,
            num_layers,
        )

    @staticmethod
    def _resolve_offload_cache_layers(
        per_layer_kv_bytes: dict[int, int],
        enable_prefill_offload: bool,
    ) -> int:
        """Count the cache-bearing layers and gate prefill-offload applicability.

        The prefill layerwise ratio must spread the shared physical buffers
        across the layers that actually hold paged KV (linear-attention layers
        are excluded upstream of this call), not the total hidden-layer count
        -- otherwise a hybrid model (e.g. 2 KV layers out of 8) silently yields
        a phantom capacity gain (2/8 instead of 2/2). Non-uniform cache-bearing
        layers are refused under prefill offload: a uniform shared-buffer ratio
        would be meaningless there.
        """
        distinct_sizes = set(per_layer_kv_bytes.values())
        if enable_prefill_offload and len(distinct_sizes) > 1:
            raise ValueError(
                "prefill layerwise KV offload assumes uniform cache-bearing "
                f"layers; got {len(distinct_sizes)} distinct per-layer KV sizes -- "
                "refusing to apply a uniform shared-buffer ratio to a non-uniform "
                "layout"
            )
        return len(per_layer_kv_bytes)

    # -----------------------------------------------------
    # public API
    # -----------------------------------------------------
    def _calculate_single_card_tps(self, execution_time_s: float) -> float:
        if not execution_time_s or execution_time_s <= 0:
            raise ValueError("execution_time_s must be positive")
        return (self.user_input.num_queries * self.user_input.query_len) / (
            execution_time_s * self.user_input.world_size
        )

    def run_inference(
        self,
        requests: Optional[List[RequestInfo]] = None,
        generate_inputs_func: Callable = generate_inputs_varlen,
        with_sampler: bool = False,
        runtime_observer: Optional[Callable[[Runtime], None]] = None,
    ) -> ModelRunnerMetrics:
        ModelRunner._validate_kv_offload_model(self)
        empirical_record_offsets = {
            id(perf_model): len(perf_model.op_records)
            for perf_model in self.perf_models
            if isinstance(perf_model, EmpiricalPerformanceModel)
        }
        data_parallel_size = self.model.model_config.parallel_config.data_parallel_size
        logger.debug("data_parallel_size: %s", data_parallel_size)

        batch_size = (self.user_input.num_queries + data_parallel_size - 1) // data_parallel_size
        logger.debug("batch_size: %s", batch_size)

        if requests is None:
            requests = self.request_info_default
        logger.debug("requests: %s", requests)

        input_kwargs = generate_inputs_func(
            self.model,
            requests,
            block_size=self.user_input.block_size,
        )
        run_start = time.perf_counter()
        if isinstance(self.model, PipelineModel):
            if getattr(self.user_input, "enable_decode_kv_offload", False) or getattr(
                self.user_input, "enable_prefill_kv_offload", False
            ):
                # The pipeline path never runs the offload accounting (KV is
                # stage-sharded), so enabling offload here would silently drop
                # the transfer cost. Refuse instead of pretending it works.
                raise ValueError(
                    "KV cache offload is not supported with pipeline parallelism "
                    "(pp > 1) yet; disable --pp-sizes or the offload flags"
                )
            pipeline_runner = PipelineRunner(
                model=self.model,
                perf_models=self.perf_models,
                device_profile=self.device_profile,
                user_input=self.user_input,
            )
            # Guard against double-sampling when MTP/DFlash already sampled
            # internally — mirrors the non-PP path's check at line 261.
            skip_sampler = _spec_decode_skips_runner_sampler(self.model, input_kwargs)
            pipeline_result = pipeline_runner.run(
                input_kwargs,
                with_sampler=with_sampler and not skip_sampler,
                sampler=self.sampler,
                runtime_observer=runtime_observer,
            )
            run_end = time.perf_counter()
            empirical_records = self._new_empirical_records(empirical_record_offsets)
            type(self)._log_empirical_model_stats(empirical_records)
            profiling_source_times_s, profiling_miss_reasons = summarize_empirical_records(empirical_records)
            if self.user_input.chrome_trace:
                PipelineRunner.export_chrome_trace(self.user_input.chrome_trace, pipeline_result.trace_events)
            return self._build_pipeline_metrics(
                pipeline_result,
                batch_size=batch_size,
                run_time_s=run_end - run_start,
                profiling_source_times_s=profiling_source_times_s,
                profiling_miss_reasons=profiling_miss_reasons,
            )

        with (
            Runtime(
                self.perf_models,
                self.device_profile,
                memory_tracker=MemoryTracker(self.device_profile),
            ) as runtime,
            torch.no_grad(),
        ):
            static_forward = getattr(self, "_static_meta_forward", None)
            if static_forward is None:
                output = self.model.forward(**input_kwargs)
            else:
                from ..compilation.static_meta import StaticMetaCaptureUnsupported

                try:
                    output = static_forward(**input_kwargs)
                except StaticMetaCaptureUnsupported as exc:
                    # Validation fails before executing/capturing the model.
                    # Unsupported inputs keep the existing guarded compiler.
                    logger.debug("Static meta capture bypassed: %s", exc)
                    output = self.model.forward(**input_kwargs)
            primary = _primary_model_output(output)
            if with_sampler and not _spec_decode_skips_runner_sampler(self.model, input_kwargs):
                _ = self.sampler(primary, input_kwargs["sampling_metadata"])
        run_end = time.perf_counter()

        # Log empirical model stats if using profiling mode
        empirical_records = self._new_empirical_records(empirical_record_offsets)
        type(self)._log_empirical_model_stats(empirical_records)
        profiling_source_times_s, profiling_miss_reasons = summarize_empirical_records(empirical_records)

        all_execution_time_s = runtime.total_execution_time_s()
        run_time_s = run_end - run_start

        calibrated_result_count = 0
        calibrated_fallback_count = 0
        for event in runtime.event_list:
            result = event.perf_results.get("calibrated")
            if result is None:
                continue
            calibration = result.statistics.get("calibration", {})
            if calibration.get("profile_id"):
                calibrated_result_count += 1
            else:
                calibrated_fallback_count += 1
        if "calibrated" in all_execution_time_s:
            logger.info(
                "Analytic calibration summary: hits=%d fallbacks=%d",
                calibrated_result_count,
                calibrated_fallback_count,
            )

        table_result = runtime.table_averages(
            group_by_input_shapes=self.user_input.dump_input_shapes,
            dump_op_bound_results=self.user_input.dump_op_bound_results,
        )

        perf_model_name = self.perf_models[0].name if self.perf_models else None
        runtime_event_list = self._aggregate_runtime_events(runtime.event_list, perf_model_name=perf_model_name)
        host_to_device_bytes = sum(
            event.op_invoke_info.get_perf_properties().host_to_device_bytes for event in runtime.event_list
        )
        host_to_device_time_s = 0.0
        if perf_model_name is not None:
            host_to_device_time_s = sum(
                float(event.perf_results[perf_model_name].statistics.get("host_to_device_time_s", 0.0))
                for event in runtime.event_list
                if perf_model_name in event.perf_results
            )

        # Calculate TPS for each model
        tps_per_model: Dict[str, float] = {}
        for model_name, exec_time in all_execution_time_s.items():
            if exec_time and exec_time > 0:
                tps_per_model[model_name] = self._calculate_single_card_tps(exec_time)

        peak_memory_usage_gb = runtime.memory_tracker.peak_mem_usage() / 1024**3

        # linear_attention 层占位张量不计入 KV cache 字节（判定集中维护于
        # kv_cache_excluded_layer_indices，确保 kv_cache_bytes 与 kv_cache_per_token 口径一致）。
        excluded_layers = kv_cache_excluded_layer_indices(self.model)
        # V4.1 CSA2 reuse/reindex layers alias the physical cache owned by a
        # source layer. Count each tensor object once while retaining the
        # per-layer lookup keys required by the attention wrappers.
        seen_kv_cache_ids = set()
        per_layer_kv_bytes = {}
        for layer_idx, kv_cache in input_kwargs["kv_cache_by_layers"].items():
            if layer_idx in excluded_layers or id(kv_cache) in seen_kv_cache_ids:
                continue
            seen_kv_cache_ids.add(id(kv_cache))
            per_layer_kv_bytes[layer_idx] = bytes_of_tensor(kv_cache)
        kv_cache_bytes = sum(per_layer_kv_bytes.values())
        seen_indexer_cache_ids = set()
        indexer_cache_bytes = 0
        for kv_cache in input_kwargs.get("indexer_cache_by_layers", {}).values():
            if id(kv_cache) in seen_indexer_cache_ids:
                continue
            seen_indexer_cache_ids.add(id(kv_cache))
            indexer_cache_bytes += bytes_of_tensor(kv_cache)
        for scale_cache in input_kwargs.get("indexer_scale_cache_by_layers", {}).values():
            if id(scale_cache) in seen_indexer_cache_ids:
                continue
            seen_indexer_cache_ids.add(id(scale_cache))
            indexer_cache_bytes += bytes_of_tensor(scale_cache)
        seen_compressed_cache_ids = set()
        compressed_cache_bytes = 0
        for kv_cache in input_kwargs.get("compressed_kv_cache_by_source", {}).values():
            if id(kv_cache) in seen_compressed_cache_ids:
                continue
            seen_compressed_cache_ids.add(id(kv_cache))
            compressed_cache_bytes += bytes_of_tensor(kv_cache)
        for scale_cache in input_kwargs.get("compressed_kv_scale_cache_by_source", {}).values():
            if id(scale_cache) in seen_compressed_cache_ids:
                continue
            seen_compressed_cache_ids.add(id(scale_cache))
            compressed_cache_bytes += bytes_of_tensor(scale_cache)
        is_decode_step = bool(requests) and all(getattr(r, "is_decode", False) for r in requests)
        # Hot-buffer sizing must follow the requests' concurrency: the optimizer
        # encodes the searched batch in requests[0].concurrency (the
        # user_input-derived batch_size is NOT the served batch on that path).
        # Mirror the per-request expansion used by generate_inputs.
        offload_num_seqs = sum(
            (int(getattr(r, "concurrency", 1) or 1) + data_parallel_size - 1) // data_parallel_size for r in requests
        )
        cache_layer_count = self._resolve_offload_cache_layers(
            per_layer_kv_bytes,
            getattr(self.user_input, "enable_prefill_kv_offload", False),
        )
        (
            peak_memory_usage_gb,
            kv_cache_size_gb,
            offload_mode,
            host_kv_pool_gb,
            offload_transfer_time_ms,
        ) = self._apply_kv_offload_accounting(
            peak_memory_usage_gb,
            kv_cache_bytes,
            indexer_cache_bytes,
            int(input_kwargs["kv_cache_per_token"]),
            offload_num_seqs,
            is_decode_step,
            cache_layer_count,
        )
        # CSA2 compressed payloads and scales remain device-resident.
        kv_cache_size_gb += compressed_cache_bytes / 1024**3
        kv_cache_per_token_gb = (
            input_kwargs["kv_cache_per_token"]
            + input_kwargs.get("indexer_cache_per_token", 0)
            + input_kwargs.get("compressed_kv_cache_per_token", 0)
        ) / 1024**3
        # DCP shards the KV cache along the token dimension, so a card can hold more
        # tokens for the same byte budget where that shard is a real saving (``dcp``
        # for MLA latent KV, capped by TP KV replication for GQA, 1 for SFA). The
        # per-token cost above stays the true physical cost; the capacity gain is
        # surfaced as a multiplier that warmup() applies to ``num_blocks``.
        kv_cache_token_capacity_factor = dcp_kv_token_capacity_factor(self.model)
        if get_visual(self.model) and input_kwargs.get("pixel_values") is None:
            # If there is no image input, the visual part does not participate
            # in the calculation and needs to be removed
            visual_weight_size_gb = self.model.get_weight_size_nested([get_visual(self.model)]) / 1024**3
            self.model_weight_size_gb = self.model_weight_size_gb - visual_weight_size_gb
            if self.model.model_config.hf_config.model_type == "deepseek_v41":
                # V4.1 stores its Aligner and boundary embeddings beside the ViT.
                unwrapped = self.model.unwrap()
                aligner = getattr(unwrapped, "aligner", None)
                if aligner is not None:
                    self.model_weight_size_gb -= self.model.get_weight_size_nested([aligner]) / 1024**3
                for name in ("image_start", "image_end", "image_newline"):
                    value = getattr(unwrapped, name, None)
                    if isinstance(value, torch.Tensor):
                        self.model_weight_size_gb -= bytes_of_tensor(value) / 1024**3

        peak_memory_usage_gb = self._check_peak_memory_usage_gb(
            peak_memory_usage_gb,
            kv_cache_size_gb,
            requests,
        )
        model_activation_size_gb = peak_memory_usage_gb - kv_cache_size_gb - self.model_weight_size_gb
        if model_activation_size_gb < 0:
            logger.warning(
                "Negative activation memory estimate (peak=%.6f GB, weight=%.6f GB, kv=%.6f GB); "
                "this may indicate inconsistent peak memory, model weight, or cache accounting. "
                "Clamping activation to 0 and adjusting peak for consistency.",
                peak_memory_usage_gb,
                self.model_weight_size_gb,
                kv_cache_size_gb,
            )
            model_activation_size_gb = 0.0
            peak_memory_usage_gb = self.model_weight_size_gb + kv_cache_size_gb
        device_memory_available_gb = (
            self.total_device_memory_gb - peak_memory_usage_gb - self.user_input.reserved_memory_gb
        )

        if self.user_input.chrome_trace:
            runtime.export_chrome_trace(self.user_input.chrome_trace)

        if runtime_observer is not None:
            runtime_observer(runtime)

        return ModelRunnerMetrics(
            total_device_memory_gb=self.total_device_memory_gb,
            model_weight_size_gb=self.model_weight_size_gb,
            engram_host_resident_size_gb=self.engram_host_resident_size_gb,
            engram_host_resident_per_rank_gb=self.engram_host_resident_per_rank_gb,
            host_to_device_bytes=host_to_device_bytes,
            host_to_device_time_s=host_to_device_time_s,
            peak_memory_usage_gb=peak_memory_usage_gb,
            kv_cache_size_gb=kv_cache_size_gb,
            kv_cache_per_token_gb=kv_cache_per_token_gb,
            kv_cache_token_capacity_factor=kv_cache_token_capacity_factor,
            indexer_cache_size_gb=indexer_cache_bytes / 1024**3,
            indexer_cache_per_token_gb=input_kwargs.get("indexer_cache_per_token", 0) / 1024**3,
            model_activation_size_gb=model_activation_size_gb,
            reserved_memory_gb=self.user_input.reserved_memory_gb,
            device_memory_available_gb=device_memory_available_gb,
            execution_time_s=all_execution_time_s,
            tps_per_model=tps_per_model,
            run_time_s=run_time_s,
            batch_size=batch_size,
            table_result=table_result,
            breakdowns=runtime.get_breakdowns(),
            runtime_event_list=runtime_event_list,
            perf_model_name=perf_model_name,
            profiling_source_times_s=profiling_source_times_s,
            profiling_miss_reasons=profiling_miss_reasons,
            offload_mode=offload_mode,
            host_kv_pool_gb=host_kv_pool_gb,
            offload_transfer_time_ms=offload_transfer_time_ms,
        )

    def _build_pipeline_metrics(
        self,
        pipeline_result: PipelineRunResult,
        *,
        batch_size: int,
        run_time_s: float,
        profiling_source_times_s: Optional[Dict[str, float]] = None,
        profiling_miss_reasons: Optional[Dict[str, int]] = None,
    ) -> ModelRunnerMetrics:
        all_execution_time_s = pipeline_result.execution_time_s
        tps_per_model = {
            model_name: self._calculate_single_card_tps(exec_time)
            for model_name, exec_time in all_execution_time_s.items()
            if exec_time and exec_time > 0
        }

        peak_memory_usage_gb = pipeline_result.peak_memory_usage_bytes / 1024**3
        model_weight_size_gb = (
            pipeline_result.model_weight_size_bytes / 1024**3
            if pipeline_result.model_weight_size_bytes > 0
            else self.model_weight_size_gb
        )
        kv_cache_bytes = pipeline_result.kv_cache_size_bytes
        indexer_cache_bytes = pipeline_result.indexer_cache_size_bytes
        kv_cache_size_gb = (kv_cache_bytes + indexer_cache_bytes) / 1024**3
        kv_cache_per_token_gb = (
            pipeline_result.kv_cache_per_token_bytes + pipeline_result.indexer_cache_per_token_bytes
        ) / 1024**3
        if pipeline_result.model_activation_size_bytes is not None:
            # PP path: activation was computed in _build_run_result for the
            # selected summary stage, using its stage-local runtime peak -
            # weight - kv - indexer (all terms same rank, so it is non-negative
            # and never needs clamping). Use it directly instead of the
            # cross-stage `peak - kv - weight` subtraction below, which can mix
            # different ranks and fabricate a non-existent rank.
            model_activation_size_gb = pipeline_result.model_activation_size_bytes / 1024**3
        else:
            model_activation_size_gb = peak_memory_usage_gb - kv_cache_size_gb - model_weight_size_gb
            if model_activation_size_gb < 0:
                logger.warning(
                    "Negative activation memory estimate (peak=%.6f GB, weight=%.6f GB, kv=%.6f GB); "
                    "this may indicate inconsistent peak memory, model weight, or cache accounting. "
                    "Clamping activation to 0 and adjusting peak for consistency.",
                    peak_memory_usage_gb,
                    model_weight_size_gb,
                    kv_cache_size_gb,
                )
                model_activation_size_gb = 0.0
                peak_memory_usage_gb = model_weight_size_gb + kv_cache_size_gb
        device_memory_available_gb = (
            self.total_device_memory_gb - peak_memory_usage_gb - self.user_input.reserved_memory_gb
        )
        perf_model_name = self.perf_models[0].name if self.perf_models else None

        return ModelRunnerMetrics(
            total_device_memory_gb=self.total_device_memory_gb,
            model_weight_size_gb=model_weight_size_gb,
            engram_host_resident_size_gb=self.engram_host_resident_size_gb,
            engram_host_resident_per_rank_gb=self.engram_host_resident_per_rank_gb,
            peak_memory_usage_gb=peak_memory_usage_gb,
            kv_cache_size_gb=kv_cache_size_gb,
            kv_cache_per_token_gb=kv_cache_per_token_gb,
            indexer_cache_size_gb=indexer_cache_bytes / 1024**3,
            indexer_cache_per_token_gb=pipeline_result.indexer_cache_per_token_bytes / 1024**3,
            model_activation_size_gb=model_activation_size_gb,
            reserved_memory_gb=self.user_input.reserved_memory_gb,
            device_memory_available_gb=device_memory_available_gb,
            execution_time_s=all_execution_time_s,
            tps_per_model=tps_per_model,
            run_time_s=run_time_s,
            batch_size=batch_size,
            table_result=pipeline_result.table_result,
            breakdowns=pipeline_result.breakdowns,
            runtime_event_list=pipeline_result.runtime_event_list,
            perf_model_name=perf_model_name,
            stage_latency_breakdown=pipeline_result.stage_latency_breakdown,
            stage_memory_breakdown=pipeline_result.stage_memory_breakdown,
            pipeline_profile=pipeline_result.pipeline_profile,
            profiling_source_times_s=profiling_source_times_s or {},
            profiling_miss_reasons=profiling_miss_reasons or {},
        )

    def _new_empirical_records(self, record_offsets: Dict[int, int]) -> List[EmpiricalOpRecord]:
        records = []
        for perf_model in self.perf_models:
            if isinstance(perf_model, EmpiricalPerformanceModel):
                records.extend(perf_model.op_records[record_offsets.get(id(perf_model), 0) :])
        return records

    @staticmethod
    def _log_empirical_model_stats(records: List[EmpiricalOpRecord]) -> None:
        if not records:
            return
        from ..performance_model.metrics_collector import MetricsCollector

        collector = MetricsCollector()
        collector.collect_from_records(records)
        collector.log_stats()

    def get_inputs_num_bytes(self, requests: List[RequestInfo]) -> int:
        return get_inputs_num_bytes(self.model, requests, self.user_input.block_size)

    def get_kv_cache_num_bytes(self, num_tokens: int) -> int:
        return get_kv_cache_info(self.model, 1, 1) * num_tokens

    def _aggregate_runtime_events(self, event_list, perf_model_name: Optional[str] = None) -> List[Dict]:
        aggregated: Dict[str, Dict[str, float]] = {}
        for event in event_list:
            name = str(event.op_invoke_info.func)
            entry = aggregated.setdefault(name, {"total": 0.0, "count": 0})
            entry["count"] += 1
            if perf_model_name is None:
                continue
            result = event.perf_results.get(perf_model_name)
            if result is not None:
                entry["total"] += result.execution_time_s

        items: List[Dict] = []
        for name, entry in aggregated.items():
            count = entry["count"]
            total = entry["total"]
            items.append(
                {
                    "name": name,
                    "perf_model": perf_model_name,
                    "perf_total": total,
                    "perf_avg": total / count if count else 0.0,
                    "call_times": count,
                }
            )
        items.sort(key=lambda x: x["perf_total"], reverse=True)
        return items


@dataclass
class ModelRunnerMetrics:
    total_device_memory_gb: float
    model_weight_size_gb: float
    peak_memory_usage_gb: float
    kv_cache_size_gb: float
    kv_cache_per_token_gb: float
    model_activation_size_gb: float
    reserved_memory_gb: float
    device_memory_available_gb: float
    execution_time_s: Dict[str, float]
    """Execution time per performance model, keyed by model name."""
    tps_per_model: Dict[str, float]
    """TPS per performance model, keyed by model name."""
    run_time_s: float
    batch_size: int
    engram_host_resident_size_gb: float = 0.0
    engram_host_resident_per_rank_gb: float = 0.0
    host_to_device_bytes: int = 0
    host_to_device_time_s: float = 0.0
    kv_cache_token_capacity_factor: int = 1
    """Per-card KV token capacity multiplier from DCP (1 when DCP is off, and when the
    token shard buys no real per-card saving). warmup() multiplies ``num_blocks`` by
    this; see ``dcp_kv_token_capacity_factor``."""
    indexer_cache_size_gb: float = 0.0
    indexer_cache_per_token_gb: float = 0.0
    table_result: str = ""
    breakdowns: Dict[str, Dict[str, float]] = field(default_factory=dict)
    runtime_event_list: List[Dict] = field(default_factory=list)
    perf_model_name: Optional[str] = None
    stage_latency_breakdown: List[Dict] = field(default_factory=list)
    stage_memory_breakdown: List[Dict] = field(default_factory=list)
    pipeline_profile: Optional[PipelineProfile] = None
    profiling_source_times_s: Dict[str, float] = field(default_factory=dict)
    profiling_miss_reasons: Dict[str, int] = field(default_factory=dict)
    # KV cache offload substrate (Wiki RFC ``rfc_kv_cache_offloading_en.md``).
    # Populated by Phase 1/2 when offload is active; 0 / "none" otherwise.
    offload_mode: str = "none"  # "none" | "decode" | "prefill"
    host_kv_pool_gb: float = 0.0
    offload_transfer_time_ms: float = 0.0

    def print_info(self):
        print(f"Number of Queries per DP rank: {self.batch_size}")
        print(f"Model compilation and execution time: {self.run_time_s:.3f} s")
        print(self.table_result)
        for model_name, exec_time in self.execution_time_s.items():
            print(f"[{model_name}] Execution time: {exec_time:.6f} s")
            tps = self.tps_per_model.get(model_name)
            if tps is not None:
                print(f"[{model_name}] TPS/Device: {tps:.4g} token/s")

        if self.stage_latency_breakdown:
            print("Pipeline stage latency breakdown:")
            for stage in self.stage_latency_breakdown:
                for model_name, compute_time_s in stage["compute_time_s"].items():
                    outgoing_comm_time_s = stage["outgoing_comm_time_s"].get(model_name, 0.0)
                    total_time_s = stage["total_time_s"].get(
                        model_name,
                        compute_time_s + outgoing_comm_time_s,
                    )
                    print(
                        f"  [{model_name}] stage {stage['stage_id']} "
                        f"(layers {stage['layer_start']}:{stage['layer_end']}): "
                        f"compute={compute_time_s:.6f} s  "
                        f"outgoing_comm={outgoing_comm_time_s:.6f} s  "
                        f"total={total_time_s:.6f} s"
                    )

        print(f"Total device memory: {self.total_device_memory_gb:.3f} GB")
        print(f"  Model weight size: {self.model_weight_size_gb:.3f} GB")
        if self.engram_host_resident_size_gb > 0:
            print(f"  Engram host-resident tables: {self.engram_host_resident_size_gb:.3f} GB")
            print(f"    Per-rank row shard: {self.engram_host_resident_per_rank_gb:.3f} GB")
        if self.host_to_device_bytes > 0:
            print(f"  Host-to-device traffic: {self.host_to_device_bytes / 1024**2:.3f} MB")
            print(f"  Host-to-device time: {self.host_to_device_time_s * 1e3:.3f} ms")
        print(f"  KV cache: {self.kv_cache_size_gb:.3f} GB")
        if self.indexer_cache_size_gb > 0:
            print(f"    Main KV cache: {self.kv_cache_size_gb - self.indexer_cache_size_gb:.3f} GB")
            print(f"    Indexer cache: {self.indexer_cache_size_gb:.3f} GB")
            print(f"    Indexer cache per token: {self.indexer_cache_per_token_gb:.6f} GB")
        print(f"  Model activation size: {self.model_activation_size_gb:.3f} GB")
        print(f"  Reserved memory: {self.reserved_memory_gb:.3f} GB")
        print(f"  Memory available: {self.device_memory_available_gb:.3f} GB")
        if self.stage_memory_breakdown:
            gib = 1024**3
            print("Pipeline stage memory breakdown (per-rank):")
            for stage in self.stage_memory_breakdown:
                print(
                    f"  stage {stage['stage_id']} (layers {stage['layer_start']}:{stage['layer_end']}): "
                    f"weight={stage['weight_bytes'] / gib:.3f} GB  "
                    f"kv={stage['kv_cache_bytes'] / gib:.6f} GB  "
                    f"indexer={stage['indexer_cache_bytes'] / gib:.6f} GB  "
                    f"peak={stage['peak_bytes'] / gib:.3f} GB  "
                    f"activation={stage['activation_bytes'] / gib:.3f} GB"
                )

        print("Stats breakdowns:")
        for breakdown_name, breakdown in self.breakdowns.items():
            total = sum(breakdown.values())
            if total == 0:
                continue
            formatted = ", ".join(f"{key}: {val * 100 / total:.2f}" for key, val in breakdown.items())
            print(f"  {breakdown_name}: {formatted}")
        profiling_total_s = sum(self.profiling_source_times_s.values())
        if profiling_total_s > 0:
            formatted = ", ".join(
                f"{source}: {latency_s * 100 / profiling_total_s:.2f}"
                for source, latency_s in self.profiling_source_times_s.items()
            )
            print(f"  profiling_sources [scope={PROFILING_SOURCE_SCOPE}]: {formatted}")
            if self.profiling_miss_reasons:
                misses = ", ".join(
                    f"{reason}: {count}"
                    for reason, count in sorted(
                        self.profiling_miss_reasons.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                )
                print(f"  profiling_misses: {misses}")

    def dump_json(self, path: str) -> None:
        breakdowns_percent: Dict[str, Dict[str, float]] = {}
        for name, breakdown in self.breakdowns.items():
            total = sum(breakdown.values())
            if total == 0:
                continue
            breakdowns_percent[name] = {k: round(v * 100 / total, 4) for k, v in breakdown.items()}
        payload = {
            "batch_size": self.batch_size,
            "run_time_s": self.run_time_s,
            "execution_time_s": dict(self.execution_time_s),
            "tps_per_model": dict(self.tps_per_model),
            "memory_gb": {
                "total_device": self.total_device_memory_gb,
                "model_weight": self.model_weight_size_gb,
                "peak_usage": self.peak_memory_usage_gb,
                "kv_cache": self.kv_cache_size_gb,
                "kv_cache_per_token": self.kv_cache_per_token_gb,
                "kv_cache_token_capacity_factor": self.kv_cache_token_capacity_factor,
                "model_activation": self.model_activation_size_gb,
                "reserved": self.reserved_memory_gb,
                "available": self.device_memory_available_gb,
            },
            "breakdowns_raw": {k: dict(v) for k, v in self.breakdowns.items()},
            "breakdowns_percent": breakdowns_percent,
            "perf_model_name": self.perf_model_name,
            "profiling_source_scope": PROFILING_SOURCE_SCOPE,
            "profiling_source_times_s": dict(self.profiling_source_times_s),
            "profiling_miss_reasons": dict(self.profiling_miss_reasons),
            "runtime_event_list": self.runtime_event_list,
            "pipeline_stage_latency": self.stage_latency_breakdown,
            "pipeline_stage_memory": self.stage_memory_breakdown,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
