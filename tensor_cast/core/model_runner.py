# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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
from ..performance_model.empirical import EmpiricalPerformanceModel
from ..performance_model.memory_tracker import MemoryTracker
from ..performance_model.profiling_database import ProfilingDataSource
from ..performance_model.utils import bytes_of_tensor
from ..runtime import Runtime
from ..transformers.custom_model_registry import get_visual
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
        perf_model_types: List[str] = getattr(user_input, "performance_model", ["analytic"])
        profiling_database = getattr(user_input, "profiling_database", None)

        self.perf_models: List[PerformanceModel] = []
        for perf_model_type in perf_model_types:
            if perf_model_type == "profiling":
                # Exact match against pre-collected Profiling CSV database
                if not profiling_database:
                    raise ValueError("--profiling-database must be specified when using --performance-model profiling")
                data_source = ProfilingDataSource(
                    profiling_database,
                    self.device_profile,
                    parallel_config=user_input.get_parallel_config(),
                )
                self.perf_models.append(
                    EmpiricalPerformanceModel(
                        self.device_profile,
                        data_source=data_source,
                        fallback_model=AnalyticPerformanceModel(self.device_profile),
                    )
                )
            elif perf_model_type == "analytic":
                # Default: analytic (Roofline)
                self.perf_models.append(AnalyticPerformanceModel(self.device_profile))
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

        logger.info("Initializing Sampler")
        self.sampler = Sampler()
        logger.debug("Sampler initialized: %s", self.sampler)

    # -----------------------------------------------------
    # public API
    # -----------------------------------------------------
    def run_inference(
        self,
        requests: Optional[List[RequestInfo]] = None,
        generate_inputs_func: Callable = generate_inputs_varlen,
        with_sampler: bool = False,
        runtime_observer: Optional[Callable[[Runtime], None]] = None,
    ) -> ModelRunnerMetrics:
        def calculate_single_card_tps(execution_time_s: float) -> float:
            if not execution_time_s or execution_time_s <= 0:
                raise ValueError("execution_time_s must be positive")
            return (self.user_input.num_queries * self.user_input.query_len) / (
                execution_time_s * self.user_input.world_size
            )

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

        with (
            Runtime(
                self.perf_models,
                self.device_profile,
                memory_tracker=MemoryTracker(self.device_profile),
            ) as runtime,
            torch.no_grad(),
        ):
            logits = self.model.forward(**input_kwargs)
            if with_sampler:
                _ = self.sampler(logits, input_kwargs["sampling_metadata"])
        run_end = time.perf_counter()

        # Log empirical model stats if using profiling mode
        for pm in self.perf_models:
            if isinstance(pm, EmpiricalPerformanceModel):
                from ..performance_model.metrics_collector import MetricsCollector

                collector = MetricsCollector()
                collector.collect_from_records(pm.op_records)
                collector.log_stats()

        all_execution_time_s = runtime.total_execution_time_s()
        run_time_s = run_end - run_start

        table_result = runtime.table_averages(
            group_by_input_shapes=self.user_input.dump_input_shapes,
            dump_op_bound_results=self.user_input.dump_op_bound_results,
        )

        perf_model_name = self.perf_models[0].name if self.perf_models else None
        runtime_event_list = self._aggregate_runtime_events(runtime.event_list, perf_model_name=perf_model_name)

        # Calculate TPS for each model
        tps_per_model: Dict[str, float] = {}
        for model_name, exec_time in all_execution_time_s.items():
            if exec_time and exec_time > 0:
                tps_per_model[model_name] = calculate_single_card_tps(exec_time)

        peak_memory_usage_gb = runtime.memory_tracker.peak_mem_usage() / 1024**3

        kv_cache_bytes = sum(bytes_of_tensor(kv_cache) for kv_cache in input_kwargs["kv_cache_by_layers"].values())
        indexer_cache_bytes = sum(
            bytes_of_tensor(kv_cache) for kv_cache in input_kwargs.get("indexer_cache_by_layers", {}).values()
        )
        kv_cache_size_gb = (kv_cache_bytes + indexer_cache_bytes) / 1024**3
        kv_cache_per_token_gb = (
            input_kwargs["kv_cache_per_token"] + input_kwargs.get("indexer_cache_per_token", 0)
        ) / 1024**3
        if get_visual(self.model) and input_kwargs.get("pixel_values") is None:
            # If there is no image input, the visual part does not participate
            # in the calculation and needs to be removed
            visual_weight_size_gb = self.model.get_weight_size_nested([get_visual(self.model)]) / 1024**3
            self.model_weight_size_gb = self.model_weight_size_gb - visual_weight_size_gb

        model_activation_size_gb = peak_memory_usage_gb - kv_cache_size_gb - self.model_weight_size_gb
        if model_activation_size_gb < 0:
            logger.warning(
                "Negative activation memory estimate (peak=%.6f GB, weight=%.6f GB, kv=%.6f GB); "
                "clamping activation to 0 and adjusting peak for consistency.",
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
            peak_memory_usage_gb=peak_memory_usage_gb,
            kv_cache_size_gb=kv_cache_size_gb,
            kv_cache_per_token_gb=kv_cache_per_token_gb,
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
        )

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
    indexer_cache_size_gb: float = 0.0
    indexer_cache_per_token_gb: float = 0.0
    table_result: str = ""
    breakdowns: Dict[str, Dict[str, float]] = field(default_factory=dict)
    runtime_event_list: List[Dict] = field(default_factory=list)
    perf_model_name: Optional[str] = None

    def print_info(self):
        print(f"Number of Queries per DP rank: {self.batch_size}")
        print(f"Model compilation and execution time: {self.run_time_s:.3f} s")
        print(self.table_result)
        for model_name, exec_time in self.execution_time_s.items():
            print(f"[{model_name}] Execution time: {exec_time:.6f} s")
            tps = self.tps_per_model.get(model_name)
            if tps is not None:
                print(f"[{model_name}] TPS/Device: {tps:.4g} token/s")

        print(f"Total device memory: {self.total_device_memory_gb:.3f} GB")
        print(f"  Model weight size: {self.model_weight_size_gb:.3f} GB")
        print(f"  KV cache: {self.kv_cache_size_gb:.3f} GB")
        if self.indexer_cache_size_gb > 0:
            print(f"    Main KV cache: {self.kv_cache_size_gb - self.indexer_cache_size_gb:.3f} GB")
            print(f"    Indexer cache: {self.indexer_cache_size_gb:.3f} GB")
            print(f"    Indexer cache per token: {self.indexer_cache_per_token_gb:.6f} GB")
        print(f"  Model activation size: {self.model_activation_size_gb:.3f} GB")
        print(f"  Reserved memory: {self.reserved_memory_gb:.3f} GB")
        print(f"  Memory available: {self.device_memory_available_gb:.3f} GB")

        print("Stats breakdowns:")
        for breakdown_name, breakdown in self.breakdowns.items():
            total = sum(breakdown.values())
            if total == 0:
                continue
            formatted = ", ".join(f"{key}: {val * 100 / total:.2f}" for key, val in breakdown.items())
            print(f"  {breakdown_name}: {formatted}")

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
                "model_activation": self.model_activation_size_gb,
                "reserved": self.reserved_memory_gb,
                "available": self.device_memory_available_gb,
            },
            "breakdowns_raw": {k: dict(v) for k, v in self.breakdowns.items()},
            "breakdowns_percent": breakdowns_percent,
            "perf_model_name": self.perf_model_name,
            "runtime_event_list": self.runtime_event_list,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
