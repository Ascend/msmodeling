# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""Coordinator-backed workload reuse for multi-device throughput optimization."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from multiprocessing.managers import SyncManager
from types import SimpleNamespace
from typing import Any, Callable

from tensor_cast.core.model_runner import ModelRunner, ModelRunnerMetrics
from tensor_cast.device import DeviceProfile
from tensor_cast.performance_model.memory_tracker import MemoryTracker
from tensor_cast.runtime import Runtime
from tensor_cast.runtime_workload import RuntimeWorkloadTrace, WorkloadFreezeError


logger = logging.getLogger(__name__)
_WORKLOAD_TRACE_SCHEMA_VERSION = 2
_DEFAULT_WORKLOAD_CACHE_MAX_BYTES = 512 * 1024**2
_DEFAULT_WORKLOAD_INFLIGHT_TIMEOUT_S = 600.0
_MODEL_KEY_FIELDS = (
    "model_id",
    "world_size",
    "tp_size",
    "pp_size",
    "dp_size",
    "ep_size",
    "moe_dp_size",
    "moe_tp_size",
    "o_proj_tp_size",
    "o_proj_dp_size",
    "mlp_tp_size",
    "mlp_dp_size",
    "lmhead_tp_size",
    "lmhead_dp_size",
    "vision_tp_size",
    "word_embedding_tp",
    "quantize_linear_action",
    "quantize_non_expert_linear_action",
    "quantize_attention_action",
    "quantize_lmhead",
    "mxfp4_group_size",
    "num_mtp_tokens",
    "mtp_acceptance_rate",
    "prefix_cache_hit_rate",
    "block_size",
    "do_compile",
    "allow_graph_break",
    "dynamic_shapes",
    "enable_multistream",
    "enable_sequence_parallel",
    "enable_matmul_allreduce",
    "enable_dispatch_ffn_combine",
    "enable_shared_expert_tp",
    "enable_external_shared_experts",
    "host_external_shared_experts",
    "enable_redundant_experts",
    "num_hidden_layers_override",
    "disable_repetition",
)


@dataclass(frozen=True)
class ModelWorkloadTemplate:
    """Model information required by serving optimizers, without a live model."""

    model_config: Any
    model_weight_size_gb: float


@dataclass(frozen=True)
class RuntimeWorkload:
    """Frozen, profile-independent workload and direct-path metric metadata."""

    trace: RuntimeWorkloadTrace
    model_weight_size_gb: float
    kv_cache_size_gb: float
    kv_cache_per_token_gb: float
    indexer_cache_size_gb: float
    indexer_cache_per_token_gb: float
    batch_size: int
    has_heterogeneous_prefill: bool


class WorkloadCache:
    """Single-flight LRU coordinator hosted by a ``SyncManager`` process.

    Values are serialized before being retained so an accidental Runtime or
    FakeTensor reference cannot cross process boundaries.  The class is also
    usable directly in unit tests.
    """

    def __init__(
        self,
        max_entries: int = 128,
        estimate_jobs: int = 8,
        max_bytes: int = _DEFAULT_WORKLOAD_CACHE_MAX_BYTES,
        inflight_timeout_s: float = _DEFAULT_WORKLOAD_INFLIGHT_TIMEOUT_S,
    ) -> None:
        self.max_entries = max_entries
        self.estimate_jobs = estimate_jobs
        self.max_bytes = max_bytes
        self.inflight_timeout_s = float(inflight_timeout_s)
        if self.inflight_timeout_s < 0:
            raise ValueError("inflight_timeout_s must be non-negative")
        self._templates: OrderedDict[str, bytes] = OrderedDict()
        self._workloads: OrderedDict[str, bytes] = OrderedDict()
        # Templates and workloads share one LRU and byte budget.  A template
        # is profile-independent, but a large parallel search can still create
        # unboundedly many model keys without this shared accounting.
        self._lru: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._current_bytes = 0
        # Store the claim time rather than only the key. If an owner process
        # is killed before it can publish/abandon, waiters can expire the
        # stale single-flight slot instead of waiting forever.
        self._inflight: dict[str, tuple[float, str]] = {}
        self._condition = threading.Condition()
        self.analysis_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self.wait_count = 0
        self.timeout_count = 0
        self.eviction_count = 0
        self.bypass_count = 0

    @staticmethod
    def _digest(value: Any) -> str:
        canonical = json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def make_model_key(self, user_input: Any) -> str:
        values = {field: getattr(user_input, field, None) for field in _MODEL_KEY_FIELDS}
        return self._digest(values)

    def make_workload_key(self, model_key: str, requests: list, generate_inputs_func: Callable) -> str:
        return self._digest(
            {
                "schema_version": _WORKLOAD_TRACE_SCHEMA_VERSION,
                "model_key": model_key,
                "input_generator": f"{generate_inputs_func.__module__}.{generate_inputs_func.__qualname__}",
                "requests": [asdict(request) for request in requests],
            }
        )

    def get_template(self, model_key: str) -> ModelWorkloadTemplate | None:
        with self._condition:
            serialized = self._templates.get(model_key)
            if serialized is not None:
                self._touch_locked("template", model_key)
        return self._deserialize(serialized) if serialized is not None else None

    def set_template(self, model_key: str, runner: ModelRunner) -> ModelWorkloadTemplate:
        template = ModelWorkloadTemplate(
            model_config=runner.model.model_config,
            model_weight_size_gb=runner.model_weight_size_gb,
        )
        serialized = pickle.dumps(template, protocol=pickle.HIGHEST_PROTOCOL)
        with self._condition:
            existing = self._templates.get(model_key)
            if existing is not None:
                self._touch_locked("template", model_key)
                return self._deserialize(existing)
            if len(serialized) > self.max_bytes:
                logger.debug(
                    "Workload template reuse bypass for %s: serialized template is %.1f MiB (limit %.1f MiB)",
                    model_key,
                    len(serialized) / 1024**2,
                    self.max_bytes / 1024**2,
                )
                return template
            self._templates[model_key] = serialized
            self._current_bytes += len(serialized)
            self._touch_locked("template", model_key)
            self._evict_locked()
            retained = self._templates.get(model_key)
            return self._deserialize(retained) if retained is not None else template

    @staticmethod
    def _deserialize(serialized: bytes) -> Any:
        """Deserialize data produced by this process-local coordinator only."""
        return pickle.loads(serialized)  # nosec B301 - bytes never originate outside this coordinator.

    def _touch_locked(self, entry_type: str, key: str) -> None:
        entry = (entry_type, key)
        self._lru.pop(entry, None)
        self._lru[entry] = None

    def _evict_locked(self) -> None:
        while len(self._lru) > self.max_entries or self._current_bytes > self.max_bytes:
            (entry_type, key), _ = self._lru.popitem(last=False)
            entries = self._templates if entry_type == "template" else self._workloads
            serialized = entries.pop(key, None)
            if serialized is not None:
                self._current_bytes -= len(serialized)
                self.eviction_count += 1

    def claim_workload(self, workload_key: str) -> tuple[str, RuntimeWorkload | None, str | None]:
        """Return ``hit``, ``owner`` or ``wait`` and an owner token for misses."""
        with self._condition:
            serialized = self._workloads.get(workload_key)
            if serialized is not None:
                self._workloads.move_to_end(workload_key)
                self._touch_locked("workload", workload_key)
                self.hit_count += 1
                return "hit", self._deserialize(serialized), None
            if self._expire_inflight_locked(workload_key):
                logger.warning("Expired stale workload capture lease for %s", workload_key)
            if workload_key in self._inflight:
                self.wait_count += 1
                return "wait", None, None
            owner_token = uuid.uuid4().hex
            self._inflight[workload_key] = (time.monotonic(), owner_token)
            self.miss_count += 1
            return "owner", None, owner_token

    def _expire_inflight_locked(self, workload_key: str, now: float | None = None) -> bool:
        lease = self._inflight.get(workload_key)
        if lease is None:
            return False
        started_at, _ = lease
        now = time.monotonic() if now is None else now
        if now - started_at < self.inflight_timeout_s:
            return False
        self._inflight.pop(workload_key, None)
        self.timeout_count += 1
        self.bypass_count += 1
        self._condition.notify_all()
        return True

    def wait_workload(self, workload_key: str) -> RuntimeWorkload | None:
        """Wait for a capture, returning ``None`` after a stale lease expires."""
        with self._condition:
            while True:
                if self._expire_inflight_locked(workload_key):
                    logger.warning(
                        "Workload capture lease timed out for %s after %.1f seconds; using direct path",
                        workload_key,
                        self.inflight_timeout_s,
                    )
                    break
                lease = self._inflight.get(workload_key)
                if lease is None:
                    break
                started_at, _ = lease
                remaining = self.inflight_timeout_s - (time.monotonic() - started_at)
                self._condition.wait(timeout=max(remaining, 0.0))
            serialized = self._workloads.get(workload_key)
            if serialized is None:
                return None
            self._workloads.move_to_end(workload_key)
            self._touch_locked("workload", workload_key)
            self.hit_count += 1
            return self._deserialize(serialized)

    def publish_workload(self, workload_key: str, workload: RuntimeWorkload, owner_token: str) -> bool:
        serialized = pickle.dumps(workload, protocol=pickle.HIGHEST_PROTOCOL)
        with self._condition:
            lease = self._inflight.get(workload_key)
            if lease is None or lease[1] != owner_token:
                logger.debug("Discarding workload publication from expired owner for %s", workload_key)
                return False
            self.analysis_count += 1
            if len(serialized) > self.max_bytes:
                self._inflight.pop(workload_key, None)
                self.bypass_count += 1
                self._condition.notify_all()
                logger.debug(
                    "Workload reuse bypass for %s: serialized trace is %.1f MiB (limit %.1f MiB)",
                    workload_key,
                    len(serialized) / 1024**2,
                    self.max_bytes / 1024**2,
                )
                return False
            self._store_workload_locked(workload_key, serialized)
            self._inflight.pop(workload_key, None)
            self._condition.notify_all()
            return workload_key in self._workloads

    def _store_workload_locked(self, workload_key: str, serialized: bytes) -> None:
        previous = self._workloads.pop(workload_key, None)
        if previous is not None:
            self._current_bytes -= len(previous)
        self._workloads[workload_key] = serialized
        self._current_bytes += len(serialized)
        self._touch_locked("workload", workload_key)
        self._evict_locked()

    def abandon_workload(self, workload_key: str, reason: str, owner_token: str) -> bool:
        with self._condition:
            lease = self._inflight.get(workload_key)
            if lease is None or lease[1] != owner_token:
                logger.debug("Ignoring workload abandonment from expired owner for %s", workload_key)
                return False
            self._inflight.pop(workload_key, None)
            self.bypass_count += 1
            self._condition.notify_all()
        logger.debug("Workload reuse bypass for %s: %s", workload_key, reason)
        return True

    # Compatibility helpers used by lightweight unit tests and callers that do
    # not need single-flight reservation.
    def get_workload(self, workload_key: str) -> RuntimeWorkload | None:
        state, workload, owner_token = self.claim_workload(workload_key)
        if state == "owner":
            self.abandon_workload(workload_key, "read-only cache lookup", owner_token)
            return None
        return workload if state == "hit" else self.wait_workload(workload_key)

    def set_workload(self, workload_key: str, workload: RuntimeWorkload) -> None:
        serialized = pickle.dumps(workload, protocol=pickle.HIGHEST_PROTOCOL)
        with self._condition:
            # This compatibility helper is used when callers have already
            # completed a workload analysis and need to seed the cache.
            # Keep its accounting aligned with ``publish_workload``.
            self.analysis_count += 1
            if len(serialized) > self.max_bytes:
                self.bypass_count += 1
                return
            self._store_workload_locked(workload_key, serialized)

    def summary(self) -> str:
        cache_size_mib = self._current_bytes / 1024**2
        return (
            f"Workload cache: enabled, capture_jobs=1, estimate_jobs={self.estimate_jobs}, "
            f"analysis={self.analysis_count}, hit={self.hit_count}, miss={self.miss_count}, wait={self.wait_count}, "
            f"bypass={self.bypass_count}, timeout={self.timeout_count}, evicted={self.eviction_count}, "
            f"entries={len(self._lru)} (workloads={len(self._workloads)}, templates={len(self._templates)}), "
            f"bytes={cache_size_mib:.1f} MiB"
        )


class WorkloadCacheManager(SyncManager):
    """Manager exposing a process-safe :class:`WorkloadCache` coordinator."""


WorkloadCacheManager.register("WorkloadCache", WorkloadCache)  # pylint: disable=no-member


def create_workload_cache_manager(
    estimate_jobs: int,
    max_entries: int = 128,
    inflight_timeout_s: float = _DEFAULT_WORKLOAD_INFLIGHT_TIMEOUT_S,
) -> tuple[WorkloadCacheManager, Any]:
    """Start a coordinator and return it with its shareable cache proxy."""
    manager = WorkloadCacheManager()
    manager.start()
    return manager, manager.WorkloadCache(  # pylint: disable=no-member
        max_entries=max_entries,
        estimate_jobs=estimate_jobs,
        inflight_timeout_s=inflight_timeout_s,
    )


class WorkloadReuseModelRunner:
    """ModelRunner-compatible adapter that captures once and estimates per device."""

    def __init__(self, user_input, workload_cache, model_key: str, capture_runner: ModelRunner | None = None) -> None:
        self.user_input = user_input
        self.device_profile = DeviceProfile.all_device_profiles[user_input.device]
        self.total_device_memory_gb = self.device_profile.memory_size_bytes / 1024**3
        self._cache = workload_cache
        self._model_key = model_key
        self._capture_runner = capture_runner
        template = workload_cache.get_template(model_key)
        if capture_runner is not None:
            try:
                template = workload_cache.set_template(model_key, capture_runner)
            except (pickle.PickleError, TypeError, AttributeError) as error:
                logger.debug("Workload template is not portable: %s", error)
        if template is None and capture_runner is None:
            raise ValueError("A workload reuse runner requires a cached model template or capture runner.")
        self.model = (
            capture_runner.model if capture_runner is not None else SimpleNamespace(model_config=template.model_config)
        )
        self.model_weight_size_gb = (
            capture_runner.model_weight_size_gb if capture_runner is not None else template.model_weight_size_gb
        )
        self.perf_models = (
            capture_runner.perf_models
            if capture_runner is not None
            else ModelRunner.create_performance_models(user_input, self.device_profile)
        )

    def _ensure_capture_runner(self) -> ModelRunner:
        if self._capture_runner is None:
            self._capture_runner = ModelRunner(self.user_input)
            self.model = self._capture_runner.model
            self.model_weight_size_gb = self._capture_runner.model_weight_size_gb
            self.perf_models = self._capture_runner.perf_models
            self._cache.set_template(self._model_key, self._capture_runner)
        return self._capture_runner

    def run_inference(self, requests=None, generate_inputs_func=None, with_sampler=False, runtime_observer=None):
        if requests is None or generate_inputs_func is None or with_sampler:
            return self._ensure_capture_runner().run_inference(
                requests,
                generate_inputs_func=generate_inputs_func,
                with_sampler=with_sampler,
                runtime_observer=runtime_observer,
            )

        workload_key = self._cache.make_workload_key(self._model_key, requests, generate_inputs_func)
        state, workload, owner_token = self._cache.claim_workload(workload_key)
        if state == "hit":
            return self._estimate(workload, runtime_observer)
        if state == "wait":
            workload = self._cache.wait_workload(workload_key)
            if workload is not None:
                return self._estimate(workload, runtime_observer)
            return self._ensure_capture_runner().run_inference(
                requests,
                generate_inputs_func=generate_inputs_func,
                with_sampler=False,
                runtime_observer=runtime_observer,
            )

        observed_runtimes = []

        def observer(runtime):
            observed_runtimes.append(runtime)
            if runtime_observer is not None:
                runtime_observer(runtime)

        metrics = None
        owner_slot_released = False
        try:
            metrics = self._ensure_capture_runner().run_inference(
                requests,
                generate_inputs_func=generate_inputs_func,
                with_sampler=False,
                runtime_observer=observer,
            )
            if len(observed_runtimes) != 1:
                raise WorkloadFreezeError("expected exactly one Runtime observer event")
            workload = RuntimeWorkload(
                trace=RuntimeWorkloadTrace.from_runtime(observed_runtimes[0]),
                model_weight_size_gb=metrics.model_weight_size_gb,
                kv_cache_size_gb=metrics.kv_cache_size_gb,
                kv_cache_per_token_gb=metrics.kv_cache_per_token_gb,
                indexer_cache_size_gb=metrics.indexer_cache_size_gb,
                indexer_cache_per_token_gb=metrics.indexer_cache_per_token_gb,
                batch_size=metrics.batch_size,
                has_heterogeneous_prefill=(
                    len({request.query_len for request in requests if not request.is_decode}) > 1
                ),
            )
            self._cache.publish_workload(workload_key, workload, owner_token)
            owner_slot_released = True
            return metrics
        except WorkloadFreezeError as error:
            self._cache.abandon_workload(workload_key, str(error), owner_token)
            owner_slot_released = True
            # The direct capture has already produced an authoritative result.
            # Do not execute a second forward merely because it was not portable.
            if metrics is not None:
                return metrics
            return self._ensure_capture_runner().run_inference(
                requests,
                generate_inputs_func=generate_inputs_func,
                with_sampler=False,
                runtime_observer=runtime_observer,
            )
        except Exception:
            self._cache.abandon_workload(workload_key, "capture failed", owner_token)
            owner_slot_released = True
            raise
        finally:
            # ``Exception`` does not include process-level interruptions such
            # as ``KeyboardInterrupt``/``SystemExit``. Release the slot for
            # those paths too; the lease in WorkloadCache remains the final
            # safeguard when the worker is killed outright.
            if not owner_slot_released:
                try:
                    self._cache.abandon_workload(workload_key, "capture interrupted", owner_token)
                except Exception:  # pragma: no cover - manager may be gone
                    logger.debug("Unable to release workload capture slot", exc_info=True)

    def _estimate(self, workload: RuntimeWorkload, runtime_observer=None) -> ModelRunnerMetrics:
        estimate_start = time.perf_counter()
        runtime = Runtime(
            self.perf_models,
            self.device_profile,
            memory_tracker=MemoryTracker(self.device_profile),
        )
        workload.trace.replay(runtime)
        runtime.memory_tracker.analyze()
        if self.user_input.chrome_trace:
            runtime.export_chrome_trace(self.user_input.chrome_trace)
        if runtime_observer is not None:
            runtime_observer(runtime)

        peak_memory_usage_gb = runtime.memory_tracker.peak_mem_usage() / 1024**3
        model_activation_size_gb = peak_memory_usage_gb - workload.kv_cache_size_gb - workload.model_weight_size_gb
        device_memory_available_gb = (
            self.total_device_memory_gb - peak_memory_usage_gb - self.user_input.reserved_memory_gb
        )
        if device_memory_available_gb < 0 and workload.has_heterogeneous_prefill:
            peak_memory_usage_gb = (
                workload.model_weight_size_gb + workload.kv_cache_size_gb
                if model_activation_size_gb <= 0
                else 0.1 * model_activation_size_gb + workload.model_weight_size_gb + workload.kv_cache_size_gb
            )
            model_activation_size_gb = peak_memory_usage_gb - workload.kv_cache_size_gb - workload.model_weight_size_gb
            device_memory_available_gb = (
                self.total_device_memory_gb - peak_memory_usage_gb - self.user_input.reserved_memory_gb
            )
        if model_activation_size_gb < 0:
            model_activation_size_gb = 0.0
            peak_memory_usage_gb = workload.model_weight_size_gb + workload.kv_cache_size_gb
            device_memory_available_gb = (
                self.total_device_memory_gb - peak_memory_usage_gb - self.user_input.reserved_memory_gb
            )

        execution_time_s = runtime.total_execution_time_s()
        tps_per_model = {
            model_name: (self.user_input.num_queries * self.user_input.query_len)
            / (duration * self.user_input.world_size)
            for model_name, duration in execution_time_s.items()
            if duration and duration > 0
        }
        perf_model_name = self.perf_models[0].name if self.perf_models else None
        table_result = runtime.table_averages(
            group_by_input_shapes=self.user_input.dump_input_shapes,
            dump_op_bound_results=self.user_input.dump_op_bound_results,
        )
        breakdowns = runtime.get_breakdowns()
        runtime_event_list = ModelRunner._aggregate_runtime_events(
            None,
            runtime.event_list,
            perf_model_name=perf_model_name,
        )
        run_time_s = time.perf_counter() - estimate_start
        return ModelRunnerMetrics(
            total_device_memory_gb=self.total_device_memory_gb,
            model_weight_size_gb=workload.model_weight_size_gb,
            peak_memory_usage_gb=peak_memory_usage_gb,
            kv_cache_size_gb=workload.kv_cache_size_gb,
            kv_cache_per_token_gb=workload.kv_cache_per_token_gb,
            indexer_cache_size_gb=workload.indexer_cache_size_gb,
            indexer_cache_per_token_gb=workload.indexer_cache_per_token_gb,
            model_activation_size_gb=model_activation_size_gb,
            reserved_memory_gb=self.user_input.reserved_memory_gb,
            device_memory_available_gb=device_memory_available_gb,
            execution_time_s=execution_time_s,
            tps_per_model=tps_per_model,
            run_time_s=run_time_s,
            batch_size=workload.batch_size,
            table_result=table_result,
            breakdowns=breakdowns,
            runtime_event_list=runtime_event_list,
            perf_model_name=perf_model_name,
        )
