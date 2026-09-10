# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
"""Built-in P/D service search and ratio recommendation."""

from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from ..config.base_config import CONCURRENCYS, REQUESTRATES
from ..config.config import (
    DataStorageConfig,
    PdDisaggConfig,
    PdDisaggPhaseConfig,
    PerformanceIndex,
    Settings,
)
from ..io_utils import open_file
from .errors import PdDisaggError
from .outcome import OptimizationResult

PREFILL_PHASE = "prefill"
DECODE_PHASE = "decode"
PHASES = (PREFILL_PHASE, DECODE_PHASE)


@dataclass(frozen=True)
class PhaseResult:
    name: str
    fitness: float
    best_params: dict[str, Any]
    performance_index: PerformanceIndex
    csv_path: str
    qps: float | None = None
    qps_source: str | None = None
    service_params: dict[str, Any] | None = None
    benchmark_params: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fitness": self.fitness,
            "best_params": self.best_params,
            "performance_index": self.performance_index.model_dump(mode="json"),
            "csv_path": self.csv_path,
            "qps": self.qps,
            "qps_source": self.qps_source,
            "service_params": self.service_params,
            "benchmark_params": self.benchmark_params,
        }


@dataclass(frozen=True)
class InstanceAllocation:
    prefill: int
    decode: int
    prefill_devices: int
    decode_devices: int
    used_devices: int
    remaining_devices: int
    actual_ratio: float
    ratio_error: float
    balanced_qps: float


class PhaseRunner(Protocol):
    def run(
        self,
        phase_name: str,
        phase_config: PdDisaggPhaseConfig,
        phase_output_dir: Path,
    ) -> tuple[OptimizationResult, Path]: ...


def _require_positive(value: Any, description: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise PdDisaggError(f"{description} is missing or invalid") from error
    if not isfinite(number) or number <= 0:
        raise PdDisaggError(f"{description} must be a finite number greater than 0")
    return number


def calculate_phase_qps(
    phase_name: str,
    phase: PdDisaggPhaseConfig,
    result: OptimizationResult,
) -> tuple[float, str]:
    """Return benchmark request throughput as phase QPS."""

    del phase  # Kept in the signature for compatibility with existing callers.
    if phase_name not in PHASES:
        raise ValueError(f"QPS is not defined for phase '{phase_name}'")
    throughput = _require_positive(result.performance_index.throughput, f"{phase_name} throughput")
    return throughput, "benchmark_throughput"


def calculate_pd_ratio(prefill_qps: float, decode_qps: float) -> float:
    return _require_positive(decode_qps, "decode QPS") / _require_positive(prefill_qps, "prefill QPS")


def allocate_instances(
    *,
    total_devices: int,
    prefill_devices_per_instance: int,
    decode_devices_per_instance: int,
    pd_ratio: float,
    prefill_qps: float,
    decode_qps: float,
    use_full_device: bool,
    top_k: int,
) -> list[InstanceAllocation]:
    """Enumerate resource-feasible P/D allocations in recommendation order."""

    if total_devices <= 0:
        return []
    _require_positive(prefill_devices_per_instance, "prefill devices per instance")
    _require_positive(decode_devices_per_instance, "decode devices per instance")
    pd_ratio = _require_positive(pd_ratio, "PD ratio")
    candidates: list[InstanceAllocation] = []
    max_prefill = total_devices // prefill_devices_per_instance
    max_decode = total_devices // decode_devices_per_instance
    for prefill in range(1, max_prefill + 1):
        for decode in range(1, max_decode + 1):
            prefill_devices = prefill * prefill_devices_per_instance
            decode_devices = decode * decode_devices_per_instance
            used_devices = prefill_devices + decode_devices
            if used_devices > total_devices:
                continue
            if use_full_device and used_devices != total_devices:
                continue
            actual_ratio = prefill / decode
            candidates.append(
                InstanceAllocation(
                    prefill=prefill,
                    decode=decode,
                    prefill_devices=prefill_devices,
                    decode_devices=decode_devices,
                    used_devices=used_devices,
                    remaining_devices=total_devices - used_devices,
                    actual_ratio=actual_ratio,
                    ratio_error=abs(actual_ratio - pd_ratio),
                    balanced_qps=min(prefill * prefill_qps, decode * decode_qps),
                )
            )
    candidates.sort(key=lambda item: (item.ratio_error, -item.balanced_qps, -item.used_devices))
    return candidates[:top_k]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    with open_file(path, "w") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_candidate_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    with open_file(path, "w") as file:
        fieldnames: list[str] = []
        for row in rows:
            for name in row:
                if name not in fieldnames:
                    fieldnames.append(name)
        writer = csv.DictWriter(file, fieldnames=fieldnames or ["pd_ratio"])
        writer.writeheader()
        writer.writerows(rows)


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _phase_qps_candidates(
    phase_name: str,
    phase: PdDisaggPhaseConfig,
    selected: PhaseResult,
    top_k: int,
) -> list[dict[str, Any]]:
    """Load the top phase CSV rows, always retaining the optimizer-selected result first."""

    candidates = [
        {
            "fitness": selected.fitness,
            "qps": selected.qps,
            "qps_source": selected.qps_source,
            "concurrency": selected.best_params.get(phase.concurrency_field),
            "selected": True,
        }
    ]
    csv_path = Path(selected.csv_path)
    if top_k <= len(candidates) or not csv_path.is_file():
        candidates[0]["rank"] = 1
        return candidates
    with open_file(csv_path, "r") as file:
        rows = list(csv.DictReader(file))
    rows.sort(
        key=lambda row: (
            float("inf") if _optional_float(row.get("fitness")) is None else _optional_float(row.get("fitness"))
        )
    )
    for row in rows:
        fitness = _optional_float(row.get("fitness"))
        if fitness is None:
            continue
        concurrency = None
        concurrency_name = None
        for name in (phase.concurrency_field, "CONCURRENCY", "MAXCONCURRENCY"):
            concurrency = _optional_float(row.get(name))
            if concurrency is not None:
                concurrency_name = name
                break
        performance = PerformanceIndex(
            generate_speed=_optional_float(row.get("generate_speed")),
            throughput=_optional_float(row.get("throughput")),
            time_to_first_token=_optional_float(row.get("time_to_first_token")),
            time_per_output_token=_optional_float(row.get("time_per_output_token")),
            success_rate=_optional_float(row.get("success_rate")),
        )
        optimization = OptimizationResult(
            fitness=fitness,
            params={} if concurrency_name is None else {concurrency_name: concurrency},
            performance_index=performance,
        )
        try:
            qps, source = calculate_phase_qps(phase_name, phase, optimization)
        except PdDisaggError:
            continue
        candidate = {
            "fitness": fitness,
            "qps": qps,
            "qps_source": source,
            "concurrency": concurrency,
            "selected": False,
        }
        identity = (candidate["fitness"], candidate["qps"], candidate["concurrency"])
        if any((item["fitness"], item["qps"], item["concurrency"]) == identity for item in candidates):
            continue
        candidates.append(candidate)
        if len(candidates) >= top_k:
            break
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
    return candidates


def _ratio_candidate_rows(
    prefill_candidates: list[dict[str, Any]],
    decode_candidates: list[dict[str, Any]],
    config: PdDisaggConfig,
) -> list[dict[str, Any]]:
    rows = []
    for prefill in prefill_candidates:
        for decode in decode_candidates:
            ratio = calculate_pd_ratio(prefill["qps"], decode["qps"])
            allocations = allocate_instances(
                total_devices=config.total_devices,
                prefill_devices_per_instance=config.prefill_devices_per_instance,
                decode_devices_per_instance=config.decode_devices_per_instance,
                pd_ratio=ratio,
                prefill_qps=prefill["qps"],
                decode_qps=decode["qps"],
                use_full_device=config.use_full_device,
                top_k=1,
            )
            row = {
                "prefill_rank": prefill["rank"],
                "decode_rank": decode["rank"],
                "prefill_qps": prefill["qps"],
                "decode_qps": decode["qps"],
                "pd_ratio": ratio,
                "prefill_fitness": prefill["fitness"],
                "decode_fitness": decode["fitness"],
                "selected_pair": prefill["selected"] and decode["selected"],
            }
            if allocations:
                row.update(asdict(allocations[0]))
            rows.append(row)
    rows.sort(
        key=lambda row: (
            not row["selected_pair"],
            row["prefill_rank"] + row["decode_rank"],
            row["prefill_fitness"] + row["decode_fitness"],
        )
    )
    return rows


@contextmanager
def _phase_settings(settings: Settings, phase: PdDisaggPhaseConfig):
    names = (
        "ttft_penalty",
        "tpot_penalty",
        "success_rate_penalty",
        "ttft_slo",
        "tpot_slo",
        "success_rate_slo",
        "fine_tune_mode",
    )
    previous = {name: getattr(settings, name) for name in names}
    for name in names:
        setattr(settings, name, getattr(phase, name))
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(settings, name, value)


class OptixPhaseRunner:
    """Adapter that executes one phase with existing OptiX plugins and PSO."""

    def __init__(
        self,
        settings: Settings,
        *,
        runtime_ctx: Any,
        deploy_env: dict[str, str],
        bak_path: Path | None,
        load_breakpoint: bool,
        top_k: int,
    ) -> None:
        self.settings = settings
        self.runtime_ctx = runtime_ctx
        self.deploy_env = deploy_env
        self.bak_path = bak_path
        self.load_breakpoint = load_breakpoint
        self.top_k = top_k

    def _resolve_and_validate_plugins(
        self,
        phase_name: str,
        phase_config: PdDisaggPhaseConfig,
    ) -> tuple[type[Any], type[Any]]:
        from ..deploy_env import validate_deploy_stack
        from .register import (
            benchmarks,
            simulates,
            validate_benchmark_policy,
            validate_simulator_policy,
        )

        simulator_cls = simulates.get(phase_config.engine)
        benchmark_cls = benchmarks.get(phase_config.benchmark_policy)
        if simulator_cls is None:
            raise PdDisaggError(f"PD phase '{phase_name}' simulator plugin '{phase_config.engine}' is not registered")
        if benchmark_cls is None:
            raise PdDisaggError(
                f"PD phase '{phase_name}' benchmark plugin '{phase_config.benchmark_policy}' is not registered"
            )
        validate_simulator_policy(phase_config.engine)
        validate_benchmark_policy(phase_config.benchmark_policy)
        validate_deploy_stack(
            engine=phase_config.engine,
            benchmark=phase_config.benchmark_policy,
            env=self.deploy_env,
            ctx=self.runtime_ctx,
        )
        return simulator_cls, benchmark_cls

    def preflight(self, config: PdDisaggConfig) -> None:
        """Validate P/D plugins before starting an expensive real search."""

        for phase_name in PHASES:
            phase = getattr(config, phase_name)
            self._resolve_and_validate_plugins(phase_name, phase)

    @staticmethod
    def _configure_phase_hook(plugin: Any, phase_name: str, phase_config: PdDisaggPhaseConfig) -> None:
        hook = getattr(plugin, "configure_pd_phase", None)
        if callable(hook):
            hook(phase_name=phase_name, phase_config=phase_config)

    @staticmethod
    def _apply_command_overrides(
        plugin: Any,
        phase_name: str,
        plugin_role: str,
        overrides: dict[str, Any],
    ) -> None:
        """Apply a phase-local command overlay without naming a plugin implementation."""

        if not overrides:
            return
        config = getattr(plugin, "config", None)
        command = getattr(config, "command", None)
        if command is None:
            raise PdDisaggError(f"PD phase '{phase_name}' {plugin_role} does not expose a configurable command")
        model_fields = getattr(type(command), "model_fields", None)
        available_fields = set(model_fields) if model_fields is not None else set(vars(command))
        unknown_fields = sorted(set(overrides) - available_fields)
        if unknown_fields:
            raise PdDisaggError(f"PD phase '{phase_name}' {plugin_role} command has no fields: {unknown_fields}")

        local_config = deepcopy(config)
        local_command = local_config.command
        model_dump = getattr(local_command, "model_dump", None)
        model_validate = getattr(type(local_command), "model_validate", None)
        try:
            if callable(model_dump) and callable(model_validate):
                command_data = model_dump()
                command_data.update(deepcopy(overrides))
                local_config.command = model_validate(command_data)
            else:
                for name, value in overrides.items():
                    setattr(local_command, name, deepcopy(value))
        except (TypeError, ValueError) as error:
            raise PdDisaggError(
                f"Invalid {plugin_role} command override for PD phase '{phase_name}': {error}"
            ) from error
        plugin.config = local_config
        update_command = getattr(plugin, "update_command", None)
        if callable(update_command):
            update_command()

    @classmethod
    def _apply_simulator_command_overrides(
        cls,
        simulator: Any,
        phase_name: str,
        phase_config: PdDisaggPhaseConfig,
    ) -> None:
        cls._apply_command_overrides(
            simulator,
            phase_name,
            "simulator",
            phase_config.simulator_command_overrides,
        )

    @classmethod
    def _apply_benchmark_command_overrides(
        cls,
        benchmark: Any,
        phase_name: str,
        phase_config: PdDisaggPhaseConfig,
    ) -> None:
        cls._apply_command_overrides(
            benchmark,
            phase_name,
            "benchmark",
            phase_config.benchmark_command_overrides,
        )

    @staticmethod
    def _set_data_fields(plugin: Any, fields: tuple[Any, ...]) -> None:
        config = getattr(plugin, "config", None)
        configured_fields = getattr(config, "target_field", None)
        if isinstance(configured_fields, list):
            # Built-in plugins usually reference the shared Settings object.
            # Filter an instance-local copy so one phase cannot alter the next
            # phase's search-space ownership.
            plugin.config = deepcopy(config)
            plugin.config.target_field = list(fields)
            return
        if hasattr(plugin, "data_field"):
            plugin.data_field = fields

    @staticmethod
    def _benchmark_load_field_names(phase: PdDisaggPhaseConfig) -> set[str]:
        """Return benchmark-agnostic load-control field names."""

        return {
            phase.concurrency_field.casefold(),
            *(name.casefold() for name in CONCURRENCYS),
            *(name.casefold() for name in REQUESTRATES),
        }

    @staticmethod
    def _deduplicate_fields(fields: tuple[Any, ...]) -> tuple[Any, ...]:
        selected = {}
        for field in fields:
            selected.setdefault(field.name.casefold(), field)
        return tuple(selected.values())

    def _resolve_target_fields(self, simulator: Any, benchmark: Any, phase: PdDisaggPhaseConfig) -> tuple[Any, ...]:
        simulator_fields = tuple(getattr(simulator, "data_field", ()) or ())
        benchmark_fields = tuple(getattr(benchmark, "data_field", ()) or ())
        load_field_names = self._benchmark_load_field_names(phase)
        declared_benchmark_names = {field.name.casefold() for field in benchmark_fields}
        benchmark_names = declared_benchmark_names | load_field_names
        if phase.target_field:
            configured = tuple(deepcopy(phase.target_field))
            selected_benchmark = tuple(field for field in configured if field.name.casefold() in benchmark_names)
            selected_simulator = tuple(field for field in configured if field.name.casefold() not in benchmark_names)
            self._set_data_fields(simulator, selected_simulator)
            self._set_data_fields(benchmark, selected_benchmark)
            return configured

        inherited_load_fields = tuple(field for field in simulator_fields if field.name.casefold() in load_field_names)
        selected_benchmark = self._deduplicate_fields((*benchmark_fields, *inherited_load_fields))
        selected_benchmark_names = {field.name.casefold() for field in selected_benchmark}
        selected_simulator = tuple(
            field for field in simulator_fields if field.name.casefold() not in selected_benchmark_names
        )
        self._set_data_fields(simulator, selected_simulator)
        self._set_data_fields(benchmark, selected_benchmark)
        return (*selected_simulator, *selected_benchmark)

    def _resolve_search_budget(
        self,
        phase_config: PdDisaggPhaseConfig,
    ) -> tuple[int, int]:
        n_particles = phase_config.n_particles if phase_config.n_particles is not None else self.settings.n_particles
        iters = phase_config.iters if phase_config.iters is not None else self.settings.iters
        return n_particles, iters

    def run(
        self,
        phase_name: str,
        phase_config: PdDisaggPhaseConfig,
        phase_output_dir: Path,
    ) -> tuple[OptimizationResult, Path]:
        if not self.settings.manage_simulator_lifecycle:
            raise PdDisaggError(
                f"PD phase '{phase_name}' requires manage_simulator_lifecycle=true to manage the service during search"
            )
        from .experience_fine_tunning import FineTune
        from .optimizer import PSOOptimizer
        from .scheduler import Scheduler
        from .store import DataStorage

        simulator_cls, benchmark_cls = self._resolve_and_validate_plugins(phase_name, phase_config)
        simulator = simulator_cls(
            bak_path=self.bak_path,
            runtime_ctx=self.runtime_ctx,
            deploy_env=self.deploy_env,
        )
        benchmark = benchmark_cls(
            bak_path=self.bak_path,
            runtime_ctx=self.runtime_ctx,
            deploy_env=self.deploy_env,
        )
        self._apply_simulator_command_overrides(simulator, phase_name, phase_config)
        self._configure_phase_hook(simulator, phase_name, phase_config)
        self._apply_benchmark_command_overrides(benchmark, phase_name, phase_config)
        self._configure_phase_hook(benchmark, phase_name, phase_config)
        target_fields = self._resolve_target_fields(simulator, benchmark, phase_config)
        if not target_fields:
            raise PdDisaggError(f"PD phase '{phase_name}' has no optimization fields")
        if self.top_k > 0 and phase_config.fine_tune_mode == "pd_disaggregation":
            try:
                FineTune.validate_pd_disaggregation_fields(target_fields)
            except ValueError as error:
                raise PdDisaggError(f"PD phase '{phase_name}' has invalid fine-tune fields: {error}") from error
        service_param_names = tuple(field.name for field in getattr(simulator, "data_field", ()) or ())
        benchmark_param_names = tuple(field.name for field in getattr(benchmark, "data_field", ()) or ())

        storage_config = DataStorageConfig(store_dir=phase_output_dir, pso_top_k=self.top_k)
        data_storage = DataStorage(storage_config, simulator, benchmark)
        benchmark_run_count = phase_config.benchmark_run_count
        if self.settings.use_request_rate_calibration and benchmark_run_count > 1:
            raise PdDisaggError(
                f"PD phase '{phase_name}' benchmark_run_count={benchmark_run_count} requires "
                "use_request_rate_calibration=false"
            )
        scheduler = Scheduler(
            simulator,
            benchmark,
            data_storage,
            bak_path=self.bak_path,
            wait_start_time=self.settings.wait_start_time,
            engine=phase_config.engine,
            benchmark_run_count=benchmark_run_count,
        )
        fine_tune = FineTune(
            ttft_penalty=phase_config.ttft_penalty,
            tpot_penalty=phase_config.tpot_penalty,
            target_field=target_fields,
            ttft_slo=phase_config.ttft_slo,
            tpot_slo=phase_config.tpot_slo,
            slo_coefficient=self.settings.slo_coefficient,
            step_size=self.settings.step_size,
            fine_tune_mode=phase_config.fine_tune_mode,
        )
        n_particles, iters = self._resolve_search_budget(phase_config)
        optimizer = PSOOptimizer(
            scheduler,
            n_particles=n_particles,
            iters=iters,
            target_field=target_fields,
            ttft_penalty=phase_config.ttft_penalty,
            tpot_penalty=phase_config.tpot_penalty,
            success_rate_penalty=phase_config.success_rate_penalty,
            ttft_slo=phase_config.ttft_slo,
            tpot_slo=phase_config.tpot_slo,
            success_rate_slo=phase_config.success_rate_slo,
            generate_speed_target=self.settings.generate_speed_target,
            load_breakpoint=self.load_breakpoint,
            fine_tune=fine_tune,
            max_fine_tune=self.settings.max_fine_tune,
            skip_pso=self.settings.skip_pso,
            use_request_rate_calibration=self.settings.use_request_rate_calibration,
            manage_simulator_lifecycle=self.settings.manage_simulator_lifecycle,
            pso_init_kwargs={"ftol": self.settings.ftol, "ftol_iter": self.settings.ftol_iter},
        )
        with _phase_settings(self.settings, phase_config):
            result = optimizer.run_plugin()
        return replace(
            result,
            service_param_names=service_param_names,
            benchmark_param_names=benchmark_param_names,
        ), data_storage.save_file


class PdDisaggOrchestrator:
    """Run P/D service searches and recommend an instance ratio."""

    def __init__(self, settings: Settings, run_id: str, phase_runner: PhaseRunner) -> None:
        self.settings = settings
        self.config: PdDisaggConfig = settings.pd_disagg
        self.run_id = run_id
        self.phase_runner = phase_runner
        self.run_dir = settings.output.joinpath(self.config.phase_output_dir, self.run_id)
        self.summary_path = self.run_dir.joinpath("pd_disagg_summary.json")

    def _phase_config(self, phase_name: str) -> PdDisaggPhaseConfig:
        return getattr(self.config, phase_name)

    def _phase_result_path(self, phase_name: str) -> Path:
        return self.run_dir.joinpath(phase_name, "phase_result.json")

    @staticmethod
    def _summary_phase(result: PhaseResult) -> dict[str, Any]:
        performance = result.performance_index
        summary = {
            "status": "success",
            "qps": result.qps,
            "qps_source": result.qps_source,
            "ttft_s": performance.time_to_first_token,
            "tpot_s": performance.time_per_output_token,
            "throughput": performance.throughput,
            "success_rate": performance.success_rate,
            "csv": result.csv_path,
        }
        summary["best_params"] = result.best_params
        return summary

    def _run_phase(self, phase_name: str) -> PhaseResult:
        phase_dir = self.run_dir.joinpath(phase_name)
        phase_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
        optimization, csv_path = self.phase_runner.run(
            phase_name,
            self._phase_config(phase_name),
            phase_dir,
        )
        qps, qps_source = calculate_phase_qps(
            phase_name,
            self._phase_config(phase_name),
            optimization,
        )
        result = PhaseResult(
            name=phase_name,
            fitness=optimization.fitness,
            best_params=optimization.params,
            performance_index=optimization.performance_index,
            csv_path=str(csv_path),
            qps=qps,
            qps_source=qps_source,
            service_params=(
                {
                    name: optimization.params[name]
                    for name in optimization.service_param_names
                    if name in optimization.params
                }
                if optimization.service_param_names or optimization.benchmark_param_names
                else None
            ),
            benchmark_params=(
                {
                    name: optimization.params[name]
                    for name in optimization.benchmark_param_names
                    if name in optimization.params
                }
                if optimization.service_param_names or optimization.benchmark_param_names
                else None
            ),
        )
        _write_json(self._phase_result_path(phase_name), result.to_dict())
        return result

    def run(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
        results: dict[str, PhaseResult] = {}
        summary: dict[str, Any] = {
            "run_id": self.run_id,
            "status": "running",
            "artifacts": {"summary": str(self.summary_path)},
        }
        _write_json(self.summary_path, summary)
        current_phase = None
        try:
            preflight = getattr(self.phase_runner, "preflight", None)
            if callable(preflight):
                preflight(self.config)
            for phase_name in (PREFILL_PHASE, DECODE_PHASE):
                current_phase = phase_name
                summary[phase_name] = {"status": "running"}
                _write_json(self.summary_path, summary)
                results[phase_name] = self._run_phase(phase_name)
                summary[phase_name] = self._summary_phase(results[phase_name])
                summary["artifacts"][f"{phase_name}_csv"] = results[phase_name].csv_path
                _write_json(self.summary_path, summary)

            prefill = results[PREFILL_PHASE]
            decode = results[DECODE_PHASE]
            pd_ratio = calculate_pd_ratio(prefill.qps, decode.qps)
            # top_k=0 disables P/D phase FineTune. Ratio recommendation still
            # needs the optimizer-selected P/D pair and one feasible allocation.
            recommendation_top_k = max(1, self.config.top_k)
            candidates = allocate_instances(
                total_devices=self.config.total_devices,
                prefill_devices_per_instance=self.config.prefill_devices_per_instance,
                decode_devices_per_instance=self.config.decode_devices_per_instance,
                pd_ratio=pd_ratio,
                prefill_qps=prefill.qps,
                decode_qps=decode.qps,
                use_full_device=self.config.use_full_device,
                top_k=recommendation_top_k,
            )
            prefill_candidates = _phase_qps_candidates(
                PREFILL_PHASE,
                self.config.prefill,
                prefill,
                recommendation_top_k,
            )
            decode_candidates = _phase_qps_candidates(
                DECODE_PHASE,
                self.config.decode,
                decode,
                recommendation_top_k,
            )
            ratio_candidate_rows = _ratio_candidate_rows(prefill_candidates, decode_candidates, self.config)
            candidates_path = self.run_dir.joinpath("pd_ratio_candidates.csv")
            _write_candidate_rows(candidates_path, ratio_candidate_rows)
            summary["pd_ratio"] = pd_ratio
            summary["instances"] = asdict(candidates[0]) if candidates else None
            summary["artifacts"]["pd_ratio_candidates"] = str(candidates_path)
            if self.config.total_devices and not candidates:
                summary["allocation_warning"] = "No integer instance allocation satisfies the resource constraints"
            summary["status"] = "service_search_completed"
            _write_json(self.summary_path, summary)
            logger.success("PD service parameter search finished: {}", self.summary_path)
            return summary
        except Exception as error:
            summary["status"] = "failed"
            summary["failed_phase"] = current_phase
            summary["error"] = {"type": type(error).__name__, "message": str(error)}
            if current_phase:
                summary[current_phase] = {
                    "status": "failed",
                    "error": summary["error"],
                }
            _write_json(self.summary_path, summary)
            if isinstance(error, PdDisaggError):
                raise
            raise PdDisaggError(f"PD phase '{current_phase}' failed: {error}") from error
