# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Deploy PSO-selected P/D instances and refine the complete service load."""

import json
import shlex
from copy import deepcopy

from loguru import logger

from ..config.config import OptimizerConfigField
from .errors import PdDisaggError
from .optimizer import PSOOptimizer
from .pd_common import PHASES, write_json


class FullServiceOptimizer(PSOOptimizer):
    """Prefer highest request throughput among fully SLO-compliant trials."""

    def best_params(self, fitness_list, params_list, performance_list):
        feasible = [
            index
            for index, metric in enumerate(performance_list)
            if metric.success_rate is not None
            and metric.success_rate >= self.success_rate_slo
            and metric.time_to_first_token is not None
            and metric.time_to_first_token <= self.ttft_slo
            and metric.time_per_output_token is not None
            and metric.time_per_output_token <= self.tpot_slo
        ]
        if feasible:
            best = max(feasible, key=lambda index: self._performance_speed(performance_list[index]) or 0)
            return fitness_list[best], params_list[best], performance_list[best]
        return super().best_params(fitness_list, params_list, performance_list)


class OptixFullServiceRunner:
    """Own the complete PD deployment from handoff through cleanup."""

    def __init__(self, phase_runner):
        self.runner = phase_runner
        self.settings = phase_runner.settings
        self.phase = self.settings.pd_disagg.full_service
        self.cluster_config = None
        self.cleanup_errors = []

    def _try_cleanup(self, operation, callback, **kwargs):
        """Record an ordinary cleanup failure while allowing remaining actions."""
        try:
            callback(**kwargs)
        except Exception as error:
            self.cleanup_errors.append({"operation": operation, "type": type(error).__name__, "message": str(error)})
            logger.error("PD full-service cleanup failed in {}: {}: {}", operation, type(error).__name__, error)

    def _cleanup_benchmark(self, benchmark):
        restore = getattr(benchmark, "restore_service_endpoint", None)
        self._try_cleanup("benchmark.stop", benchmark.stop)
        if callable(restore):
            self._try_cleanup("benchmark.restore_service_endpoint", restore)

    def preflight(self):
        """Reject missing plugins or invalid resources before single-side searches."""
        self.runner._resolve_and_validate_plugins("full_service", self.phase)
        try:
            from vllm_pd_simulator.config import load_pd_config
            from vllm_pd_simulator.pipeline import validate_resource_pool

            self.cluster_config = load_pd_config(strict=True)
            if self.cluster_config is None:
                raise ValueError("Cannot load [vllm_pd] configuration")
            validate_resource_pool(self.cluster_config, self.settings.pd_disagg.total_devices)
            self._plans()
        except (ImportError, ValueError, KeyError) as error:
            raise PdDisaggError(f"Full-service deployment preflight failed: {error}") from error

    def _plans(self, results=None):
        from vllm_pd_simulator.pipeline import plan_role

        plans = {}
        for role in PHASES:
            command = self.settings.vllm.command.model_dump()
            command.update(getattr(self.settings.pd_disagg, role).simulator_command_overrides)
            deployment_others = getattr(self.phase, f"{role}_others")
            if deployment_others is not None:
                command["others"] = deployment_others
            additional = getattr(self.phase, f"{role}_additional_config")
            if additional:
                # Some engine settings require a real KV producer/consumer and
                # cannot be enabled during standalone service measurements.
                command["others"] = (
                    command.get("others", "") + " --additional-config " + shlex.quote(json.dumps(additional))
                )
            definitions = getattr(self.settings.pd_disagg, role).target_field or self.settings.vllm.target_field
            if results is None:
                fields = [
                    field
                    for field in definitions
                    if field.name not in {"CONCURRENCY", "MAXCONCURRENCY", "REQUESTRATE", "NUM_PROMPTS"}
                ]
            else:
                result = results[role]
                if result.service_params is None:
                    raise PdDisaggError(f"Missing service parameter handoff for {role}")
                positions = {field.name: field.config_position for field in definitions}
                fields = [
                    OptimizerConfigField.model_construct(
                        name=name,
                        value=value,
                        config_position=positions.get(name, "run"),
                    )
                    for name, value in result.service_params.items()
                ]
            plans[role] = plan_role(
                command,
                fields,
                getattr(self.settings.pd_disagg, f"{role}_devices_per_instance"),
                self.settings.vllm.env,
            )
        if plans["prefill"]["served_model_name"] != plans["decode"]["served_model_name"]:
            raise PdDisaggError("P and D must use the same served_model_name for full-service deployment")
        return plans

    def _benchmark(self, benchmark_cls, simulator, plans):
        benchmark = benchmark_cls(
            bak_path=self.runner.bak_path,
            runtime_ctx=self.runner.runtime_ctx,
            deploy_env=self.runner.deploy_env,
        )
        try:
            self.runner._apply_benchmark_command_overrides(benchmark, "full_service", self.phase)
            self.runner._configure_phase_hook(benchmark, "full_service", self.phase)
            proxy = simulator.config.proxy
            host, port = proxy.bind_ip, proxy.service_port
            model = plans["decode"]["served_model_name"]
            if self.phase.benchmark_policy == "vllm_benchmark":
                self.runner._apply_command_overrides(
                    benchmark,
                    "full_service",
                    "benchmark",
                    {
                        "host": host,
                        "port": str(port),
                        "model": plans["decode"]["model"],
                        "served_model_name": model,
                    },
                )
            else:
                bind = getattr(benchmark, "bind_service_endpoint", None)
                if not callable(bind):
                    raise PdDisaggError("Full-service benchmark must support bind_service_endpoint")
                bind(host=host, port=port, model=model)
            self.runner._set_data_fields(benchmark, tuple(deepcopy(self.phase.target_field)))
            return benchmark
        except BaseException:
            self._cleanup_benchmark(benchmark)
            raise

    def run(self, results, allocation, output_dir):
        """Freeze serving configuration, launch once, then tune benchmark fields only."""
        self.cleanup_errors = []
        if allocation is None:
            raise PdDisaggError("No resource-feasible integer P/D allocation for full-service deployment")
        plans = self._plans(results)
        simulator_cls, benchmark_cls = self.runner._resolve_and_validate_plugins("full_service", self.phase)
        simulator = simulator_cls(
            config=deepcopy(self.cluster_config),
            bak_path=self.runner.bak_path,
            runtime_ctx=self.runner.runtime_ctx,
            deploy_env=self.runner.deploy_env,
        )
        benchmark = None
        failed = False
        try:
            simulator.configure_pd_pipeline(plans, allocation, self.settings.vllm.env)
            self.runner._set_data_fields(simulator, ())
            benchmark = self._benchmark(benchmark_cls, simulator, plans)
            output_dir.mkdir(parents=True, exist_ok=True)
            pool_total = len(
                {
                    (node.ssh_ip, node.ssh_port, card)
                    for node in self.cluster_config.nodes
                    if node.role != "proxy"
                    for card in node.gpu_ids
                }
            )
            used = allocation["prefill_devices"] + allocation["decode_devices"]
            budget = self.settings.pd_disagg.total_devices
            write_json(
                output_dir / "deployment.json",
                {
                    "instances": allocation,
                    "resource_usage": {
                        "pool_total_devices": pool_total,
                        "budget_devices": budget,
                        "allocated_devices": used,
                        "budget_remaining_devices": budget - used,
                        "pool_idle_devices": pool_total - used,
                    },
                    "plans": plans,
                    "config": simulator.config.model_dump(mode="json"),
                },
            )
            result, csv_path = self.runner.optimize_phase(
                self.phase,
                output_dir,
                simulator,
                benchmark,
                tuple(benchmark.data_field),
                mode="full_service",
            )
            performance = result.performance_index
            slo_met = (
                performance.success_rate is not None
                and performance.success_rate >= self.phase.success_rate_slo
                and performance.time_to_first_token is not None
                and performance.time_to_first_token <= self.phase.ttft_slo
                and performance.time_per_output_token is not None
                and performance.time_per_output_token <= self.phase.tpot_slo
            )
            payload = {
                "status": "success",
                "slo_met": slo_met,
                "fitness": result.fitness,
                "benchmark_params": result.params,
                "performance_index": performance.model_dump(mode="json"),
                "csv": str(csv_path),
                "deployment": str(output_dir / "deployment.json"),
            }
        except BaseException:
            failed = True
            raise
        finally:
            if benchmark is not None:
                self._cleanup_benchmark(benchmark)
            self._try_cleanup("simulator.stop", simulator.stop, del_log=False)
            if self.cleanup_errors and not failed:
                details = "; ".join(
                    f"{error['operation']}: {error['type']}: {error['message']}" for error in self.cleanup_errors
                )
                raise RuntimeError(f"PD full-service cleanup failed: {details}")
        write_json(output_dir / "phase_result.json", payload)
        return payload
