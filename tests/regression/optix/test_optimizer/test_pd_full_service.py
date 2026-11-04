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

"""CPU tests of the PSO-to-PD deployment handoff and lifecycle."""

import ast
import importlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from optix.config.config import (
    OptimizerConfigField,
    PdDisaggConfig,
    PdFullServiceConfig,
    PerformanceIndex,
    Settings,
)
from optix.optimizer.aisbench_endpoint import bind_endpoint
from optix.optimizer.errors import PdDisaggError
from optix.optimizer.outcome import OptimizationResult
from optix.optimizer.pd_disagg import (
    OptixPhaseRunner,
    PdDisaggOrchestrator,
    PhaseResult,
)
from optix.optimizer.pd_full_service import FullServiceOptimizer, OptixFullServiceRunner


@pytest.fixture(scope="module")
def pd_plugin(pytestconfig):
    repo_root = pytestconfig.rootpath
    plugin_root = repo_root / "contrib/optix/vllm_pd_simulator"
    if not (plugin_root / "vllm_pd_simulator").is_dir():
        pytest.skip("PD contrib plugin sources are not available")
    cached = {
        name: module
        for name, module in sys.modules.items()
        if name == "vllm_pd_simulator" or name.startswith("vllm_pd_simulator.")
    }
    with pytest.MonkeyPatch.context() as context:
        context.syspath_prepend(str(plugin_root))
        try:
            pytest.importorskip("vllm_pd_simulator", reason="PD plugin is not available", exc_type=ModuleNotFoundError)
            yield SimpleNamespace(
                repo_root=repo_root,
                config=importlib.import_module("vllm_pd_simulator.config"),
                pipeline=importlib.import_module("vllm_pd_simulator.pipeline"),
            )
        finally:
            for name in tuple(sys.modules):
                if name == "vllm_pd_simulator" or name.startswith("vllm_pd_simulator."):
                    if name not in cached:
                        del sys.modules[name]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("qwen3_5_moe_397b_a17b", True),
        ("qwen3_5_27b", False),
        ("qwen3_moe_30b_a3b", True),
        ("qwen3_dense_8b", False),
    ],
)
def test_pd_model_detection_from_model_configs(model, expected, pd_plugin):
    from vllm_pd_simulator.tools.model_tools import detect_is_moe

    model_path = pd_plugin.repo_root / "tests/assets/model_config" / model
    assert (model_path / "config.json").is_file()
    assert detect_is_moe(str(model_path)) is expected


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"text_config": {"num_experts_per_tok": 2}}, True),
        ({"num_local_experts": 8, "text_config": None}, True),
        ({"text_config": None}, False),
        ({"text_config": "num_experts_per_tok"}, False),
        ({"vision_config": {"num_experts_per_tok": 2}}, False),
    ],
)
@pytest.mark.usefixtures("pd_plugin")
def test_pd_model_detection_checks_only_language_config(tmp_path, config, expected):
    from vllm_pd_simulator.tools.model_tools import detect_is_moe

    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert detect_is_moe(str(tmp_path)) is expected


def load_fields():
    return [
        OptimizerConfigField(name=name, config_position="env", dtype="int", min=1, max=32, value=4)
        for name in ("CONCURRENCY", "REQUESTRATE")
    ]


def settings(tmp_path):
    return Settings(
        output=tmp_path,
        use_request_rate_calibration=False,
        pd_disagg=PdDisaggConfig(
            total_devices=12,
            prefill_devices_per_instance=4,
            decode_devices_per_instance=4,
            top_k=0,
            full_service=PdFullServiceConfig(enabled=True, target_field=load_fields()),
        ),
    )


def cluster(pd_plugin):
    return pd_plugin.config.VLLMPDDisaggConfig(
        nodes=[
            pd_plugin.config.ClusterNodeConfig(
                ssh_ip="192.0.2.10", bind_ip="192.0.2.10", gpu_ids=list(range(12)), role="auto"
            )
        ],
        proxy=pd_plugin.config.ClusterNodeConfig(ssh_ip="192.0.2.10", bind_ip="192.0.2.10", service_port=8000),
    )


def phase_results():
    return {
        role: PhaseResult(
            name=role,
            fitness=1,
            best_params={"MAX_NUM_SEQS": value, "CONCURRENCY": 99},
            service_params={"MAX_NUM_SEQS": value},
            benchmark_params={"CONCURRENCY": 99},
            performance_index=PerformanceIndex(throughput=qps),
            csv_path="unused",
            qps=qps,
        )
        for role, value, qps in (("prefill", 16, 20), ("decode", 64, 10))
    }


def make_runner(tmp_path, pd_plugin):
    configured = settings(tmp_path)
    configured.vllm.command.model = "/models/demo"
    configured.vllm.command.served_model_name = "demo"
    configured.vllm.command.others = "--tensor-parallel-size 4 --quantization ascend"
    phase_runner = OptixPhaseRunner(
        configured, runtime_ctx=None, deploy_env=None, bak_path=None, load_breakpoint=False, top_k=0
    )
    runner = OptixFullServiceRunner(phase_runner)
    runner.cluster_config = cluster(pd_plugin)
    return runner


@pytest.mark.parametrize("mode", ["search", "full_service"])
@pytest.mark.parametrize("global_flags", [False, True])
def test_optimize_phase_mode_owns_optimizer_constraints(tmp_path, mode, global_flags, pd_plugin):
    runner = make_runner(tmp_path, pd_plugin).runner
    runner.top_k = 3
    runner.load_breakpoint = global_flags
    runner.settings.skip_pso = global_flags
    runner.settings.use_request_rate_calibration = global_flags
    phase = runner.settings.pd_disagg.prefill if mode == "search" else runner.settings.pd_disagg.full_service
    simulator = MagicMock(data_field=())
    benchmark = MagicMock(data_field=tuple(load_fields()))
    result = OptimizationResult(fitness=1, params={}, performance_index=PerformanceIndex(throughput=10))
    with (
        patch("optix.optimizer.store.DataStorage") as storage_cls,
        patch("optix.optimizer.scheduler.Scheduler") as scheduler_cls,
        patch("optix.optimizer.experience_fine_tunning.FineTune"),
        patch("optix.optimizer.optimizer.PSOOptimizer") as search_cls,
        patch("optix.optimizer.pd_full_service.FullServiceOptimizer") as full_cls,
    ):
        full_service = mode == "full_service"
        selected_cls, unused_cls = (full_cls, search_cls) if full_service else (search_cls, full_cls)
        selected_cls.return_value.run_plugin.return_value = result
        actual, csv_path = runner.optimize_phase(phase, tmp_path, simulator, benchmark, benchmark.data_field, mode=mode)

        selected_cls.assert_called_once()
        unused_cls.assert_not_called()
        options = selected_cls.call_args.kwargs
        assert options["skip_pso"] is (True if full_service else global_flags)
        assert options["load_breakpoint"] is (False if full_service else global_flags)
        assert options["use_request_rate_calibration"] is (False if full_service else global_flags)
        assert scheduler_cls.call_args.kwargs["benchmark_run_count"] == phase.benchmark_run_count
        assert storage_cls.call_args.args[0].pso_top_k == (1 if full_service else 3)
        selected_cls.return_value.run_plugin.assert_called_once_with()
        assert csv_path == storage_cls.return_value.save_file
        assert actual.performance_index == result.performance_index
        assert actual.benchmark_param_names == ("CONCURRENCY", "REQUESTRATE")
    assert runner.top_k == 3
    assert runner.load_breakpoint is global_flags
    assert runner.settings.skip_pso is global_flags
    assert runner.settings.use_request_rate_calibration is global_flags


@pytest.mark.parametrize("rounds", [0, -1])
def test_full_service_mode_rejects_invalid_round_count_before_constructing_scheduler(tmp_path, pd_plugin, rounds):
    runner = make_runner(tmp_path, pd_plugin).runner
    phase = runner.settings.pd_disagg.full_service.model_copy(update={"benchmark_run_count": rounds})
    with (
        patch("optix.optimizer.store.DataStorage") as storage_cls,
        patch("optix.optimizer.scheduler.Scheduler") as scheduler_cls,
        pytest.raises(PdDisaggError, match="benchmark_run_count must be greater than or equal to 1"),
    ):
        runner.optimize_phase(phase, tmp_path, MagicMock(), MagicMock(), (), mode="full_service")
    storage_cls.assert_not_called()
    scheduler_cls.assert_not_called()


@pytest.mark.parametrize("rounds", [1, 2, 4])
def test_full_service_round_count_reaches_scheduler_with_global_calibration_enabled(tmp_path, pd_plugin, rounds):
    runner = make_runner(tmp_path, pd_plugin).runner
    runner.settings.use_request_rate_calibration = True
    phase = runner.settings.pd_disagg.full_service.model_copy(update={"benchmark_run_count": rounds})
    expected = OptimizationResult(fitness=1, params={}, performance_index=PerformanceIndex(throughput=10))
    with (
        patch("optix.optimizer.store.DataStorage"),
        patch("optix.optimizer.scheduler.Scheduler") as scheduler_cls,
        patch("optix.optimizer.experience_fine_tunning.FineTune"),
        patch("optix.optimizer.pd_full_service.FullServiceOptimizer") as optimizer_cls,
    ):
        optimizer_cls.return_value.run_plugin.return_value = expected
        actual, _ = runner.optimize_phase(phase, tmp_path, MagicMock(), MagicMock(), (), mode="full_service")
    assert scheduler_cls.call_args.kwargs["benchmark_run_count"] == rounds
    assert optimizer_cls.call_args.kwargs["use_request_rate_calibration"] is False
    assert actual.performance_index == expected.performance_index


@pytest.mark.parametrize("rounds", [1, 2, 4])
def test_full_service_config_accepts_positive_round_counts(rounds):
    phase = PdFullServiceConfig(enabled=True, target_field=load_fields(), benchmark_run_count=rounds)
    assert phase.benchmark_run_count == rounds


@pytest.mark.parametrize("rounds", [0, -1])
def test_full_service_config_rejects_nonpositive_round_counts(rounds):
    with pytest.raises(ValidationError, match="benchmark_run_count"):
        PdFullServiceConfig(benchmark_run_count=rounds)


def test_full_service_defaults_warm_up_without_relaxing_slo_or_single_side_defaults(tmp_path):
    configured = settings(tmp_path)
    assert configured.pd_disagg.full_service.benchmark_run_count == 2
    assert configured.pd_disagg.full_service.ttft_slo == 0.5
    assert configured.pd_disagg.prefill.benchmark_run_count == 1
    assert configured.pd_disagg.decode.benchmark_run_count == 1
    reloaded = Settings.model_validate({**configured.model_dump(), "use_request_rate_calibration": True})
    assert reloaded.pd_disagg.full_service.benchmark_run_count == 2


def test_optimize_phase_rejects_unknown_mode(tmp_path, pd_plugin):
    runner = make_runner(tmp_path, pd_plugin).runner
    with pytest.raises(PdDisaggError, match="Unknown PD optimization mode"):
        runner.optimize_phase(runner.settings.pd_disagg.prefill, tmp_path, MagicMock(), MagicMock(), (), mode="typo")


def test_pd_only_additional_config_preserves_search_results_and_other_role(tmp_path, pd_plugin):
    runner = make_runner(tmp_path, pd_plugin)
    configured = runner.settings
    configured.vllm.command.others += ' --additional-config \'{"enable_cpu_binding":true}\''
    configured.pd_disagg.full_service.decode_additional_config = {"recompute_scheduler_enable": True}
    plans = runner._plans(phase_results())
    for role in ("prefill", "decode"):
        args = shlex.split(plans[role]["others"])
        additional = json.loads(args[args.index("--additional-config") + 1])
        assert additional["enable_cpu_binding"] is True
        assert additional.get("recompute_scheduler_enable", False) == (role == "decode")
        assert args[args.index("--max-num-seqs") + 1] == ("16" if role == "prefill" else "64")
    assert "recompute_scheduler_enable" not in configured.vllm.command.others


@pytest.mark.parametrize("with_results", [False, True])
def test_deployment_decode_can_disable_search_only_prefix_cache(tmp_path, with_results, pd_plugin):
    runner = make_runner(tmp_path, pd_plugin)
    configured = runner.settings
    search_others = "--tensor-parallel-size 4 --enable-prefix-caching --quantization ascend"
    deployment_others = "--tensor-parallel-size 4 --no-enable-prefix-caching --quantization ascend --max-num-seqs 8"
    configured.pd_disagg.decode.simulator_command_overrides = {"others": search_others}
    configured.pd_disagg.full_service = PdFullServiceConfig.model_validate(
        {**runner.phase.model_dump(), "decode_others": deployment_others}
    )
    runner = OptixFullServiceRunner(runner.runner)
    results = phase_results() if with_results else None
    plans = runner._plans(results)
    decode = shlex.split(plans["decode"]["others"])
    assert "--no-enable-prefix-caching" in decode
    assert "--enable-prefix-caching" not in decode
    assert decode[decode.index("--quantization") + 1] == "ascend"
    if with_results:
        assert decode[decode.index("--max-num-seqs") + 1] == "64"
    assert "--no-enable-prefix-caching" not in shlex.split(plans["prefill"]["others"])
    assert configured.pd_disagg.decode.simulator_command_overrides["others"] == search_others
    assert configured.pd_disagg.full_service.decode_others == deployment_others


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_deployment_others_are_optional_and_role_specific(tmp_path, role, pd_plugin):
    runner = make_runner(tmp_path, pd_plugin)
    phase = getattr(runner.settings.pd_disagg, role)
    phase.simulator_command_overrides = {"others": "--tensor-parallel-size 4 --enable-prefix-caching"}
    assert "--enable-prefix-caching" in shlex.split(runner._plans()[role]["others"])
    runner.settings.pd_disagg.full_service = PdFullServiceConfig.model_validate(
        {**runner.phase.model_dump(), f"{role}_others": "--tensor-parallel-size 4 --no-enable-prefix-caching"}
    )
    runner = OptixFullServiceRunner(runner.runner)
    plans = runner._plans()
    assert "--no-enable-prefix-caching" in shlex.split(plans[role]["others"])
    other = "decode" if role == "prefill" else "prefill"
    assert "--no-enable-prefix-caching" not in shlex.split(plans[other]["others"])


def test_full_service_config_requires_load_only_and_device_budgets():
    with pytest.raises(ValidationError, match="requires CONCURRENCY"):
        PdFullServiceConfig(enabled=True)
    with pytest.raises(ValidationError, match="requires total_devices"):
        PdDisaggConfig(full_service=PdFullServiceConfig(enabled=True, target_field=load_fields()))
    with pytest.raises(ValidationError, match="may tune only"):
        PdFullServiceConfig(enabled=True, target_field=[*load_fields(), OptimizerConfigField(name="TP")])


def test_pool_allocation_splits_cards_offsets_ports_and_preserves_input(pd_plugin):
    pool = cluster(pd_plugin)
    pd_plugin.pipeline.validate_resource_pool(pool, 12)
    allocated = pd_plugin.pipeline.allocate_resource_pool(
        pool, {"prefill": 1, "decode": 2, "prefill_devices": 4, "decode_devices": 8}
    )
    p, d = allocated.nodes
    assert p.gpu_ids == [0, 1, 2, 3]
    assert d.gpu_ids == list(range(4, 12))
    assert d.service_port == p.service_port + 4
    assert d.ascend_base_port == p.ascend_base_port + 4
    assert len(pool.nodes) == 1 and pool.nodes[0].role == "auto"
    assert allocated.prefill_instances == 1 and allocated.decode_instances == 2


def test_pool_rejects_overlap_and_honors_pinned_roles(pd_plugin):
    pool = cluster(pd_plugin)
    pool.nodes.append(pool.nodes[0].model_copy())
    with pytest.raises(ValueError, match="overlapping device"):
        pd_plugin.pipeline.validate_resource_pool(pool, 12)
    pool = cluster(pd_plugin)
    pool.nodes[0].role = "decode"
    with pytest.raises(ValueError, match="prefill"):
        pd_plugin.pipeline.allocate_resource_pool(
            pool, {"prefill": 1, "decode": 2, "prefill_devices": 4, "decode_devices": 8}
        )


@pytest.mark.parametrize(
    "address",
    ["", "   ", "0.0.0.0", "0.0.0.0 ", "::", "::0", "0:0:0:0:0:0:0:0", "0", "00", "0x0", "0.0"],
)
def test_proxy_rejects_equivalent_unspecified_addresses(address, pd_plugin):
    pool = cluster(pd_plugin)
    pool.proxy.bind_ip = address
    with pytest.raises(ValueError, match="explicit reachable Proxy"):
        pd_plugin.pipeline.validate_resource_pool(pool, 12)


@pytest.mark.parametrize("address", ["192.0.2.10", "2001:db8::1", "Proxy.Example.COM"])
def test_proxy_keeps_explicit_ip_and_hostname_support(address, pd_plugin):
    pool = cluster(pd_plugin)
    pool.proxy.bind_ip = address
    pd_plugin.pipeline.validate_resource_pool(pool, 12)
    assert pool.proxy.bind_ip == address


def test_pool_ignores_fragmented_capacity_when_tp_requires_whole_replicas(pd_plugin):
    pool = cluster(pd_plugin)
    pool.nodes = [
        pd_plugin.config.ClusterNodeConfig(ssh_ip=f"192.0.2.{10 + index}", gpu_ids=list(range(count)), role="auto")
        for index, count in enumerate((2, 8, 4))
    ]
    allocated = pd_plugin.pipeline.allocate_resource_pool(
        pool,
        {"prefill": 1, "decode": 2, "prefill_devices": 4, "decode_devices": 8},
        {"prefill": {"tp": 4}, "decode": {"tp": 4}},
    )
    assert all(len(node.gpu_ids) % 4 == 0 for node in allocated.nodes)
    assert sum(len(node.gpu_ids) for node in allocated.nodes) == 12
    assert allocated.nodes[0].ssh_ip == "192.0.2.11"


def test_role_plan_keeps_selected_flags_json_and_environment(pd_plugin):
    fields = [
        OptimizerConfigField.model_construct(name="MAX_NUM_SEQS", value=64, config_position="run"),
        OptimizerConfigField.model_construct(name="TASK_QUEUE_ENABLE", value=1, config_position="env"),
    ]
    result = pd_plugin.pipeline.plan_role(
        {
            "model": "/models/demo",
            "served_model_name": "demo",
            "others": '--tensor-parallel-size=4 --max-num-seqs 16 --speculative-config \'{"method":"mtp","num_speculative_tokens":3}\'',
        },
        fields,
        4,
    )
    assert result["tp"] == 4 and result["dp"] == 1
    assert "--tensor-parallel-size" not in result["others"]
    assert "--max-num-seqs 64" in result["others"]
    assert result["env"] == {"TASK_QUEUE_ENABLE": "1"}
    with pytest.raises(ValueError, match=r"TP \* DP"):
        pd_plugin.pipeline.plan_role(
            {"model": "m", "served_model_name": "m", "others": "--tensor-parallel-size 2"}, [], 4
        )


@pytest.mark.parametrize("literal", [False, True])
def test_aisbench_binding_preserves_path_and_chinese_text(literal):
    source = (
        "# 中文配置\nmodels = [dict(path='/models/tokenizer', model='old', host_ip='127.0.0.1', "
        "host_port=8900, url='http://old:8900/v1/chat/completions', batch_size=4)]\n"
    )
    if literal:
        source = "models = [{'path':'/models/tokenizer', 'model':'old', 'host_ip':'old', 'host_port':'8900'}]\n"
    updated = bind_endpoint(source, host="192.0.2.10", port=8000, model="demo")
    ast.parse(updated)
    assert "/models/tokenizer" in updated and "192.0.2.10" in updated
    assert "demo" in updated and "8900" not in updated
    if not literal:
        assert "/v1/chat/completions" in updated and "batch_size=4" in updated


def test_aisbench_binding_refuses_unverifiable_address():
    with pytest.raises(TypeError, match="literal"):
        bind_endpoint(
            "models = [dict(model='m', host_ip=get_host(), host_port=8000)]", host="192.0.2.10", port=8000, model="m"
        )


def test_aisbench_url_uses_generated_http_proxy_and_brackets_ipv6():
    bound = bind_endpoint(
        "models = [dict(model='m', url='https://old/v1/chat/completions')]",
        host="::1",
        port=8000,
        model="demo",
    )
    assert "http://[::1]:8000/v1/chat/completions" in bound


@pytest.mark.parametrize("failure", [None, "startup", "benchmark"])
@pytest.mark.parametrize(
    ("pool_size", "budget", "p_cards", "d_cards"), [(12, 12, 4, 8), (32, 24, 8, 16), (32, 24, 4, 16)]
)
def test_full_service_handoff_cleanup_and_slo(tmp_path, failure, pd_plugin, pool_size, budget, p_cards, d_cards):
    runner = make_runner(tmp_path, pd_plugin)
    runner.settings.pd_disagg.total_devices = budget
    runner.cluster_config.nodes[0].gpu_ids = list(range(pool_size))
    runner.cluster_config.nodes.append(
        runner.cluster_config.proxy.model_copy(update={"role": "proxy", "gpu_ids": [99]})
    )
    simulator = MagicMock()
    simulator.config = cluster(pd_plugin)
    benchmark = MagicMock()
    benchmark.config.command = runner.settings.vllm_benchmark.command.model_copy(deep=True)
    benchmark.data_field = tuple(load_fields())
    used = p_cards + d_cards
    allocation = {
        "prefill": p_cards // 4,
        "decode": d_cards // 4,
        "prefill_devices": p_cards,
        "decode_devices": d_cards,
        "used_devices": used,
        "remaining_devices": budget - used,
    }
    result = OptimizationResult(
        fitness=1,
        params={"CONCURRENCY": 4, "REQUESTRATE": 4},
        performance_index=PerformanceIndex(
            throughput=18, generate_speed=100, time_to_first_token=0.4, time_per_output_token=0.04, success_rate=1
        ),
    )
    if failure == "startup":
        simulator.configure_pd_pipeline.side_effect = RuntimeError("startup failed")
    with (
        patch.object(
            runner.runner,
            "_resolve_and_validate_plugins",
            return_value=(MagicMock(return_value=simulator), MagicMock(return_value=benchmark)),
        ),
        patch.object(runner.runner, "optimize_phase", return_value=(result, tmp_path / "data.csv")) as optimize,
    ):
        if failure == "benchmark":
            optimize.side_effect = RuntimeError("benchmark failed")
        if failure:
            with pytest.raises(RuntimeError, match=failure):
                runner.run(phase_results(), allocation, tmp_path / "full_service")
        else:
            payload = runner.run(phase_results(), allocation, tmp_path / "full_service")
            assert payload["slo_met"]
            assert optimize.call_args.kwargs == {"mode": "full_service"}
            assert runner.runner.top_k == 0
            assert benchmark.config.command.host == "192.0.2.10"
            assert benchmark.config.command.port == "8000"
            assert (tmp_path / "full_service/deployment.json").exists()
            deployment = json.loads((tmp_path / "full_service/deployment.json").read_text())
            assert deployment["resource_usage"] == {
                "pool_total_devices": pool_size,
                "budget_devices": budget,
                "allocated_devices": used,
                "budget_remaining_devices": budget - used,
                "pool_idle_devices": pool_size - used,
            }
            from optix.optimizer.pd_report import _deployment_section

            report = "\n".join(_deployment_section(deployment))
            assert "| 设备池总卡数 | 预算卡数 | 已分配 | 预算内余量 | 池内闲置 |" in report
            assert f"| {pool_size} | {budget} | {used} | {budget - used} | {pool_size - used} |" in report
            plans = simulator.configure_pd_pipeline.call_args.args[0]
            assert "--max-num-seqs 16" in plans["prefill"]["others"]
            assert "--max-num-seqs 64" in plans["decode"]["others"]
            assert "CONCURRENCY" not in str(plans)
        simulator.stop.assert_called_once_with(del_log=False)


@pytest.mark.parametrize("failure", ["startup", "benchmark_setup", "benchmark", "interrupt", None])
def test_cleanup_failures_preserve_primary_error_and_attempt_all_actions(tmp_path, pd_plugin, failure):
    runner = make_runner(tmp_path, pd_plugin)
    simulator = MagicMock(config=cluster(pd_plugin))
    benchmark = MagicMock()
    benchmark.config.command = runner.settings.vllm_benchmark.command.model_copy(deep=True)
    benchmark.data_field = tuple(load_fields())
    simulator.stop.side_effect = RuntimeError("service cleanup failed")
    benchmark.stop.side_effect = RuntimeError("benchmark stop failed")
    benchmark.restore_service_endpoint.side_effect = ValueError("endpoint restore failed")
    primary = (
        KeyboardInterrupt("original interruption") if failure == "interrupt" else ValueError(f"original {failure}")
    )
    if failure == "startup":
        simulator.configure_pd_pipeline.side_effect = primary
    result = OptimizationResult(
        fitness=1,
        params={"CONCURRENCY": 4, "REQUESTRATE": 4},
        performance_index=PerformanceIndex(
            throughput=18, time_to_first_token=0.4, time_per_output_token=0.04, success_rate=1
        ),
    )
    allocation = {"prefill": 1, "decode": 2, "prefill_devices": 4, "decode_devices": 8}
    with (
        patch.object(
            runner.runner,
            "_resolve_and_validate_plugins",
            return_value=(MagicMock(return_value=simulator), MagicMock(return_value=benchmark)),
        ),
        patch.object(
            runner.runner,
            "_apply_benchmark_command_overrides",
            side_effect=primary if failure == "benchmark_setup" else None,
        ),
        patch.object(runner.runner, "optimize_phase", return_value=(result, tmp_path / "data.csv")) as optimize,
        patch("optix.optimizer.pd_full_service.logger.error") as log,
    ):
        if failure in {"benchmark", "interrupt"}:
            optimize.side_effect = primary
        with pytest.raises(type(primary) if failure else RuntimeError) as caught:
            runner.run(phase_results(), allocation, tmp_path / "full_service")
    if failure:
        assert caught.value is primary
    else:
        assert "PD full-service cleanup failed" in str(caught.value)
    expected = [{"operation": "simulator.stop", "type": "RuntimeError", "message": "service cleanup failed"}]
    if failure != "startup":
        benchmark.stop.assert_called_once_with()
        benchmark.restore_service_endpoint.assert_called_once_with()
        expected[:0] = [
            {"operation": "benchmark.stop", "type": "RuntimeError", "message": "benchmark stop failed"},
            {
                "operation": "benchmark.restore_service_endpoint",
                "type": "ValueError",
                "message": "endpoint restore failed",
            },
        ]
    else:
        benchmark.stop.assert_not_called()
        benchmark.restore_service_endpoint.assert_not_called()
    simulator.stop.assert_called_once_with(del_log=False)
    assert runner.cleanup_errors == expected
    assert log.call_count == len(expected)
    assert not (tmp_path / "full_service/phase_result.json").exists()


@pytest.mark.parametrize("failed_action", ["benchmark.stop", "benchmark.restore_service_endpoint", "simulator.stop"])
def test_each_cleanup_failure_marks_otherwise_successful_run_failed(tmp_path, pd_plugin, failed_action):
    runner = make_runner(tmp_path, pd_plugin)
    simulator = MagicMock(config=cluster(pd_plugin))
    benchmark = MagicMock()
    benchmark.config.command = runner.settings.vllm_benchmark.command.model_copy(deep=True)
    benchmark.data_field = tuple(load_fields())
    actions = {
        "benchmark.stop": benchmark.stop,
        "benchmark.restore_service_endpoint": benchmark.restore_service_endpoint,
        "simulator.stop": simulator.stop,
    }
    actions[failed_action].side_effect = RuntimeError("isolated cleanup failure")
    result = OptimizationResult(
        fitness=1,
        params={"CONCURRENCY": 4, "REQUESTRATE": 4},
        performance_index=PerformanceIndex(
            throughput=18, time_to_first_token=0.4, time_per_output_token=0.04, success_rate=1
        ),
    )
    with (
        patch.object(
            runner.runner,
            "_resolve_and_validate_plugins",
            return_value=(MagicMock(return_value=simulator), MagicMock(return_value=benchmark)),
        ),
        patch.object(runner.runner, "optimize_phase", return_value=(result, tmp_path / "data.csv")),
        pytest.raises(RuntimeError, match="isolated cleanup failure"),
    ):
        runner.run(
            phase_results(),
            {"prefill": 1, "decode": 2, "prefill_devices": 4, "decode_devices": 8},
            tmp_path / "full_service",
        )
    assert runner.cleanup_errors == [
        {"operation": failed_action, "type": "RuntimeError", "message": "isolated cleanup failure"}
    ]
    benchmark.stop.assert_called_once_with()
    benchmark.restore_service_endpoint.assert_called_once_with()
    simulator.stop.assert_called_once_with(del_log=False)


@pytest.mark.parametrize("main_failed", [True, False])
def test_cleanup_diagnostics_reach_summary_and_final_report(tmp_path, pd_plugin, main_failed):
    runner = make_runner(tmp_path, pd_plugin)
    simulator = MagicMock(config=cluster(pd_plugin))
    simulator.stop.side_effect = RuntimeError("service cleanup failed")
    benchmark = MagicMock()
    benchmark.config.command = runner.settings.vllm_benchmark.command.model_copy(deep=True)
    benchmark.data_field = tuple(load_fields())
    benchmark.stop.side_effect = RuntimeError("benchmark stop failed")
    benchmark.restore_service_endpoint.side_effect = ValueError("endpoint restore failed")
    primary = ValueError("original benchmark failure")
    results = phase_results()

    def run_phase(name, *args):
        result = results[name]
        return OptimizationResult(
            fitness=1,
            params=result.best_params,
            performance_index=result.performance_index,
            service_param_names=("MAX_NUM_SEQS",),
            benchmark_param_names=("CONCURRENCY",),
        ), tmp_path / "missing.csv"

    full_result = OptimizationResult(
        fitness=1,
        params={"CONCURRENCY": 4, "REQUESTRATE": 4},
        performance_index=PerformanceIndex(
            throughput=18, time_to_first_token=0.4, time_per_output_token=0.04, success_rate=1
        ),
    )
    with (
        patch("optix.optimizer.pd_full_service.OptixFullServiceRunner", return_value=runner),
        patch.object(runner, "preflight"),
        patch.object(runner.runner, "preflight"),
        patch.object(runner.runner, "run", side_effect=run_phase),
        patch.object(
            runner.runner,
            "_resolve_and_validate_plugins",
            return_value=(MagicMock(return_value=simulator), MagicMock(return_value=benchmark)),
        ),
        patch.object(
            runner.runner,
            "optimize_phase",
            side_effect=primary if main_failed else None,
            return_value=(full_result, tmp_path / "data.csv"),
        ),
        pytest.raises(PdDisaggError) as caught,
    ):
        PdDisaggOrchestrator(runner.settings, "cleanup", runner.runner).run()
    saved = json.loads((tmp_path / "pd_disagg/cleanup/pd_disagg_summary.json").read_text(encoding="utf-8"))
    assert saved["status"] == "failed" and saved["failed_phase"] == "full_service"
    if main_failed:
        assert caught.value.__cause__ is primary
        assert saved["error"] == {"type": "ValueError", "message": "original benchmark failure"}
    else:
        assert saved["error"]["type"] == "RuntimeError"
        assert "PD full-service cleanup failed" in saved["error"]["message"]
    assert saved["full_service"]["cleanup_errors"] == runner.cleanup_errors
    report = Path(saved["artifacts"]["final_report"]).read_text(encoding="utf-8")
    assert "## 清理异常（附加信息）" in report
    assert "失败阶段：full_service" in report
    assert "完整服务 SLO：未完成" in report
    if main_failed:
        assert "错误：ValueError: original benchmark failure" in report
    for error in runner.cleanup_errors:
        assert f"| {error['operation']} | {error['type']} | {error['message']} |" in report


def test_orchestrator_runs_full_service_after_search_and_preserves_failure(tmp_path):
    configured = settings(tmp_path)
    results = phase_results()
    phase_runner = MagicMock()

    def run_phase(name, *args):
        result = results[name]
        return OptimizationResult(
            fitness=1,
            params=result.best_params,
            performance_index=result.performance_index,
            service_param_names=("MAX_NUM_SEQS",),
            benchmark_param_names=("CONCURRENCY",),
        ), tmp_path / "missing.csv"

    phase_runner.run.side_effect = run_phase
    with patch("optix.optimizer.pd_full_service.OptixFullServiceRunner") as full_cls:
        full_cls.return_value.run.return_value = {"status": "success", "slo_met": False}
        summary = PdDisaggOrchestrator(configured, "full", phase_runner).run()
        assert summary["status"] == "completed" and not summary["full_service"]["slo_met"]
        report = tmp_path / "pd_disagg/full/final_result.md"
        assert summary["artifacts"]["final_report"] == str(report)
        assert "完整服务 SLO：未达标" in report.read_text(encoding="utf-8")
        full_cls.return_value.preflight.assert_called_once()
        assert [call.args[0] for call in phase_runner.run.call_args_list] == ["prefill", "decode"]
        full_cls.return_value.run.side_effect = RuntimeError("cluster failed")
        with pytest.raises(Exception, match="cluster failed"):
            PdDisaggOrchestrator(configured, "failed", phase_runner).run()
        saved = json.loads((tmp_path / "pd_disagg/failed/pd_disagg_summary.json").read_text())
        assert saved["failed_phase"] == "full_service"
        failed_report = Path(saved["artifacts"]["final_report"]).read_text(encoding="utf-8")
        assert "失败阶段：full_service" in failed_report
        assert "cluster failed" in failed_report
        assert "完整服务 SLO：未完成" in failed_report
        assert "MAX_NUM_SEQS" in failed_report
        assert saved["prefill"]["status"] == "success" and saved["decode"]["status"] == "success"


def test_selection_prefers_successful_slo_compliant_request_throughput():
    optimizer = FullServiceOptimizer.__new__(FullServiceOptimizer)
    optimizer.success_rate_slo, optimizer.ttft_slo, optimizer.tpot_slo = 1, 0.5, 0.05
    metrics = [
        PerformanceIndex(
            throughput=qps,
            generate_speed=tokens,
            success_rate=success,
            time_to_first_token=ttft,
            time_per_output_token=0.05,
        )
        for qps, tokens, success, ttft in [
            (100, 1000, 0.9, 0.4),
            (20, 500, 1, 0.5),
            (30, 400, 1, 0.4),
            (50, 2000, 1, 0.6),
        ]
    ]
    _, params, best = optimizer.best_params([1, 2, 3, 4], [0, 1, 2, 3], metrics)
    assert params == 2 and best.throughput == 30


def test_real_plugin_consumes_frozen_plans_without_benchmark_parameters(tmp_path, pd_plugin):
    from vllm_pd_simulator.pd_cluster_simulator import PdClusterSimulator

    runner = make_runner(tmp_path, pd_plugin)
    plans = runner._plans(phase_results())
    simulator = PdClusterSimulator.__new__(PdClusterSimulator)
    simulator.config = cluster(pd_plugin)
    simulator._is_moe = False
    simulator._prefill_env_vars = simulator._decode_env_vars = {}
    simulator._prefill_run_vars = simulator._decode_run_vars = {}
    simulator.configure_pd_pipeline(plans, {"prefill": 1, "decode": 2, "prefill_devices": 4, "decode_devices": 8}, {})
    assert len(simulator.config.prefill_groups) == 1 and len(simulator.config.decode_groups) == 2
    simulator.before_run(load_fields())
    assert simulator._prefill_env_vars == {} and simulator._decode_env_vars == {}
    for role, groups, expected in [
        ("P", simulator.config.prefill_groups, 16),
        ("D", simulator.config.decode_groups, 64),
    ]:
        context = simulator._build_context(role, groups[0].nodes[0], 0, [0, 1, 2, 3], 18080, 30100, 29500)
        assert f"--max-num-seqs {expected}" in context["vllm.command"].others
    assert simulator._engine_id_for("D", 0) != simulator._engine_id_for("D", 1)


@pytest.mark.parametrize("is_moe", [False, True])
def test_deployment_port_plan_matches_scripts_without_opening_ssh(tmp_path, pd_plugin, is_moe):
    from vllm_pd_simulator.pd_cluster_simulator import PdClusterSimulator

    runner = make_runner(tmp_path, pd_plugin)
    if is_moe:
        runner.settings.pd_disagg.decode.simulator_command_overrides = {
            "others": "--tensor-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel"
        }
    plans = runner._plans(phase_results())
    simulator = PdClusterSimulator.__new__(PdClusterSimulator)
    simulator.config = cluster(pd_plugin)
    simulator.config.nodes[0].docker_container_id = "test-container"
    simulator._is_moe = is_moe
    simulator._prefill_env_vars = simulator._decode_env_vars = {}
    simulator._prefill_run_vars = simulator._decode_run_vars = {}
    with (
        patch.object(simulator, "_exec_for_node") as executor,
        patch("vllm_pd_simulator.pd_cluster_simulator.logger.info") as log,
    ):
        simulator.configure_pd_pipeline(
            plans, {"prefill": 1, "decode": 2, "prefill_devices": 4, "decode_devices": 8}, {}
        )
        executor.assert_not_called()
    messages = [call.args[0] for call in log.call_args_list if "Configured PD deployment ports" in call.args[0]]
    assert len(messages) == 1
    inventory = messages[0]
    assert "live availability NOT checked" in inventory
    assert "backend-selected ports are not listed" in inventory
    assert f"service={simulator.config.proxy.bind_ip}:8000" in inventory
    with patch.object(simulator, "_get_round_dir", return_value=tmp_path):
        nodes = simulator._generate_all_scripts()
    workers = [node for node in nodes if not node["label"].startswith("proxy_")]
    assert len(workers) == (5 if is_moe else 3)
    lines = inventory.splitlines()
    for info in workers:
        label = info["label"].split("-T", 1)[0]
        line = next(line for line in lines if line.startswith(f"[{label}] "))
        tokens = shlex.split((tmp_path / f"{info['label']}.sh").read_text())
        service_port = int(tokens[tokens.index("--port") + 1])
        kv_port = json.loads(tokens[tokens.index("--kv-transfer-config") + 1])["kv_port"]
        tp = int(tokens[tokens.index("--tensor-parallel-size") + 1])
        node = info["_node_config"]
        assert f"ssh={node.ssh_ip}:{node.ssh_port}" in line
        assert "container=test-container" in line
        assert f"service=0.0.0.0:{service_port} " in line
        assert f"kv={kv_port}-{kv_port + tp - 1} " in line
        assert f"configured_rpc={node.rpc_port + info['_local_dp_rank']} " in line
        assert f"ASCEND_BASE_PORT={node.ascend_base_port}" in line
        if is_moe:
            coordinator = tokens[tokens.index("--data-parallel-address") + 1]
            coordinator_port = tokens[tokens.index("--data-parallel-rpc-port") + 1]
            group_label = label.rsplit("-R", 1)[0]
            assert f"[{group_label}] dp_coordinator={coordinator}:{coordinator_port}" in lines
    assert ("dp_coordinator=" in inventory) == is_moe


def test_invalid_deployment_does_not_print_validated_port_plan(tmp_path, pd_plugin):
    from vllm_pd_simulator.pd_cluster_simulator import PdClusterSimulator

    simulator = PdClusterSimulator.__new__(PdClusterSimulator)
    simulator.config = cluster(pd_plugin)
    simulator.config.proxy.service_port = simulator.config.nodes[0].service_port
    simulator._is_moe = False
    with (
        patch.object(simulator, "_exec_for_node") as executor,
        patch("vllm_pd_simulator.pd_cluster_simulator.logger.info") as log,
        pytest.raises(ValueError, match="conflicting PD endpoint"),
    ):
        simulator.configure_pd_pipeline(
            make_runner(tmp_path, pd_plugin)._plans(phase_results()),
            {"prefill": 1, "decode": 2, "prefill_devices": 4, "decode_devices": 8},
            {},
        )
    executor.assert_not_called()
    assert not any("Configured PD deployment ports" in call.args[0] for call in log.call_args_list)


def test_preflight_rejects_resource_errors_before_search(tmp_path, pd_plugin):
    runner = make_runner(tmp_path, pd_plugin)
    with (
        patch.object(runner.runner, "_resolve_and_validate_plugins"),
        patch("vllm_pd_simulator.config.load_pd_config", return_value=pd_plugin.config.VLLMPDDisaggConfig()),
        pytest.raises(Exception, match="resource pool"),
    ):
        runner.preflight()


def test_invalid_allocation_does_not_cleanup_a_cluster_that_never_started(tmp_path, pd_plugin):
    from vllm_pd_simulator.pd_cluster_simulator import PdClusterSimulator

    simulator = PdClusterSimulator.__new__(PdClusterSimulator)
    simulator.config = cluster(pd_plugin)
    simulator.config.nodes[0].role = "decode"
    with pytest.raises(ValueError, match="prefill"):
        simulator.configure_pd_pipeline(
            make_runner(tmp_path, pd_plugin)._plans(),
            {"prefill": 1, "decode": 2, "prefill_devices": 4, "decode_devices": 8},
            {},
        )
    with patch.object(simulator, "_cleanup_all_nodes") as cleanup:
        simulator.stop()
    cleanup.assert_not_called()


@pytest.mark.usefixtures("pd_plugin")
def test_strict_loader_rejects_unknown_host_reference(tmp_path):
    from vllm_pd_simulator.config import load_pd_config

    config = tmp_path / "pd.toml"
    config.write_text('[vllm_pd.proxy]\nhosts = "missing"\n')
    with (
        patch("vllm_pd_simulator.config._find_config_file", return_value=config),
        pytest.raises(ValueError, match="Unknown deployment host"),
    ):
        load_pd_config(strict=True)


def test_generated_deployment_rejects_port_collision_and_cross_node_tp(tmp_path, pd_plugin):
    from vllm_pd_simulator.pd_cluster_simulator import PdClusterSimulator
    from vllm_pd_simulator.pipeline import validate_deployment

    runner = make_runner(tmp_path, pd_plugin)
    simulator = PdClusterSimulator.__new__(PdClusterSimulator)
    simulator.config = cluster(pd_plugin)
    simulator._is_moe = False
    simulator.configure_pd_pipeline(
        runner._plans(phase_results()),
        {"prefill": 1, "decode": 2, "prefill_devices": 4, "decode_devices": 8},
        {},
    )
    node = simulator.config.prefill_groups[0].nodes[0]
    simulator.config.proxy.service_port = node.service_port
    with pytest.raises(ValueError, match="conflicting PD endpoint"):
        validate_deployment(simulator.config, runner._plans(phase_results()))
    simulator.config.proxy.service_port = 8000
    node.gpu_ids = [0, 1]
    with pytest.raises(ValueError, match="each TP replica"):
        validate_deployment(simulator.config, runner._plans(phase_results()))


@pytest.mark.parametrize(
    ("prefill_updates", "decode_updates", "conflict"),
    [
        ({}, {"bind_ip": "0.0.0.0"}, True),
        ({}, {"bind_ip": "192.0.2.11"}, True),
        ({"bind_ip": "0.0.0.0"}, {"ssh_ip": "192.0.2.11", "bind_ip": "0.0.0.0"}, False),
        ({"bind_ip": "::"}, {"ssh_ip": "192.0.2.11", "bind_ip": "0:0:0:0:0:0:0:0"}, False),
        (
            {"ssh_ip": "2001:db8::1"},
            {"ssh_ip": "2001:0db8:0:0:0:0:0:1", "bind_ip": "192.0.2.11"},
            True,
        ),
        ({"ssh_ip": "Node.Example.COM"}, {"ssh_ip": "node.example.com", "bind_ip": "192.0.2.11"}, True),
        (
            {"docker_container_id": "container-p"},
            {"docker_container_id": "container-d", "bind_ip": "192.0.2.11"},
            False,
        ),
        (
            {"bind_ip": "2001:db8::1"},
            {"ssh_ip": "192.0.2.11", "bind_ip": "2001:0db8:0:0:0:0:0:1"},
            True,
        ),
        ({}, {"ssh_ip": "192.0.2.11", "bind_ip": "192.0.2.11"}, False),
    ],
)
def test_port_reservations_follow_listeners_and_execution_context(prefill_updates, decode_updates, conflict, pd_plugin):
    prefill = pd_plugin.config.ClusterNodeConfig(
        ssh_ip="192.0.2.10", bind_ip="192.0.2.10", gpu_ids=[0], service_port=9000, kv_port=10000, rpc_port=12000
    )
    decode = prefill.model_copy(update={"gpu_ids": [1], "kv_port": 11000, "rpc_port": 13000, **decode_updates})
    prefill = prefill.model_copy(update=prefill_updates)
    pool = cluster(pd_plugin)
    pool.prefill_groups = [pd_plugin.config.PDGroup(nodes=[prefill])]
    pool.decode_groups = [pd_plugin.config.PDGroup(nodes=[decode])]
    plans = {"prefill": {"tp": 1}, "decode": {"tp": 1}}
    if conflict:
        with pytest.raises(ValueError, match="conflicting PD endpoint"):
            pd_plugin.pipeline.validate_deployment(pool, plans)
    else:
        pd_plugin.pipeline.validate_deployment(pool, plans)


def test_proxy_port_conflicts_with_worker_wildcard_listener_on_another_interface(pd_plugin):
    pool = cluster(pd_plugin)
    pool.proxy.bind_ip = "192.0.2.11"
    pool.proxy.ssh_ip = pool.nodes[0].ssh_ip
    pool.proxy.service_port = pool.nodes[0].service_port
    pool.prefill_groups = [pd_plugin.config.PDGroup(nodes=[pool.nodes[0]])]
    with pytest.raises(ValueError, match="conflicting PD endpoint"):
        pd_plugin.pipeline.validate_deployment(pool, {"prefill": {"tp": 4}, "decode": {"tp": 4}})


def test_aisbench_restores_original_config_bytes_after_refinement(tmp_path):
    from optix.optimizer.interfaces.benchmark import BenchmarkInterface
    from optix.optimizer.plugins.benchmark import AisBench

    config = tmp_path / "model.py"
    original = b"models = [dict(model='old', host_ip='old', host_port=8900, batch_size=4, request_rate=0,)]\r\n"
    config.write_bytes(original)
    benchmark = AisBench.__new__(AisBench)
    benchmark.config = SimpleNamespace(output_path=tmp_path / "output")
    benchmark.model_config_path = config
    benchmark.bind_service_endpoint(host="192.0.2.10", port=8000, model="demo")
    try:
        with patch.object(BenchmarkInterface, "before_run"):
            benchmark.before_run(tuple(load_fields()))
        bound = config.read_text()
        assert "192.0.2.10" in bound and "8000" in bound and "demo" in bound
    finally:
        benchmark.restore_service_endpoint()
    assert config.read_bytes() == original


def test_real_optimizer_launches_once_and_refines_only_benchmark(tmp_path, pd_plugin):
    from optix.optimizer.scheduler import Scheduler

    runner = make_runner(tmp_path, pd_plugin)
    runner.settings.max_fine_tune = 4
    simulator = MagicMock()
    simulator.config = cluster(pd_plugin)
    simulator.data_field = ()
    benchmark = SimpleNamespace(data_field=tuple(load_fields()), stop=MagicMock())
    baseline = PerformanceIndex(
        throughput=10, generate_speed=100, time_to_first_token=0.7, time_per_output_token=0.08, success_rate=1
    )
    improved = PerformanceIndex(
        throughput=18, generate_speed=180, time_to_first_token=0.48, time_per_output_token=0.048, success_rate=1
    )

    def start(scheduler, params, fields):
        scheduler.simulator.run(fields)
        return baseline

    with (
        patch.object(
            runner.runner,
            "_resolve_and_validate_plugins",
            return_value=(MagicMock(return_value=simulator), MagicMock()),
        ),
        patch.object(runner, "_benchmark", return_value=benchmark),
        patch.object(Scheduler, "run", autospec=True, side_effect=start),
        patch.object(Scheduler, "run_benchmark_only", return_value=improved) as benchmark_only,
        patch.object(Scheduler, "save_result"),
    ):
        result = runner.run(
            phase_results(),
            {"prefill": 1, "decode": 2, "prefill_devices": 4, "decode_devices": 8},
            tmp_path / "full_service",
        )
    simulator.run.assert_called_once()
    assert benchmark_only.call_count >= 1
    assert result["slo_met"] and result["performance_index"]["throughput"] == 18


@pytest.mark.parametrize(
    ("initial", "lower", "upper", "tpot", "expected"),
    [(21, 20, 32, 1.0, 20), (31, 20, 32, 0.001, 32), (24, 24, 24, 1.0, 24)],
)
def test_full_service_refinement_respects_concurrency_bounds(
    tmp_path, initial, lower, upper, tpot, expected, pd_plugin
):
    from optix.optimizer.scheduler import Scheduler

    runner = make_runner(tmp_path, pd_plugin)
    runner.settings.use_request_rate_calibration = True
    runner.settings.max_fine_tune = 2
    fields = [
        OptimizerConfigField(
            name="CONCURRENCY", config_position="env", dtype="int", min=lower, max=upper, value=initial
        ),
        OptimizerConfigField(name="REQUESTRATE", config_position="env", dtype="float", min=0, max=0, value=0),
    ]
    runner.phase.target_field = fields
    simulator = MagicMock(data_field=())
    simulator.config = cluster(pd_plugin)
    benchmark = SimpleNamespace(data_field=tuple(fields), stop=MagicMock())
    baseline = PerformanceIndex(
        throughput=10, generate_speed=100, time_to_first_token=0.4, time_per_output_token=tpot, success_rate=1
    )
    improved = PerformanceIndex(
        throughput=18, generate_speed=180, time_to_first_token=0.4, time_per_output_token=0.05, success_rate=1
    )
    with (
        patch.object(
            runner.runner,
            "_resolve_and_validate_plugins",
            return_value=(MagicMock(return_value=simulator), MagicMock()),
        ),
        patch.object(runner, "_benchmark", return_value=benchmark),
        patch.object(Scheduler, "run", return_value=baseline),
        patch.object(Scheduler, "run_benchmark_only", return_value=improved) as benchmark_only,
        patch.object(Scheduler, "save_result"),
    ):
        result = runner.run(
            phase_results(),
            {"prefill": 1, "decode": 2, "prefill_devices": 4, "decode_devices": 8},
            tmp_path / "full_service",
        )
    assert result["benchmark_params"]["CONCURRENCY"] == expected
    if lower == upper:
        benchmark_only.assert_not_called()
    else:
        benchmark_only.assert_called_once()
    for call in benchmark_only.call_args_list:
        concurrency = next(field for field in call.args[1] if field.name == "CONCURRENCY")
        assert lower <= concurrency.value <= upper
    assert runner.phase.target_field[0].min == lower
    assert runner.phase.target_field[0].max == upper
    assert runner.phase.target_field[0].value == initial


def test_complete_template_loads_and_generates_pd_launch_scripts(tmp_path, pd_plugin):
    from vllm_pd_simulator.config import load_pd_config
    from vllm_pd_simulator.pd_cluster_simulator import PdClusterSimulator

    template = pd_plugin.repo_root / "docs/zh/use_cases/examples/optix_pd_disaggregation/full_pipeline.toml"

    class TemplateSettings(Settings):
        model_config = SettingsConfigDict(**{**Settings.model_config, "toml_file": [template], "extra": "allow"})

    configured = TemplateSettings(output=tmp_path)
    phase_runner = OptixPhaseRunner(
        configured, runtime_ctx=None, deploy_env=None, bak_path=None, load_breakpoint=False, top_k=0
    )
    runner = OptixFullServiceRunner(phase_runner)
    with patch("vllm_pd_simulator.config._find_config_file", return_value=template):
        runner.cluster_config = load_pd_config(strict=True)
    pd_plugin.pipeline.validate_resource_pool(runner.cluster_config, 24)
    plans = runner._plans()
    search_decode = shlex.split(configured.pd_disagg.decode.simulator_command_overrides["others"])
    assert "--enable-prefix-caching" in search_decode
    assert "--no-enable-prefix-caching" not in search_decode
    assert "--no-enable-prefix-caching" in shlex.split(plans["decode"]["others"])
    assert "--enable-prefix-caching" not in shlex.split(plans["decode"]["others"])
    simulator = PdClusterSimulator.__new__(PdClusterSimulator)
    simulator.config = runner.cluster_config
    from vllm_pd_simulator.tools.model_tools import detect_is_moe

    model_path = pd_plugin.repo_root / "tests/assets/model_config/qwen3_5_moe_397b_a17b"
    simulator._is_moe = detect_is_moe(str(model_path))
    simulator._prefill_env_vars = simulator._decode_env_vars = {}
    simulator._prefill_run_vars = simulator._decode_run_vars = {}
    simulator.configure_pd_pipeline(plans, {"prefill": 1, "decode": 2, "prefill_devices": 8, "decode_devices": 16}, {})
    with patch.object(simulator, "_get_round_dir", return_value=tmp_path):
        nodes = simulator._generate_all_scripts()
    assert len(nodes) == 4  # One P, two D and one Proxy.
    assert simulator._engine_id_for("D", 0, 0) == simulator._engine_id_for("D", 0, 1)
    bash = shutil.which("bash")
    for node in nodes:
        script = tmp_path / f"{node['label']}.sh"
        content = script.read_text()
        assert "<%" not in content
        if node["label"].startswith("D-"):
            assert "--no-enable-prefix-caching" in shlex.split(content)
            assert "--enable-prefix-caching" not in shlex.split(content)
        if bash:
            checked = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True, check=False)
            assert checked.returncode == 0, checked.stderr


def test_cleanup_aggregates_all_replica_ports_and_cards_per_container(tmp_path, pd_plugin):
    from vllm_pd_simulator.pd_cluster_simulator import PdClusterSimulator

    simulator = PdClusterSimulator.__new__(PdClusterSimulator)
    simulator._executors = {}
    simulator._ssh_cmd_timeout = 120
    simulator._node_infos = None
    simulator.config = cluster(pd_plugin)
    simulator._is_moe = False
    simulator.configure_pd_pipeline(
        make_runner(tmp_path, pd_plugin)._plans(phase_results()),
        {"prefill": 1, "decode": 2, "prefill_devices": 4, "decode_devices": 8},
        {},
    )
    for node in simulator._all_nodes():
        args = shlex.split(simulator._cleanup_command_args(node))
        assert args[args.index("--gpus") + 1] == ",".join(map(str, range(12)))
        ports = set(map(int, args[args.index("--port") + 1].split(",")))
        expected = {8000}
        for group in simulator.config.prefill_groups + simulator.config.decode_groups:
            for member in group.nodes:
                expected.update((member.service_port, member.kv_port))
        assert ports == expected


def test_cleanup_decode_uses_own_service_port_without_remote_proxy(tmp_path, pd_plugin):
    from vllm_pd_simulator.pd_cluster_simulator import PdClusterSimulator

    simulator = PdClusterSimulator.__new__(PdClusterSimulator)
    simulator._executors = {}
    simulator._ssh_cmd_timeout = 120
    simulator.config = cluster(pd_plugin)
    simulator._is_moe = True
    runner = make_runner(tmp_path, pd_plugin)
    runner.settings.pd_disagg.decode.simulator_command_overrides = {
        "others": "--tensor-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel"
    }
    simulator.configure_pd_pipeline(
        runner._plans(phase_results()),
        {"prefill": 1, "decode": 2, "prefill_devices": 4, "decode_devices": 8},
        {},
    )
    for group in simulator.config.decode_groups:
        for node in group.nodes:
            node.ssh_ip = "192.0.2.11"
    simulator._node_infos = [
        {"label": f"D-I{index}-R{rank}", "_node_config": node, "_local_dp_rank": rank}
        for index, group in enumerate(simulator.config.decode_groups)
        for node in group.nodes
        for rank in range(2)
    ]
    decode = simulator.config.decode_groups[0].nodes[0]
    args = shlex.split(simulator._cleanup_command_args(decode))
    ports = set(map(int, args[args.index("--port") + 1].split(",")))
    assert decode.service_port in ports and decode.service_port + 1 in ports
    assert decode.kv_port in ports and decode.kv_port + 2 in ports
    assert simulator.config.proxy.service_port not in ports


def test_cleanup_failure_is_reported_after_attempting_every_container(pd_plugin):
    from vllm_pd_simulator.pd_cluster_simulator import PdClusterSimulator

    simulator = PdClusterSimulator.__new__(PdClusterSimulator)
    simulator._executors = {}
    simulator._ssh_cmd_timeout = 120
    nodes = [pd_plugin.config.ClusterNodeConfig(ssh_ip=f"192.0.2.{index}") for index in (10, 11)]
    with (
        patch.object(simulator, "_all_nodes", return_value=nodes),
        patch.object(
            simulator,
            "_stop_node",
            side_effect=[
                SimpleNamespace(stdout="still alive", failed=True, returncode=1),
                SimpleNamespace(stdout="stopped", failed=False, returncode=0),
            ],
        ) as stop,
        pytest.raises(RuntimeError, match="PD cleanup failed.*still alive"),
    ):
        simulator._kill_all_processes()
    assert stop.call_count == 2


@pytest.mark.parametrize("full_service", [None, {}, {"enabled": False, "target_field": load_fields()}])
def test_disabled_full_service_retains_config_and_skips_deployment(tmp_path, full_service):
    configured = Settings(output=tmp_path, pd_disagg=PdDisaggConfig(full_service=full_service, top_k=0))
    if full_service is not None:
        assert configured.pd_disagg.full_service.enabled is False
    results = phase_results()
    phase_runner = MagicMock()
    phase_runner.run.side_effect = [
        (
            OptimizationResult(
                fitness=1,
                params=result.best_params,
                performance_index=result.performance_index,
                service_param_names=("MAX_NUM_SEQS",),
                benchmark_param_names=("CONCURRENCY",),
            ),
            tmp_path / "missing.csv",
        )
        for result in results.values()
    ]
    with patch("optix.optimizer.pd_full_service.OptixFullServiceRunner") as full_cls:
        summary = PdDisaggOrchestrator(configured, "disabled", phase_runner).run()
        full_cls.assert_not_called()
    assert [call.args[0] for call in phase_runner.run.call_args_list] == ["prefill", "decode"]
    assert summary["status"] == "service_search_completed"
    assert "full_service" not in summary
    assert not (tmp_path / "pd_disagg/disabled/full_service").exists()
    assert not (tmp_path / "pd_disagg/disabled/final_result.md").exists()
    assert "final_report" not in summary["artifacts"]


def test_enabling_full_service_revalidates_retained_config(tmp_path):
    configured = settings(tmp_path).pd_disagg.model_dump()
    configured["full_service"]["enabled"] = False
    assert PdDisaggConfig.model_validate(configured).full_service.target_field
    configured["full_service"]["enabled"] = True
    assert PdDisaggConfig.model_validate(configured).full_service.enabled
    configured["full_service"]["target_field"] = []
    with pytest.raises(ValidationError, match="requires CONCURRENCY"):
        PdDisaggConfig.model_validate(configured)


@pytest.mark.parametrize("slo_met", [False, True])
@pytest.mark.parametrize("rounds", [1, 2, 3])
@pytest.mark.parametrize("ttft_slo", [None, 7.25])
def test_final_report_contains_selected_results_and_complete_deployment(tmp_path, slo_met, pd_plugin, rounds, ttft_slo):
    from optix.optimizer.pd_report import write_final_report

    runner = make_runner(tmp_path, pd_plugin)
    config = runner.settings.pd_disagg
    config.decode.benchmark_run_count = 3
    config.full_service.benchmark_run_count = rounds
    if ttft_slo is not None:
        config.full_service.ttft_slo = ttft_slo
    config.full_service.benchmark_command_overrides = {"others": "--random-input-len 3500 --random-output-len 1500"}
    results = phase_results()
    summary = {"run_id": "one-file", "status": "completed", "pd_ratio": 0.5, "artifacts": {}}
    for role, result in results.items():
        directory = tmp_path / role
        directory.mkdir()
        (directory / "phase_result.json").write_text(json.dumps(result.to_dict()), encoding="utf-8")
        summary[role] = PdDisaggOrchestrator._summary_phase(result)
    summary["full_service"] = {
        "slo_met": slo_met,
        "benchmark_params": {"CONCURRENCY": 4, "REQUESTRATE": 0},
        "performance_index": {
            "throughput": 0.03,
            "generate_speed": 44.9876,
            "time_to_first_token": 0.4,
            "time_per_output_token": 0.01 if slo_met else 0.0851,
            "success_rate": 1.0,
        },
    }
    deployment = {
        "instances": {"prefill": 1, "decode": 2, "prefill_devices": 4, "decode_devices": 8},
        "plans": runner._plans(results),
        "config": {
            "prefill_groups": [{"nodes": [{"ssh_ip": "192.0.2.10", "gpu_ids": [0, 1, 2, 3]}]}],
            "decode_groups": [{"nodes": [{"ssh_ip": "192.0.2.20", "gpu_ids": list(range(8))}]}],
            "proxy": {"bind_ip": "192.0.2.10", "service_port": 8000},
        },
    }
    (tmp_path / "full_service").mkdir()
    (tmp_path / "full_service/deployment.json").write_text(json.dumps(deployment), encoding="utf-8")
    path = write_final_report(tmp_path, summary, config)
    text = path.read_text(encoding="utf-8")
    assert ("完整服务 SLO：达标" if slo_met else "完整服务 SLO：未达标") in text
    assert "0.0851" in text or slo_met
    assert "44.9876" in text and "0.03" in text
    assert "| MAX_NUM_SEQS | 16 |" in text and "| MAX_NUM_SEQS | 64 |" in text
    assert "| CONCURRENCY | 4 |" in text and "| REQUESTRATE | 0 |" in text
    assert "pd_mixed，在各字段配置范围内联合调整并发和请求率" in text
    assert "顶层请求率标定开关不作用于本阶段" in text
    assert "| CONCURRENCY | 1.0 | 32.0 | 4 | 4 | 完整服务实测候选择优（含基线） |" in text
    assert "192.0.2.10" in text and "192.0.2.20" in text
    assert "--random-input-len 3500 --random-output-len 1500" in text
    assert "仅最后一轮计分" in text
    assert f"完整服务每个负载候选压测轮数：{rounds}" in text
    if rounds > 1:
        assert f"前 {rounds - 1} 轮预热，仅第 {rounds} 轮计分" in text
        assert "各轮复用同一套 PD 服务，候选负载参数保持不变" in text
    else:
        assert "单轮直接计分，未额外安排压测预热轮次" in text
    origin = "默认值：0.5" if ttft_slo is None else "显式配置：7.25"
    assert f"TTFT 目标来源：pd_disagg.full_service.ttft_slo（{origin} 秒）" in text
    assert json.dumps(deployment, ensure_ascii=False, indent=2) in text
    assert "此部署记录未保存原始设备池总量，无法确定池内闲置卡数" in text
    assert "理论 P:D = 0.5:1" in text


def test_preflight_failure_still_outputs_one_report(tmp_path):
    configured = settings(tmp_path)
    runner = MagicMock()
    with patch("optix.optimizer.pd_full_service.OptixFullServiceRunner") as full_cls:
        full_cls.return_value.preflight.side_effect = RuntimeError("missing plugin")
        with pytest.raises(PdDisaggError, match="missing plugin"):
            PdDisaggOrchestrator(configured, "preflight", runner).run()
    runner.run.assert_not_called()
    report = tmp_path / "pd_disagg/preflight/final_result.md"
    text = report.read_text(encoding="utf-8")
    assert "失败阶段：full_service_preflight" in text
    assert "missing plugin" in text and "未测量" in text
    assert "尚未生成完整服务部署配置" in text
    assert "未生成完整服务选参结果" in text


def test_report_write_failure_does_not_hide_original_failure(tmp_path):
    with (
        patch("optix.optimizer.pd_full_service.OptixFullServiceRunner") as full_cls,
        patch("optix.optimizer.pd_report.write_final_report", side_effect=OSError("report disk error")),
    ):
        full_cls.return_value.preflight.side_effect = RuntimeError("original preflight error")
        with pytest.raises(PdDisaggError, match="original preflight error"):
            PdDisaggOrchestrator(settings(tmp_path), "failure", MagicMock()).run()
    saved = json.loads((tmp_path / "pd_disagg/failure/pd_disagg_summary.json").read_text(encoding="utf-8"))
    assert saved["error"]["message"] == "original preflight error"
