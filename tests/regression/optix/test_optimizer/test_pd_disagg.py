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

"""Tests for the built-in PD disaggregation tuning workflow."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from optix.config.config import (
    OptimizerConfigField,
    PdDisaggConfig,
    PdDisaggPhaseConfig,
    PerformanceIndex,
    Settings,
)
from optix.optimizer.errors import PdDisaggError
from optix.optimizer.outcome import OptimizationResult
from optix.optimizer.pd_disagg import (
    DECODE_PHASE,
    PREFILL_PHASE,
    OptixPhaseRunner,
    PdDisaggOrchestrator,
    allocate_instances,
    calculate_pd_ratio,
    calculate_phase_qps,
)


def _optimization_result(
    *,
    concurrency: float,
    throughput: float | None = None,
    ttft: float = 0.4,
    tpot: float = 0.2,
) -> OptimizationResult:
    return OptimizationResult(
        fitness=1.0,
        params={"CONCURRENCY": concurrency},
        performance_index=PerformanceIndex(
            generate_speed=throughput or 1.0,
            throughput=throughput,
            time_to_first_token=ttft,
            time_per_output_token=tpot,
            success_rate=1.0,
        ),
    )


class _FakePhaseRunner:
    def __init__(self, *, fail_phase: str | None = None) -> None:
        self.fail_phase = fail_phase
        self.calls = []

    def run(self, phase_name, phase_config, phase_output_dir):
        self.calls.append(phase_name)
        if phase_name == self.fail_phase:
            raise RuntimeError(f"{phase_name} failed")
        phase_output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = phase_output_dir / "data_storage_test.csv"
        if phase_name == PREFILL_PHASE:
            csv_path.write_text(
                "fitness,throughput,time_to_first_token,time_per_output_token\n1,10,0.4,0.2\n2,8,0.5,0.2\n",
                encoding="utf-8",
            )
            result = _optimization_result(concurrency=4, throughput=10)
            result.params.clear()
            return result, csv_path
        if phase_name == DECODE_PHASE:
            csv_path.write_text(
                "fitness,throughput,time_to_first_token,time_per_output_token\n1,15,0.4,0.2\n2,12,0.4,0.2\n",
                encoding="utf-8",
            )
            result = _optimization_result(concurrency=3, throughput=15, tpot=0.2)
            result.params.clear()
            return result, csv_path
        raise AssertionError(f"unexpected phase: {phase_name}")


class TestPdDisaggConfig:
    def test_default_mode_contains_only_prefill_and_decode_phases(self):
        config = PdDisaggConfig()

        assert not hasattr(config, "enabled")
        assert config.prefill.fine_tune_mode == "pd_mixed"
        assert config.decode.fine_tune_mode == "pd_disaggregation"
        assert config.prefill.engine == "vllm"
        assert config.decode.engine == "vllm"
        assert config.prefill.benchmark_policy == "vllm_benchmark"
        assert config.decode.benchmark_policy == "vllm_benchmark"
        assert not hasattr(config, "resume_phase")
        assert not hasattr(config, "resume_run_id")
        assert not hasattr(config, "final_benchmark")
        assert not hasattr(config.prefill, "search_scope")
        assert config.prefill.n_particles is None
        assert config.prefill.iters is None
        assert config.decode.n_particles is None
        assert config.decode.iters is None
        assert config.prefill.benchmark_run_count == 1
        assert config.decode.benchmark_run_count == 1

    def test_benchmark_run_count_must_be_positive(self):
        with pytest.raises(ValidationError, match="benchmark_run_count"):
            PdDisaggPhaseConfig(benchmark_run_count=0)

    def test_request_rate_calibration_rejects_repeated_phase_benchmarks(self, tmp_path):
        pd_disagg = PdDisaggConfig(
            prefill=PdDisaggPhaseConfig(benchmark_run_count=2),
            decode=PdDisaggPhaseConfig(benchmark_run_count=3),
        )

        with pytest.raises(
            ValidationError,
            match=r"benchmark_run_count > 1 requires use_request_rate_calibration=false.*prefill, decode",
        ):
            Settings(output=tmp_path, use_request_rate_calibration=True, pd_disagg=pd_disagg)

    def test_repeated_phase_benchmarks_are_allowed_without_request_rate_calibration(self, tmp_path):
        settings = Settings(
            output=tmp_path,
            use_request_rate_calibration=False,
            pd_disagg=PdDisaggConfig(decode=PdDisaggPhaseConfig(benchmark_run_count=2)),
        )

        assert settings.pd_disagg.decode.benchmark_run_count == 2

    def test_top_k_zero_disables_phase_fine_tune(self):
        assert PdDisaggConfig(top_k=0).top_k == 0

    def test_negative_top_k_is_rejected(self):
        with pytest.raises(ValidationError, match="top_k"):
            PdDisaggConfig(top_k=-1)

    def test_prefill_and_decode_accept_independent_search_budgets(self):
        config = PdDisaggConfig(
            prefill=PdDisaggPhaseConfig(n_particles=8, iters=4),
            decode=PdDisaggPhaseConfig(n_particles=12, iters=6),
        )

        assert (config.prefill.n_particles, config.prefill.iters) == (8, 4)
        assert (config.decode.n_particles, config.decode.iters) == (12, 6)

    def test_partial_device_configuration_is_rejected(self):
        with pytest.raises(ValidationError, match="must be configured together"):
            PdDisaggConfig(total_devices=16, prefill_devices_per_instance=4)

    def test_non_vllm_phase_engine_is_rejected(self):
        with pytest.raises(ValidationError, match="supports only the 'vllm' engine"):
            PdDisaggConfig(prefill=PdDisaggPhaseConfig(engine="mindie"))

    @pytest.mark.parametrize(
        ("field_name", "value"),
        (
            ("enabled", True),
            ("resume_phase", "decode"),
            ("resume_run_id", "run123"),
            ("final_benchmark", {}),
        ),
    )
    def test_removed_orchestration_fields_are_rejected(self, field_name, value):
        with pytest.raises(ValidationError) as error_info:
            PdDisaggConfig(**{field_name: value})

        assert any(
            error["loc"] == (field_name,) and error["type"] == "extra_forbidden" for error in error_info.value.errors()
        )

    @pytest.mark.parametrize("field_name", ("search_scope", "reuse_service"))
    def test_removed_phase_fields_are_rejected(self, field_name):
        with pytest.raises(ValidationError) as error_info:
            PdDisaggPhaseConfig(**{field_name: "benchmark" if field_name == "search_scope" else True})

        assert any(
            error["loc"] == (field_name,) and error["type"] == "extra_forbidden" for error in error_info.value.errors()
        )


class TestPdRatioCalculation:
    def test_prefill_uses_benchmark_throughput_without_concurrency(self):
        result = _optimization_result(concurrency=4, throughput=12.5, ttft=0.4)
        result.params.clear()

        qps, source = calculate_phase_qps(PREFILL_PHASE, PdDisaggPhaseConfig(), result)

        assert qps == 12.5
        assert source == "benchmark_throughput"

    def test_prefill_missing_throughput_fails_instead_of_using_ttft(self):
        result = _optimization_result(concurrency=4, throughput=None, ttft=0.4)

        with pytest.raises(PdDisaggError, match="prefill throughput is missing or invalid"):
            calculate_phase_qps(PREFILL_PHASE, PdDisaggPhaseConfig(), result)

    def test_decode_uses_benchmark_throughput_without_concurrency(self):
        result = _optimization_result(concurrency=3, throughput=15, tpot=0.2)
        result.params.clear()

        qps, source = calculate_phase_qps(DECODE_PHASE, PdDisaggPhaseConfig(), result)

        assert qps == 15
        assert source == "benchmark_throughput"
        assert calculate_pd_ratio(10, qps) == 1.5

    @pytest.mark.parametrize("throughput", (None, 0, float("nan"), float("inf")))
    def test_decode_invalid_throughput_fails_instead_of_using_concurrency_or_tpot(self, throughput):
        result = _optimization_result(concurrency=3, throughput=throughput, tpot=0.2)

        with pytest.raises(PdDisaggError, match="decode throughput"):
            calculate_phase_qps(DECODE_PHASE, PdDisaggPhaseConfig(), result)


class TestInstanceAllocation:
    def test_rfc_example_selects_three_prefill_and_two_decode_instances(self):
        candidates = allocate_instances(
            total_devices=16,
            prefill_devices_per_instance=4,
            decode_devices_per_instance=2,
            pd_ratio=1.5,
            prefill_qps=10,
            decode_qps=15,
            use_full_device=True,
            top_k=3,
        )

        assert (candidates[0].prefill, candidates[0].decode) == (3, 2)
        assert candidates[0].remaining_devices == 0
        assert candidates[0].balanced_qps == 30

    def test_no_feasible_full_device_allocation_returns_empty_list(self):
        candidates = allocate_instances(
            total_devices=3,
            prefill_devices_per_instance=2,
            decode_devices_per_instance=2,
            pd_ratio=1,
            prefill_qps=10,
            decode_qps=10,
            use_full_device=True,
            top_k=3,
        )

        assert candidates == []

    def test_resource_limit_can_force_one_prefill_and_one_decode_instance(self):
        candidates = allocate_instances(
            total_devices=16,
            prefill_devices_per_instance=8,
            decode_devices_per_instance=8,
            pd_ratio=0.15,
            prefill_qps=7.5,
            decode_qps=1.125,
            use_full_device=True,
            top_k=3,
        )

        assert len(candidates) == 1
        assert (candidates[0].prefill, candidates[0].decode) == (1, 1)
        assert candidates[0].actual_ratio == 1
        assert candidates[0].ratio_error == pytest.approx(0.85)
        assert candidates[0].balanced_qps == 1.125


class TestPdDisaggOrchestrator:
    def _settings(self, tmp_path: Path) -> Settings:
        return Settings(
            output=tmp_path,
            pd_disagg=PdDisaggConfig(
                total_devices=16,
                prefill_devices_per_instance=4,
                decode_devices_per_instance=2,
            ),
        )

    def test_service_search_stops_after_ratio_recommendation(self, tmp_path):
        runner = _FakePhaseRunner()
        orchestrator = PdDisaggOrchestrator(self._settings(tmp_path), "run123", runner)

        summary = orchestrator.run()

        assert runner.calls == [PREFILL_PHASE, DECODE_PHASE]
        assert summary["status"] == "service_search_completed"
        assert summary["run_id"] == "run123"
        assert summary["pd_ratio"] == 1.5
        assert summary["decode"]["qps"] == 15
        assert summary["decode"]["qps_source"] == "benchmark_throughput"
        assert summary["instances"]["prefill"] == 3
        assert "next_step" not in summary
        summary_path = tmp_path / "pd_disagg" / "run123" / "pd_disagg_summary.json"
        assert json.loads(summary_path.read_text(encoding="utf-8"))["status"] == "service_search_completed"
        assert (summary_path.parent / "pd_ratio_candidates.csv").is_file()
        candidate_lines = (summary_path.parent / "pd_ratio_candidates.csv").read_text(encoding="utf-8").splitlines()
        assert len(candidate_lines) == 5  # header plus two P x two D combinations

    def test_top_k_zero_still_outputs_selected_ratio_and_instance_recommendation(self, tmp_path):
        settings = self._settings(tmp_path)
        settings.pd_disagg.top_k = 0
        orchestrator = PdDisaggOrchestrator(settings, "run123", _FakePhaseRunner())

        summary = orchestrator.run()

        assert summary["status"] == "service_search_completed"
        assert summary["instances"] is not None
        candidates_path = tmp_path / "pd_disagg" / "run123" / "pd_ratio_candidates.csv"
        assert len(candidates_path.read_text(encoding="utf-8").splitlines()) == 2

    def test_phase_artifact_separates_service_and_benchmark_params(self, tmp_path):
        phase_runner = MagicMock()
        csv_path = tmp_path / "phase.csv"
        phase_runner.run.return_value = (
            OptimizationResult(
                fitness=1.0,
                params={"SERVICE": 7, "CONCURRENCY": 4},
                performance_index=PerformanceIndex(throughput=12.5),
                service_param_names=("SERVICE",),
                benchmark_param_names=("CONCURRENCY",),
            ),
            csv_path,
        )
        orchestrator = PdDisaggOrchestrator(self._settings(tmp_path), "run123", phase_runner)

        result = orchestrator._run_phase(PREFILL_PHASE)

        assert result.service_params == {"SERVICE": 7}
        assert result.benchmark_params == {"CONCURRENCY": 4}
        artifact = json.loads(orchestrator._phase_result_path(PREFILL_PHASE).read_text(encoding="utf-8"))
        assert artifact["service_params"] == {"SERVICE": 7}
        assert artifact["benchmark_params"] == {"CONCURRENCY": 4}

    def test_decode_failure_preserves_prefill_artifacts_and_failed_summary(self, tmp_path):
        runner = _FakePhaseRunner(fail_phase=DECODE_PHASE)
        orchestrator = PdDisaggOrchestrator(self._settings(tmp_path), "run123", runner)

        with pytest.raises(PdDisaggError, match="decode"):
            orchestrator.run()

        run_dir = tmp_path / "pd_disagg" / "run123"
        assert (run_dir / PREFILL_PHASE / "phase_result.json").is_file()
        summary = json.loads((run_dir / "pd_disagg_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "failed"
        assert summary["failed_phase"] == DECODE_PHASE
        assert summary[PREFILL_PHASE]["status"] == "success"


class _BenchmarkPlugin:
    def __init__(self, **kwargs):
        self.data_field = (
            OptimizerConfigField(name="CONCURRENCY", config_position="env", min=1, max=8, dtype="int", value=4),
        )

    def stop(self, del_log=True):
        del del_log


class _SimulatorPlugin:
    def __init__(self, **kwargs):
        self.data_field = (OptimizerConfigField(name="SERVICE", min=1, max=2, value=1),)


class TestOptixPhaseRunner:
    @pytest.mark.parametrize(
        ("global_others", "phase_others", "should_warn"),
        [
            ("--datasets decode_data", None, True),
            ("--datasets decode_data --summarizer stable_stage", None, False),
            ("--datasets decode_data", "--datasets decode_data --summarizer stable_stage", False),
            ("--datasets decode_data --summarizer stable_stage", "--datasets decode_data", True),
        ],
    )
    def test_decode_aisbench_stable_stage_warning(self, tmp_path, global_others, phase_others, should_warn):
        settings = Settings(output=tmp_path)
        settings.ais_bench.command.others = global_others
        runner = OptixPhaseRunner(
            settings,
            runtime_ctx=MagicMock(),
            deploy_env={},
            bak_path=None,
            load_breakpoint=False,
            top_k=3,
        )
        phase = PdDisaggPhaseConfig(
            benchmark_policy="ais_bench",
            benchmark_command_overrides={} if phase_others is None else {"others": phase_others},
        )
        with patch("optix.optimizer.pd_disagg.logger.warning") as warning:
            runner._warn_if_decode_aisbench_not_stable(PdDisaggConfig(decode=phase))

        assert warning.called is should_warn

    def test_decode_aisbench_warning_does_not_apply_to_other_benchmarks(self, tmp_path):
        settings = Settings(output=tmp_path)
        settings.ais_bench.command.others = "--datasets decode_data"
        runner = OptixPhaseRunner(
            settings,
            runtime_ctx=MagicMock(),
            deploy_env={},
            bak_path=None,
            load_breakpoint=False,
            top_k=3,
        )
        with patch("optix.optimizer.pd_disagg.logger.warning") as warning:
            runner._warn_if_decode_aisbench_not_stable(PdDisaggConfig())

        warning.assert_not_called()

    def test_preflight_checks_decode_aisbench_summary_mode(self, tmp_path):
        settings = Settings(output=tmp_path)
        runner = OptixPhaseRunner(
            settings,
            runtime_ctx=MagicMock(),
            deploy_env={},
            bak_path=None,
            load_breakpoint=False,
            top_k=3,
        )
        config = PdDisaggConfig()
        with (
            patch.object(runner, "_resolve_and_validate_plugins"),
            patch.object(runner, "_warn_if_decode_aisbench_not_stable") as warn,
        ):
            runner.preflight(config)

        warn.assert_called_once_with(config)

    def test_resolve_and_validate_plugins_returns_registered_classes(self, tmp_path):
        runtime_ctx = MagicMock()
        deploy_env = {"PATH": "/usr/bin"}
        runner = OptixPhaseRunner(
            Settings(output=tmp_path),
            runtime_ctx=runtime_ctx,
            deploy_env=deploy_env,
            bak_path=None,
            load_breakpoint=False,
            top_k=3,
        )
        phase = PdDisaggPhaseConfig(engine="pd_plugin", benchmark_policy="pd_benchmark")

        with (
            patch.dict("optix.optimizer.register.simulates", {"pd_plugin": _SimulatorPlugin}),
            patch.dict("optix.optimizer.register.benchmarks", {"pd_benchmark": _BenchmarkPlugin}),
            patch("optix.optimizer.register.validate_simulator_policy") as validate_simulator,
            patch("optix.optimizer.register.validate_benchmark_policy") as validate_benchmark,
            patch("optix.deploy_env.validate_deploy_stack") as validate_stack,
        ):
            simulator_cls, benchmark_cls = runner._resolve_and_validate_plugins(PREFILL_PHASE, phase)

        assert simulator_cls is _SimulatorPlugin
        assert benchmark_cls is _BenchmarkPlugin
        validate_simulator.assert_called_once_with("pd_plugin")
        validate_benchmark.assert_called_once_with("pd_benchmark")
        validate_stack.assert_called_once_with(
            engine="pd_plugin",
            benchmark="pd_benchmark",
            env=deploy_env,
            ctx=runtime_ctx,
        )

    def test_run_rejects_disabled_simulator_lifecycle(self, tmp_path):
        settings = Settings(
            output=tmp_path,
            skip_pso=True,
            n_particles=0,
            iters=0,
            manage_simulator_lifecycle=False,
        )
        runner = OptixPhaseRunner(
            settings,
            runtime_ctx=MagicMock(),
            deploy_env={},
            bak_path=None,
            load_breakpoint=False,
            top_k=0,
        )

        with pytest.raises(PdDisaggError, match="requires manage_simulator_lifecycle=true"):
            runner.run(PREFILL_PHASE, PdDisaggPhaseConfig(), tmp_path / PREFILL_PHASE)

    def test_preflight_rejects_missing_plugin_before_search(self, tmp_path):
        runner = OptixPhaseRunner(
            Settings(output=tmp_path),
            runtime_ctx=MagicMock(),
            deploy_env={},
            bak_path=None,
            load_breakpoint=False,
            top_k=3,
        )

        with (
            patch.dict("optix.optimizer.register.simulates", {}, clear=True),
            patch.dict("optix.optimizer.register.benchmarks", {}, clear=True),
            pytest.raises(PdDisaggError, match="is not registered"),
        ):
            runner.preflight(PdDisaggConfig())

    def test_explicit_target_fields_replace_config_backed_plugin_fields(self, tmp_path):
        runner = OptixPhaseRunner(
            Settings(output=tmp_path),
            runtime_ctx=MagicMock(),
            deploy_env={},
            bak_path=None,
            load_breakpoint=False,
            top_k=3,
        )
        service = OptimizerConfigField(name="SERVICE", min=1, max=2, value=1)
        ignored_service = OptimizerConfigField(name="OTHER_SERVICE", min=1, max=2, value=1)
        concurrency = OptimizerConfigField(name="CONCURRENCY", min=1, max=8, value=4)

        class ConfigBackedPlugin:
            def __init__(self, fields):
                self.config = SimpleNamespace(target_field=list(fields))

            @property
            def data_field(self):
                return tuple(self.config.target_field)

        simulator = ConfigBackedPlugin([service, ignored_service])
        benchmark = ConfigBackedPlugin([concurrency])
        phase = PdDisaggPhaseConfig(target_field=[service, concurrency])

        resolved = runner._resolve_target_fields(simulator, benchmark, phase)

        assert [field.name for field in resolved] == ["SERVICE", "CONCURRENCY"]
        assert [field.name for field in simulator.config.target_field] == ["SERVICE"]
        assert [field.name for field in benchmark.config.target_field] == ["CONCURRENCY"]

    def test_native_load_fields_are_owned_by_benchmark_without_mutating_shared_config(self, tmp_path):
        runner = OptixPhaseRunner(
            Settings(output=tmp_path),
            runtime_ctx=MagicMock(),
            deploy_env={},
            bak_path=None,
            load_breakpoint=False,
            top_k=3,
        )

        class ConfigBackedPlugin:
            def __init__(self, fields):
                self.config = SimpleNamespace(target_field=list(fields))

            @property
            def data_field(self):
                return tuple(self.config.target_field)

        service = OptimizerConfigField(name="SERVICE", min=1, max=2, value=1)
        concurrency = OptimizerConfigField(name="CONCURRENCY", min=1, max=8, value=4)
        request_rate = OptimizerConfigField(name="REQUESTRATE", min=1, max=8, value=4)
        simulator = ConfigBackedPlugin([service, concurrency, request_rate])
        benchmark = ConfigBackedPlugin([])
        shared_simulator_config = simulator.config

        resolved = runner._resolve_target_fields(simulator, benchmark, PdDisaggPhaseConfig())

        assert [field.name for field in resolved] == ["SERVICE", "CONCURRENCY", "REQUESTRATE"]
        assert [field.name for field in simulator.data_field] == ["SERVICE"]
        assert [field.name for field in benchmark.data_field] == ["CONCURRENCY", "REQUESTRATE"]
        assert [field.name for field in shared_simulator_config.target_field] == [
            "SERVICE",
            "CONCURRENCY",
            "REQUESTRATE",
        ]

    def test_phase_local_benchmark_command_override_does_not_mutate_shared_config(self, tmp_path):
        runner = OptixPhaseRunner(
            Settings(output=tmp_path),
            runtime_ctx=MagicMock(),
            deploy_env={},
            bak_path=None,
            load_breakpoint=False,
            top_k=3,
        )
        shared_config = SimpleNamespace(
            command=SimpleNamespace(others="--random-output-len 256"),
            target_field=[],
        )
        benchmark = SimpleNamespace(config=shared_config, update_command=MagicMock())
        phase = PdDisaggPhaseConfig(benchmark_command_overrides={"others": "--random-output-len 1"})

        runner._apply_benchmark_command_overrides(benchmark, PREFILL_PHASE, phase)

        assert shared_config.command.others == "--random-output-len 256"
        assert benchmark.config.command.others == "--random-output-len 1"
        benchmark.update_command.assert_called_once_with()

    def test_prefill_and_decode_have_independent_simulator_commands_and_search_fields(self, tmp_path):
        runner = OptixPhaseRunner(
            Settings(output=tmp_path),
            runtime_ctx=MagicMock(),
            deploy_env={},
            bak_path=None,
            load_breakpoint=False,
            top_k=3,
        )
        shared_config = SimpleNamespace(command=SimpleNamespace(others="--common"), target_field=[])
        prefill_simulator = SimpleNamespace(config=shared_config, update_command=MagicMock())
        decode_simulator = SimpleNamespace(config=shared_config, update_command=MagicMock())
        prefill_field = OptimizerConfigField(name="P_SERVICE", min=1, max=2, value=1)
        decode_field = OptimizerConfigField(name="D_SERVICE", min=3, max=4, value=3)
        prefill = PdDisaggPhaseConfig(
            simulator_command_overrides={"others": "--tensor-parallel-size 4"},
            target_field=[prefill_field],
        )
        decode = PdDisaggPhaseConfig(
            simulator_command_overrides={"others": "--tensor-parallel-size 2"},
            target_field=[decode_field],
        )

        runner._apply_simulator_command_overrides(prefill_simulator, PREFILL_PHASE, prefill)
        runner._apply_simulator_command_overrides(decode_simulator, DECODE_PHASE, decode)

        assert shared_config.command.others == "--common"
        assert prefill_simulator.config.command.others == "--tensor-parallel-size 4"
        assert decode_simulator.config.command.others == "--tensor-parallel-size 2"
        assert [field.name for field in prefill.target_field] == ["P_SERVICE"]
        assert [field.name for field in decode.target_field] == ["D_SERVICE"]
        prefill_simulator.update_command.assert_called_once_with()
        decode_simulator.update_command.assert_called_once_with()

    def test_unknown_benchmark_command_override_is_rejected(self, tmp_path):
        runner = OptixPhaseRunner(
            Settings(output=tmp_path),
            runtime_ctx=MagicMock(),
            deploy_env={},
            bak_path=None,
            load_breakpoint=False,
            top_k=3,
        )
        benchmark = SimpleNamespace(config=SimpleNamespace(command=SimpleNamespace(others=""), target_field=[]))
        phase = PdDisaggPhaseConfig(benchmark_command_overrides={"unknown": "value"})

        with pytest.raises(PdDisaggError, match="has no fields"):
            runner._apply_benchmark_command_overrides(benchmark, PREFILL_PHASE, phase)

    def test_unknown_simulator_command_override_is_rejected(self, tmp_path):
        runner = OptixPhaseRunner(
            Settings(output=tmp_path),
            runtime_ctx=MagicMock(),
            deploy_env={},
            bak_path=None,
            load_breakpoint=False,
            top_k=3,
        )
        simulator = SimpleNamespace(config=SimpleNamespace(command=SimpleNamespace(others=""), target_field=[]))
        phase = PdDisaggPhaseConfig(simulator_command_overrides={"unknown": "value"})

        with pytest.raises(PdDisaggError, match="simulator command has no fields"):
            runner._apply_simulator_command_overrides(simulator, PREFILL_PHASE, phase)

    def test_decode_rejects_fixed_request_rate_before_search(self, tmp_path):
        runner = OptixPhaseRunner(
            Settings(output=tmp_path),
            runtime_ctx=MagicMock(),
            deploy_env={},
            bak_path=None,
            load_breakpoint=False,
            top_k=3,
        )
        phase = PdDisaggPhaseConfig(
            engine="pd_plugin",
            benchmark_policy="pd_benchmark",
            fine_tune_mode="pd_disaggregation",
            target_field=[
                OptimizerConfigField(name="SERVICE", min=1, max=2, value=1),
                OptimizerConfigField(name="CONCURRENCY", min=1, max=8, value=4),
                OptimizerConfigField(name="REQUESTRATE", min=0, max=0, value=0),
            ],
        )

        with (
            patch.dict("optix.optimizer.register.simulates", {"pd_plugin": _SimulatorPlugin}),
            patch.dict("optix.optimizer.register.benchmarks", {"pd_benchmark": _BenchmarkPlugin}),
            patch("optix.optimizer.register.validate_simulator_policy"),
            patch("optix.optimizer.register.validate_benchmark_policy"),
            patch("optix.deploy_env.validate_deploy_stack"),
            pytest.raises(PdDisaggError, match="configurable REQUESTRATE range"),
        ):
            runner.run(DECODE_PHASE, phase, tmp_path / DECODE_PHASE)

    def test_top_k_zero_skips_decode_fine_tune_field_validation(self, tmp_path):
        runner = OptixPhaseRunner(
            Settings(output=tmp_path),
            runtime_ctx=MagicMock(),
            deploy_env={},
            bak_path=None,
            load_breakpoint=False,
            top_k=0,
        )
        phase = PdDisaggPhaseConfig(
            engine="pd_plugin",
            benchmark_policy="pd_benchmark",
            fine_tune_mode="pd_disaggregation",
            target_field=[
                OptimizerConfigField(name="SERVICE", min=1, max=2, value=1),
                OptimizerConfigField(name="CONCURRENCY", min=1, max=8, value=4),
                OptimizerConfigField(name="REQUESTRATE", min=0, max=0, value=0),
            ],
        )
        storage = MagicMock(save_file=tmp_path / "data.csv")
        optimizer = MagicMock()
        optimizer.run_plugin.return_value = _optimization_result(concurrency=4, throughput=20)

        with (
            patch.dict("optix.optimizer.register.simulates", {"pd_plugin": _SimulatorPlugin}),
            patch.dict("optix.optimizer.register.benchmarks", {"pd_benchmark": _BenchmarkPlugin}),
            patch("optix.optimizer.register.validate_simulator_policy"),
            patch("optix.optimizer.register.validate_benchmark_policy"),
            patch("optix.deploy_env.validate_deploy_stack"),
            patch("optix.optimizer.store.DataStorage", return_value=storage) as storage_cls,
            patch("optix.optimizer.scheduler.Scheduler"),
            patch("optix.optimizer.experience_fine_tunning.FineTune"),
            patch("optix.optimizer.optimizer.PSOOptimizer", return_value=optimizer),
        ):
            runner.run(DECODE_PHASE, phase, tmp_path / DECODE_PHASE)

        storage_config = storage_cls.call_args.args[0]
        assert storage_config.pso_top_k == 0
        optimizer.run_plugin.assert_called_once_with()

    def test_prefill_and_decode_use_independent_search_budgets(self, tmp_path):
        settings = Settings(
            output=tmp_path,
            n_particles=5,
            iters=10,
            use_request_rate_calibration=False,
        )
        runner = OptixPhaseRunner(
            settings,
            runtime_ctx=MagicMock(),
            deploy_env={},
            bak_path=None,
            load_breakpoint=False,
            top_k=3,
        )
        storage = MagicMock(save_file=tmp_path / "data.csv")
        optimizer = MagicMock()
        optimizer.run_plugin.return_value = _optimization_result(concurrency=4, throughput=20)
        phases = (
            (
                PREFILL_PHASE,
                PdDisaggPhaseConfig(
                    engine="pd_plugin",
                    benchmark_policy="pd_benchmark",
                    benchmark_run_count=1,
                    n_particles=8,
                    iters=4,
                ),
            ),
            (
                DECODE_PHASE,
                PdDisaggPhaseConfig(
                    engine="pd_plugin",
                    benchmark_policy="pd_benchmark",
                    benchmark_run_count=2,
                    n_particles=12,
                    iters=6,
                ),
            ),
        )
        with (
            patch.dict("optix.optimizer.register.simulates", {"pd_plugin": _SimulatorPlugin}),
            patch.dict("optix.optimizer.register.benchmarks", {"pd_benchmark": _BenchmarkPlugin}),
            patch("optix.optimizer.register.validate_simulator_policy"),
            patch("optix.optimizer.register.validate_benchmark_policy"),
            patch("optix.deploy_env.validate_deploy_stack"),
            patch("optix.optimizer.store.DataStorage", return_value=storage),
            patch("optix.optimizer.scheduler.Scheduler") as scheduler_cls,
            patch("optix.optimizer.experience_fine_tunning.FineTune"),
            patch("optix.optimizer.optimizer.PSOOptimizer", return_value=optimizer) as pso_cls,
        ):
            for phase_name, phase_config in phases:
                runner.run(phase_name, phase_config, tmp_path / phase_name)

        assert [(call.kwargs["n_particles"], call.kwargs["iters"]) for call in pso_cls.call_args_list] == [
            (8, 4),
            (12, 6),
        ]
        assert all(call.kwargs["skip_pso"] is False for call in pso_cls.call_args_list)
        assert all(call.kwargs["manage_simulator_lifecycle"] is True for call in pso_cls.call_args_list)
        assert [call.kwargs["benchmark_run_count"] for call in scheduler_cls.call_args_list] == [1, 2]

    def test_request_rate_calibration_rejects_runtime_phase_benchmark_run_count(self, tmp_path):
        settings = Settings(output=tmp_path, use_request_rate_calibration=True)
        runner = OptixPhaseRunner(
            settings,
            runtime_ctx=MagicMock(),
            deploy_env={},
            bak_path=None,
            load_breakpoint=False,
            top_k=0,
        )
        phase = PdDisaggPhaseConfig(
            engine="pd_plugin",
            benchmark_policy="pd_benchmark",
            benchmark_run_count=2,
            target_field=[OptimizerConfigField(name="SERVICE", min=1, max=2, value=1)],
        )
        storage = MagicMock(save_file=tmp_path / "data.csv")
        optimizer = MagicMock()
        optimizer.run_plugin.return_value = _optimization_result(concurrency=4, throughput=20)

        with (
            patch.dict("optix.optimizer.register.simulates", {"pd_plugin": _SimulatorPlugin}),
            patch.dict("optix.optimizer.register.benchmarks", {"pd_benchmark": _BenchmarkPlugin}),
            patch("optix.optimizer.register.validate_simulator_policy"),
            patch("optix.optimizer.register.validate_benchmark_policy"),
            patch("optix.deploy_env.validate_deploy_stack"),
            patch("optix.optimizer.store.DataStorage", return_value=storage),
            patch("optix.optimizer.scheduler.Scheduler") as scheduler_cls,
            patch("optix.optimizer.experience_fine_tunning.FineTune"),
            patch("optix.optimizer.optimizer.PSOOptimizer", return_value=optimizer),
        ):
            with pytest.raises(
                PdDisaggError,
                match="benchmark_run_count=2 requires use_request_rate_calibration=false",
            ):
                runner.run(PREFILL_PHASE, phase, tmp_path / PREFILL_PHASE)

        scheduler_cls.assert_not_called()

    def test_phase_search_budget_falls_back_to_top_level_settings(self, tmp_path):
        runner = OptixPhaseRunner(
            Settings(output=tmp_path, n_particles=5, iters=10),
            runtime_ctx=MagicMock(),
            deploy_env={},
            bak_path=None,
            load_breakpoint=False,
            top_k=3,
        )

        assert runner._resolve_search_budget(PdDisaggPhaseConfig()) == (5, 10)
