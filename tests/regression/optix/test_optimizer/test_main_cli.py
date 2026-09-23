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

"""Regression tests for optimizer.main() CLI branches."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from optix.config.config import OptimizerConfigField, PdDisaggConfig
from tests.helpers.cli_runner import run_cli_main


def _settings_mock() -> MagicMock:
    settings = MagicMock()
    settings.n_particles = 3
    settings.iters = 2
    settings.ttft_penalty = 0
    settings.tpot_penalty = 0
    settings.success_rate_penalty = 0
    settings.ttft_slo = 1.0
    settings.tpot_slo = 0.1
    settings.success_rate_slo = 1.0
    settings.generate_speed_target = 100
    settings.max_fine_tune = 1
    settings.skip_pso = False
    settings.manage_simulator_lifecycle = True
    settings.output = MagicMock()
    settings.step_size = 0.1
    settings.slo_coefficient = 1.0
    settings.ftol = 1e-3
    settings.ftol_iter = 2
    settings.data_storage = MagicMock()
    settings.deploy.path_prefix = None
    settings.pd_disagg = PdDisaggConfig()
    return settings


def _registry_mocks():
    mock_simu = MagicMock()
    mock_simu.data_field = [OptimizerConfigField(name="max_batch_size", min=10, max=100, dtype="int")]
    mock_bench = MagicMock()
    mock_bench.data_field = [OptimizerConfigField(name="CONCURRENCY", min=1, max=64, dtype="int")]
    return mock_simu, mock_bench


class TestOptimizerMainCli(unittest.TestCase):
    @patch("optix.optimizer.pd_disagg.PdDisaggOrchestrator")
    @patch("optix.optimizer.pd_disagg.OptixPhaseRunner")
    @patch("optix.deploy_env.emit_runtime_hints")
    @patch("optix.deploy_env.resolve_deploy_context", return_value=(MagicMock(), {}))
    @patch("optix.config.config.get_settings")
    @patch("optix.optimizer.register.register_ori_functions")
    @patch("optix.plugins.load_general_plugins")
    @patch("optix.optimizer.optimizer.is_root", return_value=False)
    def test_main_pd_disagg_mode_dispatches_to_orchestrator(
        self,
        mock_is_root,
        mock_load_plugins,
        mock_register,
        mock_get_settings,
        mock_resolve_deploy,
        mock_emit_hints,
        mock_phase_runner,
        mock_orchestrator,
    ):
        from optix.optimizer.optimizer import main as optix_main

        settings = _settings_mock()
        settings.pd_disagg = PdDisaggConfig()
        mock_get_settings.return_value = settings
        mock_simu, mock_bench = _registry_mocks()
        argv = ["optix", "--mode", "pd_disagg"]
        with (
            patch.dict("optix.optimizer.register.simulates", {"vllm": lambda **kw: mock_simu}),
            patch.dict("optix.optimizer.register.benchmarks", {"ais_bench": lambda **kw: mock_bench}),
            patch.object(sys, "argv", argv),
        ):
            optix_main()

        mock_phase_runner.assert_called_once()
        mock_orchestrator.return_value.run.assert_called_once_with()

    @patch("optix.optimizer.optimizer.PSOOptimizer")
    @patch("optix.optimizer.scheduler.Scheduler")
    @patch("optix.optimizer.store.DataStorage")
    @patch("optix.optimizer.experience_fine_tunning.FineTune")
    @patch("optix.config.config.get_settings")
    @patch("optix.optimizer.register.register_ori_functions")
    @patch("optix.plugins.load_general_plugins")
    @patch("optix.optimizer.optimizer.is_root", return_value=False)
    def test_main_missing_config_returns_early(
        self,
        mock_is_root,
        mock_load_plugins,
        mock_register,
        mock_get_settings,
        mock_fine_tune,
        mock_ds,
        mock_scheduler,
        mock_pso,
    ):
        from optix.optimizer.optimizer import main as optix_main

        mock_get_settings.return_value = _settings_mock()
        mock_simu, mock_bench = _registry_mocks()
        missing = "/tmp/optix-does-not-exist-config.toml"
        argv = ["optix", "-c", missing, "-e", "vllm", "-b", "ais_bench"]
        with (
            patch.dict("optix.optimizer.register.simulates", {"vllm": lambda **kw: mock_simu}),
            patch.dict("optix.optimizer.register.benchmarks", {"ais_bench": lambda **kw: mock_bench}),
            patch.object(sys, "argv", argv),
            self.assertRaises(SystemExit) as ctx,
        ):
            optix_main()
        self.assertEqual(ctx.exception.code, 1)
        mock_pso.assert_not_called()

    @patch("optix.optimizer.optimizer.PSOOptimizer")
    @patch("optix.optimizer.scheduler.Scheduler")
    @patch("optix.optimizer.store.DataStorage")
    @patch("optix.optimizer.experience_fine_tunning.FineTune")
    @patch("optix.config.config.get_settings")
    @patch("optix.optimizer.register.register_ori_functions")
    @patch("optix.plugins.load_general_plugins")
    @patch("optix.optimizer.optimizer.is_root", return_value=False)
    def test_main_invalid_toml_raises(
        self,
        mock_is_root,
        mock_load_plugins,
        mock_register,
        mock_get_settings,
        mock_fine_tune,
        mock_ds,
        mock_scheduler,
        mock_pso,
    ):
        from optix.optimizer.optimizer import main as optix_main

        mock_get_settings.return_value = _settings_mock()
        mock_simu, mock_bench = _registry_mocks()
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as handle:
            handle.write("not = [valid\n")
            bad_config = handle.name
        argv = ["optix", "-c", bad_config, "-e", "vllm", "-b", "ais_bench"]
        with (
            patch.dict("optix.optimizer.register.simulates", {"vllm": lambda **kw: mock_simu}),
            patch.dict("optix.optimizer.register.benchmarks", {"ais_bench": lambda **kw: mock_bench}),
            patch.object(sys, "argv", argv),
            self.assertRaises(SystemExit) as ctx,
        ):
            optix_main()
        self.assertEqual(ctx.exception.code, 1)

    @patch("optix.optimizer.register.shutil.which", return_value="/usr/bin/ais_bench")
    @patch("optix.optimizer.optimizer.PSOOptimizer")
    @patch("optix.optimizer.scheduler.Scheduler")
    @patch("optix.optimizer.store.DataStorage")
    @patch("optix.optimizer.experience_fine_tunning.FineTune")
    @patch("optix.config.config.get_settings")
    @patch("optix.optimizer.register.register_ori_functions")
    @patch("optix.plugins.load_general_plugins")
    @patch("optix.optimizer.optimizer.is_root", return_value=False)
    def test_main_backup_creates_directory(
        self,
        mock_is_root,
        mock_load_plugins,
        mock_register,
        mock_get_settings,
        mock_fine_tune,
        mock_ds,
        mock_scheduler,
        mock_pso,
        mock_which,
    ):
        from optix.optimizer.optimizer import main as optix_main

        tmp_path = Path(tempfile.mkdtemp())
        settings = _settings_mock()
        settings.output = tmp_path
        mock_get_settings.return_value = settings
        mock_simu, mock_bench = _registry_mocks()
        argv = ["optix", "--backup", "-e", "vllm", "-b", "ais_bench"]
        with (
            patch.dict("optix.optimizer.register.simulates", {"vllm": lambda **kw: mock_simu}),
            patch.dict("optix.optimizer.register.benchmarks", {"ais_bench": lambda **kw: mock_bench}),
            patch.object(sys, "argv", argv),
        ):
            optix_main()
        self.assertTrue((tmp_path / "back_up").is_dir())
        mock_pso.assert_called_once()
        self.assertFalse(mock_pso.call_args.kwargs["load_breakpoint"])

    @patch("optix.optimizer.register.shutil.which", return_value="/usr/bin/ais_bench")
    @patch("optix.optimizer.optimizer.PSOOptimizer")
    @patch("optix.optimizer.scheduler.Scheduler")
    @patch("optix.optimizer.store.DataStorage")
    @patch("optix.optimizer.experience_fine_tunning.FineTune")
    @patch("optix.config.config.get_settings")
    @patch("optix.optimizer.register.register_ori_functions")
    @patch("optix.plugins.load_general_plugins")
    @patch("optix.optimizer.optimizer.is_root", return_value=False)
    def test_main_standard_mode_load_breakpoint_flag_forwarded(
        self,
        mock_is_root,
        mock_load_plugins,
        mock_register,
        mock_get_settings,
        mock_fine_tune,
        mock_ds,
        mock_scheduler,
        mock_pso,
        mock_which,
    ):
        from optix.optimizer.optimizer import main as optix_main

        mock_get_settings.return_value = _settings_mock()
        mock_simu, mock_bench = _registry_mocks()
        argv = ["optix", "--mode", "standard", "-lb", "-e", "vllm", "-b", "ais_bench"]
        with (
            patch.dict("optix.optimizer.register.simulates", {"vllm": lambda **kw: mock_simu}),
            patch.dict("optix.optimizer.register.benchmarks", {"ais_bench": lambda **kw: mock_bench}),
            patch.object(sys, "argv", argv),
        ):
            optix_main()
        self.assertTrue(mock_pso.call_args.kwargs["load_breakpoint"])

    @patch("optix.optimizer.register.shutil.which", return_value="/usr/bin/ais_bench")
    @patch("optix.optimizer.optimizer.PSOOptimizer")
    @patch("optix.optimizer.scheduler.Scheduler")
    @patch("optix.optimizer.store.DataStorage")
    @patch("optix.optimizer.experience_fine_tunning.FineTune")
    @patch("optix.config.config.get_settings")
    @patch("optix.optimizer.register.register_ori_functions")
    @patch("optix.plugins.load_general_plugins")
    @patch("optix.optimizer.optimizer.is_root", return_value=False)
    def test_main_run_plugin_exception_propagates(
        self,
        mock_is_root,
        mock_load_plugins,
        mock_register,
        mock_get_settings,
        mock_fine_tune,
        mock_ds,
        mock_scheduler,
        mock_pso,
        mock_which,
    ):
        from optix.optimizer.optimizer import main as optix_main

        mock_get_settings.return_value = _settings_mock()
        mock_simu, mock_bench = _registry_mocks()
        mock_pso.return_value.run_plugin.side_effect = RuntimeError("optimizer failed")
        argv = ["optix", "-e", "vllm", "-b", "ais_bench"]
        with (
            patch.dict("optix.optimizer.register.simulates", {"vllm": lambda **kw: mock_simu}),
            patch.dict("optix.optimizer.register.benchmarks", {"ais_bench": lambda **kw: mock_bench}),
        ):
            result = run_cli_main(optix_main, argv[1:], prog="optix")
        self.assertEqual(result.returncode, 1)
        mock_pso.return_value.run_plugin.assert_called_once()


class TestOptimizerCliBindings(unittest.TestCase):
    def test_module_still_exports_is_root_and_json_helper(self):
        from optix.optimizer.optimizer import get_required_field_from_json, is_root

        self.assertTrue(callable(is_root))
        self.assertTrue(callable(get_required_field_from_json))

    def test_pd_disagg_mode_rejects_skip_pso(self):
        from optix.optimizer.errors import PdDisaggError
        from optix.optimizer.optimizer import _validate_optimization_mode_settings

        with self.assertRaisesRegex(PdDisaggError, "pd_disagg mode requires skip_pso=false"):
            _validate_optimization_mode_settings(
                "pd_disagg",
                skip_pso=True,
                manage_simulator_lifecycle=True,
            )

    def test_pd_disagg_mode_accepts_pso_search(self):
        from optix.optimizer.optimizer import _validate_optimization_mode_settings

        _validate_optimization_mode_settings(
            "pd_disagg",
            skip_pso=False,
            manage_simulator_lifecycle=True,
        )

    def test_pd_disagg_mode_rejects_disabled_simulator_lifecycle(self):
        from optix.optimizer.errors import PdDisaggError
        from optix.optimizer.optimizer import _validate_optimization_mode_settings

        with self.assertRaisesRegex(PdDisaggError, "pd_disagg mode requires manage_simulator_lifecycle=true"):
            _validate_optimization_mode_settings(
                "pd_disagg",
                skip_pso=True,
                manage_simulator_lifecycle=False,
            )

    def test_standard_mode_accepts_skip_pso(self):
        from optix.optimizer.optimizer import _validate_optimization_mode_settings

        _validate_optimization_mode_settings(
            "standard",
            skip_pso=True,
            manage_simulator_lifecycle=False,
        )

    @patch("optix.optimizer.optimizer.PSOOptimizer")
    @patch("optix.optimizer.scheduler.Scheduler")
    @patch("optix.optimizer.store.DataStorage")
    @patch("optix.optimizer.experience_fine_tunning.FineTune")
    @patch("optix.config.config.get_settings")
    @patch("optix.optimizer.register.register_ori_functions")
    @patch("optix.plugins.load_general_plugins")
    @patch("optix.optimizer.optimizer.is_root", return_value=False)
    @patch("cli.logo.print_logo", lambda: None)
    @patch("optix.optimizer.optimizer.set_log_level")
    def test_env_log_level_used_when_cli_omits_log_level(
        self,
        mock_set_log_level,
        mock_is_root,
        mock_load_plugins,
        mock_register,
        mock_get_settings,
        mock_fine_tune,
        mock_ds,
        mock_scheduler,
        mock_pso,
    ):
        from optix.optimizer.optimizer import main as optix_main

        mock_get_settings.return_value = _settings_mock()
        mock_simu, mock_bench = _registry_mocks()
        missing = "/tmp/optix-does-not-exist-config.toml"
        argv = ["optix", "-c", missing, "-e", "vllm", "-b", "ais_bench"]
        with (
            patch.dict("optix.optimizer.register.simulates", {"vllm": lambda **kw: mock_simu}),
            patch.dict("optix.optimizer.register.benchmarks", {"ais_bench": lambda **kw: mock_bench}),
            patch.object(sys, "argv", argv),
            patch.dict(os.environ, {"OPTIX_LOG_LEVEL": "DEBUG"}, clear=False),
            self.assertRaises(SystemExit),
        ):
            optix_main()
        mock_set_log_level.assert_called_once_with("DEBUG")

    @patch("optix.optimizer.optimizer.PSOOptimizer")
    @patch("optix.optimizer.scheduler.Scheduler")
    @patch("optix.optimizer.store.DataStorage")
    @patch("optix.optimizer.experience_fine_tunning.FineTune")
    @patch("optix.config.config.get_settings")
    @patch("optix.optimizer.register.register_ori_functions")
    @patch("optix.plugins.load_general_plugins")
    @patch("optix.optimizer.optimizer.is_root", return_value=False)
    @patch("cli.logo.print_logo", lambda: None)
    @patch("optix.optimizer.optimizer.set_log_level")
    def test_cli_log_level_overrides_env(
        self,
        mock_set_log_level,
        mock_is_root,
        mock_load_plugins,
        mock_register,
        mock_get_settings,
        mock_fine_tune,
        mock_ds,
        mock_scheduler,
        mock_pso,
    ):
        from optix.optimizer.optimizer import main as optix_main

        mock_get_settings.return_value = _settings_mock()
        mock_simu, mock_bench = _registry_mocks()
        missing = "/tmp/optix-does-not-exist-config.toml"
        argv = ["optix", "--log-level", "warning", "-c", missing, "-e", "vllm", "-b", "ais_bench"]
        with (
            patch.dict("optix.optimizer.register.simulates", {"vllm": lambda **kw: mock_simu}),
            patch.dict("optix.optimizer.register.benchmarks", {"ais_bench": lambda **kw: mock_bench}),
            patch.object(sys, "argv", argv),
            patch.dict(os.environ, {"OPTIX_LOG_LEVEL": "DEBUG"}, clear=False),
            self.assertRaises(SystemExit),
        ):
            optix_main()
        mock_set_log_level.assert_called_once_with("warning")
