"""Regression tests for optimizer.main() CLI branches."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from optix.config.config import OptimizerConfigField
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
    settings.output = MagicMock()
    settings.step_size = 0.1
    settings.slo_coefficient = 1.0
    settings.ftol = 1e-3
    settings.ftol_iter = 2
    settings.data_storage = MagicMock()
    return settings


def _registry_mocks():
    mock_simu = MagicMock()
    mock_simu.data_field = [OptimizerConfigField(name="max_batch_size", min=10, max=100, dtype="int")]
    mock_bench = MagicMock()
    mock_bench.data_field = [OptimizerConfigField(name="CONCURRENCY", min=1, max=64, dtype="int")]
    return mock_simu, mock_bench


class TestOptimizerMainCli(unittest.TestCase):
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
        self.assertTrue((tmp_path / "trial_logs").is_dir())
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
    def test_main_load_breakpoint_flag_forwarded(
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
        argv = ["optix", "-lb", "-e", "vllm", "-b", "ais_bench"]
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
