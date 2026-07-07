"""Smoke guard for optix optimizer CLI path."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

from optix.config.config import OptimizerConfigField
from tests.helpers.cli_runner import run_cli_main


class TestOptixOptimizerSmoke(unittest.TestCase):
    @patch("optix.optimizer.register.shutil.which", return_value="/usr/bin/ais_bench")
    @patch("optix.optimizer.optimizer.PSOOptimizer")
    @patch("optix.optimizer.scheduler.Scheduler")
    @patch("optix.optimizer.store.DataStorage")
    @patch("optix.optimizer.experience_fine_tunning.FineTune")
    @patch("optix.config.config.get_settings")
    @patch("optix.optimizer.register.register_ori_functions")
    @patch("optix.plugins.load_general_plugins")
    @patch("optix.optimizer.optimizer.is_root", return_value=False)
    @patch("cli.logo.print_logo")
    def test_optix_optimizer_main_smoke(
        self,
        mock_logo,
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

        settings = MagicMock()
        settings.n_particles = 2
        settings.iters = 1
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
        settings.ftol_iter = 1
        settings.data_storage = MagicMock()
        mock_get_settings.return_value = settings

        mock_simu = MagicMock()
        mock_simu.data_field = [OptimizerConfigField(name="max_batch_size", min=10, max=100, dtype="int")]
        mock_bench = MagicMock()
        mock_bench.data_field = [OptimizerConfigField(name="CONCURRENCY", min=1, max=64, dtype="int")]

        argv = ["optix", "-e", "vllm", "-b", "ais_bench"]
        with (
            patch.dict("optix.optimizer.register.simulates", {"vllm": lambda **kw: mock_simu}),
            patch.dict("optix.optimizer.register.benchmarks", {"ais_bench": lambda **kw: mock_bench}),
            patch.object(sys, "argv", argv),
        ):
            result = run_cli_main(optix_main, argv[1:], prog="optix")

        self.assertEqual(result.returncode, 0)
        mock_pso.assert_called_once()
        mock_pso.return_value.run_plugin.assert_called_once()
