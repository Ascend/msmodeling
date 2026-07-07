"""Plugin ``.run()`` integration tests via MRO to ``CustomProcess::run``."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from optix.optimizer.plugins.benchmark import AisBench, VllmBenchMark
from optix.optimizer.plugins.simulate import Simulator, VllmSimulator
from tests.helpers.fake_subprocess import FakePopen
from tests.regression.optix.test_optimizer.plugin_run_support import (
    make_mindie_simulator_config,
    prepare_plugin_for_run,
)

_POPEN_TARGET = "optix.optimizer.interfaces.custom_process.subprocess.Popen"
_KILL_RESIDUAL_TARGET = "optix.optimizer.interfaces.custom_process.CustomProcess.kill_residual_process"
_AIS_BEFORE_RUN = "optix.optimizer.plugins.benchmark.AisBench.before_run"
_VLLM_BENCH_BEFORE_RUN = "optix.optimizer.plugins.benchmark.VllmBenchMark.before_run"
_MINDIE_BEFORE_RUN = "optix.optimizer.plugins.simulate.Simulator.before_run"
_VLLM_SIM_BEFORE_RUN = "optix.optimizer.interfaces.custom_process.CustomProcess.before_run"

_AIS_BENCH_CMD = [
    "/usr/bin/ais_bench",
    "--models",
    "test_model",
    "--datasets",
    "test_dataset",
    "--mode",
    "perf",
    "--num-prompts",
    "8",
    "--work-dir",
    "/tmp/work",
    "--debug",
]


class TestPluginRun(unittest.TestCase):
    @patch(_AIS_BEFORE_RUN)
    @patch(_POPEN_TARGET)
    def test_ais_bench_run_invokes_popen(self, mock_popen, mock_before_run):
        mock_popen.return_value = FakePopen()
        bench = AisBench.__new__(AisBench)
        prepare_plugin_for_run(
            bench,
            command=_AIS_BENCH_CMD,
            work_path="/tmp/work",
            run_log="/tmp/ais_run.log",
        )
        bench.run()
        mock_before_run.assert_called_once()
        mock_popen.assert_called_once()
        assert mock_popen.call_args[0][0] == _AIS_BENCH_CMD

    @patch(_KILL_RESIDUAL_TARGET)
    @patch(_AIS_BEFORE_RUN)
    @patch(_POPEN_TARGET)
    def test_ais_bench_run_kill_residual_failure_continues(self, mock_popen, mock_before_run, mock_kill_residual):
        mock_popen.return_value = FakePopen()
        mock_kill_residual.side_effect = Exception("kill_residual_process failed")
        bench = AisBench.__new__(AisBench)
        prepare_plugin_for_run(
            bench,
            command=_AIS_BENCH_CMD,
            work_path="/tmp/work",
            run_log="/tmp/ais_run.log",
            process_name="ais_bench",
        )
        bench.run()
        mock_kill_residual.assert_called_once_with("ais_bench")
        mock_before_run.assert_called_once()
        mock_popen.assert_called_once()

    @patch(_VLLM_BENCH_BEFORE_RUN)
    @patch(_POPEN_TARGET)
    @patch("optix.config.custom_command.shutil.which", return_value="/usr/bin/vllm")
    def test_vllm_benchmark_run_invokes_popen(self, mock_which, mock_popen, mock_before_run):
        mock_popen.return_value = FakePopen()
        benchmark = VllmBenchMark.__new__(VllmBenchMark)
        cmd = [
            "/usr/bin/vllm",
            "bench",
            "serve",
            "--host",
            "localhost",
            "--port",
            "8000",
            "--model",
            "gpt2",
            "--served-model-name",
            "gpt2",
            "--dataset-name",
            "sharegpt",
            "--num-prompts",
            "4",
            "--max-concurrency",
            "$CONCURRENCY",
            "--request-rate",
            "$REQUESTRATE",
            "--result-dir",
            "/tmp/results",
        ]
        prepare_plugin_for_run(
            benchmark,
            command=cmd,
            work_path="/tmp/work",
            run_log="/tmp/vllm_bench.log",
        )
        benchmark.run()
        mock_before_run.assert_called_once()
        mock_popen.assert_called_once()
        assert mock_popen.call_args[0][0] == cmd

    @patch(_VLLM_BENCH_BEFORE_RUN)
    @patch(_POPEN_TARGET)
    @patch("optix.config.custom_command.shutil.which", return_value="/usr/bin/vllm")
    def test_vllm_benchmark_run_invalid_env_logs_error(self, mock_which, mock_popen, mock_before_run):
        mock_popen.return_value = FakePopen()
        benchmark = VllmBenchMark.__new__(VllmBenchMark)
        cmd = ["/usr/bin/vllm", "bench", "serve"]
        prepare_plugin_for_run(
            benchmark,
            command=cmd,
            work_path="/tmp/work",
            run_log="/tmp/vllm_bench.log",
            env={"INVALID_KEY": 123},
        )
        benchmark.run()
        mock_popen.assert_called_once()

    @patch(_KILL_RESIDUAL_TARGET)
    @patch(_MINDIE_BEFORE_RUN)
    @patch(_POPEN_TARGET)
    @patch(
        "optix.config.custom_command.shutil.which",
        return_value="/usr/bin/mindie_llm_server",
    )
    def test_mindie_simulator_run_invokes_popen(
        self,
        mock_cmd_which,
        mock_popen,
        mock_before_run,
        mock_kill_residual,
    ):
        mock_popen.return_value = FakePopen()
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        mock_config = make_mindie_simulator_config(tmp_dir)
        simulator = Simulator(config=mock_config)
        assert simulator.command is not None
        prepare_plugin_for_run(
            simulator,
            command=simulator.command,
            work_path=tmp_dir,
            run_log="/tmp/mindie_run.log",
            process_name="mindie",
        )
        simulator.run()
        mock_kill_residual.assert_called_once_with("mindie")
        mock_before_run.assert_called_once()
        mock_popen.assert_called_once()
        assert mock_popen.call_args[0][0] == simulator.command

    @patch(_VLLM_SIM_BEFORE_RUN)
    @patch(_POPEN_TARGET)
    @patch("optix.config.custom_command.shutil.which", return_value="/usr/local/bin/vllm")
    def test_vllm_simulator_run_invokes_popen(self, mock_which, mock_popen, mock_before_run):
        mock_popen.return_value = FakePopen()
        mock_config = MagicMock()
        mock_config.process_name = "vllm"
        mock_config.command = MagicMock()
        mock_config.command.host = "localhost"
        mock_config.command.port = "8000"
        mock_config.command.model = "gpt2"
        mock_config.command.served_model_name = "gpt2"
        mock_config.command.others = ""

        simulator = VllmSimulator(mock_config)
        assert simulator.command is not None
        prepare_plugin_for_run(
            simulator,
            command=simulator.command,
            work_path="/tmp/work",
            run_log="/tmp/vllm_run.log",
            process_name="vllm",
        )
        simulator.run()
        mock_before_run.assert_called_once()
        mock_popen.assert_called_once()
        assert mock_popen.call_args[0][0] == simulator.command
