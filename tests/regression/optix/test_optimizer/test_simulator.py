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
import os
import shutil
import subprocess
import unittest
from unittest.mock import MagicMock, patch
from optix.optimizer.plugins.simulate import Simulator


class TestSimulate(unittest.TestCase):
    def test_set_config_dict(self):
        origin_config = {"a": {"b": {"c": 3}}}
        Simulator.set_config(origin_config, "a.b.c", 4)
        assert origin_config["a"]["b"]["c"] == 4

    def test_set_config_list(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.b.0.c", 4)
        assert origin_config["a"]["b"][0]["c"] == 4

    def test_set_config_new_key(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.b.0.d", 4)
        assert origin_config["a"]["b"][0]["d"] == 4

    def test_set_config_add_dict_list_dict(self):
        origin_config = {"a": {"b": {"c": 3}}}
        Simulator.set_config(origin_config, "a.d.0.c", 4)
        assert origin_config["a"]["d"][0]["c"] == 4

    def test_set_config_add_dict(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.b.1.c", 4)
        assert origin_config["a"]["b"][1]["c"] == 4

    def test_set_config_add_dict_list_dict_dict(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        Simulator.set_config(origin_config, "a.d.0.c.e", 4)
        assert origin_config["a"]["d"][0]["c"]["e"] == 4

    def test_is_int(self):
        # Test the is_int static method.
        self.assertTrue(Simulator.is_int(1))
        self.assertTrue(Simulator.is_int("1"))
        self.assertFalse(Simulator.is_int("a"))


class TestVllmSimulator(unittest.TestCase):
    """Test the VllmSimulator class."""

    def setUp(self):
        # Create a mock VllmConfig.
        self.mock_config = MagicMock()
        self.mock_config.process_name = "vllm"
        self.mock_config.command = MagicMock()
        self.mock_config.command.host = "localhost"
        self.mock_config.command.port = "8000"
        self.mock_config.command.model = "gpt2"
        self.mock_config.command.served_model_name = "gpt2"
        self.mock_config.command.others = ""

    @patch("optix.config.custom_command.shutil.which")
    def test_init(self, mock_which):
        """Test VllmSimulator initialization."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        self.assertEqual(simulator.config, self.mock_config)

    @patch("optix.config.custom_command.shutil.which")
    def test_base_url_property(self, mock_which):
        """Test the base_url property."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        expected_url = "http://localhost:8000/health"
        self.assertEqual(simulator.base_url, expected_url)

    @patch("optix.config.custom_command.shutil.which")
    @patch("optix.optimizer.plugins.simulate.subprocess.run")
    @patch("optix.optimizer.interfaces.custom_process.CustomProcess.stop")
    def test_stop(self, mock_super_stop, mock_run, mock_which):
        """Test the stop method."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        # Mock _is_vllm_running to return False, indicating no process needs to be stopped.
        with patch.object(simulator, "_is_vllm_running", return_value=False):
            simulator.stop()
        mock_super_stop.assert_called_once()

    @patch("optix.config.custom_command.shutil.which")
    @patch("optix.optimizer.plugins.simulate.subprocess.run")
    def test_stop_vllm_process_success(self, mock_run, mock_which):
        """Test _stop_vllm_process successfully stopping a process."""
        mock_which.return_value = "/usr/local/bin/vllm"
        mock_run.return_value = MagicMock(returncode=0)
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)

        # Simulate process checks: running first, then absent.
        with patch.object(simulator, "_is_vllm_running", side_effect=[True, False]):
            result = simulator._stop_vllm_process(max_attempts=1, timeout=1)
            self.assertTrue(result)

    @patch("optix.config.custom_command.shutil.which")
    def test_stop_vllm_process_with_psutil_targeting(self, mock_which):
        """Test _stop_vllm_process with active process via psutil PID-based targeting."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        # Setup a mock running process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # process is running
        mock_process.pid = 12345
        simulator.process = mock_process

        mock_parent = MagicMock()
        mock_child = MagicMock()
        mock_parent.children.return_value = [mock_child]

        with patch("psutil.Process", return_value=mock_parent):
            with patch("psutil.wait_procs") as mock_wait:
                mock_wait.return_value = (
                    [mock_parent, mock_child],
                    [],
                )  # all gone, none alive
                with patch.object(simulator, "_is_vllm_running", return_value=False):
                    result = simulator._stop_vllm_process(max_attempts=1, timeout=1)
                    self.assertTrue(result)
                    mock_child.terminate.assert_called_once()
                    mock_parent.terminate.assert_called_once()

    @patch("optix.config.custom_command.shutil.which")
    def test_stop_vllm_process_psutil_fails_fallback_pkill(self, mock_which):
        """Test _stop_vllm_process fallback to pkill when psutil targeting fails."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        simulator.process = mock_process

        import psutil

        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(12345)):
            with patch("shutil.which", return_value="/usr/bin/pkill"):
                with patch("subprocess.run", return_value=MagicMock(returncode=0)):
                    with patch.object(simulator, "_is_vllm_running", side_effect=[True, False]):
                        result = simulator._stop_vllm_process(max_attempts=1, timeout=0)
                        self.assertTrue(result)

    @patch("optix.config.custom_command.shutil.which")
    @patch("optix.optimizer.plugins.simulate.subprocess.run")
    def test_stop_vllm_process_already_stopped(self, mock_run, mock_which):
        """Test _stop_vllm_process when the process is already stopped."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)

        # Simulate that the process no longer exists.
        with patch.object(simulator, "_is_vllm_running", return_value=False):
            result = simulator._stop_vllm_process(max_attempts=1, timeout=1)
            self.assertTrue(result)

    @patch("optix.config.custom_command.shutil.which")
    def test_stop_vllm_process_no_pkill(self, mock_which):
        """Test _stop_vllm_process when pkill is unavailable."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        # Simulate a running process with pkill unavailable.
        with patch.object(simulator, "_is_vllm_running", return_value=True):
            # Also mock shutil.which in the simulate module.
            with patch(
                "optix.optimizer.plugins.simulate.shutil.which",
                return_value=None,
            ):
                result = simulator._stop_vllm_process()
                self.assertFalse(result)

    @patch("optix.config.custom_command.shutil.which")
    @patch("optix.optimizer.plugins.simulate.shutil.which")
    @patch("optix.optimizer.plugins.simulate.subprocess.run")
    def test_is_vllm_running_true(self, mock_run, mock_simulate_which, mock_config_which):
        """Test _is_vllm_running returning True."""
        mock_config_which.return_value = "/usr/local/bin/vllm"
        mock_simulate_which.return_value = "/usr/bin/pgrep"  # _is_vllm_running uses shutil.which from simulate.
        mock_run.return_value = MagicMock(stdout="5\n", returncode=0)
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        result = simulator._is_vllm_running()
        self.assertTrue(result)

    @patch("optix.config.custom_command.shutil.which")
    @patch("optix.optimizer.plugins.simulate.shutil.which")
    @patch("optix.optimizer.plugins.simulate.subprocess.run")
    def test_is_vllm_running_false(self, mock_run, mock_simulate_which, mock_config_which):
        """Test _is_vllm_running returning False."""
        mock_config_which.return_value = "/usr/local/bin/vllm"
        mock_simulate_which.return_value = "/usr/bin/pgrep"
        mock_run.return_value = MagicMock(stdout="0\n", returncode=0)
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        result = simulator._is_vllm_running()
        self.assertFalse(result)

    @patch("optix.config.custom_command.shutil.which")
    @patch("optix.optimizer.plugins.simulate.shutil.which")
    @patch("optix.optimizer.plugins.simulate.subprocess.run")
    def test_is_vllm_running_exception(self, mock_run, mock_simulate_which, mock_config_which):
        """Test _is_vllm_running exception handling."""
        mock_config_which.return_value = "/usr/local/bin/vllm"
        mock_simulate_which.return_value = "/usr/bin/pgrep"
        mock_run.side_effect = subprocess.SubprocessError("Command failed")
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        result = simulator._is_vllm_running()
        self.assertFalse(result)

    @patch("optix.config.custom_command.shutil.which")
    @patch("optix.optimizer.plugins.simulate.time.time")
    def test_wait_for_process_exit_success(self, mock_time, mock_which):
        """Test _wait_for_process_exit on successful exit."""
        mock_which.return_value = "/usr/local/bin/vllm"
        mock_time.side_effect = [0, 0.3, 0.6]  # Simulate elapsed time.
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)

        with patch.object(simulator, "_is_vllm_running", side_effect=[True, False]):
            result = simulator._wait_for_process_exit(timeout=1)
            self.assertTrue(result)

    @patch("optix.config.custom_command.shutil.which")
    @patch("optix.optimizer.plugins.simulate.time.time")
    def test_wait_for_process_exit_timeout(self, mock_time, mock_which):
        """Test _wait_for_process_exit timeout."""
        mock_which.return_value = "/usr/local/bin/vllm"
        mock_time.side_effect = [0, 1, 2, 3]  # Simulate timeout.
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)

        with patch.object(simulator, "_is_vllm_running", return_value=True):
            result = simulator._wait_for_process_exit(timeout=1)
            self.assertFalse(result)

    @patch("optix.config.custom_command.shutil.which")
    @patch("optix.optimizer.plugins.simulate.subprocess.run")
    def test_log_residual_processes(self, mock_run, mock_which):
        """Test the _log_residual_processes method."""
        mock_which.return_value = "/usr/local/bin/vllm"
        mock_run.return_value = MagicMock(stdout="1234 /usr/bin/vllm\n5678 /usr/bin/vllm\n")
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        simulator._log_residual_processes()
        mock_run.assert_called_once()

    @patch("optix.config.custom_command.shutil.which")
    @patch("optix.optimizer.plugins.simulate.subprocess.run")
    def test_log_residual_processes_exception(self, mock_run, mock_which):
        """Test _log_residual_processes exception handling."""
        mock_which.return_value = "/usr/local/bin/vllm"
        mock_run.side_effect = subprocess.SubprocessError("Command failed")
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        simulator._log_residual_processes()  # Should not raise an exception.

    @patch("optix.config.custom_command.shutil.which")
    def test_update_command(self, mock_which):
        """Test the update_command method."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        simulator.update_command()
        self.assertIsNotNone(simulator.command)


class TestSimulatorSetConfigEdgeCases(unittest.TestCase):
    def test_set_config_simple_dict_key(self):
        origin_config = {"key": "old"}
        Simulator.set_config(origin_config, "key", "new")
        assert origin_config["key"] == "new"

    def test_set_config_simple_list_index(self):
        origin_config = [10, 20, 30]
        Simulator.set_config(origin_config, "1", 99)
        assert origin_config[1] == 99

    def test_set_config_list_append(self):
        origin_config = [10]
        Simulator.set_config(origin_config, "1", 20)
        assert origin_config[1] == 20

    def test_set_config_recursion_limit(self):
        origin_config = {"a": {"b": {"c": {"d": {"e": 1}}}}}
        with self.assertRaises(RecursionError):
            Simulator.set_config(origin_config, ".".join(["x"] * 12), "val")

    def test_set_config_unsupported_type(self):
        """When origin_config is not dict/list and next_level is None, set_config silently returns"""
        result = Simulator.set_config("not_a_dict_or_list", "key", "val")
        # No exception raised, just returns None
        assert result is None

    def test_set_config_invalid_list_index(self):
        """Raises IndexError when list index exceeds current length (gap > 0)"""
        origin_config = [10]
        with self.assertRaises(IndexError):
            Simulator.set_config(origin_config, "5", "val")

    def test_set_config_new_dict_key_in_dict(self):
        origin_config = {}
        Simulator.set_config(origin_config, "new_key.sub_key", "val")
        assert origin_config["new_key"]["sub_key"] == "val"

    def test_set_config_list_append_nested(self):
        origin_config = {"a": []}
        Simulator.set_config(origin_config, "a.0.key", "val")
        assert origin_config["a"][0]["key"] == "val"

    def test_update_config_with_params(self):
        """Test Simulator.update_config with BackendConfig params"""
        import json
        import tempfile
        from pathlib import Path

        config_data = {"BackendConfig": {"ScheduleConfig": {"maxBatchSize": 100}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(config_data, tmp)
            tmp_path = tmp.name
        self.addCleanup(os.unlink, tmp_path)

        mock_config = MagicMock()
        mock_config.config_path = Path(tmp_path)
        mock_config.config_bak_path = Path(tmp_path + ".bak")
        self.addCleanup(os.unlink, tmp_path + ".bak")
        mock_config.command = MagicMock()
        mock_config.process_name = "mindie"

        with patch("optix.config.custom_command.os.path.isfile", return_value=True):
            with patch(
                "optix.config.custom_command.shutil.which",
                return_value="/usr/bin/mindie_llm_server",
            ):
                simulator = Simulator(config=mock_config)

        from optix.config.config import OptimizerConfigField

        params = (
            OptimizerConfigField(
                name="max_batch_size",
                config_position="BackendConfig.ScheduleConfig.maxBatchSize",
                value=200,
            ),
        )
        simulator.update_config(params)

        with open(mock_config.config_path, "r", encoding="utf-8") as f:
            updated = json.load(f)
        assert updated["BackendConfig"]["ScheduleConfig"]["maxBatchSize"] == 200


class TestSimulatorInterfaceHealth(unittest.TestCase):
    """Test SimulatorInterface.health method (lines 72-96)"""

    @patch("optix.config.custom_command.shutil.which")
    def test_health_returns_running_on_200(self, mock_which):
        mock_which.return_value = "/usr/local/bin/vllm"
        from optix.optimizer.plugins.simulate import VllmSimulator
        from optix.config.constant import Stage

        mock_config = MagicMock()
        mock_config.process_name = "vllm"
        mock_config.command = MagicMock()
        mock_config.command.host = "localhost"
        mock_config.command.port = "8000"
        mock_config.command.model = "gpt2"
        mock_config.command.served_model_name = "gpt2"
        mock_config.command.others = ""

        simulator = VllmSimulator(mock_config)
        simulator.process = MagicMock()
        simulator.process.poll.return_value = None
        simulator.print_log = False

        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch("requests.get", return_value=mock_response):
            result = simulator.health()
        assert result.stage == Stage.running

    @patch("optix.config.custom_command.shutil.which")
    def test_health_returns_error_on_non_200(self, mock_which):
        mock_which.return_value = "/usr/local/bin/vllm"
        from optix.optimizer.plugins.simulate import VllmSimulator
        from optix.config.constant import Stage

        mock_config = MagicMock()
        mock_config.process_name = "vllm"
        mock_config.command = MagicMock()
        mock_config.command.host = "localhost"
        mock_config.command.port = "8000"
        mock_config.command.model = "gpt2"
        mock_config.command.served_model_name = "gpt2"
        mock_config.command.others = ""

        simulator = VllmSimulator(mock_config)
        simulator.process = MagicMock()
        simulator.process.poll.return_value = None
        simulator.print_log = False

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        with patch("requests.get", return_value=mock_response):
            result = simulator.health()
        assert result.stage == Stage.error

    @patch("optix.config.custom_command.shutil.which")
    def test_health_request_exception_during_start(self, mock_which):
        mock_which.return_value = "/usr/local/bin/vllm"
        from optix.optimizer.plugins.simulate import VllmSimulator
        from optix.config.constant import Stage, ProcessState
        import requests

        mock_config = MagicMock()
        mock_config.process_name = "vllm"
        mock_config.command = MagicMock()
        mock_config.command.host = "localhost"
        mock_config.command.port = "8000"
        mock_config.command.model = "gpt2"
        mock_config.command.served_model_name = "gpt2"
        mock_config.command.others = ""

        simulator = VllmSimulator(mock_config)
        simulator.process = MagicMock()
        simulator.process.poll.return_value = None
        simulator.print_log = False
        simulator._process_stage = ProcessState(stage=Stage.start)

        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
            result = simulator.health()
        assert result.stage == Stage.start

    @patch("optix.config.custom_command.shutil.which")
    def test_health_request_exception_during_running(self, mock_which):
        mock_which.return_value = "/usr/local/bin/vllm"
        from optix.optimizer.plugins.simulate import VllmSimulator
        from optix.config.constant import Stage, ProcessState
        import requests

        mock_config = MagicMock()
        mock_config.process_name = "vllm"
        mock_config.command = MagicMock()
        mock_config.command.host = "localhost"
        mock_config.command.port = "8000"
        mock_config.command.model = "gpt2"
        mock_config.command.served_model_name = "gpt2"
        mock_config.command.others = ""

        simulator = VllmSimulator(mock_config)
        simulator.process = MagicMock()
        simulator.process.poll.return_value = None
        simulator.print_log = False
        simulator._process_stage = ProcessState(stage=Stage.running)

        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
            result = simulator.health()
        assert result.stage == Stage.error

    @patch("optix.config.custom_command.shutil.which")
    def test_enable_simulation_model(self, mock_which):
        mock_which.return_value = "/usr/local/bin/vllm"
        from optix.optimizer.plugins.simulate import VllmSimulator

        mock_config = MagicMock()
        mock_config.process_name = "vllm"
        mock_config.command = MagicMock()
        mock_config.command.host = "localhost"
        mock_config.command.port = "8000"
        mock_config.command.model = "gpt2"
        mock_config.command.served_model_name = "gpt2"
        mock_config.command.others = ""

        simulator = VllmSimulator(mock_config)
        with simulator.enable_simulation_model() as flag:
            assert flag is True


class TestMindieSimulatorInit(unittest.TestCase):
    """Test Simulator (Mindie) initialization"""

    def _create_config_file(self, tmp_dir, config_data=None):
        import json
        from pathlib import Path

        if config_data is None:
            config_data = {"BackendConfig": {"ScheduleConfig": {"maxBatchSize": 100}}}
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        return config_path

    @patch("optix.config.custom_command.os.path.isfile", return_value=True)
    @patch(
        "optix.config.custom_command.shutil.which",
        return_value="/usr/bin/mindie",
    )
    def test_init_success(self, mock_which, mock_isfile):
        import json
        import tempfile
        from pathlib import Path

        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        config_data = {"BackendConfig": {"ScheduleConfig": {"maxBatchSize": 100}}}
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        bak_path = Path(tmp_dir) / "config.json.bak"

        mock_config = MagicMock()
        mock_config.config_path = config_path
        mock_config.config_bak_path = bak_path
        mock_config.process_name = "mindie"
        mock_config.command = MagicMock()

        simulator = Simulator(config=mock_config)
        assert simulator.default_config == config_data
        assert bak_path.exists()

    @patch("optix.config.custom_command.os.path.isfile", return_value=True)
    @patch(
        "optix.config.custom_command.shutil.which",
        return_value="/usr/bin/mindie",
    )
    def test_init_config_not_found(self, mock_which, mock_isfile):
        mock_config = MagicMock()
        mock_config_path = MagicMock()
        mock_config_path.exists.return_value = False
        mock_config.config_path = mock_config_path
        mock_config.process_name = "mindie"

        with self.assertRaises(FileNotFoundError):
            Simulator(config=mock_config)


class TestMindieSimulatorBeforeRun(unittest.TestCase):
    """Test Simulator.before_run"""

    @patch("optix.optimizer.plugins.simulate.subprocess.run")
    @patch("optix.optimizer.plugins.simulate.shutil.which")
    @patch("optix.config.custom_command.os.path.isfile", return_value=True)
    @patch(
        "optix.config.custom_command.shutil.which",
        return_value="/usr/bin/mindie",
    )
    def test_before_run_calls_pkill_and_npu_smi(self, mock_cmd_which, mock_isfile, mock_sim_which, mock_run):
        import json
        import tempfile
        from pathlib import Path
        from optix.config.config import OptimizerConfigField

        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        config_data = {"BackendConfig": {"ScheduleConfig": {"maxBatchSize": 100}}}
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        bak_path = Path(tmp_dir) / "config.json.bak"

        mock_config = MagicMock()
        mock_config.config_path = config_path
        mock_config.config_bak_path = bak_path
        mock_config.process_name = "mindie"
        mock_config.command = MagicMock()

        mock_sim_which.side_effect = lambda cmd: f"/usr/bin/{cmd}" if cmd in ("pkill", "npu-smi") else None

        simulator = Simulator(config=mock_config)
        simulator.env = {}
        simulator.run_log_fp = MagicMock()
        simulator.work_path = tmp_dir

        params = (
            OptimizerConfigField(
                name="max_batch_size",
                config_position="BackendConfig.ScheduleConfig.maxBatchSize",
                value=200,
            ),
        )
        simulator.before_run(params)
        # pkill and npu-smi should be called
        assert mock_run.call_count == 2

    @patch("optix.optimizer.plugins.simulate.shutil.which", return_value=None)
    @patch("optix.config.custom_command.os.path.isfile", return_value=True)
    @patch(
        "optix.config.custom_command.shutil.which",
        return_value="/usr/bin/mindie",
    )
    def test_before_run_no_pkill(self, mock_cmd_which, mock_isfile, mock_sim_which):
        import json
        import tempfile
        from pathlib import Path
        from optix.config.config import OptimizerConfigField

        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        config_data = {"BackendConfig": {"ScheduleConfig": {"maxBatchSize": 100}}}
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        bak_path = Path(tmp_dir) / "config.json.bak"

        mock_config = MagicMock()
        mock_config.config_path = config_path
        mock_config.config_bak_path = bak_path
        mock_config.process_name = "mindie"
        mock_config.command = MagicMock()

        simulator = Simulator(config=mock_config)
        simulator.env = {}
        simulator.run_log_fp = MagicMock()
        simulator.work_path = tmp_dir

        params = (
            OptimizerConfigField(
                name="max_batch_size",
                config_position="BackendConfig.ScheduleConfig.maxBatchSize",
                value=200,
            ),
        )
        # Should not crash even when pkill not found
        simulator.before_run(params)


class TestMindieSimulatorHealth(unittest.TestCase):
    """Test Simulator.health (Mindie variant with daemon check)"""

    @patch("optix.config.custom_command.os.path.isfile", return_value=True)
    @patch(
        "optix.config.custom_command.shutil.which",
        return_value="/usr/bin/mindie",
    )
    def test_health_running_via_daemon(self, mock_cmd_which, mock_isfile):
        import json
        import tempfile
        from pathlib import Path
        from optix.config.constant import Stage, ProcessState

        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        config_data = {"BackendConfig": {"ScheduleConfig": {"maxBatchSize": 100}}}
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        bak_path = Path(tmp_dir) / "config.json.bak"

        mock_config = MagicMock()
        mock_config.config_path = config_path
        mock_config.config_bak_path = bak_path
        mock_config.process_name = "mindie"
        mock_config.command = MagicMock()

        simulator = Simulator(config=mock_config)
        simulator.process = MagicMock()
        simulator.process.poll.return_value = None

        # First super().health() returns non-running
        non_running_state = ProcessState(stage=Stage.start)
        # Simulating the proxy_status returns running
        running_state = ProcessState(stage=Stage.running)
        simulator.run_log_offset = 0

        with patch.object(type(simulator).__mro__[1], "health", return_value=non_running_state):
            with patch.object(type(simulator).__mro__[2], "health", return_value=running_state):
                with patch.object(simulator, "get_log", return_value="Daemon start success!"):
                    result = simulator.health()
        assert result.stage == Stage.running

    @patch("optix.config.custom_command.os.path.isfile", return_value=True)
    @patch(
        "optix.config.custom_command.shutil.which",
        return_value="/usr/bin/mindie",
    )
    def test_health_returns_process_result_when_running(self, mock_cmd_which, mock_isfile):
        import json
        import tempfile
        from pathlib import Path
        from optix.config.constant import Stage, ProcessState

        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        config_data = {"BackendConfig": {"ScheduleConfig": {"maxBatchSize": 100}}}
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        bak_path = Path(tmp_dir) / "config.json.bak"

        mock_config = MagicMock()
        mock_config.config_path = config_path
        mock_config.config_bak_path = bak_path
        mock_config.process_name = "mindie"
        mock_config.command = MagicMock()

        simulator = Simulator(config=mock_config)
        simulator.process = MagicMock()
        simulator.process.poll.return_value = None

        running_state = ProcessState(stage=Stage.running)
        with patch.object(type(simulator).__mro__[1], "health", return_value=running_state):
            result = simulator.health()
        assert result.stage == Stage.running


class TestMindieSimulatorStop(unittest.TestCase):
    """Test Simulator.stop restores config"""

    @patch("optix.config.custom_command.os.path.isfile", return_value=True)
    @patch(
        "optix.config.custom_command.shutil.which",
        return_value="/usr/bin/mindie",
    )
    def test_stop_restores_default_config(self, mock_cmd_which, mock_isfile):
        import json
        import tempfile
        from pathlib import Path

        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        config_data = {"BackendConfig": {"ScheduleConfig": {"maxBatchSize": 100}}}
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        bak_path = Path(tmp_dir) / "config.json.bak"

        mock_config = MagicMock()
        mock_config.config_path = config_path
        mock_config.config_bak_path = bak_path
        mock_config.process_name = "mindie"
        mock_config.command = MagicMock()

        simulator = Simulator(config=mock_config)
        # Modify config
        config_path.write_text(json.dumps({"modified": True}), encoding="utf-8")

        simulator.process = None
        simulator.run_log_fp = None
        simulator.run_log = None
        with patch("optix.optimizer.plugins.simulate.remove_file"):
            simulator.stop(del_log=True)

        restored = json.loads(config_path.read_text())
        assert restored == config_data


class TestMindieSimulatorUpdateConfig(unittest.TestCase):
    """Test Simulator.update_config"""

    @patch("optix.config.custom_command.os.path.isfile", return_value=True)
    @patch(
        "optix.config.custom_command.shutil.which",
        return_value="/usr/bin/mindie",
    )
    def test_update_config_no_params(self, mock_cmd_which, mock_isfile):
        import json
        import tempfile
        from pathlib import Path

        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        config_data = {"BackendConfig": {"ScheduleConfig": {"maxBatchSize": 100}}}
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        bak_path = Path(tmp_dir) / "config.json.bak"

        mock_config = MagicMock()
        mock_config.config_path = config_path
        mock_config.config_bak_path = bak_path
        mock_config.process_name = "mindie"
        mock_config.command = MagicMock()

        simulator = Simulator(config=mock_config)
        simulator.update_config(None)
        # Config should remain unchanged
        current = json.loads(config_path.read_text())
        assert current == config_data

    @patch("optix.config.custom_command.os.path.isfile", return_value=True)
    @patch(
        "optix.config.custom_command.shutil.which",
        return_value="/usr/bin/mindie",
    )
    def test_update_config_skips_non_backend_params(self, mock_cmd_which, mock_isfile):
        import json
        import tempfile
        from pathlib import Path
        from optix.config.config import OptimizerConfigField

        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        config_data = {"BackendConfig": {"ScheduleConfig": {"maxBatchSize": 100}}}
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        bak_path = Path(tmp_dir) / "config.json.bak"

        mock_config = MagicMock()
        mock_config.config_path = config_path
        mock_config.config_bak_path = bak_path
        mock_config.process_name = "mindie"
        mock_config.command = MagicMock()

        simulator = Simulator(config=mock_config)
        params = (
            OptimizerConfigField(
                name="CONCURRENCY",
                config_position="env",
                value=32,
                min=1,
                max=100,
                dtype="int",
            ),
        )
        simulator.update_config(params)
        current = json.loads(config_path.read_text())
        # No BackendConfig param, so config should remain same
        assert current == config_data
