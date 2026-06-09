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
import subprocess
import unittest
from unittest.mock import MagicMock, patch
from experimental.optix.optimizer.plugins.simulate import Simulator


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

    @patch("experimental.optix.config.custom_command.shutil.which")
    def test_init(self, mock_which):
        """Test VllmSimulator initialization."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        self.assertEqual(simulator.config, self.mock_config)

    @patch("experimental.optix.config.custom_command.shutil.which")
    def test_base_url_property(self, mock_which):
        """Test the base_url property."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        expected_url = "http://localhost:8000/health"
        self.assertEqual(simulator.base_url, expected_url)

    @patch("experimental.optix.config.custom_command.shutil.which")
    @patch("experimental.optix.optimizer.plugins.simulate.subprocess.run")
    @patch("experimental.optix.optimizer.interfaces.custom_process.CustomProcess.stop")
    def test_stop(self, mock_super_stop, mock_run, mock_which):
        """Test the stop method."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        # Mock _is_vllm_running to return False, indicating no process needs to be stopped.
        with patch.object(simulator, "_is_vllm_running", return_value=False):
            simulator.stop()
        mock_super_stop.assert_called_once()

    @patch("experimental.optix.config.custom_command.shutil.which")
    @patch("experimental.optix.optimizer.plugins.simulate.subprocess.run")
    def test_stop_vllm_process_success(self, mock_run, mock_which):
        """Test _stop_vllm_process successfully stopping a process."""
        mock_which.return_value = "/usr/local/bin/vllm"
        mock_run.return_value = MagicMock(returncode=0)
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)

        # Simulate process checks: running first, then absent.
        with patch.object(simulator, "_is_vllm_running", side_effect=[True, False]):
            result = simulator._stop_vllm_process(max_attempts=1, timeout=1)
            self.assertTrue(result)

    @patch("experimental.optix.config.custom_command.shutil.which")
    @patch("experimental.optix.optimizer.plugins.simulate.subprocess.run")
    def test_stop_vllm_process_already_stopped(self, mock_run, mock_which):
        """Test _stop_vllm_process when the process is already stopped."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)

        # Simulate that the process no longer exists.
        with patch.object(simulator, "_is_vllm_running", return_value=False):
            result = simulator._stop_vllm_process(max_attempts=1, timeout=1)
            self.assertTrue(result)

    @patch("experimental.optix.config.custom_command.shutil.which")
    def test_stop_vllm_process_no_pkill(self, mock_which):
        """Test _stop_vllm_process when pkill is unavailable."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        # Simulate a running process with pkill unavailable.
        with patch.object(simulator, "_is_vllm_running", return_value=True):
            # Also mock shutil.which in the simulate module.
            with patch(
                "experimental.optix.optimizer.plugins.simulate.shutil.which",
                return_value=None,
            ):
                result = simulator._stop_vllm_process()
                self.assertFalse(result)

    @patch("experimental.optix.config.custom_command.shutil.which")
    @patch("experimental.optix.optimizer.plugins.simulate.shutil.which")
    @patch("experimental.optix.optimizer.plugins.simulate.subprocess.run")
    def test_is_vllm_running_true(self, mock_run, mock_simulate_which, mock_config_which):
        """Test _is_vllm_running returning True."""
        mock_config_which.return_value = "/usr/local/bin/vllm"
        mock_simulate_which.return_value = "/usr/bin/pgrep"  # _is_vllm_running uses shutil.which from simulate.
        mock_run.return_value = MagicMock(stdout="5\n", returncode=0)
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        result = simulator._is_vllm_running()
        self.assertTrue(result)

    @patch("experimental.optix.config.custom_command.shutil.which")
    @patch("experimental.optix.optimizer.plugins.simulate.shutil.which")
    @patch("experimental.optix.optimizer.plugins.simulate.subprocess.run")
    def test_is_vllm_running_false(self, mock_run, mock_simulate_which, mock_config_which):
        """Test _is_vllm_running returning False."""
        mock_config_which.return_value = "/usr/local/bin/vllm"
        mock_simulate_which.return_value = "/usr/bin/pgrep"
        mock_run.return_value = MagicMock(stdout="0\n", returncode=0)
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        result = simulator._is_vllm_running()
        self.assertFalse(result)

    @patch("experimental.optix.config.custom_command.shutil.which")
    @patch("experimental.optix.optimizer.plugins.simulate.shutil.which")
    @patch("experimental.optix.optimizer.plugins.simulate.subprocess.run")
    def test_is_vllm_running_exception(self, mock_run, mock_simulate_which, mock_config_which):
        """Test _is_vllm_running exception handling."""
        mock_config_which.return_value = "/usr/local/bin/vllm"
        mock_simulate_which.return_value = "/usr/bin/pgrep"
        mock_run.side_effect = subprocess.SubprocessError("Command failed")
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        result = simulator._is_vllm_running()
        self.assertFalse(result)

    @patch("experimental.optix.config.custom_command.shutil.which")
    @patch("experimental.optix.optimizer.plugins.simulate.time.time")
    def test_wait_for_process_exit_success(self, mock_time, mock_which):
        """Test _wait_for_process_exit on successful exit."""
        mock_which.return_value = "/usr/local/bin/vllm"
        mock_time.side_effect = [0, 0.3, 0.6]  # Simulate elapsed time.
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)

        with patch.object(simulator, "_is_vllm_running", side_effect=[True, False]):
            result = simulator._wait_for_process_exit(timeout=1)
            self.assertTrue(result)

    @patch("experimental.optix.config.custom_command.shutil.which")
    @patch("experimental.optix.optimizer.plugins.simulate.time.time")
    def test_wait_for_process_exit_timeout(self, mock_time, mock_which):
        """Test _wait_for_process_exit timeout."""
        mock_which.return_value = "/usr/local/bin/vllm"
        mock_time.side_effect = [0, 1, 2, 3]  # Simulate timeout.
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)

        with patch.object(simulator, "_is_vllm_running", return_value=True):
            result = simulator._wait_for_process_exit(timeout=1)
            self.assertFalse(result)

    @patch("experimental.optix.config.custom_command.shutil.which")
    @patch("experimental.optix.optimizer.plugins.simulate.subprocess.run")
    def test_log_residual_processes(self, mock_run, mock_which):
        """Test the _log_residual_processes method."""
        mock_which.return_value = "/usr/local/bin/vllm"
        mock_run.return_value = MagicMock(stdout="1234 /usr/bin/vllm\n5678 /usr/bin/vllm\n")
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        simulator._log_residual_processes()
        mock_run.assert_called_once()

    @patch("experimental.optix.config.custom_command.shutil.which")
    @patch("experimental.optix.optimizer.plugins.simulate.subprocess.run")
    def test_log_residual_processes_exception(self, mock_run, mock_which):
        """Test _log_residual_processes exception handling."""
        mock_which.return_value = "/usr/local/bin/vllm"
        mock_run.side_effect = subprocess.SubprocessError("Command failed")
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        simulator._log_residual_processes()  # Should not raise an exception.

    @patch("experimental.optix.config.custom_command.shutil.which")
    def test_update_command(self, mock_which):
        """Test the update_command method."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        simulator.update_command()
        self.assertIsNotNone(simulator.command)
