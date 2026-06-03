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
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import shutil
import yaml
import requests
from msguard import GlobalConfig
import experimental.optix
from experimental.optix.config.config import KubectlConfig, OptimizerConfigField, get_settings
from experimental.optix.config.custom_command import MindieCommand
from experimental.optix.optimizer.simulator import enable_simulate_old
from experimental.optix.optimizer.plugins.simulate import Simulator, DisaggregationSimulator


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


def test_enable_simulate_with_simulator(tmpdir, monkeypatch):
    config_path = Path(tmpdir).joinpath("config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("""{
    "Version": "1.0.0",
    "ServerConfig": {
        "tlsCaFile": [
            "ca.pem"
        ],
        "tlsCert": "security/certs/server.pem"
    },
    "BackendConfig": {
        "backendName": "mindieservice_llm_engine",
        "ModelDeployConfig": {
            "maxSeqLen": 2560,
            "maxInputTokenLen": 2048,
            "truncation": false,
            "ModelConfig": [
                {
                    "modelInstanceType": "Standard"
                }
            ]
        },
        "ScheduleConfig": {
            "templateType": "Standard"
        }
    }
}""")
    get_settings().mindie.config_path = config_path
    get_settings().mindie.config_bak_path = Path(tmpdir).joinpath("config_bak.json")
    monkeypatch.setattr(MindieCommand, 'command', property(lambda self: ["echo"]))
    simulator = Simulator(get_settings().mindie)
    monkeypatch.setattr(experimental.optix.optimizer.simulator, "simulate_flag", True)
    with enable_simulate_old(simulator):
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert (
                data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["plugin_params"]
                == '{"plugin_type": "simulate"}'
            )
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "plugin_params" not in data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]


def test_enable_simulate_with_simulator_plugin_params_exists(tmpdir, monkeypatch):
    config_path = Path(tmpdir).joinpath("config.json")
    data = {
        "BackendConfig": {
            "backendName": "mindieservice_llm_engine",
            "ModelDeployConfig": {
                "maxSeqLen": 2560,
                "ModelConfig": [{"modelInstanceType": "Standard", "plugin_params": "{\"plugin_type\":\"tp\"}"}],
            },
            "ScheduleConfig": {"templateType": "Standard"},
        }
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    get_settings().mindie.config_path = config_path
    get_settings().mindie.config_bak_path = Path(tmpdir).joinpath("config_bak.json")
    monkeypatch.setattr(MindieCommand, 'command', property(lambda self: ["echo"]))
    simulator = Simulator(get_settings().mindie)
    monkeypatch.setattr(experimental.optix.optimizer.simulator, "simulate_flag", True)
    with enable_simulate_old(simulator):
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert (
                data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["plugin_params"]
                == '{"plugin_type": "tp,simulate"}'
            )
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["plugin_params"] == '{"plugin_type":"tp"}'


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

    @patch('experimental.optix.config.custom_command.shutil.which')
    def test_init(self, mock_which):
        """Test VllmSimulator initialization."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        self.assertEqual(simulator.config, self.mock_config)

    @patch('experimental.optix.config.custom_command.shutil.which')
    def test_base_url_property(self, mock_which):
        """Test the base_url property."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        expected_url = "http://localhost:8000/health"
        self.assertEqual(simulator.base_url, expected_url)

    @patch('experimental.optix.config.custom_command.shutil.which')
    @patch('experimental.optix.optimizer.plugins.simulate.subprocess.run')
    @patch('experimental.optix.optimizer.interfaces.custom_process.CustomProcess.stop')
    def test_stop(self, mock_super_stop, mock_run, mock_which):
        """Test the stop method."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        # Mock _is_vllm_running to return False, indicating no process needs to be stopped.
        with patch.object(simulator, '_is_vllm_running', return_value=False):
            simulator.stop()
        mock_super_stop.assert_called_once()

    @patch('experimental.optix.config.custom_command.shutil.which')
    @patch('experimental.optix.optimizer.plugins.simulate.subprocess.run')
    def test_stop_vllm_process_success(self, mock_run, mock_which):
        """Test _stop_vllm_process successfully stopping a process."""
        mock_which.return_value = "/usr/local/bin/vllm"
        mock_run.return_value = MagicMock(returncode=0)
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)

        # Simulate process checks: running first, then absent.
        with patch.object(simulator, '_is_vllm_running', side_effect=[True, False]):
            result = simulator._stop_vllm_process(max_attempts=1, timeout=1)
            self.assertTrue(result)

    @patch('experimental.optix.config.custom_command.shutil.which')
    @patch('experimental.optix.optimizer.plugins.simulate.subprocess.run')
    def test_stop_vllm_process_already_stopped(self, mock_run, mock_which):
        """Test _stop_vllm_process when the process is already stopped."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)

        # Simulate that the process no longer exists.
        with patch.object(simulator, '_is_vllm_running', return_value=False):
            result = simulator._stop_vllm_process(max_attempts=1, timeout=1)
            self.assertTrue(result)

    @patch('experimental.optix.config.custom_command.shutil.which')
    def test_stop_vllm_process_no_pkill(self, mock_which):
        """Test _stop_vllm_process when pkill is unavailable."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        # Simulate a running process with pkill unavailable.
        with patch.object(simulator, '_is_vllm_running', return_value=True):
            # Also mock shutil.which in the simulate module.
            with patch('experimental.optix.optimizer.plugins.simulate.shutil.which', return_value=None):
                result = simulator._stop_vllm_process()
                self.assertFalse(result)

    @patch('experimental.optix.config.custom_command.shutil.which')
    @patch('experimental.optix.optimizer.plugins.simulate.shutil.which')
    @patch('experimental.optix.optimizer.plugins.simulate.subprocess.run')
    def test_is_vllm_running_true(self, mock_run, mock_simulate_which, mock_config_which):
        """Test _is_vllm_running returning True."""
        mock_config_which.return_value = "/usr/local/bin/vllm"
        mock_simulate_which.return_value = "/usr/bin/pgrep"  # _is_vllm_running uses shutil.which from simulate.
        mock_run.return_value = MagicMock(stdout="5\n", returncode=0)
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        result = simulator._is_vllm_running()
        self.assertTrue(result)

    @patch('experimental.optix.config.custom_command.shutil.which')
    @patch('experimental.optix.optimizer.plugins.simulate.shutil.which')
    @patch('experimental.optix.optimizer.plugins.simulate.subprocess.run')
    def test_is_vllm_running_false(self, mock_run, mock_simulate_which, mock_config_which):
        """Test _is_vllm_running returning False."""
        mock_config_which.return_value = "/usr/local/bin/vllm"
        mock_simulate_which.return_value = "/usr/bin/pgrep"
        mock_run.return_value = MagicMock(stdout="0\n", returncode=0)
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        result = simulator._is_vllm_running()
        self.assertFalse(result)

    @patch('experimental.optix.config.custom_command.shutil.which')
    @patch('experimental.optix.optimizer.plugins.simulate.shutil.which')
    @patch('experimental.optix.optimizer.plugins.simulate.subprocess.run')
    def test_is_vllm_running_exception(self, mock_run, mock_simulate_which, mock_config_which):
        """Test _is_vllm_running exception handling."""
        mock_config_which.return_value = "/usr/local/bin/vllm"
        mock_simulate_which.return_value = "/usr/bin/pgrep"
        mock_run.side_effect = subprocess.SubprocessError("Command failed")
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        result = simulator._is_vllm_running()
        self.assertFalse(result)

    @patch('experimental.optix.config.custom_command.shutil.which')
    @patch('experimental.optix.optimizer.plugins.simulate.time.time')
    def test_wait_for_process_exit_success(self, mock_time, mock_which):
        """Test _wait_for_process_exit on successful exit."""
        mock_which.return_value = "/usr/local/bin/vllm"
        mock_time.side_effect = [0, 0.3, 0.6]  # Simulate elapsed time.
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)

        with patch.object(simulator, '_is_vllm_running', side_effect=[True, False]):
            result = simulator._wait_for_process_exit(timeout=1)
            self.assertTrue(result)

    @patch('experimental.optix.config.custom_command.shutil.which')
    @patch('experimental.optix.optimizer.plugins.simulate.time.time')
    def test_wait_for_process_exit_timeout(self, mock_time, mock_which):
        """Test _wait_for_process_exit timeout."""
        mock_which.return_value = "/usr/local/bin/vllm"
        mock_time.side_effect = [0, 1, 2, 3]  # Simulate timeout.
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)

        with patch.object(simulator, '_is_vllm_running', return_value=True):
            result = simulator._wait_for_process_exit(timeout=1)
            self.assertFalse(result)

    @patch('experimental.optix.config.custom_command.shutil.which')
    @patch('experimental.optix.optimizer.plugins.simulate.subprocess.run')
    def test_log_residual_processes(self, mock_run, mock_which):
        """Test the _log_residual_processes method."""
        mock_which.return_value = "/usr/local/bin/vllm"
        mock_run.return_value = MagicMock(stdout="1234 /usr/bin/vllm\n5678 /usr/bin/vllm\n")
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        simulator._log_residual_processes()
        mock_run.assert_called_once()

    @patch('experimental.optix.config.custom_command.shutil.which')
    @patch('experimental.optix.optimizer.plugins.simulate.subprocess.run')
    def test_log_residual_processes_exception(self, mock_run, mock_which):
        """Test _log_residual_processes exception handling."""
        mock_which.return_value = "/usr/local/bin/vllm"
        mock_run.side_effect = subprocess.SubprocessError("Command failed")
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        simulator._log_residual_processes()  # Should not raise an exception.

    @patch('experimental.optix.config.custom_command.shutil.which')
    def test_update_command(self, mock_which):
        """Test the update_command method."""
        mock_which.return_value = "/usr/local/bin/vllm"
        from experimental.optix.optimizer.plugins.simulate import VllmSimulator

        simulator = VllmSimulator(self.mock_config)
        simulator.update_command()
        self.assertIsNotNone(simulator.command)


class TestDisaggregationSimulator(unittest.TestCase):
    def setUp(self):
        # Create a temporary test environment.
        self.test_dir = Path("conf")
        self.yaml_dir = Path("deployment")
        self.test_dir.mkdir(exist_ok=True)
        self.yaml_dir.mkdir(exist_ok=True)
        self.config_single_path = self.test_dir / "config.json"
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"MindIE-MS coordinator is ready!!!")
            self.temp_file_name = temp_file.name
        data = {
            "BackendConfig": {
                "backendName": "mindieservice_llm_engine",
                "ModelDeployConfig": {
                    "maxSeqLen": 2560,
                    "ModelConfig": [{"modelInstanceType": "Standard", "plugin_params": "{\"plugin_type\":\"tp\"}"}],
                },
                "ScheduleConfig": {"templateType": "Standard"},
            }
        }
        with open(self.config_single_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        pd_data = {"default_p_rate": 1, "default_d_rate": 3}
        self.kubectl_single_path = self.test_dir / "deploy.sh"
        self.config_single_pd_path = self.test_dir / "ms_controller.json"
        self.yaml_path = self.yaml_dir / "mindie_service_single_container.yaml"
        service_config = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "mindie-service", "labels": {"app": "mindie-server"}},
            "spec": {
                "selector": {"app": "mindie-server"},
                "ports": [{"name": "http", "port": 1025, "targetPort": 1025, "nodePort": 31015, "protocol": "TCP"}],
                "type": "NodePort",
                "sessionAffinity": "None",
            },
        }
        with open(self.yaml_path, "w", encoding="utf-8") as file:
            yaml.dump(service_config, file, default_flow_style=False)
        with open(self.config_single_pd_path, "w", encoding="utf-8") as fout:
            json.dump(pd_data, fout)
        self.config_single_bak_path = self.test_dir / "config_bak.json"
        self.config_single_pd_bak_path = self.test_dir / "ms_bak_controller.json"

    def tearDown(self):
        # Clean up temporary directories.
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.yaml_dir)
        os.unlink(self.temp_file_name)

    def test_set_config_dict(self):
        origin_config = {"a": {"b": {"c": 3}}}
        DisaggregationSimulator.set_config(origin_config, "a.b.c", 4)
        assert origin_config["a"]["b"]["c"] == 4

    def test_set_config_list(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        DisaggregationSimulator.set_config(origin_config, "a.b.0.c", 4)
        assert origin_config["a"]["b"][0]["c"] == 4

    def test_set_config_new_key(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        DisaggregationSimulator.set_config(origin_config, "a.b.0.d", 4)
        assert origin_config["a"]["b"][0]["d"] == 4

    def test_set_config_add_dict_list_dict(self):
        origin_config = {"a": {"b": {"c": 3}}}
        DisaggregationSimulator.set_config(origin_config, "a.d.0.c", 4)
        assert origin_config["a"]["d"][0]["c"] == 4

    def test_set_config_add_dict(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        DisaggregationSimulator.set_config(origin_config, "a.b.1.c", 4)
        assert origin_config["a"]["b"][1]["c"] == 4

    def test_set_config_add_dict_list_dict_dict(self):
        origin_config = {"a": {"b": [{"c": 3}]}}
        DisaggregationSimulator.set_config(origin_config, "a.d.0.c.e", 4)
        assert origin_config["a"]["d"][0]["c"]["e"] == 4

    @patch('experimental.optix.optimizer.plugins.simulate.logger')
    def test_is_int(self, mock_logger):
        # Test the is_int method.
        self.assertTrue(DisaggregationSimulator.is_int('123'))
        self.assertFalse(DisaggregationSimulator.is_int('abc'))

    @patch('experimental.optix.optimizer.plugins.simulate.logger')
    def test_stop(self, mock_logger):
        # Test the stop method.
        mindie_config = KubectlConfig()
        simulator = DisaggregationSimulator(mindie_config)
        simulator.stop()
        # Verify that logging was performed correctly.
        mock_logger.debug.assert_called()

    @patch('requests.post')
    def test_curl_success(self, mock_post):
        # Arrange
        mindie_config = KubectlConfig()
        mindie_config.config_single_path = self.config_single_path
        mindie_config.kubectl_single_path = self.kubectl_single_path
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        test_class = DisaggregationSimulator(mindie_config)

        # Act
        result = test_class.test_curl()

        # Assert
        self.assertTrue(result)
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_curl_failure(self, mock_post):
        # Arrange
        mindie_config = KubectlConfig()
        mindie_config.kubectl_single_path = self.kubectl_single_path
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response
        test_class = DisaggregationSimulator(mindie_config)

        # Act
        result = test_class.test_curl()

        # Assert
        self.assertFalse(result)
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_curl_exception(self, mock_post):
        # Arrange
        mindie_config = KubectlConfig()
        mindie_config.kubectl_single_path = self.kubectl_single_path
        mock_post.side_effect = requests.exceptions.RequestException
        test_class = DisaggregationSimulator(mindie_config)

        # Act
        result = test_class.test_curl()

        # Assert
        self.assertFalse(result)
        mock_post.assert_called_once()

    def test_update_config(self):
        # Arrange
        mindie_config = KubectlConfig()
        mindie_config.config_single_path = self.config_single_path
        mindie_config.config_single_pd_path = self.config_single_pd_path
        simulator = DisaggregationSimulator(mindie_config)

        # Create test parameters.
        params = [
            OptimizerConfigField(config_position="BackendConfig.ModelDeployConfig.maxSeqLen", value=4096),
            OptimizerConfigField(config_position="default_p_rate", value=2),
        ]

        # Act
        simulator.update_config(params)

        # Assert
        with open(self.config_single_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            self.assertEqual(config_data["BackendConfig"]["ModelDeployConfig"]["maxSeqLen"], 4096)

        with open(self.config_single_pd_path, "r", encoding="utf-8") as f:
            pd_config_data = json.load(f)
            self.assertEqual(pd_config_data["default_p_rate"], 2)

    @patch('experimental.optix.optimizer.plugins.simulate.DisaggregationSimulator.test_curl')
    @patch('msguard.security.io.open_s')
    def test_health(self, mock_open, mock_test_curl):
        # Arrange
        GlobalConfig.custom_return = True
        mindie_config = KubectlConfig()
        simulator = DisaggregationSimulator(mindie_config)
        simulator.run_log = self.temp_file_name
        simulator.mindie_log_offset = 0

        # Simulate reading file content that contains the success message.
        mock_file = Mock()
        mock_file.read.return_value = "MindIE-MS coordinator is ready!!!"
        mock_file.tell.return_value = 100
        mock_open.return_value.__enter__.return_value = mock_file

        # Simulate test_curl returning True.
        mock_test_curl.return_value = True

        # Act
        result = simulator.health()

        # Assert
        self.assertTrue(result)
        mock_test_curl.assert_called_once()
        GlobalConfig.reset()

    @patch('experimental.optix.optimizer.plugins.simulate.DisaggregationSimulator.update_config')
    @patch('experimental.optix.optimizer.plugins.simulate.DisaggregationSimulator.start_server')
    @patch('experimental.optix.optimizer.plugins.simulate.logger')
    def test_run(self, mock_logger, mock_start_server, mock_update_config):
        # Arrange
        mindie_config = KubectlConfig()
        simulator = DisaggregationSimulator(mindie_config)

        # Create test parameters.
        params = [OptimizerConfigField(config_position="BackendConfig.ModelDeployConfig.maxSeqLen", value=4096)]

        # Act
        simulator.run(params)

        # Assert
        mock_logger.info.assert_called_once()
        mock_update_config.assert_called_once_with(params)
        mock_start_server.assert_called_once_with(params)
