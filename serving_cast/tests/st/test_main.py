# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
import shutil
import subprocess
import tempfile

import pytest
import yaml

# Configuration content required by test cases
VALID_INSTANCE_CONFIG = {
    "instance_groups": [
        {
            "num_instances": 2,
            "num_devices_per_instance": 4,
            "pd_role": "both",
            "parallel_config": {
                "world_size": 4,
                "tp_size": 4,
                "dp_size": 1,
                "ep_size": 1,
                "moe_tp_size": 4,
                "moe_dp_size": 1,
            },
        }
    ]
}

VALID_COMMON_CONFIG = {
    "model_config": {
        "name": "Qwen/Qwen3-32B",
    },
    "load_gen": {
        "load_gen_type": "fixed_length",
        "num_requests": 10,  # Reduce request count to speed up tests
        "num_input_tokens": 30,
        "num_output_tokens": 5,
        "request_rate": 2.0,
    },
}

INVALID_INSTANCE_CONFIG = {
    "instance_groups": [
        {
            "num_instances": 2,
            # Missing required field num_devices_per_instance
            "pd_role": "both",
            "parallel_config": {"tp_size": 4},
        }
    ]
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class TestCLI:
    """Command-line system test class"""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Environment preparation and cleanup before and after tests"""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.profiling_dir = os.path.join(self.temp_dir, "profiling")

        # Create valid configuration files
        self.valid_instance_path = self._create_config_file(VALID_INSTANCE_CONFIG, "instance_valid.yaml")
        self.valid_common_path = self._create_config_file(VALID_COMMON_CONFIG, "common_valid.yaml")

        # Create invalid configuration file
        self.invalid_instance_path = self._create_config_file(INVALID_INSTANCE_CONFIG, "instance_invalid.yaml")

        yield  # Test execution

        # Clean up temporary files
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_config_file(self, content, filename):
        """Create configuration file for testing"""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(content, f)
        return file_path

    def _run_command(self, args, check=True):
        """Run command line and return result"""
        cmd = ["python", "-m", "serving_cast.main"] + args
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return result

    def test_basic_functionality(self):
        """Test basic functionality: run command with valid configuration"""
        # Build command arguments
        args = [
            f"--instance_config_path={self.valid_instance_path}",
            f"--common_config_path={self.valid_common_path}",
        ]

        # Execute command
        result = self._run_command(args)

        # Verify results
        assert result.returncode == 0, "Command execution failed"
        assert "Simulation" in result.stdout or "Summary" in result.stdout, "Expected output not found"

    def test_missing_required_args(self):
        """Test case of missing required arguments"""
        # Provide only one required argument
        args = [f"--instance_config_path={self.valid_instance_path}"]

        # Execute command, expected to fail
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            self._run_command(args)

        # Verify error code
        assert exc_info.value.returncode != 0, "Command should not succeed when required arguments are missing"
        assert "error: the following arguments are required" in exc_info.value.stderr

    def test_invalid_file_path(self):
        """Test case of invalid file path"""
        # Use non-existent file path
        args = [
            "--instance_config_path=./nonexistent_instance.yaml",
            f"--common_config_path={self.valid_common_path}",
        ]

        # Execute command, expected to fail
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            self._run_command(args)

        # Verify error message
        assert "invalid validate_file_path value" in exc_info.value.stderr

    def test_invalid_config_content(self):
        """Test case of invalid configuration file content"""
        # Use invalid instance configuration
        args = [
            f"--instance_config_path={self.invalid_instance_path}",
            f"--common_config_path={self.valid_common_path}",
        ]

        # Execute command, expected to fail
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            self._run_command(args)

        # Verify command execution failed
        assert exc_info.value.returncode != 0
