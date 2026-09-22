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
import json
import shutil
import re
from pathlib import Path
import tempfile
import pytest
from unittest.mock import patch, MagicMock
import csv
from optix.config.config import get_settings, AisBenchConfig, OptimizerConfigField
from optix.optimizer.plugins.benchmark import AisBench


settings = get_settings()


@pytest.fixture(scope="function")
def aisbench_test_environment():
    """Fixture creating the AisBench test environment."""
    # Use tempfile to create a temporary directory with automatic cleanup
    test_dir = Path(tempfile.mkdtemp())
    result_dir = test_dir / "ais_bench"
    result_dir.mkdir(exist_ok=True)
    aisbench_dir = result_dir / "outputs"
    aisbench_dir.mkdir(exist_ok=True)
    performance_dir = aisbench_dir / "performances"
    performance_dir.mkdir(exist_ok=True)
    output_dir = performance_dir / "api_file"
    output_dir.mkdir(exist_ok=True)

    # Create CSV and JSON test file paths
    csv_path = output_dir / "gsm8kdataset.csv"
    json_path = output_dir / "gsm8kdataset.json"

    # Create JSON test data
    json_data = {
        "Total Requests": {"total": 84},
        "Success Requests": {"total": 84},
        "Request Throughput": {"total": "2.4221 req/s"},
        "Output Token Throughput": {"total": "1240.1267 token/s"},
        "Concurrency": {"total": 40},
        "Max Concurrency": {"total": 100},
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f)

    # Create CSV test data
    data = [
        {
            "Performance Parameters": "TTFT",
            "Stage": "total",
            "Average": "146.1383 ms",
        },
        {
            "Performance Parameters": "TPOT",
            "Stage": "total",
            "Average": "30.2947 ms",
        },
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    # Return test environment data
    yield {"test_dir": test_dir, "csv_path": csv_path, "json_path": json_path}

    # Automatic cleanup after tests
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def before_run_test_environment():
    """Fixture creating the BeforeRun test environment."""
    test_dir = Path(tempfile.mkdtemp())
    benchmark_dir = test_dir / "benchmark"
    benchmark_dir.mkdir(exist_ok=True)
    config_dir = benchmark_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    model_dir = config_dir / "models"
    model_dir.mkdir(exist_ok=True)

    # Create the test config file
    code_content = '''from ais_bench.benchmark.models import VLLMCustomAPIChatStream
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="/data/models/llama3-8b",
        model="llama3-8b",
        request_rate=36,
        retry=2,
        host_ip="127.0.0.1",
        host_port=31015,
        max_out_len=512,
        batch_size=1000,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.5,
            top_k=10,
            top_p=0.95,
            seed=None,
            repetition_penalty=1.03,
        )
    )
]'''

    target_file = model_dir / "api.py"
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(code_content)

    yield {"test_dir": test_dir, "model_dir": model_dir, "api_file_path": target_file}

    # Automatic cleanup after tests
    shutil.rmtree(test_dir, ignore_errors=True)


class TestAisbench:
    """AisBench test class using pytest fixtures."""

    @patch("optix.deploy_env.shutil.which")
    @patch('optix.optimizer.plugins.benchmark.AisBench.get_models_config_path')
    @patch('optix.optimizer.plugins.benchmark.glob.glob')
    def test_get_performance_metric(self, mock_glob, mock_path, mock_which, aisbench_test_environment):
        """Test getting a performance metric."""
        mock_which.return_value = "/usr/local/bin/aisbench"
        mock_path.return_value = aisbench_test_environment["csv_path"]
        # Mock glob.glob directly to return the required csv_path
        mock_glob.return_value = [str(aisbench_test_environment["csv_path"])]

        config = AisBenchConfig()
        config.output_path = aisbench_test_environment["test_dir"]

        result = AisBench(config=config).get_performance_metric('ttft')
        assert result == 0.1461383

    @patch("optix.deploy_env.shutil.which")
    @patch('optix.optimizer.plugins.benchmark.AisBench.get_models_config_path')
    @patch('optix.optimizer.plugins.benchmark.glob.glob')
    def test_get_performance_index(self, mock_glob, mock_path, mock_which, aisbench_test_environment):
        """Test getting a performance index."""
        mock_which.return_value = "/usr/local/bin/aisbench"
        mock_path.return_value = aisbench_test_environment["csv_path"]
        # Mock glob.glob directly to return the required csv_path and json_path
        mock_glob.side_effect = [
            [str(aisbench_test_environment["csv_path"])],  # First call returns the CSV file
            [str(aisbench_test_environment["csv_path"])],  # Second call returns the CSV file
            [str(aisbench_test_environment["csv_path"])],  # Third call returns the JSON file
        ]

        config = AisBenchConfig()
        config.output_path = aisbench_test_environment["test_dir"]

        performance_index = AisBench(config=config).get_performance_index()
        assert performance_index.generate_speed == 1240.1267

    @patch("optix.deploy_env.shutil.which")
    @patch('optix.optimizer.plugins.benchmark.AisBench.get_models_config_path')
    @patch('optix.optimizer.plugins.benchmark.glob.glob')
    def test_get_best_concurrency(self, mock_glob, mock_path, mock_which, aisbench_test_environment):
        """Test getting the best concurrency."""
        mock_which.return_value = "/usr/local/bin/aisbench"
        mock_path.return_value = aisbench_test_environment["json_path"]
        # Mock glob.glob directly to return the required json_path
        mock_glob.return_value = [str(aisbench_test_environment["json_path"])]

        config = AisBenchConfig()
        config.output_path = aisbench_test_environment["test_dir"]
        config.best_concurrency_coefficient = 5.0  # Set coefficient to 5: 40*5=200
        config.best_concurrency_threshold = 200  # Set threshold to 200

        result = AisBench(config=config).get_best_concurrency()
        # 40*5=200 is not below the threshold(200), so it is truncated by Max Concurrency=100
        assert result == 100


class TestBeforeRun:
    """BeforeRun test class using pytest fixtures."""

    @patch("optix.deploy_env.shutil.which")
    @patch('optix.optimizer.plugins.benchmark.AisBench.get_models_config_path')
    def test_before_run_file_exists(self, mock_path, mock_which, before_run_test_environment):
        """Test the case where the file exists and request_rate/batch_size are successfully modified."""
        mock_which.return_value = "/usr/local/bin/aisbench"
        mock_module = MagicMock()
        mock_module.__file__ = 'ais_bench/__init__.py'
        mock_path.return_value = before_run_test_environment["api_file_path"]

        support_field = [
            OptimizerConfigField(name="CONCURRENCY", config_position="env", min=25, max=300, dtype="int", value=100),
            OptimizerConfigField(name="REQUESTRATE", config_position="env", min=1, max=25, dtype="int", value=100),
        ]

        AisBench().before_run(support_field)

        pattern = re.compile(r"request_rate\s*=\s*(\d+)")
        with open(before_run_test_environment["api_file_path"], 'r', encoding='utf-8') as f:
            content = f.read()
            match = pattern.search(content)
            assert int(match.group(1)) == 100
