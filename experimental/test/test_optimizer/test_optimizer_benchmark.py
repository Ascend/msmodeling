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
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import pytest
from experimental.optix.config.config import PerformanceIndex, get_settings
from experimental.optix.optimizer.plugins.benchmark import (
    parse_result,
    VllmBenchMark,
)


settings = get_settings()


class TestParseResult(unittest.TestCase):
    def test_string_with_ms(self):
        # Test input is a string with unit 'ms'
        self.assertAlmostEqual(parse_result("123 ms"), 0.123)

    def test_string_with_us(self):
        # Test input is a string with unit 'us'
        self.assertAlmostEqual(parse_result("456 us"), 0.000456)

    def test_string_with_other_unit(self):
        # Test input is a string with unit other than ms or us
        self.assertAlmostEqual(parse_result("789 s"), 789.0)

    def test_string_without_unit(self):
        # Test input is a string without unit
        self.assertAlmostEqual(parse_result("1010"), 1010.0)


@pytest.fixture
def results_per_request_file(tmpdir):
    file_path = Path(tmpdir).joinpath("results_per_request_202507181613.json")
    data = {
        "1": {
            "input_len": 1735,
            "output_len": 1,
            "prefill_bsz": 4,
            "decode_bsz": [],
            "req_latency": 13058.372889645398,
            "latency": [13058.23168065399],
            "queue_latency": [12598012],
            "input_data": "",
            "output": "",
        },
        "2": {
            "prefill_bsz": 4,
            "decode_bsz": [],
            "req_latency": 15173.639830201864,
            "latency": [15173.517209477723],
            "queue_latency": [14708480],
            "input_data": "",
            "output": "",
        },
        "3": {
            "input_len": 1777,
            "output_len": 3,
            "prefill_bsz": 4,
            "decode_bsz": [157, 157],
            "req_latency": 15456.984990276396,
            "latency": [15178.787489421666, 208.4683496505022, 69.54238004982471],
            "queue_latency": [14711475, 127888, 3709],
            "input_data": "",
            "output": "\t\tif (",
        },
        "4": {
            "input_len": 1770,
            "output_len": 3,
            "prefill_bsz": 4,
            "decode_bsz": [157, 157],
            "req_latency": 15481.421849690378,
            "latency": [14745.695400051773, 670.0493693351746, 64.66158013790846],
            "queue_latency": [14280800, 584221, 3686],
            "input_data": "",
            "output": "Passage ",
        },
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return file_path


class TestBenchMarkGetPerformanceIndex(unittest.TestCase):
    @patch("experimental.optix.config.custom_command.shutil.which")
    def setUp(self, mock_which):
        # Create a mock benchmark_config object
        self.mock_benchmark_config = MagicMock()
        mock_which.return_value = "/usr/local/bin/vllm"
        # Create test object and pass benchmark_config
        self.benchmark = VllmBenchMark(self.mock_benchmark_config)

        # Set command attribute
        self.benchmark.config.command = MagicMock()
        self.test_dir = Path("test_dir")
        self.benchmark.config.command.result_dir = self.test_dir
        self.test_dir.mkdir(exist_ok=True)
        self.json_path = self.test_dir / "result.json"
        json_data = {
            "output_throughput": 2000.0,
            "mean_ttft_ms": 600.0,
            "mean_tpot_ms": 140.0,
            "num_prompts": 10,
            "completed": 10,
            "request_throughput": 4.0,
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_get_performance_index_normal(self):
        """Test the get_performance_index method in normal case"""

        # Call the method
        result = self.benchmark.get_performance_index()

        # Verify the result
        self.assertIsInstance(result, PerformanceIndex)
        self.assertEqual(result.generate_speed, 2000.0)
        self.assertEqual(result.time_to_first_token, 0.6)
        self.assertEqual(result.time_per_output_token, 0.14)
        self.assertEqual(result.success_rate, 1.0)
        self.assertEqual(result.throughput, 4.0)
