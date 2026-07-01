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
from optix.config.config import PerformanceIndex, get_settings
from optix.deploy_env import RuntimeContext
from optix.optimizer.plugins.benchmark import (
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
    @patch("optix.deploy_env.shutil.which")
    def setUp(self, mock_which):
        # Create a mock benchmark_config object with proper command attributes
        self.mock_benchmark_config = MagicMock()
        self.mock_benchmark_config.command.host = "127.0.0.1"
        self.mock_benchmark_config.command.port = "8000"
        self.mock_benchmark_config.command.model = "test-model"
        self.mock_benchmark_config.command.served_model_name = "test-model"
        self.mock_benchmark_config.command.dataset_name = "test-dataset"
        self.mock_benchmark_config.command.num_prompts = 10
        self.mock_benchmark_config.command.result_dir = "test_dir"
        self.mock_benchmark_config.command.others = ""
        mock_which.return_value = "/usr/local/bin/vllm"
        # Create test object and pass benchmark_config
        self.benchmark = VllmBenchMark(self.mock_benchmark_config)

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


class TestVllmBenchMarkExtended(unittest.TestCase):
    @patch("optix.deploy_env.shutil.which")
    def setUp(self, mock_which):
        self.mock_benchmark_config = MagicMock()
        self.mock_benchmark_config.command.host = "127.0.0.1"
        self.mock_benchmark_config.command.port = "8000"
        self.mock_benchmark_config.command.model = "test-model"
        self.mock_benchmark_config.command.served_model_name = "test-model"
        self.mock_benchmark_config.command.dataset_name = "test-dataset"
        self.mock_benchmark_config.command.num_prompts = 10
        self.mock_benchmark_config.command.result_dir = "test_dir_ext"
        self.mock_benchmark_config.command.others = ""
        mock_which.return_value = "/usr/local/bin/vllm"
        self.benchmark = VllmBenchMark(self.mock_benchmark_config)

    def test_num_prompts_property(self):
        assert self.benchmark.num_prompts == 10

    def test_num_prompts_setter(self):
        self.benchmark.num_prompts = 200
        assert self.benchmark.config.command.num_prompts == 200

    @patch("optix.deploy_env.shutil.which")
    def test_update_command(self, mock_which):
        mock_which.return_value = "/usr/local/bin/vllm"
        self.benchmark.update_command()
        assert self.benchmark.command is not None

    def test_stop_removes_output(self):
        import tempfile

        tmp_dir = tempfile.mkdtemp()
        self.benchmark.config.command.result_dir = tmp_dir
        Path(tmp_dir).joinpath("test_file.json").write_text("{}", encoding="utf-8")
        self.benchmark.process = None
        self.benchmark.run_log_fp = None
        self.benchmark.run_log = None
        self.benchmark.stop(del_log=False)

    def test_before_run(self):
        from optix.config.config import OptimizerConfigField
        import tempfile

        tmp_dir = tempfile.mkdtemp()
        self.benchmark.config.command.result_dir = tmp_dir
        params = (
            OptimizerConfigField(
                name="CONCURRENCY",
                config_position="env",
                value=10,
                min=1,
                max=100,
                dtype="int",
            ),
        )
        self.benchmark.command = ["vllm", "bench", "$CONCURRENCY"]
        with patch("optix.deploy_env.shutil.which", return_value="/usr/local/bin/vllm"):
            self.benchmark.before_run(params)


class TestParseResultEdgeCases(unittest.TestCase):
    def test_numeric_input(self):
        assert parse_result(42.5) == 42.5

    def test_single_number_string(self):
        assert parse_result("100") == 100.0

    def test_none_input(self):
        assert parse_result(None) is None

    def test_integer_input(self):
        assert parse_result(100) == 100


class TestAisBenchInit(unittest.TestCase):
    """Test AisBench initialization and methods"""

    @patch("optix.optimizer.plugins.benchmark.subprocess.run")
    @patch("optix.optimizer.plugins.benchmark.open_file")
    @patch("optix.deploy_env.shutil.which")
    def test_init_with_config(self, mock_which, mock_open_file, mock_run):
        from optix.optimizer.plugins.benchmark import AisBench

        mock_which.return_value = "/usr/bin/ais_bench"
        mock_config = MagicMock()
        mock_config.work_path = "/work"
        mock_config.command.models = "model1"
        mock_config.command.datasets = "ds1"
        mock_config.command.mode = "perf"
        mock_config.command.num_prompts = 100
        mock_config.command.work_dir = "/work"
        mock_config.output_path = "/output"

        # Mock get_models_config_path subprocess.run
        # Format: 7 items after split(), containing "--models"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Type │ --models │ /path/config.py │ extra\n",
            stderr="",
        )
        # Mock open_file for reading the config
        mock_open_file.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value="data")))
        mock_open_file.return_value.__exit__ = MagicMock(return_value=False)

        bench = AisBench(config=mock_config)
        assert bench.work_path == "/work"

    @patch("optix.optimizer.plugins.benchmark.subprocess.run")
    @patch("optix.deploy_env.shutil.which")
    def test_get_models_config_path_failure(self, mock_which, mock_run):
        from optix.optimizer.plugins.benchmark import AisBench

        mock_which.return_value = "/usr/bin/ais_bench"
        mock_config = MagicMock()
        mock_config.work_path = "/work"
        mock_config.command.models = "model1"
        mock_config.command.datasets = "ds1"
        mock_config.command.mode = "perf"
        mock_config.command.num_prompts = 100
        mock_config.command.work_dir = "/work"

        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        with pytest.raises(ValueError, match="execution failed"):
            AisBench(config=mock_config)


class TestAisBenchGetPerformanceMetric(unittest.TestCase):
    """Test AisBench.get_performance_metric"""

    def setUp(self):
        self.bench = MagicMock()
        self.bench.mindie_benchmark_perf_columns = ["average", "max", "min", "p99"]

    @patch("optix.optimizer.plugins.benchmark.glob.glob")
    @patch("optix.optimizer.plugins.benchmark.pd.read_csv")
    def test_get_performance_metric_success(self, mock_read_csv, mock_glob):
        from optix.optimizer.plugins.benchmark import AisBench

        mock_glob.return_value = ["/output/run1/performances/task1/result.csv"]
        df = MagicMock()
        df.__getitem__ = MagicMock(return_value=MagicMock(tolist=MagicMock(return_value=["TTFT", "TPOT"])))
        df.columns = ["Performance Parameters", "Average", "Max"]
        df.iloc.__getitem__ = MagicMock(return_value="123 ms")
        mock_read_csv.return_value = df

        # Call via unbound method to test logic
        bench = MagicMock()
        bench.config.output_path = "/output"
        bench.mindie_benchmark_perf_columns = ["average", "max", "min"]
        result = AisBench.get_performance_metric(bench, "ttft", "average")
        assert result is not None

    @patch("optix.optimizer.plugins.benchmark.glob.glob")
    def test_get_performance_metric_no_csv(self, mock_glob):
        from optix.optimizer.plugins.benchmark import AisBench

        mock_glob.return_value = []
        bench = MagicMock()
        bench.config.output_path = "/output"
        bench.mindie_benchmark_perf_columns = ["average"]
        with pytest.raises(ValueError, match="Not Found value"):
            AisBench.get_performance_metric(bench, "ttft", "average")


class TestAisBenchGetBestConcurrency(unittest.TestCase):
    """Test AisBench.get_best_concurrency"""

    def test_get_best_concurrency_normal(self, tmp_path=None):
        import tempfile
        import os

        from optix.optimizer.plugins.benchmark import AisBench

        tmp_dir = tempfile.mkdtemp()
        perf_dir = os.path.join(tmp_dir, "run1", "performances", "task1")
        os.makedirs(perf_dir)

        # Create CSV file
        csv_path = os.path.join(perf_dir, "result.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Performance Parameters,Average\nTTFT,100 ms\n")

        # Create JSON file
        json_path = os.path.join(perf_dir, "result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "Concurrency": {"total": "20"},
                    "Max Concurrency": {"total": "100"},
                },
                f,
            )

        bench = MagicMock()
        bench.config.output_path = tmp_dir
        bench.config.best_concurrency_coefficient = 1.5
        bench.config.best_concurrency_threshold = 10

        with patch(
            "optix.optimizer.plugins.benchmark.glob.glob",
            return_value=[csv_path],
        ):
            with patch(
                "optix.optimizer.plugins.benchmark.open_file",
                side_effect=lambda *a, **kw: open(*a, **kw),  # pylint: disable=unspecified-encoding
            ):
                result = AisBench.get_best_concurrency(bench)
        assert result == 30  # 20 * 1.5 = 30

    def test_get_best_concurrency_below_threshold(self):
        from optix.optimizer.plugins.benchmark import AisBench
        import tempfile
        import os

        tmp_dir = tempfile.mkdtemp()
        perf_dir = os.path.join(tmp_dir, "run1", "performances", "task1")
        os.makedirs(perf_dir)
        csv_path = os.path.join(perf_dir, "result.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Performance Parameters,Average\nTTFT,100 ms\n")
        json_path = os.path.join(perf_dir, "result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "Concurrency": {"total": "2"},
                    "Max Concurrency": {"total": "100"},
                },
                f,
            )

        bench = MagicMock()
        bench.config.output_path = tmp_dir
        bench.config.best_concurrency_coefficient = 1.0
        bench.config.best_concurrency_threshold = 10

        with patch(
            "optix.optimizer.plugins.benchmark.glob.glob",
            return_value=[csv_path],
        ):
            with patch(
                "optix.optimizer.plugins.benchmark.open_file",
                side_effect=lambda *a, **kw: open(*a, **kw),  # pylint: disable=unspecified-encoding
            ):
                result = AisBench.get_best_concurrency(bench)
        assert result == 10  # Below threshold, use threshold


class TestAisBenchGetPerformanceIndex(unittest.TestCase):
    """Test AisBench.get_performance_index"""

    def test_get_performance_index_success(self):
        from optix.optimizer.plugins.benchmark import AisBench
        import tempfile
        import os

        tmp_dir = tempfile.mkdtemp()
        perf_dir = os.path.join(tmp_dir, "run1", "performances", "task1")
        os.makedirs(perf_dir)
        csv_path = os.path.join(perf_dir, "result.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Performance Parameters,Average\nTTFT,100 ms\nTPOT,50 ms\n")
        json_path = os.path.join(perf_dir, "result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "Total Requests": {"total": 100},
                    "Success Requests": {"total": 95},
                    "Request Throughput": {"total": "4.5 req/s"},
                    "Output Token Throughput": {"total": "2000 tokens/s"},
                },
                f,
            )

        bench = MagicMock()
        bench.config.output_path = tmp_dir
        bench.config.performance_config.time_to_first_token.metric = "ttft"
        bench.config.performance_config.time_to_first_token.algorithm = "average"
        bench.config.performance_config.time_per_output_token.metric = "tpot"
        bench.config.performance_config.time_per_output_token.algorithm = "average"
        bench.get_performance_metric = MagicMock(side_effect=[0.1, 0.05])

        with patch(
            "optix.optimizer.plugins.benchmark.glob.glob",
            return_value=[csv_path],
        ):
            with patch(
                "optix.optimizer.plugins.benchmark.open_file",
                side_effect=lambda *a, **kw: open(*a, **kw),  # pylint: disable=unspecified-encoding
            ):
                result = AisBench.get_performance_index(bench)
        assert result.throughput == 4.5
        assert result.success_rate == 0.95
        assert result.generate_speed == 2000.0


class TestAisBenchBeforeRun(unittest.TestCase):
    """Test AisBench.before_run"""

    def test_before_run_modifies_config(self):
        from optix.optimizer.plugins.benchmark import AisBench
        from optix.config.config import OptimizerConfigField
        import tempfile
        import os

        tmp_dir = tempfile.mkdtemp()
        config_file = os.path.join(tmp_dir, "config.py")
        with open(config_file, "w", encoding="utf-8") as f:
            f.write("request_rate = 10,\nbatch_size = 100,\n")

        # Create a proper AisBench mock by patching __init__
        with patch.object(AisBench, "__init__", lambda self, **kwargs: None):
            bench = AisBench()
            bench.config = MagicMock()
            bench.config.output_path = tmp_dir
            bench.model_config_path = Path(config_file)
            bench.command = ["ais_bench", "--models", "model1", "$CONCURRENCY"]
            bench.env = {}
            bench._runtime_ctx = RuntimeContext(
                in_virtualenv=False,
                virtualenv_root=None,
                python_executable=Path("/usr/bin/python"),
            )
            bench.run_log_fp = None
            bench.run_log = None
            bench.run_log_offset = 0

            params = (
                OptimizerConfigField(
                    name="CONCURRENCY",
                    config_position="env",
                    value=64,
                    min=1,
                    max=100,
                    dtype="int",
                ),
                OptimizerConfigField(
                    name="REQUESTRATE",
                    config_position="env",
                    value=5.0,
                    min=0.1,
                    max=100,
                    dtype="float",
                ),
            )

            with patch("optix.deploy_env.shutil.which", return_value="/usr/bin/ais_bench"):
                with patch("optix.optimizer.plugins.benchmark.remove_file"):
                    with patch(
                        "optix.optimizer.plugins.benchmark.open_file",
                        side_effect=lambda *a, **kw: open(*a, **kw),  # pylint: disable=unspecified-encoding
                    ):
                        bench.before_run(params)

        with open(config_file, encoding="utf-8") as f:
            content = f.read()
        assert "batch_size = 64," in content
        assert "request_rate = 5.0," in content


class TestAisBenchBackup(unittest.TestCase):
    """Test AisBench.backup"""

    def test_backup_calls_utility(self):
        from optix.optimizer.plugins.benchmark import AisBench
        import types

        bench = MagicMock()
        bench.config.output_path = "/output"
        bench.bak_path = "/bak"
        bench.run_log = "/log"
        bench.backup = types.MethodType(AisBench.backup, bench)

        with patch("optix.optimizer.plugins.benchmark.backup") as mock_backup:
            bench.backup(del_log=True)
            mock_backup.assert_called_once()

    def test_backup_with_log(self):
        from optix.optimizer.plugins.benchmark import AisBench
        import types

        bench = MagicMock()
        bench.config.output_path = "/output"
        bench.bak_path = "/bak"
        bench.run_log = "/log"
        bench.backup = types.MethodType(AisBench.backup, bench)

        with patch("optix.optimizer.plugins.benchmark.backup") as mock_backup:
            bench.backup(del_log=False)
            assert mock_backup.call_count == 2


class TestVllmBenchMarkBeforeRun(unittest.TestCase):
    """Test VllmBenchMark.before_run"""

    @patch("optix.deploy_env.shutil.which")
    def test_before_run_cleans_output(self, mock_which):
        mock_which.return_value = "/usr/bin/vllm"
        from optix.config.config import OptimizerConfigField

        mock_config = MagicMock()
        mock_config.command.host = "127.0.0.1"
        mock_config.command.port = "8000"
        mock_config.command.model = "test"
        mock_config.command.served_model_name = "test"
        mock_config.command.dataset_name = "ds"
        mock_config.command.num_prompts = 10
        mock_config.command.result_dir = "/tmp/vllm_results"
        mock_config.command.others = ""
        bench = VllmBenchMark(mock_config)

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
        bench.command = ["vllm", "bench", "$CONCURRENCY", "$REQUESTRATE"]
        with patch("optix.optimizer.plugins.benchmark.remove_file") as mock_rm:
            bench.before_run(params)
            mock_rm.assert_called_once()


class TestVllmBenchMarkGetPerformanceIndexNoJson(unittest.TestCase):
    """Test VllmBenchMark.get_performance_index with invalid json"""

    @patch("optix.deploy_env.shutil.which")
    def test_get_performance_index_json_decode_error(self, mock_which):
        import tempfile
        import os

        mock_which.return_value = "/usr/bin/vllm"
        mock_config = MagicMock()
        mock_config.command.host = "127.0.0.1"
        mock_config.command.port = "8000"
        mock_config.command.model = "test"
        mock_config.command.served_model_name = "test"
        mock_config.command.dataset_name = "ds"
        mock_config.command.num_prompts = 10
        mock_config.command.others = ""

        tmp_dir = tempfile.mkdtemp()
        mock_config.command.result_dir = tmp_dir
        # Write invalid JSON file
        json_path = os.path.join(tmp_dir, "result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write("{invalid json}")

        bench = VllmBenchMark(mock_config)
        result = bench.get_performance_index()
        # Should not crash, returns empty PerformanceIndex
        assert result is not None

    @patch("optix.deploy_env.shutil.which")
    def test_get_performance_index_zero_prompts(self, mock_which):
        import tempfile
        import os

        mock_which.return_value = "/usr/bin/vllm"
        mock_config = MagicMock()
        mock_config.command.host = "127.0.0.1"
        mock_config.command.port = "8000"
        mock_config.command.model = "test"
        mock_config.command.served_model_name = "test"
        mock_config.command.dataset_name = "ds"
        mock_config.command.num_prompts = 10
        mock_config.command.others = ""

        tmp_dir = tempfile.mkdtemp()
        mock_config.command.result_dir = tmp_dir
        json_path = os.path.join(tmp_dir, "result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "output_throughput": 1000.0,
                    "mean_ttft_ms": 100.0,
                    "mean_tpot_ms": 50.0,
                    "num_prompts": 0,
                    "completed": 0,
                    "request_throughput": 2.0,
                },
                f,
            )

        bench = VllmBenchMark(mock_config)
        result = bench.get_performance_index()
        assert result.success_rate == 0
