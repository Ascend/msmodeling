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
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from optix.config.config import (
    PerformanceIndex,
    OptimizerConfigField,
    get_settings,
)
from optix.optimizer.store import DataStorage


class TestDataStorage(unittest.TestCase):
    def setUp(self):
        self.data_storage = DataStorage(get_settings().data_storage, MagicMock(), MagicMock())

    @patch("optix.optimizer.store.Path")
    @patch("optix.optimizer.store.csv")
    @patch("optix.optimizer.store.sanitize_csv_value")
    def test_save_existing_file(self, mock_sanitize_csv_value, mock_csv, mock_path):
        # Configure mock behavior.
        mock_path.exists.return_value = True
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_path.open.return_value = mock_file

        # Create a DataStorage instance.
        config = MagicMock()
        config.store_dir = Path("/tmp/fake/dir")
        storage = DataStorage(config)

        # Create test data.
        performance_index = PerformanceIndex()
        params = [
            OptimizerConfigField(name="param1", value=1),
            OptimizerConfigField(name="param2", value=2),
        ]
        kwargs = {"key1": "value1", "key2": "value2"}

        # Call the save method.
        storage.save(performance_index, params, **kwargs)

    @patch("optix.optimizer.store.Path")
    def test_load_history_position_dir_not_exist(self, mock_path):
        mock_path.exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            DataStorage.load_history_position(mock_path)

    @patch("optix.optimizer.store.Path")
    def test_load_history_position_not_a_dir(self, mock_path):
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = False
        with self.assertRaises(ValueError):
            DataStorage.load_history_position(mock_path)

    @patch("optix.optimizer.store.Path")
    @patch("optix.optimizer.store.read_csv_s")
    def test_load_history_position_no_data(self, mock_read_csv_s, mock_path):
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True
        mock_path.iterdir.return_value = []
        result = DataStorage.load_history_position(mock_path)
        self.assertIsNone(result)

    @patch("optix.optimizer.store.Path")
    @patch("optix.optimizer.store.read_csv_s")
    def test_load_history_position_with_data(self, mock_read_csv_s, mock_path):
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True
        mock_file = MagicMock()
        mock_file.name.startswith.return_value = True
        mock_file.suffix = ".csv"
        mock_path.iterdir.return_value = [mock_file]
        mock_read_csv_s.return_value.to_dict.return_value = [{"data": "value"}]

        result = DataStorage.load_history_position(mock_path)
        self.assertEqual(result, [{"data": "value"}])

    def test_filter_data_no_filter_field(self):
        data_rows = [{"data": "value"}, {"data": "value2"}]
        result = DataStorage.filter_data(data_rows)
        self.assertEqual(result, data_rows)

    def test_filter_data_with_filter_field(self):
        data_rows = [
            {"data": "value", "filter": "field1"},
            {"data": "value2", "filter": "field2"},
            {"data": "value3", "filter": "field1"},
        ]
        filter_field = {"filter": "field1"}
        result = DataStorage.filter_data(data_rows, filter_field)
        self.assertEqual(
            result,
            [
                {"data": "value", "filter": "field1"},
                {"data": "value3", "filter": "field1"},
            ],
        )

    def test_filter_data_with_non_matching_filter_field(self):
        data_rows = [
            {"data": "value", "filter": "field1"},
            {"data": "value2", "filter": "field2"},
            {"data": "value3", "filter": "field1"},
        ]
        filter_field = {"filter": "field3"}
        result = DataStorage.filter_data(data_rows, filter_field)
        self.assertEqual(result, [])

    def test_filter_data_with_int_values(self):
        data_rows = [
            {"name": "a", "count": 10},
            {"name": "b", "count": 20},
        ]
        filter_field = {"count": 10}
        result = DataStorage.filter_data(data_rows, filter_field)
        self.assertEqual(result, [{"name": "a", "count": 10}])

    def test_filter_data_with_float_values(self):
        data_rows = [
            {"name": "a", "score": 3.14},
            {"name": "b", "score": 2.71},
        ]
        filter_field = {"score": 3.14}
        result = DataStorage.filter_data(data_rows, filter_field)
        self.assertEqual(result, [{"name": "a", "score": 3.14}])

    def test_filter_data_key_not_in_record(self):
        data_rows = [{"name": "a"}, {"name": "b", "extra": "val"}]
        filter_field = {"extra": "val"}
        result = DataStorage.filter_data(data_rows, filter_field)
        self.assertEqual(result, [{"name": "b", "extra": "val"}])

    def test_get_run_info_no_benchmark(self):
        config = MagicMock()
        config.store_dir = Path("/tmp/fake/dir")
        storage = DataStorage(config, benchmark=None)
        assert storage.get_run_info() == {}

    def test_get_run_info_with_vllm_benchmark(self):
        from optix.optimizer.plugins.benchmark import VllmBenchMark

        config = MagicMock()
        config.store_dir = Path("/tmp/fake/dir")
        mock_benchmark = MagicMock()
        mock_benchmark.__class__ = VllmBenchMark
        mock_benchmark.config.command.num_prompts = 100
        storage = DataStorage(config, benchmark=mock_benchmark)
        info = storage.get_run_info()
        assert info["num_prompts"] == 100

    def test_get_run_info_with_generic_benchmark(self):
        config = MagicMock()
        config.store_dir = Path("/tmp/fake/dir")
        mock_benchmark = MagicMock()
        mock_benchmark.num_prompts = 50
        # Make isinstance check fail for AisBench/VllmBenchMark
        mock_benchmark.__class__ = type("GenericBench", (), {})
        storage = DataStorage(config, benchmark=mock_benchmark)
        info = storage.get_run_info()
        assert info["num_prompts"] == 50

    def test_save_creates_new_file(self, tmp_path=None):
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp())
        config = MagicMock()
        config.store_dir = tmp_dir
        storage = DataStorage(config)
        performance_index = PerformanceIndex()
        params = (OptimizerConfigField(name="p1", value=42),)
        storage.save(performance_index, params, extra_key="extra_val")
        assert storage.save_file.exists()

    def test_save_appends_to_existing_file(self):
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp())
        config = MagicMock()
        config.store_dir = tmp_dir
        storage = DataStorage(config)
        performance_index = PerformanceIndex()
        params = (OptimizerConfigField(name="p1", value=1),)
        storage.save(performance_index, params)
        storage.save(performance_index, params)
        lines = storage.save_file.read_text().strip().split("\n")
        # Header + 2 data rows
        assert len(lines) == 3

    def test_filter_data_with_bool_values(self):
        data_rows = [
            {"name": "a", "enabled": True},
            {"name": "b", "enabled": False},
        ]
        filter_field = {"enabled": True}
        result = DataStorage.filter_data(data_rows, filter_field)
        self.assertEqual(result, [{"name": "a", "enabled": True}])

    def test_get_best_result_with_both_penalties(self):
        """Test get_best_result filters by both ttft and tpot SLOs"""
        import tempfile
        import csv

        tmp_dir = Path(tempfile.mkdtemp())
        config = MagicMock()
        config.store_dir = tmp_dir
        config.pso_top_k = 3
        mock_benchmark = MagicMock()
        mock_benchmark.config.command.num_prompts = 10
        storage = DataStorage(config, benchmark=mock_benchmark)

        # Create a CSV file with test data
        save_file = tmp_dir / "data.csv"
        storage.save_file = save_file
        with open(save_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "fitness",
                    "generate_speed",
                    "time_to_first_token",
                    "time_per_output_token",
                    "success_rate",
                    "throughput",
                    "num_prompts",
                ]
            )
            writer.writerow([0.5, 2000, 0.3, 0.04, 1.0, 4.0, 10])
            writer.writerow([0.8, 1500, 0.4, 0.03, 1.0, 3.5, 10])
            writer.writerow([0.3, 2500, 0.2, 0.02, 1.0, 5.0, 10])

        with patch("optix.optimizer.store.get_settings") as mock_settings:
            mock_settings.return_value.ttft_penalty = 3.0
            mock_settings.return_value.tpot_penalty = 3.0
            mock_settings.return_value.ttft_slo = 0.5
            mock_settings.return_value.tpot_slo = 0.05
            mock_settings.return_value.slo_coefficient = 0.1
            result = storage.get_best_result()
        assert len(result) > 0

    def test_get_best_result_tpot_only(self):
        """Test get_best_result filters by tpot penalty only"""
        import tempfile
        import csv

        tmp_dir = Path(tempfile.mkdtemp())
        config = MagicMock()
        config.store_dir = tmp_dir
        config.pso_top_k = 3
        storage = DataStorage(config, benchmark=None)

        save_file = tmp_dir / "data.csv"
        storage.save_file = save_file
        with open(save_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "fitness",
                    "generate_speed",
                    "time_to_first_token",
                    "time_per_output_token",
                    "success_rate",
                    "throughput",
                ]
            )
            writer.writerow([0.5, 2000, 0.3, 0.04, 1.0, 4.0])
            writer.writerow([0.3, 2500, 0.2, 0.02, 1.0, 5.0])

        with patch("optix.optimizer.store.get_settings") as mock_settings:
            mock_settings.return_value.ttft_penalty = 0
            mock_settings.return_value.tpot_penalty = 3.0
            mock_settings.return_value.tpot_slo = 0.05
            mock_settings.return_value.slo_coefficient = 0.1
            result = storage.get_best_result()
        assert len(result) > 0

    def test_get_best_result_no_penalty(self):
        """Test get_best_result with no penalty"""
        import tempfile
        import csv

        tmp_dir = Path(tempfile.mkdtemp())
        config = MagicMock()
        config.store_dir = tmp_dir
        config.pso_top_k = 3
        storage = DataStorage(config, benchmark=None)

        save_file = tmp_dir / "data.csv"
        storage.save_file = save_file
        with open(save_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "fitness",
                    "generate_speed",
                    "time_to_first_token",
                    "time_per_output_token",
                    "success_rate",
                    "throughput",
                ]
            )
            writer.writerow([0.5, 2000, 0.3, 0.04, 1.0, 4.0])
            writer.writerow([0.3, 2500, 0.2, 0.02, 1.0, 5.0])

        with patch("optix.optimizer.store.get_settings") as mock_settings:
            mock_settings.return_value.ttft_penalty = 0
            mock_settings.return_value.tpot_penalty = 0
            result = storage.get_best_result()
        assert len(result) > 0
