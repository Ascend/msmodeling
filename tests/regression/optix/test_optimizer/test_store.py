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
    OptimizerConfigField,
    PerformanceIndex,
    get_settings,
)
from optix.optimizer.store import DataStorage


class TestDataStorage(unittest.TestCase):
    def setUp(self):
        self.data_storage = DataStorage(get_settings().data_storage, MagicMock(), MagicMock())

    @patch("optix.optimizer.store.open_file")
    @patch("optix.optimizer.store.csv")
    @patch("optix.optimizer.store.sanitize_csv_value", side_effect=lambda value: value)
    def test_save_existing_file(self, mock_sanitize_csv_value, mock_csv, mock_open_file):
        mock_file = MagicMock()
        mock_open_file.return_value.__enter__.return_value = mock_file
        mock_writer = MagicMock()
        mock_csv.writer.return_value = mock_writer

        config = MagicMock()
        config.store_dir = Path("/tmp/fake/dir")
        storage = DataStorage(config, MagicMock(), MagicMock())
        storage.save_file = MagicMock()
        storage.save_file.exists.return_value = True

        performance_index = PerformanceIndex()
        params = (
            OptimizerConfigField(name="param1", value=1),
            OptimizerConfigField(name="param2", value=2),
        )
        kwargs = {"key1": "value1", "key2": "value2"}

        storage.save(performance_index, params, **kwargs)

        mock_open_file.assert_called_once_with(storage.save_file, "a+")
        mock_csv.writer.assert_called_once_with(mock_file)
        mock_writer.writerow.assert_called_once()
        written_row = mock_writer.writerow.call_args[0][0]
        assert len(written_row) > 0
        assert mock_sanitize_csv_value.call_count == len(written_row)

    def test_save_strips_leading_double_dash_before_sanitize(self):
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp())
        config = MagicMock()
        config.store_dir = tmp_dir
        storage = DataStorage(config)
        performance_index = PerformanceIndex()
        params = (OptimizerConfigField(name="cmd", value="--max-batch-size=32"),)
        storage.save(performance_index, params)
        content = storage.save_file.read_text(encoding="utf-8")
        assert "--max-batch-size" not in content
        assert "max-batch-size=32" in content

    def test_save_strips_embedded_double_dash(self):
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp())
        config = MagicMock()
        config.store_dir = tmp_dir
        storage = DataStorage(config)
        performance_index = PerformanceIndex()
        params = (OptimizerConfigField(name="cmd", value="foo--bar"),)
        storage.save(performance_index, params)
        content = storage.save_file.read_text(encoding="utf-8")
        assert "foo--bar" not in content
        assert "foobar" in content

    def test_save_metrics_sample_creates_joinable_trace_csv(self):
        import csv
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp())
        config = MagicMock()
        config.store_dir = tmp_dir
        storage = DataStorage(config)
        case_id = storage.next_case_id()

        storage.save_metrics_sample(
            {"timestamp": 10.0, "running_requests": 52, "warmup_end_reason": "load_ready"},
            case_id=case_id,
            benchmark_phase="evaluation",
            benchmark_pass=1,
        )

        with storage.metrics_save_file.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(case_id, "case_000001")
        self.assertEqual(rows[0]["case_id"], case_id)
        self.assertEqual(rows[0]["running_requests"], "52")
        self.assertEqual(rows[0]["warmup_end_reason"], "load_ready")

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
        lines = [line for line in storage.save_file.read_text().splitlines() if line]
        # Header + 2 data rows
        assert len(lines) == 3

    @patch("optix.optimizer.store.logger")
    def test_save_logs_csv_path_for_every_case(self, mock_logger):
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp())
        config = MagicMock()
        config.store_dir = tmp_dir
        storage = DataStorage(config)
        performance_index = PerformanceIndex()
        params = (OptimizerConfigField(name="p1", value=1),)

        storage.save(performance_index, params)
        storage.save(performance_index, params)

        assert mock_logger.info.call_count == 2
        mock_logger.info.assert_called_with(
            "Save result with DataStorage. File path: {!r}",
            storage.save_file,
        )

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
        import csv
        import tempfile

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
        import csv
        import tempfile

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
        import csv
        import tempfile

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

    def test_get_best_result_excludes_early_exit_and_unusable_rows(self):
        import csv
        import tempfile

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
                    "early_exit",
                    "usable_as_best",
                    "candidate_id",
                ]
            )
            writer.writerow([0.1, 3000, 0.2, 0.02, 1.0, 6.0, True, False, "early"])
            writer.writerow([0.2, 2500, 0.3, 0.03, 1.0, 5.0, False, False, "unusable"])
            writer.writerow([0.5, 2000, 0.4, 0.04, 1.0, 4.0, False, True, "complete"])

        with patch("optix.optimizer.store.get_settings") as mock_settings:
            mock_settings.return_value.ttft_penalty = 0
            mock_settings.return_value.tpot_penalty = 0
            result = storage.get_best_result()

        assert result["candidate_id"].tolist() == ["complete"]

    def test_get_reference_performance_index_is_independent_of_pso_top_k(self):
        import csv
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp())
        config = MagicMock()
        config.store_dir = tmp_dir
        config.pso_top_k = 0
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
                    "early_exit",
                    "usable_as_best",
                    "metrics_window_generate_speed",
                    "metrics_window_time_to_first_token",
                    "metrics_window_time_per_output_token",
                    "metrics_window_success_rate",
                ]
            )
            writer.writerow([0.1, 3000, 0.2, 0.02, 1.0, 6.0, True, False, 2900, 0.21, 0.021, 1.0])
            writer.writerow([0.8, 1500, 0.5, 0.05, 1.0, 3.0, False, True, 1400, 0.55, 0.055, 0.97])
            writer.writerow([0.5, 2000, 0.4, 0.04, 1.0, 4.0, False, True, 1800, 0.45, 0.045, 0.98])

        with patch("optix.optimizer.store.get_settings") as mock_settings:
            mock_settings.return_value.ttft_penalty = 0
            mock_settings.return_value.tpot_penalty = 0
            reference = storage.get_reference_performance_index()

        assert reference is not None
        assert reference.generate_speed == 1800
        assert reference.time_to_first_token == 0.45
        assert reference.time_per_output_token == 0.045
        assert reference.success_rate == 0.98

    def test_get_reference_performance_index_does_not_fallback_to_benchmark_metrics(self):
        import csv
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp())
        config = MagicMock()
        config.store_dir = tmp_dir
        config.pso_top_k = 3
        storage = DataStorage(config, benchmark=None)
        storage.save_file = tmp_dir / "data.csv"
        with open(storage.save_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "fitness",
                    "generate_speed",
                    "time_to_first_token",
                    "time_per_output_token",
                    "success_rate",
                    "throughput",
                    "early_exit",
                    "usable_as_best",
                ]
            )
            writer.writerow([0.5, 2000, 0.4, 0.04, 1.0, 4.0, False, True])

        with patch("optix.optimizer.store.get_settings") as mock_settings:
            mock_settings.return_value.ttft_penalty = 0
            mock_settings.return_value.tpot_penalty = 0
            reference = storage.get_reference_performance_index()

        assert reference is None
