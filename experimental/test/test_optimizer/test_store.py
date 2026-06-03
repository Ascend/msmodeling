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

from experimental.optix.config.config import PerformanceIndex, OptimizerConfigField, get_settings
from experimental.optix.optimizer.store import DataStorage


class TestDataStorage(unittest.TestCase):
    def setUp(self):
        self.data_storage = DataStorage(get_settings().data_storage, MagicMock(), MagicMock())

    @patch('experimental.optix.optimizer.store.Path')
    @patch('experimental.optix.optimizer.store.csv')
    @patch('msguard.security.sanitize_csv_value')
    def test_save_existing_file(self, mock_sanitize_csv_value, mock_csv, mock_path):
        # Configure mock behavior.
        mock_path.exists.return_value = True
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_path.open.return_value = mock_file

        # Create a DataStorage instance.
        config = MagicMock()
        config.store_dir = Path('/tmp/fake/dir')
        storage = DataStorage(config)

        # Create test data.
        performance_index = PerformanceIndex()
        params = [OptimizerConfigField(name='param1', value=1), OptimizerConfigField(name='param2', value=2)]
        kwargs = {'key1': 'value1', 'key2': 'value2'}

        # Call the save method.
        storage.save(performance_index, params, **kwargs)

    @patch('experimental.optix.optimizer.store.Path')
    def test_load_history_position_dir_not_exist(self, mock_path):
        mock_path.exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            DataStorage.load_history_position(mock_path)

    @patch('experimental.optix.optimizer.store.Path')
    def test_load_history_position_not_a_dir(self, mock_path):
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = False
        with self.assertRaises(ValueError):
            DataStorage.load_history_position(mock_path)

    @patch('experimental.optix.optimizer.store.Path')
    @patch('experimental.optix.optimizer.store.read_csv_s')
    def test_load_history_position_no_data(self, mock_read_csv_s, mock_path):
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True
        mock_path.iterdir.return_value = []
        result = DataStorage.load_history_position(mock_path)
        self.assertIsNone(result)

    @patch('experimental.optix.optimizer.store.Path')
    @patch('experimental.optix.optimizer.store.read_csv_s')
    def test_load_history_position_with_data(self, mock_read_csv_s, mock_path):
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True
        mock_file = MagicMock()
        mock_file.name.startswith.return_value = True
        mock_file.suffix = '.csv'
        mock_path.iterdir.return_value = [mock_file]
        mock_read_csv_s.return_value.to_dict.return_value = [{'data': 'value'}]

        result = DataStorage.load_history_position(mock_path)
        self.assertEqual(result, [{'data': 'value'}])

    def test_filter_data_no_filter_field(self):
        data_rows = [{'data': 'value'}, {'data': 'value2'}]
        result = DataStorage.filter_data(data_rows)
        self.assertEqual(result, data_rows)

    def test_filter_data_with_filter_field(self):
        data_rows = [
            {'data': 'value', 'filter': 'field1'},
            {'data': 'value2', 'filter': 'field2'},
            {'data': 'value3', 'filter': 'field1'},
        ]
        filter_field = {'filter': 'field1'}
        result = DataStorage.filter_data(data_rows, filter_field)
        self.assertEqual(result, [{'data': 'value', 'filter': 'field1'}, {'data': 'value3', 'filter': 'field1'}])

    def test_filter_data_with_non_matching_filter_field(self):
        data_rows = [
            {'data': 'value', 'filter': 'field1'},
            {'data': 'value2', 'filter': 'field2'},
            {'data': 'value3', 'filter': 'field1'},
        ]
        filter_field = {'filter': 'field3'}
        result = DataStorage.filter_data(data_rows, filter_field)
        self.assertEqual(result, [])
