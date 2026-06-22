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
from unittest.mock import patch, MagicMock


import optix.plugins as plugins_module
from optix.plugins import (
    _iter_entry_points,
    load_plugins_by_group,
    load_general_plugins,
)


class TestIterEntryPoints:
    @patch("optix.plugins.entry_points")
    def test_with_select(self, mock_eps):
        mock_result = MagicMock()
        mock_result.select.return_value = ["ep1", "ep2"]
        mock_eps.return_value = mock_result
        result = list(_iter_entry_points("test.group"))
        assert result == ["ep1", "ep2"]

    @patch("optix.plugins.entry_points")
    def test_without_select_fallback(self, mock_eps):
        mock_result = {"test.group": ["ep1"]}
        mock_eps.return_value = mock_result
        result = list(_iter_entry_points("test.group"))
        assert result == ["ep1"]

    @patch("optix.plugins.entry_points")
    def test_without_select_empty(self, mock_eps):
        mock_result = {}
        mock_eps.return_value = mock_result
        result = list(_iter_entry_points("missing.group"))
        assert result == []


class TestLoadPluginsByGroup:
    @patch("optix.plugins._iter_entry_points")
    def test_load_success(self, mock_iter):
        ep = MagicMock()
        ep.name = "plugin_a"
        ep.value = "some.module:func"
        ep.load.return_value = lambda: None
        mock_iter.return_value = [ep]
        result = load_plugins_by_group("test.group")
        assert "plugin_a" in result

    @patch("optix.plugins._iter_entry_points")
    def test_load_exception(self, mock_iter):
        ep = MagicMock()
        ep.name = "bad_plugin"
        ep.value = "bad.module:func"
        ep.load.side_effect = ImportError("no module")
        mock_iter.return_value = [ep]
        result = load_plugins_by_group("test.group")
        assert "bad_plugin" not in result

    @patch("optix.plugins._iter_entry_points")
    def test_empty_group(self, mock_iter):
        mock_iter.return_value = []
        result = load_plugins_by_group("empty.group")
        assert result == {}

    @patch("optix.plugins._iter_entry_points")
    def test_iter_exception(self, mock_iter):
        mock_iter.side_effect = Exception("fail")
        result = load_plugins_by_group("bad.group")
        assert result == {}


class TestLoadGeneralPlugins:
    def setup_method(self):
        plugins_module._PLUGINS_LOADED_FLAG = False

    @patch("optix.plugins.load_plugins_by_group")
    def test_load_once(self, mock_load):
        mock_load.return_value = {"p1": MagicMock()}
        result = load_general_plugins()
        assert result is not None
        mock_load.assert_called_once_with(group="optix.plugins")

    @patch("optix.plugins.load_plugins_by_group")
    def test_load_idempotent(self, mock_load):
        mock_load.return_value = {}
        load_general_plugins()
        result = load_general_plugins()
        assert result is None
        mock_load.assert_called_once()

    @patch("optix.plugins.load_plugins_by_group")
    def test_plugin_execution_failure(self, mock_load):
        failing_func = MagicMock(side_effect=RuntimeError("boom"))
        mock_load.return_value = {"bad": failing_func}
        result = load_general_plugins()
        assert result is not None
        failing_func.assert_called_once()
