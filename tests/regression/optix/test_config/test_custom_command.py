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
from unittest.mock import patch
import pytest

from optix.config.custom_command import VllmCommand


class MockVllmCommandConfig:
    pass


class TestVllmCommand:
    @patch("shutil.which")
    def test_init_success(self, mock_which):
        """Test successful initialization when vllm is found in PATH"""
        # Setup
        mock_which.return_value = "/usr/bin/vllm"
        config = MockVllmCommandConfig()

        # Execute
        command = VllmCommand(config)

        # Verify
        assert command.process == "/usr/bin/vllm"
        assert command.command_config == config

    @patch("shutil.which")
    def test_init_failure_vllm_not_found(self, mock_which):
        """Test initialization fails when vllm is not found in PATH"""
        # Setup
        mock_which.return_value = None
        config = MockVllmCommandConfig()

        # Execute & Verify
        with pytest.raises(ValueError) as excinfo:
            VllmCommand(config)

        assert "Error: The 'vllm' executable was not found in the system PATH." in str(excinfo.value)
