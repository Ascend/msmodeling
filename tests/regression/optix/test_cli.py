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
import sys
from unittest.mock import patch


from cli.main import main


class TestCli:
    @patch("optix.optimizer.optimizer.main")
    def test_optix_command(self, mock_optix_main):
        with patch.object(sys, "argv", ["msmodeling", "optix"]):
            main()
            mock_optix_main.assert_called_once()

    def test_no_command_prints_help(self, capsys):
        with patch.object(sys, "argv", ["msmodeling"]):
            main()
        captured = capsys.readouterr()
        assert "msmodeling" in captured.out or "usage" in captured.out.lower()

    @patch("optix.optimizer.optimizer.main")
    def test_optix_passes_remaining_args(self, mock_optix_main):
        with patch.object(sys, "argv", ["msmodeling", "optix", "--some-arg", "value"]):
            main()
            assert sys.argv == ["msmodeling", "--some-arg", "value"]

    def test_unknown_command_prints_help(self, capsys):
        with patch.object(sys, "argv", ["msmodeling", "--unknown"]):
            main()
        captured = capsys.readouterr()
        assert "msmodeling" in captured.out or "commands" in captured.out.lower()
