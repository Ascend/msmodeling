# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
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

"""Coverage for ``render_hardware_profile_comparison`` (serving_cast UT suite)."""

import builtins
import importlib.util
import unittest
from unittest import TestCase
from unittest.mock import patch

from serving_cast.service import optimizer_summary as osum


def _torch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


class TestRenderHardwareProfileComparison(TestCase):
    def test_torch_import_error_returns_empty_string(self):
        real_import = builtins.__import__

        def _selective_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                raise ImportError("torch unavailable in ut")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=_selective_import):
            with self.assertLogs(osum.logger, level="WARNING") as log_ctx:
                txt = osum.render_hardware_profile_comparison(["ANY"])
        self.assertEqual(txt, "")
        self.assertTrue(any("skipped" in m.lower() or "import" in m.lower() for m in log_ctx.output))

    @unittest.skipUnless(_torch_installed(), "torch not installed")
    def test_unknown_device_prints_placeholder_row(self):
        txt = osum.render_hardware_profile_comparison(["__no_such_profile_for_ut__"])
        self.assertIn("__no_such_profile_for_ut__", txt)
        self.assertIn("-", txt)
