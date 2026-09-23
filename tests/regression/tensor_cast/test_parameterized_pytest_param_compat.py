# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Verify parameterized + pytest markers for unittest TestCase."""

import unittest

import pytest
from parameterized import parameterized

# parameterized.expand + pytest.param(marks=...) is incompatible with unittest
# (pytest passes mark node ids as extra positional args). Use method/class-level
# @pytest.mark.nightly + dedicated nightly TestCase instead (see T10b fallback path).


class TestCompatMethodMark(unittest.TestCase):
    @pytest.mark.nightly
    @parameterized.expand([["marked_compile"]])
    def test_method_level_nightly(self, name):
        self.assertIsNotNone(name)


@pytest.mark.nightly
class TestCompatClassMark(unittest.TestCase):
    @parameterized.expand([["class_marked"]])
    def test_class_level_nightly(self, name):
        self.assertIsNotNone(name)
