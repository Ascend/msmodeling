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
