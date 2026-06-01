# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import argparse
import unittest

from serving_cast.service.utils import (
    BatchRangeAction,
    OptimizerData,
    check_positive_float,
    check_positive_integer,
    check_string_valid,
)


class TestServiceUtils(unittest.TestCase):
    def test_check_string_valid_within_limit_and_valid_chars(self):
        """Test check_string_valid with valid string"""
        valid_string = "valid_string123/test-path.file"
        result = check_string_valid(valid_string, max_len=100)
        self.assertEqual(result, valid_string)

    def test_check_positive_integer_valid(self):
        """Test check_positive_integer with valid integers"""
        self.assertEqual(check_positive_integer("1"), 1)
        self.assertEqual(check_positive_integer("100"), 100)
        self.assertEqual(check_positive_integer(5), 5)

    def test_check_positive_integer_invalid_string(self):
        """Test check_positive_integer with invalid string"""
        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_integer("abc")

    def test_check_positive_integer_non_positive(self):
        """Test check_positive_integer with non-positive values"""
        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_integer("0")

        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_integer("-1")

    def test_check_positive_integer_too_large(self):
        """Test check_positive_integer with very large value"""
        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_integer("2000000")  # Greater than 1e6

    def test_check_positive_float_valid(self):
        """Test check_positive_float with valid floats"""
        self.assertEqual(check_positive_float("1.5"), 1.5)
        self.assertEqual(check_positive_float("100"), 100.0)
        self.assertEqual(check_positive_float("inf"), float("inf"))
        self.assertEqual(check_positive_float("INF"), float("inf"))

    def test_check_positive_float_invalid(self):
        """Test check_positive_float with invalid values"""
        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_float("abc")

        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_float("0")

        with self.assertRaises(argparse.ArgumentTypeError):
            check_positive_float("-1.5")

    def test_optimizer_data_creation(self):
        """Test OptimizerData creation with default values"""
        config = OptimizerData()
        self.assertIsNone(config.input_length)
        self.assertIsNone(config.output_length)
        self.assertEqual(config.prefix_cache_hit_rate, 0.0)

    def test_optimizer_data_effective_input_length_with_prefix_cache(self):
        config = OptimizerData(input_length=200, prefix_cache_hit_rate=0.5)
        self.assertEqual(config.get_effective_input_length(), 100)

    def test_optimizer_data_effective_input_length_ignores_prefix_cache_in_decode(self):
        config = OptimizerData(input_length=200, prefix_cache_hit_rate=0.5)
        self.assertEqual(config.get_effective_input_length(is_decode=True), 200)


class TestBatchRangeAction(unittest.TestCase):
    """Test BatchRangeAction class functionality"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.parser = argparse.ArgumentParser()
        self.namespace = argparse.Namespace()
        self.action = BatchRangeAction(option_strings=["--batch-range"], dest="batch_range")

    def test_valid_single_value(self):
        """Test BatchRangeAction with valid single value"""
        parser = argparse.ArgumentParser()
        namespace = argparse.Namespace()

        # Test single value (e.g., --batch-range 100)
        self.action(parser, namespace, [100])
        self.assertEqual(namespace.batch_range, [100])

    def test_valid_range_values(self):
        """Test BatchRangeAction with valid range values"""
        parser = argparse.ArgumentParser()
        namespace = argparse.Namespace()

        # Test range values (e.g., --batch-range 10 100)
        self.action(parser, namespace, [10, 100])
        self.assertEqual(namespace.batch_range, [10, 100])

    def test_invalid_range_order(self):
        """Test BatchRangeAction with invalid range order"""
        parser = argparse.ArgumentParser()
        namespace = argparse.Namespace()

        # Test with min > max (should raise ArgumentTypeError)
        with self.assertRaises(argparse.ArgumentTypeError):
            self.action(parser, namespace, [100, 10])
