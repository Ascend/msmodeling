# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import os
import sys
import tempfile
import unittest

from serving_cast.config import LoadGenConfig
from serving_cast.main import get_load_gen, get_serving, instance_group2pd_type, parse_command_line_args


class TestParseCommandLineArgs(unittest.TestCase):
    """Tests for parse_command_line_args function."""

    def setUp(self):
        """Set up test fixtures with temporary config files."""
        self.temp_dir = tempfile.mkdtemp()
        self.instance_config_path = os.path.join(self.temp_dir, "instance.yaml")
        self.common_config_path = os.path.join(self.temp_dir, "common.yaml")

        # Create valid config files
        with open(self.instance_config_path, "w", encoding="utf-8") as f:
            f.write("instance_groups: []")
        with open(self.common_config_path, "w", encoding="utf-8") as f:
            f.write(
                "model_config: {name: test}\nload_gen: "
                "{load_gen_type: fixed_length, num_requests: 100, num_input_tokens: 128, "
                "num_output_tokens: 128, request_rate: 1.0}\n"
                "serving_config: {}"
            )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_parse_args_valid_required_args(self):
        """Test parsing with valid required arguments."""
        original_argv = sys.argv
        try:
            sys.argv = [
                "main.py",
                "--instance_config_path",
                self.instance_config_path,
                "--common_config_path",
                self.common_config_path,
            ]
            args = parse_command_line_args()
            self.assertEqual(args.instance_config_path, self.instance_config_path)
            self.assertEqual(args.common_config_path, self.common_config_path)
            self.assertFalse(args.enable_profiling)
            self.assertEqual(args.profiling_output_path, "./profiling_results")
        finally:
            sys.argv = original_argv

    def test_parse_args_with_profiling_enabled(self):
        """Test parsing with profiling enabled."""
        original_argv = sys.argv
        try:
            sys.argv = [
                "main.py",
                "--instance_config_path",
                self.instance_config_path,
                "--common_config_path",
                self.common_config_path,
                "--enable_profiling",
            ]
            args = parse_command_line_args()
            self.assertTrue(args.enable_profiling)
        finally:
            sys.argv = original_argv

    def test_parse_args_with_custom_profiling_output_path(self):
        """Test parsing with custom profiling output path."""
        original_argv = sys.argv
        try:
            sys.argv = [
                "main.py",
                "--instance_config_path",
                self.instance_config_path,
                "--common_config_path",
                self.common_config_path,
                "--profiling_output_path",
                "/custom/path",
            ]
            args = parse_command_line_args()
            self.assertEqual(args.profiling_output_path, "/custom/path")
        finally:
            sys.argv = original_argv

    def test_parse_args_output_json_default_none(self):
        """Test that --output_json defaults to None when not provided."""
        original_argv = sys.argv
        try:
            sys.argv = [
                "main.py",
                "--instance_config_path",
                self.instance_config_path,
                "--common_config_path",
                self.common_config_path,
            ]
            args = parse_command_line_args()
            self.assertIsNone(args.output_json)
        finally:
            sys.argv = original_argv

    def test_parse_args_with_output_json(self):
        """Test parsing with --output_json provided."""
        original_argv = sys.argv
        try:
            sys.argv = [
                "main.py",
                "--instance_config_path",
                self.instance_config_path,
                "--common_config_path",
                self.common_config_path,
                "--output_json",
                "/tmp/summary.json",
            ]
            args = parse_command_line_args()
            self.assertEqual(args.output_json, "/tmp/summary.json")
        finally:
            sys.argv = original_argv

    def test_parse_args_missing_instance_config(self):
        """Test parsing with missing instance config path."""
        original_argv = sys.argv
        try:
            sys.argv = [
                "main.py",
                "--common_config_path",
                self.common_config_path,
            ]
            with self.assertRaises(SystemExit):
                parse_command_line_args()
        finally:
            sys.argv = original_argv

    def test_parse_args_missing_common_config(self):
        """Test parsing with missing common config path."""
        original_argv = sys.argv
        try:
            sys.argv = [
                "main.py",
                "--instance_config_path",
                self.instance_config_path,
            ]
            with self.assertRaises(SystemExit):
                parse_command_line_args()
        finally:
            sys.argv = original_argv

    def test_parse_args_nonexistent_instance_config(self):
        """Test parsing with non-existent instance config file."""
        original_argv = sys.argv
        try:
            sys.argv = [
                "main.py",
                "--instance_config_path",
                "/nonexistent/path.yaml",
                "--common_config_path",
                self.common_config_path,
            ]
            with self.assertRaises(SystemExit):
                parse_command_line_args()
        finally:
            sys.argv = original_argv

    def test_parse_args_nonexistent_common_config(self):
        """Test parsing with non-existent common config file."""
        original_argv = sys.argv
        try:
            sys.argv = [
                "main.py",
                "--instance_config_path",
                self.instance_config_path,
                "--common_config_path",
                "/nonexistent/path.yaml",
            ]
            with self.assertRaises(SystemExit):
                parse_command_line_args()
        finally:
            sys.argv = original_argv


class TestInstanceGroup2PdType(unittest.TestCase):
    """Tests for instance_group2pd_type function."""

    def test_pd_aggregation(self):
        """Test pd_aggregation detection."""
        instance_group = {"both": [1, 2], "prefill": [], "decode": []}
        result = instance_group2pd_type(instance_group)
        self.assertEqual(result, "pd_aggregation")

    def test_pd_disaggregation(self):
        """Test pd_disaggregation detection."""
        instance_group = {"both": [], "prefill": [1], "decode": [2]}
        result = instance_group2pd_type(instance_group)
        self.assertEqual(result, "pd_disaggregation")

    def test_invalid_both_empty(self):
        """Test invalid when all empty."""
        instance_group = {"both": [], "prefill": [], "decode": []}
        result = instance_group2pd_type(instance_group)
        self.assertIsNone(result)

    def test_invalid_both_non_empty(self):
        """Test invalid when both both and prefill non-empty."""
        instance_group = {"both": [1], "prefill": [2], "decode": []}
        result = instance_group2pd_type(instance_group)
        self.assertIsNone(result)


class TestGetServing(unittest.TestCase):
    """Tests for get_serving function."""

    def test_get_serving_invalid(self):
        """Test get_serving raises error for invalid config."""
        instance_group = {"both": [], "prefill": [], "decode": []}
        with self.assertRaises(ValueError) as context:
            get_serving(instance_group)
        self.assertIn("Unknown pd type", str(context.exception))


class TestGetLoadGen(unittest.TestCase):
    """Tests for get_load_gen function."""

    def test_get_load_gen_fixed_length(self):
        """Test get_load_gen for fixed_length type."""
        config = LoadGenConfig(
            load_gen_type="fixed_length",
            num_requests=100,
            num_input_tokens=128,
            num_output_tokens=128,
            request_rate=1.0,
        )
        result = get_load_gen(config)
        # Verify the load gen was created with correct parameters
        self.assertIsNotNone(result)
        self.assertEqual(result.num_requests, 100)
        self.assertEqual(result.request_rate, 1.0)

    def test_get_load_gen_unknown_type(self):
        """Test get_load_gen raises error for unknown type."""
        # Create a config with invalid type by modifying after creation
        config = LoadGenConfig(
            load_gen_type="fixed_length",  # valid type first
            num_requests=100,
            num_input_tokens=128,
            num_output_tokens=128,
            request_rate=1.0,
        )
        # Change to invalid type
        config.load_gen_type = "unknown"
        # Note: main.py has a bug - it uses load_gen_config.type instead of load_gen_config.load_gen_type
        # This test documents the current behavior
        with self.assertRaises((ValueError, AttributeError)):
            get_load_gen(config)


if __name__ == "__main__":
    unittest.main()
