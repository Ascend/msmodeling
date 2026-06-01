# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from typing import Optional

from serving_cast.request import Request, RequestState
from serving_cast.utils import dataclass2dict, gen_profiling_config_set_env_variable, get_basic_timestamp, summarize


@dataclass
class NestedDataclass:
    value: int


@dataclass
class SampleDataclass:
    name: str
    count: int
    nested: Optional[NestedDataclass] = None


@dataclass
class ComplexDataclass:
    items: list[int]
    nested_list: list[NestedDataclass]
    dict_field: dict


class TestDataclass2Dict(unittest.TestCase):
    def test_simple_dataclass(self):
        """Test converting a simple dataclass to dict."""
        obj = SampleDataclass(name="test", count=42)
        result = dataclass2dict(obj)
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["count"], 42)
        self.assertIsNone(result["nested"])

    def test_nested_dataclass(self):
        """Test converting a dataclass with nested dataclass."""
        nested = NestedDataclass(value=100)
        obj = SampleDataclass(name="test", count=42, nested=nested)
        result = dataclass2dict(obj)
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["count"], 42)
        self.assertEqual(result["nested"]["value"], 100)

    def test_skip_none_true(self):
        """Test skip_none=True removes None fields."""
        obj = SampleDataclass(name="test", count=42, nested=None)
        result = dataclass2dict(obj, skip_none=True)
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["count"], 42)
        self.assertNotIn("nested", result)

    def test_skip_none_false(self):
        """Test skip_none=False keeps None fields."""
        obj = SampleDataclass(name="test", count=42, nested=None)
        result = dataclass2dict(obj, skip_none=False)
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["count"], 42)
        self.assertIn("nested", result)
        self.assertIsNone(result["nested"])

    def test_list_of_dataclasses(self):
        """Test converting dataclass with list of dataclasses."""
        obj = ComplexDataclass(
            items=[1, 2, 3],
            nested_list=[NestedDataclass(value=i) for i in range(3)],
            dict_field={"key": "value"},
        )
        result = dataclass2dict(obj)
        self.assertEqual(result["items"], [1, 2, 3])
        self.assertEqual(len(result["nested_list"]), 3)
        for i, item in enumerate(result["nested_list"]):
            self.assertEqual(item["value"], i)
        self.assertEqual(result["dict_field"], {"key": "value"})

    def test_non_dataclass_raises_error(self):
        """Test that non-dataclass raises TypeError."""
        with self.assertRaises(TypeError):
            dataclass2dict({"not": "a dataclass"})

        with self.assertRaises(TypeError):
            dataclass2dict([1, 2, 3])

        with self.assertRaises(TypeError):
            dataclass2dict("string")


class TestGetBasicTimestamp(unittest.TestCase):
    def test_timestamp_format(self):
        """Test that timestamp has correct format."""
        timestamp = get_basic_timestamp()
        # Format: YYYY-MM-DD_HH-MM-SS
        parts = timestamp.split("_")
        self.assertEqual(len(parts), 2)
        date_part, time_part = parts
        self.assertEqual(len(date_part.split("-")), 3)
        self.assertEqual(len(time_part.split("-")), 3)

    def test_timestamp_is_string(self):
        """Test that timestamp is a string."""
        timestamp = get_basic_timestamp()
        self.assertIsInstance(timestamp, str)

    def test_timestamp_not_empty(self):
        """Test that timestamp is not empty."""
        timestamp = get_basic_timestamp()
        self.assertTrue(len(timestamp) > 0)


class TestGenProfilingConfigSetEnvVariable(unittest.TestCase):
    def test_creates_config_file(self):
        """Test that config file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen_profiling_config_set_env_variable(tmpdir)
            config_path = os.path.join(tmpdir, "profiling_config.json")
            self.assertTrue(os.path.exists(config_path))

    def test_config_content(self):
        """Test that config file has correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen_profiling_config_set_env_variable(tmpdir)
            config_path = os.path.join(tmpdir, "profiling_config.json")
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            self.assertEqual(config["enable"], 1)
            self.assertEqual(config["prof_dir"], tmpdir)
            self.assertEqual(config["profiler_level"], "INFO")

    def test_env_variable_set(self):
        """Test that environment variable is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen_profiling_config_set_env_variable(tmpdir)
            config_path = os.path.join(tmpdir, "profiling_config.json")
            self.assertEqual(os.environ.get("SERVICE_PROF_CONFIG_PATH"), config_path)


class TestSummarize(unittest.TestCase):
    def test_summarize_basic(self):
        """Test summarize with basic request data."""
        # Create requests with time data
        request1 = Request(num_input_tokens=100, num_output_tokens=50)
        request1.leaves_client_time = 0.0
        request1.arrives_server_time = 0.1
        request1.prefill_done_time = 1.0
        request1.decode_done_time = 5.0
        request1._state = RequestState.DECODE_DONE

        request2 = Request(num_input_tokens=200, num_output_tokens=100)
        request2.leaves_client_time = 0.5
        request2.arrives_server_time = 0.6
        request2.prefill_done_time = 2.0
        request2.decode_done_time = 10.0
        request2._state = RequestState.DECODE_DONE

        # summarize prints output, just verify it doesn't raise
        summarize([request1, request2])

    def test_summarize_single_request(self):
        """Test summarize with single request."""
        request = Request(num_input_tokens=100, num_output_tokens=10)
        request.leaves_client_time = 0.0
        request.arrives_server_time = 0.0
        request.prefill_done_time = 1.0
        request.decode_done_time = 2.0
        request._state = RequestState.DECODE_DONE

        summarize([request])


if __name__ == "__main__":
    unittest.main()
