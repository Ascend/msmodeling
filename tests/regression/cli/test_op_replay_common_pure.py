"""Tests for op_replay/common.py — pure functions (no NPU needed)."""

import argparse
import unittest

from tools.perf_data_collection.op_replay.common import (
    SUPPORTED_DEVICES,
    DEFAULT_DEVICE,
    DEFAULT_REPLAY_REPEAT_COUNT,
    SUPPORTED_UPDATE_MODES,
    MICROBENCH_DURATION,
    check_version,
    normalize_device_name,
    normalize_vllm_ascend_version,
    parse_list_field,
    split_metadata_field,
    parse_shape,
    parse_shape_or_none,
    normalize_dtype_name,
    normalize_op_name,
    expand_fractal_nz_shape,
    normalize_shape,
    build_version_dir_name,
    is_version_dir_name,
    _normalize_stack_component,
    INVALID_REPLAY_ROWS,
)


class TestConstants(unittest.TestCase):
    def test_supported_devices(self):
        self.assertIn(DEFAULT_DEVICE, SUPPORTED_DEVICES)
        self.assertGreater(len(SUPPORTED_DEVICES), 3)

    def test_default_replay_repeat_count(self):
        self.assertGreater(DEFAULT_REPLAY_REPEAT_COUNT, 0)

    def test_supported_update_modes(self):
        self.assertIn("all", SUPPORTED_UPDATE_MODES)
        self.assertIn("missing-only", SUPPORTED_UPDATE_MODES)

    def test_microbench_duration_column(self):
        self.assertEqual(MICROBENCH_DURATION, "Average Duration(us)")

    def test_invalid_replay_rows_is_list(self):
        self.assertIsInstance(INVALID_REPLAY_ROWS, list)


class TestCheckVersion(unittest.TestCase):
    def test_valid_simple(self):
        self.assertEqual(check_version("0.9.2"), "0.9.2")

    def test_valid_with_v(self):
        self.assertIsNotNone(check_version("v0.13.0"))

    def test_valid_with_underscore(self):
        self.assertIsNotNone(check_version("vllm0.18.0_torch2.9.0_cann8.5"))

    def test_invalid_raises(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            check_version("bad version with spaces")


class TestNormalizeDeviceName(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(normalize_device_name("  ATLAS_800  "), "ATLAS_800")


class TestNormalizeVllmAscendVersion(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(normalize_vllm_ascend_version("  0.13.0  "), "0.13.0")


class TestNormalizeStackComponent(unittest.TestCase):
    def test_vllm_prefix(self):
        self.assertEqual(_normalize_stack_component("vllm", "vllm0.18.0"), "0.18.0")

    def test_v_prefix(self):
        self.assertEqual(_normalize_stack_component("vllm", "v0.18.0"), "0.18.0")

    def test_torch_prefix(self):
        result = _normalize_stack_component("torch", "torch2.9.0+cpu")
        self.assertIn("2.9.0", result)

    def test_cann_prefix(self):
        self.assertEqual(_normalize_stack_component("cann", "cann8.5"), "8.5")


class TestBuildVersionDirName(unittest.TestCase):
    def test_standard(self):
        result = build_version_dir_name(
            vllm_ascend_version="0.18.0",
            torch_version="2.9.0",
            cann_version="8.5",
        )
        self.assertEqual(result, "vllm0.18.0_torch2.9.0_cann8.5")

    def test_with_prefixes(self):
        result = build_version_dir_name(
            vllm_ascend_version="v0.18.0",
            torch_version="torch2.9.0",
            cann_version="cann8.5",
        )
        self.assertEqual(result, "vllm0.18.0_torch2.9.0_cann8.5")


class TestIsVersionDirName(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(is_version_dir_name("vllm0.18.0_torch2.9.0_cann8.5"))

    def test_invalid(self):
        self.assertFalse(is_version_dir_name("not_a_version"))


class TestParseListField(unittest.TestCase):
    def test_semicolon(self):
        self.assertEqual(parse_list_field("a;b;c"), ["a", "b", "c"])

    def test_quoted(self):
        self.assertEqual(parse_list_field('"a;b;c"'), ["a", "b", "c"])

    def test_empty(self):
        self.assertEqual(parse_list_field(""), [])


class TestSplitMetadataField(unittest.TestCase):
    def test_semicolon(self):
        self.assertEqual(split_metadata_field("a;b"), ["a", "b"])

    def test_quoted(self):
        self.assertEqual(split_metadata_field('"a;b"'), ["a", "b"])

    def test_empty(self):
        self.assertEqual(split_metadata_field(""), [""])


class TestParseShape(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(parse_shape("128,5120"), (128, 5120))

    def test_single_dim(self):
        self.assertEqual(parse_shape("4096"), (4096,))


class TestParseShapeOrNone(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_shape_or_none("128,5120"), (128, 5120))

    def test_empty_returns_none(self):
        self.assertIsNone(parse_shape_or_none("  "))


class TestNormalizeDtypeName(unittest.TestCase):
    def test_with_prefix(self):
        self.assertEqual(normalize_dtype_name("DT_BF16"), "DT_BF16")

    def test_without_prefix(self):
        self.assertEqual(normalize_dtype_name("BF16"), "DT_BF16")

    def test_empty_returns_undefined(self):
        self.assertEqual(normalize_dtype_name(""), "DT_UNDEFINED")


class TestNormalizeOpName(unittest.TestCase):
    def test_removes_run_py(self):
        self.assertEqual(normalize_op_name("MatMulV2_run.py"), "MatMulV2")

    def test_removes_run(self):
        self.assertEqual(normalize_op_name("PadV3_run"), "PadV3")

    def test_removes_csv(self):
        self.assertEqual(normalize_op_name("SoftmaxV2.csv"), "SoftmaxV2")

    def test_passthrough(self):
        self.assertEqual(normalize_op_name("MatMulV2"), "MatMulV2")


class TestExpandFractalNzShape(unittest.TestCase):
    def test_valid(self):
        result = expand_fractal_nz_shape((2, 3, 4, 5))
        self.assertEqual(result, (12, 10))

    def test_invalid_dims_raises(self):
        with self.assertRaises(ValueError):
            expand_fractal_nz_shape((2, 3))


class TestNormalizeShape(unittest.TestCase):
    def test_regular_passthrough(self):
        self.assertEqual(normalize_shape((128, 5120), "ND"), (128, 5120))

    def test_fractal_nz_expands(self):
        result = normalize_shape((2, 3, 4, 5), "FRACTAL_NZ")
        self.assertEqual(result, (12, 10))


if __name__ == "__main__":
    unittest.main()
