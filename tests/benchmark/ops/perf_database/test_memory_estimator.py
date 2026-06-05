"""Tests for memory_estimator.py — HBM memory estimation utilities."""

from tools.perf_data_collection.memory_estimator import (
    DTYPE_BYTES,
    DEFAULT_BYTES_PER_ELEMENT,
    DEFAULT_MAX_BYTES,
    dtype_to_bytes,
    estimate_row_memory,
    estimate_tensor_bytes,
    exceeds_memory_budget,
    format_bytes,
    parse_dtype_from_template_row,
    _parse_dtype_list,
)


class TestDtypeToBytes:
    def test_known_dtype_with_prefix(self):
        assert dtype_to_bytes("DT_FLOAT") == 4
        assert dtype_to_bytes("DT_FLOAT16") == 2
        assert dtype_to_bytes("DT_BF16") == 2
        assert dtype_to_bytes("DT_BFLOAT16") == 2
        assert dtype_to_bytes("DT_FLOAT32") == 4
        assert dtype_to_bytes("DT_FLOAT64") == 8
        assert dtype_to_bytes("DT_INT8") == 1
        assert dtype_to_bytes("DT_INT16") == 2
        assert dtype_to_bytes("DT_INT32") == 4
        assert dtype_to_bytes("DT_INT64") == 8
        assert dtype_to_bytes("DT_UINT8") == 1
        assert dtype_to_bytes("DT_UINT16") == 2
        assert dtype_to_bytes("DT_UINT32") == 4
        assert dtype_to_bytes("DT_UINT64") == 8
        assert dtype_to_bytes("DT_BOOL") == 1
        assert dtype_to_bytes("DT_COMPLEX64") == 8
        assert dtype_to_bytes("DT_COMPLEX128") == 16
        assert dtype_to_bytes("DT_FLOAT8_E4M3") == 1
        assert dtype_to_bytes("DT_FLOAT8_E5M2") == 1
        assert dtype_to_bytes("DT_FLOAT8") == 1

    def test_known_dtype_without_prefix(self):
        assert dtype_to_bytes("FLOAT") == 4
        assert dtype_to_bytes("INT8") == 1
        assert dtype_to_bytes("INT16") == 2
        assert dtype_to_bytes("INT32") == 4
        assert dtype_to_bytes("INT64") == 8
        assert dtype_to_bytes("UINT8") == 1
        assert dtype_to_bytes("BOOL") == 1

    def test_case_insensitive(self):
        assert dtype_to_bytes("dt_float16") == 2
        assert dtype_to_bytes("Dt_Int32") == 4
        assert dtype_to_bytes("dt_bfloat16") == 2

    def test_whitespace_handling(self):
        assert dtype_to_bytes("  DT_FLOAT  ") == 4
        assert dtype_to_bytes("DT_BF16\n") == 2

    def test_unknown_dtype_falls_back_to_fp16(self):
        assert dtype_to_bytes("UNKNOWN_TYPE") == DEFAULT_BYTES_PER_ELEMENT
        assert dtype_to_bytes("") == DEFAULT_BYTES_PER_ELEMENT

    def test_all_known_dtypes_have_valid_size(self):
        for name, size in DTYPE_BYTES.items():
            assert isinstance(size, int) and size > 0, f"Bad size for {name}: {size}"


class TestParseDtypeList:
    def test_semicolon_separated(self):
        assert _parse_dtype_list("DT_BF16;DT_BF16;DT_INT32") == [
            "DT_BF16",
            "DT_BF16",
            "DT_INT32",
        ]

    def test_space_separated(self):
        assert _parse_dtype_list("DT_BF16 DT_BF16  DT_INT32") == [
            "DT_BF16",
            "DT_BF16",
            "DT_INT32",
        ]

    def test_single_dtype(self):
        assert _parse_dtype_list("DT_FLOAT16") == ["DT_FLOAT16"]

    def test_empty_string(self):
        assert _parse_dtype_list("") == []

    def test_quoted_string(self):
        assert _parse_dtype_list('"DT_BF16;DT_INT32"') == ["DT_BF16", "DT_INT32"]

    def test_none_coerced_to_empty_string_list(self):
        result = _parse_dtype_list(None)
        assert result == []


class TestEstimateTensorBytes:
    def test_basic(self):
        assert estimate_tensor_bytes((128, 5120), 2) == 128 * 5120 * 2

    def test_empty_shape(self):
        assert estimate_tensor_bytes((), 4) == 0

    def test_single_dim(self):
        assert estimate_tensor_bytes((1024,), 4) == 1024 * 4

    def test_large_shape(self):
        assert estimate_tensor_bytes((4096, 8192), 1) == 4096 * 8192

    def test_zero_in_shape(self):
        assert estimate_tensor_bytes((0, 100), 2) == 0


class TestEstimateRowMemory:
    def test_basic_two_inputs(self):
        total = estimate_row_memory(
            input_shapes=[(128, 5120), (5120, 768)],
            output_shapes=[(128, 768)],
            input_dtypes=["DT_BF16", "DT_BF16"],
            output_dtypes=["DT_BF16"],
        )
        expected = 128 * 5120 * 2 + 5120 * 768 * 2 + 128 * 768 * 2
        assert total == expected

    def test_output_dtypes_default_to_input_dtypes(self):
        total = estimate_row_memory(
            input_shapes=[(100, 200)],
            output_shapes=[(100, 300)],
            input_dtypes=["DT_FLOAT32"],
            output_dtypes=None,
        )
        expected = 100 * 200 * 4 + 100 * 300 * 4
        assert total == expected

    def test_missing_dtype_falls_back(self):
        total = estimate_row_memory(
            input_shapes=[(10, 10), (10, 20)],
            output_shapes=[(10, 20)],
            input_dtypes=["DT_BF16"],
            output_dtypes=["DT_BF16"],
        )
        expected = 10 * 10 * 2 + 10 * 20 * 2 + 10 * 20 * 2
        assert total == expected

    def test_fp8_estimation(self):
        total = estimate_row_memory(
            input_shapes=[(4096, 8192)],
            output_shapes=[(4096, 4096)],
            input_dtypes=["DT_FLOAT8_E4M3"],
            output_dtypes=["DT_FLOAT8_E4M3"],
        )
        expected = 4096 * 8192 * 1 + 4096 * 4096 * 1
        assert total == expected

    def test_empty_inputs(self):
        total = estimate_row_memory(
            input_shapes=[],
            output_shapes=[(100, 200)],
            input_dtypes=[],
            output_dtypes=["DT_BF16"],
        )
        assert total == 100 * 200 * 2


class TestExceedsMemoryBudget:
    def test_under_budget(self):
        exceeded, est = exceeds_memory_budget(
            input_shapes=[(128, 5120)],
            output_shapes=[(128, 768)],
            input_dtypes=["DT_BF16"],
            output_dtypes=["DT_BF16"],
        )
        assert not exceeded
        assert est < DEFAULT_MAX_BYTES

    def test_over_budget(self):
        exceeded, est = exceeds_memory_budget(
            input_shapes=[(1 << 20, 4096), (1 << 20, 4096)],
            output_shapes=[(1 << 20, 4096)],
            input_dtypes=["DT_FLOAT32", "DT_FLOAT32"],
            output_dtypes=["DT_FLOAT32"],
        )
        assert exceeded
        assert est > DEFAULT_MAX_BYTES

    def test_custom_budget(self):
        tiny_budget = 1000
        exceeded, est = exceeds_memory_budget(
            input_shapes=[(100, 100)],
            output_shapes=[],
            input_dtypes=["DT_FLOAT32"],
            max_bytes=tiny_budget,
        )
        assert exceeded
        assert est > tiny_budget

    def test_exact_budget_boundary(self):
        budget = 100 * 200 * 2
        exceeded, _ = exceeds_memory_budget(
            input_shapes=[(100, 200)],
            output_shapes=[],
            input_dtypes=["DT_BF16"],
            max_bytes=budget,
        )
        assert not exceeded


class TestFormatBytes:
    def test_bytes(self):
        assert format_bytes(500) == "500 B"

    def test_kib(self):
        assert format_bytes(2048) == "2.00 KiB"

    def test_mib(self):
        assert "MiB" in format_bytes(5 * 1024 * 1024)

    def test_gib(self):
        assert "GiB" in format_bytes(3 * 1024**3)

    def test_zero(self):
        assert format_bytes(0) == "0 B"

    def test_gib_precise(self):
        assert format_bytes(2 * 1024**3) == "2.00 GiB"


class TestParseDtypeFromTemplateRow:
    def test_basic(self):
        row = {
            "Input Data Types": '"DT_BF16;DT_BF16;INT32"',
            "Output Data Types": '"DT_BF16;DT_BF16;DT_BF16"',
        }
        inputs, outputs = parse_dtype_from_template_row(row)
        assert inputs == ["DT_BF16", "DT_BF16", "INT32"]
        assert outputs == ["DT_BF16", "DT_BF16", "DT_BF16"]

    def test_empty_row(self):
        row = {}
        inputs, outputs = parse_dtype_from_template_row(row)
        assert inputs == []
        assert outputs == []
