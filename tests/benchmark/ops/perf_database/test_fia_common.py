"""Tests for fia_common.py — shared FIA parsing utilities."""

import pytest

from tools.perf_data_collection.fia_common import (
    parse_runtime_int,
    parse_runtime_int_list,
    parse_shape_or_none,
    shape_numel,
    shape_to_text,
    split_metadata_field,
)


class TestSplitMetadataField:
    def test_semicolon_separated(self):
        assert split_metadata_field("a;b;c") == ["a", "b", "c"]

    def test_quoted(self):
        assert split_metadata_field('"a;b;c"') == ["a", "b", "c"]

    def test_whitespace_trim(self):
        assert split_metadata_field("  a ; b  ; c ") == ["a", "b", "c"]

    def test_single_value(self):
        assert split_metadata_field("hello") == ["hello"]

    def test_empty_string(self):
        assert split_metadata_field("") == [""]

    def test_none_returns_empty_string_list(self):
        assert split_metadata_field(None) == [""]


class TestParseShapeOrNone:
    def test_valid_shape(self):
        assert parse_shape_or_none("128,5120") == (128, 5120)

    def test_single_dim(self):
        assert parse_shape_or_none("4096") == (4096,)

    def test_empty_string(self):
        assert parse_shape_or_none("") is None

    def test_none(self):
        assert parse_shape_or_none(None) is None

    def test_whitespace(self):
        assert parse_shape_or_none("  128 , 5120  ") == (128, 5120)

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            parse_shape_or_none("128,abc")


class TestParseRuntimeInt:
    def test_valid(self):
        assert parse_runtime_int("42") == 42

    def test_negative(self):
        assert parse_runtime_int("-1") == -1

    def test_zero(self):
        assert parse_runtime_int("0") == 0

    def test_empty(self):
        assert parse_runtime_int("") is None

    def test_none(self):
        assert parse_runtime_int(None) is None

    def test_whitespace(self):
        assert parse_runtime_int("  99  ") == 99

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_runtime_int("not_a_number")


class TestParseRuntimeIntList:
    def test_csv_separated(self):
        assert parse_runtime_int_list("1,2,3") == [1, 2, 3]

    def test_semicolon_separated(self):
        assert parse_runtime_int_list("1;2;3") == [1, 2, 3]

    def test_mixed_separators(self):
        assert parse_runtime_int_list("1,2;3") == [1, 2, 3]

    def test_empty(self):
        assert parse_runtime_int_list("") is None

    def test_none(self):
        assert parse_runtime_int_list(None) is None

    def test_single_value(self):
        assert parse_runtime_int_list("42") == [42]

    def test_whitespace(self):
        assert parse_runtime_int_list("  1 , 2 , 3  ") == [1, 2, 3]

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            parse_runtime_int_list("1,abc,3")


class TestShapeNumel:
    def test_basic(self):
        assert shape_numel((2, 3, 4)) == 24

    def test_single_dim(self):
        assert shape_numel((1024,)) == 1024

    def test_empty_tuple(self):
        assert shape_numel(()) == 0

    def test_none(self):
        assert shape_numel(None) == 0

    def test_large_shape(self):
        assert shape_numel((4096, 8192)) == 4096 * 8192


class TestShapeToText:
    def test_basic(self):
        assert shape_to_text([128, 5120]) == "128,5120"

    def test_single_dim(self):
        assert shape_to_text([1]) == "1"

    def test_none(self):
        assert shape_to_text(None) == ""

    def test_empty_list(self):
        assert shape_to_text([]) == ""

    def test_tuple_input(self):
        assert shape_to_text((64, 128)) == "64,128"

    def test_floats_cast_to_int(self):
        assert shape_to_text([1.0, 2.0]) == "1,2"
