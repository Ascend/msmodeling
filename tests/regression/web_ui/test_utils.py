"""Tests for web_ui.utils module."""

from __future__ import annotations

import pytest

from web_ui.utils import (
    bool_from_ui,
    normalize_value,
    parse_optional_number,
    parse_scalar_or_list,
    stable_hash,
)


class TestParseScalarOrList:
    """Tests for parse_scalar_or_list function."""

    def test_parse_scalar_string(self) -> None:
        """Test parsing a simple string value."""
        result = parse_scalar_or_list("test_value", str)
        assert result == ["test_value"]

    def test_parse_scalar_int(self) -> None:
        """Test parsing an integer value."""
        result = parse_scalar_or_list(42, int)
        assert result == [42]

    def test_parse_scalar_float(self) -> None:
        """Test parsing a float value."""
        result = parse_scalar_or_list(3.14, float)
        assert result == [3.14]

    def test_parse_none(self) -> None:
        """Test parsing None returns empty list."""
        result = parse_scalar_or_list(None, str)
        assert result == []

    def test_parse_empty_string(self) -> None:
        """Test parsing empty string returns empty list."""
        result = parse_scalar_or_list("", str)
        assert result == []

    def test_parse_list_of_strings(self) -> None:
        """Test parsing a list of strings."""
        result = parse_scalar_or_list(["a", "b", "c"], str)
        assert result == ["a", "b", "c"]

    def test_parse_list_of_ints(self) -> None:
        """Test parsing a list of integers."""
        result = parse_scalar_or_list([1, 2, 3], int)
        assert result == [1, 2, 3]

    def test_parse_tuple(self) -> None:
        """Test parsing a tuple."""
        result = parse_scalar_or_list((1, 2, 3), int)
        assert result == [1, 2, 3]

    def test_parse_bool(self) -> None:
        """Test parsing boolean value."""
        result = parse_scalar_or_list(True, bool)
        assert result == [True]

    def test_parse_bracket_list_string(self) -> None:
        """Test parsing a bracket-enclosed list string."""
        result = parse_scalar_or_list("[1,2,3]", int)
        assert result == [1, 2, 3]

    def test_parse_bracket_list_strings(self) -> None:
        """Test parsing bracket-enclosed string list."""
        result = parse_scalar_or_list("['a','b','c']", str)
        assert result == ["a", "b", "c"]

    def test_parse_bracket_empty_list(self) -> None:
        """Test parsing empty bracket list."""
        result = parse_scalar_or_list("[]", int)
        assert result == []

    def test_parse_bracket_list_with_spaces(self) -> None:
        """Test parsing bracket list with spaces."""
        result = parse_scalar_or_list("[ 1, 2 , 3 ]", int)
        assert result == [1, 2, 3]

    def test_parse_comma_separated_string(self) -> None:
        """Test parsing comma-separated string."""
        result = parse_scalar_or_list("a,b,c", str)
        assert result == ["a", "b", "c"]

    def test_parse_comma_separated_with_spaces(self) -> None:
        """Test parsing comma-separated string with spaces."""
        result = parse_scalar_or_list("1, 2, 3", int)
        assert result == [1, 2, 3]

    def test_parse_invalid_bracket_fallback(self) -> None:
        """Test parsing invalid bracket format treats as comma separated."""
        # Original implementation doesn't handle invalid bracket format specially
        # It just splits on comma, keeping the leading bracket
        result = parse_scalar_or_list("[a,b,c", str)
        assert result == ["[a", "b", "c"]

    def test_parse_string_with_quotes(self) -> None:
        """Test parsing strings with quotes keeps quotes."""
        # Original implementation doesn't strip quotes from comma-separated strings
        result = parse_scalar_or_list("'a','b','c'", str)
        assert result == ["'a'", "'b'", "'c'"]

    def test_parse_double_quoted_strings(self) -> None:
        """Test parsing double-quoted strings keeps quotes."""
        # Original implementation doesn't strip quotes from comma-separated strings
        result = parse_scalar_or_list('"x","y","z"', str)
        assert result == ['"x"', '"y"', '"z"']


class TestParseOptionalNumber:
    """Tests for parse_optional_number function."""

    def test_parse_none_returns_none(self) -> None:
        """Test parsing None returns None."""
        result = parse_optional_number(None, int)
        assert result is None

    def test_parse_empty_string_returns_none(self) -> None:
        """Test parsing empty string returns None."""
        result = parse_optional_number("", int)
        assert result is None

    def test_parse_none_string_returns_none(self) -> None:
        """Test parsing 'None' string returns None."""
        result = parse_optional_number("None", int)
        assert result is None

    def test_parse_none_lowercase_returns_none(self) -> None:
        """Test parsing 'none' string returns None."""
        result = parse_optional_number("none", float)
        assert result is None

    def test_parse_auto_returns_none(self) -> None:
        """Test parsing 'auto' string returns None."""
        result = parse_optional_number("auto", int)
        assert result is None

    def test_parse_auto_uppercase_returns_none(self) -> None:
        """Test parsing 'AUTO' string - original only matches lowercase 'auto'."""
        # Original implementation only checks for exact string match (case-sensitive)
        # "AUTO" doesn't match "auto" in the original, so it tries float("AUTO")
        # which raises ValueError
        with pytest.raises(ValueError):
            parse_optional_number("AUTO", float)

    def test_parse_auto_lowercase_returns_none(self) -> None:
        """Test parsing 'auto' string (lowercase) returns None."""
        result = parse_optional_number("auto", float)
        assert result is None

    def test_parse_valid_int(self) -> None:
        """Test parsing valid integer."""
        result = parse_optional_number("42", int)
        assert result == 42

    def test_parse_valid_float(self) -> None:
        """Test parsing valid float."""
        result = parse_optional_number("3.14", float)
        assert result == 3.14

    def test_parse_number_directly(self) -> None:
        """Test parsing a number value directly."""
        result = parse_optional_number(100, int)
        assert result == 100

    def test_parse_negative_number(self) -> None:
        """Test parsing negative number."""
        result = parse_optional_number("-5", int)
        assert result == -5

    def test_parse_zero(self) -> None:
        """Test parsing zero."""
        result = parse_optional_number("0", int)
        assert result == 0


class TestStableHash:
    """Tests for stable_hash function."""

    def test_hash_consistent_for_same_input(self) -> None:
        """Test hash is consistent for same input."""
        data = {"model": "test", "device": "D1"}
        h1 = stable_hash(data)
        h2 = stable_hash(data)
        assert h1 == h2
        assert len(h1) == 16

    def test_hash_differs_for_different_input(self) -> None:
        """Test hash differs for different input."""
        h1 = stable_hash({"model": "test"})
        h2 = stable_hash({"model": "different"})
        assert h1 != h2

    def test_hash_order_independent(self) -> None:
        """Test hash is independent of key order."""
        h1 = stable_hash({"a": 1, "b": 2, "c": 3})
        h2 = stable_hash({"c": 3, "b": 2, "a": 1})
        assert h1 == h2

    def test_hash_with_nested_dict(self) -> None:
        """Test hash with nested dictionary."""
        data = {"outer": {"inner": "value"}, "other": "test"}
        h = stable_hash(data)
        assert len(h) == 16
        assert isinstance(h, str)

    def test_hash_with_list_values(self) -> None:
        """Test hash with list values."""
        data = {"items": [1, 2, 3, 4]}
        h = stable_hash(data)
        assert len(h) == 16

    def test_hash_with_special_characters(self) -> None:
        """Test hash with special characters in values."""
        data = {"path": "C:\\Users\\test", "unicode": "test_unicode"}
        h = stable_hash(data)
        assert len(h) == 16

    def test_hash_with_numeric_values(self) -> None:
        """Test hash with various numeric types."""
        data = {"int": 42, "float": 3.14, "neg": -10}
        h = stable_hash(data)
        assert len(h) == 16

    def test_hash_with_boolean_values(self) -> None:
        """Test hash with boolean values."""
        data = {"flag1": True, "flag2": False}
        h = stable_hash(data)
        assert len(h) == 16

    def test_hash_with_none_value(self) -> None:
        """Test hash with None value."""
        data = {"value": None, "other": "test"}
        h = stable_hash(data)
        assert len(h) == 16


class TestBoolFromUi:
    """Tests for bool_from_ui function."""

    def test_true_boolean(self) -> None:
        """Test True boolean returns True."""
        assert bool_from_ui(True) is True

    def test_false_boolean(self) -> None:
        """Test False boolean returns False."""
        assert bool_from_ui(False) is False

    def test_string_one(self) -> None:
        """Test '1' string returns True."""
        assert bool_from_ui("1") is True

    def test_string_zero(self) -> None:
        """Test '0' string returns False."""
        assert bool_from_ui("0") is False

    def test_string_true_lowercase(self) -> None:
        """Test 'true' string returns True."""
        assert bool_from_ui("true") is True

    def test_string_true_uppercase(self) -> None:
        """Test 'TRUE' string returns True."""
        assert bool_from_ui("TRUE") is True

    def test_string_true_mixed_case(self) -> None:
        """Test 'True' string returns True."""
        assert bool_from_ui("True") is True

    def test_string_yes(self) -> None:
        """Test 'yes' string returns True."""
        assert bool_from_ui("yes") is True

    def test_string_y(self) -> None:
        """Test 'y' string returns True."""
        assert bool_from_ui("y") is True

    def test_string_on(self) -> None:
        """Test 'on' string returns True."""
        assert bool_from_ui("on") is True

    def test_string_false_returns_false(self) -> None:
        """Test 'false' string returns False."""
        assert bool_from_ui("false") is False

    def test_string_no_returns_false(self) -> None:
        """Test 'no' string returns False."""
        assert bool_from_ui("no") is False

    def test_string_off_returns_false(self) -> None:
        """Test 'off' string returns False."""
        assert bool_from_ui("off") is False

    def test_arbitrary_string_returns_false(self) -> None:
        """Test arbitrary string returns False."""
        assert bool_from_ui("random") is False

    def test_empty_string_returns_false(self) -> None:
        """Test empty string returns False."""
        assert bool_from_ui("") is False


class TestNormalizeValue:
    """Tests for normalize_value function."""

    def test_normalize_scalar(self) -> None:
        """Test normalizing a scalar value."""
        result = normalize_value(42)
        assert result == 42

    def test_normalize_string(self) -> None:
        """Test normalizing a string."""
        result = normalize_value("test")
        assert result == "test"

    def test_normalize_list(self) -> None:
        """Test normalizing a list."""
        result = normalize_value([1, 2, 3])
        assert result == [1, 2, 3]

    def test_normalize_tuple(self) -> None:
        """Test normalizing a tuple converts to list."""
        result = normalize_value((1, 2, 3))
        assert result == [1, 2, 3]

    def test_normalize_nested_list(self) -> None:
        """Test normalizing nested lists."""
        result = normalize_value([1, [2, 3], 4])
        assert result == [1, [2, 3], 4]

    def test_normalize_nested_tuples(self) -> None:
        """Test normalizing nested tuples."""
        result = normalize_value((1, (2, 3), 4))
        assert result == [1, [2, 3], 4]

    def test_normalize_dict_values(self) -> None:
        """Test normalizing preserves dict as-is."""
        result = normalize_value({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_normalize_mixed_nested(self) -> None:
        """Test normalizing mixed nested structures."""
        result = normalize_value([1, (2, [3, 4]), {"x": 5}])
        assert result == [1, [2, [3, 4]], {"x": 5}]

    def test_normalize_none(self) -> None:
        """Test normalizing None."""
        result = normalize_value(None)
        assert result is None

    def test_normalize_empty_list(self) -> None:
        """Test normalizing empty list."""
        result = normalize_value([])
        assert result == []

    def test_normalize_empty_tuple(self) -> None:
        """Test normalizing empty tuple."""
        result = normalize_value(())
        assert result == []

    def test_normalize_list_with_none(self) -> None:
        """Test normalizing list containing None."""
        result = normalize_value([1, None, 3])
        assert result == [1, None, 3]

    def test_normalize_deeply_nested(self) -> None:
        """Test normalizing deeply nested structure."""
        result = normalize_value(([((1,))],))
        assert result == [[[1]]]
