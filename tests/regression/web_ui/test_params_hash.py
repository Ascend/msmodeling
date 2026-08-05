"""Real unit tests for services/params_hash.py module.

Tests params hash computation using real module imports.
"""

from __future__ import annotations

from enum import Enum

from services.params_hash import _normalize, compute_params_hash


class TestNormalizeString:
    """Tests for _normalize with string values."""

    def test_normalize_string(self):
        """String values pass through unchanged."""
        result = _normalize("hello")
        assert result == "hello"

    def test_normalize_string_unicode(self):
        """Unicode string values pass through unchanged."""
        result = _normalize("café")
        assert result == "café"

    def test_normalize_string_empty(self):
        """Empty string passes through unchanged."""
        result = _normalize("")
        assert result == ""


class TestNormalizeInt:
    """Tests for _normalize with integer values."""

    def test_normalize_int(self):
        """Integer values pass through unchanged."""
        result = _normalize(42)
        assert result == 42

    def test_normalize_int_zero(self):
        """Zero integer passes through unchanged."""
        result = _normalize(0)
        assert result == 0

    def test_normalize_int_negative(self):
        """Negative integer passes through unchanged."""
        result = _normalize(-10)
        assert result == -10

    def test_normalize_int_large(self):
        """Large integer passes through unchanged."""
        result = _normalize(9999999999)
        assert result == 9999999999


class TestNormalizeFloat:
    """Tests for _normalize with float values."""

    def test_normalize_float(self):
        """Float values pass through unchanged."""
        result = _normalize(3.14)
        assert result == 3.14

    def test_normalize_float_zero(self):
        """Zero float passes through unchanged."""
        result = _normalize(0.0)
        assert result == 0.0

    def test_normalize_float_negative(self):
        """Negative float passes through unchanged."""
        result = _normalize(-2.5)
        assert result == -2.5

    def test_normalize_float_precision(self):
        """Float precision is preserved."""
        result = _normalize(0.123456789)
        assert result == 0.123456789


class TestNormalizeBool:
    """Tests for _normalize with boolean values."""

    def test_normalize_bool_true(self):
        """True boolean passes through unchanged."""
        result = _normalize(True)
        assert result is True

    def test_normalize_bool_false(self):
        """False boolean passes through unchanged."""
        result = _normalize(False)
        assert result is False


class TestNormalizeNone:
    """Tests for _normalize with None values."""

    def test_normalize_none(self):
        """None passes through unchanged."""
        result = _normalize(None)
        assert result is None


class TestNormalizeDict:
    """Tests for _normalize with dict values."""

    def test_normalize_dict(self):
        """Dict values are recursively normalized."""
        result = _normalize({"a": 1, "b": "test"})
        assert result == {"a": 1, "b": "test"}

    def test_normalize_nested_dict(self):
        """Nested dict values are recursively normalized."""
        result = _normalize({"outer": {"inner": "value"}})
        assert result == {"outer": {"inner": "value"}}

    def test_normalize_dict_deeply_nested(self):
        """Deeply nested dict values are recursively normalized."""
        result = _normalize({"a": {"b": {"c": {"d": "deep"}}}})
        assert result == {"a": {"b": {"c": {"d": "deep"}}}}

    def test_normalize_dict_empty(self):
        """Empty dict passes through unchanged."""
        result = _normalize({})
        assert result == {}


class TestNormalizeList:
    """Tests for _normalize with list values."""

    def test_normalize_list(self):
        """List values are recursively normalized."""
        result = _normalize([1, 2, 3])
        assert result == [1, 2, 3]

    def test_normalize_nested_list(self):
        """Nested list values are recursively normalized."""
        result = _normalize([1, [2, 3], 4])
        assert result == [1, [2, 3], 4]

    def test_normalize_list_deeply_nested(self):
        """Deeply nested list values are recursively normalized."""
        result = _normalize([1, [2, [3, [4]]]])
        assert result == [1, [2, [3, [4]]]]

    def test_normalize_list_empty(self):
        """Empty list passes through unchanged."""
        result = _normalize([])
        assert result == []

    def test_normalize_list_of_dicts(self):
        """List of dicts is recursively normalized."""
        result = _normalize([{"a": 1}, {"b": 2}])
        assert result == [{"a": 1}, {"b": 2}]


class TestNormalizeMixed:
    """Tests for _normalize with mixed/nested structures."""

    def test_normalize_mixed(self):
        """Mixed types are recursively normalized."""
        result = _normalize({"a": [1, "test", {"b": 2}]})
        assert result == {"a": [1, "test", {"b": 2}]}

    def test_normalize_complex_mixed(self):
        """Complex nested mixed types are recursively normalized."""
        result = _normalize(
            {
                "numbers": [1, 2.5, -3],
                "strings": ["a", "b"],
                "nested": {"inner": [True, None, {"deep": "value"}]},
                "empty_list": [],
                "empty_dict": {},
            }
        )
        assert result == {
            "numbers": [1, 2.5, -3],
            "strings": ["a", "b"],
            "nested": {"inner": [True, None, {"deep": "value"}]},
            "empty_list": [],
            "empty_dict": {},
        }


class TestNormalizeEnum:
    """Tests for _normalize with Enum values."""

    def test_normalize_enum(self):
        """Enum values are normalized to their value."""

        class TestEnum(Enum):
            A = "value_a"
            B = "value_b"

        result = _normalize(TestEnum.A)
        assert result == "value_a"

    def test_normalize_enum_int_value(self):
        """Enum with integer values are normalized to their value."""

        class IntEnum(Enum):
            ONE = 1
            TWO = 2

        result = _normalize(IntEnum.ONE)
        assert result == 1

    def test_normalize_enum_in_dict(self):
        """Enum values in dict are normalized."""

        class TestEnum(Enum):
            X = "enum_x"

        result = _normalize({"key": TestEnum.X})
        assert result == {"key": "enum_x"}

    def test_normalize_enum_in_list(self):
        """Enum values in list are normalized."""

        class TestEnum(Enum):
            Y = "enum_y"

        result = _normalize([TestEnum.Y])
        assert result == ["enum_y"]


class TestComputeParamsHash:
    """Tests for compute_params_hash function."""

    def test_simple_params(self):
        """Simple params produce a hash."""
        result = compute_params_hash("test_module", "1.0.0", {"model": "gpt2"})
        assert isinstance(result, str)
        assert len(result) == 64  # sha256 hex length

    def test_deterministic_behavior(self):
        """Same inputs always produce the same hash."""
        params = {"model": "gpt2", "batch_size": 32}
        hash1 = compute_params_hash("test_module", "1.0.0", params)
        hash2 = compute_params_hash("test_module", "1.0.0", params)
        assert hash1 == hash2

    def test_different_module_id(self):
        """Different module_id produces different hash."""
        params = {"model": "gpt2"}
        hash1 = compute_params_hash("module_a", "1.0.0", params)
        hash2 = compute_params_hash("module_b", "1.0.0", params)
        assert hash1 != hash2

    def test_different_version(self):
        """Different form_schema_version produces different hash."""
        params = {"model": "gpt2"}
        hash1 = compute_params_hash("test_module", "1.0.0", params)
        hash2 = compute_params_hash("test_module", "2.0.0", params)
        assert hash1 != hash2

    def test_different_params(self):
        """Different params produce different hash."""
        hash1 = compute_params_hash("test_module", "1.0.0", {"model": "gpt2"})
        hash2 = compute_params_hash("test_module", "1.0.0", {"model": "gpt3"})
        assert hash1 != hash2

    def test_key_order_independence(self):
        """Key order in dict doesn't affect hash."""
        hash1 = compute_params_hash("test_module", "1.0.0", {"a": 1, "b": 2})
        hash2 = compute_params_hash("test_module", "1.0.0", {"b": 2, "a": 1})
        assert hash1 == hash2

    def test_nested_params(self):
        """Nested params are hashed correctly."""
        params = {"config": {"layers": 5, "activation": "relu"}}
        result = compute_params_hash("test_module", "1.0.0", params)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_list_params(self):
        """List params are hashed correctly."""
        params = {"layers": [1, 2, 3, 4]}
        result = compute_params_hash("test_module", "1.0.0", params)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_empty_params(self):
        """Empty params produce a valid hash."""
        result = compute_params_hash("test_module", "1.0.0", {})
        assert isinstance(result, str)
        assert len(result) == 64

    def test_numeric_params(self):
        """Numeric params (int and float) are hashed correctly."""
        params = {"int": 42, "float": 3.14, "negative": -10}
        result = compute_params_hash("test_module", "1.0.0", params)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_unicode_params(self):
        """Unicode params are hashed correctly."""
        params = {"text": "café", "emoji": "😀"}
        result = compute_params_hash("test_module", "1.0.0", params)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_special_characters(self):
        """Special characters in params are hashed correctly."""
        params = {"special": "!@#$%^&*()", "newlines": "line1\nline2"}
        result = compute_params_hash("test_module", "1.0.0", params)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_enum_normalization(self):
        """Enum values are normalized for hashing."""

        class TestEnum(Enum):
            OPTION_A = "option_a"

        # Enum and its value should produce same hash
        params1 = {"option": TestEnum.OPTION_A}
        params2 = {"option": "option_a"}

        hash1 = compute_params_hash("test_module", "1.0.0", params1)
        hash2 = compute_params_hash("test_module", "1.0.0", params2)
        assert hash1 == hash2

    def test_consistent_results(self):
        """Hash is consistent across multiple calls."""
        params = {"model": "gpt2", "batch": 32}
        hashes = [compute_params_hash("test", "1.0", params) for _ in range(10)]
        assert len(set(hashes)) == 1  # All hashes are identical


class TestHashProperties:
    """Tests for hash properties."""

    def test_hash_format(self):
        """Hash is hexadecimal string."""
        result = compute_params_hash("test", "1.0", {"a": 1})
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_length(self):
        """Hash is 64 characters (sha256 hex)."""
        result = compute_params_hash("test", "1.0", {"a": 1})
        assert len(result) == 64

    def test_collision_avoidance_simple(self):
        """Different inputs produce different hashes (simple test)."""
        hashes = []
        for i in range(10):
            h = compute_params_hash("test", "1.0", {"value": i})
            hashes.append(h)

        # All hashes should be different
        assert len(set(hashes)) == 10

    def test_collision_avoidance_complex(self):
        """Different inputs produce different hashes (complex test)."""
        hashes = []
        params_variations = [
            {"a": 1},
            {"a": 2},
            {"b": 1},
            {"a": 1, "b": 2},
            {"b": 2, "a": 1},  # Same as above, different order
            {"nested": {"x": 1}},
            {"nested": {"x": 2}},
        ]

        for params in params_variations:
            h = compute_params_hash("test", "1.0", params)
            hashes.append(h)

        # Only 6 unique hashes (params 4 and 5 should collide)
        assert len(set(hashes)) == 6

    def test_module_id_effects(self):
        """Module_id significantly affects hash."""
        base_params = {"model": "gpt2"}
        hash_base = compute_params_hash("module_a", "1.0", base_params)

        # Changing module_id changes hash even with same params
        hash_different = compute_params_hash("module_b", "1.0", base_params)
        assert hash_base != hash_different

    def test_version_effects(self):
        """Version significantly affects hash."""
        base_params = {"model": "gpt2"}
        hash_v1 = compute_params_hash("test", "1.0", base_params)
        hash_v2 = compute_params_hash("test", "2.0", base_params)
        assert hash_v1 != hash_v2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_none_values(self):
        """None values in params are handled correctly."""
        params = {"value": None, "nested": {"inner": None}}
        result = compute_params_hash("test", "1.0", params)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_zero_values(self):
        """Zero values (int and float) are handled correctly."""
        params = {"zero_int": 0, "zero_float": 0.0}
        hash1 = compute_params_hash("test", "1.0", params)
        hash2 = compute_params_hash("test", "1.0", params)
        assert hash1 == hash2  # Should be deterministic

    def test_very_long_strings(self):
        """Very long strings are hashed correctly."""
        params = {"long": "x" * 10000}
        result = compute_params_hash("test", "1.0", params)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_deeply_nested_structures(self):
        """Deeply nested structures are handled correctly."""
        params = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        result = compute_params_hash("test", "1.0", params)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_list_of_dicts(self):
        """List of dicts is normalized correctly."""
        params = {"items": [{"id": 1}, {"id": 2}, {"id": 3}]}
        result = compute_params_hash("test", "1.0", params)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_complex_mixed_structures(self):
        """Complex mixed structures are handled correctly."""
        params = {
            "simple": "value",
            "numbers": [1, 2, 3.5],
            "nested": {"a": [1, 2], "b": {"c": "deep"}},
            "bools": [True, False],
            "none": None,
            "empty": [],
        }
        result = compute_params_hash("test", "1.0", params)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_empty_strings(self):
        """Empty strings are handled correctly."""
        params = {"empty": "", "mixed": ["", "value", ""]}
        result = compute_params_hash("test", "1.0", params)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_negative_numbers(self):
        """Negative numbers are handled correctly."""
        params = {"neg_int": -42, "neg_float": -3.14}
        result = compute_params_hash("test", "1.0", params)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_float_precision(self):
        """Float precision is preserved in hash."""
        params1 = {"value": 0.123456789}
        params2 = {"value": 0.123456788}

        hash1 = compute_params_hash("test", "1.0", params1)
        hash2 = compute_params_hash("test", "1.0", params2)
        assert hash1 != hash2  # Different precision, different hash

    def test_boolean_values(self):
        """Boolean values are handled correctly."""
        params = {"flag_true": True, "flag_false": False}
        result = compute_params_hash("test", "1.0", params)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_empty_module_id(self):
        """Empty module_id is handled."""
        result = compute_params_hash("", "1.0", {"a": 1})
        assert isinstance(result, str)
        assert len(result) == 64

    def test_empty_version(self):
        """Empty version is handled."""
        result = compute_params_hash("test", "", {"a": 1})
        assert isinstance(result, str)
        assert len(result) == 64

    def test_all_empty(self):
        """All empty inputs are handled."""
        result = compute_params_hash("", "", {})
        assert isinstance(result, str)
        assert len(result) == 64
