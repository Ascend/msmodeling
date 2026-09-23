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

"""Tests for services/enum_utils.py — enum coercion utilities.

Tests the coerce_enum() function that converts string values (kebab-case or
UPPER_SNAKE) into StrEnum members. Used by runners to coerce frontend-sent
string values back to typed enums.
"""

import pytest
import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from strenum import StrEnum

from services.enum_utils import coerce_enum


class ColorEnum(StrEnum):
    """Test enum with UPPER_SNAKE names."""

    RED_COLOR = "red_color"
    BLUE_COLOR = "blue_color"
    GREEN_COLOR = "green_color"


class StatusEnum(StrEnum):
    """Test enum with kebab-case values."""

    ACTIVE = "active"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"


class TestCoerceEnum:
    """Tests for coerce_enum() function."""

    def test_direct_name_match_upper_snake(self):
        """Matches by member name (UPPER_SNAKE)."""
        result = coerce_enum(ColorEnum, "RED_COLOR")
        assert result == ColorEnum.RED_COLOR
        assert isinstance(result, ColorEnum)

    def test_direct_name_match_multiple(self):
        """Different names match different members."""
        assert coerce_enum(ColorEnum, "BLUE_COLOR") == ColorEnum.BLUE_COLOR
        assert coerce_enum(ColorEnum, "GREEN_COLOR") == ColorEnum.GREEN_COLOR

    def test_value_match_kebab_case(self):
        """Matches by value after to_kebab conversion."""
        # in-progress is already kebab-case
        result = coerce_enum(StatusEnum, "in-progress")
        assert result == StatusEnum.IN_PROGRESS

    def test_value_match_upper_snake_to_kebab(self):
        """UPPER_SNAKE value is converted to kebab for matching."""
        # IN_PROGRESS name -> in-progress value
        result = coerce_enum(StatusEnum, "IN_PROGRESS")
        assert result == StatusEnum.IN_PROGRESS

    def test_enum_instance_passthrough(self):
        """Enum instance is returned as-is."""
        original = ColorEnum.RED_COLOR
        result = coerce_enum(ColorEnum, original)
        assert result is original

    def test_invalid_name_raises_valueerror(self):
        """Invalid name raises ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            coerce_enum(ColorEnum, "INVALID_COLOR")
        assert "INVALID_COLOR" in str(exc_info.value)
        assert "ColorEnum" in str(exc_info.value)
        assert "RED_COLOR" in str(exc_info.value)  # Shows valid options

    def test_invalid_value_raises_valueerror(self):
        """Invalid value raises ValueError."""
        with pytest.raises(ValueError):
            coerce_enum(StatusEnum, "unknown-status")

    def test_lowercase_matches_via_kebab_conversion(self):
        """Lowercase matches via to_kebab conversion (case-insensitive value match)."""
        # red_color -> red-color via to_kebab, matches RED_COLOR value
        result = coerce_enum(ColorEnum, "red_color")
        assert result == ColorEnum.RED_COLOR

    def test_real_world_quantize_enum(self):
        """Test with actual quantization enum from CLI."""
        from tensor_cast.core.quantization.datatypes import QuantizeLinearAction

        # UPPER_SNAKE name match
        result = coerce_enum(QuantizeLinearAction, "W8A8_DYNAMIC")
        assert result == QuantizeLinearAction.W8A8_DYNAMIC

        # Value match (kebab-case)
        result = coerce_enum(QuantizeLinearAction, "w8a8-dynamic")
        assert result == QuantizeLinearAction.W8A8_DYNAMIC

    def test_real_world_log_level_enum(self):
        """Test with log level enum if it exists."""
        try:
            from cli.spec_cli import LogLevel

            result = coerce_enum(LogLevel, "DEBUG")
            assert result == LogLevel.DEBUG
            result = coerce_enum(LogLevel, "debug")
            assert result == LogLevel.DEBUG
        except (ImportError, AttributeError):
            # LogLevel enum may not exist in all versions
            pytest.skip("LogLevel enum not available")


class TestCoerceEnumEdgeCases:
    """Edge cases for coerce_enum()."""

    def test_empty_string_raises(self):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError):
            coerce_enum(ColorEnum, "")

    def test_none_raises_value_error(self):
        """None raises ValueError (not a valid enum value)."""
        with pytest.raises(ValueError):
            coerce_enum(ColorEnum, None)

    def test_numeric_string_raises(self):
        """Numeric string raises ValueError."""
        with pytest.raises(ValueError):
            coerce_enum(ColorEnum, "123")
