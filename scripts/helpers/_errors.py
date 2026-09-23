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

"""Stdlib-only error types shared by helpers that must not import pydantic."""

from __future__ import annotations


class ConfigError(Exception):
    """Raised on configuration, environment, or data validation errors."""


def format_expected_got(field: str, expected: str, got: object) -> str:
    """Format a human-readable error message for an unexpected value.

    Args:
        field: The name of the field or variable being validated.
        expected: A description of the expected value or type.
        got: The actual value received.

    Returns:
        A formatted error string.
    """
    return f"Expected {field!r} to be {expected}. Got {got!r} instead."
