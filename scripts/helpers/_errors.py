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
