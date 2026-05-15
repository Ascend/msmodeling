import argparse
import logging


LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def check_positive_integer(value):
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid integer value: %r", value) from None
    if value <= 0:
        raise argparse.ArgumentTypeError("%r is not a positive integer", value)

    return value


def parse_int_range(value: str, name: str) -> tuple[int, int]:
    """Parse a range string in the form 'start,end'.

    Semantics:
    - Surrounding spaces are allowed around both numbers.
    - Both values must be integers and non-negative.
    - `end` must be greater than or equal to `start`.

    Args:
        value: Raw CLI string, for example '11,45' or ' 0 , 54 '.
        name: Argument name used in error messages, for example '--cache-step-range'.

    Returns:
        A tuple `(start, end)`.

    Raises:
        ValueError: If input format or bounds are invalid.
    """
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"{name} must be 'start,end', got {value!r}.")
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"{name} must be 'start,end', got {value!r}.") from exc
    if start < 0 or end < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    if end < start:
        raise ValueError(f"{name} must be 'start,end' with end >= start, got {value!r}.")
    return start, end
