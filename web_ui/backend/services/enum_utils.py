"""Enum coercion utilities.

Converts string values (kebab-case or UPPER_SNAKE) into StrEnum members.
Used by runners to coerce frontend-sent string values back to the typed enums
the CLI path would have produced via argparse ``type=``.
"""

from __future__ import annotations

import sys
from typing import TypeVar

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from strenum import StrEnum

T = TypeVar("T", bound=StrEnum)


def coerce_enum(enum_cls: type[T], value: str) -> T:
    """Coerce a string into a StrEnum member.

    Matches by:
    1. Member ``name`` (UPPER_SNAKE, e.g. ``W8A8_DYNAMIC``)
    2. Member ``value`` after ``to_kebab`` (kebab-case, e.g. ``w8a8-dynamic``)

    Raises ``ValueError`` when no member matches.
    """
    from cli.spec_cli import to_kebab

    if isinstance(value, enum_cls):
        return value

    # Try direct name match (UPPER_SNAKE)
    try:
        return enum_cls[value]
    except KeyError:
        pass

    # Try value match (kebab-case)
    target = to_kebab(value)
    for member in enum_cls:
        if to_kebab(str(member.value)) == target:
            return member

    raise ValueError(
        f"{value!r} is not a valid {enum_cls.__name__}. Expected one of: {', '.join(m.name for m in enum_cls)}"
    )
