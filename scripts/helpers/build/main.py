"""Root build.py entry — dispatch build or test."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.helpers.build.argv import parse_argv
from scripts.helpers.build.run_build import run_build
from scripts.helpers.build.run_test import run_test

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Parse argv and run build or test."""
    options = parse_argv(argv)
    if options.is_test:
        return run_test(options)
    return run_build(options)


if __name__ == "__main__":
    raise SystemExit(main())
