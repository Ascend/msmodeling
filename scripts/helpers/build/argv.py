"""Argument parsing for the root build.py entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_VALID_TOKENS = frozenset({"local", "test"})
_TEST_EXTRA_KEYS = frozenset({"test_map_path", "base_branch", "offline", "weights_prune"})


@dataclass(frozen=True)
class BuildOptions:
    is_test: bool
    is_local: bool
    version: str | None
    version_explicit: bool
    extras: Mapping[str, str]


def _parse_extras(raw_extras: Sequence[str], parser: argparse.ArgumentParser) -> dict[str, str]:
    extras: dict[str, str] = {}
    for item in raw_extras:
        if "=" not in item:
            parser.error(f"--extra must be KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        if not key:
            parser.error(f"--extra key must be non-empty, got: {item!r}")
        if key in extras:
            parser.error(f"duplicate --extra key: {key!r}")
        extras[key] = value
    return extras


def _validate_extras(
    is_test: bool,
    extras: Mapping[str, str],
    parser: argparse.ArgumentParser,
) -> None:
    if not extras:
        return
    if not is_test:
        parser.error("--extra is only supported with the test command")
    unknown = sorted(set(extras) - _TEST_EXTRA_KEYS)
    if unknown:
        allowed = ", ".join(sorted(_TEST_EXTRA_KEYS))
        parser.error(f"unknown --extra key(s): {', '.join(unknown)}; allowed: {allowed}")


def _parse_tokens(tokens: Sequence[str], parser: argparse.ArgumentParser) -> tuple[bool, bool]:
    seen: set[str] = set()
    is_test = False
    is_local = False
    for token in tokens:
        if token not in _VALID_TOKENS:
            parser.error(f"unknown positional argument: {token!r}")
        if token in seen:
            parser.error(f"duplicate positional argument: {token!r}")
        seen.add(token)
        if token == "test":
            is_test = True
        elif token == "local":
            is_local = True
    return is_test, is_local


def parse_argv(argv: Sequence[str] | None = None) -> BuildOptions:
    """Parse CLI arguments into :class:`BuildOptions`."""
    parser = argparse.ArgumentParser(prog="build.py")
    parser.add_argument(
        "-v",
        "--version",
        default=None,
        metavar="VERSION",
        help="wheel artifact version label (default: pyproject.toml project.version)",
    )
    parser.add_argument(
        "-e",
        "--extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "test-only key/value pair (repeatable); allowed keys: test_map_path, base_branch, offline, weights_prune"
        ),
    )
    parser.add_argument(
        "tokens",
        nargs="*",
        metavar="COMMAND",
        help="optional commands: test, local",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    is_test, is_local = _parse_tokens(args.tokens, parser)
    extras = _parse_extras(args.extra, parser)
    _validate_extras(is_test, extras, parser)
    return BuildOptions(
        is_test=is_test,
        is_local=is_local,
        version=args.version,
        version_explicit=args.version is not None,
        extras=extras,
    )
