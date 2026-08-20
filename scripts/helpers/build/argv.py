"""Argument parsing for the root build.py entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_VALID_TOKENS = frozenset({"local", "test"})
_TEST_EXTRA_KEYS = frozenset({"test_map_path", "base_branch", "offline", "weights_prune"})
_BUILD_EXTRA_KEYS = frozenset({"only_down_deps"})
_SUITE_VALUES = ("ci_gate", "full", "smoke", "regression", "benchmark")


class BuildSuite(str, Enum):
    """Test modes for ``python build.py test --suite``."""

    CI_GATE = "ci_gate"
    FULL = "full"
    SMOKE = "smoke"
    REGRESSION = "regression"
    BENCHMARK = "benchmark"


@dataclass(frozen=True)
class BuildOptions:
    is_test: bool
    is_local: bool
    version: str | None
    version_explicit: bool
    extras: Mapping[str, str]
    suite: BuildSuite
    only_down_deps: bool = False


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
    allowed = _TEST_EXTRA_KEYS if is_test else _BUILD_EXTRA_KEYS
    unknown = sorted(set(extras) - allowed)
    if unknown:
        formatted = ", ".join(sorted(allowed))
        parser.error(f"unknown --extra key(s): {', '.join(unknown)}; allowed: {formatted}")
    if not is_test and "only_down_deps" in extras and extras["only_down_deps"] not in {"true", "false"}:
        parser.error(f"--extra only_down_deps must be 'true' or 'false', got: {extras['only_down_deps']!r}")


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
    suite_choices = ", ".join(_SUITE_VALUES)
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
            "key/value pair (repeatable); test keys: test_map_path, base_branch, offline, "
            "weights_prune; build keys: only_down_deps"
        ),
    )
    parser.add_argument(
        "--suite",
        default=None,
        choices=list(_SUITE_VALUES),
        help=f"test-only suite to run (default: full); choices: {suite_choices}",
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
    if args.suite is not None and not is_test:
        parser.error("--suite is only supported with the test command")
    suite = BuildSuite(args.suite) if args.suite is not None else BuildSuite.FULL
    only_down_deps = extras.get("only_down_deps") == "true" if not is_test else False
    return BuildOptions(
        is_test=is_test,
        is_local=is_local,
        version=args.version,
        version_explicit=args.version is not None,
        extras=extras,
        suite=suite,
        only_down_deps=only_down_deps,
    )
