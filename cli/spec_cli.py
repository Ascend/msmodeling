"""MindStudio unified CLI helpers: names, help layout, version, and aliases.

Implements the public-command subset of the MindStudio CLI spec for msmodeling:
kebab-case long options, single-character short options, hidden deprecated
aliases with a one-shot stderr warning, metavar/choices style, and the
Description/Usage/Required/Optional/Examples help layout.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from enum import Enum
from importlib import metadata
from typing import Any

from cli.logo import render_logo

STANDARD_LOG_LEVELS = ("debug", "info", "warning", "error", "critical")
LOG_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

METAVAR_DIR = "<DIR>"
METAVAR_FILE = "<FILE>"
METAVAR_N = "<N>"
METAVAR_FLOAT = "<FLOAT>"
METAVAR_NAME = "<NAME>"
METAVAR_SEC = "<SEC>"
METAVAR_ID = "<ID>"
METAVAR_RANGE = "<RANGE>"

_DEPRECATED_ATTR = "_spec_deprecated"
_WARNED_ALIASES: set[str] = set()


def to_kebab(value: str) -> str:
    return str(value).replace("_", "-").lower()


def warn_deprecated(old: str, new: str) -> None:
    """Print a one-shot deprecation warning for a compatibility alias."""
    key = f"{old}->{new}"
    if key in _WARNED_ALIASES:
        return
    _WARNED_ALIASES.add(key)
    print(
        f"WARNING: {old} is deprecated; use {new} instead. The old form remains accepted for compatibility.",
        file=sys.stderr,
    )


def reset_deprecation_warnings() -> None:
    _WARNED_ALIASES.clear()


def _root_parser(target: Any) -> argparse.ArgumentParser:
    obj = target
    while hasattr(obj, "_container"):
        obj = obj._container
    return obj


def _deprecated_map(parser: argparse.ArgumentParser) -> dict[str, str]:
    mapping = getattr(parser, _DEPRECATED_ATTR, None)
    if mapping is None:
        mapping = {}
        setattr(parser, _DEPRECATED_ATTR, mapping)
    return mapping


def inherit_deprecated(child: argparse.ArgumentParser, parent: argparse.ArgumentParser) -> None:
    merged = {**_deprecated_map(parent), **_deprecated_map(child)}
    setattr(child, _DEPRECATED_ATTR, merged)


def add_option(target: Any, *option_strings: str, aliases: Sequence[str] = (), **kwargs: Any) -> argparse.Action:
    """Register a public option plus hidden compatibility aliases."""
    action = target.add_argument(*option_strings, **kwargs)
    if not aliases:
        return action
    dest = kwargs.get("dest", action.dest)
    alias_kwargs = dict(kwargs)
    alias_kwargs["dest"] = dest
    alias_kwargs["help"] = argparse.SUPPRESS
    alias_kwargs.pop("required", None)
    parser = _root_parser(target)
    canonical = next((opt for opt in option_strings if opt.startswith("--")), option_strings[0])
    for alias in aliases:
        alias_action = target.add_argument(alias, **alias_kwargs)
        alias_action.spec_replacement = canonical  # type: ignore[attr-defined]
        original_call = alias_action.__call__

        def _call(
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: Any,
            option_string: str | None = None,
            *,
            _original=original_call,
            _canonical=canonical,
        ) -> None:
            if option_string:
                warn_deprecated(option_string, _canonical)
            _original(parser, namespace, values, option_string)

        alias_action.__call__ = _call  # type: ignore[method-assign]
        _deprecated_map(parser)[alias] = canonical
    return action


def _collect_alias_map(parser: argparse.ArgumentParser) -> dict[str, str]:
    mapping = dict(getattr(parser, _DEPRECATED_ATTR, {}))
    for action in parser._actions:
        replacement = getattr(action, "spec_replacement", None)
        if replacement:
            for opt in action.option_strings:
                mapping[opt] = replacement
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                mapping.update(_collect_alias_map(subparser))
    return mapping


def warn_deprecated_from_argv(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> None:
    mapping = _collect_alias_map(parser)
    if not mapping:
        return
    tokens = list(sys.argv[1:] if argv is None else argv)
    seen: set[str] = set()
    for token in tokens:
        key = token.split("=", 1)[0]
        replacement = mapping.get(key)
        if replacement and key not in seen:
            seen.add(key)
            warn_deprecated(key, replacement)


def parse_args(parser: argparse.ArgumentParser, args: Sequence[str] | None = None) -> argparse.Namespace:
    namespace = parser.parse_args(args)
    warn_deprecated_from_argv(parser, args)
    resolve_log_level(namespace, argv=list(sys.argv[1:] if args is None else args))
    return namespace


def _git_hash() -> str:
    git = shutil.which("git")
    if not git:
        return "unknown"
    try:
        result = subprocess.run(
            [git, "rev-parse", "--short=7", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    hash_ = (result.stdout or "").strip()
    return hash_ if result.returncode == 0 and hash_ else "unknown"


def package_version() -> str:
    try:
        return metadata.version("msmodeling")
    except metadata.PackageNotFoundError:
        return "0.2.0"


def format_version(tool_name: str = "msmodeling") -> str:
    logo = render_logo(color=False, terminal_cols=65)
    version = package_version()
    git_hash = _git_hash()
    return (
        f"{logo}\n"
        f"{tool_name} {version} ({git_hash})\n"
        "Copyright (C) 2026 Huawei Technologies Co., Ltd.\n"
        "License: Mulan PSL v2.\n"
        "\n"
        "Build Info:\n"
        f"  Repo : https://gitcode.com/Ascend/msmodeling\n"
    )


class VersionAction(argparse.Action):
    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str = argparse.SUPPRESS,
        default: str = argparse.SUPPRESS,
        help: str = "Show version information.",  # noqa: A002
    ) -> None:
        super().__init__(option_strings=option_strings, dest=dest, default=default, nargs=0, help=help)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        tool = parser.prog.split()[0] if parser.prog else "msmodeling"
        sys.stdout.write(format_version(tool))
        parser.exit(0)


def add_version_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-V", "--version", action=VersionAction)


_LOG_LEVEL_FLAGS = ("--log-level", "--log_level")


def _argv_has_log_level(argv: Sequence[str] | None) -> bool | None:
    if argv is None:
        return None
    return any(token.split("=", 1)[0] in _LOG_LEVEL_FLAGS for token in argv)


def add_log_options(parser: argparse.ArgumentParser | argparse._ArgumentGroup) -> None:
    add_option(
        parser,
        "--log-level",
        choices=list(STANDARD_LOG_LEVELS),
        default="error",
        help=(
            "Specifies the verbosity level for log output. "
            "Available levels: 'debug' (most verbose), 'info', 'warning', 'error', 'critical' (least verbose)."
        ),
        aliases=("--log_level",),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output detail (equivalent to --log-level debug).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet output (equivalent to --log-level error).",
    )


def resolve_log_level(args: argparse.Namespace, argv: Sequence[str] | None = None) -> str:
    explicit = getattr(args, "log_level", None)
    on_cli = _argv_has_log_level(argv)
    verbose = getattr(args, "verbose", False)
    quiet = getattr(args, "quiet", False)
    if explicit and on_cli is not False:
        level = str(explicit).lower()
    elif verbose:
        level = "debug"
    elif quiet:
        level = "error"
    elif explicit:
        level = str(explicit).lower()
    else:
        level = "error"
    args.log_level = level
    return level


def configure_std_logging(args: argparse.Namespace, *, log_format: str | None = None) -> str:
    level = resolve_log_level(args, argv=sys.argv[1:])
    kwargs: dict[str, Any] = {"level": LOG_LEVEL_MAP[level]}
    if log_format:
        kwargs["format"] = log_format
    logging.basicConfig(**kwargs)
    return level


def kebab_choice_metavar(values: Iterable[str]) -> str:
    return "{" + ",".join(to_kebab(value) for value in values) + "}"


def native_choice_metavar(values: Iterable[str]) -> str:
    return "{" + ",".join(str(value) for value in values) + "}"


def make_enum_type(enum_cls: type[Enum], option_name: str):
    """Parse native enum values; also accept kebab-case spellings."""
    members = list(enum_cls)
    kebab_map = {to_kebab(member.value): member for member in members}

    def parser(value: str):
        if isinstance(value, enum_cls):
            return value
        kebab = to_kebab(value)
        member = kebab_map.get(kebab)
        if member is None:
            allowed = ", ".join(str(item.value) for item in members)
            raise argparse.ArgumentTypeError(f"invalid choice {value!r} for {option_name} (choose from {allowed})")
        return member

    parser.__name__ = f"parse_{enum_cls.__name__}"
    return parser, native_choice_metavar(member.value for member in members)


def make_token_type(
    canonical_values: Sequence[str],
    option_name: str,
    *,
    store_canonical: str = "kebab",
    registered_names: bool = False,
):
    """Parse token values.

    Help shows ``canonical_values`` as passed. Kebab and snake spellings both
    parse. ``store_canonical`` is ``kebab`` or ``snake`` (the form returned to dest).
    ``registered_names=True`` is for registry keys such as ``ais_bench``.
    """
    kebab_values = [to_kebab(value) for value in canonical_values]
    snake_map = {to_kebab(value): value.replace("-", "_") for value in canonical_values}
    display_values = list(canonical_values)

    def parser(value: str) -> str:
        kebab = to_kebab(value)
        if kebab not in kebab_values:
            allowed = ", ".join(display_values)
            raise argparse.ArgumentTypeError(f"invalid choice {value!r} for {option_name} (choose from {allowed})")
        if store_canonical == "snake":
            return snake_map[kebab]
        if registered_names:
            return next(item for item in canonical_values if to_kebab(item) == kebab)
        return kebab

    parser.__name__ = f"parse_{option_name.strip('-').replace('-', '_')}"
    return parser, native_choice_metavar(display_values)


def _is_suppressed(action: argparse.Action) -> bool:
    return action.help is argparse.SUPPRESS or action.dest is argparse.SUPPRESS


def _is_required(action: argparse.Action) -> bool:
    if isinstance(action, argparse._SubParsersAction):
        return bool(action.required)
    if action.option_strings:
        return bool(action.required)
    return action.nargs not in (argparse.OPTIONAL, argparse.ZERO_OR_MORE, argparse.REMAINDER, "*")


def _format_option_name(action: argparse.Action) -> str:
    shorts = [
        opt for opt in action.option_strings if len(opt) == 2 and opt.startswith("-") and not opt.startswith("--")
    ]
    longs = [opt for opt in action.option_strings if opt.startswith("--")]
    metavar = _metavar_for(action)
    name_parts: list[str] = []
    if shorts:
        name_parts.append(f"{shorts[0]},")
    else:
        name_parts.append("   ")
    if longs:
        long_text = ", ".join(longs)
        name_parts.append(f"{long_text} {metavar}".rstrip() if metavar else long_text)
    elif action.option_strings:
        name_parts.append(action.option_strings[0])
    else:
        name_parts = [metavar] if metavar else [action.dest]
    return " ".join(part for part in name_parts if part).rstrip()


def _metavar_for(action: argparse.Action) -> str:
    if action.nargs == 0:
        return ""
    if action.choices and not action.metavar:
        base = "{" + ",".join(str(choice) for choice in action.choices) + "}"
    elif action.metavar:
        metavar = action.metavar
        if isinstance(metavar, tuple):
            return " ".join(str(item) for item in metavar)
        base = str(metavar)
    elif not action.option_strings:
        return action.dest
    else:
        base = METAVAR_NAME
    if "[" in base:
        return base
    if action.nargs in ("+", argparse.ONE_OR_MORE):
        return f"{base} [{base} ...]"
    if action.nargs in ("*", argparse.ZERO_OR_MORE):
        return f"[{base} ...]"
    return base


def _format_default_value(default: Any) -> str:
    if isinstance(default, Enum):
        return str(default.value)
    if isinstance(default, list):
        return "[" + ", ".join(_format_default_value(item) for item in default) + "]"
    return str(default)


def _default_annotation(action: argparse.Action) -> str:
    if action.required or action.nargs == 0 and action.const is True and action.default in (None, False):
        if action.option_strings and action.nargs == 0:
            if action.default in (None, False):
                return " [default: off]"
            if action.default is True:
                return " [default: on]"
        if action.required:
            return ""
    default = action.default
    if default in (None, argparse.SUPPRESS) or default == []:
        return ""
    if action.nargs == 0:
        return " [default: on]" if default else " [default: off]"
    return f" [default: {_format_default_value(default)}]"


def _format_action_line(action: argparse.Action, name_width: int) -> str:
    name = _format_option_name(action)
    help_text = (action.help or "").replace("%(default)s", str(action.default))
    help_text = help_text.rstrip()
    if help_text and "[default:" not in help_text.lower() and "(default:" not in help_text.lower():
        help_text += _default_annotation(action)
    return f"  {name.ljust(name_width)}  {help_text}".rstrip()


def _public_actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    return [action for action in parser._actions if not _is_suppressed(action)]


class SpecHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keep argparse usage generation; SpecArgumentParser rewrites the full help."""


class SpecArgumentParser(argparse.ArgumentParser):
    def __init__(
        self,
        *args: Any,
        examples: str | None = None,
        output_help: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("formatter_class", SpecHelpFormatter)
        kwargs.pop("parser_class", None)
        super().__init__(*args, **kwargs)
        self.examples = examples
        self.output_help = output_help
        for action in self._actions:
            if isinstance(action, argparse._HelpAction):
                action.help = "Show help message."

    def format_help(self) -> str:
        description = (self.description or "").strip()
        usage = self.format_usage()
        usage_line = usage.replace("usage:", "", 1).replace("Usage:", "", 1).strip()
        actions = _public_actions(self)
        subparsers = [action for action in actions if isinstance(action, argparse._SubParsersAction)]
        required = [
            action for action in actions if _is_required(action) and not isinstance(action, argparse._SubParsersAction)
        ]
        optional = [
            action
            for action in actions
            if not _is_required(action) and not isinstance(action, argparse._SubParsersAction)
        ]
        named = required + optional
        name_width = max((_format_option_name(action) for action in named), key=len, default="")
        width = min(max(len(name_width), 24), 48)

        parts = [
            "Description:",
            f"  {description}" if description else "  MindStudio Modeling command.",
            "",
            "Usage:",
            f"  {usage_line}",
            "",
        ]
        if subparsers:
            parts.append("Commands:")
            cmd_width = max(
                (len(name) for action in subparsers for name in action.choices),
                default=12,
            )
            cmd_width = min(max(cmd_width, 12), 32)
            for action in subparsers:
                help_by_name = {
                    choice.dest: (choice.help or "").strip() for choice in getattr(action, "_choices_actions", [])
                }
                for name, subparser in action.choices.items():
                    help_text = help_by_name.get(name) or (subparser.description or "").strip()
                    parts.append(f"  {name.ljust(cmd_width)}  {help_text}".rstrip())
            parts.append("")
        if required:
            parts.append("Required arguments:")
            parts.extend(_format_action_line(action, width) for action in required)
            parts.append("")
        if optional:
            parts.append("Optional arguments:")
            parts.extend(_format_action_line(action, width) for action in optional)
            parts.append("")
        examples = (self.examples or self.epilog or "").strip()
        if examples:
            parts.append("Examples:")
            for line in examples.splitlines():
                parts.append(line if line.startswith("  ") else f"  {line}")
            parts.append("")
        if self.output_help:
            parts.append("Output:")
            for line in self.output_help.strip().splitlines():
                parts.append(line if line.startswith("  ") else f"  {line}")
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"
