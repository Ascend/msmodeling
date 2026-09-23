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

"""argparse adapter — generate CLI ArgumentParser from ModuleSpec.

Translates ``Param`` definitions into ``cli.spec_cli`` calls. spec_cli owns
the parser class, option registration, log/version injection, and deprecated-alias
warnings. This adapter maps Param → spec_cli calls.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable

from cli import spec_cli

from .datatypes import ModuleSpec, Param


def build_argparser(spec: ModuleSpec) -> spec_cli.SpecArgumentParser:
    """ModuleSpec → SpecArgumentParser.

    Args:
        spec: Module specification with fields, validators, and CLI metadata.

    Returns:
        Configured SpecArgumentParser ready for argument parsing.
    """
    parser = spec_cli.SpecArgumentParser(
        prog=spec.cli_prog,
        description=spec.cli_description or spec.title,
        examples=spec.cli_examples,
        output_help=spec.cli_output_help,
    )
    spec_cli.add_version_option(parser)
    groups: dict[str, argparse._ArgumentGroup] = {}
    log_options_added = False
    log_options_after = spec.log_options_after_group or "General Options"

    for p in spec.fields:
        grp_name = p.group or "Optional arguments"
        # Insert log options after specified field (e.g. image_generate)
        if (
            spec.log_options
            and not log_options_added
            and spec.log_options_after_field is not None
            and p.py_name == spec.log_options_after_field
        ):
            grp = groups.setdefault(grp_name, parser.add_argument_group(grp_name))
            _register_param(grp, p)
            spec_cli.add_log_options(parser)
            log_options_added = True
            continue
        # Add log options after the specified group
        if spec.log_options and not log_options_added and grp_name != log_options_after and log_options_after in groups:
            spec_cli.add_log_options(parser)
            log_options_added = True
        grp = groups.setdefault(grp_name, parser.add_argument_group(grp_name))
        _register_param(grp, p)

    if spec.log_options and not log_options_added:
        spec_cli.add_log_options(parser)

    return parser


def _register_param(grp: argparse._ArgumentGroup, p: Param) -> None:
    """Register a single Param in its group. Dispatches on cli_positional."""
    if p.cli_positional:
        _register_positional(grp, p)
    else:
        _register_flag(grp, p)


def _register_positional(grp: argparse._ArgumentGroup, p: Param) -> None:
    """Positional argument + optional same-dest formal --flag (model_id pattern).

    Args:
        grp: argparse group to register in.
        p: Param with cli_positional=True.
    """
    positional_kw = _build_kwargs(p, is_positional=True)
    # Use dedicated name for positional+flag combos to detect "passed both" conflict
    positional_name = f"{p.py_name}_positional" if p.cli_flag else p.dest_name
    if p.cli_flag:
        positional_kw.setdefault("metavar", positional_kw.get("metavar") or p.cli_metavar or "<NAME>")
    # For nargs="?" positionals sharing dest with a flag, use SUPPRESS as default
    # to prevent the positional's default from overriding the flag's value
    if p.nargs == "?":
        positional_kw["default"] = argparse.SUPPRESS
    grp.add_argument(positional_name, **positional_kw)

    # Same-dest formal flag(s)
    if p.cli_flag:
        alias_flags = tuple(f"--{a}" for a in p.cli_aliases)
        flag_kw = _build_kwargs(p, is_positional=False)
        flag_kw.pop("required", None)  # Deferred to require_model_id()
        flag_kw.pop("nargs", None)  # Flag uses default nargs (exactly one value)
        # model-path is hidden; model-id is the visible formal flag
        if p.cli_flag == "model-path":
            hidden_kw = dict(flag_kw)
            hidden_kw["help"] = argparse.SUPPRESS
            spec_cli.add_option(grp, "--model-path", dest=p.dest_name, **hidden_kw)
            visible_help = getattr(p, "cli_flag_help", None) or (
                "Model source. Recommended safe mode: a reviewed absolute local model path. "
                "Equivalent to the positional model id."
            )
            visible_kw = dict(flag_kw)
            visible_kw["help"] = visible_help
            spec_cli.add_option(grp, "--model-id", dest=p.dest_name, aliases=alias_flags, **visible_kw)
        else:
            if p.py_name == "model_id":
                flag_kw["help"] = getattr(p, "cli_flag_help", None) or (
                    "Model source. Recommended safe mode: a reviewed absolute local model path. "
                    "Equivalent to the positional model id."
                )
            spec_cli.add_option(grp, f"--{p.cli_flag}", dest=p.dest_name, aliases=alias_flags, **flag_kw)


def _register_flag(grp: argparse._ArgumentGroup, p: Param) -> None:
    """Regular --flag argument (the common case)."""
    formal_flag = f"--{p.name}"
    extra_formal_flags = tuple(f"--{a}" for a in p.cli_extra_flags)
    alias_flags = tuple(f"--{a}" for a in p.cli_aliases)
    kw = _build_kwargs(p, is_positional=False)
    spec_cli.add_option(
        grp,
        formal_flag,
        *extra_formal_flags,
        *([f"-{p.cli_short}"] if p.cli_short else []),
        dest=p.dest_name,
        aliases=alias_flags,
        **kw,
    )
    # Boolean dual-toggle: --no-X off-switch
    if p.cli_off_flag:
        off_help = p.cli_off_help or f"Enable {p.cli_off_flag.replace('-', ' ')} (default)."
        grp.add_argument(
            f"--{p.cli_off_flag}",
            dest=p.dest_name,
            action="store_false",
            default=argparse.SUPPRESS,
            help=off_help,
        )


def parse_module_args(
    spec: ModuleSpec, parser: spec_cli.SpecArgumentParser, argv: list[str] | None = None
) -> argparse.Namespace:
    """Unified parse entry: spec_cli.parse_args + validators + require_model_id.

    Args:
        spec: Module specification with fields and validators.
        parser: Configured SpecArgumentParser.
        argv: CLI tokens (None = use sys.argv).

    Returns:
        Parsed arguments Namespace.
    """
    from cli.utils import require_model_id

    args = spec_cli.parse_args(parser, argv)

    # Cross-field validators
    if spec.validators:
        params = {p.name: getattr(args, p.dest_name, None) for p in spec.fields}
        provided = _explicitly_provided(spec, parser, args, argv)
        for v in spec.validators:
            msg = v.fn(params, provided) if v.wants_provided else v.fn(params)
            if msg is not None:
                parser.error(msg)

    if spec.requires_model_id:
        require_model_id(parser, args)
    return args


def _explicitly_provided(
    spec: ModuleSpec,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str] | None,
) -> set[str]:
    """Kebab-case Param.name values explicitly set on the command line.

    Args:
        spec: Module specification.
        parser: Configured ArgumentParser (unused; kept for API compatibility).
        args: Parsed arguments.
        argv: CLI tokens (None = use sys.argv).

    Returns:
        Set of kebab-case param names explicitly passed.
    """
    import sys as _sys

    tokens = list(_sys.argv[1:] if argv is None else argv)
    # Build flag→param mapping directly from Param definitions.
    # This avoids accessing parser._actions (private argparse API).
    flag_to_param: dict[str, str] = {}
    for p in spec.fields:
        # Positional with same-dest flag (e.g. model_id: positional + --model-id)
        if p.cli_positional:
            if p.cli_flag:
                flag_to_param[f"--{p.cli_flag}"] = p.name
                for alias in p.cli_aliases:
                    flag_to_param[f"--{alias}"] = p.name
            continue
        # Regular flag
        flag_to_param[f"--{p.name}"] = p.name
        for flag in p.cli_extra_flags:
            flag_to_param[f"--{flag}"] = p.name
        if p.cli_short:
            flag_to_param[f"-{p.cli_short}"] = p.name
        for alias in p.cli_aliases:
            flag_to_param[f"--{alias}"] = p.name
        # Boolean off-flag (e.g. --no-foo)
        if p.cli_off_flag:
            flag_to_param[f"--{p.cli_off_flag}"] = p.name
    provided: set[str] = set()
    for tok in tokens:
        name = flag_to_param.get(tok.split("=", 1)[0])
        if name is not None:
            provided.add(name)
    return provided


# ── kwargs generation ───────────────────────────────────────────────


def _build_kwargs(p: Param, *, is_positional: bool) -> dict:
    """Param → add_argument() kwargs.

    Args:
        p: Parameter definition.
        is_positional: True for positional args (no required=/default=).

    Returns:
        Kwargs dict for argparse add_argument().
    """
    kw: dict = {}

    # type: custom > auto-inferred from data_type + constraints
    if p.cli_type:
        kw["type"] = p.cli_type
    elif p.choices and _is_enum_list(p.choices):
        enum_cls = type(p.choices[0])
        parse_fn, metavar = spec_cli.make_enum_type(enum_cls, f"--{p.name}")
        kw["type"] = parse_fn
        kw["metavar"] = metavar
    elif p.choices and p.data_type == "string[]":
        parse_fn, metavar = spec_cli.make_token_type(list(p.choices), f"--{p.name}", store_canonical="snake")
        kw["type"] = parse_fn
        kw["metavar"] = metavar
    elif p.data_type == "integer" or p.data_type == "integer[]":
        if any(v is not None for v in (p.min, p.max, p.exclusive_min, p.exclusive_max)):
            kw["type"] = _int_type(p.min, p.max, p.exclusive_min, p.exclusive_max)
        else:
            kw["type"] = int
    elif p.data_type == "number" or p.data_type == "number[]":
        if any(v is not None for v in (p.min, p.max, p.exclusive_min, p.exclusive_max)):
            kw["type"] = _float_type(p.min, p.max, p.exclusive_min, p.exclusive_max)
        else:
            kw["type"] = _finite_float_type()
    elif p.data_type == "string[]":
        if p.nargs is None:
            kw["nargs"] = "+"
        kw["type"] = str
    elif p.data_type == "string" and (p.pattern is not None or p.max_length is not None):
        kw["type"] = _string_type(p.pattern, p.max_length)

    # choices: DEVICE gets runtime lookup; others use static choices
    if p.name == "device" and p.data_type == "string":
        from .shared import get_device_choices

        kw["choices"] = get_device_choices()
    elif p.choices and not (p.data_type == "string[]" and not _is_enum_list(p.choices)):
        if not _is_enum_list(p.choices):
            kw["choices"] = list(p.choices)
        else:
            kw["choices"] = _enum_values(p.choices)

    if p.nargs is not None:
        kw["nargs"] = p.nargs

    # action / default / required
    if p.data_type == "boolean":
        kw["action"] = p.cli_action if p.cli_action != "store" else "store_true"
        if not is_positional and p.default is not None:
            kw["default"] = p.default
    else:
        if p.cli_action != "store":
            kw["action"] = _resolve_custom_action(p.cli_action) if p.cli_action in ("batch_range",) else p.cli_action
        if not is_positional:
            if p.required:
                kw["required"] = True
            if p.default is not None:
                kw["default"] = p.default

    if p.cli_metavar:
        kw["metavar"] = p.cli_metavar
    if p.cli_help:
        kw["help"] = p.cli_help

    return kw


# ── Named custom actions (lazy import to avoid heavy deps) ─────────


_CUSTOM_ACTIONS: dict[str, type] = {}


def _resolve_custom_action(name: str) -> type:
    """Resolve named custom action class (lazy import)."""
    if name not in _CUSTOM_ACTIONS:
        from serving_cast.service.utils import BatchRangeAction

        _CUSTOM_ACTIONS["batch_range"] = BatchRangeAction
    return _CUSTOM_ACTIONS[name]


# ── Internal type builders ──────────────────────────────────────────


def _int_type(
    lo: float | None,
    hi: float | None,
    ex_lo: float | None = None,
    ex_hi: float | None = None,
) -> Callable[[str], int]:
    """Build int parser with range bounds."""

    def check(v: str) -> int:
        if v is argparse.SUPPRESS:  # nargs="?" default passthrough
            return v
        try:
            n = int(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid integer value: {v!r}") from None
        if lo is not None and n < lo:
            raise argparse.ArgumentTypeError(f"{n} is out of range; must be >= {lo}")
        if hi is not None and n > hi:
            raise argparse.ArgumentTypeError(f"{n} is out of range; must be <= {hi}")
        if ex_lo is not None and n <= ex_lo:
            raise argparse.ArgumentTypeError(f"{n} is out of range; must be > {ex_lo}")
        if ex_hi is not None and n >= ex_hi:
            raise argparse.ArgumentTypeError(f"{n} is out of range; must be < {ex_hi}")
        return n

    return check


def _float_type(
    lo: float | None,
    hi: float | None,
    ex_lo: float | None = None,
    ex_hi: float | None = None,
) -> Callable[[str], float]:
    """Build float parser with range bounds."""

    def check(v: str) -> float:
        if v is argparse.SUPPRESS:  # nargs="?" default passthrough
            return v
        try:
            n = float(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid float value: {v!r}") from None
        if not math.isfinite(n):
            raise argparse.ArgumentTypeError(f"Invalid finite number: {v!r}")
        if lo is not None and n < lo:
            raise argparse.ArgumentTypeError(f"{n} is out of range; must be >= {lo}")
        if hi is not None and n > hi:
            raise argparse.ArgumentTypeError(f"{n} is out of range; must be <= {hi}")
        if ex_lo is not None and n <= ex_lo:
            raise argparse.ArgumentTypeError(f"{n} is out of range; must be > {ex_lo}")
        if ex_hi is not None and n >= ex_hi:
            raise argparse.ArgumentTypeError(f"{n} is out of range; must be < {ex_hi}")
        return n

    return check


def _finite_float_type() -> Callable[[str], float]:
    """Build float parser that rejects NaN/Inf (used when no range bounds are set)."""

    def check(v: str) -> float:
        if v is argparse.SUPPRESS:
            return v
        try:
            n = float(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid float value: {v!r}") from None
        if not math.isfinite(n):
            raise argparse.ArgumentTypeError(f"Invalid finite number: {v!r}")
        return n

    return check


def _string_type(pattern: str | None, max_length: int | None) -> Callable[[str], str]:
    """Build string parser with optional pattern and max_length."""
    import re

    compiled = re.compile(pattern) if pattern else None

    def check(v: str) -> str:
        if v is argparse.SUPPRESS:  # nargs="?" default passthrough
            return v
        if max_length is not None and len(v) > max_length:
            raise argparse.ArgumentTypeError(f"String length exceeds {max_length} characters: {v!r}")
        if compiled is not None and not compiled.match(v):
            raise argparse.ArgumentTypeError(f"String contains invalid characters: {v!r}")
        return v

    return check


def _enum_values(choices: list) -> list:
    """Extract .value from Enum members for argparse."""
    return [getattr(c, "value", c) for c in choices]


def _is_enum_list(choices: list) -> bool:
    """Check if choices is a list of Enum instances."""
    return len(choices) > 0 and hasattr(choices[0], "value") and hasattr(choices[0], "name")
