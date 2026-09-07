"""Core dataclasses for the CLI/WebUI parameter registry.

This module is the single source of truth for CLI-side parameter definitions:

- ``Param``: immutable parameter definition (CLI-focused). The argparse_adapter
  reads all fields to generate an ArgumentParser; the json_adapter reads a subset
  (name, data_type, default, required, min/max/choices/pattern/max_length/group/
  nargs/cli_help) to generate the frontend form JSON.

- ``ValidatorRef``: cross-field validator reference (fn for CLI, name for UI).

- ``ModuleSpec``: a CLI module's complete parameter spec (fields + validators
  + CLI metadata).

UI types (I18nText, UIFieldProps) live in web_ui/backend/services/ui_props/.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Param:
    """Immutable parameter definition — the single source of truth for a field.

    frozen=True: shared singletons (e.g. MODEL_ID) cannot be accidentally
    mutated (``MODEL_ID.default = x`` raises FrozenInstanceError). Use
    ``override()`` to create a modified copy for module-specific variants.
    """

    # ── Identity ────────────────────────────────────────────────────
    name: str
    ## Kebab-case field id, CLI flag, and JSON key (e.g. "tp-size").

    data_type: str
    ## "string" | "integer" | "number" | "boolean" | "string[]".

    # ── Default value & required ────────────────────────────────────
    default: Any = None
    required: bool = False

    # ── Declarative constraints (L1 validation, auto-generated both sides) ──
    min: int | float | None = None
    max: int | float | None = None
    exclusive_min: int | float | None = None
    exclusive_max: int | float | None = None
    ## Open-bound range constraints. CLI: type fn check; Frontend: gt/lt rules.

    choices: list | None = None
    ## Allowed values; may pass Python Enum via list(EnumClass).

    pattern: str | None = None
    max_length: int | None = None
    ## String-only constraints.

    # ── Grouping ────────────────────────────────────────────────────
    group: str | None = None
    ## CLI: argparse group name; Frontend: i18n via ui_props GROUP_LABELS.

    # ── Multi-value ─────────────────────────────────────────────────
    nargs: str | int | None = None
    ## "*" = 0+, "+" = 1+, "?" = 0-1, int = exact. Frontend uses multi_values.

    # ── CLI-specific escape hatches (not used by frontend) ──────────
    cli_type: Callable | None = None
    ## Custom argparse type fn; None = auto-generate from data_type + constraints.

    cli_action: str = "store"
    ## argparse action; boolean defaults to "store_true".

    cli_positional: bool = False
    ## True = positional (no -- prefix). argparse_adapter skips required=/default=.

    cli_flag: str | None = None
    ## Formal --flag name override; only for positional+flag combos (e.g. MODEL_ID).

    cli_flag_help: str | None = None
    ## Help text for the formal --flag variant of a positional+flag combo.

    cli_aliases: tuple[str, ...] = ()
    ## Deprecated hidden aliases (no -- prefix); one-shot warning, not in --help.

    cli_extra_flags: tuple[str, ...] = ()
    ## Additional public --flag names (no -- prefix); shown in --help, no warning.

    cli_short: str | None = None
    ## Single-char short option (no - prefix); cross-command vocabulary-fixed.

    cli_off_flag: str | None = None
    ## Boolean off-switch (no -- prefix); generates --name/--no-name pair.

    cli_off_help: str | None = None
    ## Help text for off-flag; None = auto-generated.

    cli_metavar: str | None = None
    ## argparse metavar; None = spec_cli convention.

    cli_help: str | None = None
    ## --help text; Frontend tooltip fallback (zh=en=cli_help when unset).

    cli_dest: str | None = None
    ## argparse dest override; None = py_name. Rare, for downstream attr compat.

    # ── Methods ─────────────────────────────────────────────────────
    @property
    def py_name(self) -> str:
        """name with hyphens → underscores, for argparse dest."""
        return self.name.replace("-", "_")

    @property
    def dest_name(self) -> str:
        """Effective argparse dest: cli_dest override or py_name."""
        return self.cli_dest or self.py_name

    # Allowlist of fields that override() may modify. Prevents accidental
    # modification of key fields like name, data_type.
    # group IS allowed: modules that register everything on the bare parser
    # (no argument groups, e.g. image_generate baseline) override(group=None)
    # to drop the shared grouping.
    _OVERRIDE_ALLOWED = frozenset(
        {
            "default",
            "required",
            "min",
            "max",
            "exclusive_min",
            "exclusive_max",
            "choices",
            "nargs",
            "group",
            "cli_help",
            "cli_action",
            "cli_positional",
            "cli_flag",
            "cli_flag_help",
            "cli_aliases",
            "cli_extra_flags",
            "cli_short",
            "cli_off_flag",
            "cli_off_help",
            "cli_metavar",
            "cli_dest",
        }
    )

    def override(self, **kwargs: Any) -> Param:
        """Create a modified copy. For cross-module sharing with tweaks.
        Example: RESERVED_MEMORY_GB.override(default=10.0)

        Only fields in _OVERRIDE_ALLOWED may be overridden. Key identity fields
        (name, data_type) cannot be changed. Note that group IS overridable for
        modules that drop shared grouping (e.g. override(group=None)).
        """
        forbidden = set(kwargs) - self._OVERRIDE_ALLOWED
        if forbidden:
            raise ValueError(f"Cannot override fields: {forbidden}")
        return dataclasses.replace(self, **kwargs)


@dataclass
class ValidatorRef:
    """Cross-field validator reference.

    CLI: iterates validators after parsing, calls fn() — returns None (pass) or str (error).
    UI:  name is the association key linking cli/registry ↔ ui_props ↔ frontend.
    """

    name: str
    ## camelCase validator id (e.g. "productEqNumDevices"); three-way association key.

    fn: Callable[..., str | None]
    ## fn(params) or fn(params, provided) when wants_provided=True.
    ## Returns None (pass) or str (error message).
    ## ``provided`` = kebab-case names EXPLICITLY in argv (distinguishes "passed default" vs "not passed").

    wants_provided: bool = False
    ## Opt in to 2-arg invocation with ``provided`` set.


@dataclass
class ModuleSpec:
    """CLI-side module definition. One ModuleSpec per CLI module / frontend form.

    Does not contain version (schema version maintained by Web UI's ui_props,
    used for schema_registry version pinning).
    Does not contain i18n messages (maintained by ui_props.VALIDATOR_UI).
    """

    module_id: str
    ## Matches CLI module name and frontend moduleId.

    title: str
    ## English title for CLI --help; Frontend i18n in ui_props.TITLE.

    fields: list[Param]
    ## Order determines CLI --help parameter order.

    validators: list[ValidatorRef] = field(default_factory=list)
    ## CLI: calls fn() after parsing; Frontend: uses name only.

    # ── CLI metadata ────────────────────────────────────────────────
    cli_prog: str | None = None
    cli_description: str | None = None
    ## --help description; None = falls back to title.

    cli_examples: str | None = None
    cli_output_help: str | None = None

    log_options: bool = True
    ## Inject spec_cli.add_log_options() (--log-level/-v/-q/--debug/--log-file).

    log_options_after_group: str | None = None
    ## Insert log options after this group; None = "General Options".

    log_options_after_field: str | None = None
    ## Insert log options after this field; takes precedence over group.

    requires_model_id: bool = True
    ## Run cli.utils.require_model_id() after parsing.
