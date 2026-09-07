"""JSON adapter — generate frontend form JSON from ModuleSpec + UI config.

Merges Param (CLI-side) with UIFieldProps (UI-side) to produce frontend form JSON.
UIFieldProps can override data_type/choices/default/required for divergence.

CLI: ``--generate`` regenerates the form bundle; ``--check`` diffs against bundle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cli.registry.datatypes import (
    ModuleSpec,
    Param,
)
from .ui_props import I18nText, UIFieldProps, _UNSET

# Default bundle dir (var/config/forms) for --check; sibling of services/.
_DEFAULT_BUNDLE_DIR = Path(__file__).resolve().parent.parent / "var" / "config"


# ── Module-level Web UI configuration ─────────────────────────────


@dataclass
class FormConfig:
    """Module's Web UI configuration, assembled by ui_props module file."""

    version: str
    ## Schema version (for schema_registry version pinning + refuse-on-mismatch).

    title: I18nText
    ## Module title (bilingual). Frontend header.

    runner: str
    ## Frontend runner class name (e.g. "ModelRunner", "ParallelRunner").

    ui: dict[str, UIFieldProps] = field(default_factory=dict)
    ## Per-field UI attributes, keyed by Param.name.
    ## Every Param should have an entry (UT enforced); missing entries fall back
    ## to empty UIFieldProps (defensive fallback, shouldn't trigger in prod).

    group_labels: dict[str, I18nText] = field(default_factory=dict)
    ## CLI English group name → bilingual label.
    ## Example: {"General Options": I18nText("通用选项", "General Options")}

    groups: list[dict] = field(default_factory=list)
    ## Group collapse/expand metadata (frontend-only).

    option_source_registry: dict = field(default_factory=dict)
    ## Dynamic option sources (e.g. {"devices": {"endpoint": "/api/options/devices", "cache": "session"}}).

    validator_ui: dict[str, dict] = field(default_factory=dict)
    ## Cross-field validator UI mappings.
    ## Key: validator name (camelCase, e.g. "productEqNumDevices").
    ## Value: {"fields": [...], "message": I18nText(...)}.


# ── Main entry ────────────────────────────────────────────────────


def export_form_json(spec: ModuleSpec, cfg: FormConfig) -> dict:
    """Generate frontend form JSON from ModuleSpec + FormConfig.

    Args:
        spec: Module specification with fields.
        cfg: Form configuration with UI properties.

    Returns:
        Frontend form JSON dict.
    """
    fields = [_build_field(p, cfg) for p in spec.fields]
    spec_names = {p.name for p in spec.fields}
    for fid, ui in cfg.ui.items():
        if fid in spec_names:
            continue  # regular field, built from Param above
        f = _build_convention_field(fid, ui, cfg)
        if f:
            fields.append(f)

    return {
        "$schema": "form-schema/v1",
        "moduleId": spec.module_id,
        "title": _i18n(cfg.title),
        "version": cfg.version,
        "runner": cfg.runner,
        "optionSourceRegistry": cfg.option_source_registry,
        "formValidation": _build_form_validation(spec, cfg),
        "groups": cfg.groups,
        "fields": fields,
    }


# ── Internal builders ─────────────────────────────────────────────


def _build_convention_field(fid: str, ui, cfg: FormConfig) -> dict | None:
    """Build frontend field dict for UI-only field (no Param).

    Args:
        fid: Field ID.
        ui: UIFieldProps.
        cfg: Form configuration.

    Returns:
        Field dict or None if hidden.
    """
    if getattr(ui, "hidden", False):
        return None
    f = {"id": fid, "control": ui.control or "text", "dataType": ui.data_type or "string"}
    if ui.label:
        f["label"] = _i18n(ui.label)
    if ui.tooltip:
        f["tooltip"] = _i18n(ui.tooltip)
    if ui.placeholder:
        f["placeholder"] = _i18n(ui.placeholder)
    if ui.default is not _UNSET and ui.default is not None:
        f["default"] = ui.default
    if ui.group:
        f["group"] = ui.group  # already i18n-shaped in ui_props
    if ui.option_source:
        f["optionSource"] = ui.option_source
    if ui.required:
        label = ui.label
        zh = f"{label.zh}为必填项" if label else f"{fid}为必填项"
        en = f"{label.en} is required" if label else f"{fid} is required"
        f["validation"] = [{"rule": "required", "message": {"zh": zh, "en": en}, "trigger": ["change", "blur"]}]
    return f


def _build_form_validation(spec: ModuleSpec, cfg: FormConfig) -> list[dict]:
    """Build formValidation array from cross-field validators.

    Args:
        spec: Module specification.
        cfg: Form configuration.

    Returns:
        List of validator rule dicts.
    """
    result = []
    for v in spec.validators:
        vu = cfg.validator_ui.get(v.name, {})
        msg = vu.get("message")
        # msg is I18nText → convert to dict; else use v.name as fallback
        message = _i18n(msg) if isinstance(msg, I18nText) else {"zh": v.name, "en": v.name}
        result.append(
            {
                "rule": "validator",
                "value": v.name,
                "message": message,
            }
        )
    return result


def _infer_control(data_type: str, choices: list | None, option_source: dict | None) -> str:
    """Auto-infer frontend control type from data_type and options.

    Mapping:
    - boolean → "switch"
    - string[] with choices/option_source → "multi-select"
    - string[] without choices → "text" (comma-separated)
    - integer[] / number[] → "text" (comma-separated)
    - string/integer/number with choices/option_source → "select"
    - string/integer/number without choices → "text"
    """
    if data_type == "boolean":
        return "switch"
    if data_type.endswith("[]"):
        # Array types
        if choices or option_source:
            return "multi-select"
        return "text"  # Comma-separated input
    # Scalar types
    if choices or option_source:
        return "select"
    if data_type in ("integer", "number"):
        return "number"  # numeric spinner (el-input-number), matches baseline forms
    return "text"


def _build_field(p: Param, cfg: FormConfig) -> dict:
    """Build frontend field dict from Param + UIFieldProps.

    Args:
        p: Parameter definition.
        cfg: Form configuration.

    Returns:
        Field dict.
    """
    ui = cfg.ui.get(p.name, UIFieldProps())

    # data_type / choices / default / required may be overridden by UIFieldProps
    data_type = ui.data_type or p.data_type
    choices = ui.choices if ui.choices is not None else p.choices
    default = p.default if ui.default is _UNSET else ui.default
    required = ui.required if ui.required is not None else p.required

    field_dict: dict = {"id": p.name, "dataType": data_type, "default": default}

    # i18n attributes (with fallback)
    _set_if(field_dict, "label", _i18n_or(ui.label, I18nText(p.name, p.name)))
    _set_if(
        field_dict,
        "tooltip",
        _i18n_or(ui.tooltip, I18nText(p.cli_help, p.cli_help) if p.cli_help else None),
    )
    _set_if(field_dict, "placeholder", _i18n(ui.placeholder) if ui.placeholder else None)
    # Group: per-field ui.group override wins (baseline parity — a Param group key may
    # need a different i18n label per module); else map Param.group via GROUP_LABELS.
    if ui.group:
        field_dict["group"] = ui.group
    else:
        _set_if(field_dict, "group", _resolve_group(p.group, cfg.group_labels))

    # UI control attributes (auto-infer from data_type if not explicitly set)
    control = ui.control or _infer_control(data_type, choices, ui.option_source)
    field_dict["control"] = control
    _set_if(field_dict, "optionSource", ui.option_source)
    _set_if(field_dict, "disabled", True if ui.disabled else None)
    _set_if(field_dict, "hidden", True if ui.hidden else None)
    _set_if(field_dict, "conditions", ui.conditions)
    _set_if(field_dict, "multiValues", True if ui.multi_values else None)
    _set_if(field_dict, "clearable", True if ui.clearable else None)

    # Options list (choices override takes priority)
    # Output [{value, label}] format compatible with existing TS form config.
    # StrEnum: .name as label (e.g. "W8A8_DYNAMIC"), .value as value (kebab-cased).
    # Plain strings: value=label=string itself.
    if choices:
        field_dict["options"] = _build_options(choices)

    # Declarative constraints → async-validator rules
    label_zh = field_dict.get("label", {}).get("zh", p.name)
    label_en = field_dict.get("label", {}).get("en", p.name)
    field_dict["validation"] = _build_validation(p, label_zh, label_en, required, data_type)
    return field_dict


def _build_validation(
    p: Param, label_zh: str, label_en: str, required: bool = False, data_type: str = "string"
) -> list[dict]:
    """Build async-validator rules from Param constraints.

    Args:
        p: Parameter definition.
        label_zh: Chinese label.
        label_en: English label.
        required: Whether field is required.
        data_type: Field data type.

    Returns:
        List of validation rule dicts.
    """
    # Numeric range rules need type 'number' (or 'integer' — 'number' accepts both).
    range_type = "number" if data_type in ("integer", "number") else None
    # Array fields: type 'array' so required checks non-empty list (device multi-select).
    if data_type.endswith("[]"):
        range_type = "array"
    rules = []
    if required:
        rule: dict = {"rule": "required", "message": {"zh": f"{label_zh}为必填项", "en": f"{label_en} is required"}}
        # For array types, required needs type:'array' to check non-empty list
        # (otherwise async-validator treats any non-None value as "present").
        if data_type.endswith("[]"):
            rule["type"] = "array"
        # NOTE: deliberately no `type: "number"` here — text controls store
        # strings; type check would reject "4" as a string-vs-number mismatch.
        rules.append(rule)
    if p.min is not None:
        _r: dict = {"rule": "min", "value": p.min, "message": {"zh": f"必须 ≥ {p.min}", "en": f"must be ≥ {p.min}"}}
        if range_type:
            _r["type"] = range_type
        rules.append(_r)
    if p.max is not None:
        _r: dict = {"rule": "max", "value": p.max, "message": {"zh": f"必须 ≤ {p.max}", "en": f"must be ≤ {p.max}"}}
        if range_type:
            _r["type"] = range_type
        rules.append(_r)
    if p.exclusive_min is not None:
        _g: dict = {
            "rule": "gt",
            "value": p.exclusive_min,
            "message": {"zh": f"必须 > {p.exclusive_min}", "en": f"must be > {p.exclusive_min}"},
        }
        if range_type and range_type != "array":
            _g["type"] = range_type
        rules.append(_g)
    if p.exclusive_max is not None:
        _l: dict = {
            "rule": "lt",
            "value": p.exclusive_max,
            "message": {"zh": f"必须 < {p.exclusive_max}", "en": f"must be < {p.exclusive_max}"},
        }
        if range_type and range_type != "array":
            _l["type"] = range_type
        rules.append(_l)
    if p.pattern:
        rules.append({"rule": "pattern", "value": p.pattern, "message": {"zh": "格式不匹配", "en": "format mismatch"}})
    if p.max_length is not None:
        # async-validator has no 'length' rule type — a string max-length check is
        # expressed as type:'string' + max (chars). 'type' must be a REAL async-validator
        # type; emitting 'length' crashes Schema.validate with "Unknown rule type".
        rules.append(
            {
                "rule": "max",
                "type": "string",
                "value": p.max_length,
                "message": {
                    "zh": f"长度不能超过 {p.max_length} 字符",
                    "en": f"length must be ≤ {p.max_length} characters",
                },
            }
        )
    return rules


# ── Helpers ───────────────────────────────────────────────────────


def _build_options(choices: list) -> list[dict]:
    """Convert choices to [{value, label}] format.

    Args:
        choices: List of Enum members, dicts, or strings.

    Returns:
        List of option dicts.
    """
    opts = []
    for c in choices:
        if isinstance(c, dict):
            # Pre-formatted — pass through directly
            opts.append(c)
        elif hasattr(c, "value") and hasattr(c, "name"):
            # Enum member — use native value to match CLI registry encoding.
            # The CLI argparse type accepts both native and kebab-case spellings,
            # so sending native values keeps the form consistent with the enum
            # definitions (e.g. QuantizeLinearAction.W8A8_DYNAMIC → "W8A8_DYNAMIC").
            opts.append({"value": c.value, "label": c.name})
        else:
            # Plain string
            opts.append({"value": c, "label": str(c)})
    return opts


def _i18n(t: I18nText | None) -> dict | None:
    """Convert I18nText to {zh, en} dict."""
    return {"zh": t.zh, "en": t.en} if t else None


def _i18n_or(t: I18nText | None, fallback: I18nText | None) -> dict | None:
    """Convert I18nText to dict, using fallback if unset."""
    return _i18n(t) or _i18n(fallback)


def _resolve_group(group: str | None, labels: dict[str, I18nText]) -> dict | None:
    """Convert CLI English group name to i18n dict."""
    if not group:
        return None
    i18n = labels.get(group)
    return _i18n(i18n) if i18n else {"zh": group, "en": group}


def _set_if(d: dict, key: str, value: Any) -> None:
    """Write to dict only when value is not None."""
    if value is not None:
        d[key] = value


# ── CLI entry point (for testing / generation) ────────────────────

# Modules whose form JSON lives in var/config/forms/ (the schema_registry
# bundle). image_generate is registry-complete (23 Params + ui_props) but
# CLI-only: web_ui/backend/runners/registry.py has no ImageGenerateRunner
# adapter, so exposing a form would submit jobs that cannot execute.
# Wiring it up needs a runner adapter first (see docs/design/rfc_gap_analysis.md).
BUNDLED_MODULES = ("text_generate", "throughput_optimizer", "video_generate")


def check_module(module_id: str, bundle_dir: Path | None = None) -> list[str]:
    """Diff generated form JSON against bundled JSON file.

    Args:
        module_id: Module identifier.
        bundle_dir: Bundle directory (default: var/config).

    Returns:
        List of drift descriptions; empty if in sync.
    """
    import tempfile

    from services.schema_registry import canonical_json_bytes, generate_form_configs

    if module_id not in BUNDLED_MODULES:
        return [f"module '{module_id}' has no bundled form JSON (not in BUNDLED_MODULES)"]

    path = (bundle_dir or _DEFAULT_BUNDLE_DIR) / "forms" / f"{module_id}.json"
    if not path.exists():
        return [f"bundle file missing: {path}"]
    on_disk = json.loads(path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp:
        generate_form_configs(Path(tmp))
        gen_path = Path(tmp) / "forms" / f"{module_id}.json"
        generated = json.loads(gen_path.read_text(encoding="utf-8"))

    if canonical_json_bytes(on_disk) == canonical_json_bytes(generated):
        return []

    diffs: list[str] = []
    gen_version = generated.get("version")
    disk_version = on_disk.get("version")
    if gen_version != disk_version:
        diffs.append(
            f"version: bundle={disk_version} registry={gen_version} "
            f"(bump ui_props.{module_id}.VERSION when registry/ui_props content changes)"
        )
    gen_fields = {f["id"]: f for f in generated.get("fields", [])}
    disk_fields = {f["id"]: f for f in on_disk.get("fields", [])}
    for fid in sorted(set(gen_fields) - set(disk_fields)):
        diffs.append(f"field '{fid}' generated but missing from bundle")
    for fid in sorted(set(disk_fields) - set(gen_fields)):
        diffs.append(f"field '{fid}' in bundle but not generated (stale?)")
    for fid in sorted(set(gen_fields) & set(disk_fields)):
        if canonical_json_bytes(gen_fields[fid]) != canonical_json_bytes(disk_fields[fid]):
            diffs.append(f"field '{fid}' content differs")
    if not diffs:
        # Field-level comparison found nothing but the envelopes differ
        # (groups/formValidation/metadata drift).
        diffs.append("envelope metadata differs (groups/formValidation/title/...)")
    return diffs


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    parser = argparse.ArgumentParser(
        description="JSON adapter CLI: generate / check form JSONs against the registry",
    )
    parser.add_argument("--generate", action="store_true", help="(Re)generate all bundled form JSONs")
    parser.add_argument("--check", action="store_true", help="Verify bundled JSONs match the registry; exit 1 on drift")
    args = parser.parse_args()

    if args.generate:
        from services.schema_registry import generate_form_configs

        dst = generate_form_configs()
        print(f"[json_adapter] regenerated bundle at {dst}")
        sys.exit(0)

    if args.check:
        from pathlib import Path as _P

        root = _P(__file__).resolve().parents[1]
        bundle = root / "var" / "config"
        failures = 0
        for module_id in BUNDLED_MODULES:
            diffs = check_module(module_id, bundle)
            if diffs:
                failures += 1
                print(f"[json_adapter] DRIFT {module_id}:")
                for d in diffs:
                    print(f"  - {d}")
            else:
                print(f"[json_adapter] OK {module_id}")
        if failures:
            print(
                f"[json_adapter] {failures} module(s) out of sync — run "
                f"'python -m services.json_adapter --generate' and bump ui_props VERSION",
                file=sys.stderr,
            )
            sys.exit(1)
        sys.exit(0)

    parser.print_help()
