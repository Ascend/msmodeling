"""Integrity tests for CLI registry ↔ Web UI ui_props consistency (RFC §3.6).

These tests guard against silent drift between the CLI registry (single source
of truth for parameters and validators) and the Web UI layer (UIFieldProps +
VALIDATOR_UI). Without these tests, the following failures go unnoticed:

- cli/registry adds a new Param but ui_props forgets the UIFieldProps entry
  → frontend falls back to raw snake_case id as label
- ValidatorRef.name has a typo vs VALIDATOR_UI key (e.g. "productEqNumDevice"
  vs "productEqNumDevices")
- VALIDATOR_UI["someValidator"].fields references a field id that doesn't
  exist in spec.fields
- UIFieldProps.choices contains values not in Param.choices
  → frontend offers options CLI will reject at submit time (400)

CI runs these on every commit; failures block the merge.
"""

from __future__ import annotations

import pytest

from cli.registry.modules import get_spec
from services.json_adapter import BUNDLED_MODULES
from services.ui_props import get_ui, get_validator_ui


@pytest.mark.parametrize("module_id", BUNDLED_MODULES)
def test_all_fields_have_ui_props(module_id):
    """Every field in spec.fields must have a UIFieldProps entry in the UI dict.

    Forces developers to explicitly consider UI attributes (i18n label/tooltip,
    etc.) for every field, preventing the frontend from falling back to the raw
    snake_case id as the label when a new field is added without UI config.

    Notes:
    - Fields that don't need UI exposure must still have an entry (set
      hidden=True).
    - Even fields needing no UI customization require an empty UIFieldProps().
    - json_adapter has fallback logic (uses Param defaults when a field is
      missing from UI dict), but this test enforces explicit configuration —
      the fallback is only for transient development states.
    """
    spec = get_spec(module_id)
    ui = get_ui(module_id)
    for p in spec.fields:
        assert p.name in ui, (
            f"Field '{p.name}' in {module_id} has no UIFieldProps entry. "
            f"Add it to UI dict (set hidden=True if CLI-only, "
            f"or use empty UIFieldProps() if no customization needed)."
        )


@pytest.mark.parametrize("module_id", BUNDLED_MODULES)
def test_validator_has_ui_entry(module_id):
    """Every ValidatorRef must have a matching entry in VALIDATOR_UI.

    Catches typos in ValidatorRef.name or missing VALIDATOR_UI entries, which
    would leave the frontend unable to display cross-field validation messages.
    """
    spec = get_spec(module_id)
    validator_ui = get_validator_ui(module_id)
    for v in spec.validators:
        assert v.name in validator_ui, (
            f"ValidatorRef '{v.name}' has no VALIDATOR_UI entry in {module_id}. "
            f"Add it to VALIDATOR_UI dict with 'fields' and 'message' keys."
        )


@pytest.mark.parametrize("module_id", BUNDLED_MODULES)
def test_validator_ui_fields_exist(module_id):
    """Field ids referenced by VALIDATOR_UI['...'].fields must exist in spec.fields.

    Catches typos in VALIDATOR_UI.fields that would prevent the frontend from
    highlighting the relevant fields when a cross-field validation fails.
    """
    spec = get_spec(module_id)
    validator_ui = get_validator_ui(module_id)
    field_ids = {p.name for p in spec.fields}
    for v in spec.validators:
        vu = validator_ui.get(v.name)
        if not vu:
            continue  # Caught by test_validator_has_ui_entry
        for f in vu.get("fields", []):
            assert f in field_ids, (
                f"VALIDATOR_UI['{v.name}'].fields references unknown field "
                f"'{f}' in {module_id}. Available fields: {sorted(field_ids)}"
            )


@pytest.mark.parametrize("module_id", BUNDLED_MODULES)
def test_ui_choices_is_subset_of_param_choices(module_id):
    """UIFieldProps.choices must be a subset of Param.choices.

    Prevents the frontend from offering options the CLI would reject at submit
    time with a 400 error.
    """
    spec = get_spec(module_id)
    ui = get_ui(module_id)

    def _value(c):
        """Extract comparable value from a choice entry.

        Choices may be: dict ({value, label}), Enum member (.value), or plain string.
        """
        if isinstance(c, dict):
            return c.get("value")
        return getattr(c, "value", c)

    for p in spec.fields:
        u = ui.get(p.name)
        if u and u.choices is not None and p.choices is not None:
            param_vals = {_value(c) for c in p.choices}
            ui_vals = {_value(c) for c in u.choices}
            extra = ui_vals - param_vals
            assert not extra, (
                f"UIFieldProps.choices for '{p.name}' in {module_id} contains values not in Param.choices: {extra}"
            )
