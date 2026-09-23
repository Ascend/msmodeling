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

"""Case-level validation helpers.

Shared between jobs.py (submit-time) and runners (execution-time).
Avoids circular imports by not depending on either.

This module provides:
- L1 declarative constraint validation (min/max/choices/pattern/required)
- L2 cross-field validator execution
- Type coercion for frontend-submitted params
- Provided set computation for wants_provided=True validators
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def check_declarative(
    spec: Any,
    params: dict[str, Any],
) -> str | None:
    """L1 declarative constraint backend validation (RFC §3.5).

    Called per-case (after multi-value expansion), so all values in ``params``
    are scalars. Checks each field's declarative constraints:
    - required: mandatory field presence
    - min/max: inclusive numeric range
    - exclusive_min/exclusive_max: exclusive numeric range
    - choices: enum value legality
    - pattern: regex match (string only)
    - max_length: string max length (string only)

    Args:
        spec: ModuleSpec from cli.registry
        params: Single case's scalar params dict (kebab-case keys, type-coerced)

    Returns:
        str: First failing constraint's error message
        None: All constraints pass
    """
    for field in spec.fields:
        val = params.get(field.name)

        # Handle None/empty values
        if val is None or val == "":
            if field.required:
                return f"Field '{field.name}' is required"
            continue  # Non-required and empty → skip further constraint checks

        # Numeric range constraints (integer/number)
        if field.data_type in ("integer", "number"):
            if not isinstance(val, (int, float)):
                return f"Field '{field.name}' must be a number, got {type(val).__name__}"

            if field.min is not None and val < field.min:
                return f"Field '{field.name}' must be ≥ {field.min}, got {val}"
            if field.max is not None and val > field.max:
                return f"Field '{field.name}' must be ≤ {field.max}, got {val}"
            if field.exclusive_min is not None and val <= field.exclusive_min:
                return f"Field '{field.name}' must be > {field.exclusive_min}, got {val}"
            if field.exclusive_max is not None and val >= field.exclusive_max:
                return f"Field '{field.name}' must be < {field.exclusive_max}, got {val}"

        # Enum constraints
        if field.choices is not None:
            allowed = [getattr(c, "value", c) for c in field.choices]
            # For array types (e.g. string[]), validate each element
            if field.data_type.endswith("[]"):
                if isinstance(val, list):
                    for item in val:
                        if item not in allowed:
                            allowed_str = ", ".join(repr(a) for a in allowed[:10])
                            suffix = ", ..." if len(allowed) > 10 else ""
                            return f"Field '{field.name}' elements must be one of [{allowed_str}{suffix}], got {item!r}"
            else:
                # Scalar type: validate the value itself
                if val not in allowed:
                    allowed_str = ", ".join(repr(a) for a in allowed[:10])
                    suffix = ", ..." if len(allowed) > 10 else ""
                    return f"Field '{field.name}' must be one of [{allowed_str}{suffix}], got {val!r}"

        # String constraints
        if field.data_type == "string":
            if not isinstance(val, str):
                return f"Field '{field.name}' must be a string, got {type(val).__name__}"

            if field.pattern is not None and not re.match(field.pattern, val):
                return f"Field '{field.name}' does not match pattern '{field.pattern}'"

            if field.max_length is not None and len(val) > field.max_length:
                return f"Field '{field.name}' length must be ≤ {field.max_length}, got {len(val)}"

    return None


def coerce_scalar(field: Any, val: Any) -> Any:
    """Coerce a single scalar value to Param.data_type's Python type.

    Frontend may send string representations of numbers (e.g. "42" for integer).
    Normalize before validation to avoid false negatives.

    Args:
        field: Param definition
        val: Raw value from frontend

    Returns:
        Coerced value, or original if coercion fails (L1 will report type error)
    """
    if val is None or val == "":
        return None
    try:
        if field.data_type == "integer":
            return int(val)
        elif field.data_type == "number":
            return float(val)
        elif field.data_type == "string":
            # Allow numeric values for string fields (e.g. input-length can be
            # int or YAML path). Convert to string for validation compatibility.
            if isinstance(val, (int, float)):
                return str(val)
    except (ValueError, TypeError):
        return val  # Coercion failed → keep original, L1 will report type error
    return val


def coerce_case_types(spec: Any, case: dict[str, Any]) -> dict[str, Any]:
    """Type-coerce a single case's params (scalar values after expansion).

    Args:
        spec: ModuleSpec from cli.registry
        case: Single case's raw params dict

    Returns:
        Type-coerced params dict
    """
    result = {}
    for field in spec.fields:
        val = case.get(field.name)
        result[field.name] = coerce_scalar(field, val)
    return result


def compute_provided_set(
    spec: Any,
    params: dict[str, Any],
    explicitly_touched: list[str] | None,
) -> set[str]:
    """Compute the ``provided`` set for wants_provided=True L2 validators.

    Job-level computation (shared across all cases). Prefers ``explicitly_touched``
    (frontend-sent list of user-interacted fields) over ``val != default`` fallback.

    Args:
        spec: ModuleSpec from cli.registry
        params: Raw job-level params (may contain multi-value arrays)
        explicitly_touched: Field IDs the user explicitly interacted with

    Returns:
        Set of kebab-case field names considered "explicitly provided"
    """
    if explicitly_touched:
        return set(explicitly_touched)

    # Fallback: val != default (any value in multi-value array differs from default)
    provided: set[str] = set()
    for field in spec.fields:
        val = params.get(field.name)
        if val is None or val == "":
            continue
        # For multi-value fields, check if any element differs from default
        if isinstance(val, list):
            if any(v != field.default for v in val if v is not None and v != ""):
                provided.add(field.name)
        else:
            coerced = coerce_scalar(field, val)
            if coerced is not None and coerced != field.default:
                provided.add(field.name)
    return provided


def validate_case(
    spec: Any,
    case_params: dict[str, Any],
    provided: set[str],
) -> tuple[str | None, str | None]:
    """Validate a single expanded case: L1 declarative + L2 cross-field.

    Args:
        spec: ModuleSpec from cli.registry
        case_params: Single case's scalar params dict (kebab-case keys, type-coerced)
        provided: Set of explicitly provided field names (job-level, shared)

    Returns:
        (error_message, validator_name): First failing validator's error and name
        (None, None): All validations pass
    """
    # L1: Declarative constraints
    l1_error = check_declarative(spec, case_params)
    if l1_error is not None:
        # L1 errors are field-level, extract field name from error message
        # Format: "Field 'xxx' ..."
        match = re.match(r"Field '([^']+)'", l1_error)
        field_name = match.group(1) if match else None
        return l1_error, field_name

    # L2: Cross-field validators
    for v in spec.validators:
        msg = v.fn(case_params, provided) if v.wants_provided else v.fn(case_params)
        if msg is not None:
            return msg, v.name

    return None, None


def validate_case_for_module(
    module_id: str,
    case_params: dict[str, Any],
    provided: set[str],
) -> tuple[str | None, list[str] | None]:
    """Validate a single case using CLI registry validators.

    This is the main entry point for runners. It looks up the module's spec,
    performs type coercion, runs L1+L2 validation, and returns the error
    message along with the associated field IDs (from VALIDATOR_UI) for
    frontend highlighting.

    Args:
        module_id: The module identifier (e.g. "text_generate")
        case_params: Single case's raw params dict (kebab-case keys)
        provided: Set of explicitly provided field names (job-level)

    Returns:
        (error_message, fields): Error details if validation fails
        (None, None): If validation passes or module is unknown
    """
    from cli.registry.modules import get_spec
    from services.ui_props import get_validator_ui

    try:
        spec = get_spec(module_id)
    except KeyError:
        # Unknown module: log warning and skip validation
        logger.warning("Unknown module_id=%s, skipping validation", module_id)
        return None, None

    # Type coercion
    typed_case = coerce_case_types(spec, case_params)

    # L1 + L2 validation
    error, validator_name = validate_case(spec, typed_case, provided)

    if error is None:
        return None, None

    # Lookup associated fields from VALIDATOR_UI
    fields = None
    if validator_name:
        validator_ui = get_validator_ui(module_id)
        if validator_name in validator_ui:
            fields = validator_ui[validator_name].get("fields")

    return error, fields
