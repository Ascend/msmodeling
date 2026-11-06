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


"""Assemble validated agent values into execution fields for preflight and runtime."""

import json
from copy import deepcopy

from ...config.config import OptimizerConfigField


def resolve_execution_fields(params: dict, target_fields: list, context: dict | None = None) -> list:
    """Keep configured field contracts and add active-space service fields.

    ``params`` is the full output of validate_candidate: active-space defaults,
    pinned constants and explicit candidate values have already been resolved.
    This function deliberately does not implement a second default policy.
    """
    space = (context or {}).get("search_space", {})
    schema = {item["name"]: item for group in ("parameters", "constants") for item in space.get(group, [])}
    resolved = []
    covered_names = set()
    for field in target_fields:
        active = deepcopy(field)
        item = schema.get(active.name)
        if item is not None:
            # Round-local domains govern tunable fields after TOML is frozen.
            # Keep configured types, positions and derivation contracts; candidate
            # validation checks that TOML constants cannot be unfrozen.
            definition = active.model_dump()
            definition.update(min=item["min"], max=item["max"])
            if active.dtype == "enum":
                definition["dtype_param"] = item["choices"]
            if active.constant is None:
                definition["constant"] = item.get("constant")
            active = OptimizerConfigField(**definition)
        covered_names.add(active.name)
        if active.name in params:
            value = params[active.name]
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            active.apply_value(value)
        resolved.append(active)

    for name, value in params.items():
        if name in covered_names:
            continue
        item = schema.get(name)
        if item is None:
            continue  # Candidate validation rejects unregistered names.
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        kwargs = {
            "name": name,
            "value": value,
            "dtype": item.get("dtype", "str"),
            "config_position": item.get("config_position", "run"),
            "min": item.get("min", 0.0),
            "max": item.get("max", 1.0),
            "dtype_param": item.get("choices") if item.get("dtype") == "enum" else item.get("dtype_param"),
        }
        if item.get("constant") is not None:
            kwargs["constant"] = item["constant"]
        # Invalid field definitions must fail preflight instead of silently
        # becoming untyped fields with different execution semantics.
        active = OptimizerConfigField(**kwargs)
        active.apply_value(value)
        resolved.append(active)
    return resolved


def validate_service_parameters(context, params, *, fields=None, others=None):
    """Plan effective service fields for preflight using the runtime renderer."""
    import re
    from types import SimpleNamespace
    from ...serve_plan import render_serve_flags, substitute_service_placeholders, validate_service_positions
    from .pd import render_pd_role_flags

    if context.get("engine") == "mindie":
        return
    if fields is None:
        space = context.get("search_space", {})
        fields = [item for group in ("parameters", "constants") for item in space.get(group, [])]
    resolved = []
    for item in fields:
        name = item["name"]
        value = params.get(name, item.get("default"))
        position = item.get("config_position", "run")
        if position == "run" and value is not None:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            resolved.append(SimpleNamespace(name=name, value=value, config_position=position))
    if others is None:
        others = ""
        source = context.get("source", {}).get("config_toml")
        if source:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            with open(source, "rb") as stream:
                config = tomllib.load(stream)
            section = config.get(context.get("engine", "vllm"), {})
            others = section.get("command", {}).get("others", "") or ""
    is_pd = context.get("deployment", {}).get("type") == "pd"
    position_fields = [
        SimpleNamespace(name=item["name"], config_position=item.get("config_position", "run")) for item in fields
    ]
    validate_service_positions(position_fields, others=None if is_pd else others)
    substitutions = {}
    for item in fields:
        value = params.get(item["name"], item.get("default"))
        if value is not None:
            substitutions[item["name"].upper()] = value
    substitutions.update({name.upper(): value for name, value in params.items() if value is not None})
    if is_pd:
        others = re.sub(
            r"\$([A-Za-z_][A-Za-z0-9_]*)",
            lambda match: str(substitutions[match[1]]) if match[1] in substitutions else match[0],
            others,
        )
    else:
        values = [SimpleNamespace(name=name, value=value) for name, value in substitutions.items()]
        others = substitute_service_placeholders(others, values)
    if is_pd:
        for role in ("prefill", "decode"):
            try:
                render_pd_role_flags(resolved, role, others)
            except ValueError as exc:
                raise ValueError(f"{role}: {exc}") from exc
    else:
        render_serve_flags(resolved, others)
