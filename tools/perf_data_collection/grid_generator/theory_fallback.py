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

"""Generic theory fallback for explicitly selected replay operators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .generators.base import TheoryShapeRow
from .theory_router import get_default_theory_generator
from .utils import (
    INPUT_SHAPES_COLUMN,
    OUTPUT_SHAPES_COLUMN,
    align_shape_slot_count,
    build_generated_row,
    parse_shape_text,
    profile_dedupe_key,
)


RUNTIME_SOURCE_PROFILE = "Runtime source_profile"


def theory_generation_is_skipped(
    kernel_type: str,
    config: dict[str, Any],
    op_meta: dict[str, dict[str, Any]],
) -> bool:
    """Return whether the generic theory router intentionally skips an operator."""
    if config.get("assignments", {}).get(kernel_type) == "skip":
        return True
    metadata = op_meta.get(kernel_type, {})
    return any(
        bool(metadata.get(field))
        for field in ("zero_cost", "composite", "communication")
    )


def _split_metadata_slots(value: str) -> list[str]:
    raw = str(value or "").strip().strip('"')
    if not raw:
        return []
    return [part.strip() for part in raw.split(";")]


def _clear_absent_shape_slots(
    shapes: list[tuple[int, ...]],
    dtype_cell: str,
    format_cell: str,
) -> list[tuple[int, ...]]:
    absent_values = {"", "NULL", "NONE", "UNDEFINED", "DT_UNDEFINED"}
    dtypes = _split_metadata_slots(dtype_cell)
    formats = _split_metadata_slots(format_cell)
    sanitized = list(shapes)
    for index in range(len(sanitized)):
        dtype_absent = index < len(dtypes) and dtypes[index].upper() in absent_values
        format_absent = index < len(formats) and formats[index].upper() in absent_values
        if dtype_absent or format_absent:
            sanitized[index] = ()
    return sanitized


def _theory_row_to_csv(
    row: TheoryShapeRow,
    *,
    headers: list[str],
    template_row: dict[str, str],
) -> dict[str, str]:
    template_inputs = parse_shape_text(template_row.get(INPUT_SHAPES_COLUMN, ""))
    template_output_text = template_row.get(OUTPUT_SHAPES_COLUMN, "")
    template_outputs = parse_shape_text(template_output_text) if template_output_text else []
    extra_values = {
        key: value
        for key, value in row.extra_values.items()
        if key in headers and key != RUNTIME_SOURCE_PROFILE
    }
    input_dtype_cell = extra_values.get(
        "Input Data Types",
        template_row.get("Input Data Types", ""),
    )
    input_format_cell = extra_values.get(
        "Input Formats",
        template_row.get("Input Formats", ""),
    )
    output_dtype_cell = extra_values.get(
        "Output Data Types",
        template_row.get("Output Data Types", ""),
    )
    output_format_cell = extra_values.get(
        "Output Formats",
        template_row.get("Output Formats", ""),
    )
    input_shapes = _clear_absent_shape_slots(
        align_shape_slot_count(template_inputs, row.input_shapes),
        input_dtype_cell,
        input_format_cell,
    )
    output_shapes = _clear_absent_shape_slots(
        align_shape_slot_count(template_outputs, row.output_shapes),
        output_dtype_cell,
        output_format_cell,
    )
    generated = build_generated_row(
        headers,
        template_row,
        input_shapes,
        output_shapes,
        extra_values=extra_values,
    )
    if RUNTIME_SOURCE_PROFILE in headers:
        generated[RUNTIME_SOURCE_PROFILE] = ""
    return generated


def build_theory_fallback_rows(
    *,
    kernel_type: str,
    model_names: list[str],
    config: dict[str, Any],
    op_meta: dict[str, dict[str, Any]],
    csv_path: Path,
    headers: list[str],
    source_rows: list[dict[str, str]],
    row_limit: int,
) -> tuple[list[dict[str, str]], dict[str, int]] | None:
    """Build up to ``row_limit`` new generic rows after database de-duplication."""
    generator = get_default_theory_generator(kernel_type, model_names, config, op_meta)
    if generator is None:
        return None
    if not source_rows:
        raise ValueError(f"{csv_path} does not contain a template data row")

    seen = {profile_dedupe_key(csv_path, headers, row) for row in source_rows}
    generated_rows: list[dict[str, str]] = []
    duplicate_count = 0
    attempted = 0
    for theory_row in generator:
        attempted += 1
        generated = _theory_row_to_csv(
            theory_row,
            headers=headers,
            template_row=source_rows[0],
        )
        key = profile_dedupe_key(csv_path, headers, generated)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        generated_rows.append(generated)
        if len(generated_rows) >= row_limit:
            break
    return generated_rows, {
        "attempted": attempted,
        "duplicates": duplicate_count,
        "appended": len(generated_rows),
    }
