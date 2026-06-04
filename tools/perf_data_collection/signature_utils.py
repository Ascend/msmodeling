"""Canonical signature utilities for perf database CSV rows.

Shape-grid generation and microbench backfill both need to decide whether two
profiling rows describe the same operator case. This module keeps that matching
logic in one place, including operator-specific normalization for MatMul-family
aliases, parameter slots, and DispatchFFNCombine EP size.
"""

from __future__ import annotations

import re

DISPATCH_FFN_OP = "DispatchFFNCombine"
MATMUL_FAMILY_OPS = {"MatMulV2", "MatMulV3", "MatMulCommon"}


def normalize_op_name(name: str) -> str:
    normalized = name.strip()
    if normalized.endswith("_run.py"):
        normalized = normalized.removesuffix("_run.py")
    elif normalized.endswith("_run"):
        normalized = normalized.removesuffix("_run")
    elif normalized.endswith(".csv"):
        normalized = normalized.removesuffix(".csv")
    return normalized


def _split_slot_cell(value: str) -> list[str]:
    cleaned = (value or "").strip().strip('"')
    if not cleaned:
        return []
    return [part.strip().strip('"') for part in cleaned.split(";")]


def _trim_trailing_empty(values: list[str]) -> list[str]:
    result = list(values)
    while result and result[-1] == "":
        result.pop()
    return result


def _normalize_shape_slot(slot: str) -> str:
    cleaned = (slot or "").strip().strip('"').strip()
    if cleaned in {"()", "N/A", "NA", "NULL", "None", "none"}:
        return ""
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
    return ",".join(part.strip() for part in re.split(r"[,\s]+", cleaned) if part.strip())


def _normalize_shape_attr_sig(
    shapes_text: str,
    attr_text: str,
) -> tuple[str, str]:
    shape_slots = [_normalize_shape_slot(slot) for slot in _split_slot_cell(shapes_text)]
    attr_slots = _split_slot_cell(attr_text)
    slot_count = max(len(shape_slots), len(attr_slots))
    shape_slots += [""] * (slot_count - len(shape_slots))
    attr_slots += [""] * (slot_count - len(attr_slots))
    normalized_attrs: list[str] = []
    for index in range(slot_count):
        normalized_attrs.append(attr_slots[index] if shape_slots[index] else "")
    return (
        ";".join(_trim_trailing_empty(shape_slots)),
        ";".join(_trim_trailing_empty(normalized_attrs)),
    )


def _parse_shape_slot(slot: str) -> tuple[int, ...] | None:
    cleaned = _normalize_shape_slot(slot)
    if not cleaned:
        return None
    try:
        return tuple(int(part) for part in cleaned.split(",") if part)
    except ValueError:
        return None


def _format_shape_slot(shape: tuple[int, ...]) -> str:
    return ",".join(str(dim) for dim in shape)


def is_matmul_family(op_name: str) -> bool:
    return normalize_op_name(op_name) in MATMUL_FAMILY_OPS


def canonicalize_matmul_family_signature(
    row: dict[str, str],
) -> tuple[str, str, str, str] | None:
    input_shapes = [_parse_shape_slot(slot) for slot in _split_slot_cell(row.get("Input Shapes", ""))]
    output_shapes = [_parse_shape_slot(slot) for slot in _split_slot_cell(row.get("Output Shapes", ""))]

    if len(input_shapes) < 2 or not input_shapes[0] or not input_shapes[1]:
        return None
    a_shape, b_shape = input_shapes[0], input_shapes[1]
    if len(a_shape) != 2 or len(b_shape) != 2:
        return None

    out_shape = output_shapes[0] if output_shapes else None
    if not out_shape or len(out_shape) != 2:
        return None

    m_dim, n_dim = out_shape
    k_dim: int | None = None
    if a_shape[0] == m_dim:
        if b_shape[0] == n_dim and a_shape[1] == b_shape[1]:
            k_dim = a_shape[1]
        elif b_shape[1] == n_dim and a_shape[1] == b_shape[0]:
            k_dim = a_shape[1]

    if k_dim is None:
        common_dims = set(a_shape) & set(b_shape)
        non_output_common = [dim for dim in common_dims if dim not in {m_dim, n_dim}]
        if len(non_output_common) == 1:
            k_dim = non_output_common[0]
        elif len(common_dims) == 1:
            k_dim = next(iter(common_dims))

    if k_dim is None:
        return None

    input_dtypes = _split_slot_cell(row.get("Input Data Types", ""))
    input_formats = _split_slot_cell(row.get("Input Formats", ""))
    canonical_input_shapes = f"{m_dim},{k_dim};{n_dim},{k_dim}"
    canonical_output_shapes = f"{m_dim},{n_dim}"
    canonical_input_dtypes = ";".join(input_dtypes[:2])
    canonical_input_formats = ";".join(input_formats[:2])
    return (
        canonical_input_shapes,
        canonical_input_dtypes,
        canonical_input_formats,
        canonical_output_shapes,
    )


def canonicalize_profile_signature(
    row: dict[str, str],
    op_name: str | None = None,
) -> tuple[str, str, str]:
    resolved_op_name = normalize_op_name(
        (op_name or row.get("OP Type", "") or row.get("OP State", "") or "").strip().strip('"')
    )
    input_shapes = _split_slot_cell(row.get("Input Shapes", ""))
    input_dtypes = _split_slot_cell(row.get("Input Data Types", ""))
    input_formats = _split_slot_cell(row.get("Input Formats", ""))

    def keep_input_slots(indices: list[int]) -> None:
        nonlocal input_shapes, input_dtypes, input_formats
        input_shapes = [input_shapes[index] for index in indices if index < len(input_shapes)]
        input_dtypes = [input_dtypes[index] for index in indices if index < len(input_dtypes)]
        input_formats = [input_formats[index] for index in indices if index < len(input_formats)]

    if resolved_op_name == "Index":
        output_slots = _split_slot_cell(row.get("Output Shapes", ""))
        output = _parse_shape_slot(output_slots[0]) if output_slots else None
        if output and input_shapes:
            input_shapes = [input_shapes[0], _format_shape_slot((output[0],))]
            input_dtypes = [input_dtypes[0], input_dtypes[-1]] if input_dtypes else []
            input_formats = [input_formats[0], input_formats[-1]] if input_formats else []
    elif resolved_op_name in {"Slice", "SliceAiCore", "Transpose", "TransposeAiCore"}:
        keep_input_slots([0])
        if input_formats:
            input_formats[0] = "ND"

    return (";".join(input_shapes), ";".join(input_dtypes), ";".join(input_formats))


def get_sig(
    row: dict[str, str],
    as_str: bool = False,
    op_name: str | None = None,
) -> tuple[str, ...] | str:
    resolved_op_name = normalize_op_name(
        (op_name or row.get("OP Type", "") or row.get("OP State", "") or "").strip().strip('"')
    )

    if is_matmul_family(resolved_op_name):
        matmul_sig = canonicalize_matmul_family_signature(row)
        if matmul_sig is not None:
            input_shapes, input_dtypes, input_formats, output_shapes = matmul_sig
            _, output_dtypes = _normalize_shape_attr_sig(
                row.get("Output Shapes", ""),
                row.get("Output Data Types", ""),
            )
            vals = (input_shapes, input_dtypes, input_formats, output_shapes, output_dtypes)
            if as_str:
                inp = row.get("Input Shapes", "") or "N/A"
                out = row.get("Output Shapes", "") or "N/A"
                return f"{inp} -> {out}"
            return vals

    raw_input_shapes, raw_input_dtypes, raw_input_formats = canonicalize_profile_signature(
        row, op_name=op_name
    )
    input_shapes, input_dtypes = _normalize_shape_attr_sig(raw_input_shapes, raw_input_dtypes)
    _, input_formats = _normalize_shape_attr_sig(raw_input_shapes, raw_input_formats)
    output_shapes, output_dtypes = _normalize_shape_attr_sig(
        row.get("Output Shapes", ""),
        row.get("Output Data Types", ""),
    )
    vals = (input_shapes, input_dtypes, input_formats, output_shapes, output_dtypes)

    if as_str:
        inp = row.get("Input Shapes", "") or "N/A"
        out = row.get("Output Shapes", "") or "N/A"
        return f"{inp} -> {out}"

    if resolved_op_name == normalize_op_name(DISPATCH_FFN_OP):
        vals = vals + ((row.get("EP Size", "") or "").strip(),)

    return vals
