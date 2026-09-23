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
RUNTIME_CASE_ID = "Runtime case_id"
CASE_ID_ONLY_RUNTIME_OPS = {"mla_preprocess_0_mix_aic"}
RUNTIME_AWARE_OPS = {"LightningIndexer", "SparseFlashAttention", *CASE_ID_ONLY_RUNTIME_OPS}
FIA_OP = "FusedInferAttentionScore"
FIA_RUNTIME_SIGNATURE_COLUMNS = (
    "Runtime actual_seq_lengths_shape",
    "Runtime actual_seq_lengths_values",
    "Runtime actual_seq_lengths_kv_shape",
    "Runtime actual_seq_lengths_kv_values",
    "Runtime avg_seq_len",
    "Runtime block_table_shape",
    "Runtime block_table_valid_blocks",
    "Runtime num_heads",
    "Runtime num_key_value_heads",
    "Runtime sparse_mode",
    "Runtime input_layout",
    "Runtime block_size",
    "Runtime attn_state",
    "Runtime phase",
    "Runtime topk",
    "Runtime cache_layout",
    "Runtime kv_cache_mode",
    "Runtime sparse_block_size",
    "Runtime sparse_indices_pattern",
    "Runtime sparse_indices_valid_count",
)

SHAPE_COLUMNS = ("Input Shapes", "Output Shapes")


def _strip_balanced_outer_quotes(value: str) -> str:
    """Remove storage-only quote pairs while tolerating legacy over-quoting."""
    cleaned = (value or "").strip()
    while len(cleaned) >= 2 and cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def normalize_shape_semantics(value: str) -> str:
    """Return quote-free, whitespace-normalized shape-cell semantics."""
    cleaned = _strip_balanced_outer_quotes(value)
    if not cleaned:
        return ""
    slots = []
    for slot in cleaned.split(";"):
        normalized_slot = _strip_balanced_outer_quotes(slot)
        slots.append(
            ",".join(dimension.strip() for dimension in normalized_slot.split(","))
            if normalized_slot
            else ""
        )
    return ";".join(slots)


def canonicalize_shape_storage(value: str) -> str:
    """Wrap a non-empty shape value in one literal outer quote pair."""
    semantic = normalize_shape_semantics(value)
    return f'"{semantic}"' if semantic else ""


def canonicalize_shape_columns(row: dict[str, str]) -> dict[str, str]:
    """Return a row whose core shape columns use canonical CSV storage."""
    normalized = dict(row)
    for column in SHAPE_COLUMNS:
        if column in normalized:
            normalized[column] = canonicalize_shape_storage(normalized[column])
    return normalized


_DTYPE_CANONICAL = {
    "BF16": "DT_BF16",
    "BFLOAT16": "DT_BF16",
    "DT_BFLOAT16": "DT_BF16",
    "FP16": "DT_FLOAT16",
    "FLOAT16": "DT_FLOAT16",
    "HALF": "DT_FLOAT16",
    "FLOAT": "DT_FLOAT",
    "FP32": "DT_FLOAT",
    "FLOAT32": "DT_FLOAT",
    "FLOAT64": "DT_FLOAT64",
    "FP64": "DT_FLOAT64",
    "DOUBLE": "DT_FLOAT64",
    "INT4": "DT_INT4",
    "INT8": "DT_INT8",
    "UINT8": "DT_UINT8",
    "INT16": "DT_INT16",
    "INT32": "DT_INT32",
    "INT64": "DT_INT64",
    "BOOL": "DT_BOOL",
    "UNDEFINED": "DT_UNDEFINED",
}


def _canonical_dtype_name(value: str) -> str:
    """Normalize a dtype token to its canonical ``DT_*`` form.

    Profiler op_summary rows record dtypes such as ``FLOAT16`` or
    ``BFLOAT16`` while the database CSV stores ``DT_FLOAT16`` / ``DT_BF16``;
    signatures must canonicalize both sides or otherwise replay rows can
    never be matched back (missing shapes).  Unknown tokens such as opaque
    dtype IDs used by Triton kernels are preserved verbatim.
    """
    cleaned = (value or "").strip()
    if not cleaned:
        return ""
    upper = cleaned.upper()
    if upper in _DTYPE_CANONICAL:
        return _DTYPE_CANONICAL[upper]
    return cleaned


def normalize_op_name(name: str) -> str:
    normalized = name.strip()
    if normalized.endswith("_run.py"):
        normalized = normalized.removesuffix("_run.py")
    elif normalized.endswith("_run"):
        normalized = normalized.removesuffix("_run")
    elif normalized.endswith(".csv"):
        normalized = normalized.removesuffix(".csv")
    return normalized


def get_runtime_signature_context(
    row: dict[str, str],
    op_name: str | None = None,
) -> dict[str, str]:
    """Return semantic runtime fields needed to match a replay case exactly."""
    resolved_op_name = normalize_op_name(
        (op_name or row.get("OP Type", "") or row.get("OP State", "") or "").strip().strip('"')
    )
    if resolved_op_name != FIA_OP:
        return {}
    return {
        column: (row.get(column, "") or "").strip()
        for column in FIA_RUNTIME_SIGNATURE_COLUMNS
        if column in row
    }


def _split_slot_cell(value: str) -> list[str]:
    cleaned = _strip_balanced_outer_quotes(value)
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
    canonical_input_dtypes = ";".join(_canonical_dtype_name(dtype) for dtype in input_dtypes[:2])
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
    elif resolved_op_name in {"ScatterNdUpdate", "ScatterNdUpdateAiCore"}:
        # msprof records the data tensor as a flattened (total_slots, head_dim)
        # because vLLM-Ascend calls .view(-1, head_dim) before the op.  Flatten
        # the first input's leading dims so the profiling signature matches the
        # CSV row regardless of whether the data shape is 2D or 3D paged.
        if input_shapes and input_shapes[0].count(",") >= 2:
            dims = input_shapes[0].split(",")
            total_slots = 1
            for d in dims[:-1]:
                total_slots *= int(d.strip())
            input_shapes[0] = str(total_slots) + "," + dims[-1].strip()

    return (
        ";".join(input_shapes),
        ";".join(_canonical_dtype_name(dtype) for dtype in input_dtypes),
        ";".join(input_formats),
    )


def get_sig(
    row: dict[str, str],
    as_str: bool = False,
    op_name: str | None = None,
) -> tuple[str, ...] | str:
    resolved_op_name = normalize_op_name(
        (op_name or row.get("OP Type", "") or row.get("OP State", "") or "").strip().strip('"')
    )

    if resolved_op_name in CASE_ID_ONLY_RUNTIME_OPS:
        case_id = (row.get(RUNTIME_CASE_ID, "") or "").strip()
        if case_id:
            return case_id if as_str else ("runtime_case_id", case_id)

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
                # Serialize the full canonical identity (shapes, dtypes and
                # formats) so that rows sharing shapes but differing in dtype,
                # e.g. FLOAT vs BF16 MatMul variants, do not collapse into one
                # row during parallel-shard merge dedup.
                return _serialize_sig(vals)
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
    output_dtypes = ";".join(_canonical_dtype_name(dtype) for dtype in output_dtypes.split(";"))
    vals = (input_shapes, input_dtypes, input_formats, output_shapes, output_dtypes)

    if resolved_op_name == normalize_op_name(DISPATCH_FFN_OP):
        vals = vals + ((row.get("EP Size", "") or "").strip(),)

    if resolved_op_name == FIA_OP:
        runtime_context = get_runtime_signature_context(row, resolved_op_name)
        vals = vals + tuple(runtime_context.get(column, "") for column in FIA_RUNTIME_SIGNATURE_COLUMNS)

    if resolved_op_name in RUNTIME_AWARE_OPS:
        case_id = (row.get(RUNTIME_CASE_ID, "") or "").strip()
        if case_id:
            vals = vals + (case_id,)

    if as_str:
        # Serialize the full canonical identity (shapes, dtypes and formats)
        # plus any runtime context (EP / FIA columns / case_id) so that rows
        # with identical Input Shapes but different dtype or runtime semantics
        # do not collide when deduping via the string signature (used by
        # merge_shard_results.py). Without the dtype/formats components, FLOAT
        # and BF16 variants of the same shape (e.g. MatMul) silently collapse
        # into one row during parallel-shard merge, dropping measured data.
        return _serialize_sig(vals)
    return vals


def _serialize_sig(vals: tuple[str, ...]) -> str:
    """Serialize a signature tuple into a lossless, canonical string key."""
    return "|".join(str(value) for value in vals)




def get_case_shard_key(row: dict[str, str], op_name: str) -> str:
    """Return the exact stable key used to assign one CSV row to a worker."""
    case_id = (row.get(RUNTIME_CASE_ID, "") or "").strip()
    if case_id:
        return case_id
    normalized_op = normalize_op_name(op_name)
    return f"{normalized_op}:{get_sig(row, as_str=True, op_name=normalized_op)}"
