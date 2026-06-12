"""Estimate HBM memory consumption for generated shape grid rows.

Used by ``grid_generator/runner.py`` to filter out shapes that would exceed
the available device memory during microbench replay.

Default budget: 32 GiB (50 % of Ascend 910B 64 GiB HBM).
"""

from __future__ import annotations

import math
import re
from typing import Sequence

# ── dtype → bytes-per-element mapping ─────────────────────────
DTYPE_BYTES: dict[str, int] = {
    "DT_FLOAT": 4,
    "DT_FLOAT16": 2,
    "DT_BF16": 2,
    "DT_BFLOAT16": 2,
    "DT_FLOAT32": 4,
    "DT_FLOAT64": 8,
    "DT_DOUBLE": 8,
    "DT_INT8": 1,
    "DT_INT16": 2,
    "DT_INT32": 4,
    "DT_INT64": 8,
    "DT_UINT8": 1,
    "DT_UINT16": 2,
    "DT_UINT32": 4,
    "DT_UINT64": 8,
    "DT_BOOL": 1,
    "DT_COMPLEX64": 8,
    "DT_COMPLEX128": 16,
    # FP8 variants (Ascend / NVIDIA)
    "DT_FLOAT8_E4M3": 1,
    "DT_FLOAT8_E5M2": 1,
    "DT_FLOAT8": 1,
    # Raw types found in some CSVs without DT_ prefix
    "FLOAT": 4,
    "INT8": 1,
    "INT16": 2,
    "INT32": 4,
    "INT64": 8,
    "UINT8": 1,
    "BOOL": 1,
}

DEFAULT_BYTES_PER_ELEMENT = 2  # FP16 fallback

# Note: Ascend 910B has 64 GiB HBM, so 32 GiB is 50% budget.
# While seemingly conservative, DFC operators for large MoEs (e.g. DSv3) 
# require loading all expert weights, which further reduces actual available space. 
# 32GiB is a safe baseline filter, and CLI --max-hbm-gb allows tuning downstream.
DEFAULT_MAX_BYTES = 32 * 1024 ** 3  # 32 GiB


def dtype_to_bytes(dtype_name: str) -> int:
    """Convert a dtype string to element size in bytes.

    Normalises the name by upper-casing and stripping whitespace before
    lookup.  Falls back to ``DEFAULT_BYTES_PER_ELEMENT`` for unknown types.
    """
    key = dtype_name.strip().upper()
    return DTYPE_BYTES.get(key, DEFAULT_BYTES_PER_ELEMENT)


def _parse_dtype_list(dtype_cell: str) -> list[str]:
    """Parse the ``Input Data Types`` CSV column into a dtype list."""
    raw = str(dtype_cell or "").strip().strip('"')
    if not raw:
        return []
    return [part.strip() for part in re.split(r"[;\s]+", raw) if part.strip()]


def estimate_tensor_bytes(shape: tuple[int, ...], bytes_per_elem: int) -> int:
    """Return estimated bytes for one tensor.  Empty shape → 0."""
    if not shape:
        return 0
    return math.prod(shape) * bytes_per_elem


def estimate_row_memory(
    input_shapes: Sequence[tuple[int, ...]],
    output_shapes: Sequence[tuple[int, ...]],
    input_dtypes: Sequence[str],
    output_dtypes: Sequence[str] | None = None,
) -> int:
    """Estimate total HBM memory for one shape-grid row (bytes).

    Each input/output tensor size = product(shape) * bytes_per_element.
    Missing dtype entries fall back to FP16 (2 bytes).
    """
    total = 0
    for idx, shape in enumerate(input_shapes):
        dtype = input_dtypes[idx] if idx < len(input_dtypes) else ""
        total += estimate_tensor_bytes(shape, dtype_to_bytes(dtype))
    if output_dtypes is None:
        output_dtypes = input_dtypes
    for idx, shape in enumerate(output_shapes):
        dtype = output_dtypes[idx] if idx < len(output_dtypes) else ""
        total += estimate_tensor_bytes(shape, dtype_to_bytes(dtype))
    return total


def exceeds_memory_budget(
    input_shapes: Sequence[tuple[int, ...]],
    output_shapes: Sequence[tuple[int, ...]],
    input_dtypes: Sequence[str],
    output_dtypes: Sequence[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[bool, int]:
    """Check whether a row exceeds the HBM budget.

    Returns:
        (exceeded: bool, estimated_bytes: int)
    """
    estimated = estimate_row_memory(
        input_shapes, output_shapes, input_dtypes, output_dtypes,
    )
    return estimated > max_bytes, estimated


def format_bytes(n: int) -> str:
    """Human-readable byte count (e.g. '12.34 GiB')."""
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GiB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n} B"


def parse_dtype_from_template_row(row: dict[str, str]) -> tuple[list[str], list[str]]:
    """Extract input/output dtype lists from a CSV template row."""
    input_dtypes = _parse_dtype_list(row.get("Input Data Types", ""))
    output_dtypes = _parse_dtype_list(row.get("Output Data Types", ""))
    return input_dtypes, output_dtypes
