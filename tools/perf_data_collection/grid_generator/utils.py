from __future__ import annotations

import csv
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from ..signature_utils import (
        MATMUL_FAMILY_OPS,
        canonicalize_matmul_family_signature,
        get_sig,
    )
except ImportError:
    from signature_utils import (
        MATMUL_FAMILY_OPS,
        canonicalize_matmul_family_signature,
        get_sig,
    )


KEEP_COLUMNS = {
    "OP State",
    "Accelerator Core",
    "Input Data Types",
    "Input Formats",
    "Output Data Types",
    "Output Formats",
}
INPUT_SHAPES_COLUMN = "Input Shapes"
OUTPUT_SHAPES_COLUMN = "Output Shapes"
ZERO_VALUE = "0"
LATENCY_HEADER_KEYWORDS = ("duration", "latency", "time", "cycles", "ratio", "miss", "utilization")
MISSING_SHAPE_TOKENS = {"", "N/A", "NA", "NULL", "NONE", "UNDEFINED"}


def _render_progress(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return f"[{'-' * width}]"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = min(width, int(ratio * width))
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def print_progress(
    *,
    file_index: int,
    total_files: int,
    csv_path: Path,
    row_index: int,
    total_rows: int,
    appended_rows: int,
) -> None:
    file_bar = _render_progress(file_index, total_files)
    row_bar = _render_progress(row_index, total_rows)
    message = (
        f"\rFiles {file_bar} {file_index}/{total_files} | "
        f"Rows {row_bar} {row_index}/{total_rows} | "
        f"Appended {appended_rows} | {csv_path.name}"
    )
    print(message, end="", file=sys.stderr, flush=True)


def clear_progress() -> None:
    print("\r" + " " * 160 + "\r", end="", file=sys.stderr, flush=True)


def parse_shape_text(shape_text: str) -> list[tuple[int, ...]]:
    value = str(shape_text or "").strip()
    if value.upper() in MISSING_SHAPE_TOKENS:
        return []
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    value = value.strip()
    if value.upper() in MISSING_SHAPE_TOKENS:
        return []
    if not value:
        return []

    parts = []
    depth = 0
    current = []
    for char in value:
        if char == ";" and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        current.append(char)
    if current:
        parts.append("".join(current).strip())

    shapes: list[tuple[int, ...]] = []
    for part in parts:
        cleaned = part.strip().strip('"').strip()
        if not cleaned or cleaned.upper() in MISSING_SHAPE_TOKENS:
            shapes.append(())
            continue
        if cleaned == "()":
            shapes.append(())
            continue
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = cleaned[1:-1]
        dims = [token.strip() for token in re.split(r"[,\s]+", cleaned) if token.strip()]
        if not dims:
            shapes.append(())
            continue
        shapes.append(tuple(int(dim) for dim in dims))
    return shapes


def build_shape_text(shapes: list[tuple[int, ...]]) -> str:
    if not shapes:
        return ""
    rendered = []
    for shape in shapes:
        if not shape:
            rendered.append("")
            continue
        rendered.append(",".join(str(dim) for dim in shape))
    return ";".join(rendered)


def build_shape_cell(shapes: list[tuple[int, ...]]) -> str:
    shape_text = build_shape_text(shapes)
    return f'"{shape_text}"' if shape_text else shape_text


def align_shape_slot_count(
    template_shapes: list[tuple[int, ...]],
    generated_shapes: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    # Do not truncate generated slots: FIA kernels may legitimately have more
    # runtime inputs than the template row used for metadata inheritance.
    target_count = len(template_shapes)
    if target_count == 0:
        return generated_shapes
    aligned = list(generated_shapes)
    while len(aligned) < target_count:
        aligned.append(())
    return aligned


def zero_fill_column(header: str) -> bool:
    lowered = header.strip().lower()
    return any(keyword in lowered for keyword in LATENCY_HEADER_KEYWORDS)


def build_row_template(headers: list[str], source_row: dict[str, str]) -> dict[str, str]:
    row_template: dict[str, str] = {}
    for header in headers:
        if header in KEEP_COLUMNS or header == OUTPUT_SHAPES_COLUMN:
            row_template[header] = source_row.get(header, "")
        elif header == INPUT_SHAPES_COLUMN:
            row_template[header] = ""
        elif zero_fill_column(header):
            row_template[header] = ZERO_VALUE
        else:
            row_template[header] = source_row.get(header, "")
    return row_template


def build_generated_row(
    headers: list[str],
    source_row: dict[str, str],
    input_shapes: list[tuple[int, ...]],
    output_shapes: list[tuple[int, ...]],
    *,
    extra_values: dict[str, str] | None = None,
) -> dict[str, str]:
    row = build_row_template(headers, source_row)
    row[INPUT_SHAPES_COLUMN] = build_shape_cell(input_shapes)
    if OUTPUT_SHAPES_COLUMN in headers:
        row[OUTPUT_SHAPES_COLUMN] = build_shape_cell(output_shapes)
    if extra_values:
        row.update(extra_values)
    return row


def extend_theory_headers(headers: list[str], extra_headers: list[str]) -> list[str]:
    merged = list(headers)
    for header in extra_headers:
        if header not in merged:
            merged.append(header)
    return merged


def build_input_shapes_sort_key(row: dict[str, str]) -> tuple[tuple[int, ...], ...]:
    return tuple(parse_shape_text(row.get(INPUT_SHAPES_COLUMN, "")))


def sort_generated_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=build_input_shapes_sort_key)


def replace_csv_with_generated_rows(
    csv_path: Path,
    headers: list[str],
    source_rows: list[dict[str, str]],
    generated_rows: list[dict[str, str]],
) -> None:
    temp_path = csv_path.with_name(f"{csv_path.stem}.tmp{csv_path.suffix}")
    with temp_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(source_rows)
        writer.writerows(sort_generated_rows(generated_rows))
    if csv_path.exists():
        os.chmod(csv_path, stat.S_IWRITE | stat.S_IREAD)
    try:
        os.replace(temp_path, csv_path)
    except PermissionError:
        backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
        if csv_path.exists():
            os.chmod(csv_path, stat.S_IWRITE | stat.S_IREAD)
            os.replace(csv_path, backup_path)
        try:
            os.replace(temp_path, csv_path)
        except Exception:
            if backup_path.exists() and not csv_path.exists():
                os.replace(backup_path, csv_path)
            raise
        if backup_path.exists():
            os.remove(backup_path)


def _dedupe_key(headers: list[str], row: dict[str, str]) -> tuple[str, ...]:
    return tuple((row.get(header, "") or "").strip() for header in headers if not zero_fill_column(header))


def _profile_dedupe_key(
    csv_path: Path | None,
    headers: list[str],
    row: dict[str, str],
) -> tuple[str, ...]:
    if csv_path is not None and csv_path.stem in MATMUL_FAMILY_OPS:
        matmul_key = canonicalize_matmul_family_signature(row)
        if matmul_key is not None:
            return ("_matmul_family",) + matmul_key
    if csv_path is not None:
        return ("_profile",) + tuple(get_sig(row, op_name=csv_path.stem))
    return _dedupe_key(headers, row)


def dedupe_generated_rows(
    headers: list[str],
    source_rows: list[dict[str, str]],
    generated_rows: list[dict[str, str]],
    csv_path: Path | None = None,
) -> list[dict[str, str]]:
    seen = {_profile_dedupe_key(csv_path, headers, row) for row in source_rows}
    unique_rows = []
    for row in generated_rows:
        key = _profile_dedupe_key(csv_path, headers, row)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def collect_generated_rows(
    rows: Iterable[Any],
    row_builder: Callable[[Any], dict[str, str]],
    *,
    file_index: int,
    total_files: int,
    csv_path: Path,
    total_rows: int | None,
    progress_interval: int,
) -> list[dict[str, str]]:
    generated_rows: list[dict[str, str]] = []
    appended_rows = 0
    progress_total = total_rows or 0
    print_progress(
        file_index=file_index,
        total_files=total_files,
        csv_path=csv_path,
        row_index=0,
        total_rows=progress_total,
        appended_rows=0,
    )
    for item in rows:
        result = row_builder(item)
        if result is None:
            continue
        generated_rows.append(result)
        appended_rows += 1
        if appended_rows == total_rows or appended_rows % progress_interval == 0:
            print_progress(
                file_index=file_index,
                total_files=total_files,
                csv_path=csv_path,
                row_index=appended_rows,
                total_rows=progress_total or appended_rows,
                appended_rows=appended_rows,
            )
    return generated_rows


def load_csv_template_rows(
    csv_path: Path,
    *,
    require_rows: bool,
    extra_headers: list[str] | None = None,
) -> tuple[list[str], list[dict[str, str]]] | None:
    with csv_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        headers = reader.fieldnames
        if not headers:
            return None
        if extra_headers:
            headers = extend_theory_headers(headers, extra_headers)
        if INPUT_SHAPES_COLUMN not in headers:
            return None
        source_rows = list(reader)
        if require_rows and not source_rows:
            raise ValueError(f"{csv_path} does not contain a data row.")
        if not require_rows and not source_rows:
            return None
    return headers, source_rows


def process_csv_with_generated_rows(
    csv_path: Path,
    *,
    require_rows: bool,
    extra_headers: list[str] | None = None,
    generated_rows_builder: Callable[[list[str], list[dict[str, str]]], list[dict[str, str]] | None],
) -> int | None:
    loaded = load_csv_template_rows(
        csv_path,
        require_rows=require_rows,
        extra_headers=extra_headers,
    )
    if loaded is None:
        if require_rows:
            raise ValueError(f"{csv_path} is missing a header row.")
        return None
    headers, source_rows = loaded
    generated_rows = generated_rows_builder(headers, source_rows)
    if generated_rows is None:
        return None
    generated_rows = dedupe_generated_rows(headers, source_rows, generated_rows, csv_path)
    replace_csv_with_generated_rows(csv_path, headers, source_rows, generated_rows)
    return len(generated_rows)
