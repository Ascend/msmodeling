# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
"""Run one operator microbenchmark and serialize its performance result.

The production contract is msprof-only.  The initial implementation also
provides an explicit, read-only database simulation backend so that the CLI,
matching rules, and output schemas can be developed without an NPU.  A
simulated result is always labelled and never written back to the profiling
database.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

try:
    from .op_microbench.adapters import (
        ADAPTERS,
        AdapterValidationError,
        describe_adapter,
        get_adapter,
    )
except ImportError:
    from op_microbench.adapters import (
        ADAPTERS,
        AdapterValidationError,
        describe_adapter,
        get_adapter,
    )


SCHEMA_VERSION = 1
DEFAULT_PROFILING = {
    "device_id": 0,
    "warmup_count": 5,
    "repeat_count": 30,
    "timeout_seconds": 300,
    "include_raw_records": True,
    "keep_artifacts": "on_failure",
}

BASE_CSV_COLUMNS = [
    "OP State",
    "Accelerator Core",
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
]
DURATION_CSV_COLUMNS = [
    "Average Duration(us)",
    "Median Duration(us)",
    "Std Duration(us)",
]
ACTUAL_OUTPUT_CSV_COLUMNS = [
    "Actual Output Shapes",
    "Actual Output Data Types",
    "Actual Output Formats",
]
SIMULATION_CSV_COLUMNS = ["Result Source", "Simulated", "Source CSV", "Source Row"]

PROFILING_AVERAGE_DURATION = "Profiling Average Duration(us)"
PROFILING_MEDIAN_DURATION = "Profiling Median Duration(us)"
PROFILING_STD_DURATION = "Profiling Std Duration(us)"
PROFILING_AVERAGE_PREFIX = "Profiling Average "

KERNEL_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
VERSION_DIR_PATTERN = re.compile(r"^vllm(?P<vllm>[^_]+)_torch(?P<torch>[^_]+)_cann(?P<cann>.+)$")
REPLAY_REQUIRED_COLUMNS = {
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
}
DTYPE_SIZE_BYTES = {
    "DT_BOOL": 1,
    "DT_INT8": 1,
    "DT_UINT8": 1,
    "DT_FLOAT16": 2,
    "DT_BF16": 2,
    "DT_INT16": 2,
    "DT_UINT16": 2,
    "DT_FLOAT": 4,
    "DT_FLOAT32": 4,
    "DT_INT32": 4,
    "DT_UINT32": 4,
    "DT_FLOAT64": 8,
    "DT_INT64": 8,
    "DT_UINT64": 8,
}

# When allow_internal_format=True (set by the NPU worker), the runtime may
# convert 4D ND tensors into optimized physical formats (NCHW, NHWC, etc.).
# These are logically equivalent to ND; only the physical storage layout differs.
_NPU_INTERNAL_FORMATS = frozenset({"NCHW", "NHWC", "NCDHW", "NDHWC", "NC1HWC0", "NCL", "NLC", "NCH"})


def _formats_compatible(expected: str | None, actual: str | None) -> bool:
    """Check format compatibility, treating NPU internal formats as equivalent to ND."""
    if expected == actual:
        return True
    if expected == "ND" and actual in _NPU_INTERNAL_FORMATS:
        return True
    if actual == "ND" and expected in _NPU_INTERNAL_FORMATS:
        return True
    return False


def _shapes_compatible(expected: list, actual: list) -> bool:
    """Check logical shape compatibility without conflating scalars and empty tensors."""
    return expected == actual


class MicrobenchError(Exception):
    """A structured failure returned by the microbenchmark CLI."""

    def __init__(
        self,
        code: str,
        phase: str,
        message: str,
        message_zh: str,
        details: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.phase = phase
        self.message = message
        self.message_zh = message_zh
        self.details = details or {}
        self.artifacts = artifacts


def build_argparser() -> argparse.ArgumentParser:
    """Build the public command-line interface."""
    parser = argparse.ArgumentParser(description="Run one operator microbenchmark and emit JSON (default) or CSV.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", type=Path, help="Operator request JSON path.")
    source.add_argument(
        "--list-operators",
        action="store_true",
        help="List all supported single-card op_replay operators and exit.",
    )
    source.add_argument(
        "--describe-operator",
        metavar="KERNEL_TYPE",
        help="Describe the CSV-row request contract for one operator, then exit.",
    )
    parser.add_argument(
        "--output-format",
        choices=("json", "csv"),
        default="json",
        help="Result format. Default: json. The output filename is not used to infer it.",
    )
    parser.add_argument("--output", type=Path, help="Result path. Omit to write to stdout.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Directory for msprof artifacts. It is unused by database simulation.",
    )
    parser.add_argument(
        "--simulate-from-database",
        type=Path,
        metavar="PATH",
        help=(
            "Development-only: read one matching record from a profiling database CSV "
            "file or version directory instead of running an NPU."
        ),
    )
    return parser


def _raise_request_error(message: str, message_zh: str, **details: Any) -> None:
    raise MicrobenchError("INVALID_REQUEST", "request_validation", message, message_zh, details)


def load_request(path: Path) -> dict[str, Any]:
    """Load a request JSON object from disk."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MicrobenchError(
            "REQUEST_NOT_FOUND",
            "request_validation",
            f"Request file does not exist: {path}",
            f"请求文件不存在：{path}",
            {"request_path": str(path)},
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise MicrobenchError(
            "REQUEST_READ_FAILED",
            "request_validation",
            f"Could not read request file: {exc}",
            f"无法读取请求文件：{exc}",
            {"request_path": str(path)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise MicrobenchError(
            "INVALID_REQUEST_JSON",
            "request_validation",
            f"Request is not valid JSON: {exc}",
            f"请求文件不是合法 JSON：{exc}",
            {"request_path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(raw, dict):
        _raise_request_error("Request root must be an object.", "请求 JSON 顶层必须是对象。")
    return raw


def _validate_replay_row(value: Any) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Validate a CSV-style op_replay row and derive its concrete input slots."""
    if not isinstance(value, dict) or not value:
        _raise_request_error(
            "replay_row must be a non-empty object.",
            "replay_row must be a non-empty object.",
        )
    invalid_keys = [
        key
        for key in value
        if not isinstance(key, str) or not key or len(key) > 128 or any(char in key for char in "\x00\r\n")
    ]
    if invalid_keys:
        _raise_request_error(
            "replay_row contains an invalid column name.",
            "replay_row contains an invalid column name.",
            invalid_columns=[str(key) for key in invalid_keys],
        )
    non_strings = [key for key, item in value.items() if not isinstance(item, str)]
    if non_strings:
        _raise_request_error(
            "All replay_row values must be strings copied from a replay CSV row.",
            "All replay_row values must be strings copied from a replay CSV row.",
            invalid_columns=non_strings,
        )
    missing = sorted(REPLAY_REQUIRED_COLUMNS - set(value))
    if missing:
        _raise_request_error(
            f"replay_row is missing columns: {', '.join(missing)}",
            f"replay_row is missing columns: {', '.join(missing)}",
            missing_columns=missing,
        )
    row = dict(value)
    shape_slots = _strip_outer_quotes(row["Input Shapes"]).split(";")
    dtype_slots = _strip_outer_quotes(row["Input Data Types"]).split(";")
    format_slots = _strip_outer_quotes(row["Input Formats"]).split(";")
    if not (len(shape_slots) == len(dtype_slots) == len(format_slots)):
        _raise_request_error(
            "replay_row input metadata slot counts do not match.",
            "replay_row input metadata slot counts do not match.",
            shape_slots=len(shape_slots),
            dtype_slots=len(dtype_slots),
            format_slots=len(format_slots),
        )
    inputs: list[dict[str, Any]] = []
    for slot, (shape_text, dtype, tensor_format) in enumerate(zip(shape_slots, dtype_slots, format_slots, strict=True)):
        shape_text = _strip_outer_quotes(shape_text).strip()
        dtype = _strip_outer_quotes(dtype).strip()
        tensor_format = _strip_outer_quotes(tensor_format).strip()
        if not shape_text:
            continue
        try:
            shape = [] if shape_text == "[]" else [int(dim.strip()) for dim in shape_text.split(",") if dim.strip()]
        except ValueError:
            _raise_request_error(
                f"replay_row Input Shapes slot {slot} is invalid.",
                f"replay_row Input Shapes slot {slot} is invalid.",
                slot=slot,
                value=shape_text,
            )
        if any(dim < 0 for dim in shape):
            _raise_request_error(
                f"replay_row Input Shapes slot {slot} contains a negative dimension.",
                f"replay_row 的 Input Shapes 槽位 {slot} 包含负数维度。",
                slot=slot,
                value=shape_text,
            )
        if not dtype:
            _raise_request_error(
                f"replay_row Input Data Types slot {slot} is empty.",
                f"replay_row 的 Input Data Types 槽位 {slot} 为空。",
                slot=slot,
            )
        inputs.append(
            {
                "name": f"replay_slot_{slot}",
                "shape": shape,
                "dtype": dtype if dtype.startswith("DT_") else f"DT_{dtype}",
                "format": tensor_format or "NULL",
            }
        )
    if not inputs:
        _raise_request_error(
            "replay_row has no concrete input slots.",
            "replay_row has no concrete input slots.",
        )
    return row, inputs


def _validate_profiling(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        _raise_request_error("profiling must be an object.", "profiling 必须是对象。")
    unknown = sorted(set(value) - set(DEFAULT_PROFILING))
    if unknown:
        _raise_request_error(
            f"Unknown profiling fields: {', '.join(unknown)}",
            f"存在未知的 profiling 字段：{', '.join(unknown)}",
            unknown_fields=unknown,
        )
    result = {**DEFAULT_PROFILING, **value}
    integer_rules = {
        "device_id": 0,
        "warmup_count": 0,
        "repeat_count": 1,
        "timeout_seconds": 1,
    }
    for field, minimum in integer_rules.items():
        field_value = result[field]
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < minimum:
            _raise_request_error(
                f"profiling.{field} must be an integer >= {minimum}.",
                f"profiling.{field} 必须是大于等于 {minimum} 的整数。",
                field=f"profiling.{field}",
                value=field_value,
            )
    if not isinstance(result["include_raw_records"], bool):
        _raise_request_error(
            "profiling.include_raw_records must be a boolean.",
            "profiling.include_raw_records 必须是布尔值。",
        )
    keep_artifacts = result["keep_artifacts"]
    if not isinstance(keep_artifacts, str) or keep_artifacts not in {"always", "on_failure", "never"}:
        _raise_request_error(
            "profiling.keep_artifacts must be always, on_failure, or never.",
            "profiling.keep_artifacts 必须是 always、on_failure 或 never。",
        )
    return result


def normalize_request(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a version-1 request."""
    allowed_fields = {
        "schema_version",
        "kernel_type",
        "profiling",
        "replay_row",
    }
    unknown = sorted(set(raw) - allowed_fields)
    if unknown:
        _raise_request_error(
            f"Unknown request fields: {', '.join(unknown)}",
            f"存在未知请求字段：{', '.join(unknown)}",
            unknown_fields=unknown,
        )
    if raw.get("schema_version") != SCHEMA_VERSION:
        _raise_request_error(
            f"schema_version must be {SCHEMA_VERSION}.",
            f"schema_version 必须是 {SCHEMA_VERSION}。",
            value=raw.get("schema_version"),
        )
    kernel_type = raw.get("kernel_type")
    if not isinstance(kernel_type, str) or not KERNEL_NAME_PATTERN.fullmatch(kernel_type):
        _raise_request_error(
            "kernel_type must be a simple operator name.",
            "kernel_type 必须是合法的简单算子名称，不能包含路径。",
            value=kernel_type,
        )
    replay_row, inputs = _validate_replay_row(raw.get("replay_row"))
    try:
        get_adapter(kernel_type)
    except AdapterValidationError as exc:
        _raise_request_error(exc.message, exc.message_zh, **exc.details)
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "kernel_type": kernel_type,
        "inputs": inputs,
        "replay_row": replay_row,
        "profiling": _validate_profiling(raw.get("profiling")),
    }
    return normalized


def _strip_outer_quotes(value: str) -> str:
    value = value.strip()
    while len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1].strip()
    return value


def parse_shape_cell(value: str) -> list[list[int]]:
    """Parse the semicolon-separated shape representation used by the database."""
    value = _strip_outer_quotes(value)
    if not value:
        return []
    shapes: list[list[int]] = []
    for shape_text in value.split(";"):
        shape_text = _strip_outer_quotes(shape_text)
        if shape_text in {"", "[]"}:
            shapes.append([])
            continue
        try:
            shapes.append([int(dim.strip()) for dim in shape_text.split(",")])
        except ValueError as exc:
            raise ValueError(f"invalid shape cell: {value!r}") from exc
    return shapes


def _parse_series_cell(value: str) -> list[str]:
    value = _strip_outer_quotes(value)
    if not value:
        return []
    return [_strip_outer_quotes(item).strip().upper() for item in value.split(";")]


def _format_shapes(shapes: list[list[int]]) -> str:
    return ";".join(",".join(str(dim) for dim in shape) if shape else "[]" for shape in shapes)


def _tensor_size_bytes(shape: list[int], dtype: str) -> int | None:
    element_size = DTYPE_SIZE_BYTES.get(dtype.upper())
    if element_size is None:
        return None
    elements = math.prod(shape) if shape else 1
    return elements * element_size


def _finite_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(str(value).strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_float(value: Any) -> float | None:
    parsed = _finite_float(value)
    return parsed if parsed is not None and parsed > 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_database_csv(source: Path, kernel_type: str) -> Path:
    """Resolve one unambiguous source CSV from a file or database directory."""
    try:
        source = source.resolve(strict=True)
    except FileNotFoundError as exc:
        raise MicrobenchError(
            "SIMULATION_SOURCE_NOT_FOUND",
            "database_simulation",
            f"Simulation source does not exist: {source}",
            f"模拟数据源不存在：{source}",
            {"source": str(source)},
        ) from exc
    if source.is_file():
        if source.suffix.lower() != ".csv":
            raise MicrobenchError(
                "SIMULATION_SOURCE_NOT_CSV",
                "database_simulation",
                "Simulation source file must be a CSV file.",
                "模拟数据源文件必须是 CSV。",
                {"source": str(source)},
            )
        if source.stem.casefold() != kernel_type.casefold():
            raise MicrobenchError(
                "OPERATOR_DATABASE_MISMATCH",
                "database_simulation",
                "The source CSV filename does not match kernel_type.",
                "数据源 CSV 文件名与 kernel_type 不一致。",
                {
                    "source_csv": str(source),
                    "source_operator": source.stem,
                    "kernel_type": kernel_type,
                },
            )
        return source
    direct = source / f"{kernel_type}.csv"
    if direct.is_file():
        return direct.resolve()
    candidates = sorted(
        path.resolve() for path in source.rglob("*.csv") if path.stem.casefold() == kernel_type.casefold()
    )
    if not candidates:
        raise MicrobenchError(
            "OPERATOR_DATABASE_NOT_FOUND",
            "database_simulation",
            f"No {kernel_type}.csv was found under the simulation source.",
            f"模拟数据源中没有找到 {kernel_type}.csv。",
            {"source": str(source), "kernel_type": kernel_type},
        )
    if len(candidates) > 1:
        raise MicrobenchError(
            "AMBIGUOUS_OPERATOR_DATABASE",
            "database_simulation",
            f"More than one {kernel_type}.csv was found; select a version directory or file.",
            f"找到多个 {kernel_type}.csv；请指定版本目录或具体文件。",
            {"candidates": [str(path) for path in candidates]},
        )
    return candidates[0]


def _ensure_result_is_decoupled(result_path: Path | None, source: Path) -> None:
    if result_path is None:
        return
    source_root = source if source.is_dir() else source.parent
    resolved_result = result_path.resolve()
    try:
        resolved_result.relative_to(source_root.resolve())
    except ValueError:
        return
    raise MicrobenchError(
        "OUTPUT_INSIDE_DATABASE",
        "request_validation",
        "The result path must not be inside the profiling database source directory.",
        "结果文件不能写入 profiling database 数据源目录。",
        {"output": str(resolved_result), "database_root": str(source_root.resolve())},
    )


def _load_matching_database_row(csv_path: Path, request: dict[str, Any]) -> tuple[list[str], dict[str, str], int, int]:
    replay_row = request["replay_row"]
    match_columns = sorted(
        REPLAY_REQUIRED_COLUMNS
        | {
            column
            for column in replay_row
            if column.startswith("Runtime ") or column in {"OP State", "Accelerator Core", "EP Size"}
        }
    )
    matches: list[tuple[int, dict[str, str]]] = []
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames
            if not headers:
                raise MicrobenchError(
                    "INVALID_DATABASE_CSV",
                    "database_simulation",
                    "Simulation CSV has no header.",
                    "模拟 CSV 没有表头。",
                    {"source_csv": str(csv_path)},
                )
            missing = sorted(set(match_columns) - set(headers))
            if missing:
                raise MicrobenchError(
                    "INVALID_DATABASE_CSV",
                    "database_simulation",
                    f"Simulation CSV is missing columns: {', '.join(missing)}",
                    f"模拟 CSV 缺少字段：{', '.join(missing)}",
                    {"source_csv": str(csv_path), "missing_columns": missing},
                )
            scanned = 0
            for row_number, row in enumerate(reader, start=2):
                scanned += 1
                row_matches = all(
                    _strip_outer_quotes(row.get(column, "")) == _strip_outer_quotes(replay_row.get(column, ""))
                    for column in match_columns
                )
                if row_matches:
                    matches.append(
                        (
                            row_number,
                            {key: value or "" for key, value in row.items() if key},
                        )
                    )
    except OSError as exc:
        raise MicrobenchError(
            "DATABASE_READ_FAILED",
            "database_simulation",
            f"Could not read simulation CSV: {exc}",
            f"无法读取模拟 CSV：{exc}",
            {"source_csv": str(csv_path)},
        ) from exc
    match_key = {column: replay_row.get(column, "") for column in match_columns}
    if not matches:
        raise MicrobenchError(
            "DATABASE_RECORD_NOT_FOUND",
            "database_match",
            "No database row exactly matched the requested replay descriptor.",
            "数据库中没有与请求 replay 描述严格匹配的记录。",
            {
                "source_csv": str(csv_path),
                "match_key": match_key,
                "rows_scanned": scanned,
            },
        )
    if len(matches) > 1:
        raise MicrobenchError(
            "AMBIGUOUS_DATABASE_RECORD",
            "database_match",
            "More than one database row matched the requested replay descriptor.",
            "数据库中有多条记录与请求 replay 描述匹配，无法唯一确定结果。",
            {
                "source_csv": str(csv_path),
                "match_key": match_key,
                "matching_rows": [row_number for row_number, _ in matches],
            },
        )
    row_number, row = matches[0]
    return headers, row, row_number, scanned


def _parse_outputs(row: dict[str, str]) -> list[dict[str, Any]]:
    try:
        shapes = parse_shape_cell(row.get("Output Shapes", ""))
    except ValueError as exc:
        raise MicrobenchError(
            "INVALID_DATABASE_OUTPUT",
            "database_match",
            "The matched row contains invalid output shapes.",
            "匹配记录中的输出 shape 不合法。",
            {"Output Shapes": row.get("Output Shapes")},
        ) from exc
    dtypes = _parse_series_cell(row.get("Output Data Types", ""))
    formats = _parse_series_cell(row.get("Output Formats", ""))
    if not shapes or len(shapes) != len(dtypes) or len(shapes) != len(formats):
        raise MicrobenchError(
            "INVALID_DATABASE_OUTPUT",
            "database_match",
            "Output shape, dtype, and format counts in the matched row are inconsistent.",
            "匹配记录中的输出 shape、dtype、format 数量不一致。",
            {
                "shape_count": len(shapes),
                "dtype_count": len(dtypes),
                "format_count": len(formats),
            },
        )
    return [
        {
            "path": f"$[{index}]",
            "origin": "profiling_database_simulation",
            "logical_shape": shape,
            "device_shape": None,
            "dtype": dtype if dtype.startswith("DT_") else f"DT_{dtype}",
            "format": output_format,
            "size_bytes": _tensor_size_bytes(shape, dtype if dtype.startswith("DT_") else f"DT_{dtype}"),
        }
        for index, (shape, dtype, output_format) in enumerate(zip(shapes, dtypes, formats, strict=True))
    ]


def _unknown_output_descriptor_slots(replay_row: dict[str, str]) -> list[bool]:
    shape_slots = _strip_outer_quotes(replay_row["Output Shapes"]).split(";")
    dtype_slots = _strip_outer_quotes(replay_row["Output Data Types"]).split(";")
    format_slots = _strip_outer_quotes(replay_row["Output Formats"]).split(";")
    return [
        not _strip_outer_quotes(shape).strip()
        and _strip_outer_quotes(dtype).strip().upper() in {"", "UNDEFINED", "DT_UNDEFINED"}
        and _strip_outer_quotes(tensor_format).strip().upper() in {"", "UNDEFINED", "DT_UNDEFINED"}
        for shape, dtype, tensor_format in zip(shape_slots, dtype_slots, format_slots, strict=True)
    ]


def _validate_replay_outputs(
    replay_row: dict[str, str],
    outputs: list[dict[str, Any]],
    *,
    kernel_type: str | None = None,
) -> bool:
    expected = _parse_outputs(replay_row)
    unknown_descriptors = _unknown_output_descriptor_slots(replay_row)
    dtype_compatibility = set()
    if kernel_type is not None:
        dtype_compatibility = set(get_adapter(kernel_type).api_output_dtype_compatibility)
    mismatches: list[dict[str, Any]] = []
    if len(expected) != len(outputs):
        mismatches.append({"field": "output_count", "expected": len(expected), "actual": len(outputs)})
    for index, assertion in enumerate(expected[: len(outputs)]):
        output = outputs[index]
        if unknown_descriptors[index]:
            continue
        for field in ("logical_shape", "dtype", "format"):
            expected_val = assertion[field]
            actual_val = output.get(field)
            if field == "format" and _formats_compatible(expected_val, actual_val):
                continue
            if field == "logical_shape" and _shapes_compatible(expected_val, actual_val):
                continue
            if field == "dtype" and (index, expected_val, actual_val) in dtype_compatibility:
                continue
            if expected_val != actual_val:
                mismatches.append(
                    {
                        "output_index": index,
                        "field": field,
                        "expected": expected_val,
                        "actual": actual_val,
                    }
                )
    if mismatches:
        raise MicrobenchError(
            "REPLAY_OUTPUT_MISMATCH",
            "output_validation",
            "The operator output did not match the replay-row output descriptor.",
            "算子输出与 replay_row 中的输出描述不一致。",
            {"mismatches": mismatches, "actual_outputs": outputs},
        )
    return True


def _environment_from_path(csv_path: Path, device_id: int) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "device_id": device_id,
        "device_name": None,
        "cann_version": None,
        "torch_version": None,
        "torch_npu_version": None,
        "vllm_ascend_version": None,
        "msprof_version": None,
        "simulated": True,
    }
    for parent in csv_path.parents:
        match = VERSION_DIR_PATTERN.fullmatch(parent.name)
        if match:
            environment["vllm_ascend_version"] = match.group("vllm")
            environment["torch_version"] = match.group("torch")
            environment["cann_version"] = match.group("cann")
        if parent.name.startswith(("ATLAS_", "ASCEND_")):
            environment["device_name"] = parent.name
    return environment


def _build_duration_and_metrics(
    row: dict[str, str], headers: list[str], csv_path: Path, row_number: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    median = _positive_float(row.get(PROFILING_MEDIAN_DURATION))
    if median is None:
        raise MicrobenchError(
            "NO_VALID_PROFILING_DURATION",
            "database_match",
            f"Matched row has no finite positive {PROFILING_MEDIAN_DURATION}.",
            f"匹配记录没有有限正数的 {PROFILING_MEDIAN_DURATION}。",
            {
                "source_csv": str(csv_path),
                "source_row": row_number,
                "value": row.get(PROFILING_MEDIAN_DURATION),
            },
        )
    mean = _positive_float(row.get(PROFILING_AVERAGE_DURATION))
    std = _finite_float(row.get(PROFILING_STD_DURATION))
    if std is not None and std < 0:
        std = None
    statistics = {
        "min": None,
        "p50": median,
        "p90": None,
        "p95": None,
        "mean": mean,
        "std": std,
        "max": None,
    }
    duration = {
        "source": "profiling_database_simulation",
        "source_kind": "previously_aggregated_msprof_data",
        "column": PROFILING_MEDIAN_DURATION,
        "unit": "us",
        "aggregation": "p50",
        "value_us": median,
        "warmup_count": None,
        "sample_count": None,
        "statistics": statistics,
    }
    metrics: dict[str, dict[str, Any]] = {
        "Task Duration(us)": {
            "p50": median,
            "mean": mean,
            "std": std,
            "source_columns": [
                PROFILING_MEDIAN_DURATION,
                PROFILING_AVERAGE_DURATION,
                PROFILING_STD_DURATION,
            ],
        }
    }
    for header in headers:
        if not header.startswith(PROFILING_AVERAGE_PREFIX) or header == PROFILING_AVERAGE_DURATION:
            continue
        metric_name = header.removeprefix(PROFILING_AVERAGE_PREFIX)
        metrics[metric_name] = {
            "mean": _finite_float(row.get(header)),
            "source_column": header,
        }
    return duration, metrics


def run_database_simulation(request: dict[str, Any], source: Path, request_id: str) -> dict[str, Any]:
    """Build a labelled result by strictly matching one existing database row."""
    csv_path = resolve_database_csv(source, request["kernel_type"])
    headers, row, row_number, rows_scanned = _load_matching_database_row(csv_path, request)
    outputs = _parse_outputs(row)
    output_matches_replay_row = True
    duration, aggregated_metrics = _build_duration_and_metrics(row, headers, csv_path, row_number)
    source_stat = csv_path.stat()
    source_info = {
        "path": str(csv_path),
        "row_number": row_number,
        "size_bytes": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "sha256": _sha256(csv_path),
    }
    inputs = [
        {
            "name": tensor["name"],
            "origin": "replay_metadata",
            "logical_shape": tensor["shape"],
            "device_shape": None,
            "dtype": tensor["dtype"],
            # Keep unknown formats consistent with the real NPU metadata path.
            "format": None if tensor["format"] == "NULL" else tensor["format"],
            "size_bytes": _tensor_size_bytes(tensor["shape"], tensor["dtype"]),
        }
        for tensor in request["inputs"]
    ]
    raw_target_records = []
    if request["profiling"]["include_raw_records"]:
        raw_target_records.append(
            {
                "phase": "database_simulation",
                "record_kind": "profiling_database_aggregate",
                "source_row": row_number,
                "raw": row,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "status": "succeeded",
        "phase": "database_simulation",
        "simulated": True,
        "result_source": "profiling_database_simulation",
        "operator": {
            "requested_kernel_type": request["kernel_type"],
            "resolved_adapter": None,
            "is_composite": None,
        },
        "configuration": {
            "backend": "profiling_database_simulation",
            "replay_row": request["replay_row"],
            "profiling": request["profiling"],
        },
        "environment": _environment_from_path(csv_path, request["profiling"]["device_id"]),
        "inputs": inputs,
        "outputs": outputs,
        "duration": duration,
        "msprof": {
            "executed": False,
            "simulated": True,
            "source": "profiling_database_simulation",
            "return_code": None,
            "timed_out": None,
            "tables": [
                {
                    "kind": "profiling_database_csv",
                    "path": str(csv_path),
                    "columns": headers,
                    "rows_scanned": rows_scanned,
                    "source_row": row_number,
                }
            ],
            "target_match": {
                "rule": "exact_replay_descriptor",
                "database_rows_matched": 1,
                "accelerator_core": row.get("Accelerator Core") or None,
                "warmup_records": None,
                "measurement_records": None,
            },
            "aggregated_metrics": aggregated_metrics,
            "physical_kernels": [],
            "raw_target_records": raw_target_records,
            "other_tasks": None,
        },
        "validation": {
            "request_valid": True,
            "adapter_supported": None,
            "inputs_built": None,
            "replay_legal": None,
            "msprof_completed": None,
            "profile_match_unambiguous": None,
            "finite_positive_duration": True,
            "output_matches_replay_row": output_matches_replay_row,
            "simulation_record_matched": True,
        },
        "warnings": [
            "This is a simulated result from an existing profiling database row; no NPU operator was executed.",
            "msprof was not run in this invocation; physical kernel names and raw msprof task records are unavailable.",
        ],
        "error": None,
        "artifacts": {
            "kept": False,
            "directory": None,
            "files": [],
            "simulation_source": source_info,
        },
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _numeric_statistics(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "p50": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "max": max(values),
    }


def _read_worker_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MicrobenchError(
            "WORKER_METADATA_NOT_FOUND",
            "operator_execution",
            "The profiled worker did not produce its metadata file.",
            "被采集的 worker 没有生成执行元数据文件。",
            {"metadata_path": str(path)},
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MicrobenchError(
            "INVALID_WORKER_METADATA",
            "operator_execution",
            f"Could not parse worker metadata: {exc}",
            f"无法解析 worker 执行元数据：{exc}",
            {"metadata_path": str(path)},
        ) from exc
    if not isinstance(payload, dict):
        raise MicrobenchError(
            "INVALID_WORKER_METADATA",
            "operator_execution",
            "Worker metadata root must be an object.",
            "worker 执行元数据顶层必须是对象。",
            {"metadata_path": str(path)},
        )
    return payload


def _find_msprof_tables(profile_root: Path) -> list[Path]:
    tables = sorted(profile_root.rglob("op_summary_*.csv"))
    if not tables:
        raise MicrobenchError(
            "MSPROF_SUMMARY_NOT_FOUND",
            "profile_parse",
            "msprof completed without producing an op_summary CSV.",
            "msprof 执行完成，但没有生成 op_summary CSV。",
            {"profile_root": str(profile_root)},
        )
    return tables


def _read_msprof_rows(
    table_paths: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    for path in table_paths:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                headers = reader.fieldnames or []
                table_rows = [{key: value or "" for key, value in row.items() if key} for row in reader]
        except OSError as exc:
            raise MicrobenchError(
                "MSPROF_SUMMARY_READ_FAILED",
                "profile_parse",
                f"Could not read msprof summary: {exc}",
                f"无法读取 msprof summary：{exc}",
                {"table_path": str(path)},
            ) from exc
        if "OP Type" not in headers or "Task Duration(us)" not in headers:
            raise MicrobenchError(
                "INVALID_MSPROF_SUMMARY",
                "profile_parse",
                "msprof summary is missing OP Type or Task Duration(us).",
                "msprof summary 缺少 OP Type 或 Task Duration(us) 字段。",
                {"table_path": str(path), "columns": headers},
            )
        for row_index, row in enumerate(table_rows, start=2):
            records.append({"table": str(path), "row_number": row_index, "raw": row})
        tables.append(
            {
                "kind": "op_summary",
                "path": str(path),
                "columns": headers,
                "row_count": len(table_rows),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records, tables


def _record_start_key(record: dict[str, Any]) -> tuple[float, str, int]:
    start = _finite_float(record["raw"].get("Task Start Time(us)"))
    return (
        start if start is not None else math.inf,
        record["table"],
        record["row_number"],
    )


def _metric_statistics(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    excluded = {
        "Device_id",
        "Model ID",
        "Task ID",
        "Stream ID",
        "Op Name",
        "OP Type",
        "OP State",
        "Task Type",
        "Task Start Time(us)",
        "Block Dim",
        "Mix Block Dim",
        "HF32 Eligible",
        "Input Shapes",
        "Input Data Types",
        "Input Formats",
        "Output Shapes",
        "Output Data Types",
        "Output Formats",
        "Context ID",
    }
    headers = sorted({header for row in rows for header in row if header not in excluded})
    metrics: dict[str, dict[str, Any]] = {}
    for header in headers:
        values = [_finite_float(row.get(header)) for row in rows]
        finite_values = [value for value in values if value is not None]
        if not finite_values:
            continue
        metrics[header] = {
            **_numeric_statistics(finite_values),
            "source_column": header,
        }
    return metrics


def _physical_kernels(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("Op Name", "").strip(),
            row.get("OP Type", "").strip(),
            row.get("Task Type", "").strip(),
        )
        grouped.setdefault(key, []).append(row)
    kernels = []
    for (kernel_name, op_type, task_type), group_rows in sorted(grouped.items()):
        durations = [_positive_float(row.get("Task Duration(us)")) for row in group_rows]
        finite_durations = [value for value in durations if value is not None]
        kernels.append(
            {
                "kernel_name": kernel_name or None,
                "op_type": op_type or None,
                "task_type": task_type or None,
                "sample_count": len(group_rows),
                "duration": _numeric_statistics(finite_durations) if finite_durations else None,
            }
        )
    return kernels


def _build_msprof_result(
    request: dict[str, Any],
    request_id: str,
    worker: dict[str, Any],
    table_paths: list[Path],
    return_code: int,
) -> dict[str, Any]:
    records, tables = _read_msprof_rows(table_paths)
    records.sort(key=_record_start_key)
    target_types = {value.casefold() for value in worker["operator"]["target_op_types"]}
    target_records = [
        record for record in records if record["raw"].get("OP Type", "").strip().casefold() in target_types
    ]
    warmup_count = request["profiling"]["warmup_count"]
    repeat_count = request["profiling"]["repeat_count"]
    expected_count = warmup_count + repeat_count
    if len(target_records) != expected_count:
        raise MicrobenchError(
            "AMBIGUOUS_MSPROF_TARGET_RECORDS",
            "profile_match",
            "The number of matching msprof records does not equal warmup_count + repeat_count.",
            "匹配到的 msprof 记录数不等于 warmup_count + repeat_count，无法可靠区分目标记录。",
            {
                "target_op_types": worker["operator"]["target_op_types"],
                "expected_records": expected_count,
                "matched_records": len(target_records),
                "all_op_types": sorted({record["raw"].get("OP Type", "") for record in records}),
            },
        )
    measurement_records = target_records[warmup_count:]
    measurement_rows = [record["raw"] for record in measurement_records]
    duration_values = [_positive_float(row.get("Task Duration(us)")) for row in measurement_rows]
    if len(duration_values) != repeat_count or any(value is None for value in duration_values):
        raise MicrobenchError(
            "NO_VALID_MSPROF_DURATION",
            "measurement",
            "One or more measurement records have a non-finite or non-positive Task Duration(us).",
            "一条或多条正式采集记录的 Task Duration(us) 不是有限正数。",
            {"values": [row.get("Task Duration(us)") for row in measurement_rows]},
        )
    durations = [value for value in duration_values if value is not None]
    duration_stats = _numeric_statistics(durations)
    aggregated_metrics = _metric_statistics(measurement_rows)
    aggregated_metrics["Task Duration(us)"] = {
        **duration_stats,
        "source_column": "Task Duration(us)",
    }
    raw_target_records = []
    if request["profiling"]["include_raw_records"]:
        for index, record in enumerate(target_records):
            raw_target_records.append(
                {
                    "phase": "warmup" if index < warmup_count else "measurement",
                    "sequence_index": index,
                    "source_table": record["table"],
                    "source_row": record["row_number"],
                    "raw": record["raw"],
                }
            )
    target_record_ids = {id(record) for record in target_records}
    other_rows = [record["raw"] for record in records if id(record) not in target_record_ids]
    other_by_type: dict[str, int] = {}
    for row in other_rows:
        op_type = row.get("OP Type", "").strip() or "<empty>"
        other_by_type[op_type] = other_by_type.get(op_type, 0) + 1
    output_matches_replay_row = _validate_replay_outputs(
        request["replay_row"], worker["outputs"], kernel_type=request["kernel_type"]
    )
    actual_op_types = sorted({row.get("OP Type", "").strip() for row in measurement_rows})
    warnings = []
    if request["kernel_type"] not in actual_op_types:
        warnings.append(
            f"The requested logical operator resolved to physical OP Type(s): {', '.join(actual_op_types)}."
        )
    accelerator_cores = sorted({row.get("Task Type", "").strip() for row in measurement_rows if row.get("Task Type")})
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "status": "succeeded",
        "phase": "measurement",
        "simulated": False,
        "result_source": "msprof",
        "operator": {**worker["operator"], "actual_op_types": actual_op_types},
        "configuration": {
            "backend": "msprof",
            "replay_row": request["replay_row"],
            "profiling": request["profiling"],
        },
        "environment": worker["environment"],
        "inputs": worker["inputs"],
        "outputs": worker["outputs"],
        "duration": {
            "source": "msprof",
            "source_kind": "current_measurement",
            "column": "Task Duration(us)",
            "unit": "us",
            "aggregation": "p50",
            "value_us": duration_stats["p50"],
            "warmup_count": warmup_count,
            "sample_count": repeat_count,
            "statistics": duration_stats,
        },
        "msprof": {
            "executed": True,
            "simulated": False,
            "source": "msprof",
            "return_code": return_code,
            "timed_out": False,
            "tables": tables,
            "target_match": {
                "rule": "exact_allowlisted_op_type_and_invocation_count",
                "target_op_types": worker["operator"]["target_op_types"],
                "expected_records": expected_count,
                "matched_records": len(target_records),
                "warmup_records": warmup_count,
                "measurement_records": repeat_count,
                "accelerator_core": ";".join(accelerator_cores) or None,
            },
            "aggregated_metrics": aggregated_metrics,
            "physical_kernels": _physical_kernels(measurement_rows),
            "raw_target_records": raw_target_records,
            "other_tasks": {"count": len(other_rows), "by_op_type": other_by_type},
        },
        "validation": {
            "request_valid": True,
            "adapter_supported": True,
            "inputs_built": True,
            "replay_legal": True,
            "msprof_completed": True,
            "profile_match_unambiguous": True,
            "finite_positive_duration": True,
            "output_matches_replay_row": output_matches_replay_row,
            "simulation_record_matched": None,
        },
        "warnings": warnings,
        "error": None,
        "artifacts": None,
    }


def _artifact_manifest(task_dir: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(item for item in task_dir.rglob("*") if item.is_file()):
        files.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(task_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return files


def _finalize_artifacts(task_dir: Path, keep: bool) -> dict[str, Any]:
    if keep:
        files = _artifact_manifest(task_dir)
        return {
            "kept": True,
            "directory": str(task_dir),
            "files": files,
            "simulation_source": None,
        }
    discarded_file_count = sum(1 for item in task_dir.rglob("*") if item.is_file())
    shutil.rmtree(task_dir, ignore_errors=True)
    return {
        "kept": False,
        "directory": None,
        "files": [],
        "discarded_file_count": discarded_file_count,
        "simulation_source": None,
    }


def _write_process_log(path: Path, value: str | bytes | None) -> None:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value or ""
    path.write_text(text, encoding="utf-8")


def run_msprof_backend(request: dict[str, Any], request_id: str, artifact_dir: Path | None) -> dict[str, Any]:
    """Execute one registered adapter under msprof and parse its op_summary."""
    has_torch_npu = importlib.util.find_spec("torch_npu") is not None
    msprof_path = shutil.which("msprof")
    if not has_torch_npu or msprof_path is None:
        raise MicrobenchError(
            "NPU_NOT_AVAILABLE",
            "environment_validation",
            "The msprof backend requires an Ascend NPU environment with torch_npu and msprof.",
            "真实 msprof 后端需要可用的 Ascend NPU 环境、torch_npu 和 msprof。",
            {
                "torch_npu_available": has_torch_npu,
                "msprof_available": msprof_path is not None,
            },
        )
    try:
        get_adapter(request["kernel_type"])
    except AdapterValidationError as exc:
        raise MicrobenchError(
            "UNSUPPORTED_OPERATOR",
            "adapter_validation",
            exc.message,
            exc.message_zh,
            exc.details,
        ) from exc

    if artifact_dir is None:
        task_dir = Path(tempfile.mkdtemp(prefix=f"op_microbench_{request_id}_"))
    else:
        artifact_root = artifact_dir.resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        task_dir = artifact_root / f"op_microbench_{request_id}"
        task_dir.mkdir(parents=False, exist_ok=False)
    request_path = task_dir / "normalized_request.json"
    metadata_path = task_dir / "worker_metadata.json"
    profile_root = task_dir / "msprof"
    profile_root.mkdir()
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        msprof_path,
        f"--output={profile_root}",
        sys.executable,
        "-m",
        "tools.perf_data_collection.op_microbench.worker",
        "--request",
        str(request_path),
        "--metadata-out",
        str(metadata_path),
    ]
    policy = request["profiling"]["keep_artifacts"]
    try:
        # Start msprof in its own process group so that a timeout can kill
        # the whole process tree. msprof spawns worker children; killing
        # only the direct child would leave orphaned workers holding NPU
        # resources and the artifact directory.
        process = subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=request["profiling"]["timeout_seconds"])
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            # Kill the entire process group to reclaim every descendant.
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            stdout, stderr = process.communicate()
            _write_process_log(task_dir / "msprof.stdout.log", stdout)
            _write_process_log(task_dir / "msprof.stderr.log", stderr)
            raise MicrobenchError(
                "MSPROF_TIMEOUT",
                "measurement",
                "The msprof process timed out.",
                "msprof 采集进程超时。",
                {
                    "timeout_seconds": request["profiling"]["timeout_seconds"],
                    "command": cmd,
                },
            ) from None
        _write_process_log(task_dir / "msprof.stdout.log", stdout)
        _write_process_log(task_dir / "msprof.stderr.log", stderr)
        worker = _read_worker_metadata(metadata_path)
        if worker.get("status") != "succeeded":
            raise MicrobenchError(
                "OPERATOR_EXECUTION_FAILED",
                "operator_execution",
                "The operator worker failed inside the msprof process.",
                "算子 worker 在 msprof 采集进程中执行失败。",
                {
                    "return_code": return_code,
                    "worker_error": worker.get("error"),
                },
            )
        if return_code != 0:
            raise MicrobenchError(
                "MSPROF_FAILED",
                "measurement",
                f"msprof exited with code {return_code}.",
                f"msprof 以退出码 {return_code} 结束。",
                {"return_code": return_code, "command": cmd},
            )
        result = _build_msprof_result(
            request,
            request_id,
            worker,
            _find_msprof_tables(profile_root),
            return_code,
        )
    except MicrobenchError as error:
        error.artifacts = _finalize_artifacts(task_dir, policy in {"always", "on_failure"})
        raise
    except Exception as exc:
        artifacts = _finalize_artifacts(task_dir, policy in {"always", "on_failure"})
        raise MicrobenchError(
            "MSPROF_BACKEND_INTERNAL_ERROR",
            "measurement",
            f"Unexpected msprof backend failure: {exc}",
            f"msprof 后端发生未预期错误：{exc}",
            {"exception_type": type(exc).__name__},
            artifacts=artifacts,
        ) from exc
    result["artifacts"] = _finalize_artifacts(task_dir, policy == "always")
    return result


def _failure_result(request_id: str, error: MicrobenchError) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "status": "failed",
        "phase": error.phase,
        "simulated": None,
        "result_source": None,
        "operator": None,
        "configuration": None,
        "environment": None,
        "inputs": [],
        "outputs": [],
        "duration": None,
        "msprof": None,
        "validation": {"request_valid": error.phase != "request_validation"},
        "warnings": [],
        "error": {
            "code": error.code,
            "message": error.message,
            "message_zh": error.message_zh,
            "details": error.details,
        },
        "artifacts": error.artifacts,
    }


def _format_csv_number(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"


def _result_to_csv(result: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    metrics = result["msprof"]["aggregated_metrics"]
    metric_names = sorted(name for name in metrics if name != "Task Duration(us)")
    replay_row = result["configuration"]["replay_row"]
    identity_columns = [column for column in replay_row if column.startswith("Runtime ") or column == "EP Size"]
    columns = (
        BASE_CSV_COLUMNS
        + identity_columns
        + ACTUAL_OUTPUT_CSV_COLUMNS
        + DURATION_CSV_COLUMNS
        + [f"Average {name}" for name in metric_names]
        + SIMULATION_CSV_COLUMNS
    )
    duration_stats = result["duration"]["statistics"]
    row = {
        "OP State": "static",
        "Accelerator Core": (
            result["msprof"]["raw_target_records"][0]["raw"].get("Accelerator Core")
            or result["msprof"]["raw_target_records"][0]["raw"].get("Task Type")
            or result["msprof"]["target_match"].get("accelerator_core")
            or ""
        )
        if result["msprof"]["raw_target_records"]
        else result["msprof"]["target_match"].get("accelerator_core") or "",
        "Input Shapes": replay_row["Input Shapes"],
        "Input Data Types": replay_row["Input Data Types"],
        "Input Formats": replay_row["Input Formats"],
        "Output Shapes": replay_row["Output Shapes"],
        "Output Data Types": replay_row["Output Data Types"],
        "Output Formats": replay_row["Output Formats"],
        "Actual Output Shapes": _format_shapes([tensor["logical_shape"] for tensor in result["outputs"]]),
        "Actual Output Data Types": ";".join(tensor["dtype"] for tensor in result["outputs"]),
        "Actual Output Formats": ";".join(tensor["format"] or "" for tensor in result["outputs"]),
        "Average Duration(us)": _format_csv_number(duration_stats["mean"]),
        "Median Duration(us)": _format_csv_number(duration_stats["p50"]),
        "Std Duration(us)": _format_csv_number(duration_stats["std"]),
        "Result Source": result["result_source"],
        "Simulated": str(result["simulated"]).lower(),
        "Source CSV": (
            result["artifacts"]["simulation_source"]["path"] if result["artifacts"].get("simulation_source") else ""
        ),
        "Source Row": (
            result["artifacts"]["simulation_source"]["row_number"]
            if result["artifacts"].get("simulation_source")
            else ""
        ),
    }
    for column in identity_columns:
        row[column] = replay_row[column]
    for metric_name in metric_names:
        row[f"Average {metric_name}"] = _format_csv_number(metrics[metric_name].get("mean"))
    return columns, row


def _write_json(result: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        sys.stdout.write(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _write_csv(result: dict[str, Any], output: Path | None) -> None:
    columns, row = _result_to_csv(result)
    if output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Return zero only when a valid result was emitted."""
    args = build_argparser().parse_args(argv)
    request_id = str(uuid.uuid4())
    failure_output_is_safe = args.simulate_from_database is None
    if args.simulate_from_database is not None:
        try:
            simulation_source = args.simulate_from_database.resolve()
            _ensure_result_is_decoupled(args.output, simulation_source)
            _ensure_result_is_decoupled(args.artifact_dir, simulation_source)
            failure_output_is_safe = True
        except MicrobenchError as error:
            failure = _failure_result(request_id, error)
            if args.output_format == "json":
                _write_json(failure, None)
            else:
                sys.stderr.write(json.dumps(failure["error"], ensure_ascii=False, indent=2) + "\n")
            return 1
    if args.list_operators:
        _write_json(
            {
                "supported_operators": sorted(ADAPTERS),
                "request_mode": "replay_row",
                "excluded_multi_card_operators": ["DispatchFFNCombine"],
            },
            args.output,
        )
        return 0
    if args.describe_operator:
        try:
            _write_json(describe_adapter(args.describe_operator), args.output)
            return 0
        except AdapterValidationError as error:
            failure = MicrobenchError(
                "INVALID_REQUEST",
                "request_validation",
                error.message,
                error.message_zh,
                error.details,
            )
            _write_json(_failure_result(str(uuid.uuid4()), failure), args.output)
            return 1
    try:
        request = normalize_request(load_request(args.request))
        if args.simulate_from_database is not None:
            result = run_database_simulation(request, args.simulate_from_database, request_id)
        else:
            result = run_msprof_backend(request, request_id, args.artifact_dir)
        if args.output_format == "json":
            _write_json(result, args.output)
        else:
            _write_csv(result, args.output)
        return 0
    except MicrobenchError as error:
        failure = _failure_result(request_id, error)
        if args.output_format == "json":
            failure_output = args.output if failure_output_is_safe else None
            _write_json(failure, failure_output)
        else:
            sys.stderr.write(json.dumps(failure["error"], ensure_ascii=False, indent=2) + "\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
