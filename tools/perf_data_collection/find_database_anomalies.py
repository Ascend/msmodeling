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

"""Find review candidates in a profiling database without modifying it.

The audit deliberately separates deterministic data errors from statistical
leave-one-out (LOO) evidence. LOO uses the production candidate builders and
interpolation geometry; it only flags review candidates and never rewrites CSVs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Sequence

import pandas as pd
import yaml


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tensor_cast.performance_model.profiling_database.interpolation_index import (
    CandidateGroup,
    CandidateIndex,
    CandidatePoint,
    InterpolationResult,
)
from tensor_cast.performance_model.profiling_database.interpolating_data_source import (
    InterpolatingDataSource,
    _ATTENTION_AXIS_GROUPS,
    _COMPUTE_AXIS_GROUPS,
    _COMPUTE_SCALE_AXIS_GROUPS,
    _COMPUTE_SCALE_SUBCATEGORY,
    _ELEMENTWISE_AXIS_GROUPS,
    _INTERPOLATION_MATMUL_KERNELS,
    _MOE_FUSED_AXIS_GROUPS,
    _QUANTIZED_MATMUL_SUBCATEGORY,
    _candidate_latency_cols as interpolation_latency_cols,
)
from tensor_cast.performance_model.profiling_database.profiling_data_source import (
    ProfilingDataSource,
)
from tools.perf_data_collection.signature_utils import get_sig


# The audit intentionally reuses production interpolation internals instead of
# maintaining a second candidate policy. Changes to these private APIs require
# the audit and production datasource regression tests to run together.
_IDENTITY_KEYS = (
    "version",
    "device",
    "cann_version",
    "pytorch_version",
    "op_plugin_version",
)
_DIRECT_LATENCY_COLUMNS = {
    "Average Duration(us)",
    "Median Duration(us)",
    "Duration(us)",
    "MicroBench Duration(us)",
    "Profiling Average Duration(us)",
    "Profiling Median Duration(us)",
    "Profiling Std Duration(us)",
    "Service Median Duration(us)",
    "Service Std Duration(us)",
    "Std Duration(us)",
    "Service Count",
    "bandwidth_gbps",
}
_PROVENANCE_COLUMNS = {
    "Notes",
    "Runtime case_id",
    "Runtime source_profile",
}
_COUNTER_PREFIXES = ("Average ", "MicroBench ", "Profiling Average ")
_COUNTER_SUFFIXES = (
    "_time(us)",
    "_ratio",
    "_total_cycles",
    "_miss_rate",
    "_utilization(%)",
)
# Report ordering is separate from evidence-to-status mapping in _merge_status.
_STATUS_PRIORITY = {
    "CONFIRMED_FORMAT_ERROR": 0,
    "REVIEW_REGIME": 1,
    "REMEASURE_HIGH": 2,
    "UNMEASURED_PLACEHOLDER": 3,
    "REMEASURE_NORMAL": 4,
    "INSUFFICIENT_EVIDENCE": 5,
}
_DEFAULT_RESIDUAL_THRESHOLD = 1.0
# Exact and candidate policies read the same CSV value. This only tolerates
# binary floating-point representation noise, not measurement variance.
_LATENCY_POLICY_REL_TOL = 1e-12
_EXIT_OK = 0
_EXIT_MISSING_REQUIRED_CSV = 2
_OUTPUT_COLUMNS = (
    "kernel_type",
    "csv_path",
    "row_number",
    "strict_row_signature",
    "collection_match_signature",
    "status",
    "evidence",
    "exact_latency_us",
    "exact_latency_column",
    "exact_latency_details",
    "candidate_latency_us",
    "candidate_latency_column",
    "candidate_latency_details",
    "target_axes",
    "regime_key",
    "predicted_latency_us",
    "relative_error",
    "interpolation_method",
    "interpolation_axes",
    "matched_rows",
    "reason",
    "recommended_action",
)


@dataclass
class AuditCandidate:
    kernel_type: str
    csv_path: str
    row_number: int
    strict_row_signature: str
    collection_match_signature: str
    status: str = "INSUFFICIENT_EVIDENCE"
    evidence: list[str] = field(default_factory=list)
    exact_latency_us: float | None = None
    exact_latency_column: str = ""
    exact_latency_details: str = ""
    candidate_latency_us: float | None = None
    candidate_latency_column: str = ""
    candidate_latency_details: str = ""
    target_axes: str = ""
    regime_key: str = ""
    predicted_latency_us: float | None = None
    relative_error: float | None = None
    interpolation_method: str = ""
    interpolation_axes: str = ""
    matched_rows: str = ""
    reason: str = ""
    recommended_action: str = ""


@dataclass
class ScanResult:
    database_snapshot_hash: str
    op_mapping_sha256: str
    files_scanned: int
    rows_scanned: int
    valid_latency_rows: int
    loo_evaluated: int
    loo_predicted: int
    loo_rows_attempted: int
    loo_rows_predicted: int
    loo_rows_abstained: int
    loo_indexed_kernels: tuple[str, ...]
    loo_predicted_kernels: tuple[str, ...]
    deterministic_only_csvs: tuple[str, ...]
    residual_threshold: float | None
    candidates: list[AuditCandidate]


@dataclass
class _AuditRow:
    candidate: AuditCandidate
    row: dict[str, str]
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _LooContext:
    kind: str
    kernel_type: str
    policy_kernel_type: str
    tc_input_count: int | None = None
    include_output_signature: bool = False


@dataclass
class _LooRowOutcome:
    attempts: int = 0
    predictions: int = 0
    reasons: set[str] = field(default_factory=set)


@dataclass
class _LooAnalysis:
    observations: list[tuple[float, str, CandidatePoint, InterpolationResult]]
    evaluated: int
    predicted: int
    indexed_kernels: tuple[str, ...]
    predicted_kernels: tuple[str, ...]
    row_outcomes: dict[tuple[str, int], _LooRowOutcome]
    index_errors: dict[str, str]


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalized_slots(value: str, *, shape: bool = False) -> str:
    slots = []
    for slot in _cell_text(value).strip('"').split(";"):
        cleaned = slot.strip().strip('"').strip()
        if shape:
            if cleaned == "()":
                slots.append(cleaned)
                continue
            cleaned = cleaned.strip("()")
            parts = [part.strip() for part in cleaned.split(",")] if "," in cleaned else cleaned.split()
            cleaned = ",".join(parts)
        slots.append(cleaned)
    while slots and slots[-1] == "":
        slots.pop()
    return ";".join(slots)


def _is_measurement_column(column: str) -> bool:
    if column in _DIRECT_LATENCY_COLUMNS:
        return True
    return column.startswith(_COUNTER_PREFIXES) and column.endswith(_COUNTER_SUFFIXES)


def build_strict_row_signature(
    kernel_type: str,
    row: dict[str, str],
    database_identity: dict[str, Any],
) -> str:
    """Hash all semantic fields while excluding registered measurements."""
    semantic_fields: dict[str, str] = {}
    for column in sorted(row):
        if column in _PROVENANCE_COLUMNS or _is_measurement_column(column):
            continue
        value = row[column]
        if column.endswith("Shapes"):
            semantic_fields[column] = _normalized_slots(value, shape=True)
        elif column.endswith(("Data Types", "Formats")):
            semantic_fields[column] = _normalized_slots(value)
        else:
            semantic_fields[column] = _cell_text(value)
    payload = {
        "signature_version": 1,
        "kernel_type": kernel_type,
        "database": {key: _cell_text(database_identity.get(key)) for key in _IDENTITY_KEYS},
        "fields": semantic_fields,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_database_file(path: Path, database_path: Path) -> Path:
    if path.is_symlink():
        raise ValueError(f"database file must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(database_path)
    except (OSError, ValueError) as error:
        raise ValueError(f"database file is outside the database directory: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"database entry is not a regular file: {path}")
    return resolved


def _database_snapshot_hash(database_path: Path, csv_paths: Sequence[Path]) -> str:
    files = [
        (path.relative_to(database_path).as_posix(), _sha256(path))
        for path in csv_paths
    ]
    payload = json.dumps(files, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _relative_latency_error(predicted_latency_us: float, target_latency_us: float) -> float:
    """Return a direction-neutral multiplicative latency residual."""
    return abs(predicted_latency_us - target_latency_us) / min(
        predicted_latency_us,
        target_latency_us,
    )


def _validate_shape_metadata(row: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for prefix in ("Input", "Output"):
        shape_column = f"{prefix} Shapes"
        dtype_column = f"{prefix} Data Types"
        format_column = f"{prefix} Formats"
        columns = (shape_column, dtype_column, format_column)
        if not any(column in row for column in columns):
            continue
        for column in columns:
            if column not in row:
                errors.append(f"{column} column is missing")

        shape_slots = _cell_text(row.get(shape_column)).strip('"').split(";")
        dtype_slots = _cell_text(row.get(dtype_column)).strip('"').split(";")
        format_slots = _cell_text(row.get(format_column)).strip('"').split(";")
        for index, shape in enumerate(shape_slots):
            cleaned_shape = shape.strip().strip('"').strip()
            dimensions: list[int]
            if not cleaned_shape:
                dimensions = []
            elif cleaned_shape == "()":
                dimensions = []
            else:
                dimension_tokens = (
                    [part.strip() for part in cleaned_shape.strip("()").split(",")]
                    if "," in cleaned_shape
                    else cleaned_shape.strip("()").split()
                )
                if not dimension_tokens or any(not token for token in dimension_tokens):
                    errors.append(f"{prefix.lower()} shape slot {index} has an empty dimension")
                    dimensions = []
                else:
                    try:
                        dimensions = [int(part) for part in dimension_tokens]
                    except ValueError:
                        errors.append(f"{prefix.lower()} shape slot {index} is not integral")
                        dimensions = []
            if any(dimension < 0 for dimension in dimensions):
                errors.append(f"{prefix.lower()} shape slot {index} has a negative dimension")
            if index >= len(dtype_slots) or not dtype_slots[index].strip():
                errors.append(f"{prefix.lower()} dtype slot {index} is missing")
            if index >= len(format_slots) or not format_slots[index].strip():
                errors.append(f"{prefix.lower()} format slot {index} is missing")
    return errors


def _latency_issues(row: dict[str, str], columns: Iterable[str]) -> tuple[list[str], bool]:
    values = [(column, _cell_text(row.get(column))) for column in columns if column in row]
    if not values:
        return ["no supported latency column"], False
    invalid = []
    positive = False
    for column, raw_value in values:
        if raw_value == "":
            continue
        parsed = _parse_float(raw_value)
        if parsed is None or parsed < 0:
            invalid.append(f"{column}={raw_value!r}")
        elif parsed > 0:
            positive = True
    return invalid, positive


def _collection_signature(kernel_type: str, row: dict[str, str]) -> str:
    signature = get_sig(row, op_name=kernel_type)
    return json.dumps(signature, ensure_ascii=False, separators=(",", ":"))


def _merge_status(candidate: AuditCandidate) -> None:
    evidence = set(candidate.evidence)
    if evidence.intersection(
        {"INVALID_CSV", "INVALID_LATENCY_VALUE", "INVALID_SHAPE_METADATA", "MISSING_CSV"}
    ):
        candidate.status = "CONFIRMED_FORMAT_ERROR"
        candidate.recommended_action = "Fix the deterministic format error before using this row."
    elif "LATENCY_POLICY_DIVERGENCE" in evidence:
        candidate.status = "REVIEW_REGIME"
        candidate.recommended_action = (
            "Align exact and interpolation latency policies before using this row."
        )
    elif "COLLECTION_SIGNATURE_COLLISION" in evidence:
        candidate.status = "REVIEW_REGIME"
        candidate.recommended_action = (
            "Review fields omitted by the collection signature before merging or backfilling this group."
        )
    elif "REVIEW_REGIME" in evidence:
        candidate.status = "REVIEW_REGIME"
        candidate.recommended_action = "Review the shared regime or boundary before remeasuring individual rows."
    elif "LOCAL_LOO_RESIDUAL" in evidence and len(evidence) > 1:
        candidate.status = "REMEASURE_HIGH"
        candidate.recommended_action = "Remeasure the target and its matched neighbors."
    elif "UNMEASURED_PLACEHOLDER" in evidence:
        candidate.status = "UNMEASURED_PLACEHOLDER"
        candidate.recommended_action = "Collect a positive latency or remove the unsupported placeholder."
    elif evidence == {"INSUFFICIENT_EVIDENCE"}:
        candidate.status = "INSUFFICIENT_EVIDENCE"
        candidate.recommended_action = "Add same-regime support points before judging this row."
    elif evidence:
        candidate.status = "REMEASURE_NORMAL"
        candidate.recommended_action = "Review the evidence and remeasure before changing the database."


def _add_evidence(record: _AuditRow, evidence: str, reason: str) -> None:
    if evidence not in record.candidate.evidence:
        record.candidate.evidence.append(evidence)
    if reason and reason not in record.reasons:
        record.reasons.append(reason)
    _merge_status(record.candidate)


def _load_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        first_line = file.readline().rstrip("\r\n")
        if first_line == "version https://git-lfs.github.com/spec/v1":
            raise ValueError("CSV is a Git LFS pointer; fetch its LFS content before scanning")
        file.seek(0)
        reader = csv.DictReader(file, strict=True)
        if not reader.fieldnames:
            raise ValueError("CSV header is missing")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV header contains duplicate columns")
        rows = []
        for row in reader:
            if None in row:
                raise ValueError("CSV row has more cells than the header")
            rows.append({str(key): _cell_text(value) for key, value in row.items()})
        return [str(field) for field in reader.fieldnames], rows


def _validate_kernel_type(value: Any, mapping_name: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"operator_mappings[{mapping_name!r}].{field_name} must be a non-empty string"
        )
    kernel_type = value.strip()
    windows_path = PureWindowsPath(kernel_type)
    if (
        kernel_type != value
        or kernel_type in {".", ".."}
        or kernel_type.lower().endswith(".csv")
        or "/" in kernel_type
        or "\\" in kernel_type
        or ":" in kernel_type
        or "\x00" in kernel_type
        or windows_path.is_absolute()
        or windows_path.drive
    ):
        raise ValueError(
            f"operator_mappings[{mapping_name!r}].{field_name} must be a single CSV stem"
        )
    return kernel_type


def _discover_loo_contexts(mapping: dict[str, Any]) -> list[_LooContext]:
    operator_mappings = mapping.get("operator_mappings")
    if not isinstance(operator_mappings, dict) or not operator_mappings:
        raise ValueError("op_mapping.yaml operator_mappings must be a non-empty mapping")

    contexts: set[_LooContext] = set()
    for mapping_name, operator_mapping in operator_mappings.items():
        if not isinstance(operator_mapping, dict):
            raise ValueError(f"operator_mappings[{mapping_name!r}] must be a mapping")
        for flag in ("zero_cost", "accepted_miss", "composite"):
            if flag in operator_mapping and not isinstance(operator_mapping[flag], bool):
                raise ValueError(f"operator_mappings[{mapping_name!r}].{flag} must be a boolean")
        for field_name in ("category", "query_mode", "compute_subcategory"):
            field_value = operator_mapping.get(field_name)
            if field_value is not None and (not isinstance(field_value, str) or not field_value.strip()):
                raise ValueError(
                    f"operator_mappings[{mapping_name!r}].{field_name} must be a non-empty string"
                )
        if (
            operator_mapping.get("zero_cost")
            or operator_mapping.get("accepted_miss")
            or operator_mapping.get("composite")
            or operator_mapping.get("category") == "communication"
        ):
            continue
        primary = _validate_kernel_type(operator_mapping.get("kernel_type"), mapping_name, "kernel_type")
        alternate = operator_mapping.get("alternate_kernel_types")
        if alternate is None:
            alternate = []
        if not isinstance(alternate, list) or any(
            not isinstance(kernel_type, str) or not kernel_type.strip() for kernel_type in alternate
        ):
            raise ValueError(
                f"operator_mappings[{mapping_name!r}].alternate_kernel_types must be a list of non-empty strings"
            )
        alternate = [
            _validate_kernel_type(kernel_type, mapping_name, "alternate_kernel_types")
            for kernel_type in alternate
        ]
        tc_input_count = operator_mapping.get("tc_input_count")
        if tc_input_count is not None and (
            isinstance(tc_input_count, bool) or not isinstance(tc_input_count, int) or tc_input_count <= 0
        ):
            raise ValueError(f"operator_mappings[{mapping_name!r}].tc_input_count must be a positive integer")

        kernel_types = [primary, *alternate]
        query_mode = operator_mapping.get("query_mode")
        subcategory = operator_mapping.get("compute_subcategory")
        for kernel_type in kernel_types:
            if query_mode == "elementwise":
                kind = "elementwise"
            elif query_mode == "attention_special":
                kind = "attention"
            elif query_mode == "moe_fused":
                kind = "moe_fused"
            elif subcategory == _COMPUTE_SCALE_SUBCATEGORY:
                kind = "compute_scale"
            elif kernel_type in _INTERPOLATION_MATMUL_KERNELS:
                kind = "matmul"
            else:
                kind = "generic_compute"
            contexts.add(
                _LooContext(
                    kind=kind,
                    kernel_type=str(kernel_type),
                    policy_kernel_type=str(primary),
                    tc_input_count=tc_input_count,
                    include_output_signature=subcategory == _QUANTIZED_MATMUL_SUBCATEGORY,
                )
            )
    return sorted(
        contexts,
        key=lambda context: (
            context.kernel_type,
            context.kind,
            context.policy_kernel_type,
            context.tc_input_count or -1,
        ),
    )


def _context_indexes(
    source: InterpolatingDataSource,
    context: _LooContext,
) -> list[tuple[CandidateIndex, Sequence[Sequence[str]], str | None]]:
    if context.kind == "matmul":
        index = source._get_compute_index(
            context.kernel_type,
            context.tc_input_count,
            include_output_signature=context.include_output_signature,
        )
        return [] if index is None else [(index, _COMPUTE_AXIS_GROUPS, None)]
    if context.kind == "generic_compute":
        index = source._get_generic_compute_index(
            context.kernel_type,
            context.tc_input_count,
            context.policy_kernel_type,
        )
        axis_groups = source._generic_compute_axis_groups(
            context.kernel_type,
            context.policy_kernel_type,
        )
        return [] if index is None else [(index, axis_groups, None)]
    if context.kind == "attention":
        index = source._get_attention_index(context.kernel_type)
        transform = source._kernel_overrides.get(context.kernel_type, {}).get("axis_transform")
        return [] if index is None else [(index, _ATTENTION_AXIS_GROUPS, transform)]
    if context.kind == "moe_fused":
        index, _rejected = source._get_moe_fused_index(context.kernel_type)
        return [] if index is None else [(index, _MOE_FUSED_AXIS_GROUPS, None)]
    if context.kind == "compute_scale":
        index = source._get_compute_scale_index(context.kernel_type)
        return [] if index is None else [(index, _COMPUTE_SCALE_AXIS_GROUPS, None)]
    if context.kind == "elementwise":
        dataframe = source.base._load_csv(context.kernel_type)
        if dataframe is None or "Output Data Types" not in dataframe.columns:
            return []
        dtypes = sorted(
            {
                _cell_text(value).split(";")[0]
                for value in dataframe["Output Data Types"]
                if _cell_text(value)
            }
        )
        return [
            (index, _ELEMENTWISE_AXIS_GROUPS, None)
            for dtype in dtypes
            if (index := source._get_elementwise_index(context.kernel_type, dtype)) is not None
        ]
    return []


def _point_row_indices(point: CandidatePoint) -> set[int]:
    duplicate_indices = point.row_meta.get("duplicate_row_indices")
    if isinstance(duplicate_indices, list) and all(isinstance(index, int) for index in duplicate_indices):
        return set(duplicate_indices)
    return {point.row_index}


def _loo_observations(
    source: InterpolatingDataSource,
    contexts: Sequence[_LooContext],
    excluded_rows: dict[str, set[int]] | None = None,
) -> _LooAnalysis:
    observations: list[tuple[float, str, CandidatePoint, InterpolationResult]] = []
    evaluated = 0
    predicted = 0
    indexed_kernels: set[str] = set()
    predicted_kernels: set[str] = set()
    row_outcomes: dict[tuple[str, int], _LooRowOutcome] = {}
    index_errors: dict[str, str] = {}
    seen: set[tuple[Any, ...]] = set()
    excluded_rows = excluded_rows or {}
    for context in contexts:
        try:
            context_indexes = _context_indexes(source, context)
        except (pd.errors.ParserError, OSError, UnicodeError, ValueError) as error:
            index_errors.setdefault(context.kernel_type, f"{type(error).__name__}: {error}")
            continue
        if any(index.points for index, _axis_groups, _transform in context_indexes):
            indexed_kernels.add(context.kernel_type)
        excluded = excluded_rows.get(context.kernel_type, set())
        for index_position, (index, axis_groups, transform) in enumerate(context_indexes):
            override = {
                **source._kernel_overrides.get(context.policy_kernel_type, {}),
                **source._kernel_overrides.get(context.kernel_type, {}),
            }
            for group in index._candidate_groups.values():
                for target in group.points:
                    if _point_row_indices(target).intersection(excluded):
                        continue
                    key = (
                        context,
                        index_position,
                        tuple(tuple(group) for group in axis_groups),
                        transform,
                        target.row_index,
                        target.regime_key,
                        tuple(sorted(target.axes.items())),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    evaluated += 1
                    outcome = row_outcomes.setdefault(
                        (context.kernel_type, target.row_index),
                        _LooRowOutcome(),
                    )
                    outcome.attempts += 1
                    if target.latency_us <= 0:
                        outcome.reasons.add("non-positive target latency")
                        continue
                    target_selection = source._candidate_latency_column_selection(target)
                    remaining = [
                        point
                        for point in group.points
                        if point.row_index != target.row_index
                        and source._candidate_latency_column_selection(point) == target_selection
                        and not _point_row_indices(point).intersection(excluded)
                    ]
                    if len(remaining) < 2:
                        outcome.reasons.add("fewer than two same-source neighbors")
                        continue
                    if any(point.axes == target.axes for point in remaining):
                        outcome.reasons.add("same-axis duplicate remains after leave-one-out")
                        continue
                    result = None
                    for _label, candidate_group in source._latency_column_pure_candidate_group_attempts(
                        CandidateGroup(group.regime_key, remaining)
                    ):
                        target_axes = dict(target.axes)
                        active_group = candidate_group
                        axis_transform = None
                        extra_details: dict[str, Any] = {}
                        if transform in {"sqrt", "sqrt_seq"} and "seq" in target_axes:
                            extra_details["attention_axes"] = dict(target_axes)
                            target_axes["seq"] = math.sqrt(target_axes["seq"])
                            active_group = source._sqrt_seq_group(candidate_group)
                            axis_transform = "sqrt(seq)"
                        result = active_group.interpolate(
                            target_axes,
                            axis_groups,
                            fallback_from="audit_loo",
                            axis_transform=axis_transform,
                            max_interpolation_dim=(
                                None if context.kind == "attention" else override.get("max_interpolation_dim")
                            ),
                            extra_details=extra_details,
                            axis_matchers=(
                                {"q_tokens": source._attention_q_tokens_match}
                                if context.kind == "attention"
                                else None
                            ),
                        )
                        if result is not None:
                            if transform in {"sqrt", "sqrt_seq"}:
                                result = source._mark_sqrt_interpolation(result)
                            break
                    if result is None:
                        outcome.reasons.add("no legal same-regime interpolation bracket")
                        continue
                    predicted += 1
                    predicted_kernels.add(context.kernel_type)
                    outcome.predictions += 1
                    relative_error = _relative_latency_error(result.latency_us, target.latency_us)
                    observations.append((relative_error, context.kernel_type, target, result))
    observations.sort(key=lambda item: (-item[0], item[1], item[2].row_index))
    return _LooAnalysis(
        observations=observations,
        evaluated=evaluated,
        predicted=predicted,
        indexed_kernels=tuple(sorted(indexed_kernels)),
        predicted_kernels=tuple(sorted(predicted_kernels)),
        row_outcomes=row_outcomes,
        index_errors=index_errors,
    )


def scan_database(
    database_path: Path | str,
    *,
    residual_threshold: float | None = _DEFAULT_RESIDUAL_THRESHOLD,
) -> ScanResult:
    """Scan a database and return deterministic findings plus thresholded LOO evidence."""
    database_path = Path(database_path).resolve()
    if not database_path.is_dir():
        raise ValueError(f"database directory does not exist: {database_path}")
    mapping_path = database_path / "op_mapping.yaml"
    if not mapping_path.exists():
        raise ValueError(f"op_mapping.yaml does not exist: {mapping_path}")
    if residual_threshold is not None and (
        not math.isfinite(residual_threshold) or residual_threshold < 0
    ):
        raise ValueError("residual_threshold must be finite and non-negative")

    mapping_path = _validated_database_file(mapping_path, database_path)
    csv_paths = [
        _validated_database_file(path, database_path)
        for path in sorted(database_path.glob("*.csv"), key=lambda path: path.name)
    ]
    initial_csv_names = tuple(path.relative_to(database_path) for path in csv_paths)
    database_snapshot_hash = _database_snapshot_hash(database_path, csv_paths)
    op_mapping_sha256 = _sha256(mapping_path)
    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or {}
    if not isinstance(mapping, dict):
        raise ValueError("op_mapping.yaml root must be a mapping")
    all_contexts = _discover_loo_contexts(mapping)
    run_loo = residual_threshold is not None
    contexts = all_contexts if run_loo else []
    available_kernels = {path.stem for path in csv_paths}
    expected_primary_kernels = {context.policy_kernel_type for context in all_contexts}
    base = ProfilingDataSource(database_path)
    source = InterpolatingDataSource(base)
    records: dict[tuple[str, int], _AuditRow] = {}
    signatures: dict[str, list[_AuditRow]] = {}
    collection_signatures: dict[tuple[str, str], list[_AuditRow]] = {}
    file_excluded_kernels: set[str] = set()
    excluded_rows: dict[str, set[int]] = {}
    valid_latency_rows = 0

    for kernel_type in sorted(expected_primary_kernels - available_kernels):
        relative_path = f"{kernel_type}.csv"
        signature = hashlib.sha256(f"missing:{relative_path}".encode("utf-8")).hexdigest()
        candidate = AuditCandidate(
            kernel_type=kernel_type,
            csv_path=relative_path,
            row_number=1,
            strict_row_signature=signature,
            collection_match_signature="",
        )
        record = _AuditRow(candidate, {})
        _add_evidence(record, "MISSING_CSV", "required primary kernel CSV is missing")
        records[(relative_path, 1)] = record

    for csv_path in csv_paths:
        kernel_type = csv_path.stem
        relative_path = csv_path.relative_to(database_path).as_posix()
        try:
            headers, rows = _load_rows(csv_path)
        except (csv.Error, OSError, UnicodeError, ValueError) as error:
            file_excluded_kernels.add(kernel_type)
            signature = hashlib.sha256(f"{relative_path}:{error}".encode("utf-8")).hexdigest()
            candidate = AuditCandidate(
                kernel_type=kernel_type,
                csv_path=relative_path,
                row_number=1,
                strict_row_signature=signature,
                collection_match_signature="",
            )
            record = _AuditRow(candidate, {})
            _add_evidence(record, "INVALID_CSV", str(error))
            records[(relative_path, 1)] = record
            continue

        preferred_latency_column = base._latency_col(pd.DataFrame(columns=headers))
        latency_columns = tuple(
            dict.fromkeys(
                [
                    *base._candidate_latency_cols(preferred_latency_column),
                    *interpolation_latency_cols(preferred_latency_column),
                ]
            )
        )
        for row_index, row in enumerate(rows):
            row_number = row_index + 2
            strict_signature = build_strict_row_signature(kernel_type, row, mapping)
            candidate = AuditCandidate(
                kernel_type=kernel_type,
                csv_path=relative_path,
                row_number=row_number,
                strict_row_signature=strict_signature,
                collection_match_signature=_collection_signature(kernel_type, row),
            )
            record = _AuditRow(candidate, row)
            records[(relative_path, row_number)] = record
            signatures.setdefault(strict_signature, []).append(record)
            collection_signatures.setdefault(
                (kernel_type, candidate.collection_match_signature), []
            ).append(record)

            invalid_latency, has_positive_latency = _latency_issues(row, latency_columns)
            if invalid_latency:
                excluded_rows.setdefault(kernel_type, set()).add(row_index)
                _add_evidence(
                    record,
                    "INVALID_LATENCY_VALUE",
                    "invalid latency cells: " + ", ".join(invalid_latency),
                )
            elif not has_positive_latency:
                _add_evidence(
                    record,
                    "UNMEASURED_PLACEHOLDER",
                    "all supported latency cells are empty or zero",
                )
            else:
                valid_latency_rows += 1

            metadata_errors = _validate_shape_metadata(row)
            if metadata_errors:
                excluded_rows.setdefault(kernel_type, set()).add(row_index)
                _add_evidence(
                    record,
                    "INVALID_SHAPE_METADATA",
                    "; ".join(metadata_errors),
                )

            exact = base._effective_row_latency(
                pd.Series(row),
                kernel_type,
                preferred_latency_column,
            )
            if exact is not None:
                candidate.exact_latency_column = exact[0]
                candidate.exact_latency_us = exact[1]
                candidate.exact_latency_details = json.dumps(
                    exact[2],
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                )
            interpolation_latency, interpolation_meta = source._candidate_latency(
                row,
                preferred_latency_column,
            )
            if interpolation_latency is not None:
                candidate.candidate_latency_us = interpolation_latency
                candidate.candidate_latency_column = str(interpolation_meta.get("latency_column", ""))
            candidate.candidate_latency_details = json.dumps(
                interpolation_meta,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            )
            latency_policy_differs = (exact is None) != (interpolation_latency is None)
            if exact is not None and interpolation_latency is not None:
                latency_policy_differs = (
                    exact[0] != candidate.candidate_latency_column
                    or not math.isclose(
                        exact[1],
                        interpolation_latency,
                        rel_tol=_LATENCY_POLICY_REL_TOL,
                        abs_tol=0.0,
                    )
                )
            if latency_policy_differs:
                exact_selection = exact[2].get("latency_selection") if exact is not None else None
                divergence_reason = (
                    f"registered exact policy {exact_selection} selects a different latency than interpolation"
                    if exact_selection
                    else "unregistered exact/interpolation latency source or value divergence"
                )
                _add_evidence(
                    record,
                    "LATENCY_POLICY_DIVERGENCE",
                    divergence_reason,
                )

        invalid_row_indexes = excluded_rows.get(kernel_type)
        if invalid_row_indexes:
            sanitized = pd.DataFrame(rows, columns=headers)
            shape_columns = [column for column in headers if column.endswith("Shapes")]
            for row_index in invalid_row_indexes:
                for column in shape_columns:
                    sanitized.at[row_index, column] = ""
                for column in latency_columns:
                    if column in sanitized.columns:
                        sanitized.at[row_index, column] = "0"
            base._csv_cache[kernel_type] = sanitized

    for duplicate_rows in signatures.values():
        if len(duplicate_rows) < 2:
            continue
        values = {
            (record.candidate.exact_latency_column, record.candidate.exact_latency_us)
            for record in duplicate_rows
        }
        if len(values) > 1:
            locations = ", ".join(
                f"{record.candidate.csv_path}:{record.candidate.row_number}"
                for record in duplicate_rows
            )
            for record in duplicate_rows:
                _add_evidence(
                    record,
                    "DUPLICATE_LATENCY_CONFLICT",
                    f"the same strict signature has conflicting effective latency sources or values at {locations}",
                )

    for collision_rows in collection_signatures.values():
        strict_signatures = {record.candidate.strict_row_signature for record in collision_rows}
        if len(strict_signatures) < 2:
            continue
        locations = ", ".join(
            f"{record.candidate.csv_path}:{record.candidate.row_number}"
            for record in collision_rows
        )
        reason = (
            "collection signature collapses "
            f"{len(strict_signatures)} different strict row semantics at {locations}"
        )
        for record in collision_rows:
            _add_evidence(record, "COLLECTION_SIGNATURE_COLLISION", reason)

    loo_analysis = (
        _loo_observations(
            source,
            [
                context
                for context in contexts
                if context.kernel_type in available_kernels
                and context.kernel_type not in file_excluded_kernels
            ],
            excluded_rows,
        )
        if run_loo
        else _LooAnalysis([], 0, 0, (), (), {}, {})
    )
    for kernel_type, error in sorted(loo_analysis.index_errors.items()):
        relative_path = f"{kernel_type}.csv"
        key = (relative_path, 1)
        record = records.get(key)
        if record is None:
            signature = hashlib.sha256(f"{relative_path}:{error}".encode("utf-8")).hexdigest()
            candidate = AuditCandidate(
                kernel_type=kernel_type,
                csv_path=relative_path,
                row_number=1,
                strict_row_signature=signature,
                collection_match_signature="",
            )
            record = _AuditRow(candidate, {})
            records[key] = record
        _add_evidence(record, "INVALID_CSV", f"production index rejected CSV: {error}")

    for (kernel_type, row_index), outcome in loo_analysis.row_outcomes.items():
        if outcome.predictions:
            continue
        record = records.get((f"{kernel_type}.csv", row_index + 2))
        if record is None:
            continue
        reasons = ", ".join(sorted(outcome.reasons)) or "production interpolation did not return a prediction"
        _add_evidence(
            record,
            "INSUFFICIENT_EVIDENCE",
            f"LOO abstained after {outcome.attempts} attempt(s): {reasons}",
        )

    loo_rows_attempted = len(loo_analysis.row_outcomes)
    loo_rows_predicted = sum(
        1 for outcome in loo_analysis.row_outcomes.values() if outcome.predictions > 0
    )
    loo_rows_abstained = loo_rows_attempted - loo_rows_predicted
    deterministic_only_csvs = (
        tuple(sorted(available_kernels - set(loo_analysis.predicted_kernels)))
        if run_loo
        else ()
    )
    selected_loo_rows: set[tuple[str, int]] = set()
    selected_observations: list[tuple[_AuditRow, CandidatePoint, InterpolationResult]] = []
    for relative_error, kernel_type, target, result in loo_analysis.observations:
        if residual_threshold is None or relative_error < residual_threshold:
            break
        key = (f"{kernel_type}.csv", target.row_index + 2)
        if key in selected_loo_rows:
            continue
        record = records.get(key)
        if record is None:
            continue
        selected_loo_rows.add(key)
        candidate = record.candidate
        candidate.predicted_latency_us = result.latency_us
        candidate.relative_error = relative_error
        candidate.interpolation_method = result.method
        candidate.interpolation_axes = ",".join(result.axes)
        candidate.target_axes = json.dumps(target.axes, sort_keys=True, separators=(",", ":"))
        candidate.regime_key = json.dumps(
            dict(target.regime_key),
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        candidate.matched_rows = json.dumps(
            [_matched_point_payload(point) for point in result.matched_points],
            sort_keys=True,
            separators=(",", ":"),
        )
        _add_evidence(
            record,
            "LOCAL_LOO_RESIDUAL",
            "production LOO residual reached the review threshold; "
            f"relative error={relative_error:.6g}; threshold={residual_threshold:.6g}",
        )
        selected_observations.append((record, target, result))

    neighborhood_groups: dict[tuple[str, Any, tuple[str, ...]], list[tuple[_AuditRow, set[int]]]] = {}
    for record, target, result in selected_observations:
        neighborhood = _point_row_indices(target)
        for point in result.matched_points:
            neighborhood.update(_point_row_indices(point))
        group_key = (record.candidate.kernel_type, target.regime_key, tuple(result.axes))
        neighborhood_groups.setdefault(group_key, []).append((record, neighborhood))
    for grouped_records in neighborhood_groups.values():
        for index, (record, neighborhood) in enumerate(grouped_records):
            if any(
                neighborhood.intersection(other_neighborhood)
                for other_index, (_other_record, other_neighborhood) in enumerate(grouped_records)
                if other_index != index
            ):
                _add_evidence(
                    record,
                    "REVIEW_REGIME",
                    "multiple thresholded residuals share one local interpolation neighborhood",
                )

    candidates = []
    for record in records.values():
        if not record.candidate.evidence:
            continue
        record.candidate.evidence.sort()
        record.candidate.reason = " | ".join(record.reasons)
        _merge_status(record.candidate)
        candidates.append(record.candidate)
    candidates.sort(
        key=lambda candidate: (
            _STATUS_PRIORITY[candidate.status],
            -(candidate.relative_error if candidate.relative_error is not None else -1.0),
            candidate.kernel_type,
            candidate.csv_path,
            candidate.row_number,
        )
    )
    final_csv_paths = [
        _validated_database_file(path, database_path)
        for path in sorted(database_path.glob("*.csv"), key=lambda path: path.name)
    ]
    final_csv_names = tuple(path.relative_to(database_path) for path in final_csv_paths)
    if (
        final_csv_names != initial_csv_names
        or _database_snapshot_hash(database_path, final_csv_paths) != database_snapshot_hash
        or _sha256(mapping_path) != op_mapping_sha256
    ):
        raise RuntimeError("profiling database changed during scan; discard the result and retry")

    return ScanResult(
        database_snapshot_hash=database_snapshot_hash,
        op_mapping_sha256=op_mapping_sha256,
        files_scanned=len(csv_paths),
        rows_scanned=sum(1 for key in records if key[1] >= 2),
        valid_latency_rows=valid_latency_rows,
        loo_evaluated=loo_analysis.evaluated,
        loo_predicted=loo_analysis.predicted,
        loo_rows_attempted=loo_rows_attempted,
        loo_rows_predicted=loo_rows_predicted,
        loo_rows_abstained=loo_rows_abstained,
        loo_indexed_kernels=loo_analysis.indexed_kernels,
        loo_predicted_kernels=loo_analysis.predicted_kernels,
        deterministic_only_csvs=deterministic_only_csvs,
        residual_threshold=residual_threshold,
        candidates=candidates,
    )


def _candidate_csv_row(candidate: AuditCandidate) -> dict[str, Any]:
    row = asdict(candidate)
    row["evidence"] = ";".join(candidate.evidence)
    return row


def _matched_point_payload(point: CandidatePoint) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "row_number": point.row_index + 2,
        "latency_us": point.latency_us,
        "latency_column": point.row_meta.get("latency_column", ""),
    }
    duplicate_indices = point.row_meta.get("duplicate_row_indices")
    duplicate_meta = point.row_meta.get("duplicate_row_meta")
    if not isinstance(duplicate_indices, list) or not isinstance(duplicate_meta, list):
        return payload

    contributors = []
    for row_index, row_meta in zip(duplicate_indices, duplicate_meta):
        if not isinstance(row_index, int) or not isinstance(row_meta, dict):
            continue
        contributors.append(
            {
                "row_number": row_index + 2,
                "latency_us": row_meta.get("raw_latency_us"),
                "latency_column": row_meta.get("latency_column", ""),
            }
        )
    if contributors:
        payload["aggregation"] = point.row_meta.get("aggregation", "median")
        payload["contributors"] = contributors
    return payload


def _csv_safe_cell(value: Any) -> Any:
    if isinstance(value, str):
        candidate = value.lstrip()
        while candidate and ord(candidate[0]) < 32:
            candidate = candidate[1:].lstrip()
        if candidate.startswith(("=", "+", "-", "@")):
            return "'" + value
    return value


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        delete=False,
    ) as file:
        temp_path = Path(file.name)
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {column: _csv_safe_cell(value) for column, value in row.items()}
            for row in rows
        )
    os.replace(temp_path, path)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        delete=False,
    ) as file:
        temp_path = Path(file.name)
        file.write(content)
    os.replace(temp_path, path)


def _write_report_files(result: ScanResult, output_dir: Path, *, remeasure_limit: int) -> None:
    output_dir = Path(output_dir)
    _atomic_csv(
        output_dir / "anomaly_candidates.csv",
        _OUTPUT_COLUMNS,
        (_candidate_csv_row(candidate) for candidate in result.candidates),
    )
    def evidence_count(evidence: str) -> int:
        return sum(evidence in candidate.evidence for candidate in result.candidates)

    collection_collision_kernels = sorted(
        {
            candidate.kernel_type
            for candidate in result.candidates
            if "COLLECTION_SIGNATURE_COLLISION" in candidate.evidence
        }
    )
    residual_candidates = [
        candidate
        for candidate in result.candidates
        if "LOCAL_LOO_RESIDUAL" in candidate.evidence
    ]
    residual_regime_candidates = sum(
        "REVIEW_REGIME" in candidate.evidence for candidate in residual_candidates
    )
    residual_point_candidates = len(residual_candidates) - residual_regime_candidates
    threshold_text = (
        "未运行 LOO"
        if result.residual_threshold is None
        else f"{result.residual_threshold:g}（预测值与实测值至少相差 {result.residual_threshold + 1:g} 倍）"
    )
    summary = f"""# 实测算子性能数据库异常审计报告

报告性质：只读候选筛查。确定性问题可以直接修复；性能候选必须独立复测后才能修改数据库。

## 1. 扫描范围

| 项目 | 数量 / 值 |
| --- | ---: |
| 数据库快照 SHA-256 | `{result.database_snapshot_hash}` |
| op_mapping SHA-256 | `{result.op_mapping_sha256}` |
| 扫描的 CSV / 数据行 | {result.files_scanned} / {result.rows_scanned} |
| latency 为正的有效数据行 | {result.valid_latency_rows} |
| 尝试 LOO 的唯一数据行 | {result.loo_rows_attempted} |
| 能得到同分桶预测的数据行 | {result.loo_rows_predicted} |
| 因缺少合法邻点而无法判断的数据行 | {result.loo_rows_abstained} |
| 能建立 LOO 索引 / 能成功预测的 CSV | {len(result.loo_indexed_kernels)} / {len(result.loo_predicted_kernels)} |
| 固定性能复测候选阈值 | {threshold_text} |

## 2. 已确认的数据完整性问题

| 问题 | 数据行 / 文件数 |
| --- | ---: |
| 生产映射引用、但数据库中不存在的算子 CSV | {evidence_count("MISSING_CSV")} |
| 损坏或无法读取的 CSV | {evidence_count("INVALID_CSV")} |
| 非法 latency 数据行 | {evidence_count("INVALID_LATENCY_VALUE")} |
| Shape 或字段格式错误的数据行 | {evidence_count("INVALID_SHAPE_METADATA")} |
| 没有可用正 latency 的占位数据行 | {evidence_count("UNMEASURED_PLACEHOLDER")} |

## 3. 需要检查的数据契约问题

| 问题 | 数据行 / 算子数 |
| --- | ---: |
| 同一严格数据身份存在冲突 latency 的数据行 | {evidence_count("DUPLICATE_LATENCY_CONFLICT")} |
| 采集去重规则无法区分的数据行 / 算子 | {evidence_count("COLLECTION_SIGNATURE_COLLISION")} / {len(collection_collision_kernels)} |
| 同一数据行在直接命中和插值时使用了不同 latency | {evidence_count("LATENCY_POLICY_DIVERGENCE")} |

涉及采集去重风险的算子：{", ".join(collection_collision_kernels) or "无"}。

## 4. 达到固定阈值的性能复测候选

| 候选 | 数量 |
| --- | ---: |
| 达到固定残差阈值的数据行 | {len(residual_candidates)} |
| 共享局部邻域，需要整组检查的数据行 | {residual_regime_candidates} |
| 可独立定位，需要单点复测的数据行 | {residual_point_candidates} |

这里的候选表示“预测值与实测值偏差较大”，不是已经确认的错误数据。阈值只决定是否进入复测清单，不用于自动删除或改写 CSV。

## 5. 当前无法判断

| 原因 | 数量 |
| --- | ---: |
| 缺少合法同分桶邻点，无法完成 LOO 的数据行 | {evidence_count("INSUFFICIENT_EVIDENCE")} |
| 只能做确定性检查、无法得到 LOO 预测的 CSV | {len(result.deterministic_only_csvs)} |

只能做确定性检查的 CSV：{", ".join(result.deterministic_only_csvs) or "无"}。

## 6. 结果使用方式

- 第 2 节是确定性问题，应先修复或补齐。
- 第 3 节说明现有字段或匹配规则可能混淆不同数据，应核对采集和查询契约。
- 第 4 节只生成复测候选；复测稳定复现后，才能确认原始 latency 是否异常。
- 第 5 节不是“没有问题”，而是当前数据库密度或生产插值几何不足以判断。
- 同一数据行可能同时命中多类证据，各分类数量不能相加为“异常总数”。
"""
    _atomic_text(output_dir / "anomaly_summary.md", summary)

    manifest_fields = (
        "case_id",
        "database_snapshot_hash",
        "kernel_type",
        "csv_path",
        "row_number",
        "strict_row_signature",
        "status",
        "evidence",
        "reason",
    )
    manifest_rows = []
    manifest_candidates = [
        candidate
        for candidate in result.candidates
        if candidate.row_number >= 2
        and candidate.strict_row_signature
        and candidate.status in {"REMEASURE_HIGH", "REMEASURE_NORMAL", "UNMEASURED_PLACEHOLDER"}
    ][:remeasure_limit]
    for index, candidate in enumerate(manifest_candidates, start=1):
        manifest_rows.append(
            {
                "case_id": f"audit-{index:04d}",
                "database_snapshot_hash": result.database_snapshot_hash,
                "kernel_type": candidate.kernel_type,
                "csv_path": candidate.csv_path,
                "row_number": candidate.row_number,
                "strict_row_signature": candidate.strict_row_signature,
                "status": candidate.status,
                "evidence": ";".join(candidate.evidence),
                "reason": candidate.reason,
            }
        )
    _atomic_csv(output_dir / "remeasure_manifest.csv", manifest_fields, manifest_rows)


def write_reports(result: ScanResult, output_dir: Path | str, *, remeasure_limit: int = 0) -> None:
    output_dir = Path(output_dir)
    if remeasure_limit < 0:
        raise ValueError("remeasure_limit must be non-negative")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_dir}")
    if output_dir.is_symlink():
        raise ValueError(f"output directory must not be a symbolic link: {output_dir}")

    report_names = {"anomaly_candidates.csv", "anomaly_summary.md", "remeasure_manifest.csv"}
    if output_dir.exists():
        unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in report_names)
        if unexpected:
            raise ValueError(
                "output directory contains files not owned by this report: " + ", ".join(unexpected)
            )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    backup: Path | None = None
    try:
        _write_report_files(result, staging, remeasure_limit=remeasure_limit)
        if {path.name for path in staging.iterdir()} != report_names:
            raise RuntimeError("staged report package is incomplete")
        if output_dir.exists():
            backup = Path(
                tempfile.mkdtemp(prefix=f".{output_dir.name}.previous-", dir=output_dir.parent)
            )
            backup.rmdir()
            os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except BaseException:
            if backup is not None and backup.exists() and not output_dir.exists():
                os.replace(backup, output_dir)
            raise
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only profiling database anomaly candidate audit.",
    )
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=_DEFAULT_RESIDUAL_THRESHOLD,
        help="Minimum direction-neutral production-LOO residual to report (default: 1.0, or 2x).",
    )
    parser.add_argument(
        "--skip-loo",
        action="store_true",
        help="Run deterministic checks without production LOO.",
    )
    parser.add_argument(
        "--remeasure-limit",
        type=int,
        default=0,
        help="Maximum candidates copied to the review manifest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    database_path = args.database_path.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == database_path or database_path in output_dir.parents:
        raise ValueError("output directory must be outside the database directory")
    result = scan_database(
        database_path,
        residual_threshold=None if args.skip_loo else args.residual_threshold,
    )
    write_reports(result, output_dir, remeasure_limit=args.remeasure_limit)
    print(
        f"Scanned {result.files_scanned} CSV files and {result.rows_scanned} rows; "
        f"wrote audit reports to {output_dir}."
    )
    if any("MISSING_CSV" in candidate.evidence for candidate in result.candidates):
        return _EXIT_MISSING_REQUIRED_CSV
    return _EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
